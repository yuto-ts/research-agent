#!/usr/bin/env python3
"""research-agent: ロボット研究分野の自動リサーチエージェント

arXiv, RSS, 会議ページを巡回し、Claude Code CLI でスコアリング・要約を行い、
週次 Markdown レポートを生成する。
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import re
import shutil
import sqlite3
import subprocess
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import feedparser
import requests
import yaml
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------
DEFAULT_CONFIG_PATH = "config.yaml"
USER_AGENT = "research-agent/1.0 (academic research bot)"

logger = logging.getLogger("research-agent")

# ---------------------------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------------------------


def make_fingerprint(url: str, title: str) -> str:
    """URL + タイトルから正規化 fingerprint (SHA-256) を生成する。"""
    norm_url = re.sub(r"https?://", "", url).strip().rstrip("/").lower()
    norm_title = re.sub(r"\s+", " ", title).strip().lower()
    raw = f"{norm_url}|{norm_title}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def fetch_with_retry(
    url: str,
    params: dict | None = None,
    max_retries: int = 3,
    timeout: int = 30,
    backoff_base: float = 2.0,
) -> requests.Response:
    """指数バックオフ付き HTTP GET。"""
    for attempt in range(max_retries):
        try:
            resp = requests.get(
                url,
                params=params,
                headers={"User-Agent": USER_AGENT},
                timeout=timeout,
            )
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            if attempt == max_retries - 1:
                raise
            wait = backoff_base ** attempt
            logger.warning("リトライ %d/%d (%.1fs後): %s", attempt + 1, max_retries, wait, e)
            time.sleep(wait)
    raise RuntimeError("unreachable")


# ---------------------------------------------------------------------------
# データクラス
# ---------------------------------------------------------------------------


@dataclass
class Paper:
    """論文 1 件を表すデータオブジェクト。"""

    fingerprint: str
    source: str  # 'arxiv_api' | 'rss'
    arxiv_id: str | None
    title: str
    authors: str  # カンマ区切り
    abstract: str
    url: str
    published: str | None = None  # ISO 8601
    updated: str | None = None  # ISO 8601
    score: float = 0.0
    score_detail: dict = field(default_factory=dict)
    summary: str = ""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class Config:
    """config.yaml を読み込みバリデーションする。"""

    def __init__(self, path: str = DEFAULT_CONFIG_PATH) -> None:
        self.path = Path(path)
        with open(self.path, "r", encoding="utf-8") as f:
            self._raw: dict = yaml.safe_load(f)
        self._validate()

    # -- バリデーション -------------------------------------------------------

    def _validate(self) -> None:
        for sec in ("general", "llm", "sources", "scoring", "report"):
            if sec not in self._raw:
                raise ValueError(f"config.yaml に '{sec}' セクションがありません")
        total_weight = sum(c["max_points"] for c in self._raw["scoring"]["criteria"])
        max_score = self._raw["scoring"]["max_score"]
        if total_weight > max_score + 0.01:
            raise ValueError(
                f"criteria の max_points 合計({total_weight})が "
                f"max_score({max_score})を超えています"
            )

    # -- プロパティ -----------------------------------------------------------

    @property
    def lookback_days(self) -> int:
        return int(self._raw["general"]["lookback_days"])

    @property
    def database_path(self) -> str:
        return self._raw["general"].get("database_path", ".agent.db")

    @property
    def log_level(self) -> str:
        return self._raw["general"].get("log_level", "INFO")

    @property
    def llm(self) -> dict:
        return self._raw["llm"]

    @property
    def arxiv_api(self) -> dict:
        return self._raw["sources"]["arxiv_api"]

    @property
    def rss_feeds(self) -> list[dict]:
        return self._raw["sources"]["rss"].get("feeds", [])

    @property
    def rss_enabled(self) -> bool:
        return self._raw["sources"]["rss"].get("enabled", True)

    @property
    def url_watch_targets(self) -> list[dict]:
        return self._raw["sources"]["url_watch"].get("targets", [])

    @property
    def url_watch_enabled(self) -> bool:
        return self._raw["sources"]["url_watch"].get("enabled", True)

    @property
    def scoring(self) -> dict:
        return self._raw["scoring"]

    @property
    def report(self) -> dict:
        return self._raw["report"]


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------


class Database:
    """SQLite ラッパー。"""

    def __init__(self, db_path: str) -> None:
        self.conn = sqlite3.connect(db_path, timeout=10)
        self.conn.row_factory = sqlite3.Row
        self._init_tables()

    def _init_tables(self) -> None:
        cur = self.conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS papers (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                fingerprint   TEXT    UNIQUE NOT NULL,
                source        TEXT    NOT NULL,
                arxiv_id      TEXT,
                title         TEXT    NOT NULL,
                authors       TEXT,
                abstract      TEXT,
                url           TEXT    NOT NULL,
                published     TEXT,
                updated       TEXT,
                score         REAL    DEFAULT 0.0,
                score_detail  TEXT,
                summary       TEXT    DEFAULT '',
                first_seen    TEXT    NOT NULL,
                last_seen     TEXT    NOT NULL,
                in_report     INTEGER DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_papers_fingerprint
                ON papers(fingerprint);
            CREATE INDEX IF NOT EXISTS idx_papers_score
                ON papers(score DESC);
            CREATE INDEX IF NOT EXISTS idx_papers_first_seen
                ON papers(first_seen);

            CREATE TABLE IF NOT EXISTS url_watch_snapshots (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                target_name   TEXT    NOT NULL,
                url           TEXT    NOT NULL,
                content_hash  TEXT    NOT NULL,
                snapshot_text TEXT,
                captured_at   TEXT    NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_snapshots_target
                ON url_watch_snapshots(target_name, captured_at DESC);

            CREATE TABLE IF NOT EXISTS run_log (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at    TEXT    NOT NULL,
                finished_at   TEXT,
                papers_found  INTEGER DEFAULT 0,
                papers_new    INTEGER DEFAULT 0,
                status        TEXT    DEFAULT 'running'
            );
            """
        )
        self.conn.commit()

    # -- papers --------------------------------------------------------------

    def upsert_paper(self, paper: Paper) -> bool:
        """新規なら INSERT して True、既存なら UPDATE して False を返す。"""
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        cur = self.conn.cursor()
        cur.execute("SELECT id FROM papers WHERE fingerprint = ?", (paper.fingerprint,))
        row = cur.fetchone()
        if row is None:
            cur.execute(
                """INSERT INTO papers
                   (fingerprint, source, arxiv_id, title, authors, abstract,
                    url, published, updated, score, score_detail, summary,
                    first_seen, last_seen)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    paper.fingerprint,
                    paper.source,
                    paper.arxiv_id,
                    paper.title,
                    paper.authors,
                    paper.abstract,
                    paper.url,
                    paper.published,
                    paper.updated,
                    paper.score,
                    json.dumps(paper.score_detail, ensure_ascii=False),
                    paper.summary,
                    now,
                    now,
                ),
            )
            self.conn.commit()
            return True
        else:
            cur.execute(
                """UPDATE papers
                   SET last_seen=?, score=?, score_detail=?, summary=?
                   WHERE fingerprint=?""",
                (
                    now,
                    paper.score,
                    json.dumps(paper.score_detail, ensure_ascii=False),
                    paper.summary,
                    paper.fingerprint,
                ),
            )
            self.conn.commit()
            return False

    def get_top_papers(
        self, min_score: float, limit: int, since: str
    ) -> list[sqlite3.Row]:
        cur = self.conn.cursor()
        cur.execute(
            """SELECT * FROM papers
               WHERE first_seen >= ? AND score >= ?
               ORDER BY score DESC
               LIMIT ?""",
            (since, min_score, limit),
        )
        return cur.fetchall()

    # -- url_watch -----------------------------------------------------------

    def get_latest_snapshot(self, target_name: str) -> sqlite3.Row | None:
        cur = self.conn.cursor()
        cur.execute(
            """SELECT * FROM url_watch_snapshots
               WHERE target_name = ?
               ORDER BY captured_at DESC LIMIT 1""",
            (target_name,),
        )
        return cur.fetchone()

    def save_url_snapshot(
        self, target_name: str, url: str, content_hash: str, snapshot_text: str
    ) -> bool:
        """保存し、前回から変更があれば True を返す。"""
        prev = self.get_latest_snapshot(target_name)
        changed = prev is None or prev["content_hash"] != content_hash
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.conn.execute(
            """INSERT INTO url_watch_snapshots
               (target_name, url, content_hash, snapshot_text, captured_at)
               VALUES (?,?,?,?,?)""",
            (target_name, url, content_hash, snapshot_text, now),
        )
        self.conn.commit()
        return changed

    # -- run_log -------------------------------------------------------------

    def log_run_start(self) -> int:
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        cur = self.conn.execute(
            "INSERT INTO run_log (started_at) VALUES (?)", (now,)
        )
        self.conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def log_run_end(
        self, run_id: int, papers_found: int, papers_new: int, status: str
    ) -> None:
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.conn.execute(
            """UPDATE run_log
               SET finished_at=?, papers_found=?, papers_new=?, status=?
               WHERE id=?""",
            (now, papers_found, papers_new, status, run_id),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()


# ---------------------------------------------------------------------------
# Fetchers
# ---------------------------------------------------------------------------


class ArxivApiFetcher:
    """arXiv API (Atom XML) から論文を取得する。"""

    BASE_URL = "https://export.arxiv.org/api/query"
    NS = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }

    def __init__(self, config: Config) -> None:
        cfg = config.arxiv_api
        self.category: str = cfg["category"]
        self.max_results: int = cfg["max_results"]
        self.rate_limit: float = cfg.get("rate_limit_sec", 3.0)
        self.lookback: int = config.lookback_days

    def fetch(self) -> list[Paper]:
        papers: list[Paper] = []
        start = 0
        batch_size = min(200, self.max_results)
        cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
            days=self.lookback
        )

        while start < self.max_results:
            params = {
                "search_query": f"cat:{self.category}",
                "start": start,
                "max_results": batch_size,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }
            resp = fetch_with_retry(self.BASE_URL, params=params)
            batch = self._parse_atom(resp.text, cutoff)
            papers.extend(batch)
            if len(batch) < batch_size:
                break
            start += batch_size
            time.sleep(self.rate_limit)

        return papers

    def _parse_atom(
        self, xml_text: str, cutoff: datetime.datetime
    ) -> list[Paper]:
        root = ET.fromstring(xml_text)
        papers: list[Paper] = []

        for entry in root.findall("atom:entry", self.NS):
            title = entry.findtext("atom:title", "", self.NS).strip()
            title = re.sub(r"\s+", " ", title)

            published = entry.findtext("atom:published", "", self.NS)
            if published:
                pub_dt = datetime.datetime.fromisoformat(
                    published.replace("Z", "+00:00")
                )
                if pub_dt < cutoff:
                    continue

            link = ""
            for lnk in entry.findall("atom:link", self.NS):
                if lnk.get("type") == "text/html":
                    link = lnk.get("href", "")
                    break
            if not link:
                link = entry.findtext("atom:id", "", self.NS)

            arxiv_id = link.split("/abs/")[-1] if "/abs/" in link else None

            authors_els = entry.findall("atom:author/atom:name", self.NS)
            authors = ", ".join(a.text.strip() for a in authors_els if a.text)

            abstract = entry.findtext("atom:summary", "", self.NS).strip()
            abstract = re.sub(r"\s+", " ", abstract)

            updated = entry.findtext("atom:updated", "", self.NS)

            fp = make_fingerprint(link, title)
            papers.append(
                Paper(
                    fingerprint=fp,
                    source="arxiv_api",
                    arxiv_id=arxiv_id,
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    url=link,
                    published=published,
                    updated=updated,
                )
            )
        return papers


class RssFetcher:
    """RSS フィードから論文を取得する。"""

    def __init__(self, config: Config) -> None:
        self.feeds = config.rss_feeds

    def fetch(self) -> list[Paper]:
        papers: list[Paper] = []
        for feed_cfg in self.feeds:
            try:
                parsed = feedparser.parse(feed_cfg["url"], agent=USER_AGENT)
            except Exception as e:
                logger.warning("RSS 取得失敗 [%s]: %s", feed_cfg["name"], e)
                continue

            for entry in parsed.entries:
                title = entry.get("title", "").strip()
                link = entry.get("link", "")
                desc = entry.get("description", "") or entry.get("summary", "")
                abstract = self._extract_abstract(desc)
                authors = entry.get("author", "")

                fp = make_fingerprint(link, title)
                papers.append(
                    Paper(
                        fingerprint=fp,
                        source="rss",
                        arxiv_id=self._extract_arxiv_id(link),
                        title=title,
                        authors=authors,
                        abstract=abstract,
                        url=link,
                    )
                )
        return papers

    @staticmethod
    def _extract_abstract(description: str) -> str:
        soup = BeautifulSoup(description, "html.parser")
        return soup.get_text(separator=" ").strip()

    @staticmethod
    def _extract_arxiv_id(url: str) -> str | None:
        m = re.search(r"(\d{4}\.\d{4,5})(v\d+)?", url)
        return m.group(0) if m else None


class UrlWatcher:
    """指定 URL の変更を検出する。"""

    def __init__(self, config: Config, db: Database) -> None:
        self.targets = config.url_watch_targets
        self.db = db

    def check_all(self) -> list[dict]:
        results: list[dict] = []
        for target in self.targets:
            name = target["name"]
            url = target["url"]
            selector = target.get("css_selector", "body")

            try:
                resp = fetch_with_retry(url)
            except requests.RequestException as e:
                logger.warning("URL 監視失敗 [%s]: %s", name, e)
                results.append(
                    {"name": name, "url": url, "changed": False, "snippet": "", "error": str(e)}
                )
                continue

            soup = BeautifulSoup(resp.text, "html.parser")
            elem = soup.select_one(selector)
            text = elem.get_text(separator="\n").strip() if elem else ""
            content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

            changed = self.db.save_url_snapshot(name, url, content_hash, text)
            results.append(
                {"name": name, "url": url, "changed": changed, "snippet": text[:500]}
            )
        return results


# ---------------------------------------------------------------------------
# LLM Scorer (Claude Code CLI)
# ---------------------------------------------------------------------------


class LLMScorer:
    """Claude Code CLI をサブプロセスとして起動し、論文をスコアリング・要約する。"""

    def __init__(self, config: Config) -> None:
        llm_cfg = config.llm
        self.command: str = llm_cfg.get("command", "claude")
        self.model: str = llm_cfg.get("model", "opus")
        self.max_tokens: int = llm_cfg.get("max_tokens", 4096)
        self.batch_size: int = llm_cfg.get("batch_size", 5)

        self.criteria: list[dict] = config.scoring["criteria"]
        self.max_score: float = config.scoring["max_score"]

        # CLI が利用可能か確認
        if not shutil.which(self.command):
            raise FileNotFoundError(
                f"コマンド '{self.command}' が見つかりません。"
                f"Claude Code がインストールされているか確認してください。"
            )

    def score_papers(self, papers: list[Paper]) -> list[Paper]:
        """論文リストをバッチでスコアリングし、score / score_detail / summary を設定する。"""
        for i in range(0, len(papers), self.batch_size):
            batch = papers[i : i + self.batch_size]
            self._score_batch(batch)
        return papers

    def _score_batch(self, batch: list[Paper]) -> None:
        """1 バッチ分を 1 回の Claude Code CLI 呼び出しでスコアリングする。"""
        prompt = self._build_prompt(batch)

        try:
            result = subprocess.run(
                [
                    self.command,
                    "--print",
                    "--model", self.model,
                    "--output-format", "json",
                    "--max-tokens", str(self.max_tokens),
                    "--dangerously-skip-permissions",
                    "--no-session-persistence",
                    prompt,
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode != 0:
                logger.error(
                    "Claude Code CLI エラー (exit %d): %s",
                    result.returncode,
                    result.stderr[:500],
                )
                return

            # --output-format json は {"type":"result","result":"..."} を返す
            output = json.loads(result.stdout)
            content = output.get("result", "")

            # JSON ブロックを抽出 (```json ... ``` またはそのまま)
            parsed = self._extract_json(content)
            if parsed is None:
                logger.error("Claude Code の応答から JSON をパースできませんでした")
                logger.debug("応答内容: %s", content[:1000])
                return

        except subprocess.TimeoutExpired:
            logger.error("Claude Code CLI がタイムアウトしました")
            return
        except (json.JSONDecodeError, KeyError) as e:
            logger.error("Claude Code CLI の出力パースに失敗: %s", e)
            logger.debug("stdout: %s", result.stdout[:1000] if result else "")
            return

        papers_results = parsed.get("papers", [])
        for paper, paper_result in zip(batch, papers_results):
            detail = {}
            total = 0.0
            for criterion in self.criteria:
                name = criterion["name"]
                pts = float(paper_result.get("scores", {}).get(name, 0))
                pts = max(0.0, min(pts, criterion["max_points"]))
                detail[name] = pts
                total += pts
            paper.score = min(total, self.max_score)
            paper.score_detail = detail
            paper.summary = paper_result.get("summary", "")

    @staticmethod
    def _extract_json(text: str) -> dict | None:
        """テキストから JSON オブジェクトを抽出する。

        Claude の応答が ```json ... ``` で囲まれている場合と、
        そのまま JSON の場合の両方に対応する。
        """
        # ```json ... ``` ブロックを探す
        m = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1).strip())
            except json.JSONDecodeError:
                pass

        # そのまま JSON として解析を試みる
        # 最初の { から最後の } までを抽出
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass

        return None

    def _build_prompt(self, batch: list[Paper]) -> str:
        criteria_text = "\n".join(
            f"  - {c['name']} (0〜{c['max_points']}点): {c['description']}"
            for c in self.criteria
        )

        papers_text = ""
        for i, paper in enumerate(batch):
            abstract_short = paper.abstract[:1500] if paper.abstract else "(no abstract)"
            papers_text += (
                f"--- Paper {i} ---\n"
                f"Title: {paper.title}\n"
                f"Abstract: {abstract_short}\n\n"
            )

        return f"""\
以下の論文をスコアリングしてください。

評価基準:
{criteria_text}

各論文について日本語で1〜2文の簡潔な要約も付けてください。

出力は必ず以下の JSON 形式のみで返してください（説明文は不要です）:
{{
  "papers": [
    {{
      "index": 0,
      "scores": {{
        "frontier_technology": 1.5,
        "code_available": 0.0,
        "dataset_benchmark": 2.0,
        "real_world_validation": 1.0,
        "safety_robustness": 0.0
      }},
      "summary": "日本語の要約"
    }}
  ]
}}

各スコアは 0 から各基準の max_points の範囲で、0.5 刻みで付けてください。
根拠が明確でないものは 0 としてください。

{papers_text}"""


# ---------------------------------------------------------------------------
# ReportGenerator
# ---------------------------------------------------------------------------


class ReportGenerator:
    """Markdown レポートを生成する。"""

    def __init__(self, config: Config) -> None:
        self.output_dir = Path(config.report["output_dir"])
        self.top_n: int = config.report["top_n"]
        self.filename_format: str = config.report["filename_format"]

    def generate(
        self,
        papers: list[sqlite3.Row],
        url_watch_results: list[dict],
        run_stats: dict,
    ) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        today = datetime.date.today().isoformat()
        filename = self.filename_format.replace("{date}", today)
        filepath = self.output_dir / filename

        lines: list[str] = []
        lines.append("# Research Agent Weekly Report")
        lines.append("")
        lines.append(f"**生成日時:** {datetime.datetime.now().isoformat()}")
        lines.append(f"**観測期間:** 過去 {run_stats['lookback_days']} 日間")
        lines.append(f"**候補論文数:** {run_stats['total_found']} 件")
        lines.append(f"**新規論文数:** {run_stats['new_papers']} 件")
        lines.append(
            f"**レポート掲載:** 上位 {len(papers)} 件 "
            f"(スコア {run_stats['min_score']}+)"
        )
        lines.append("")

        # -- スコア上位論文 --
        lines.append("## スコア上位論文")
        lines.append("")

        for i, row in enumerate(papers, 1):
            detail = json.loads(row["score_detail"]) if row["score_detail"] else {}
            tags = [k for k, v in detail.items() if v > 0]
            tag_str = ", ".join(tags) if tags else "-"

            lines.append(f"### {i}. [{row['title']}]({row['url']})")
            lines.append("")
            lines.append(f"- **スコア:** {row['score']:.1f} / 10.0")
            lines.append(f"- **該当項目:** {tag_str}")

            if row["authors"]:
                author_list = row["authors"].split(", ")
                if len(author_list) > 3:
                    display = ", ".join(author_list[:3]) + " et al."
                else:
                    display = row["authors"]
                lines.append(f"- **著者:** {display}")

            summary = row["summary"] if row["summary"] else ""
            if summary:
                lines.append(f"- **要約:** {summary}")
            elif row["abstract"]:
                short = row["abstract"][:200]
                if len(row["abstract"]) > 200:
                    short += "..."
                lines.append(f"- **概要:** {short}")
            lines.append("")

        # -- URL 監視 --
        if url_watch_results:
            lines.append("## URL 監視")
            lines.append("")
            for uw in url_watch_results:
                status = "**更新あり**" if uw["changed"] else "変更なし"
                lines.append(f"- [{uw['name']}]({uw['url']}): {status}")
                if uw.get("error"):
                    lines.append(f"  - エラー: {uw['error']}")
                elif uw["changed"] and uw["snippet"]:
                    snippet = uw["snippet"][:300].replace("\n", " ")
                    lines.append(f"  - スニペット: {snippet}...")
            lines.append("")

        # -- フッター --
        lines.append("---")
        lines.append("*Generated by research-agent*")

        filepath.write_text("\n".join(lines), encoding="utf-8")
        return filepath


# ---------------------------------------------------------------------------
# Agent (オーケストレーター)
# ---------------------------------------------------------------------------


class Agent:
    """全処理を統括するメインクラス。"""

    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH) -> None:
        self.config = Config(config_path)

        logging.basicConfig(
            level=getattr(logging, self.config.log_level, logging.INFO),
            format="%(asctime)s [%(levelname)s] %(message)s",
        )

        self.db = Database(self.config.database_path)
        self.scorer = LLMScorer(self.config)
        self.reporter = ReportGenerator(self.config)

    def run(self, dry_run: bool = False) -> Path | None:
        """メイン実行フロー。レポートファイルパスを返す。"""
        run_id = self.db.log_run_start()
        total_found = 0
        new_count = 0

        try:
            # --- Step 1: 論文取得 ---
            all_papers: list[Paper] = []

            if self.config.arxiv_api.get("enabled", True):
                logger.info("arXiv API から取得中...")
                fetcher = ArxivApiFetcher(self.config)
                arxiv_papers = fetcher.fetch()
                logger.info("arXiv API: %d 件取得", len(arxiv_papers))
                all_papers.extend(arxiv_papers)

            if self.config.rss_enabled:
                logger.info("RSS から取得中...")
                rss_fetcher = RssFetcher(self.config)
                rss_papers = rss_fetcher.fetch()
                logger.info("RSS: %d 件取得", len(rss_papers))
                all_papers.extend(rss_papers)

            total_found = len(all_papers)

            # --- Step 2: 重複排除 (DB で既存チェック) ---
            new_papers: list[Paper] = []
            for p in all_papers:
                cur = self.db.conn.cursor()
                cur.execute(
                    "SELECT id FROM papers WHERE fingerprint = ?", (p.fingerprint,)
                )
                if cur.fetchone() is None:
                    new_papers.append(p)

            logger.info("新規論文: %d / %d 件", len(new_papers), total_found)

            # --- Step 3: LLM スコアリング (新規のみ) ---
            if new_papers:
                logger.info(
                    "Claude Code (%s) でスコアリング中 (%d 件)...",
                    self.scorer.model,
                    len(new_papers),
                )
                self.scorer.score_papers(new_papers)

            # --- Step 4: DB 保存 ---
            for paper in new_papers:
                is_new = self.db.upsert_paper(paper)
                if is_new:
                    new_count += 1

            logger.info("DB 保存完了: 新規 %d 件", new_count)

            # --- Step 5: URL 監視 ---
            url_watch_results: list[dict] = []
            if self.config.url_watch_enabled:
                logger.info("URL 監視チェック中...")
                watcher = UrlWatcher(self.config, self.db)
                url_watch_results = watcher.check_all()

            if dry_run:
                logger.info("dry-run モード: レポート生成をスキップ")
                self.db.log_run_end(run_id, total_found, new_count, "completed")
                return None

            # --- Step 6: レポート生成 ---
            cutoff = (
                datetime.datetime.now(datetime.timezone.utc)
                - datetime.timedelta(days=self.config.lookback_days)
            )
            top_papers = self.db.get_top_papers(
                min_score=self.config.scoring["min_report_score"],
                limit=self.config.report["top_n"],
                since=cutoff.isoformat(),
            )

            run_stats = {
                "lookback_days": self.config.lookback_days,
                "total_found": total_found,
                "new_papers": new_count,
                "min_score": self.config.scoring["min_report_score"],
            }
            report_path = self.reporter.generate(top_papers, url_watch_results, run_stats)
            logger.info("レポート生成完了: %s", report_path)

            # レポートに含めた論文をマーク
            for row in top_papers:
                self.db.conn.execute(
                    "UPDATE papers SET in_report = 1 WHERE id = ?", (row["id"],)
                )
            self.db.conn.commit()

            self.db.log_run_end(run_id, total_found, new_count, "completed")
            return report_path

        except Exception as e:
            logger.error("実行エラー: %s", e, exc_info=True)
            self.db.log_run_end(run_id, total_found, new_count, "error")
            raise
        finally:
            self.db.close()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="research-agent: ロボット研究分野の自動リサーチエージェント"
    )
    parser.add_argument(
        "-c",
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="設定ファイルのパス (デフォルト: config.yaml)",
    )
    parser.add_argument(
        "-m",
        "--model",
        choices=["opus", "sonnet", "haiku"],
        default=None,
        help="使用するモデル (config.yaml の設定を上書き)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="レポートを生成せず、取得・スコアリングのみ実行",
    )
    args = parser.parse_args()

    # モデルをコマンドライン引数で上書き
    agent = Agent(config_path=args.config)
    if args.model:
        agent.scorer.model = args.model

    report_path = agent.run(dry_run=args.dry_run)
    if report_path:
        print(f"レポート: {report_path}")
    else:
        print("dry-run 完了")


if __name__ == "__main__":
    main()
