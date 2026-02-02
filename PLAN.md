# research-agent 実装計画書

## 1. プロジェクト構成

仕様では単一ファイル `agent.py` だが、保守性のためモジュール分割を推奨する。ただし「シンプルに保つ」方針を尊重し、**単一ファイル `agent.py` にすべてのクラス/関数を収める**設計とする。将来的に肥大化した場合にのみ分割する。

```
research-agent/
  .gitignore          # 既存 + .agent.db, reports/ を追加
  LICENSE             # 既存
  README.md           # 既存 → 使い方を追記
  PLAN.md             # この計画書
  agent.py            # エージェント本体 (全ロジック)
  config.yaml         # 設定ファイル
  requirements.txt    # 依存パッケージ
  reports/            # 週次レポート出力先 (gitignore)
    .gitkeep
```

## 2. 依存パッケージ (`requirements.txt`)

```
feedparser>=6.0.0
requests>=2.31.0
PyYAML>=6.0
beautifulsoup4>=4.12.0
lxml>=5.0.0
```

標準ライブラリとして `sqlite3`, `hashlib`, `logging`, `datetime`, `re`, `pathlib`, `argparse`, `textwrap`, `xml.etree.ElementTree` を使用。外部 LLM API は使わない（スコアリングはルールベース）。

## 3. `.gitignore` への追加項目

既存の `.gitignore` に以下を追加:

```
.agent.db
reports/*.md
```

## 4. 設定ファイル (`config.yaml`) の完全仕様

```yaml
general:
  lookback_days: 7          # 何日前まで遡るか
  database_path: .agent.db  # SQLiteファイルパス
  log_level: INFO           # DEBUG / INFO / WARNING / ERROR

sources:
  arxiv_api:
    enabled: true
    category: "cs.RO"
    max_results: 200
    base_url: "https://export.arxiv.org/api/query"
    rate_limit_sec: 3.0

  rss:
    enabled: true
    feeds:
      - name: "arXiv cs.RO RSS"
        url: "https://rss.arxiv.org/rss/cs.RO"

  url_watch:
    enabled: true
    targets:
      - name: "ICRA 2025 Awards"
        url: "https://2025.ieee-icra.org/awards/"
        css_selector: "main"  # 監視対象のCSSセレクタ (省略時はbody全体)

scoring:
  max_score: 10.0
  min_report_score: 3.0
  weights:
    keyword_frontier: 2.0       # 先端性キーワード
    code_available: 2.0         # GitHub等コード公開
    dataset_benchmark: 2.0      # データセット/ベンチマーク公開
    real_world: 2.0             # 実機/リアルワールド検証
    safety_robustness: 2.0      # 安全性/堅牢性

  keywords:
    frontier:
      - "foundation model"
      - "large language model"
      - "LLM"
      - "diffusion"
      - "transformer"
      - "world model"
      - "sim-to-real"
      - "zero-shot"
      - "few-shot"
      - "reinforcement learning from human feedback"
      - "RLHF"
      - "vision-language"
      - "VLA"
      - "neural radiance"
      - "NeRF"
      - "gaussian splatting"
    code_indicators:
      - "github.com"
      - "gitlab.com"
      - "bitbucket.org"
      - "code is available"
      - "open source"
      - "code release"
    dataset_indicators:
      - "benchmark"
      - "dataset"
      - "evaluation suite"
      - "leaderboard"
    real_world_indicators:
      - "real robot"
      - "real-world"
      - "hardware experiment"
      - "physical experiment"
      - "deployed"
      - "field test"
    safety_indicators:
      - "safety"
      - "robustness"
      - "robust"
      - "adversarial"
      - "fail-safe"
      - "collision avoidance"
      - "safe reinforcement learning"
      - "constraint"

report:
  output_dir: "reports"
  top_n: 15
  filename_format: "report_{date}.md"  # {date} は YYYY-MM-DD に置換
```

## 5. データベーススキーマ (SQLite)

`agent.py` 内で初回実行時に自動生成する。

### テーブル: `papers`

```sql
CREATE TABLE IF NOT EXISTS papers (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    fingerprint   TEXT    UNIQUE NOT NULL,  -- SHA-256(正規化URL + 正規化タイトル)
    source        TEXT    NOT NULL,         -- 'arxiv_api' | 'rss' | 'url_watch'
    arxiv_id      TEXT,                     -- arXiv ID (例: 2401.12345)
    title         TEXT    NOT NULL,
    authors       TEXT,                     -- カンマ区切り
    abstract      TEXT,
    url           TEXT    NOT NULL,
    published     TEXT,                     -- ISO 8601
    updated       TEXT,                     -- ISO 8601
    score         REAL    DEFAULT 0.0,
    score_detail  TEXT,                     -- JSON: {"keyword_frontier": 2.0, ...}
    first_seen    TEXT    NOT NULL,         -- ISO 8601, INSERT時の現在時刻
    last_seen     TEXT    NOT NULL,         -- ISO 8601, 最後に観測した時刻
    in_report     INTEGER DEFAULT 0        -- レポートに含めたか (0/1)
);

CREATE INDEX IF NOT EXISTS idx_papers_fingerprint ON papers(fingerprint);
CREATE INDEX IF NOT EXISTS idx_papers_score ON papers(score DESC);
CREATE INDEX IF NOT EXISTS idx_papers_first_seen ON papers(first_seen);
```

### テーブル: `url_watch_snapshots`

```sql
CREATE TABLE IF NOT EXISTS url_watch_snapshots (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    target_name   TEXT    NOT NULL,
    url           TEXT    NOT NULL,
    content_hash  TEXT    NOT NULL,         -- SHA-256(抽出テキスト)
    snapshot_text TEXT,                     -- 抽出テキスト (差分表示用)
    captured_at   TEXT    NOT NULL          -- ISO 8601
);

CREATE INDEX IF NOT EXISTS idx_snapshots_target ON url_watch_snapshots(target_name, captured_at DESC);
```

### テーブル: `run_log`

```sql
CREATE TABLE IF NOT EXISTS run_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at    TEXT    NOT NULL,
    finished_at   TEXT,
    papers_found  INTEGER DEFAULT 0,
    papers_new    INTEGER DEFAULT 0,
    status        TEXT    DEFAULT 'running'  -- 'running' | 'completed' | 'error'
);
```

## 6. `agent.py` の内部設計

### 6.1 全体構成

```python
#!/usr/bin/env python3
"""research-agent: ロボット研究分野の自動リサーチエージェント"""

import argparse
import datetime
import hashlib
import json
import logging
import re
import sqlite3
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import feedparser
import requests
import yaml
from bs4 import BeautifulSoup

# --- 定数 ---
DEFAULT_CONFIG_PATH = "config.yaml"
DEFAULT_DB_PATH = ".agent.db"
USER_AGENT = "research-agent/1.0 (academic research bot)"

# --- データクラス ---
# --- Config ---
# --- Database ---
# --- Fetchers ---
# --- Scorer ---
# --- Reporter ---
# --- Agent (オーケストレーター) ---
# --- main ---
```

### 6.2 データクラス

```python
@dataclass
class Paper:
    """論文1件を表すデータオブジェクト"""
    fingerprint: str
    source: str              # 'arxiv_api' | 'rss' | 'url_watch'
    arxiv_id: str | None
    title: str
    authors: str             # カンマ区切り
    abstract: str
    url: str
    published: str | None    # ISO 8601
    updated: str | None      # ISO 8601
    score: float = 0.0
    score_detail: dict = field(default_factory=dict)
```

### 6.3 `Config` クラス

```python
class Config:
    """config.yaml を読み込み、バリデーションを行う"""

    def __init__(self, path: str = DEFAULT_CONFIG_PATH):
        self.path = Path(path)
        self._raw: dict = {}
        self.load()

    def load(self) -> None:
        with open(self.path, "r", encoding="utf-8") as f:
            self._raw = yaml.safe_load(f)
        self._validate()

    def _validate(self) -> None:
        """必須セクションの存在確認、重み合計チェック等"""
        required_sections = ["general", "sources", "scoring", "report"]
        for sec in required_sections:
            if sec not in self._raw:
                raise ValueError(f"config.yaml に '{sec}' セクションがありません")
        weights = self._raw["scoring"]["weights"]
        total = sum(weights.values())
        max_score = self._raw["scoring"]["max_score"]
        if total > max_score + 0.01:
            raise ValueError(f"重みの合計({total})がmax_score({max_score})を超えています")

    @property
    def lookback_days(self) -> int: ...
    @property
    def database_path(self) -> str: ...
    @property
    def arxiv_api(self) -> dict: ...
    @property
    def rss_feeds(self) -> list[dict]: ...
    @property
    def url_watch_targets(self) -> list[dict]: ...
    @property
    def scoring(self) -> dict: ...
    @property
    def report(self) -> dict: ...
```

### 6.4 `Database` クラス

```python
class Database:
    """SQLiteラッパー。テーブル作成、UPSERT、クエリを提供"""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_tables()

    def _init_tables(self) -> None:
        """テーブルが存在しなければ作成"""

    def upsert_paper(self, paper: Paper) -> bool:
        """fingerprint重複時はlast_seenとscoreを更新。新規ならINSERT。
        戻り値: 新規挿入ならTrue"""

    def get_top_papers(self, min_score: float, limit: int,
                       since: str) -> list[sqlite3.Row]:
        """指定期間内のスコア上位論文を取得"""

    def save_url_snapshot(self, target_name: str, url: str,
                          content_hash: str, snapshot_text: str) -> bool:
        """URL監視スナップショットを保存。content_hashが前回と同じならFalse"""

    def get_latest_snapshot(self, target_name: str) -> sqlite3.Row | None:
        """指定ターゲットの最新スナップショットを取得"""

    def log_run_start(self) -> int:
        """実行ログ開始。run_idを返す"""

    def log_run_end(self, run_id: int, papers_found: int,
                    papers_new: int, status: str) -> None:
        """実行ログ終了"""

    def close(self) -> None:
        self.conn.close()
```

### 6.5 Fetcher 群

```python
def make_fingerprint(url: str, title: str) -> str:
    """URL + タイトルから正規化fingerprint(SHA-256)を生成"""
    norm_url = re.sub(r"https?://", "", url).strip().rstrip("/").lower()
    norm_title = re.sub(r"\s+", " ", title).strip().lower()
    raw = f"{norm_url}|{norm_title}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def fetch_with_retry(url: str, params: dict = None,
                     max_retries: int = 3, timeout: int = 30,
                     backoff_base: float = 2.0) -> requests.Response:
    """指数バックオフ付きHTTP GET"""
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params,
                                headers={"User-Agent": USER_AGENT},
                                timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            if attempt == max_retries - 1:
                raise
            wait = backoff_base ** attempt
            logging.warning("リトライ %d/%d (%.1fs後): %s",
                           attempt + 1, max_retries, wait, e)
            time.sleep(wait)


class ArxivApiFetcher:
    """arXiv API (Atom XML) から論文を取得"""

    BASE_URL = "https://export.arxiv.org/api/query"
    NS = {"atom": "http://www.w3.org/2005/Atom",
          "arxiv": "http://arxiv.org/schemas/atom"}

    def __init__(self, config: Config): ...

    def fetch(self) -> list[Paper]:
        """arXiv APIを呼び出し、Paperリストを返す。
        ページネーションでmax_results件まで取得"""

    def _parse_atom(self, xml_text: str,
                    cutoff: datetime.datetime) -> list[Paper]:
        """Atom XMLをパースしてPaperリストに変換"""


class RssFetcher:
    """RSS (feedparser) から論文を取得"""

    def __init__(self, config: Config): ...

    def fetch(self) -> list[Paper]: ...

    @staticmethod
    def _extract_abstract(description: str) -> str:
        """HTMLタグを除去してabstractテキストを取得"""

    @staticmethod
    def _extract_arxiv_id(url: str) -> str | None: ...


class UrlWatcher:
    """指定URLのページ内容を監視し、変更を検出する"""

    def __init__(self, config: Config, db: Database): ...

    def check_all(self) -> list[dict]:
        """全ターゲットをチェックし、変更があったもののリストを返す
        戻り値: [{"name": ..., "url": ..., "changed": bool, "snippet": str}, ...]"""
```

### 6.6 スコアリングエンジン

```python
class Scorer:
    """ルールベーススコアリング (max 10点)"""

    def __init__(self, config: Config): ...

    def score(self, paper: Paper) -> tuple[float, dict]:
        """論文をスコアリングし、(合計スコア, 詳細dict) を返す

        5軸それぞれ二値判定:
        1. keyword_frontier: 先端性キーワードの有無
        2. code_available: GitHub等コード公開の有無
        3. dataset_benchmark: データセット/ベンチマーク公開の有無
        4. real_world: 実機/リアルワールド検証要素の有無
        5. safety_robustness: 安全性/堅牢性に関する記述の有無
        """

    @staticmethod
    def _has_any(text: str, keywords: list[str]) -> bool:
        """テキスト中にキーワードのいずれかが含まれるか"""
```

**設計意図:**
- 各軸は二値（該当/非該当）で weight が付く。全5軸 × 2.0 = 最大10.0点
- 将来的に段階評価（複数キーワードヒットで加点）に拡張可能
- keywords は config.yaml で自由に追加/削除できる

### 6.7 レポート生成

```python
class ReportGenerator:
    """Markdownレポートを生成"""

    def __init__(self, config: Config): ...

    def generate(self, papers: list[sqlite3.Row],
                 url_watch_results: list[dict],
                 run_stats: dict) -> Path:
        """Markdownレポートを生成し、ファイルパスを返す"""
```

**レポート構成:**

```markdown
# Research Agent Weekly Report

**生成日時:** 2026-02-02T09:00:00
**観測期間:** 過去 7 日間
**候補論文数:** 150 件
**新規論文数:** 42 件
**レポート掲載:** 上位 15 件 (スコア 3.0+)

## スコア上位論文

### 1. [論文タイトル](https://arxiv.org/abs/xxxx.xxxxx)

- **スコア:** 8.0 / 10.0
- **該当項目:** keyword_frontier, code_available, real_world, safety_robustness
- **著者:** Alice, Bob, Charlie et al.
- **概要:** This paper proposes...

### 2. ...

## URL監視

- [ICRA 2025 Awards](https://...): **更新あり**
  - スニペット: ...

---
*Generated by research-agent*
```

### 6.8 Agent（オーケストレーター）

```python
class Agent:
    """全処理を統括するメインクラス"""

    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH): ...

    def run(self) -> Path:
        """メイン実行フロー:
        1. 各ソースから論文を取得
        2. スコアリング + DB保存（重複排除はfingerprintで自動）
        3. URL監視
        4. レポート生成
        5. レポートに含めた論文をマーク
        レポートファイルパスを返す"""
```

### 6.9 エントリーポイント

```python
def main():
    parser = argparse.ArgumentParser(
        description="research-agent: ロボット研究分野の自動リサーチエージェント")
    parser.add_argument("-c", "--config", default=DEFAULT_CONFIG_PATH,
                        help="設定ファイルのパス (デフォルト: config.yaml)")
    parser.add_argument("--dry-run", action="store_true",
                        help="レポートを生成せず、取得・スコアリングのみ実行")
    args = parser.parse_args()

    agent = Agent(config_path=args.config)
    report_path = agent.run()
    print(f"レポート: {report_path}")


if __name__ == "__main__":
    main()
```

## 7. エラーハンドリング方針

| 箇所 | 方針 |
|---|---|
| arXiv API タイムアウト/HTTPエラー | `requests.exceptions.RequestException` を捕捉。3回リトライ (exponential backoff)。全失敗時は該当ソースをスキップし `logger.error` で記録 |
| RSS パースエラー | `feedparser` の `bozo` フラグをチェック。パース不能時はスキップ |
| URL監視 接続エラー | 個別 target ごとに `try/except`。失敗しても他 target は続行 |
| DB ロックエラー | `sqlite3.OperationalError` を捕捉。`timeout=10` を `connect()` に設定 |
| config.yaml 不備 | `Config._validate()` で起動時に即座に `ValueError` を raise |
| 全体 | `Agent.run()` で最外周を `try/except/finally` で囲み、`run_log` に状態記録 |

## 8. テスト戦略

テストは `agent.py` と同階層に `test_agent.py` を置く。`pytest` で実行。

```python
# test_agent.py のテスト項目

class TestFingerprint:
    def test_same_paper_different_protocol(self):
        """http と https で同じfingerprintになること"""
    def test_title_normalization(self):
        """空白・大小文字の違いが吸収されること"""

class TestScorer:
    def test_max_score(self):
        """全キーワードヒット時にmax_scoreを超えないこと"""
    def test_zero_score(self):
        """無関係な論文が0点になること"""
    def test_partial_match(self):
        """一部キーワードのみヒットした場合のスコア"""

class TestArxivParser:
    def test_parse_sample_atom(self):
        """サンプルAtom XMLを正しくパースできること"""

class TestDatabase:
    def test_upsert_new(self):
        """新規論文がINSERTされること (in-memory DB使用)"""
    def test_upsert_duplicate(self):
        """重複fingerprint時にlast_seenが更新されること"""
    def test_get_top_papers(self):
        """スコア順に正しく取得できること"""

class TestUrlWatcher:
    def test_detect_change(self):
        """コンテンツ変更を検出できること (requestsをモック)"""
    def test_no_change(self):
        """変更なしを正しく判定できること"""

class TestReportGenerator:
    def test_generates_valid_markdown(self):
        """生成されたMarkdownにヘッダーとリンクが含まれること"""

class TestConfig:
    def test_missing_section_raises(self):
        """必須セクション欠落時にValueErrorが発生すること"""
    def test_weight_overflow_raises(self):
        """重みの合計がmax_scoreを超えた場合にエラーになること"""
```

テスト実行: `python -m pytest test_agent.py -v`

外部 API 呼び出しは `unittest.mock.patch` でモック。DB テストは `sqlite3.connect(":memory:")` を使用。

## 9. 実装手順（推奨順序）

| ステップ | 内容 | 依存 |
|---|---|---|
| 1 | `requirements.txt` 作成、`.gitignore` に `.agent.db` と `reports/*.md` 追加 | なし |
| 2 | `config.yaml` 作成 | なし |
| 3 | `agent.py`: `Paper` データクラス、`make_fingerprint()` | なし |
| 4 | `agent.py`: `Config` クラス + バリデーション | Step 2 |
| 5 | `agent.py`: `Database` クラス + テーブル作成 | Step 3 |
| 6 | `agent.py`: `fetch_with_retry()` ヘルパー | なし |
| 7 | `agent.py`: `ArxivApiFetcher` | Step 3, 4, 6 |
| 8 | `agent.py`: `RssFetcher` | Step 3, 4 |
| 9 | `agent.py`: `Scorer` | Step 3, 4 |
| 10 | `agent.py`: `UrlWatcher` | Step 4, 5 |
| 11 | `agent.py`: `ReportGenerator` | Step 4 |
| 12 | `agent.py`: `Agent` オーケストレーター + `main()` | Step 4-11 |
| 13 | `test_agent.py` | Step 3-11 |
| 14 | `README.md` 更新（使い方、cron設定例） | 全体完了後 |
| 15 | `reports/.gitkeep` 作成 | なし |

## 10. cron 設定例

```bash
# 毎週月曜 9:00 JST に実行
0 0 * * 1 cd /path/to/research-agent && .venv/bin/python agent.py >> /var/log/research-agent.log 2>&1
```

## 11. 将来の拡張ポイント

- **段階的スコアリング**: キーワードヒット数に応じた段階評価
- **LLMによる要約**: abstract の日本語要約を追加
- **複数分野対応**: config.yaml に複数カテゴリを定義し、1 agent = 1分野を複数起動
- **Slack/Discord通知**: レポート生成後にWebhookで通知
- **差分レポート**: 前回からの差分のみを報告するモード
