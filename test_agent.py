"""research-agent のユニットテスト。"""

from __future__ import annotations

import json
import sqlite3
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent import (
    Agent,
    ArxivApiFetcher,
    Config,
    Database,
    LLMScorer,
    Paper,
    ReportGenerator,
    RssFetcher,
    UrlWatcher,
    make_fingerprint,
)

# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------


class TestFingerprint:
    def test_same_paper_different_protocol(self):
        fp1 = make_fingerprint("http://arxiv.org/abs/2401.00001", "My Paper")
        fp2 = make_fingerprint("https://arxiv.org/abs/2401.00001", "My Paper")
        assert fp1 == fp2

    def test_title_normalization(self):
        fp1 = make_fingerprint("https://arxiv.org/abs/2401.00001", "My  Paper")
        fp2 = make_fingerprint("https://arxiv.org/abs/2401.00001", "my paper")
        assert fp1 == fp2

    def test_different_papers(self):
        fp1 = make_fingerprint("https://arxiv.org/abs/2401.00001", "Paper A")
        fp2 = make_fingerprint("https://arxiv.org/abs/2401.00002", "Paper B")
        assert fp1 != fp2


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------


class TestDatabase:
    def setup_method(self):
        self.db = Database(":memory:")

    def teardown_method(self):
        self.db.close()

    def _make_paper(self, title: str = "Test Paper", url: str = "https://example.com") -> Paper:
        return Paper(
            fingerprint=make_fingerprint(url, title),
            source="arxiv_api",
            arxiv_id="2401.00001",
            title=title,
            authors="Alice, Bob",
            abstract="A test paper about robots.",
            url=url,
            score=5.0,
            score_detail={"frontier_technology": 2.0, "code_available": 1.0},
            summary="テスト論文の要約",
        )

    def test_upsert_new(self):
        paper = self._make_paper()
        assert self.db.upsert_paper(paper) is True

    def test_upsert_duplicate(self):
        paper = self._make_paper()
        self.db.upsert_paper(paper)
        assert self.db.upsert_paper(paper) is False

    def test_get_top_papers(self):
        for i in range(5):
            p = self._make_paper(title=f"Paper {i}", url=f"https://example.com/{i}")
            p.score = float(i)
            self.db.upsert_paper(p)

        rows = self.db.get_top_papers(min_score=2.0, limit=10, since="2000-01-01")
        assert len(rows) == 3
        assert rows[0]["score"] == 4.0

    def test_url_snapshot_change_detection(self):
        changed1 = self.db.save_url_snapshot("test", "https://x.com", "hash1", "text1")
        assert changed1 is True  # first time
        changed2 = self.db.save_url_snapshot("test", "https://x.com", "hash1", "text1")
        assert changed2 is False  # same hash
        changed3 = self.db.save_url_snapshot("test", "https://x.com", "hash2", "text2")
        assert changed3 is True  # different hash

    def test_run_log(self):
        run_id = self.db.log_run_start()
        assert run_id is not None
        self.db.log_run_end(run_id, 100, 50, "completed")
        row = self.db.conn.execute("SELECT * FROM run_log WHERE id = ?", (run_id,)).fetchone()
        assert row["status"] == "completed"
        assert row["papers_found"] == 100


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_valid_config(self, tmp_path: Path):
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(
            textwrap.dedent("""\
            general:
              lookback_days: 7
              database_path: .agent.db
              log_level: INFO
            llm:
              provider: openai
              model: gpt-4o-mini
              api_key_env: OPENAI_API_KEY
            sources:
              arxiv_api:
                enabled: true
                category: cs.RO
                max_results: 200
              rss:
                enabled: false
                feeds: []
              url_watch:
                enabled: false
                targets: []
            scoring:
              max_score: 10.0
              min_report_score: 3.0
              criteria:
                - name: test
                  description: test criterion
                  max_points: 10.0
            report:
              output_dir: reports
              top_n: 15
              filename_format: "report_{date}.md"
            """),
            encoding="utf-8",
        )
        config = Config(str(cfg_path))
        assert config.lookback_days == 7

    def test_missing_section(self, tmp_path: Path):
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text("general:\n  lookback_days: 7\n", encoding="utf-8")
        with pytest.raises(ValueError, match="セクション"):
            Config(str(cfg_path))

    def test_weight_overflow(self, tmp_path: Path):
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(
            textwrap.dedent("""\
            general:
              lookback_days: 7
              database_path: .agent.db
            llm:
              provider: openai
              model: gpt-4o-mini
              api_key_env: OPENAI_API_KEY
            sources:
              arxiv_api:
                enabled: true
                category: cs.RO
                max_results: 200
              rss:
                enabled: false
                feeds: []
              url_watch:
                enabled: false
                targets: []
            scoring:
              max_score: 10.0
              min_report_score: 3.0
              criteria:
                - name: a
                  description: a
                  max_points: 6.0
                - name: b
                  description: b
                  max_points: 6.0
            report:
              output_dir: reports
              top_n: 15
              filename_format: "report_{date}.md"
            """),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="max_score"):
            Config(str(cfg_path))


# ---------------------------------------------------------------------------
# ArxivApiFetcher (parse only)
# ---------------------------------------------------------------------------

SAMPLE_ATOM = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2401.00001v1</id>
    <title>  A Test   Paper  </title>
    <published>2099-01-15T00:00:00Z</published>
    <updated>2099-01-15T00:00:00Z</updated>
    <summary>  This is a test abstract.  </summary>
    <author><name>Alice</name></author>
    <author><name>Bob</name></author>
    <link href="http://arxiv.org/abs/2401.00001v1" type="text/html"/>
  </entry>
</feed>
"""


class TestArxivParser:
    def test_parse_sample_atom(self):
        import datetime

        config = MagicMock()
        config.arxiv_api = {"category": "cs.RO", "max_results": 200, "rate_limit_sec": 0}
        config.lookback_days = 7
        fetcher = ArxivApiFetcher(config)
        cutoff = datetime.datetime(2000, 1, 1, tzinfo=datetime.timezone.utc)
        papers = fetcher._parse_atom(SAMPLE_ATOM, cutoff)
        assert len(papers) == 1
        assert papers[0].title == "A Test Paper"
        assert papers[0].authors == "Alice, Bob"
        assert papers[0].arxiv_id == "2401.00001v1"


# ---------------------------------------------------------------------------
# ReportGenerator
# ---------------------------------------------------------------------------


class TestReportGenerator:
    def test_generates_valid_markdown(self, tmp_path: Path):
        config = MagicMock()
        config.report = {
            "output_dir": str(tmp_path),
            "top_n": 15,
            "filename_format": "report_{date}.md",
        }
        gen = ReportGenerator(config)

        # fake paper rows
        db = Database(":memory:")
        paper = Paper(
            fingerprint="abc",
            source="arxiv_api",
            arxiv_id="2401.00001",
            title="Great Paper",
            authors="Alice",
            abstract="About robots",
            url="https://arxiv.org/abs/2401.00001",
            score=8.0,
            score_detail={"frontier_technology": 2.0},
            summary="ロボットの論文",
        )
        db.upsert_paper(paper)
        rows = db.get_top_papers(min_score=0, limit=10, since="2000-01-01")
        db.close()

        path = gen.generate(
            rows,
            [{"name": "ICRA", "url": "https://example.com", "changed": True, "snippet": "hello"}],
            {"lookback_days": 7, "total_found": 100, "new_papers": 50, "min_score": 3.0},
        )
        text = path.read_text(encoding="utf-8")
        assert "# Research Agent Weekly Report" in text
        assert "Great Paper" in text
        assert "ロボットの論文" in text
        assert "ICRA" in text


# ---------------------------------------------------------------------------
# LLMScorer (mocked)
# ---------------------------------------------------------------------------


class TestLLMScorer:
    def _make_config(self, tmp_path: Path) -> Config:
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(
            textwrap.dedent("""\
            general:
              lookback_days: 7
              database_path: .agent.db
            llm:
              provider: openai
              model: gpt-4o-mini
              api_key_env: TEST_API_KEY
              batch_size: 2
              rate_limit_sec: 0
            sources:
              arxiv_api:
                enabled: false
                category: cs.RO
                max_results: 200
              rss:
                enabled: false
                feeds: []
              url_watch:
                enabled: false
                targets: []
            scoring:
              max_score: 10.0
              min_report_score: 3.0
              criteria:
                - name: frontier_technology
                  description: test
                  max_points: 2.0
                - name: code_available
                  description: test
                  max_points: 2.0
            report:
              output_dir: reports
              top_n: 15
              filename_format: "report_{date}.md"
            """),
            encoding="utf-8",
        )
        return Config(str(cfg_path))

    @patch.dict("os.environ", {"TEST_API_KEY": "sk-test-key"})
    def test_score_papers(self, tmp_path: Path):
        config = self._make_config(tmp_path)
        scorer = LLMScorer(config)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "papers": [
                    {
                        "index": 0,
                        "scores": {"frontier_technology": 1.5, "code_available": 2.0},
                        "summary": "テスト要約",
                    }
                ]
            }
        )
        scorer.client = MagicMock()
        scorer.client.chat.completions.create.return_value = mock_response

        paper = Paper(
            fingerprint="abc",
            source="arxiv_api",
            arxiv_id=None,
            title="Test",
            authors="",
            abstract="Test abstract",
            url="https://example.com",
        )
        scorer.score_papers([paper])

        assert paper.score == 3.5
        assert paper.score_detail["frontier_technology"] == 1.5
        assert paper.score_detail["code_available"] == 2.0
        assert paper.summary == "テスト要約"

    @patch.dict("os.environ", {"TEST_API_KEY": "sk-test-key"})
    def test_score_clamped_to_max(self, tmp_path: Path):
        config = self._make_config(tmp_path)
        scorer = LLMScorer(config)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "papers": [
                    {
                        "index": 0,
                        "scores": {"frontier_technology": 999, "code_available": 999},
                        "summary": "",
                    }
                ]
            }
        )
        scorer.client = MagicMock()
        scorer.client.chat.completions.create.return_value = mock_response

        paper = Paper(
            fingerprint="abc",
            source="arxiv_api",
            arxiv_id=None,
            title="Test",
            authors="",
            abstract="",
            url="https://example.com",
        )
        scorer.score_papers([paper])
        # clamped: 2.0 + 2.0 = 4.0, but total capped at max_score=10.0
        assert paper.score_detail["frontier_technology"] == 2.0
        assert paper.score_detail["code_available"] == 2.0
        assert paper.score == 4.0
