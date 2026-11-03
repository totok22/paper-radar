from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .models import Paper, norm_title

SCHEMA = """
CREATE TABLE IF NOT EXISTS pushed (
  key TEXT PRIMARY KEY, norm_title TEXT, title TEXT, link TEXT, s2_ref TEXT,
  topic TEXT, score REAL, pushed_at TEXT, payload TEXT,
  arxiv_id TEXT, doi TEXT
);
CREATE INDEX IF NOT EXISTS idx_pushed_title ON pushed(norm_title);
CREATE TABLE IF NOT EXISTS feedback (
  key TEXT PRIMARY KEY, label INTEGER, topic TEXT, title TEXT, s2_ref TEXT, created_at TEXT
);
CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS runs (id INTEGER PRIMARY KEY AUTOINCREMENT, at TEXT, stats TEXT);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        cols = {r[1] for r in self.db.execute("PRAGMA table_info(pushed)")}
        if "arxiv_id" not in cols:
            self.db.execute("ALTER TABLE pushed ADD COLUMN arxiv_id TEXT")
        if "doi" not in cols:
            self.db.execute("ALTER TABLE pushed ADD COLUMN doi TEXT")
        self.db.commit()

    # --- history ---------------------------------------------------------
    def seen_keys(self) -> set[str]:
        keys = {r[0] for r in self.db.execute("SELECT key FROM pushed")}
        for r in self.db.execute("SELECT arxiv_id, doi FROM pushed"):
            if r[0]:
                keys.add(f"arxiv:{r[0]}")
            if r[1]:
                keys.add(f"doi:{str(r[1]).lower()}")
        return keys

    def seen_titles(self) -> set[str]:
        return {r[0] for r in self.db.execute("SELECT norm_title FROM pushed")}

    def mark_pushed(self, papers: list[Paper]) -> None:
        for p in papers:
            payload = {
                "llm": p.llm, "signals": p.signals,
                "pdf_url": p.pdf_link, "code_url": p.code_url,
                "direction_topic": p.best_topic,
            }
            self.db.execute(
                """INSERT OR REPLACE INTO pushed
                   (key, norm_title, title, link, s2_ref, topic, score, pushed_at, payload, arxiv_id, doi)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (p.key, norm_title(p.title), p.title, p.link, p.s2_ref, p.best_topic,
                 p.final_score, now(), json.dumps(payload, ensure_ascii=False), p.arxiv_id, p.doi),
            )
        self.db.commit()

    def find_pushed(self, needle: str):
        q = f"%{needle.lower()}%"
        return self.db.execute(
            "SELECT * FROM pushed WHERE lower(key) LIKE ? OR lower(title) LIKE ? ORDER BY pushed_at DESC",
            (q, q)).fetchall()

    def recent_pushed(self, limit: int = 30):
        return self.db.execute(
            "SELECT * FROM pushed ORDER BY pushed_at DESC, score DESC LIMIT ?", (limit,)).fetchall()

    def pushed_since(self, days: int):
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        return self.db.execute(
            "SELECT * FROM pushed WHERE pushed_at >= ? ORDER BY score DESC", (cutoff,)).fetchall()

    # --- feedback --------------------------------------------------------
    def add_feedback(self, key: str, label: int, topic: str, title: str, s2_ref: str | None) -> None:
        self.db.execute("INSERT OR REPLACE INTO feedback VALUES (?,?,?,?,?,?)",
                        (key, label, topic, title, s2_ref, now()))
        self.db.commit()

    def feedback_rows(self):
        return self.db.execute("SELECT * FROM feedback").fetchall()

    # --- cache -----------------------------------------------------------
    def cache_get(self, k: str, max_age_days: float | None = None, *, empty_max_age_days: float | None = None):
        r = self.db.execute("SELECT v, created_at FROM cache WHERE k=?", (k,)).fetchone()
        if not r:
            return None
        try:
            value = json.loads(r[0])
        except json.JSONDecodeError:
            return None
        age_days = None
        if r[1]:
            try:
                created = datetime.fromisoformat(r[1])
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                age_days = (datetime.now(timezone.utc) - created).total_seconds() / 86400
            except ValueError:
                age_days = None
        empty = value in ("", None) or (isinstance(value, dict) and not value.get("url"))
        if empty and empty_max_age_days is not None and age_days is not None and age_days > empty_max_age_days:
            return None
        if max_age_days is not None and age_days is not None and age_days > max_age_days:
            return None
        return value

    def cache_set(self, k: str, v) -> None:
        self.cache_set_many([(k, v)])

    def cache_set_many(self, items: list[tuple[str, object]]) -> None:
        if not items:
            return
        ts = now()
        self.db.executemany(
            "INSERT OR REPLACE INTO cache VALUES (?,?,?)",
            [(k, json.dumps(v), ts) for k, v in items],
        )
        self.db.commit()

    def log_run(self, stats: dict) -> None:
        self.db.execute("INSERT INTO runs(at, stats) VALUES (?,?)", (now(), json.dumps(stats, ensure_ascii=False)))
        self.db.commit()
