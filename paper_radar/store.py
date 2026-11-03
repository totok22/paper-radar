from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .models import Paper, norm_title

SCHEMA = """
CREATE TABLE IF NOT EXISTS pushed (
  key TEXT PRIMARY KEY, norm_title TEXT, title TEXT, link TEXT, s2_ref TEXT,
  topic TEXT, score REAL, pushed_at TEXT, payload TEXT
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

    # --- history ---------------------------------------------------------
    def seen_keys(self) -> set[str]:
        return {r[0] for r in self.db.execute("SELECT key FROM pushed")}

    def seen_titles(self) -> set[str]:
        return {r[0] for r in self.db.execute("SELECT norm_title FROM pushed")}

    def mark_pushed(self, papers: list[Paper]) -> None:
        for p in papers:
            self.db.execute(
                "INSERT OR REPLACE INTO pushed VALUES (?,?,?,?,?,?,?,?,?)",
                (p.key, norm_title(p.title), p.title, p.link, p.s2_ref, p.best_topic,
                 p.final_score, now(), json.dumps({"llm": p.llm, "signals": p.signals}, ensure_ascii=False)),
            )
        self.db.commit()

    def find_pushed(self, needle: str):
        q = f"%{needle.lower()}%"
        return self.db.execute(
            "SELECT * FROM pushed WHERE lower(key) LIKE ? OR lower(title) LIKE ? ORDER BY pushed_at DESC",
            (q, q)).fetchall()

    def recent_pushed(self, limit: int = 30):
        return self.db.execute("SELECT * FROM pushed ORDER BY pushed_at DESC, score DESC LIMIT ?", (limit,)).fetchall()

    # --- feedback --------------------------------------------------------
    def add_feedback(self, key: str, label: int, topic: str, title: str, s2_ref: str | None) -> None:
        self.db.execute("INSERT OR REPLACE INTO feedback VALUES (?,?,?,?,?,?)",
                        (key, label, topic, title, s2_ref, now()))
        self.db.commit()

    def feedback_rows(self):
        return self.db.execute("SELECT * FROM feedback").fetchall()

    # --- cache -----------------------------------------------------------
    def cache_get(self, k: str):
        r = self.db.execute("SELECT v FROM cache WHERE k=?", (k,)).fetchone()
        return json.loads(r[0]) if r else None

    def cache_set(self, k: str, v) -> None:
        self.db.execute("INSERT OR REPLACE INTO cache VALUES (?,?,?)", (k, json.dumps(v, ensure_ascii=False), now()))
        self.db.commit()

    def log_run(self, stats: dict) -> None:
        self.db.execute("INSERT INTO runs(at, stats) VALUES (?,?)", (now(), json.dumps(stats, ensure_ascii=False)))
        self.db.commit()
