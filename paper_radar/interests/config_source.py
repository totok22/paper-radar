from __future__ import annotations

from .base import InterestSource, Topic


class ConfigInterestSource(InterestSource):
    """Topics defined in config.yaml (the primary, Zotero-free source)."""

    name = "config"

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def contribute(self, topics: dict[str, Topic]) -> None:
        for t in self.cfg.get("topics", []):
            pos = t.get("positive", []) or []
            neg = t.get("negative", []) or []
            topics[t["id"]] = Topic(
                id=t["id"],
                name=t["name"],
                direction=t["direction"],
                description=t.get("description", ""),
                query=t.get("query", ""),
                extra_queries=[str(q).strip() for q in (t.get("extra_queries") or []) if str(q).strip()],
                positive_ids=[p["id"] for p in pos if p.get("id")],
                positive_texts=[p.get("title", "") for p in pos],
                negative_ids=[p["id"] for p in neg if isinstance(p, dict) and p.get("id")],
                negative_texts=[p if isinstance(p, str) else p.get("title", "") for p in neg],
            )
