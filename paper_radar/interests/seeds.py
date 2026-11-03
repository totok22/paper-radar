"""Enrich seed papers with title+abstract from Semantic Scholar (cached in SQLite)."""
from __future__ import annotations

import logging

from ..http import request

log = logging.getLogger("paper_radar")


def enrich_seed_texts(topics, store, api_key: str = "") -> int:
    refs = sorted({r for t in topics.values() for r in t.positive_ids + t.negative_ids})
    cache = {r: store.cache_get(f"seed:{r}") for r in refs}
    missing = [r for r, v in cache.items() if v is None]
    if missing:
        headers = {"x-api-key": api_key} if api_key else {}
        r = request("POST", "https://api.semanticscholar.org/graph/v1/paper/batch",
                    params={"fields": "title,abstract"}, json={"ids": missing},
                    headers=headers, retries=5, backoff=6)
        if r is not None and r.status_code == 200:
            for ref, d in zip(missing, r.json()):
                if d:
                    v = f"{d.get('title') or ''}. {d.get('abstract') or ''}"
                    store.cache_set(f"seed:{ref}", v)
                    cache[ref] = v
        else:
            log.warning("seed enrichment failed (S2 %s); using titles only", None if r is None else r.status_code)
    n = 0
    for t in topics.values():
        for ref in t.positive_ids:
            if cache.get(ref):
                t.positive_texts.append(cache[ref][:1500])
                n += 1
    return n
