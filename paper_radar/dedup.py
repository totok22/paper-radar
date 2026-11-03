"""Normalize + dedup by arXiv ID / DOI / fuzzy title, and drop already-pushed."""
from __future__ import annotations

from difflib import SequenceMatcher

from .models import Paper, norm_title


def _merge(a: Paper, b: Paper) -> Paper:
    for f in ("abstract", "venue", "comments", "doi", "arxiv_id", "s2_id", "published", "url"):
        va, vb = getattr(a, f), getattr(b, f)
        if not va or (f == "abstract" and vb and len(vb) > len(va)):
            setattr(a, f, vb)
    a.authors = a.authors or b.authors
    a.categories = list(dict.fromkeys(a.categories + b.categories))
    a.sources = list(dict.fromkeys(a.sources + b.sources))
    return a


def dedup(papers: list[Paper], seen_keys: set[str] = frozenset(), seen_titles: set[str] = frozenset(),
          fuzzy: float = 0.93) -> list[Paper]:
    by_id: dict[str, Paper] = {}
    out: list[Paper] = []
    titles: dict[str, Paper] = {}
    for p in papers:
        if not p.title or len(p.title) < 8:
            continue
        nt = norm_title(p.title)
        ids = [i for i in (p.arxiv_id and f"a:{p.arxiv_id}", p.doi and f"d:{p.doi.lower()}") if i]
        hit = next((by_id[i] for i in ids if i in by_id), None) or titles.get(nt)
        if hit is None and fuzzy < 1:
            prefix = nt[:12]
            for t2, q in titles.items():
                if t2[:12] == prefix and SequenceMatcher(None, nt, t2).ratio() >= fuzzy:
                    hit = q
                    break
        if hit:
            _merge(hit, p)
            for i in ids:
                by_id[i] = hit
            continue
        out.append(p)
        titles[nt] = p
        for i in ids:
            by_id[i] = p
    return [p for p in out
            if p.key not in seen_keys and norm_title(p.title) not in seen_titles
            and not (p.arxiv_id and f"arxiv:{p.arxiv_id}" in seen_keys)
            and not (p.doi and f"doi:{p.doi.lower()}" in seen_keys)]
