"""Normalize + dedup across arXiv ID / DOI / fuzzy title, and drop already-pushed."""
from __future__ import annotations

from difflib import SequenceMatcher

from .models import Paper, arxiv_from_doi, norm_arxiv, norm_doi, norm_title
from .pdf import annotate_local, looks_like_pdf


def _prefer_pdf(a: str | None, b: str | None) -> str | None:
    if looks_like_pdf(a) and not looks_like_pdf(b):
        return a
    if looks_like_pdf(b) and not looks_like_pdf(a):
        return b
    return a or b


def _merge(a: Paper, b: Paper) -> Paper:
    for f in ("abstract", "venue", "comments", "doi", "arxiv_id", "s2_id", "published", "url", "code_url"):
        va, vb = getattr(a, f), getattr(b, f)
        if not va or (f == "abstract" and vb and len(vb) > len(va)):
            setattr(a, f, vb)
    a.pdf_url = _prefer_pdf(a.pdf_url, b.pdf_url)
    a.authors = a.authors or b.authors
    if len(b.affiliations) > len(a.affiliations):
        a.affiliations = b.affiliations
    if b.citation_count and (not a.citation_count or b.citation_count > a.citation_count):
        a.citation_count = b.citation_count
    a.categories = list(dict.fromkeys(a.categories + b.categories))
    a.sources = list(dict.fromkeys(a.sources + b.sources))
    return a


def _ids(p: Paper) -> list[str]:
    ids: list[str] = []
    aid = norm_arxiv(p.arxiv_id) or arxiv_from_doi(p.doi)
    if aid:
        p.arxiv_id = aid
        ids.append(f"a:{aid}")
    doi = norm_doi(p.doi)
    if doi:
        p.doi = doi
        ids.append(f"d:{doi}")
    return ids


def _seen_hit(p: Paper, seen_keys: set[str]) -> bool:
    if p.key in seen_keys:
        return True
    if p.arxiv_id and f"arxiv:{p.arxiv_id}" in seen_keys:
        return True
    if p.doi and f"doi:{p.doi}" in seen_keys:
        return True
    return False


def dedup(papers: list[Paper], seen_keys: set[str] = frozenset(), seen_titles: set[str] = frozenset(),
          fuzzy: float = 0.93) -> list[Paper]:
    by_id: dict[str, Paper] = {}
    out: list[Paper] = []
    titles: dict[str, Paper] = {}
    for p in papers:
        if not p.title or len(p.title) < 8:
            continue
        nt = norm_title(p.title)
        ids = _ids(p)
        hit = next((by_id[i] for i in ids if i in by_id), None) or titles.get(nt)
        if hit is None and fuzzy < 1 and len(nt) >= 12:
            prefix = nt[:12]
            for t2, q in titles.items():
                if t2[:12] == prefix and SequenceMatcher(None, nt, t2).ratio() >= fuzzy:
                    hit = q
                    break
        if hit is not None:
            _merge(hit, p)
            for i in _ids(hit):
                by_id[i] = hit
            continue
        out.append(p)
        titles[nt] = p
        for i in ids:
            by_id[i] = p
    kept = []
    for p in out:
        annotate_local(p)
        if _seen_hit(p, seen_keys) or norm_title(p.title) in seen_titles:
            continue
        kept.append(p)
    return kept
