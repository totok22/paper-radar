"""Direct PDF and code links.

arXiv papers always get `https://arxiv.org/pdf/{id}`. Open-access PDFs found
by OpenAlex / Unpaywall (DOI) / Semantic Scholar `openAccessPdf` are kept
when the URL looks like a real PDF. Closed-access papers stay without a
fake link.
"""
from __future__ import annotations

import logging
import re

from .http import request
from .models import Paper

log = logging.getLogger("paper_radar")

_GH = re.compile(r"https?://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", re.I)
_PDFISH = re.compile(r"(\.pdf($|[?#]))|arxiv\.org/pdf/|openreview\.net/pdf", re.I)


def looks_like_pdf(url: str | None) -> bool:
    if not url:
        return False
    u = url.strip()
    if not u.lower().startswith(("http://", "https://")):
        return False
    return bool(_PDFISH.search(u))


def _clean_pdf(url: str | None) -> str | None:
    if not url:
        return None
    u = url.strip()
    if "arxiv.org/abs/" in u:
        u = u.replace("arxiv.org/abs/", "arxiv.org/pdf/")
    return u if looks_like_pdf(u) else None


def annotate_local(p: Paper) -> None:
    """Fill arXiv PDF and a GitHub code link from text already on the paper."""
    cleaned = _clean_pdf(p.pdf_url)
    if p.arxiv_id:
        p.pdf_url = cleaned or f"https://arxiv.org/pdf/{p.arxiv_id}"
    else:
        # Drop landing pages so the card button is a direct PDF or absent.
        p.pdf_url = cleaned
    if not p.code_url:
        m = _GH.search(f"{p.abstract or ''}\n{p.comments or ''}\n{p.url or ''}")
        if m:
            p.code_url = m.group(0).rstrip(").,;]}>\"'")


def _unpaywall_pdf(doi: str, email: str) -> str | None:
    r = request("GET", f"https://api.unpaywall.org/v2/{doi}", params={"email": email}, retries=2, timeout=25)
    if r is None or r.status_code != 200:
        return None
    try:
        data = r.json()
    except ValueError:
        return None
    best = data.get("best_oa_location") or {}
    url = _clean_pdf(best.get("url_for_pdf"))
    if url:
        return url
    for loc in data.get("oa_locations") or []:
        url = _clean_pdf(loc.get("url_for_pdf"))
        if url:
            return url
    return None


def resolve_oa(papers: list[Paper], cfg: dict, store=None) -> int:
    """Fill missing direct-PDF URLs for papers that have a DOI. Returns how many were filled."""
    pdf_cfg = cfg.get("pdf") or {}
    if not pdf_cfg.get("unpaywall", True):
        return 0
    email = pdf_cfg.get("email") or ""
    if not email:
        log.info("no Unpaywall email (OPENALEX_EMAIL / UNPAYWALL_EMAIL); skip OA lookup")
        return 0
    limit = int(pdf_cfg.get("lookup_limit", 20))
    n = 0
    looked = 0
    for p in papers:
        if p.pdf_link and looks_like_pdf(p.pdf_link):
            p.pdf_url = p.pdf_link
            continue
        if not p.doi or looked >= limit:
            continue
        key = f"unpaywall:{p.doi}"
        cached = store.cache_get(key) if store is not None else None
        if cached is None:
            cached = _unpaywall_pdf(p.doi, email) or ""
            if store is not None:
                store.cache_set(key, cached)
            looked += 1
        if cached:
            p.pdf_url = cached
            n += 1
    if looked:
        log.info("Unpaywall lookups=%d filled=%d", looked, n)
    return n
