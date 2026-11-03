"""Direct PDF and code links.

arXiv papers always get `https://arxiv.org/pdf/{id}`. Open-access PDFs found
by OpenAlex / Unpaywall (DOI) / Semantic Scholar `openAccessPdf` are kept
when the URL looks like a real PDF. Closed-access papers stay without a
fake link.
"""
from __future__ import annotations

import logging
import re
from urllib.parse import quote

from .http import request
from .models import Paper, norm_doi

log = logging.getLogger("paper_radar")

_GH = re.compile(r"https?://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", re.I)
# Publisher OA links often have no ".pdf" suffix (PLOS type=printable, IEEE getPDF).
_PDFISH = re.compile(
    r"(\.pdf($|[?#]))|arxiv\.org/pdf/|openreview\.net/pdf|"
    r"/pdf(?:/|\?|$)|type=printable|getpdf|stamppdf|application/pdf",
    re.I,
)
_DOI_PAGE = re.compile(r"^https?://(?:dx\.)?doi\.org/", re.I)


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


def unpaywall_endpoint(doi: str) -> str | None:
    """Bare DOI in the path. A `https://doi.org/` prefix makes Unpaywall 404."""
    bare = norm_doi(doi)
    if not bare:
        return None
    return "https://api.unpaywall.org/v2/" + quote(bare, safe="/")


def pdf_from_unpaywall(data: dict) -> str | None:
    """`url_for_pdf` is the PDF even when the URL does not end in .pdf."""
    if not isinstance(data, dict):
        return None
    candidates = []
    best = data.get("best_oa_location") or {}
    if best.get("url_for_pdf"):
        candidates.append(best["url_for_pdf"])
    for loc in data.get("oa_locations") or []:
        if isinstance(loc, dict) and loc.get("url_for_pdf"):
            candidates.append(loc["url_for_pdf"])
    for url in candidates:
        u = str(url).strip()
        if u.lower().startswith(("http://", "https://")) and not _DOI_PAGE.search(u):
            return u
    for loc in [best, *(data.get("oa_locations") or [])]:
        if isinstance(loc, dict) and looks_like_pdf(loc.get("url")):
            return loc["url"].strip()
    return None


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
    url = unpaywall_endpoint(doi)
    if not url or not email:
        return None
    r = request("GET", url, params={"email": email}, retries=2, timeout=25)
    if r is None or r.status_code != 200:
        if r is not None and r.status_code in (401, 422):
            log.warning("Unpaywall rejected the email (%s). Set OPENALEX_EMAIL or UNPAYWALL_EMAIL.", r.status_code)
        elif r is not None and r.status_code != 404:
            log.info("Unpaywall HTTP %s for %s", r.status_code, doi)
        return None
    try:
        data = r.json()
    except ValueError:
        return None
    return pdf_from_unpaywall(data)


def _openalex_pdf_by_doi(doi: str, email: str) -> str | None:
    """Fallback when Unpaywall has no url_for_pdf. OpenAlex `pdf_url` is a direct file."""
    bare = norm_doi(doi)
    if not bare:
        return None
    params = {"mailto": email} if email else None
    r = request(
        "GET", "https://api.openalex.org/works/doi:" + quote(bare, safe="/"),
        params=params, retries=1, timeout=20, backoff=1,
    )
    if r is None or r.status_code != 200:
        return None
    try:
        work = r.json()
    except ValueError:
        return None
    pdf = (work.get("best_oa_location") or {}).get("pdf_url")
    if pdf and str(pdf).lower().startswith(("http://", "https://")) and not _DOI_PAGE.search(str(pdf)):
        return str(pdf).strip()
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
        # v2: previous runs cached "" when url_for_pdf had no ".pdf" suffix.
        key = f"oa-pdf:v2:{norm_doi(p.doi) or p.doi}"
        cached = store.cache_get(key) if store is not None else None
        if cached is None:
            cached = _unpaywall_pdf(p.doi, email) or ""
            if not cached:
                try:
                    cached = _openalex_pdf_by_doi(p.doi, email) or ""
                except Exception as e:  # noqa: BLE001
                    log.info("OpenAlex PDF lookup skipped: %s", e)
                    cached = ""
            if store is not None:
                store.cache_set(key, cached)
            looked += 1
        if cached:
            p.pdf_url = cached
            n += 1
    if looked:
        log.info("Unpaywall lookups=%d filled=%d", looked, n)
    return n
