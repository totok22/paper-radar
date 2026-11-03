"""OpenAlex recall: per-topic search + journal (ISSN) feeds.

Without an API key OpenAlex shares a small daily budget per IP; set
OPENALEX_API_KEY (free at https://openalex.org/settings/api) for reliability.
"""
from __future__ import annotations

import logging
import re

from ..http import request
from ..models import Paper
from .base import Source

log = logging.getLogger("paper_radar")
API = "https://api.openalex.org/works"


def _abstract(inv: dict | None) -> str:
    if not inv:
        return ""
    pos = {}
    for w, idx in inv.items():
        for i in idx:
            pos[i] = w
    return " ".join(pos[i] for i in sorted(pos))


def _oa_pdf(w: dict) -> str | None:
    """OpenAlex `pdf_url` is a direct PDF even when the link has no `.pdf` suffix."""
    ordered = []
    best = (w.get("best_oa_location") or {}).get("pdf_url")
    if best:
        ordered.append(best)
    for loc in w.get("locations") or []:
        if loc.get("pdf_url"):
            ordered.append(loc["pdf_url"])
    for url in ordered:
        if url and str(url).lower().startswith(("http://", "https://")):
            return str(url).strip()
    return None


def _pdf_candidates(w: dict) -> list[str]:
    urls = []
    best = w.get("best_oa_location") or {}
    if best.get("pdf_url"):
        urls.append(best["pdf_url"])
    oa = w.get("open_access") or {}
    if oa.get("oa_url"):
        urls.append(oa["oa_url"])
    for loc in w.get("locations") or []:
        if loc.get("pdf_url"):
            urls.append(loc["pdf_url"])
    return urls


def _affiliations(w: dict) -> list[str]:
    names = []
    for a in w.get("authorships") or []:
        for inst in a.get("institutions") or []:
            name = inst.get("display_name")
            if name and name not in names:
                names.append(name)
    return names[:8]


def _to_paper(w: dict, src: str) -> Paper:
    doi = (w.get("doi") or "").replace("https://doi.org/", "") or None
    aid = None
    blob = " ".join(_pdf_candidates(w))
    for loc in w.get("locations") or []:
        blob += " " + (loc.get("landing_page_url") or "")
    m = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", blob)
    if m:
        aid = m.group(1)
    if doi and doi.lower().startswith("10.48550/arxiv."):
        aid = aid or doi.split("arxiv.", 1)[-1].split("v")[0]
    venue = ((w.get("primary_location") or {}).get("source") or {}).get("display_name") or ""
    pdf = _oa_pdf(w)
    cites = w.get("cited_by_count")
    return Paper(
        title=w.get("title") or w.get("display_name") or "",
        abstract=_abstract(w.get("abstract_inverted_index")),
        authors=[a["author"]["display_name"] for a in (w.get("authorships") or [])[:20] if a.get("author")],
        published=w.get("publication_date") or "",
        url=(w.get("primary_location") or {}).get("landing_page_url") or w.get("id", ""),
        arxiv_id=aid, doi=doi, venue=venue if "arxiv" not in venue.lower() else "",
        pdf_url=pdf, affiliations=_affiliations(w),
        citation_count=cites if isinstance(cites, int) else None,
        sources=[src],
    )


class _OpenAlexBase(Source):
    def _params(self, extra: dict) -> dict:
        p = {"per_page": int(self.scfg.get("per_query", 50)), **extra}
        if self.scfg.get("api_key"):
            p["api_key"] = self.scfg["api_key"]
        if self.scfg.get("email"):
            p["mailto"] = self.scfg["email"]
        return p

    def _get(self, params: dict) -> list[dict] | None:
        r = request("GET", API, params=self._params(params), timeout=25, retries=2, backoff=2)
        if r is None or r.status_code != 200:
            log.warning("OpenAlex failed: %s", None if r is None else f"{r.status_code} {r.text[:120]}")
            return None
        try:
            return r.json().get("results", [])
        except ValueError:
            log.warning("OpenAlex returned non-JSON")
            return None


class OpenAlexSource(_OpenAlexBase):
    name = "openalex"

    def fetch(self, topics) -> list[Paper]:
        out, fails = [], 0
        limit = int(self.scfg.get("max_failures", 2))
        for t in topics.values():
            if fails >= limit:
                log.warning("OpenAlex: repeated failures, skipping remaining topics")
                break
            queries = [t.query or t.name, *list(getattr(t, "extra_queries", None) or [])]
            topic_failed = False
            for i, q in enumerate(queries):
                res = self._get({"search": q, "filter": f"from_publication_date:{self.since.isoformat()}",
                                 "sort": "relevance_score:desc"})
                if res is None:
                    # Only the primary query spends the failure budget. Extras are optional.
                    if i == 0:
                        topic_failed = True
                    continue
                out += [_to_paper(w, "openalex") for w in res if w.get("title")]
            if topic_failed:
                fails += 1
                continue
            fails = 0
        return out


class OpenAlexJournalSource(_OpenAlexBase):
    """New articles from target journals (by ISSN) — replaces fragile IEEE RSS."""

    name = "journals"

    def fetch(self, topics) -> list[Paper]:
        issns = self.scfg.get("issns", {})
        if not issns:
            return []
        days = int(self.scfg.get("lookback_days", max(self.days, 7)))  # journals index slower
        from datetime import date, timedelta
        since = (date.today() - timedelta(days=days)).isoformat()
        flt = f"primary_location.source.issn:{'|'.join(issns.values())},from_publication_date:{since}"
        out, cursor = [], "*"
        for _ in range(int(self.scfg.get("max_pages", 4))):
            r = request("GET", API, params=self._params({"filter": flt, "cursor": cursor, "per_page": 200}),
                        timeout=25, retries=2, backoff=2)
            if r is None or r.status_code != 200:
                log.warning("OpenAlex journals failed: %s", None if r is None else r.status_code)
                break
            js = r.json()
            out += [_to_paper(w, "journals") for w in js.get("results", []) if w.get("title")]
            cursor = js.get("meta", {}).get("next_cursor")
            if not cursor:
                break
        return out
