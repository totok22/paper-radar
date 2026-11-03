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


def _to_paper(w: dict, src: str) -> Paper:
    doi = (w.get("doi") or "").replace("https://doi.org/", "") or None
    aid = None
    for loc in w.get("locations") or []:
        u = (loc.get("landing_page_url") or "") + " " + (loc.get("pdf_url") or "")
        m = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", u)
        if m:
            aid = m.group(1)
            break
    if doi and doi.lower().startswith("10.48550/arxiv."):
        aid = aid or doi.split("arxiv.", 1)[-1].split("v")[0]
    venue = ((w.get("primary_location") or {}).get("source") or {}).get("display_name") or ""
    return Paper(
        title=w.get("title") or w.get("display_name") or "",
        abstract=_abstract(w.get("abstract_inverted_index")),
        authors=[a["author"]["display_name"] for a in (w.get("authorships") or [])[:20]],
        published=w.get("publication_date") or "",
        url=(w.get("primary_location") or {}).get("landing_page_url") or w.get("id", ""),
        arxiv_id=aid, doi=doi, venue=venue if "arxiv" not in venue.lower() else "",
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

    def _get(self, params: dict) -> list[dict]:
        r = request("GET", API, params=self._params(params))
        if r is None or r.status_code != 200:
            log.warning("OpenAlex failed: %s", None if r is None else f"{r.status_code} {r.text[:120]}")
            return []
        return r.json().get("results", [])


class OpenAlexSource(_OpenAlexBase):
    name = "openalex"

    def fetch(self, topics) -> list[Paper]:
        out = []
        for t in topics.values():
            q = t.query or t.name
            res = self._get({"search": q, "filter": f"from_publication_date:{self.since.isoformat()}",
                             "sort": "relevance_score:desc"})
            out += [_to_paper(w, "openalex") for w in res if w.get("title")]
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
            r = request("GET", API, params=self._params({"filter": flt, "cursor": cursor, "per_page": 200}))
            if r is None or r.status_code != 200:
                log.warning("OpenAlex journals failed")
                break
            js = r.json()
            out += [_to_paper(w, "journals") for w in js.get("results", []) if w.get("title")]
            cursor = js.get("meta", {}).get("next_cursor")
            if not cursor:
                break
        return out
