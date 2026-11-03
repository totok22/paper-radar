"""arXiv API recall.

Two groups of categories:
  full_categories   – take everything (e.g. cs.RO, eess.SY)
  keyword_categories – high-volume (cs.CV/cs.LG/cs.AI) restricted by driving keywords
"""
from __future__ import annotations

import html
import logging
import re
from datetime import datetime

import feedparser

from ..http import request
from ..models import Paper
from .base import Source

log = logging.getLogger("paper_radar")
API = "https://export.arxiv.org/api/query"


class ArxivSource(Source):
    name = "arxiv"

    def build_queries(self) -> list[str]:
        """Simple queries are far faster/more reliable on the arXiv API than one giant OR."""
        full = self.scfg.get("full_categories", ["cs.RO", "eess.SY"])
        kwc = self.scfg.get("keyword_categories", ["cs.CV", "cs.LG", "cs.AI"])
        kws = self.scfg.get("keywords", ["autonomous driving"])
        qs = [f"cat:{c}" for c in full]
        if kwc and kws:
            kq = " OR ".join(f'abs:"{k}"' if " " in k else f"abs:{k}" for k in kws)
            qs.append("(" + " OR ".join(f"cat:{c}" for c in kwc) + f") AND ({kq})")
        return qs

    def fetch(self, topics) -> list[Paper]:
        out, failed = [], False
        if self.scfg.get("mode", "api") == "api":
            for q in self.build_queries():
                got = self._fetch_query(q)
                if got is None:
                    failed = True
                    break
                log.info("arXiv API %-40s %d", q[:40], len(got))
                out += got
        if failed or self.scfg.get("mode") == "listing" or (not out and self.scfg.get("listing_fallback", True)):
            log.warning("arXiv API unavailable/rate-limited -> falling back to /list/<cat>/new pages")
            out += self._fetch_listings()
        return out

    # ---- fallback: daily "new" listing pages (latest announcement only) ----
    def _fetch_listings(self) -> list[Paper]:
        full = self.scfg.get("full_categories", ["cs.RO", "eess.SY"])
        kwc = self.scfg.get("keyword_categories", ["cs.CV", "cs.LG", "cs.AI"])
        kws = [k.lower() for k in self.scfg.get("keywords", [])]
        out = []
        for cat in full + kwc:
            r = request("GET", f"https://arxiv.org/list/{cat}/new", params={"skip": 0, "show": 2000},
                        timeout=60, retries=3, backoff=10)
            if r is None or r.status_code != 200:
                log.warning("arXiv listing %s failed", cat)
                continue
            got = parse_listing(r.text, cat)
            if cat in kwc:
                got = [p for p in got if any(k in (p.title + " " + p.abstract).lower() for k in kws)]
            log.info("arXiv listing %-8s %d", cat, len(got))
            out += got
        return out

    def _fetch_query(self, q: str) -> list[Paper]:
        page, start, out = int(self.scfg.get("page_size", 200)), 0, []
        max_total = int(self.scfg.get("max_results", 1500))
        since = self.since
        while start < max_total:
            r = request("GET", API, params={"search_query": q, "sortBy": "submittedDate", "sortOrder": "descending",
                                            "start": start, "max_results": page},
                        timeout=int(self.scfg.get("timeout", 30)), retries=int(self.scfg.get("retries", 2)), backoff=10)
            if r is None or r.status_code != 200:
                log.warning("arXiv request failed (%s) q=%s start=%d", None if r is None else r.status_code, q[:30], start)
                return out or None
            feed = feedparser.parse(r.text)
            if not feed.entries:
                break
            stop = False
            for e in feed.entries:
                pub = datetime(*e.published_parsed[:6]).date()
                if pub < since:
                    stop = True
                    continue
                aid = re.sub(r"v\d+$", "", e.id.rsplit("/abs/", 1)[-1])
                out.append(Paper(
                    title=" ".join(e.title.split()),
                    abstract=" ".join(e.summary.split()),
                    authors=[a.name for a in e.get("authors", [])],
                    published=pub.isoformat(),
                    url=e.link,
                    arxiv_id=aid,
                    doi=e.get("arxiv_doi"),
                    comments=" ".join([e.get("arxiv_comment", ""), e.get("arxiv_journal_ref", "")]).strip(),
                    categories=[t["term"] for t in e.get("tags", [])],
                    sources=["arxiv"],
                    pdf_url=f"https://arxiv.org/pdf/{aid}",
                ))
            if stop or len(feed.entries) < page:
                break
            start += page
        return out


_TAG = re.compile(r"<[^>]+>")


def _clean(x: str) -> str:
    return " ".join(html.unescape(_TAG.sub(" ", x)).split())


def parse_listing(text: str, cat: str) -> list[Paper]:
    m = re.search(r"Showing new listings for \w+, (\d+ \w+ \d{4})", text)
    pub = datetime.strptime(m.group(1), "%d %B %Y").date().isoformat() if m else ""
    out = []
    # only the "New submissions" + "Cross-lists" sections (skip replacements)
    body = text.split("Replacement submissions")[0]
    for dt, dd in re.findall(r"<dt>(.*?)</dt>\s*<dd>(.*?)</dd>", body, re.S):
        idm = re.search(r'id="(\d{4}\.\d{4,5})"', dt)
        tm = re.search(r"list-title[^>]*>(.*?)</div>", dd, re.S)
        if not (idm and tm):
            continue
        am = re.search(r"list-authors'>(.*?)</div>", dd, re.S)
        cm = re.search(r"list-comments[^>]*>(.*?)</div>", dd, re.S)
        jm = re.search(r"list-journal-ref[^>]*>(.*?)</div>", dd, re.S)
        sm = re.search(r"<p class='mathjax'>(.*?)</p>", dd, re.S)
        aid = idm.group(1)
        out.append(Paper(
            title=_clean(tm.group(1)).replace("Title:", "", 1).strip(),
            abstract=_clean(sm.group(1)) if sm else "",
            authors=[_clean(a) for a in re.findall(r">([^<]+)</a>", am.group(1))] if am else [],
            published=pub, url=f"https://arxiv.org/abs/{aid}", arxiv_id=aid,
            pdf_url=f"https://arxiv.org/pdf/{aid}",
            comments=" ".join(_clean(x.group(1)).replace("Comments:", "").replace("Journal-ref:", "")
                              for x in (cm, jm) if x).strip(),
            categories=[cat], sources=["arxiv"]))
    return out
