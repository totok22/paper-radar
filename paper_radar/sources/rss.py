"""Generic RSS/Atom slot (e.g. a journal TOC feed, a lab blog).

sources:
  rss:
    enabled: true
    feeds: {"Some journal": "https://example.org/feed.xml"}
"""
from __future__ import annotations

import logging
import re
from datetime import datetime

import feedparser

from ..http import request
from ..models import Paper
from .base import Source

log = logging.getLogger("paper_radar")


class RSSSource(Source):
    name = "rss"

    def fetch(self, topics) -> list[Paper]:
        out = []
        for venue, url in (self.scfg.get("feeds") or {}).items():
            r = request("GET", url)
            if r is None or r.status_code != 200:
                log.warning("RSS failed: %s", venue)
                continue
            for e in feedparser.parse(r.content).entries:
                tp = e.get("published_parsed") or e.get("updated_parsed")
                pub = datetime(*tp[:6]).date() if tp else None
                if pub and pub < self.since:
                    continue
                summ = re.sub(r"<[^>]+>", " ", e.get("summary", ""))
                m = re.search(r"10\.\d{4,9}/\S+", e.get("link", "") + " " + e.get("id", ""))
                out.append(Paper(title=" ".join(e.get("title", "").split()), abstract=" ".join(summ.split()),
                                 published=pub.isoformat() if pub else "", url=e.get("link", ""),
                                 doi=m.group(0) if m else None, venue=venue, sources=["rss"]))
        return out
