"""Semantic Scholar Recommendations API: seeds (+feedback) -> similar new papers."""
from __future__ import annotations

import logging
import time

from ..http import request
from ..models import Paper
from .base import Source

log = logging.getLogger("paper_radar")
REC = "https://api.semanticscholar.org/recommendations/v1/papers"
FIELDS = "title,abstract,authors,publicationDate,externalIds,venue,url"


def s2_to_paper(d: dict, src: str = "semantic_scholar") -> Paper:
    ext = d.get("externalIds") or {}
    return Paper(
        title=d.get("title") or "",
        abstract=d.get("abstract") or "",
        authors=[a.get("name", "") for a in (d.get("authors") or [])[:20]],
        published=d.get("publicationDate") or "",
        url=d.get("url") or "",
        arxiv_id=ext.get("ArXiv"), doi=ext.get("DOI"), s2_id=d.get("paperId"),
        venue=d.get("venue") or "", sources=[src],
    )


class S2RecommendSource(Source):
    name = "semantic_scholar"

    def fetch(self, topics) -> list[Paper]:
        headers = {"x-api-key": self.scfg["api_key"]} if self.scfg.get("api_key") else {}
        since = self.since.isoformat()
        out, fails = [], 0
        for t in topics.values():
            if fails >= int(self.scfg.get("max_failures", 2)):
                log.warning("S2 recs: too many failures (rate limited?), skipping remaining topics")
                break
            pos = list(dict.fromkeys(t.positive_ids))[-100:]
            if not pos:
                continue
            body = {"positivePaperIds": pos, "negativePaperIds": list(dict.fromkeys(t.negative_ids))[-100:]}
            r = request("POST", REC, params={"fields": FIELDS, "limit": int(self.scfg.get("limit", 100))},
                        json=body, headers=headers, retries=3, backoff=5)
            if r is None or r.status_code != 200:
                log.warning("S2 recs failed for %s: %s", t.id, None if r is None else r.status_code)
                fails += 1
                continue
            fails = 0
            for d in r.json().get("recommendedPapers", []):
                if (d.get("publicationDate") or "") >= since:
                    out.append(s2_to_paper(d))
            time.sleep(1.2 if headers else 3)
        return out
