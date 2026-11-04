"""Semantic Scholar Recommendations API: seeds (+feedback) -> similar new papers."""
from __future__ import annotations

import logging
from datetime import date, timedelta

from ..pdf import looks_like_pdf
from ..models import Paper
from ..s2 import get_client
from .base import Source

log = logging.getLogger("paper_radar")
REC = "https://api.semanticscholar.org/recommendations/v1/papers"
FIELDS = "title,abstract,authors,publicationDate,externalIds,venue,url,openAccessPdf,citationCount"


def s2_to_paper(d: dict, src: str = "semantic_scholar") -> Paper:
    ext = d.get("externalIds") or {}
    oa = d.get("openAccessPdf") or {}
    pdf = oa.get("url") if isinstance(oa, dict) else None
    if pdf and not looks_like_pdf(pdf):
        pdf = None
    cites = d.get("citationCount")
    return Paper(
        title=d.get("title") or "",
        abstract=d.get("abstract") or "",
        authors=[a.get("name", "") for a in (d.get("authors") or [])[:20]],
        published=d.get("publicationDate") or "",
        url=d.get("url") or "",
        arxiv_id=ext.get("ArXiv"), doi=ext.get("DOI"), s2_id=d.get("paperId"),
        venue=d.get("venue") or "", pdf_url=pdf,
        citation_count=cites if isinstance(cites, int) else None,
        sources=[src],
    )


class S2RecommendSource(Source):
    name = "semantic_scholar"

    def fetch(self, topics) -> list[Paper]:
        client = get_client(self.scfg.get("api_key") or "")
        # Recommendations are usually months old; a daily 3-day window would
        # throw almost all of them away, so this source has its own floor.
        days = self.window_days(minimum=14)
        since = (date.today() - timedelta(days=days)).isoformat()
        retries = int(self.scfg.get("retries", 2))
        out, fails = [], 0
        seen_queries: set[tuple] = set()
        max_failures = int(self.scfg.get("max_failures", 2))
        for t in topics.values():
            if fails >= max_failures:
                log.warning("S2 recs: too many failures, skipping remaining topics")
                break
            pos = list(dict.fromkeys(t.positive_ids))[-100:]
            if not pos:
                continue
            neg = list(dict.fromkeys(t.negative_ids))[-100:]
            sig = (tuple(pos), tuple(neg))
            if sig in seen_queries:
                log.info("S2 recs skip %s (same seeds as an earlier topic)", t.id)
                continue
            seen_queries.add(sig)
            r = client.request(
                "POST", REC,
                params={"fields": FIELDS, "limit": int(self.scfg.get("limit", 100))},
                json={"positivePaperIds": pos, "negativePaperIds": neg},
                retries=retries,
            )
            if r is None or r.status_code != 200:
                log.warning("S2 recs failed for %s: %s", t.id, None if r is None else r.status_code)
                fails += 1
                continue
            fails = 0
            try:
                recs = r.json().get("recommendedPapers", [])
            except ValueError:
                fails += 1
                continue
            for d in recs:
                if (d.get("publicationDate") or "") >= since:
                    out.append(s2_to_paper(d))
        log.info("S2 requests so far: %d/%d", client.sent, client.max_requests)
        return out
