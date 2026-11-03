"""DeepSeek web search as an extra recall source.

POST https://api.deepseek.com/v1/tools/search with the same Bearer key as the
LLM (`LLM_API_KEY`). Each sub-topic gets a few short queries (arXiv, IEEE /
ScienceDirect, and one OpenReview variant) limited by `after:`. Hits outside
a scholarly allowlist are dropped. arXiv URLs are normalized and then filled
from the arXiv API so the embedding prefilter sees a real abstract. Other
hits keep a DOI when the URL has one, otherwise a title lookup against
OpenAlex and Crossref. The `date` field is crawl time, not a publication
date, so it is stored and never used as a filter.

A failed query, a missing key, or a lookup error skips that step. The run
continues. Search is billed per successful request ($0.75 / 1000), so each
run has a hard request cap.
"""
from __future__ import annotations

import logging
import re
from datetime import date, timedelta
from difflib import SequenceMatcher
from urllib.parse import quote, urlparse

import feedparser

from ..http import request
from ..models import Paper, arxiv_from_doi, norm_doi, norm_title
from ..util import shanghai_today
from .base import Source

log = logging.getLogger("paper_radar")

SEARCH_PATH = "/v1/tools/search"
QUERY_LIMIT = 128
DEFAULT_ALLOW = (
    "arxiv.org",
    "doi.org",
    "ieeexplore.ieee.org",
    "sciencedirect.com",
    "link.springer.com",
    "dl.acm.org",
    "openreview.net",
    "proceedings.mlr.press",
    "proceedings.neurips.cc",
    "openaccess.thecvf.com",
    "thecvf.com",
    "aaai.org",
    "ijcai.org",
    "onlinelibrary.wiley.com",
    "iopscience.iop.org",
    "mdpi.com",
    "frontiersin.org",
    "nature.com",
    "science.org",
    "ascelibrary.org",
    "journals.sagepub.com",
    "tandfonline.com",
    "hal.science",
    "semanticscholar.org",
)

# abs / pdf / html, optional version, optional .pdf, optional #fragment.
_ARXIV_IN_URL = re.compile(
    r"arxiv\.org/(?:abs|pdf|html|ftp)/"
    r"((?:[a-z\-]+/\d{7})|(?:\d{4}\.\d{4,5}))"
    r"(?:v\d+)?"
    r"(?:\.pdf)?",
    re.I,
)
_DOI_IN_TEXT = re.compile(r"\b(10\.\d{4,9}/[-._;()/:A-Z0-9]+)", re.I)
_TRAILING = re.compile(r"[)\].,;:]+$")


def search_endpoint(base: str) -> str:
    b = (base or "https://api.deepseek.com").rstrip("/")
    if b.endswith("/v1"):
        return b + "/tools/search"
    return b + SEARCH_PATH


def fit_query(core: str, suffix: str, limit: int = QUERY_LIMIT) -> str:
    """Keep operators in `suffix` and trim the free-text core to the API's 128-char cap."""
    core = " ".join((core or "").split())
    suffix = " ".join((suffix or "").split())
    if not suffix:
        return core[:limit]
    room = limit - len(suffix) - 1
    if room < 8:
        return suffix[:limit]
    if len(core) > room:
        trimmed = core[:room]
        core = trimmed.rsplit(" ", 1)[0] if " " in trimmed else trimmed
    return f"{core} {suffix}".strip()[:limit]


def queries_for_topic(topic, after: str, n: int = 3) -> list[str]:
    """Base queries, then any topic-specific extras (control sub-topics).

    Extras are appended after the `n` base queries so a cap of 3 still means
    the arXiv / journal / OpenReview trio. Each extra is a plain keyword
    string; this function adds the scholarly sites and `after:` date.
    """
    core = (getattr(topic, "query", None) or getattr(topic, "name", None) or "autonomous driving").strip()
    low = core.lower()
    if "driving" not in low and "vehicle" not in low:
        core = f"{core} autonomous driving"
    built = [
        fit_query(core, f"site:arxiv.org after:{after}"),
        fit_query(core, f"site:ieeexplore.ieee.org OR site:sciencedirect.com after:{after}"),
        fit_query(core, f"site:openreview.net OR site:arxiv.org lang:en after:{after}"),
    ]
    out = built[: max(0, int(n))]
    for extra in getattr(topic, "extra_queries", None) or []:
        text = " ".join(str(extra).split())
        if not text:
            continue
        q = fit_query(text, f"site:arxiv.org OR site:ieeexplore.ieee.org after:{after}")
        if q not in out:
            out.append(q)
    return out


def host_of(url: str) -> str:
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""
    if host.startswith("www."):
        host = host[4:]
    return host


def host_allowed(url: str, allow: list[str] | tuple[str, ...]) -> bool:
    host = host_of(url)
    if not host:
        return False
    for dom in allow:
        d = (dom or "").lower().strip()
        if d.startswith("www."):
            d = d[4:]
        if d and (host == d or host.endswith("." + d)):
            return True
    return False


def arxiv_id_from_url(url: str) -> str | None:
    if not url:
        return None
    m = _ARXIV_IN_URL.search(url.split("#", 1)[0])
    if not m:
        return None
    return m.group(1)


def doi_from_url(url: str) -> str | None:
    if not url:
        return None
    m = _DOI_IN_TEXT.search(url)
    if not m:
        return None
    raw = _TRAILING.sub("", m.group(1))
    return norm_doi(raw)


def drop_reason(hit: dict, *, min_rerank: float, allow) -> str | None:
    """None means keep. Otherwise a short reason for the log/tests."""
    try:
        score = float(hit.get("rerank_score"))
    except (TypeError, ValueError):
        return "no_score"
    if score < min_rerank:
        return "low_score"
    url = hit.get("url") or ""
    if not host_allowed(url, allow):
        return "domain"
    if not (hit.get("title") or "").strip():
        return "no_title"
    return None


def hit_to_paper(hit: dict) -> Paper:
    url = hit.get("url") or ""
    aid = arxiv_id_from_url(url)
    doi = None if aid else doi_from_url(url)
    # Summary is the long model-facing text. Snippet is the short fallback.
    abstract = (hit.get("summary") or hit.get("snippet") or "").strip()
    paper = Paper(
        title=" ".join((hit.get("title") or "").split()),
        abstract=abstract,
        url=url,
        arxiv_id=aid,
        doi=doi,
        sources=["deepseek_search"],
    )
    # `date` is often crawl time. Keep it off `published` until a bibliographic
    # source (arXiv, OpenAlex, Crossref) says otherwise.
    if hit.get("date"):
        paper.signals["search_date"] = str(hit.get("date"))
    if hit.get("rerank_score") is not None:
        paper.signals["rerank_score"] = hit.get("rerank_score")
    if hit.get("authority") is not None:
        paper.signals["authority"] = hit.get("authority")
    if aid:
        paper.pdf_url = f"https://arxiv.org/pdf/{aid}"
    return paper


def titles_match(a: str, b: str, floor: float = 0.88) -> bool:
    na, nb = norm_title(a), norm_title(b)
    if not na or not nb:
        return False
    if na == nb or na[:80] == nb[:80]:
        return True
    return SequenceMatcher(None, na, nb).ratio() >= floor


def ids_from_openalex(work: dict, title: str) -> tuple[str | None, str | None] | None:
    if not isinstance(work, dict) or not titles_match(work.get("title") or work.get("display_name") or "", title):
        return None
    doi = norm_doi(work.get("doi"))
    aid = arxiv_from_doi(doi)
    return aid, doi


def ids_from_crossref(message: dict, title: str) -> tuple[str | None, str | None] | None:
    items = (message or {}).get("items") or []
    if not items:
        return None
    item = items[0]
    got = " ".join(item.get("title") or [])
    if not titles_match(got, title):
        return None
    doi = norm_doi(item.get("DOI"))
    return arxiv_from_doi(doi), doi


def parse_arxiv_atom(xml: str) -> dict[str, dict]:
    """Map a bare arXiv id to metadata parsed from an API Atom feed."""
    feed = feedparser.parse(xml or "")
    out = {}
    for e in feed.entries:
        raw = e.get("id") or ""
        aid = arxiv_id_from_url(raw) or arxiv_id_from_url("https://arxiv.org/abs/" + raw)
        if not aid:
            continue
        published = ""
        if e.get("published_parsed"):
            y, m, d = e.published_parsed[:3]
            published = f"{y:04d}-{m:02d}-{d:02d}"
        out[aid] = {
            "title": " ".join((e.get("title") or "").split()),
            "abstract": " ".join((e.get("summary") or "").split()),
            "authors": [a.get("name", "") for a in e.get("authors", []) if a.get("name")],
            "published": published,
            "comments": " ".join([e.get("arxiv_comment", ""), e.get("arxiv_journal_ref", "")]).strip(),
            "categories": [t.get("term", "") for t in e.get("tags", []) if t.get("term")],
        }
    return out


def _inverted_abstract(inv) -> str:
    if not isinstance(inv, dict) or not inv:
        return ""
    pos = {}
    for word, idx in inv.items():
        if not isinstance(idx, list):
            continue
        for i in idx:
            pos[i] = word
    if not pos:
        return ""
    return " ".join(pos[i] for i in sorted(pos))


def biblio_from_openalex(work: dict) -> dict:
    """Authors, date, venue, and PDF from one OpenAlex work. Empty fields stay empty."""
    if not isinstance(work, dict):
        return {}
    authors = []
    for row in (work.get("authorships") or [])[:20]:
        name = ((row or {}).get("author") or {}).get("display_name")
        if name:
            authors.append(name)
    venue = ((work.get("primary_location") or {}).get("source") or {}).get("display_name") or ""
    if "arxiv" in venue.lower():
        venue = ""
    pdf = (work.get("best_oa_location") or {}).get("pdf_url") or ""
    pdf = str(pdf).strip()
    if not pdf.lower().startswith(("http://", "https://")):
        pdf = ""
    return {
        "authors": authors,
        "published": (work.get("publication_date") or "").strip(),
        "venue": venue,
        "abstract": _inverted_abstract(work.get("abstract_inverted_index")),
        "pdf_url": pdf,
        "doi": norm_doi(work.get("doi")),
    }


def _crossref_date(message: dict) -> str:
    for key in ("published-print", "published-online", "issued"):
        parts = ((message.get(key) or {}).get("date-parts") or [None])[0] or []
        if not parts or not parts[0]:
            continue
        year = int(parts[0])
        month = int(parts[1]) if len(parts) > 1 and parts[1] else 1
        day = int(parts[2]) if len(parts) > 2 and parts[2] else 1
        return f"{year:04d}-{month:02d}-{day:02d}"
    return ""


def biblio_from_crossref(message: dict) -> dict:
    """One Crossref work message (GET /works/{doi}), not a search result list."""
    if not isinstance(message, dict) or not (message.get("DOI") or message.get("author")):
        return {}
    authors = []
    for row in (message.get("author") or [])[:20]:
        name = " ".join(x for x in (row.get("given"), row.get("family")) if x).strip()
        if name:
            authors.append(name)
    container = message.get("container-title") or []
    venue = container[0] if container else ""
    return {
        "authors": authors,
        "published": _crossref_date(message),
        "venue": venue,
        "abstract": "",
        "pdf_url": "",
        "doi": norm_doi(message.get("DOI")),
    }


def apply_biblio(paper: Paper, info: dict) -> bool:
    """Fill only blank bibliographic fields. A real abstract is left alone."""
    if not info:
        return False
    changed = False
    authors = [a for a in (info.get("authors") or []) if a]
    if authors and not paper.authors:
        paper.authors = authors
        changed = True
    published = (info.get("published") or "").strip()
    if published and not paper.published:
        paper.published = published
        changed = True
    venue = (info.get("venue") or "").strip()
    if venue and not paper.venue:
        paper.venue = venue
        changed = True
    pdf = (info.get("pdf_url") or "").strip()
    if pdf and not paper.pdf_url:
        paper.pdf_url = pdf
        changed = True
    doi = info.get("doi")
    if doi and not paper.doi:
        paper.doi = doi
        changed = True
    abstract = " ".join((info.get("abstract") or "").split())
    if abstract and len(paper.abstract or "") < 80 and len(abstract) > len(paper.abstract or ""):
        paper.abstract = abstract
        changed = True
    return changed


def apply_arxiv_meta(papers: list[Paper], meta: dict[str, dict]) -> int:
    n = 0
    for p in papers:
        e = meta.get(p.arxiv_id or "")
        if not e:
            continue
        if e.get("abstract"):
            p.abstract = e["abstract"]
            n += 1
        if e.get("title"):
            p.title = e["title"]
        if e.get("authors"):
            p.authors = e["authors"]
        if e.get("published"):
            p.published = e["published"]
        if e.get("comments"):
            p.comments = e["comments"]
        if e.get("categories"):
            p.categories = e["categories"]
        p.pdf_url = p.pdf_url or f"https://arxiv.org/pdf/{p.arxiv_id}"
    return n


class DeepSeekSearchSource(Source):
    name = "deepseek_search"

    def __init__(self, scfg: dict, cfg: dict):
        super().__init__(scfg, cfg)
        self.sent = 0

    def _key(self) -> str:
        return (self.scfg.get("api_key") or (self.cfg.get("llm") or {}).get("api_key") or "").strip()

    def _allow(self) -> list[str]:
        raw = self.scfg.get("allow_domains")
        if isinstance(raw, list) and raw:
            return [str(x) for x in raw if str(x).strip()]
        return list(DEFAULT_ALLOW)

    def _after(self) -> str:
        days = int(self.scfg.get("lookback_days", 7))
        try:
            today = date.fromisoformat(shanghai_today())
        except ValueError:
            today = date.today()
        return (today - timedelta(days=max(0, days))).isoformat()

    def _search(self, query: str) -> list[dict]:
        cap = int(self.scfg.get("max_requests", 40))
        if self.sent >= cap:
            log.info("DeepSeek search budget %d used up", cap)
            return []
        self.sent += 1
        count = max(1, min(50, int(self.scfg.get("count", 50))))
        url = search_endpoint(self.scfg.get("base_url") or "https://api.deepseek.com")
        try:
            r = request(
                "POST", url,
                json={"query": query, "count": count, "need_content": False},
                headers={"Authorization": f"Bearer {self._key()}", "Content-Type": "application/json"},
                retries=2, backoff=2, timeout=30,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("DeepSeek search request crashed: %s", e)
            return []
        if r is None or r.status_code != 200:
            log.warning("DeepSeek search failed (%s) q=%s", None if r is None else r.status_code, query[:80])
            return []
        try:
            payload = r.json()
        except ValueError:
            log.warning("DeepSeek search returned non-JSON")
            return []
        rows = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            log.warning("DeepSeek search payload has no results list")
            return []
        return [row for row in rows if isinstance(row, dict)]

    def _mailto(self) -> dict:
        email = (self.scfg.get("email") or "").strip()
        return {"mailto": email} if email else {}

    def _lookup_ids(self, title: str) -> tuple[str | None, str | None]:
        aid, doi, _info = self._resolve_title(title)
        return aid, doi

    def _resolve_title(self, title: str) -> tuple[str | None, str | None, dict]:
        """OpenAlex, then Crossref. Bibliographic fields come from the same hit."""
        try:
            r = request(
                "GET", "https://api.openalex.org/works",
                params={"search": title, "per-page": 1, **self._mailto()},
                timeout=20, retries=1, backoff=1,
            )
            if r is not None and r.status_code == 200:
                results = (r.json().get("results") or [])
                if results:
                    got = ids_from_openalex(results[0], title)
                    if got and (got[0] or got[1]):
                        return got[0], got[1], biblio_from_openalex(results[0])
        except Exception as e:  # noqa: BLE001
            log.info("OpenAlex title lookup skipped: %s", e)
        try:
            r = request(
                "GET", "https://api.crossref.org/works",
                params={"query.bibliographic": title, "rows": 1, **self._mailto()},
                headers=self._crossref_headers(),
                timeout=20, retries=1, backoff=1,
            )
            if r is not None and r.status_code == 200:
                message = r.json().get("message") or {}
                got = ids_from_crossref(message, title)
                if got and (got[0] or got[1]):
                    item = (message.get("items") or [{}])[0]
                    return got[0], got[1], biblio_from_crossref(item)
        except Exception as e:  # noqa: BLE001
            log.info("Crossref title lookup skipped: %s", e)
        return None, None, {}

    def _crossref_headers(self) -> dict:
        email = (self.scfg.get("email") or "").strip()
        agent = f"paper-radar/1.0 (mailto:{email})" if email else "paper-radar/1.0"
        return {"User-Agent": agent}

    def _biblio_by_doi(self, doi: str) -> dict:
        """OpenAlex work-by-DOI, then Crossref. One cheap fill for journal hits."""
        bare = norm_doi(doi)
        if not bare:
            return {}
        try:
            r = request(
                "GET", "https://api.openalex.org/works/doi:" + quote(bare, safe="/"),
                params=self._mailto() or None,
                timeout=20, retries=1, backoff=1,
            )
            if r is not None and r.status_code == 200:
                info = biblio_from_openalex(r.json())
                if info.get("authors") or info.get("published"):
                    return info
        except Exception as e:  # noqa: BLE001
            log.info("OpenAlex DOI lookup skipped: %s", e)
        try:
            r = request(
                "GET", "https://api.crossref.org/works/" + quote(bare, safe="/"),
                headers=self._crossref_headers(),
                timeout=20, retries=1, backoff=1,
            )
            if r is not None and r.status_code == 200:
                return biblio_from_crossref((r.json() or {}).get("message") or {})
        except Exception as e:  # noqa: BLE001
            log.info("Crossref DOI lookup skipped: %s", e)
        return {}

    def _enrich_arxiv(self, papers: list[Paper]) -> None:
        ids = list(dict.fromkeys(p.arxiv_id for p in papers if p.arxiv_id))
        if not ids:
            return
        meta: dict[str, dict] = {}
        for start in range(0, len(ids), 40):
            chunk = ids[start:start + 40]
            try:
                r = request(
                    "GET", "https://export.arxiv.org/api/query",
                    params={"id_list": ",".join(chunk)},
                    timeout=30, retries=2, backoff=5,
                )
            except Exception as e:  # noqa: BLE001
                log.warning("arXiv enrich crashed: %s", e)
                continue
            if r is None or r.status_code != 200:
                log.warning("arXiv enrich failed: %s", None if r is None else r.status_code)
                continue
            meta.update(parse_arxiv_atom(r.text))
        n = apply_arxiv_meta(papers, meta)
        log.info("DeepSeek search arXiv abstracts filled: %d/%d", n, len(ids))

    def fetch(self, topics) -> list[Paper]:
        if not self._key():
            log.info("DeepSeek search skipped: no LLM_API_KEY")
            return []
        allow = self._allow()
        after = self._after()
        min_score = float(self.scfg.get("min_rerank_score", 1.0))
        per_topic = int(self.scfg.get("queries_per_topic", 3))
        out: list[Paper] = []
        seen: set[str] = set()
        try:
            for topic in topics.values():
                for query in queries_for_topic(topic, after, per_topic):
                    if self.sent >= int(self.scfg.get("max_requests", 40)):
                        break
                    try:
                        hits = self._search(query)
                    except Exception as e:  # noqa: BLE001
                        log.warning("DeepSeek search query crashed: %s", e)
                        continue
                    kept = 0
                    for hit in hits:
                        why = drop_reason(hit, min_rerank=min_score, allow=allow)
                        if why:
                            continue
                        paper = hit_to_paper(hit)
                        if paper.key in seen:
                            continue
                        seen.add(paper.key)
                        out.append(paper)
                        kept += 1
                    log.info("DeepSeek search kept %d/%d for %s", kept, len(hits), getattr(topic, "id", ""))
        except Exception as e:  # noqa: BLE001
            log.warning("DeepSeek search stopped early: %s", e)
        lookups = 0
        limit = int(self.scfg.get("doi_lookups", 20))
        for paper in out:
            if paper.arxiv_id or paper.doi or lookups >= limit:
                continue
            lookups += 1
            try:
                aid, doi, info = self._resolve_title(paper.title)
            except Exception as e:  # noqa: BLE001
                log.info("id lookup crashed: %s", e)
                continue
            if aid:
                paper.arxiv_id = aid
                paper.pdf_url = paper.pdf_url or f"https://arxiv.org/pdf/{aid}"
            if doi and not paper.doi:
                paper.doi = doi
            apply_biblio(paper, info)
        try:
            self._enrich_arxiv(out)
        except Exception as e:  # noqa: BLE001
            log.warning("arXiv enrich skipped: %s", e)
        # Journal hits already carry a DOI, so the title lookup above skipped them.
        # Fill authors and publication date from that DOI, without touching arXiv papers.
        meta_used = 0
        meta_filled = 0
        meta_limit = int(self.scfg.get("biblio_lookups", 24))
        for paper in out:
            if paper.arxiv_id or not paper.doi or meta_used >= meta_limit:
                continue
            if paper.authors and paper.published:
                continue
            meta_used += 1
            try:
                info = self._biblio_by_doi(paper.doi)
            except Exception as e:  # noqa: BLE001
                log.info("DOI biblio crashed: %s", e)
                continue
            if apply_biblio(paper, info):
                meta_filled += 1
        if meta_used:
            log.info("DeepSeek search DOI biblio lookups=%d filled=%d", meta_used, meta_filled)
        log.info("DeepSeek search requests=%d papers=%d", self.sent, len(out))
        return out
