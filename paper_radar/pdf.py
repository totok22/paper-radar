"""Legal open-access PDF links.

Waterfall for the papers that will actually be pushed:

1. arXiv id already on the paper (canonical ``/pdf/{id}``)
2. Semantic Scholar ``externalIds.ArXiv`` and ``openAccessPdf`` (filled earlier)
3. OpenAlex locations
4. Unpaywall, every ``oa_locations`` entry
5. HAL, CORE and OpenAIRE green OA (author copies in repositories)
6. arXiv title search
7. DeepSeek web search, only while a pick still has no PDF
8. A ranged GET that must start with ``%PDF``

Canonical arXiv PDFs and HAL deposit files are not byte-checked. Every other
URL is. Repository links (``/bitstream/``, ``?download=1``, ``?pdf=render``) are
eligible; a DOI landing page is not. Empty lookups are cached for about ten days.
Sci-Hub and LibGen are never queried.
"""
from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from urllib.parse import quote, urlparse

import feedparser
import requests

from .http import _with_deadline, request
from .institution import access_buttons, publisher_from_crossref, publisher_url_from
from .models import Paper, norm_arxiv, norm_doi, norm_title

log = logging.getLogger("paper_radar")

_GH = re.compile(r"https?://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", re.I)
_PDFISH = re.compile(
    r"(\.pdf($|[?#]))|arxiv\.org/pdf/|openreview\.net/pdf|"
    r"/pdf(?:/|\?|$)|type=printable|getpdf|stamppdf|application/pdf|"
    r"/bitstream/|[?&]download=1(?:&|$)|[?&]pdf=render(?:&|$)|[?&]format=pdf(?:&|$)|"
    r"hal\.science/[^?#\s]+/document(?:$|[?#])",
    re.I,
)
_DOI_PAGE = re.compile(r"^https?://(?:dx\.)?doi\.org/", re.I)
# HAL and some publishers sit behind a JS proof-of-work wall. An HTML 200 is
# then "unknown", not "no PDF".
_CHALLENGE = re.compile(
    rb"(not a bot|verifying you are human|cf-browser-verification|checking your browser|"
    rb"just a moment\.\.\.|captcha|enable javascript and cookies)",
    re.I,
)
_CANON_ARXIV = re.compile(r"^https://arxiv\.org/pdf/(?:\d{4}\.\d{4,5}|[a-z\-]+/\d{7})$", re.I)
_ARXIV_IN_URL = re.compile(
    r"arxiv\.org/(?:abs|pdf|html)/((?:[a-z\-]+/\d{7})|(?:\d{4}\.\d{4,5}))",
    re.I,
)
_STOP = {
    "for", "with", "and", "the", "via", "from", "into", "under", "toward", "towards",
    "using", "based", "its", "are", "how", "why", "that", "this", "than", "over",
    "between", "into", "onto", "per", "not",
}
_PUBLISHER_HOSTS = (
    "ieeexplore.ieee.org", "ieee.org", "sciencedirect.com", "springer.com",
    "doi.org", "tandfonline.com", "wiley.com", "acm.org", "nature.com",
    "science.org", "elsevier.com", "xplorestaging.ieee.org",
)
_BROWSER = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,*/*",
}


def normalize_pdf_url(url: str | None) -> str:
    if not url:
        return ""
    u = url.strip().split("#", 1)[0]
    # DeepSeek and repository mirrors hand back export./www. hosts and proxies.
    u = re.sub(r"^https?://(?:export\.|www\.)?arxiv\.org/", "https://arxiv.org/", u, flags=re.I)
    if "arxiv.org/abs/" in u:
        u = u.replace("arxiv.org/abs/", "arxiv.org/pdf/")
    return u


def looks_like_pdf(url: str | None) -> bool:
    """Cheap candidate test. Acceptance of a non-arXiv URL is `confirm_pdf`."""
    u = normalize_pdf_url(url)
    if not u.lower().startswith(("http://", "https://")):
        return False
    if _DOI_PAGE.search(u):
        return False
    return bool(_PDFISH.search(u))


def _clean_pdf(url: str | None) -> str | None:
    u = normalize_pdf_url(url)
    return u if looks_like_pdf(u) else None


class DomainBreaker:
    """Stop calling a host after N consecutive byte-check failures."""

    def __init__(self, limit: int = 3):
        self.limit = max(1, int(limit))
        self._fail: dict[str, int] = {}
        self._open: set[str] = set()

    def allowed(self, host: str) -> bool:
        return (host or "") not in self._open

    def record(self, host: str, ok: bool) -> None:
        if not host:
            return
        if ok:
            self._fail[host] = 0
            return
        n = self._fail.get(host, 0) + 1
        self._fail[host] = n
        if n >= self.limit:
            self._open.add(host)
            log.info("PDF byte-check paused for %s after %d failures", host, n)


class _Unverifiable(Exception):
    """The host answered, but not with a body we can judge (auth wall, rate limit, 5xx)."""


def confirm_pdf(url: str, breaker: DomainBreaker | None = None, timeout: float = 8) -> bool | None:
    """Ranged GET. True only when the body starts with %PDF.

    ``None`` means "could not tell" (403, 429, 5xx): a bot challenge must not be
    cached as "no PDF", because the link may work from a browser.
    """
    url = normalize_pdf_url(url)
    if not url.lower().startswith(("http://", "https://")):
        return False
    if _CANON_ARXIV.match(url):
        return True
    host = (urlparse(url).hostname or "").lower()
    if breaker is not None and not breaker.allowed(host):
        return False

    def _read(with_range: bool) -> bool:
        headers = dict(_BROWSER)
        if with_range:
            headers["Range"] = "bytes=0-1023"
        response = requests.get(url, headers=headers, timeout=timeout, stream=True, allow_redirects=True)
        try:
            if with_range and response.status_code == 416:
                return _read(False)
            if response.status_code in (401, 403, 407, 429) or response.status_code >= 500:
                raise _Unverifiable(f"HTTP {response.status_code}")
            chunk = next(response.iter_content(1024), b"") or b""
            if isinstance(chunk, str):
                chunk = chunk.encode("latin1", "ignore")
            ctype = (getattr(response, "headers", {}) or {}).get("Content-Type") or ""
            if "text/html" in ctype.lower() and _CHALLENGE.search(chunk):
                raise _Unverifiable("bot challenge")
            return chunk.startswith(b"%PDF") or b"%PDF" in chunk[:20]
        finally:
            response.close()

    result: bool | None = False
    try:
        result = bool(_with_deadline(lambda: _read(True), float(timeout) + 2))
    except _Unverifiable as e:
        log.info("PDF byte-check unverifiable for %s: %s", url[:90], e)
        result = None
    except Exception as e:  # noqa: BLE001
        log.info("PDF byte-check failed for %s: %s", url[:90], e)
        result = False
    if breaker is not None and result is not None:
        breaker.record(host, result)
    return result


def unpaywall_endpoint(doi: str) -> str | None:
    """Bare DOI in the path. A `https://doi.org/` prefix makes Unpaywall 404."""
    bare = norm_doi(doi)
    if not bare:
        return None
    return "https://api.unpaywall.org/v2/" + quote(bare, safe="/")


def candidates_from_unpaywall(data: dict) -> list[tuple[str, str]]:
    """Every OA location, url_for_pdf first. (url, version)."""
    if not isinstance(data, dict):
        return []
    locs = []
    best = data.get("best_oa_location")
    if isinstance(best, dict):
        locs.append(best)
    for loc in data.get("oa_locations") or []:
        if isinstance(loc, dict):
            locs.append(loc)
    out, seen = [], set()
    for loc in locs:
        ver = str(loc.get("version") or "")
        for key in ("url_for_pdf", "url"):
            raw = loc.get(key)
            if not raw:
                continue
            u = normalize_pdf_url(str(raw))
            if not u.lower().startswith(("http://", "https://")) or _DOI_PAGE.search(u):
                continue
            if key == "url" and "arxiv.org" not in u and not looks_like_pdf(u):
                continue
            if u in seen:
                continue
            seen.add(u)
            out.append((u, ver))
    return out


def pdf_from_unpaywall(data: dict) -> str | None:
    """First Unpaywall candidate. `url_for_pdf` need not end in .pdf."""
    rows = candidates_from_unpaywall(data)
    return rows[0][0] if rows else None


def candidates_from_openalex(work: dict) -> tuple[list[tuple[str, str]], str]:
    """PDF candidates plus the best publisher article URL found on the work."""
    if not isinstance(work, dict):
        return [], ""
    locs = []
    best = work.get("best_oa_location")
    if isinstance(best, dict):
        locs.append(best)
    for loc in work.get("locations") or []:
        if isinstance(loc, dict):
            locs.append(loc)
    publisher = ""
    out, seen = [], set()

    def add(raw: str, ver: str) -> None:
        nonlocal publisher
        if not raw:
            return
        pub = publisher_url_from(raw)
        if pub and not publisher:
            publisher = pub
        u = normalize_pdf_url(raw)
        if not u.lower().startswith(("http://", "https://")) or u in seen:
            return
        if _DOI_PAGE.search(u):
            return
        if "arxiv.org" not in u and not looks_like_pdf(u):
            return
        seen.add(u)
        out.append((u, ver))

    for loc in locs:
        ver = str(loc.get("version") or "")
        add(str(loc.get("pdf_url") or ""), ver)
        add(str(loc.get("landing_page_url") or ""), ver or "submittedVersion")
    oa = work.get("open_access") or {}
    if oa.get("oa_url"):
        add(str(oa["oa_url"]), "")
    return out, publisher


def candidates_from_openaire(record: dict) -> list[tuple[str, str]]:
    """Direct PDF URLs from the instances of an OpenAIRE research product."""
    if not isinstance(record, dict):
        return []
    access = str((record.get("bestAccessRight") or {}).get("label") or "").upper()
    if access and access != "OPEN":
        return []
    rows, seen = [], set()
    for inst in record.get("instances") or []:
        if not isinstance(inst, dict):
            continue
        for raw in inst.get("urls") or []:
            u = normalize_pdf_url(str(raw or ""))
            if not u.lower().startswith(("http://", "https://")) or u in seen:
                continue
            if _DOI_PAGE.search(u) or not looks_like_pdf(u):
                continue
            seen.add(u)
            rows.append((u, ""))
    return rows


def candidates_from_core(item: dict) -> list[tuple[str, str]]:
    """CORE downloadUrl and full-text links. Landing pages are filtered out."""
    if not isinstance(item, dict):
        return []
    rows, seen = [], set()

    def add(raw) -> None:
        u = normalize_pdf_url(str(raw or ""))
        if not u.lower().startswith(("http://", "https://")) or u in seen:
            return
        if _DOI_PAGE.search(u) or not looks_like_pdf(u):
            return
        seen.add(u)
        rows.append((u, ""))

    add(item.get("downloadUrl"))
    for link in item.get("links") or []:
        if isinstance(link, dict) and str(link.get("type") or "").lower() == "download":
            add(link.get("url"))
    for url in item.get("sourceFulltextUrls") or []:
        add(url)
    for row in item.get("urls") or []:
        if isinstance(row, dict):
            add(row.get("url"))
    return rows


def hal_pdf_url(doc: dict) -> str | None:
    """HAL deposit file, usually ``.../hal-xxxx/document`` or ``.../file/name.pdf``."""
    if not isinstance(doc, dict):
        return None
    for key in ("fileMain_s", "files_s"):
        raw = doc.get(key) or ""
        if isinstance(raw, list):
            raw = raw[0] if raw else ""
        u = normalize_pdf_url(str(raw))
        if u.lower().startswith(("http://", "https://")):
            return u
    return None


def is_trusted_pdf_url(url: str) -> bool:
    """HAL file URLs are real PDFs behind a proof-of-work wall a bot cannot clear."""
    host = (urlparse(url).hostname or "").lower()
    if not (host == "hal.science" or host.endswith(".hal.science")):
        return False
    return bool(re.search(r"(/document|/file/[^?#]+\.pdf)(?:$|[?#])", url, re.I))


def same_work(doi: str, title: str, paper: Paper) -> bool:
    """Guard against DOI/title search returning a neighbour."""
    got = norm_doi(doi)
    want = norm_doi(paper.doi)
    if got and want and got == want:
        return True
    return _sim(title, paper.title) >= 0.92 if title else False


def apply_s2_oa(p: Paper, data: dict) -> None:
    """Copy ArXiv externalId and openAccessPdf.url onto the paper."""
    if not isinstance(data, dict):
        return
    ext = data.get("externalIds") or {}
    aid = ""
    if isinstance(ext, dict):
        aid = str(ext.get("ArXiv") or ext.get("arXiv") or "")
    if aid and not p.arxiv_id:
        p.arxiv_id = norm_arxiv(aid)
    oa = data.get("openAccessPdf") or {}
    url = oa.get("url") if isinstance(oa, dict) else None
    if url:
        p.signals["s2_oa_pdf"] = normalize_pdf_url(str(url))
    if p.arxiv_id:
        p.pdf_url = f"https://arxiv.org/pdf/{p.arxiv_id}"
        p.signals["pdf_source"] = p.signals.get("pdf_source") or "s2"
        p.signals["pdf_version"] = "preprint"


def arxiv_title_query(title: str) -> str:
    words = [w for w in norm_title(title).split() if len(w) > 2 and w not in _STOP][:8]
    if not words:
        return ""
    return " AND ".join(f"ti:{w}" for w in words)


def _surnames(names) -> set[str]:
    out = set()
    for name in names or []:
        parts = re.findall(r"[A-Za-z]{2,}", name or "")
        if parts:
            out.add(parts[-1].lower())
    return out


def _year_of(published: str) -> int | None:
    m = re.match(r"(\d{4})", published or "")
    return int(m.group(1)) if m else None


def arxiv_id_year(aid: str) -> int | None:
    m = re.match(r"(\d{2})\d{2}\.\d{4,5}", aid or "")
    if not m:
        return None
    yy = int(m.group(1))
    return 2000 + yy if yy < 90 else 1900 + yy


def author_year_sanity(paper_authors, paper_published: str, hit_authors, arxiv_id: str) -> str:
    """`ok` when a surname or a close year matches, `fail` on a contradiction, else `unchecked`."""
    want, got = _surnames(paper_authors), _surnames(hit_authors)
    if want and not (want & got):
        return "fail"
    paper_year, ax_year = _year_of(paper_published), arxiv_id_year(arxiv_id)
    if paper_year is not None and ax_year is not None and not (paper_year - 3 <= ax_year <= paper_year + 1):
        return "fail"
    if want or (paper_year is not None and ax_year is not None):
        return "ok"
    return "unchecked"


def _sim(a: str, b: str) -> float:
    na, nb = norm_title(a), norm_title(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    return SequenceMatcher(None, na, nb).ratio()


def pick_arxiv_entry(entries: list[dict], title: str, authors, published: str) -> str | None:
    best_id, best_score = None, 0.0
    for entry in entries:
        aid = norm_arxiv(entry.get("arxiv_id"))
        if not aid:
            continue
        score = _sim(entry.get("title") or "", title)
        sanity = author_year_sanity(authors, published, entry.get("authors") or [], aid)
        if sanity == "fail" or score < 0.90:
            continue
        if sanity == "unchecked" and score < 0.95:
            continue
        if score > best_score:
            best_id, best_score = aid, score
    return best_id


def parse_arxiv_search(xml: str) -> list[dict]:
    feed = feedparser.parse(xml or "")
    out = []
    for entry in feed.entries:
        raw = entry.get("id") or ""
        m = _ARXIV_IN_URL.search(raw) or _ARXIV_IN_URL.search("https://arxiv.org/abs/" + raw)
        if not m:
            continue
        published = ""
        if entry.get("published_parsed"):
            y, mo, d = entry.published_parsed[:3]
            published = f"{y:04d}-{mo:02d}-{d:02d}"
        out.append({
            "title": " ".join((entry.get("title") or "").split()),
            "arxiv_id": m.group(1),
            "authors": [a.get("name", "") for a in entry.get("authors") or [] if a.get("name")],
            "published": published,
        })
    return out


def deepseek_pdf_queries(title: str) -> list[str]:
    core = " ".join((title or "").split())[:115].rstrip()
    if not core:
        return []
    return [f"{core} arxiv", f"{core} pdf"]


def _publisher_host(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return any(host == item or host.endswith("." + item) for item in _PUBLISHER_HOSTS)


def deepseek_hit_url(hit: dict, title: str) -> str | None:
    """arXiv abs/pdf/html, or a non-publisher .pdf, when the title is close enough."""
    if not isinstance(hit, dict):
        return None
    url = hit.get("url") or ""
    shown = (hit.get("title") or "").split(" - ")[0]
    if _sim(shown, title) < 0.85 and _sim(hit.get("title") or "", title) < 0.85:
        return None
    m = _ARXIV_IN_URL.search(url)
    if m:
        aid = norm_arxiv(m.group(1))
        return f"https://arxiv.org/pdf/{aid}" if aid else None
    if looks_like_pdf(url) and not _publisher_host(url):
        return normalize_pdf_url(url)
    return None


def version_name(raw: str, url: str) -> str:
    if "arxiv.org" in (url or ""):
        return "preprint"
    text = (raw or "").lower()
    if "accept" in text:
        return "accepted"
    if "publish" in text:
        return "published"
    if "submit" in text or "preprint" in text:
        return "preprint"
    return ""


def pdf_button_label(p: Paper) -> str:
    url = p.pdf_link or ""
    version = p.signals.get("pdf_version") or ""
    source = p.signals.get("pdf_source") or ""
    if "arxiv.org" in url or source in {"arxiv", "arxiv_title"} or (p.arxiv_id and "arxiv.org" in url):
        return "PDF (arXiv preprint)"
    if version == "accepted":
        return "PDF (accepted)"
    if version == "published":
        return "PDF (published)"
    if version == "preprint":
        return "PDF (preprint)"
    return "PDF"


def annotate_local(p: Paper) -> None:
    """Fill arXiv PDF and a GitHub code link from text already on the paper."""
    cleaned = _clean_pdf(p.pdf_url)
    if p.arxiv_id:
        p.pdf_url = cleaned or f"https://arxiv.org/pdf/{p.arxiv_id}"
        p.signals.setdefault("pdf_source", "arxiv")
        p.signals.setdefault("pdf_version", "preprint")
    else:
        p.pdf_url = cleaned
    if not p.code_url:
        m = _GH.search(f"{p.abstract or ''}\n{p.comments or ''}\n{p.url or ''}")
        if m:
            p.code_url = m.group(0).rstrip(").,;]}>\"'")


def _remember_publisher(p: Paper, url: str) -> None:
    got = publisher_url_from(url)
    if got and not p.signals.get("publisher_url"):
        p.signals["publisher_url"] = got


def _set_pdf(p: Paper, url: str, source: str, version: str) -> None:
    p.pdf_url = url
    p.signals["pdf_source"] = source
    p.signals["pdf_version"] = version_name(version, url) or version


class _Resolver:
    def __init__(self, papers, cfg: dict, store):
        pdf_cfg = cfg.get("pdf") or {}
        self.papers = papers
        self.cfg = cfg
        self.store = store
        self.email = pdf_cfg.get("email") or ""
        self.limit = int(pdf_cfg.get("lookup_limit", 12))
        self.lookups = 0
        self.timeout = float(pdf_cfg.get("byte_timeout", 8))
        self.breaker = DomainBreaker(int(pdf_cfg.get("byte_fail_limit", 3)))
        self.empty_days = float(pdf_cfg.get("empty_cache_days", 10))
        self.do_arxiv = bool(pdf_cfg.get("arxiv_title", True))
        self.do_deepseek = bool(pdf_cfg.get("deepseek_fallback", True))
        self.deepseek_left = int(pdf_cfg.get("deepseek_limit", 8))
        self.do_hal = bool(pdf_cfg.get("hal", True))
        self.hal_left = int(pdf_cfg.get("hal_limit", 8))
        self.do_core = bool(pdf_cfg.get("core", True))
        self.core_key = str(pdf_cfg.get("core_api_key") or "").strip()
        self.core_left = int(pdf_cfg.get("core_limit", 6))
        self.do_openaire = bool(pdf_cfg.get("openaire", True))
        self.openaire_left = int(pdf_cfg.get("openaire_limit", 4))
        self.uncertain = False
        self.institution = cfg.get("institution") or {}
        self.filled = 0

    def _budget(self) -> bool:
        if self.lookups >= self.limit:
            return False
        self.lookups += 1
        return True

    def _cache_key(self, p: Paper) -> str:
        ident = norm_doi(p.doi) or p.arxiv_id or ""
        return f"pdf-resolve:v1:{ident}" if ident else ""

    def _read_cache(self, p: Paper):
        key = self._cache_key(p)
        if not key or self.store is None:
            return None
        return self.store.cache_get(key, empty_max_age_days=self.empty_days)

    def _write_cache(self, p: Paper, payload: dict) -> None:
        key = self._cache_key(p)
        if key and self.store is not None:
            self.store.cache_set(key, payload)

    def _accept(self, p: Paper, url: str, source: str, version: str) -> bool:
        url = normalize_pdf_url(url)
        if not url or _DOI_PAGE.search(url):
            return False
        aid = None
        m = _ARXIV_IN_URL.search(url)
        if m:
            aid = norm_arxiv(m.group(1))
        if aid and not p.arxiv_id:
            p.arxiv_id = aid
        if p.arxiv_id and (aid or "arxiv.org" in url):
            _set_pdf(p, f"https://arxiv.org/pdf/{p.arxiv_id}", source, "preprint")
            return True
        if is_trusted_pdf_url(url):
            # A browser clears HAL's JS challenge; a bot only sees the HTML wall.
            _set_pdf(p, url, source, version)
            return True
        ok = confirm_pdf(url, self.breaker, self.timeout)
        if ok is None:
            # A 403/429 says nothing about the file. Keep the URL out of the
            # negative cache so the next run can try again.
            self.uncertain = True
            return False
        if not ok:
            return False
        _set_pdf(p, url, source, version)
        return True

    def _try_urls(self, p: Paper, rows: list[tuple[str, str]], source: str) -> bool:
        for url, version in rows:
            _remember_publisher(p, url)
            if self._accept(p, url, source, version):
                return True
        return False

    def _s2(self, p: Paper) -> bool:
        url = p.signals.get("s2_oa_pdf") or ""
        if not url:
            return False
        _remember_publisher(p, url)
        return self._accept(p, url, "s2", "preprint" if "arxiv.org" in url else "")

    def _openalex(self, p: Paper) -> bool:
        if not p.doi or not self._budget():
            return False
        params = {"mailto": self.email} if self.email else None
        try:
            response = request(
                "GET", "https://api.openalex.org/works/doi:" + quote(p.doi, safe="/"),
                params=params, retries=1, timeout=20, backoff=1,
            )
        except Exception as e:  # noqa: BLE001
            log.info("OpenAlex PDF lookup skipped: %s", e)
            return False
        if response is None or response.status_code != 200:
            return False
        try:
            work = response.json()
        except ValueError:
            return False
        rows, publisher = candidates_from_openalex(work)
        if publisher:
            p.signals.setdefault("publisher_url", publisher)
        for loc in work.get("locations") or []:
            if isinstance(loc, dict):
                _remember_publisher(p, loc.get("landing_page_url") or "")
        return self._try_urls(p, rows, "openalex")

    def _unpaywall(self, p: Paper) -> bool:
        if not p.doi or not self.email or not self._budget():
            return False
        endpoint = unpaywall_endpoint(p.doi)
        if not endpoint:
            return False
        response = request("GET", endpoint, params={"email": self.email}, retries=1, timeout=25)
        if response is None or response.status_code != 200:
            if response is not None and response.status_code in (401, 422):
                log.warning("Unpaywall rejected the email (%s).", response.status_code)
            return False
        try:
            data = response.json()
        except ValueError:
            return False
        return self._try_urls(p, candidates_from_unpaywall(data), "unpaywall")

    def _hal(self, p: Paper) -> bool:
        """French repositories (Inria, LAAS, ISIR) deposit author copies in HAL."""
        if not self.do_hal or not p.doi or self.hal_left <= 0 or not self._budget():
            return False
        self.hal_left -= 1
        try:
            response = request(
                "GET", "https://api.archives-ouvertes.fr/search/",
                params={
                    "q": f'doiId_s:"{norm_doi(p.doi)}"',
                    "fl": "title_s,fileMain_s,uri_s,openAccess_bool",
                    "wt": "json",
                    "rows": 3,
                },
                retries=1, timeout=20, backoff=1,
            )
        except Exception as e:  # noqa: BLE001
            log.info("HAL PDF lookup skipped: %s", e)
            return False
        if response is None or response.status_code != 200:
            log.info("HAL PDF lookup HTTP %s", "none" if response is None else response.status_code)
            return False
        try:
            docs = ((response.json().get("response") or {}).get("docs") or [])
        except ValueError:
            return False
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            title = (doc.get("title_s") or [""])[0] if isinstance(doc.get("title_s"), list) else doc.get("title_s")
            if not same_work(p.doi, str(title or ""), p):
                continue
            url = hal_pdf_url(doc)
            if url and re.search(r"(\.pdf($|\?)|/document($|\?))", url, re.I):
                if self._accept(p, url, "hal", ""):
                    return True
        return False

    def _core(self, p: Paper) -> bool:
        if not self.do_core or not self.core_key or not p.doi or self.core_left <= 0 or not self._budget():
            return False
        self.core_left -= 1
        try:
            response = request(
                "GET", "https://api.core.ac.uk/v3/search/outputs/",
                params={"q": f'doi:"{norm_doi(p.doi)}"', "limit": 3},
                headers={"Authorization": f"Bearer {self.core_key}"},
                retries=1, timeout=25, backoff=2,
            )
        except Exception as e:  # noqa: BLE001
            log.info("CORE PDF lookup skipped: %s", e)
            return False
        if response is None or response.status_code != 200:
            code = "none" if response is None else response.status_code
            if response is not None and response.status_code in (401, 403):
                log.warning("CORE rejected the API key (HTTP %s).", response.status_code)
            else:
                log.info("CORE PDF lookup HTTP %s", code)
            return False
        try:
            items = response.json().get("results") or []
        except ValueError:
            return False
        if not items:
            log.info("CORE returned no record for %s", p.doi)
            return False
        for item in items:
            if not isinstance(item, dict):
                continue
            if not same_work(str(item.get("doi") or ""), str(item.get("title") or ""), p):
                continue
            if self._try_urls(p, candidates_from_core(item), "core"):
                return True
        return False

    def _openaire(self, p: Paper) -> bool:
        if not self.do_openaire or not p.doi or self.openaire_left <= 0 or not self._budget():
            return False
        self.openaire_left -= 1
        try:
            response = request(
                "GET", "https://api.openaire.eu/graph/v1/researchProducts",
                params={"pid": norm_doi(p.doi)},
                retries=1, timeout=15, backoff=2,
            )
        except Exception as e:  # noqa: BLE001
            log.info("OpenAIRE PDF lookup skipped: %s", e)
            return False
        if response is None or response.status_code != 200:
            log.info("OpenAIRE PDF lookup HTTP %s", "none" if response is None else response.status_code)
            return False
        try:
            results = response.json().get("results") or []
        except ValueError:
            return False
        if not results or not isinstance(results[0], dict):
            log.info("OpenAIRE returned no record for %s", p.doi)
            return False
        return self._try_urls(p, candidates_from_openaire(results[0]), "openaire")

    def _arxiv_title(self, p: Paper) -> bool:
        if p.arxiv_id or not self.do_arxiv or not self._budget():
            return False
        query = arxiv_title_query(p.title)
        if not query:
            return False
        try:
            response = request(
                "GET", "https://export.arxiv.org/api/query",
                params={"search_query": query, "max_results": 5},
                timeout=25, retries=1, backoff=3,
            )
        except Exception as e:  # noqa: BLE001
            log.info("arXiv title search skipped: %s", e)
            return False
        if response is None or response.status_code != 200:
            return False
        aid = pick_arxiv_entry(parse_arxiv_search(response.text), p.title, p.authors, p.published)
        if not aid:
            return False
        p.arxiv_id = aid
        return self._accept(p, f"https://arxiv.org/pdf/{aid}", "arxiv_title", "preprint")

    def _deepseek(self, p: Paper) -> bool:
        if not self.do_deepseek or self.deepseek_left <= 0:
            return False
        ds = (self.cfg.get("sources") or {}).get("deepseek_search") or {}
        key = (ds.get("api_key") or (self.cfg.get("llm") or {}).get("api_key") or "").strip()
        if not key:
            return False
        from .sources.deepseek_search import search_endpoint
        base = ds.get("base_url") or (self.cfg.get("llm") or {}).get("base_url") or "https://api.deepseek.com"
        self.deepseek_left -= 1
        for query in deepseek_pdf_queries(p.title):
            if len(query) > 128 or not self._budget():
                continue
            try:
                response = request(
                    "POST", search_endpoint(base),
                    json={"query": query, "count": 10, "need_content": False},
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    retries=1, timeout=30, backoff=2,
                )
            except Exception as e:  # noqa: BLE001
                log.info("DeepSeek PDF search skipped: %s", e)
                return False
            if response is None or response.status_code != 200:
                continue
            try:
                rows = response.json().get("results") or []
            except ValueError:
                continue
            for hit in rows:
                url = deepseek_hit_url(hit, p.title)
                if url and self._accept(p, url, "deepseek", "preprint" if "arxiv.org" in url else ""):
                    return True
        return False

    def _crossref_publisher(self, p: Paper) -> None:
        if p.signals.get("publisher_url") or not p.doi or not self.institution.get("enabled"):
            return
        if not self._budget():
            return
        try:
            response = request(
                "GET", "https://api.crossref.org/works/" + quote(p.doi, safe="/"),
                headers={"User-Agent": "paper-radar/1.0 (mailto:%s)" % (self.email or "paper-radar@users.noreply.github.com")},
                timeout=20, retries=1, backoff=1,
            )
        except Exception as e:  # noqa: BLE001
            log.info("Crossref publisher lookup skipped: %s", e)
            return
        if response is None or response.status_code != 200:
            return
        try:
            message = response.json().get("message") or {}
        except ValueError:
            return
        got = publisher_from_crossref(message, p.doi)
        if got:
            p.signals["publisher_url"] = got

    def _access(self, p: Paper) -> None:
        if p.pdf_url or not p.doi:
            p.signals.pop("access", None)
            return
        self._crossref_publisher(p)
        target = p.signals.get("publisher_url") or f"https://doi.org/{p.doi}"
        buttons = access_buttons(target, self.institution)
        if buttons:
            p.signals["access"] = [{"label": label, "url": url} for label, url in buttons]
        else:
            p.signals.pop("access", None)

    def _s2_batch(self) -> None:
        s2cfg = (self.cfg.get("sources") or {}).get("semantic_scholar") or {}
        if not (s2cfg.get("api_key") or ""):
            return
        need = [p for p in self.papers if p.s2_ref and not p.arxiv_id and not p.signals.get("s2_oa_pdf")]
        if not need:
            return
        from .s2 import get_client
        try:
            rows = get_client(s2cfg.get("api_key") or "").paper_batch(
                [p.s2_ref for p in need], "externalIds,openAccessPdf",
            )
        except Exception as e:  # noqa: BLE001
            log.info("S2 PDF batch skipped: %s", e)
            return
        for paper, row in zip(need, rows):
            if isinstance(row, dict):
                apply_s2_oa(paper, row)

    def run(self) -> int:
        if self.do_core and not self.core_key:
            log.info("CORE skipped: CORE_API_KEY is empty")
        self._s2_batch()
        for paper in self.papers:
            if paper.arxiv_id:
                _set_pdf(paper, f"https://arxiv.org/pdf/{paper.arxiv_id}", paper.signals.get("pdf_source") or "arxiv", "preprint")
                self.filled += 1
                paper.signals.pop("access", None)
                continue
            pending = []
            if paper.pdf_url:
                pending.append((paper.pdf_url, paper.signals.get("pdf_version") or ""))
                paper.pdf_url = None
            cached = self._read_cache(paper)
            if isinstance(cached, dict) and cached.get("url"):
                _set_pdf(paper, cached["url"], cached.get("source") or "cache", cached.get("version") or "")
                if cached.get("publisher_url"):
                    paper.signals.setdefault("publisher_url", cached["publisher_url"])
                self.filled += 1
                paper.signals.pop("access", None)
                continue
            if isinstance(cached, dict) and not cached.get("url"):
                if cached.get("publisher_url"):
                    paper.signals.setdefault("publisher_url", cached["publisher_url"])
                self._access(paper)
                continue
            known_source = paper.signals.get("pdf_source") or "known"
            self.uncertain = False
            found = (
                self._try_urls(paper, pending, known_source)
                or self._s2(paper)
                or self._openalex(paper)
                or self._unpaywall(paper)
                or self._hal(paper)
                or self._core(paper)
                or self._openaire(paper)
                or self._arxiv_title(paper)
                or self._deepseek(paper)
            )
            if found:
                self.filled += 1
                paper.signals.pop("access", None)
            else:
                self._access(paper)
            if paper.pdf_url or not self.uncertain:
                self._write_cache(paper, {
                    "url": paper.pdf_url or "",
                    "source": paper.signals.get("pdf_source") or "",
                    "version": paper.signals.get("pdf_version") or "",
                    "publisher_url": paper.signals.get("publisher_url") or "",
                })
        if self.lookups or self.filled:
            log.info("PDF resolve lookups=%d filled=%d", self.lookups, self.filled)
        return self.filled


def resolve_pdfs(papers: list[Paper], cfg: dict, store=None) -> int:
    """Resolve PDFs for the final picks. Returns how many gained a direct PDF."""
    pdf_cfg = cfg.get("pdf") or {}
    if pdf_cfg.get("enabled", True) is False:
        return 0
    try:
        return _Resolver(papers, cfg, store).run()
    except Exception as e:  # noqa: BLE001
        log.warning("PDF resolve stopped: %s", e)
        return 0


def resolve_oa(papers: list[Paper], cfg: dict, store=None) -> int:
    """Backward-compatible name. Same waterfall as `resolve_pdfs`."""
    return resolve_pdfs(papers, cfg, store)
