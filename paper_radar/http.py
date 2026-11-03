"""HTTP with retry, Retry-After, and per-host pacing.

arXiv asks for at most one request every 3 seconds. Semantic Scholar and
OpenAlex rate-limit aggressively without a key. A failed source must not
kill the run: callers treat a None response as "skip".
"""
from __future__ import annotations

import logging
import random
import threading
import time
from urllib.parse import urlparse

import requests

log = logging.getLogger("paper_radar")
UA = {"User-Agent": "paper-radar/0.2 (academic paper digest; mailto:paper-radar@users.noreply.github.com)"}

# Default polite gaps (seconds). Callers can override with pace=.
# Semantic Scholar is NOT listed: every S2 call goes through paper_radar.s2,
# which enforces one shared ~1.1s gap across all endpoints.
_MIN_INTERVAL = {
    "export.arxiv.org": 3.0,
    "arxiv.org": 3.0,
    "api.openalex.org": 0.2,
    "api.unpaywall.org": 0.25,
    "api2.openreview.net": 0.4,
    "huggingface.co": 0.15,
    "open.feishu.cn": 0.2,
}
_lock = threading.Lock()
_next_ok: dict[str, float] = {}


class _Deadline(Exception):
    pass


def _with_deadline(fn, seconds: float):
    """Run fn in a daemon thread with a wall-clock limit.

    requests' read timeout does not fire when a server trickles bytes.
    """
    box: dict = {}

    def target():
        try:
            box["r"] = fn()
        except Exception as e:  # noqa: BLE001
            box["e"] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(seconds)
    if t.is_alive():
        raise _Deadline(f"deadline {seconds}s exceeded")
    if "e" in box:
        raise box["e"]
    return box["r"]


def _pace(host: str, gap: float | None) -> None:
    if gap is None:
        gap = _MIN_INTERVAL.get(host, 0.0)
    if gap <= 0:
        return
    with _lock:
        now = time.monotonic()
        wait = _next_ok.get(host, 0.0) - now
        _next_ok[host] = max(now, _next_ok.get(host, 0.0)) + gap
    if wait > 0:
        time.sleep(wait)


def _retry_after(resp) -> float | None:
    raw = resp.headers.get("Retry-After") if resp is not None else None
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None


def request(method: str, url: str, *, retries: int = 3, backoff: float = 3.0, timeout: int = 40,
            pace: float | None = None, _s2_internal: bool = False, **kw):
    """Return a Response, or None after retries are exhausted.

    `pace` overrides the per-host gap (seconds). Pass 0 to disable pacing.
    Any URL on api.semanticscholar.org is handed to the shared S2 client
    unless `_s2_internal` is set (that client is the only caller of the flag).
    """
    headers = {**UA, **kw.pop("headers", {})}
    host = urlparse(url).netloc
    if "semanticscholar.org" in host and not _s2_internal:
        from .s2 import get_client
        return get_client().request(
            method, url, retries=retries, backoff=backoff, timeout=timeout, headers=headers, **kw,
        )
    for i in range(retries):
        _pace(host, pace)
        try:
            r = _with_deadline(
                lambda: requests.request(method, url, headers=headers, timeout=(10, timeout), **kw),
                timeout + 15,
            )
            if r.status_code == 429 or r.status_code >= 500:
                wait = _retry_after(r)
                # A multi-hour Retry-After means the daily budget is gone.
                # Sleeping it (or even the old 30s cap, once per call) blows the Actions window.
                if r.status_code == 429 and wait is not None and wait > 15:
                    log.warning("HTTP 429 Retry-After %.0fs; giving up on %s", wait, url[:90])
                    return None
                if wait is None:
                    wait = backoff * (i + 1) * random.uniform(0.85, 1.15)
                wait = min(wait, 15)
                log.debug("HTTP %s %s, retry %d in %.1fs", r.status_code, url[:80], i, wait)
                if i < retries - 1:
                    time.sleep(wait)
                continue
            return r
        except (requests.RequestException, _Deadline) as e:
            log.debug("HTTP error %s: %r", url[:80], e)
            if i < retries - 1:
                time.sleep(min(backoff * (i + 1), 30))
    return None


def download_capped(url: str, max_bytes: int, timeout: int = 60) -> bytes | None:
    """Download a URL into memory, aborting above `max_bytes`. None on failure."""
    if not url.lower().startswith(("http://", "https://")):
        return None
    try:
        with requests.get(url, headers=UA, stream=True, timeout=(10, timeout)) as r:
            if r.status_code != 200:
                log.info("PDF download HTTP %s %s", r.status_code, url[:80])
                return None
            cl = r.headers.get("Content-Length")
            if cl and cl.isdigit() and int(cl) > max_bytes:
                log.info("PDF too large (%s bytes): %s", cl, url[:80])
                return None
            buf = bytearray()
            for chunk in r.iter_content(64 * 1024):
                if not chunk:
                    continue
                buf.extend(chunk)
                if len(buf) > max_bytes:
                    log.info("PDF exceeded %d bytes while streaming: %s", max_bytes, url[:80])
                    return None
    except requests.RequestException as e:
        log.info("PDF download failed: %s", e)
        return None
    data = bytes(buf)
    if not data.startswith(b"%PDF"):
        log.info("download is not a PDF: %s", url[:80])
        return None
    return data
