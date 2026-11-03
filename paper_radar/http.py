from __future__ import annotations

import logging
import time
import threading

import requests

log = logging.getLogger("paper_radar")
UA = {"User-Agent": "paper-radar/0.1 (academic paper digest)"}


class _Deadline(Exception):
    pass


def _with_deadline(fn, seconds: float):
    """Run fn in a daemon thread (won't block interpreter exit) with a wall-clock limit."""
    box = {}

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


def request(method: str, url: str, *, retries: int = 3, backoff: float = 3.0, timeout: int = 40, **kw):
    """HTTP with retry on 429/5xx and a hard wall-clock deadline per attempt
    (servers that trickle bytes never trip requests' read timeout). Returns Response or None."""
    headers = {**UA, **kw.pop("headers", {})}
    for i in range(retries):
        try:
            r = _with_deadline(lambda: requests.request(method, url, headers=headers, timeout=(10, timeout), **kw),
                               timeout + 15)
            if r.status_code == 429 or r.status_code >= 500:
                log.debug("HTTP %s %s, retry %d", r.status_code, url[:80], i)
                if i < retries - 1:
                    time.sleep(backoff * (i + 1))
                continue
            return r
        except (requests.RequestException, _Deadline) as e:
            log.debug("HTTP error %s: %r", url[:80], e)
            if i < retries - 1:
                time.sleep(backoff * (i + 1))
    return None
