"""One Semantic Scholar client for the whole run.

The key on this account allows 1 request/second, cumulative across every
endpoint (graph, recommendations, paper, author). All calls share one
limiter (~1.1s), back off on 429, and prefer paper/batch and author/batch.
A hard cap keeps a bug from issuing more than a few hundred requests.
"""
from __future__ import annotations

import logging
import threading
import time

from .http import _retry_after, request as http_request

log = logging.getLogger("paper_radar")

GRAPH = "https://api.semanticscholar.org/graph/v1"
PAPER_BATCH = GRAPH + "/paper/batch"
AUTHOR_BATCH = GRAPH + "/author/batch"
# S2 documents 500 ids per paper batch. Stay under that.
BATCH_SIZE = 500

_singleton_lock = threading.Lock()
_client: "S2Client | None" = None


class S2BudgetExceeded(RuntimeError):
    pass


class S2Client:
    def __init__(self, api_key: str = "", min_interval: float = 1.1, max_requests: int = 300,
                 batch_size: int = BATCH_SIZE):
        self.api_key = api_key or ""
        self.min_interval = float(min_interval)
        self.max_requests = int(max_requests)
        self.batch_size = max(1, min(int(batch_size), BATCH_SIZE))
        self.sent = 0
        self._lock = threading.Lock()
        self._next = 0.0

    def _acquire(self) -> float | None:
        """Reserve one request slot. Returns seconds to sleep, or None if the budget is spent."""
        with self._lock:
            if self.sent >= self.max_requests:
                return None
            now = time.monotonic()
            wait = max(0.0, self._next - now)
            self._next = max(now, self._next) + self.min_interval
            self.sent += 1
            return wait

    def _delay_next(self, extra: float) -> None:
        with self._lock:
            self._next = max(self._next, time.monotonic() + extra)

    def request(self, method: str, url: str, *, retries: int = 4, backoff: float = 2.0,
                timeout: int = 40, headers: dict | None = None, **kw):
        """Rate-limited request. 429/5xx are retried with backoff. None when the budget is gone."""
        hdrs = dict(headers or {})
        if self.api_key and "x-api-key" not in hdrs:
            hdrs["x-api-key"] = self.api_key
        last = None
        for attempt in range(max(1, retries)):
            wait = self._acquire()
            if wait is None:
                log.warning("S2 budget exhausted (%d); skip %s", self.max_requests, url[:90])
                return None
            if wait:
                time.sleep(wait)
            # pace=0: the generic host pacer must not add a second gap.
            last = http_request(
                method, url, retries=1, timeout=timeout, headers=hdrs, pace=0,
                _s2_internal=True, **kw,
            )
            if last is not None and last.status_code != 429 and last.status_code < 500:
                return last
            status = None if last is None else last.status_code
            extra = _retry_after(last) if last is not None else None
            if status == 429 and extra is not None and extra > 15:
                log.warning("S2 429 Retry-After %.0fs; not waiting out %s", extra, url[:80])
                return last
            if extra is None:
                extra = min(backoff * (2 ** attempt), 15.0)
            else:
                extra = min(extra, 15.0)
            log.info("S2 HTTP %s on %s; backoff %.1fs (attempt %d)", status, url[:80], extra, attempt + 1)
            self._delay_next(extra)
        return last

    def _chunked(self, url: str, ids: list[str], fields: str) -> list:
        """POST a batch endpoint. On HTTP 400, split the chunk so one bad id does not drop the rest."""
        ids = [i for i in ids if i]
        if not ids:
            return []
        out: list = [None] * len(ids)

        def run(start: int, chunk: list[str]) -> None:
            if not chunk:
                return
            r = self.request("POST", url, params={"fields": fields}, json={"ids": chunk})
            if r is not None and r.status_code == 400 and len(chunk) > 1:
                mid = len(chunk) // 2
                log.info("S2 batch 400 on %d ids; splitting", len(chunk))
                run(start, chunk[:mid])
                run(start + mid, chunk[mid:])
                return
            if r is None or r.status_code != 200:
                log.warning("S2 batch failed (%s) n=%d", None if r is None else r.status_code, len(chunk))
                return
            try:
                rows = r.json()
            except ValueError:
                log.warning("S2 batch returned non-JSON")
                return
            if not isinstance(rows, list):
                log.warning("S2 batch payload was not a list")
                return
            for i, row in enumerate(rows[:len(chunk)]):
                out[start + i] = row

        for start in range(0, len(ids), self.batch_size):
            run(start, ids[start:start + self.batch_size])
        return out

    def paper_batch(self, ids: list[str], fields: str) -> list:
        # Preserve order, drop duplicate round-trips.
        return self._chunked(PAPER_BATCH, list(dict.fromkeys(ids)), fields)

    def author_batch(self, ids: list[str], fields: str = "name,hIndex") -> list:
        return self._chunked(AUTHOR_BATCH, list(dict.fromkeys(ids)), fields)


def get_client(api_key: str = "", **kwargs) -> S2Client:
    """Process-wide client so recommendations, paper/batch and author/batch share one clock."""
    global _client
    with _singleton_lock:
        if _client is None:
            _client = S2Client(api_key, **kwargs)
            return _client
        if api_key:
            _client.api_key = api_key
        if _client.sent == 0:
            if "min_interval" in kwargs:
                _client.min_interval = float(kwargs["min_interval"])
            if "max_requests" in kwargs:
                _client.max_requests = int(kwargs["max_requests"])
            if "batch_size" in kwargs:
                _client.batch_size = max(1, min(int(kwargs["batch_size"]), BATCH_SIZE))
        return _client


def reset_client() -> None:
    """Tests only."""
    global _client
    with _singleton_lock:
        _client = None
