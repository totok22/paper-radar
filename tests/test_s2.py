"""Shared Semantic Scholar limiter: one clock, batch endpoints, 429 backoff, budget."""
from paper_radar.models import Paper
from paper_radar.rank import signals
from paper_radar.s2 import AUTHOR_BATCH, PAPER_BATCH, get_client


class Resp:
    def __init__(self, status, payload=None, headers=None):
        self.status_code = status
        self._payload = {} if payload is None else payload
        self.headers = headers or {}
        self.text = ""

    def json(self):
        return self._payload


def _clock(monkeypatch):
    state = {"t": 0.0}
    sleeps = []

    def monotonic():
        return state["t"]

    def sleep(seconds):
        sleeps.append(seconds)
        state["t"] += seconds

    monkeypatch.setattr("paper_radar.s2.time.monotonic", monotonic)
    monkeypatch.setattr("paper_radar.s2.time.sleep", sleep)
    return sleeps


def test_one_limiter_spaces_every_endpoint(monkeypatch):
    sleeps = _clock(monkeypatch)
    seen = []

    def fake(method, url, **kw):
        seen.append((url, kw.get("headers", {}).get("x-api-key"), kw.get("_s2_internal"), kw.get("pace")))
        return Resp(200, {"ok": True})

    monkeypatch.setattr("paper_radar.s2.http_request", fake)
    client = get_client("secret-key", min_interval=1.1, max_requests=10)
    assert get_client() is client
    client.request("GET", "https://api.semanticscholar.org/graph/v1/paper/A")
    client.request("POST", "https://api.semanticscholar.org/recommendations/v1/papers", json={})
    assert client.sent == 2
    assert sleeps == [1.1]
    assert all(key == "secret-key" and internal is True and pace == 0 for _, key, internal, pace in seen)


def test_huge_retry_after_does_not_block(monkeypatch):
    sleeps = []
    monkeypatch.setattr("paper_radar.http.time.sleep", lambda s: sleeps.append(s))

    class Slow:
        status_code = 429
        headers = {"Retry-After": "26881"}
        text = "budget"

    monkeypatch.setattr("paper_radar.http.requests.request", lambda *a, **k: Slow())
    from paper_radar import http
    assert http.request("GET", "https://api.openalex.org/works", retries=3, pace=0) is None
    assert sleeps == []


def test_429_backoff_then_success(monkeypatch):
    sleeps = _clock(monkeypatch)
    queue = [Resp(429, {}, {"Retry-After": "3"}), Resp(200, {"ok": 1})]

    def fake(method, url, **kw):
        return queue.pop(0)

    monkeypatch.setattr("paper_radar.s2.http_request", fake)
    client = get_client("k", min_interval=1.1, max_requests=10)
    resp = client.request("GET", "https://api.semanticscholar.org/graph/v1/paper/A", retries=3)
    assert resp.status_code == 200
    assert client.sent == 2
    assert sleeps[0] == 3


def test_budget_stops_further_calls(monkeypatch):
    _clock(monkeypatch)
    calls = {"n": 0}

    def fake(method, url, **kw):
        calls["n"] += 1
        return Resp(200, {})

    monkeypatch.setattr("paper_radar.s2.http_request", fake)
    client = get_client("", min_interval=1.1, max_requests=1)
    assert client.request("GET", "https://api.semanticscholar.org/x") is not None
    assert client.request("GET", "https://api.semanticscholar.org/y") is None
    assert calls["n"] == 1


def test_paper_batch_chunks_and_author_batch(monkeypatch):
    _clock(monkeypatch)
    bodies = []

    def fake(method, url, **kw):
        ids = kw["json"]["ids"]
        bodies.append((url, ids))
        return Resp(200, [{"paperId": i} for i in ids])

    monkeypatch.setattr("paper_radar.s2.http_request", fake)
    client = get_client("k", min_interval=1.1, max_requests=20, batch_size=2)
    rows = client.paper_batch(["a", "b", "a", "c", "d", "e"], "title")
    assert [r["paperId"] for r in rows] == ["a", "b", "c", "d", "e"]
    assert [url for url, _ in bodies] == [PAPER_BATCH, PAPER_BATCH, PAPER_BATCH]
    assert [len(ids) for _, ids in bodies] == [2, 2, 1]
    authors = client.author_batch(["1", "2"], "name,hIndex")
    assert len(authors) == 2
    assert bodies[-1][0] == AUTHOR_BATCH


def test_batch_400_splits_without_dropping_ids(monkeypatch):
    _clock(monkeypatch)

    def fake(method, url, **kw):
        ids = kw["json"]["ids"]
        if len(ids) > 1:
            return Resp(400, {"error": "bad id"})
        return Resp(200, [{"title": ids[0]}])

    monkeypatch.setattr("paper_radar.s2.http_request", fake)
    rows = get_client("k", min_interval=0, max_requests=20).paper_batch(["good", "bad"], "title")
    assert [r["title"] for r in rows] == ["good", "bad"]


def test_http_entry_points_share_the_limiter(monkeypatch):
    sleeps = _clock(monkeypatch)

    def fake(method, url, **kw):
        assert kw.get("_s2_internal") is True
        return Resp(200, {})

    monkeypatch.setattr("paper_radar.s2.http_request", fake)
    from paper_radar import http
    assert http.request("GET", "https://api.semanticscholar.org/graph/v1/paper/X", retries=1) is not None
    assert http.request("POST", "https://api.semanticscholar.org/recommendations/v1/papers", retries=1) is not None
    assert get_client().sent == 2
    assert sleeps == [1.1]


def test_hindex_uses_author_batch_only_when_missing(monkeypatch):
    calls = []

    class Stub:
        def paper_batch(self, ids, fields):
            calls.append(("paper", fields))
            return [{
                "authors": [{"authorId": "99", "name": "Ada"}],
                "citationCount": 4,
                "venue": "",
                "openAccessPdf": {"url": "https://arxiv.org/pdf/2401.00001"},
            }]

        def author_batch(self, ids, fields="name,hIndex"):
            calls.append(("author", ids, fields))
            return [{"authorId": "99", "name": "Ada", "hIndex": 17}]

    monkeypatch.setattr(signals, "get_client", lambda api_key="": Stub())
    paper = Paper(title="Diffusion traffic scenario generation study", arxiv_id="2401.00001",
                  abstract="x" * 50)
    assert signals.add_author_hindex([paper], "k") == 1
    assert paper.signals["max_hindex"] == 17
    assert paper.pdf_url == "https://arxiv.org/pdf/2401.00001"
    assert [c[0] for c in calls] == ["paper", "author"]

    calls.clear()

    class Stub2(Stub):
        def paper_batch(self, ids, fields):
            calls.append(("paper", fields))
            return [{"authors": [{"authorId": "99", "name": "Ada", "hIndex": 8}]}]

        def author_batch(self, ids, fields="name,hIndex"):
            raise AssertionError("author/batch should not run when paper/batch has hIndex")

    monkeypatch.setattr(signals, "get_client", lambda api_key="": Stub2())
    paper2 = Paper(title="Another driving paper with a long title", arxiv_id="2401.00002", abstract="y" * 40)
    assert signals.add_author_hindex([paper2], "k") == 1
    assert paper2.signals["max_hindex"] == 8
    assert calls == [("paper", calls[0][1])]
