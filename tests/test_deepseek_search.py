"""DeepSeek search URL normalization, allowlist, and a mocked fetch."""
from paper_radar.interests.base import Topic
from paper_radar.models import Paper
from paper_radar.rank.score import prefilter
from paper_radar.sources.deepseek_search import (
    DeepSeekSearchSource,
    apply_arxiv_meta,
    apply_biblio,
    arxiv_id_from_url,
    biblio_from_crossref,
    biblio_from_openalex,
    doi_from_url,
    drop_reason,
    fit_query,
    host_allowed,
    ids_from_crossref,
    ids_from_openalex,
    parse_arxiv_atom,
    queries_for_topic,
)


ALLOW = ["arxiv.org", "doi.org", "ieeexplore.ieee.org", "sciencedirect.com", "openreview.net"]

ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2401.12345v2</id>
    <title>Real driving title</title>
    <summary>This abstract describes an autonomous vehicle on a highway.</summary>
    <published>2026-10-28T00:00:00Z</published>
    <author><name>Ada</name></author>
  </entry>
</feed>
"""


class Resp:
    def __init__(self, status, payload=None, text=""):
        self.status_code = status
        self._payload = payload or {}
        self.text = text
        self.headers = {}

    def json(self):
        return self._payload


def test_arxiv_url_shapes_and_doi():
    cases = {
        "https://arxiv.org/abs/2401.12345": "2401.12345",
        "https://arxiv.org/abs/2401.12345v2": "2401.12345",
        "https://arxiv.org/pdf/2401.12345": "2401.12345",
        "https://arxiv.org/pdf/2401.12345v3.pdf": "2401.12345",
        "https://arxiv.org/html/2401.12345v2#S3.1": "2401.12345",
        "https://ar5iv.labs.arxiv.org/html/2401.12345": "2401.12345",
        "http://arxiv.org/abs/cs/0211011": "cs/0211011",
        "https://arxiv.org/abs/2401.12345?utm=1": "2401.12345",
    }
    for url, aid in cases.items():
        assert arxiv_id_from_url(url) == aid, url
    assert arxiv_id_from_url("https://github.com/lab/paper") is None
    assert doi_from_url("https://doi.org/10.1109/TIV.2026.1") == "10.1109/tiv.2026.1"
    assert doi_from_url("https://dx.doi.org/10.48550/arXiv.2401.12345)") == "10.48550/arxiv.2401.12345"
    assert doi_from_url("https://ieeexplore.ieee.org/document/123") is None


def test_allowlist_drops_blogs_and_low_scores():
    assert host_allowed("https://blog.csdn.net/a/b", ALLOW) is False
    assert host_allowed("https://github.com/lab/x", ALLOW) is False
    assert host_allowed("https://www.arxiv.org/abs/1", ALLOW) is True
    assert host_allowed("https://export.arxiv.org/api", ALLOW) is True
    assert host_allowed("https://person.github.io/notes", ALLOW) is False
    good = {"title": "A paper", "url": "https://arxiv.org/abs/2401.1", "rerank_score": 1.0}
    assert drop_reason(good, min_rerank=1.0, allow=ALLOW) is None
    assert drop_reason({**good, "rerank_score": 0.9}, min_rerank=1.0, allow=ALLOW) == "low_score"
    assert drop_reason(
        {"title": "notes", "url": "https://blog.csdn.net/x", "rerank_score": 2},
        min_rerank=1.0, allow=ALLOW,
    ) == "domain"
    assert drop_reason({"title": "x", "url": "https://arxiv.org/abs/1", "rerank_score": None},
                       min_rerank=1.0, allow=ALLOW) == "no_score"


def test_queries_stay_within_128_and_carry_after():
    topic = Topic(
        id="ctrl", name="MPC", direction="运动控制", description="vehicle control",
        query="model predictive control for a very long autonomous vehicle trajectory tracking " * 4,
    )
    qs = queries_for_topic(topic, "2026-10-27", 3)
    assert len(qs) == 3
    assert all(len(q) <= 128 for q in qs)
    assert "site:arxiv.org" in qs[0] and "after:2026-10-27" in qs[0]
    assert "site:ieeexplore.ieee.org OR site:sciencedirect.com" in qs[1]
    assert "site:openreview.net" in qs[2] and "lang:en" in qs[2]
    bare = Topic(id="dmp", name="DMP", direction="运动规划", description="primitives",
                 query="dynamical movement primitives")
    dmp = queries_for_topic(bare, "2026-10-27", 1)[0]
    assert "autonomous driving" in dmp and "after:2026-10-27" in dmp
    assert fit_query("hello", "after:2026-01-01").endswith("after:2026-01-01")
    topic.extra_queries = ["vehicle path tracking MPC tire model"]
    extra = queries_for_topic(topic, "2026-10-27", 3)
    assert len(extra) == 4
    assert extra[:3] == qs
    assert "vehicle path tracking" in extra[3]
    assert "site:arxiv.org OR site:ieeexplore.ieee.org" in extra[3]
    assert "after:2026-10-27" in extra[3]
    assert len(extra[3]) <= 128


def test_title_lookup_and_atom_enrich():
    work = {
        "title": "MPC tracking for an autonomous vehicle on the highway",
        "doi": "https://doi.org/10.1109/TIV.2026.1",
    }
    assert ids_from_openalex(work, work["title"]) == (None, "10.1109/tiv.2026.1")
    assert ids_from_openalex(work, "Completely different title about cooking") is None
    msg = {"items": [{"title": ["MPC tracking for an autonomous vehicle on the highway"],
                      "DOI": "10.1109/TIV.2026.1"}]}
    assert ids_from_crossref(msg, work["title"])[1] == "10.1109/tiv.2026.1"
    meta = parse_arxiv_atom(ATOM)
    paper = Paper(title="placeholder", abstract="short", arxiv_id="2401.12345", sources=["deepseek_search"])
    assert apply_arxiv_meta([paper], meta) == 1
    assert paper.abstract.startswith("This abstract")
    assert paper.published == "2026-10-28"
    assert paper.authors == ["Ada"]


def test_fetch_filters_enriches_and_survives_errors(monkeypatch):
    calls = []

    def fake(method, url, **kw):
        calls.append(url)
        if "tools/search" in url:
            query = kw["json"]["query"]
            assert kw["json"]["need_content"] is False
            assert kw["json"]["count"] == 50
            assert kw["headers"]["Authorization"] == "Bearer test-key"
            if "RAISE" in query:
                raise RuntimeError("search down")
            if "site:arxiv.org" in query and "openreview" not in query:
                return Resp(200, {"results": [
                    {"title": "Html hit", "url": "https://arxiv.org/html/2401.12345v2#S3",
                     "summary": "snippet long enough to stand in before enrich. " * 3,
                     "rerank_score": 2.2, "authority": 3, "date": "2026-11-03T00:00:00+00:00"},
                    {"title": "CSDN notes", "url": "https://blog.csdn.net/user/1",
                     "summary": "not a paper " * 10, "rerank_score": 2.5},
                    {"title": "Repo", "url": "https://github.com/lab/code",
                     "summary": "code " * 20, "rerank_score": 2.4},
                    {"title": "Weak hit", "url": "https://arxiv.org/abs/2401.00001",
                     "summary": "nope " * 20, "rerank_score": 0.2},
                ]})
            if "ieeexplore" in query:
                return Resp(200, {"results": [
                    {"title": "MPC tracking for an autonomous vehicle on the highway",
                     "url": "https://ieeexplore.ieee.org/document/999",
                     "summary": "Journal vehicle MPC. " * 8,
                     "rerank_score": 1.1, "date": "2020-01-01T00:00:00Z"},
                ]})
            return Resp(200, {"results": []})
        if "export.arxiv.org" in url:
            assert "2401.12345" in kw["params"]["id_list"]
            return Resp(200, text=ATOM)
        if "openalex.org/works/doi:" in url or "api.crossref.org/works/" in url:
            return Resp(404, {})
        if "api.openalex.org" in url:
            return Resp(200, {"results": [{
                "title": "MPC tracking for an autonomous vehicle on the highway",
                "doi": "https://doi.org/10.1109/TIV.2026.1",
            }]})
        raise AssertionError(url)

    monkeypatch.setattr("paper_radar.sources.deepseek_search.request", fake)
    bad = Topic(id="bad", name="bad", direction="运动控制", description="x", query="RAISE controller")
    good = Topic(id="ctrl", name="MPC", direction="运动控制", description="vehicle",
                 query="mpc autonomous vehicle")
    src = DeepSeekSearchSource(
        {"api_key": "test-key", "queries_per_topic": 3, "count": 50, "min_rerank_score": 1.0,
         "max_requests": 10, "doi_lookups": 5, "lookback_days": 7,
         "allow_domains": ALLOW},
        {},
    )
    papers = src.fetch({"bad": bad, "ctrl": good})
    by_url = {p.url: p for p in papers}
    assert "https://blog.csdn.net/user/1" not in by_url
    assert "https://github.com/lab/code" not in by_url
    arxiv = next(p for p in papers if p.arxiv_id == "2401.12345")
    assert arxiv.abstract.startswith("This abstract")
    assert arxiv.published == "2026-10-28"
    assert arxiv.signals["search_date"].startswith("2026-11-03")
    assert arxiv.pdf_url == "https://arxiv.org/pdf/2401.12345"
    ieee = next(p for p in papers if "ieeexplore" in p.url)
    assert ieee.published == ""
    assert ieee.doi == "10.1109/tiv.2026.1"
    assert ieee.signals["search_date"].startswith("2020-01-01")
    assert src.sent >= 1
    assert any("export.arxiv.org" in u for u in calls)

    empty = DeepSeekSearchSource({}, {"llm": {}})
    assert empty.fetch({"ctrl": good}) == []


def test_request_cap(monkeypatch):
    def fake(method, url, **kw):
        return Resp(200, {"results": []})

    monkeypatch.setattr("paper_radar.sources.deepseek_search.request", fake)
    topic = Topic(id="a", name="a", direction="运动规划", description="driving",
                  query="autonomous driving planning")
    src = DeepSeekSearchSource(
        {"api_key": "k", "queries_per_topic": 3, "max_requests": 1, "doi_lookups": 0},
        {},
    )
    assert src.fetch({"a": topic, "b": topic}) == []
    assert src.sent == 1


def test_prefilter_keeps_vehicles_and_drops_non_vehicle_robotics():
    topics = {
        "ctrl": Topic(
            id="ctrl", name="MPC", direction="运动控制",
            description="Model predictive control for autonomous vehicle path tracking on public roads",
            query="mpc autonomous vehicle",
        ),
        "plan": Topic(
            id="plan", name="规划", direction="运动规划",
            description="Motion planning for autonomous driving lane change and merging",
            query="autonomous driving planning",
        ),
    }
    papers = [
        Paper(title="MPC path tracking for an autonomous vehicle",
              abstract=("We track a reference path for an on-road autonomous vehicle with nonlinear MPC. " * 4)),
        Paper(title="Energy-Based Policy Orchestration for a Robot Manipulator",
              abstract=("Energy-based policies for a 7-DoF robotic arm gripper. " * 6)),
        Paper(title="Warehouse multi-robot task planning",
              abstract=("Multi-robot task allocation and motion planning inside a warehouse. " * 4)),
        Paper(title="Diffusion traffic scenario generation",
              abstract=("A diffusion model generates dynamic traffic scenarios for autonomous driving. " * 4)),
    ]
    kept, _ = prefilter(papers, topics, {}, top_k=10, min_emb=0.0, min_keep=4)
    titles = " ".join(p.title for p in kept)
    assert "autonomous vehicle" in titles
    assert "traffic scenario" in titles
    assert "Manipulator" not in titles
    assert "Warehouse" not in titles
    assert all(not p.signals.get("off_domain") for p in kept)


def test_biblio_fills_missing_authors_and_date_only():
    work = {
        "doi": "https://doi.org/10.1080/00423114.2024.1",
        "publication_date": "2024-05-02",
        "authorships": [{"author": {"display_name": "Li Wei"}}],
        "primary_location": {"source": {"display_name": "Vehicle System Dynamics"}},
        "abstract_inverted_index": {"Tire": [0], "model": [1], "identification": [2]},
        "best_oa_location": {"pdf_url": "https://oa.example.org/vsd.pdf"},
    }
    info = biblio_from_openalex(work)
    paper = Paper(title="Tire model", abstract="short", doi="10.1080/00423114.2024.1")
    assert apply_biblio(paper, info)
    assert paper.authors == ["Li Wei"]
    assert paper.published == "2024-05-02"
    assert paper.venue == "Vehicle System Dynamics"
    assert paper.abstract == "Tire model identification"
    assert paper.pdf_url.endswith(".pdf")
    paper.abstract = "x" * 120
    paper.authors = ["Keep Me"]
    apply_biblio(paper, info)
    assert paper.authors == ["Keep Me"]
    assert paper.abstract == "x" * 120
    cr = biblio_from_crossref({
        "DOI": "10.1016/j.conengprac.2024.1",
        "author": [{"given": "Ann", "family": "Lee"}],
        "issued": {"date-parts": [[2024, 7]]},
        "container-title": ["Control Engineering Practice"],
    })
    assert cr["authors"] == ["Ann Lee"]
    assert cr["published"] == "2024-07-01"
    assert cr["venue"] == "Control Engineering Practice"
    from datetime import date
    from paper_radar.sources.deepseek_search import _crossref_date, sane_date
    issue = {
        "published-print": {"date-parts": [[2027, 2, 1]]},
        "published-online": {"date-parts": [[2025, 11, 1]]},
    }
    assert _crossref_date(issue, date(2026, 11, 3)) == "2025-11-01"
    assert _crossref_date({"issued": {"date-parts": [[2027, 2, 1]]}}, date(2026, 11, 3)) == ""
    assert sane_date("2999-01-01", date(2026, 11, 3)) == ""
    assert biblio_from_openalex({"publication_date": "2999-01-01"})["published"] == ""


def test_fetch_fills_journal_biblio_by_doi(monkeypatch):
    def fake(method, url, **kw):
        if "tools/search" in url:
            return Resp(200, {"results": [{
                "title": "Tire model identification for path tracking",
                "url": "https://doi.org/10.1080/00423114.2024.9",
                "summary": "A long search summary that must stay in place. " * 4,
                "rerank_score": 1.6,
            }]})
        if "openalex.org/works/doi:" in url:
            assert url.endswith("doi:10.1080/00423114.2024.9")
            assert kw["params"]["mailto"] == "lab@example.edu"
            return Resp(200, {
                "publication_date": "2024-05-02",
                "authorships": [{"author": {"display_name": "Li Wei"}}],
                "primary_location": {"source": {"display_name": "Vehicle System Dynamics"}},
                "abstract_inverted_index": {"Different": [0], "abstract": [1]},
                "doi": "https://doi.org/10.1080/00423114.2024.9",
            })
        if "export.arxiv.org" in url:
            return Resp(200, text="<feed></feed>")
        raise AssertionError(url)

    monkeypatch.setattr("paper_radar.sources.deepseek_search.request", fake)
    topic = Topic(id="ctrl", name="MPC", direction="运动控制", description="vehicle",
                  query="mpc autonomous vehicle",
                  extra_queries=["tire model path tracking vehicle"])
    src = DeepSeekSearchSource(
        {"api_key": "k", "queries_per_topic": 3, "count": 50, "min_rerank_score": 1.0,
         "max_requests": 10, "doi_lookups": 0, "biblio_lookups": 4,
         "email": "lab@example.edu", "allow_domains": ALLOW + ["tandfonline.com"]},
        {},
    )
    # doi.org is already allowlisted; the hit URL is doi.org so it is kept.
    papers = src.fetch({"ctrl": topic})
    assert len(papers) == 1
    paper = papers[0]
    assert paper.authors == ["Li Wei"]
    assert paper.published == "2024-05-02"
    assert paper.venue == "Vehicle System Dynamics"
    assert paper.abstract.startswith("A long search summary")
    assert src.sent == 4  # 3 base queries + 1 extra
