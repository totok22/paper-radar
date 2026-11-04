"""Legal PDF waterfall, BIT access links, and the control-slot content gate."""
import json
from datetime import date

from paper_radar.institution import carsi_url, publisher_from_crossref, publisher_url_from, webvpn_url
from paper_radar.models import Paper
from paper_radar.pdf import (
    DomainBreaker,
    arxiv_title_query,
    candidates_from_core,
    candidates_from_openaire,
    confirm_pdf,
    deepseek_hit_url,
    deepseek_pdf_queries,
    hal_pdf_url,
    normalize_pdf_url,
    pdf_button_label,
    pick_arxiv_entry,
    resolve_pdfs,
)
from paper_radar.push import feishu
from paper_radar.rank.score import lock_control_topics
from paper_radar.store import Store


def test_webvpn_known_hosts_and_carsi_needs_pii():
    ieee = webvpn_url("https://ieeexplore.ieee.org/document/10902072")
    assert ieee.startswith(
        "https://webvpn.bit.edu.cn/https/77726476706e69737468656265737421f9f244993f20645f6c0dc7a59d50267b1ab4a9"
    )
    assert ieee.endswith("/document/10902072")
    assert webvpn_url("https://doi.org/10.1109/tits.2025.1").startswith(
        "https://webvpn.bit.edu.cn/https/77726476706e69737468656265737421f4f848d228226f"
    )
    assert webvpn_url("https://www.sciencedirect.com/science/article/pii/S0001457526000011").startswith(
        "https://webvpn.bit.edu.cn/https/77726476706e69737468656265737421e7e056d234336155700b8ca891472636a6d29e640e"
    )
    assert webvpn_url("https://link.springer.com/article/10.1007/s00158-000-0000-0").startswith(
        "https://webvpn.bit.edu.cn/https/77726476706e69737468656265737421fcfe4f976923784277068ea98a1b203a54"
    )
    odd = webvpn_url("https://ieeexplore.ieee.org:8443/document/5")
    assert "/https-8443/" in odd
    staging = "http://xplorestaging.ieee.org/ielx8/6979/10945245/10902072.pdf?arnumber=10902072"
    assert publisher_url_from(staging) == "https://ieeexplore.ieee.org/document/10902072"
    pii = publisher_from_crossref({"alternative-id": ["S0001457526000011"]}, "10.1016/j.aap.2026.1")
    assert pii.endswith("/pii/S0001457526000011")
    idp = "https://idp.bit.edu.cn/idp/shibboleth"
    ieee_url = "https://ieeexplore.ieee.org/document/10902072"
    carsi = carsi_url(ieee_url, idp=idp)
    assert carsi.startswith("https://ieeexplore.ieee.org/servlet/wayf.jsp?entityId=")
    assert "idp.bit.edu.cn" in carsi and "10902072" in carsi
    assert carsi_url("https://doi.org/10.1016/j.aap.2026.1", idp=idp) is None
    assert "ShibAuth" in carsi_url(pii, idp=idp)


def test_byte_check_and_domain_breaker(monkeypatch):
    seen = {}

    class Resp:
        status_code = 206

        def iter_content(self, n):
            yield b"%PDF-1.5\nrest"

        def close(self):
            pass

    def fake_get(url, headers=None, timeout=None, stream=None, allow_redirects=None):
        seen["url"] = url
        seen["range"] = headers.get("Range")
        seen["ua"] = headers.get("User-Agent")
        return Resp()

    monkeypatch.setattr("paper_radar.pdf.requests.get", fake_get)
    assert confirm_pdf("http://repo.example.edu/bitstream/1/paper.pdf#page=4")
    assert seen["url"].startswith("https://repo.example.edu") is False
    assert seen["url"].endswith("paper.pdf")
    assert "#" not in seen["url"]
    assert seen["range"] == "bytes=0-1023"
    assert "Mozilla" in seen["ua"]

    breaker = DomainBreaker(2)
    breaker.record("closed.example", False)
    assert breaker.allowed("closed.example")
    breaker.record("closed.example", False)
    assert not breaker.allowed("closed.example")


def test_arxiv_title_and_deepseek_acceptance():
    query = arxiv_title_query("Model Predictive Control for Vehicle Path Tracking on Highways")
    assert " AND " in query and query.startswith("ti:")
    assert len(query.split(" AND ")) <= 8
    title = "Model Predictive Control for Vehicle Path Tracking on Highways"
    hit = {"title": title, "arxiv_id": "2401.12345", "authors": ["Ada Lee"]}
    assert pick_arxiv_entry([hit], title, ["Wei Lee"], "2024-06-01") == "2401.12345"
    assert pick_arxiv_entry([hit], title, ["Other Name"], "2024-06-01") is None
    old = {"title": title, "arxiv_id": "1901.00001", "authors": ["Ada Lee"]}
    assert pick_arxiv_entry([old], title, ["Ada Lee"], "2025-01-01") is None
    queries = deepseek_pdf_queries("T" * 200)
    assert queries and all(len(q) <= 128 for q in queries)
    assert queries[0].endswith(" arxiv") and queries[1].endswith(" pdf")
    assert deepseek_hit_url(
        {"title": title + " - arXiv", "url": "https://arxiv.org/html/2401.12345v2"}, title,
    ) == "https://arxiv.org/pdf/2401.12345"
    assert deepseek_hit_url(
        {"title": title + " - IEEE", "url": "https://ieeexplore.ieee.org/ielx7/1/1/1.pdf"}, title,
    ) is None
    assert deepseek_hit_url(
        {"title": "A different paper entirely", "url": "https://oa.example.org/x.pdf"}, title,
    ) is None


def test_core_openaire_hal_candidates():
    core = {
        "downloadUrl": "https://core.ac.uk/download/pdf/123.pdf",
        "links": [
            {"type": "download", "url": "https://core.ac.uk/download/456.pdf"},
            {"type": "display", "url": "https://core.ac.uk/outputs/1"},
        ],
        "sourceFulltextUrls": ["https://inria.hal.science/hal-04191464/document"],
        "urls": [{"url": "https://repo.example.edu/bitstream/1/paper.pdf", "type": "fulltext"}],
    }
    assert [u for u, _ in candidates_from_core(core)] == [
        "https://core.ac.uk/download/pdf/123.pdf",
        "https://core.ac.uk/download/456.pdf",
        "https://inria.hal.science/hal-04191464/document",
        "https://repo.example.edu/bitstream/1/paper.pdf",
    ]
    assert candidates_from_core({}) == []

    openaire = {
        "bestAccessRight": {"label": "OPEN"},
        "instances": [
            {"urls": ["https://doi.org/10.1/x",
                      "https://journals.plos.org/plosone/article/file?id=1&type=printable"]},
            {"urls": ["https://inria.hal.science/hal-1/document"]},
        ],
    }
    assert [u for u, _ in candidates_from_openaire(openaire)] == [
        "https://journals.plos.org/plosone/article/file?id=1&type=printable",
        "https://inria.hal.science/hal-1/document",
    ]
    assert candidates_from_openaire({"bestAccessRight": {"label": "RESTRICTED"},
                                     "instances": [{"urls": ["https://x.example/a.pdf"]}]}) == []

    assert hal_pdf_url({"fileMain_s": "https://inria.hal.science/hal-04191464/document"}) == (
        "https://inria.hal.science/hal-04191464/document"
    )
    assert hal_pdf_url({}) is None
    assert normalize_pdf_url("https://export.arxiv.org/pdf/2502.10363#1#5#1") == (
        "https://arxiv.org/pdf/2502.10363"
    )


def test_hal_fills_pdf_and_unverifiable_is_not_cached(tmp_path, monkeypatch):
    class Resp:
        status_code = 200
        headers = {}

        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    def fake(method, url, **kw):
        if "openalex.org" in url:
            return Resp({"locations": []})
        if "archives-ouvertes" in url:
            return Resp({"response": {"docs": [{
                "title_s": ["Robust Robot Planning for Human-Robot Collaboration"],
                "fileMain_s": "https://inria.hal.science/hal-04191464/document",
            }]}})
        raise AssertionError(url)

    monkeypatch.setattr("paper_radar.pdf.request", fake)
    monkeypatch.setattr("paper_radar.pdf.confirm_pdf", lambda *a, **k: True)
    paper = Paper(
        title="Robust Robot Planning for Human-Robot Collaboration",
        doi="10.1109/ICRA48891.2023.10161406",
        abstract="a" * 40,
    )
    cfg = {"pdf": {"lookup_limit": 5, "email": ""}, "institution": {"enabled": False}}
    assert resolve_pdfs([paper], cfg) == 1
    assert paper.pdf_url == "https://inria.hal.science/hal-04191464/document"
    assert paper.signals["pdf_source"] == "hal"

    # 403 / 429 must not poison the ten-day negative cache.
    def fake_oa(method, url, **kw):
        if "openalex.org" in url:
            return Resp({"best_oa_location": {"pdf_url": "https://publisher.example/getpdf?id=9"},
                         "locations": []})
        return Resp({"response": {"docs": []}, "results": []})

    monkeypatch.setattr("paper_radar.pdf.request", fake_oa)
    monkeypatch.setattr("paper_radar.pdf.confirm_pdf", lambda *a, **k: None)
    store = Store(tmp_path / "c.sqlite")
    blocked = Paper(title="Blocked repository copy of a vehicle MPC paper",
                    doi="10.1109/TIV.2024.9", abstract="a" * 40)
    assert resolve_pdfs([blocked], cfg, store) == 0
    assert store.cache_get("pdf-resolve:v1:10.1109/tiv.2024.9") is None


def test_byte_check_reports_unverifiable(monkeypatch):
    class Resp:
        status_code = 403

        def iter_content(self, n):
            yield b"<html>challenge</html>"

        def close(self):
            pass

    monkeypatch.setattr("paper_radar.pdf.requests.get", lambda *a, **k: Resp())
    assert confirm_pdf("https://repo.example.edu/paper.pdf") is None
    breaker = DomainBreaker(1)
    assert confirm_pdf("https://repo.example.edu/paper.pdf", breaker) is None
    assert breaker.allowed("repo.example.edu")


def test_empty_pdf_cache_expires(tmp_path):
    store = Store(tmp_path / "c.sqlite")
    store.cache_set("pdf-resolve:v1:10.1/abc", {"url": "", "publisher_url": "https://doi.org/10.1/abc"})
    store.db.execute("UPDATE cache SET created_at=? WHERE k=?", ("2020-01-01T00:00:00+00:00", "pdf-resolve:v1:10.1/abc"))
    store.db.commit()
    assert store.cache_get("pdf-resolve:v1:10.1/abc", empty_max_age_days=10) is None
    store.cache_set("pdf-resolve:v1:10.1/hit", {"url": "https://arxiv.org/pdf/2401.00001", "source": "arxiv"})
    store.db.execute("UPDATE cache SET created_at=? WHERE k=?", ("2020-01-01T00:00:00+00:00", "pdf-resolve:v1:10.1/hit"))
    store.db.commit()
    assert store.cache_get("pdf-resolve:v1:10.1/hit", empty_max_age_days=10)["url"].endswith("2401.00001")


def test_resolve_uses_s2_arxiv_then_stops(monkeypatch):
    def explode(*a, **k):
        raise AssertionError("no further lookup")

    monkeypatch.setattr("paper_radar.pdf.request", explode)
    paper = Paper(title="Path tracking MPC", doi="10.1109/tiv.2024.1", abstract="a" * 40)
    paper.signals["s2_oa_pdf"] = "https://arxiv.org/pdf/2401.12345"
    n = resolve_pdfs([paper], {"pdf": {"email": "a@b.edu", "lookup_limit": 3}, "institution": {"enabled": False}})
    assert n == 1
    assert paper.arxiv_id == "2401.12345"
    assert paper.signals["pdf_source"] == "s2"
    assert pdf_button_label(paper) == "PDF (arXiv preprint)"


def test_missing_pdf_gets_institution_buttons(monkeypatch):
    def fake(method, url, **kw):
        class Resp:
            status_code = 200
            text = ""
            headers = {}

            def json(self):
                if "openalex" in url:
                    return {"locations": [{"landing_page_url": "https://ieeexplore.ieee.org/document/10902072"}]}
                return {"best_oa_location": None, "oa_locations": []}

        return Resp()

    monkeypatch.setattr("paper_radar.pdf.request", fake)
    monkeypatch.setattr("paper_radar.pdf.confirm_pdf", lambda *a, **k: False)
    paper = Paper(title="Closed IEEE MPC paper about path tracking control", doi="10.1109/tits.2025.1", abstract="a" * 50)
    cfg = {
        "pdf": {"email": "a@b.edu", "lookup_limit": 5, "arxiv_title": False, "deepseek_fallback": False},
        "institution": {"enabled": True},
    }
    assert resolve_pdfs([paper], cfg) == 0
    labels = [item["label"] for item in paper.signals["access"]]
    assert labels == ["WebVPN", "CARSI"]
    assert "ieeexplore.ieee.org/document/10902072" in paper.signals["publisher_url"]
    topics = {"t": type("T", (), {"name": "MPC", "direction": "运动控制"})()}
    cards = feishu.build_cards({"运动控制": [paper]}, topics, "2026-11-03", {"candidates": 1, "selected": 1})
    blob = json.dumps(cards, ensure_ascii=False)
    assert "webvpn.bit.edu.cn" in blob and "wayf.jsp" in blob
    assert len(blob.encode()) < 20000
    assert "暂无开放获取 PDF" in blob


def test_markdown_abstract_and_control_rehome():
    paper = Paper(
        title="Path tracking",
        abstract="## Abstract:\n\nWe track a vehicle.\n## Method\nA controller.",
    )
    assert "##" not in paper.abstract
    assert paper.abstract.startswith("We track")
    assert "Method" in paper.abstract

    class Topic:
        def __init__(self, tid, direction):
            self.id = tid
            self.direction = direction

    topics = {"plan": Topic("plan", "运动规划"), "ctrl": Topic("ctrl", "运动控制")}
    moved = Paper(
        title="Lane-change trajectory planning for an autonomous vehicle",
        abstract="We use MPC as a baseline planner.",
    )
    moved.best_topic = "ctrl"
    moved.signals["direction"] = "运动控制"
    moved.signals["prefilter_topic"] = "plan"
    moved.emb_scores = {"plan": 0.7, "ctrl": 0.72}
    assert lock_control_topics([moved], topics) == 1
    assert moved.best_topic == "plan"
    assert moved.signals["direction"] == "运动规划"


def test_crossref_date_is_not_a_future_issue():
    # imported lazily so the helper stays covered with the biblio tests too
    from paper_radar.sources.deepseek_search import _crossref_date
    assert _crossref_date(
        {"published-print": {"date-parts": [[2027, 2, 1]]}, "published-online": {"date-parts": [[2025, 8, 2]]}},
        date(2026, 11, 3),
    ) == "2025-08-02"
