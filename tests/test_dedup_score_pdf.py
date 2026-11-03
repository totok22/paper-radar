import numpy as np

from paper_radar.dedup import dedup
from paper_radar.models import Paper, norm_title
from paper_radar.pdf import annotate_local, looks_like_pdf, pdf_from_unpaywall, resolve_oa, unpaywall_endpoint
from paper_radar.rank.score import apply_direction_quota, blend_scores, mmr_select, prefer_control_topic, select_indices
from paper_radar.rank.signals import _openreview_summary, compute_basic, in_domain
from paper_radar.sources.openalex import _to_paper


def test_cross_source_dedup_merges_arxiv_doi_and_pdf():
    a = Paper(
        title="Planning with Language Models for Urban Driving",
        abstract="short abstract " * 5,
        arxiv_id="2401.00001v2",
        sources=["arxiv"],
        authors=["Ann"],
    )
    b = Paper(
        title="Planning with Language Models for Urban Driving",
        abstract=("A longer abstract about motion planning. " * 20) + " code https://github.com/lab/planner.",
        doi="https://doi.org/10.48550/arXiv.2401.00001",
        pdf_url="https://arxiv.org/pdf/2401.00001",
        sources=["openalex"],
        affiliations=["Tsinghua University"],
    )
    short = len(a.abstract)
    out = dedup([a, b])
    assert len(out) == 1
    p = out[0]
    assert p.arxiv_id == "2401.00001"
    assert p.doi == "10.48550/arxiv.2401.00001"
    assert "openalex" in p.sources and "arxiv" in p.sources
    assert p.pdf_link == "https://arxiv.org/pdf/2401.00001"
    assert p.code_url == "https://github.com/lab/planner"
    assert p.affiliations == ["Tsinghua University"]
    assert len(p.abstract) > short


def test_dedup_drops_seen_and_fuzzy_titles():
    seen = Paper(title="Model Predictive Control for Vehicle Path Tracking", abstract="a" * 100,
                 arxiv_id="2402.00002")
    again = Paper(title="Model Predictive Control for Vehicle Path Tracking!", abstract="b" * 100,
                  doi="10.1000/xyz")
    dropped = dedup([again], seen_keys={seen.key}, seen_titles={norm_title(seen.title)})
    assert dropped == []
    near = Paper(title="Model Predictive Control for Vehicle Path Tracking Now", abstract="c" * 100,
                 arxiv_id="2402.00003")
    # Same 12-char prefix and high ratio should collapse; a clearly different title should not.
    twin = Paper(title="Model Predictive Control for Vehicle Path Tracking Now.", abstract="d" * 120,
                 doi="10.1000/twin")
    other = Paper(title="Completely Unrelated Medical Imaging Survey Paper", abstract="e" * 100,
                  arxiv_id="2402.00009")
    merged = dedup([near, twin, other])
    assert len(merged) == 2
    assert any(p.arxiv_id == "2402.00009" for p in merged)


def test_calibrated_threshold_and_exploration_slot():
    scores = np.array([0.1, 0.95, 0.4, 0.2])
    assert select_indices(scores, top_k=2, min_emb=0.35, min_keep=1) == [1, 2]
    # Nothing clears the floor: still keep min_keep, highest first.
    assert select_indices(np.array([0.1, 0.2, 0.05]), top_k=5, min_emb=0.8, min_keep=2)[:2] == [1, 0]
    # Off-domain sentinels must not refill min_keep.
    assert select_indices(np.array([-1e9, 0.9, -1e9]), top_k=5, min_emb=0.35, min_keep=3) == [1]

    profile = np.array([[0.2, 0.9], [0.8, 0.1]])
    seeds = np.array([[np.nan, 0.4], [0.2, np.nan]])
    neg = np.zeros_like(profile)
    has_neg = np.array([False, False])
    blended = blend_scores(profile, seeds, neg, has_neg)
    # Topic 0 has no seed for paper 0, so the profile similarity is kept.
    assert blended[0, 0] == profile[0, 0]
    # Topic 1 blends profile and seed.
    assert abs(blended[0, 1] - (0.55 * 0.9 + 0.45 * 0.4)) < 1e-9

    papers = []
    for i, topic, score, emb in (
        (0, "plan", 0.9, 0.2),
        (1, "plan", 0.8, 0.2),
        (2, "scen", 0.3, 0.95),
    ):
        p = Paper(title=f"Paper title number {i} about driving", abstract="abs")
        p.best_topic = topic
        p.final_score = score
        p.emb_score = emb
        papers.append(p)
    vecs = np.eye(3)
    picked = mmr_select(papers, vecs, n=2, lam=0.9, per_topic_max=2, exploration=1)
    assert [p.best_topic for p in picked] == ["plan", "scen"]
    assert picked[1].signals.get("explore") is True

    covered = []
    for i, direction, topic, score in (
        (0, "运动控制", "mpc", 0.9),
        (1, "运动控制", "dyn", 0.85),
        (2, "运动规划", "llm", 0.4),
    ):
        p = Paper(title=f"Coverage paper title number {i}", abstract="abs")
        p.best_topic = topic
        p.signals["direction"] = direction
        p.final_score = score
        p.emb_score = score
        covered.append(p)
    picked = mmr_select(covered, np.eye(3), n=2, lam=1.0, per_topic_max=3, exploration=1)
    assert {p.signals["direction"] for p in picked} == {"运动控制", "运动规划"}
    assert any(p.signals.get("explore") for p in picked)

    vehicle = Paper(title="Vehicle MPC on the highway", abstract="path tracking")
    vehicle.best_topic = "mpc"
    vehicle.signals = {"direction": "运动控制"}
    vehicle.final_score = 0.8
    vehicle.emb_score = 0.8
    extra = Paper(title="Another vehicle controller", abstract="lane keeping")
    extra.best_topic = "dyn"
    extra.signals = {"direction": "运动控制"}
    extra.final_score = 0.7
    extra.emb_score = 0.7
    warehouse = Paper(title="Warehouse multi-robot planning", abstract="mobile robots")
    warehouse.best_topic = "plan"
    warehouse.signals = {"direction": "运动规划", "off_domain": True}
    warehouse.final_score = 0.99
    warehouse.emb_score = 0.99
    picked = mmr_select([vehicle, extra, warehouse], np.eye(3), n=2, lam=1.0, per_topic_max=2, exploration=1)
    assert all(not p.signals.get("off_domain") for p in picked)
    assert {p.signals["direction"] for p in picked} == {"运动控制"}


def test_openalex_pdf_and_quality_signals():
    work = {
        "title": "Scenario generation under kinematic constraints",
        "doi": "https://doi.org/10.48550/arXiv.2401.00002",
        "publication_date": "2026-11-01",
        "authorships": [{
            "author": {"display_name": "Li"},
            "institutions": [{"display_name": "Tsinghua University"}],
        }],
        "best_oa_location": {"pdf_url": "https://arxiv.org/pdf/2401.00002"},
        "abstract_inverted_index": {"Driving": [0], "scenario": [1]},
        "primary_location": {"source": {"display_name": "arXiv"}},
    }
    paper = _to_paper(work, "openalex")
    assert paper.arxiv_id == "2401.00002"
    assert paper.pdf_url == "https://arxiv.org/pdf/2401.00002"
    assert paper.affiliations == ["Tsinghua University"]
    paper.comments = "Accepted to ICRA 2026"
    paper.abstract = "Closed-loop evaluation on nuPlan. Code: https://github.com/lab/demo"
    compute_basic([paper])
    assert paper.signals["venue_name"] == "ICRA"
    assert paper.signals["institution"] == "tsinghua"
    assert paper.code_url == "https://github.com/lab/demo"
    assert "nuplan" in paper.signals["rigor_hits"]

    web = Paper(title="Progressive Skill Exploration for LLM Web Agents",
                abstract="Large language models browse websites and complete long-horizon tasks.")
    drive = Paper(title="MPC path tracking for an autonomous vehicle",
                  abstract="We control a vehicle along a reference trajectory.")
    assert in_domain(drive) and not in_domain(web)
    assert not in_domain(Paper(
        title="Energy-Based Policy Orchestration for a Robot Manipulator",
        abstract="We orchestrate energy-based policies for a 7-DoF robotic arm and its gripper.",
    ))
    assert not in_domain(Paper(
        title="Warehouse multi-robot task planning",
        abstract="Multi-robot motion planning and task allocation for mobile robots in a warehouse.",
    ))
    assert not in_domain(Paper(
        title="RLiD: Robust Local Interaction Dynamics",
        abstract="A policy for coordinating indoor service robots in an office building.",
    ))
    assert not in_domain(Paper(
        title="Nonlinear MPC for a robotic manipulator",
        abstract="Real-time model predictive control of a 6-axis robot arm.",
    ))
    assert in_domain(Paper(
        title="Manipulator primitives transferred to autonomous driving",
        abstract="A robot-arm DMP library is adapted for ego-vehicle lane change on public roads.",
    ))
    # Control papers often never say "vehicle". Those phrases count only after the
    # off-vehicle screen, so a gripper that mentions a tire stays out.
    assert in_domain(Paper(
        title="Yaw-rate feedback for path tracking",
        abstract="Steering and a tire model on a chassis with sideslip.",
    ))
    assert in_domain(Paper(
        title="Cornering stiffness from electric vehicle data",
        abstract="Vehicle dynamics parameter identification for adaptive cruise control.",
    ))
    assert not in_domain(Paper(
        title="Tire-inspired gripper",
        abstract="A robotic manipulator grasps with a tire-like pad and yaw control of the wrist.",
    ))
    assert not in_domain(Paper(
        title="Warehouse fleet yaw scheduling",
        abstract="Multi-robot yaw control for a warehouse fleet.",
    ))
    assert not in_domain(Paper(
        title="Soft actor-critic for continuous control",
        abstract="A generic reinforcement learning policy trained on MuJoCo locomotion.",
    ))

    summary = _openreview_summary({
        "count": 2,
        "notes": [
            {"content": {"rating": {"value": "6: marginally above the acceptance threshold"}}},
            {"content": {"rating": "8"}},
        ],
    })
    assert summary["ratings"] == [6.0, 8.0]
    assert _openreview_summary({"count": 500, "notes": []}) == {"ratings": [], "venues": []}


def test_unpaywall_fills_missing_pdf(monkeypatch):
    paper = Paper(title="Journal article about vehicle MPC tracking", doi="10.1109/TIV.2024.1",
                  abstract="a" * 40)

    class Resp:
        status_code = 200
        headers = {}

        def json(self):
            return {"best_oa_location": {"url_for_pdf": "https://oa.example.org/paper.pdf"}}

    monkeypatch.setattr("paper_radar.pdf.request", lambda *a, **k: Resp())
    n = resolve_oa([paper], {"pdf": {"email": "me@lab.edu", "unpaywall": True, "lookup_limit": 5}})
    assert n == 1
    assert paper.pdf_url == "https://oa.example.org/paper.pdf"
    landing = Paper(title="Landing page is not a direct PDF file", pdf_url="https://publisher.example/article")
    annotate_local(landing)
    assert landing.pdf_url is None
    assert looks_like_pdf("https://arxiv.org/pdf/2401.00001")
    assert looks_like_pdf(
        "https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0000217&type=printable"
    )
    assert not looks_like_pdf("https://doi.org/10.1/abc")
    assert not looks_like_pdf("https://publisher.example/article")


def test_unpaywall_printable_url_and_bare_doi(monkeypatch):
    seen = {}

    class Resp:
        status_code = 200
        headers = {}

        def json(self):
            return {"best_oa_location": {
                "url_for_pdf": "https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0000217&type=printable",
            }}

    def fake(method, url, **kw):
        seen["url"] = url
        seen["params"] = kw.get("params")
        return Resp()

    monkeypatch.setattr("paper_radar.pdf.request", fake)
    paper = Paper(
        title="An open vehicle dynamics note",
        doi="https://doi.org/10.1371/journal.pone.0000217",
        abstract="a" * 40,
    )
    n = resolve_oa([paper], {"pdf": {"email": "me@lab.edu", "unpaywall": True, "lookup_limit": 5}})
    assert n == 1
    assert "type=printable" in paper.pdf_url
    assert seen["url"] == unpaywall_endpoint("https://doi.org/10.1371/journal.pone.0000217")
    assert "/doi.org/" not in seen["url"]
    assert seen["params"]["email"] == "me@lab.edu"
    kept = pdf_from_unpaywall({"best_oa_location": {"url_for_pdf": "https://doi.org/10.1/x"}})
    assert kept is None


def test_openalex_pdf_when_unpaywall_has_none(monkeypatch):
    def fake(method, url, **kw):
        class Resp:
            headers = {}

            def __init__(self, code, payload):
                self.status_code = code
                self._payload = payload

            def json(self):
                return self._payload

        if "unpaywall.org" in url:
            assert kw["params"]["email"] == "me@lab.edu"
            return Resp(200, {"best_oa_location": {"url_for_pdf": None, "url": "https://doi.org/10.1109/tiv.2024.1"}})
        if "openalex.org/works/doi:" in url:
            assert url.endswith("doi:10.1109/tiv.2024.1")
            assert kw["params"]["mailto"] == "me@lab.edu"
            return Resp(200, {"best_oa_location": {"pdf_url": "https://publisher.example/getpdf?id=9"}})
        raise AssertionError(url)

    monkeypatch.setattr("paper_radar.pdf.request", fake)
    paper = Paper(title="Closed-looking IEEE article", doi="10.1109/TIV.2024.1", abstract="a" * 40)
    n = resolve_oa([paper], {"pdf": {"email": "me@lab.edu", "unpaywall": True, "lookup_limit": 5}})
    assert n == 1
    assert paper.pdf_url == "https://publisher.example/getpdf?id=9"


def test_control_near_tie_and_direction_minimum():
    row = np.array([0.80, 0.70, 0.10])
    directions = ["运动规划", "运动控制", "场景生成"]
    assert prefer_control_topic(row, directions, "Path tracking MPC with a tire model") == 1
    assert prefer_control_topic(row, directions, "Diffusion traffic scenario generation") == 0
    assert prefer_control_topic(np.array([0.90, 0.50, 0.1]), directions, "path tracking") == 0

    papers = []
    for i in range(5):
        p = Paper(title=f"Planning paper {i}", abstract="autonomous driving lane change " * 3)
        p.signals["direction"] = "运动规划"
        p.final_score = 10 - i
        p.emb_score = 0.8
        p.best_topic = "plan"
        papers.append(p)
    for i in range(2):
        p = Paper(title=f"Control paper {i}", abstract="path tracking steering yaw-rate " * 3)
        p.signals["direction"] = "运动控制"
        p.final_score = 1.0 - i * 0.1
        p.emb_score = 0.3
        p.best_topic = "ctrl"
        papers.append(p)
    warehouse = Paper(title="Warehouse filler", abstract="warehouse multi-robot " * 3)
    warehouse.signals["direction"] = "运动控制"
    warehouse.signals["off_domain"] = True
    warehouse.final_score = 100
    warehouse.emb_score = 0.99
    warehouse.best_topic = "ctrl"
    papers.append(warehouse)

    chosen = mmr_select(
        papers, None, n=6, lam=1.0, per_topic_max=10, exploration=0,
        direction_mins={"运动控制": 2}, per_direction_min=1,
    )
    ctrl = [p for p in chosen if p.signals.get("direction") == "运动控制"]
    assert len(chosen) == 6
    assert len(ctrl) == 2
    assert all("Warehouse" not in p.title for p in chosen)

    only_plan = papers[:5]
    stayed = mmr_select(
        only_plan, None, n=4, lam=1.0, per_topic_max=10, exploration=0,
        direction_mins={"运动控制": 2}, per_direction_min=1,
    )
    assert len(stayed) == 4
    assert all(p.signals["direction"] == "运动规划" for p in stayed)

    scores = np.array([0.9, 0.8, 0.7, 0.25, 0.24])
    quota_papers = []
    for i, direction in enumerate(["运动规划", "运动规划", "运动规划", "运动控制", "运动控制"]):
        p = Paper(title=f"q{i}", abstract="vehicle " * 10)
        p.signals["direction"] = direction
        p.emb_score = float(scores[i])
        quota_papers.append(p)
    seated = apply_direction_quota(quota_papers, scores, [0, 1, 2], per_direction=2, floor=0.22, cap=4)
    seated_dirs = [quota_papers[i].signals["direction"] for i in seated]
    assert seated_dirs.count("运动控制") == 2
    assert len(seated) == 4
