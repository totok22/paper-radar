import numpy as np

from paper_radar.dedup import dedup
from paper_radar.models import Paper, norm_title
from paper_radar.pdf import annotate_local, looks_like_pdf, resolve_oa
from paper_radar.rank.score import blend_scores, mmr_select, select_indices
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
    assert not looks_like_pdf("https://doi.org/10.1/abc")
