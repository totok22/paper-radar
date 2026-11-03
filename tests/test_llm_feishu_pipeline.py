import json
from pathlib import Path

from paper_radar.models import Paper
from paper_radar.push import feishu
from paper_radar.push.feishu_files import deliver, pdf_filename
from paper_radar.rank.llm import extract_json, fill_heuristic_text, normalize_review
from paper_radar.sources.base import Source
from paper_radar.store import Store
from paper_radar.validate import ConfigError, validate_config
from paper_radar.weekly import collect, render_md


class Resp:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload
        self.headers = {"content-type": "application/json"}
        self.text = ""

    def json(self):
        return self._payload


def test_llm_json_and_heuristic_chinese():
    from paper_radar.rank.llm import BATCH, SINGLE
    assert "机械臂" in SINGLE and "道路车辆" in BATCH
    raw = "```json\n{\"reviews\": [{\"i\": 1, \"topic\": \"plan\", \"relevance\": 9, \"novelty\": 4, \"rigor\": 7, \"beginner\": 6, \"tldr\": \"用语言模型扩展驾驶基元。\", \"reason\": \"贴近规划方向。\"}]}\n```"
    data = extract_json(raw)
    review = normalize_review(data["reviews"][0], {"plan"})
    assert review["topic"] == "plan"
    assert review["relevance"] == 9
    assert normalize_review({"topic": "nope", "relevance": 3, "tldr": "无关。"}, {"plan"})["topic"] == ""
    paper = Paper(title="A diffusion model for traffic scenarios", abstract="We roll out agents. " * 5,
                  arxiv_id="2401.00008")
    paper.best_topic = "scen"
    paper.signals = {"emb_norm": 0.8, "venue_name": "CoRL", "rigor_hits": ["carla"]}
    topics = {"scen": type("T", (), {"name": "扩散场景", "direction": "场景生成"})()}
    fill_heuristic_text([paper], topics)
    assert paper.llm["heuristic"] is True
    assert "扩散场景" in paper.llm["tldr"]
    assert "CoRL" in paper.llm["reason"]


def test_feishu_card_has_pdf_button_and_splits():
    topics = {"t": type("T", (), {"name": "MPC", "direction": "运动控制"})()}
    papers = []
    for i in range(3):
        p = Paper(title=f"Vehicle MPC paper number {i} with a reasonably long title",
                  abstract="Tracking control. " * 10, arxiv_id=f"2401.0001{i}", authors=["A", "B", "C", "D"])
        p.best_topic = "t"
        p.final_score = 0.66
        p.code_url = "https://github.com/lab/mpc"
        p.llm = {"tldr": "中文摘要 " * 30, "reason": "因为和 MPC 相关 " * 10, "relevance": 8, "novelty": 6, "rigor": 7}
        papers.append(p)
    cards = feishu.build_cards({"运动控制": papers}, topics, "2026-11-03", {"candidates": 10, "prefiltered": 3, "selected": 3, "scorer": "启发式"}, max_bytes=700)
    assert len(cards) >= 2
    blob = json.dumps(cards, ensure_ascii=False)
    assert "https://arxiv.org/pdf/2401.00010" in blob
    assert "https://github.com/lab/mpc" in blob
    actions = [e for c in cards for e in c["card"]["elements"] if e.get("tag") == "action"]
    urls = [a["url"] for block in actions for a in block["actions"]]
    assert "https://arxiv.org/pdf/2401.00010" in urls
    assert feishu.sign("secret", 1700000000) == feishu.sign("secret", 1700000000)
    assert feishu.sign("secret", 1700000000) != feishu.sign("other", 1700000000)


def test_pdf_delivery_disabled_unless_app_env_and_dry_run_does_not_upload(monkeypatch):
    paper = Paper(title="Drive paper", arxiv_id="2401.00077", abstract="a" * 20)
    assert deliver([paper], {}, dry_run=False)["enabled"] is False
    fs = {"app_id": "cli", "app_secret": "sec", "receive_id": "oc_1", "receive_id_type": "chat_id",
          "open_base": "https://open.feishu.cn"}
    plan = deliver([paper], fs, {"push_top_n": 3}, dry_run=True)
    assert plan["sent"] == 0 and plan["would_send"][0]["url"].endswith("2401.00077")
    assert pdf_filename(paper).endswith(".pdf")

    def fake_download(url, max_bytes, timeout=60):
        return b"%PDF-1.4\n" + b"0" * 20

    def fake_request(method, url, **kw):
        if url.endswith("/tenant_access_token/internal"):
            return Resp(200, {"code": 0, "tenant_access_token": "tok"})
        if url.endswith("/im/v1/files"):
            assert kw["data"]["file_type"] == "pdf"
            return Resp(200, {"code": 0, "data": {"file_key": "file_1"}})
        if "/im/v1/messages" in url:
            assert kw["params"]["receive_id_type"] == "chat_id"
            assert kw["json"]["msg_type"] == "file"
            assert "file_1" in kw["json"]["content"]
            return Resp(200, {"code": 0, "data": {}})
        raise AssertionError(url)

    monkeypatch.setattr("paper_radar.push.feishu_files.download_capped", fake_download)
    monkeypatch.setattr("paper_radar.push.feishu_files.request", fake_request)
    sent = deliver([paper], fs, {"push_top_n": 1, "max_bytes": 1000}, dry_run=False)
    assert sent["sent"] == 1


def test_config_flag_works_after_subcommand(tmp_path, monkeypatch):
    from paper_radar.cli import main
    monkeypatch.setattr("paper_radar.pipeline.run", lambda cfg, dry_run=False, no_mark=False: {"stats": {"ok": True, "dry": dry_run}})
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text("topics:\n- {id: t, name: n, direction: d, description: autonomous driving planning}\nlookback_days: 2\n", encoding="utf-8")
    main(["run", "-c", str(cfg_path), "--dry-run"])


def test_shipped_config_validates():
    from paper_radar.config import load_config
    cfg = load_config("config.yaml")
    assert cfg["llm"]["model"] == "deepseek-flash"
    assert cfg["sources"]["semantic_scholar"]["min_interval"] == 1.1
    assert cfg["sources"]["semantic_scholar"]["max_requests"] <= 400
    assert cfg["sources"]["deepseek_search"]["enabled"] is True
    assert cfg["sources"]["deepseek_search"]["max_requests"] <= 50
    assert any(t["id"] == "scen_diffusion" for t in cfg["topics"])


def test_validate_and_weekly_and_dry_run(tmp_path, monkeypatch):
    try:
        validate_config({"topics": [], "lookback_days": 3})
        raise AssertionError("expected ConfigError")
    except ConfigError:
        pass
    try:
        validate_config({
            "lookback_days": 3,
            "topics": [
                {"id": "a", "name": "A", "direction": "运动规划", "description": "planning"},
                {"id": "a", "name": "B", "direction": "运动规划", "description": "planning"},
            ],
        })
        raise AssertionError("duplicate id")
    except ConfigError:
        pass

    store = Store(tmp_path / "state.sqlite")
    liked = Paper(title="Liked paper about MPC tracking", arxiv_id="2401.11111", abstract="abs " * 30)
    liked.best_topic = "ctrl"
    liked.final_score = 0.7
    store.mark_pushed([liked])
    topics = {"ctrl": type("T", (), {"direction": "运动控制", "id": "ctrl", "name": "MPC"})()}
    grouped = collect(store, topics, days=7)
    text = render_md(grouped, "2026-11-03", 7)
    assert "Liked paper" in text and "运动控制" in text

    sent = []
    monkeypatch.setattr("paper_radar.pipeline.feishu.send", lambda *a, **k: sent.append("card"))

    class Fake(Source):
        def fetch(self, topics):
            return [Paper(
                title="Diffusion model for traffic scenario generation with kinematic constraints",
                abstract=("We generate dynamic driving scenarios with a diffusion model and vehicle kinematic constraints. " * 4),
                authors=["A Researcher"],
                published="2026-11-02",
                arxiv_id="2611.01234",
                sources=["arxiv"],
                comments="Accepted to CoRL 2026. Code: https://github.com/lab/scenarios",
            )]

    monkeypatch.setattr("paper_radar.pipeline.REGISTRY", {"arxiv": Fake})
    cfg = {
        "lookback_days": 2,
        "state_db": str(tmp_path / "run.sqlite"),
        "report_dir": str(tmp_path / "reports"),
        "dedup_fuzzy": 0.93,
        "min_abstract_chars": 80,
        "topics": [{
            "id": "scen",
            "name": "扩散动态场景",
            "direction": "场景生成",
            "description": "diffusion models for dynamic traffic scenario generation under kinematic constraints",
            "query": "diffusion traffic scenario",
            "positive": [],
            "negative": [],
        }],
        "sources": {"arxiv": {"enabled": True}},
        "embedding": {},
        "llm": {},
        "quality": {"hf_upvotes": False, "author_hindex": False, "openreview": False},
        "pdf": {"unpaywall": False},
        "ranking": {"top_k": 5, "top_n": 2, "per_topic_max": 3, "mmr_lambda": 0.7,
                    "exploration_slots": 0, "min_emb": 0.0, "min_keep": 1, "min_relevance": 5},
        "weekly": {"weekday": -1},
        "feishu": {},
    }
    from paper_radar.pipeline import run
    res = run(cfg, dry_run=True)
    assert res["stats"]["selected"] == 1
    assert res["stats"]["with_pdf"] == 1
    assert sent == []
    report = Path(res["stats"]["report"]).read_text(encoding="utf-8")
    assert "https://arxiv.org/pdf/2611.01234" in report
    assert "github.com/lab/scenarios" in report
    assert "启发式" in report or "扩散" in report
    cards = json.loads((tmp_path / "reports").joinpath(
        [p.name for p in (tmp_path / "reports").glob("*.feishu.json")][0]
    ).read_text(encoding="utf-8"))
    blob = json.dumps(cards, ensure_ascii=False)
    assert "https://arxiv.org/pdf/2611.01234" in blob
    assert "代码" in blob


def test_embedding_cache_and_http_retry(monkeypatch, tmp_path):
    from paper_radar.rank.embed import Embedder
    from paper_radar import http

    calls = {"n": 0}

    def fake(method, url, **kw):
        calls["n"] += 1
        payload = kw["json"]["input"]
        return Resp(200, {"data": [{"index": i, "embedding": [1.0, 0.0, 0.0]} for i in range(len(payload))]})

    monkeypatch.setattr("paper_radar.rank.embed.request", fake)
    store = Store(tmp_path / "emb.sqlite")
    emb = Embedder({"api_key": "k", "base_url": "https://example.com/v1", "model": "m", "cache": True}, store)
    a = emb.encode(["autonomous driving motion planning"])
    b = emb.encode(["autonomous driving motion planning"])
    assert calls["n"] == 1
    assert a.shape == b.shape

    sleeps = []
    monkeypatch.setattr(http.time, "sleep", lambda s: sleeps.append(s))
    seq = [Resp(429, {}), Resp(200, {"ok": True})]
    # 429 response needs Retry-After via headers; Resp above has content-type only.
    seq[0].headers["Retry-After"] = "1.25"

    def raw(method, url, headers=None, timeout=None, **kw):
        return seq.pop(0)

    monkeypatch.setattr(http.requests, "request", raw)
    got = http.request("GET", "https://example.com/resource", retries=2, pace=0)
    assert got.status_code == 200
    assert sleeps and abs(sleeps[0] - 1.25) < 1e-6
