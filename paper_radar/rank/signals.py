"""Cheap quality signals: venue acceptance, benchmarks/real-vehicle, code, S2 h-index, HF upvotes."""
from __future__ import annotations

import logging
import re
from datetime import date, timedelta

from ..http import request

log = logging.getLogger("paper_radar")

TIER1 = ["CoRL", "RSS", "CVPR", "ICCV", "ECCV", "NeurIPS", "ICLR", "ICML", "T-RO", "TRO",
         "Transactions on Robotics", "Science Robotics", "IJRR", "International Journal of Robotics Research"]
TIER2 = ["ICRA", "IROS", "RA-L", "RAL", "Robotics and Automation Letters", "T-ITS", "Transactions on Intelligent Transportation",
         "T-IV", "Transactions on Intelligent Vehicles", "TCST", "Control Systems Technology", "AAAI", "IJCAI",
         "TVT", "Transactions on Vehicular Technology", "TPAMI", "Pattern Analysis and Machine Intelligence"]
TIER3 = ["ITSC", "IV 20", "Intelligent Vehicles Symposium", "WACV", "ACC", "CDC", "L4DC", "Vehicle System Dynamics"]
ACCEPT = re.compile(r"accept|to appear|published|camera[- ]ready|in proceedings|oral|spotlight", re.I)

RIGOR = {
    "benchmark": ["nuplan", "nuscenes", "waymo", "womd", "carla", "argoverse", "interaction dataset", "highd",
                  "bench2drive", "navsim", "metadrive", "highway-env", "sumo", "commonroad", "ind dataset"],
    "real": ["real vehicle", "real-vehicle", "real-world vehicle", "full-scale", "test vehicle", "on-road",
             "real car", "hardware experiment", "real-world experiment", "closed-loop"],
    "code": ["github.com", "code is available", "code will be", "open-source", "open source", "project page"],
}


def venue_signal(p) -> tuple[float, str]:
    txt = f"{p.comments} {p.venue}"
    accepted = bool(ACCEPT.search(p.comments)) or bool(p.venue and "arxiv" not in p.venue.lower())
    for tier, score in ((TIER1, 1.0), (TIER2, 0.75), (TIER3, 0.5)):
        for v in tier:
            if re.search(rf"(?<![A-Za-z]){re.escape(v)}(?![a-z])", txt):
                return (score if accepted else score * 0.4), v
    return 0.0, ""


def rigor_signal(p) -> tuple[float, list[str]]:
    t = f"{p.abstract} {p.comments}".lower()
    hits = [k for grp in RIGOR.values() for k in grp if k in t]
    s = 0.0
    if any(k in t for k in RIGOR["benchmark"]):
        s += 0.5
    if any(k in t for k in RIGOR["real"]):
        s += 0.3
    if any(k in t for k in RIGOR["code"]):
        s += 0.2
    return min(s, 1.0), hits


def compute_basic(papers) -> None:
    for p in papers:
        vs, vname = venue_signal(p)
        rs, hits = rigor_signal(p)
        p.signals.update({"venue": vs, "venue_name": vname, "rigor": rs, "rigor_hits": hits[:6]})


def add_hf_upvotes(papers, days: int = 4) -> int:
    ups = {}
    for i in range(days):
        d = (date.today() - timedelta(days=i)).isoformat()
        r = request("GET", "https://huggingface.co/api/daily_papers", params={"date": d}, retries=2)
        if r is not None and r.status_code == 200:
            for it in r.json():
                ups[it["paper"]["id"]] = it["paper"].get("upvotes", 0)
    n = 0
    for p in papers:
        if p.arxiv_id in ups:
            p.signals["hf_upvotes"] = ups[p.arxiv_id]
            n += 1
    return n


def add_author_hindex(papers, api_key: str = "", store=None) -> int:
    refs = [p.s2_ref for p in papers if p.s2_ref]
    if not refs:
        return 0
    headers = {"x-api-key": api_key} if api_key else {}
    r = request("POST", "https://api.semanticscholar.org/graph/v1/paper/batch",
                params={"fields": "authors.hIndex,citationCount,venue"}, json={"ids": refs[:500]},
                headers=headers, retries=4, backoff=5)
    if r is None or r.status_code != 200:
        log.warning("S2 batch (h-index) failed: %s", None if r is None else r.status_code)
        return 0
    n = 0
    for ref, d in zip(refs, r.json()):
        if not d:
            continue
        p = next(x for x in papers if x.s2_ref == ref)
        hs = [a.get("hIndex") or 0 for a in d.get("authors") or []]
        if hs:
            p.signals["max_hindex"] = max(hs)
            n += 1
        if d.get("venue") and not p.venue:
            p.venue = d["venue"]
    return n


def quality_score(p) -> float:
    s = p.signals
    h = min((s.get("max_hindex") or 0) / 60.0, 1.0)
    hf = min((s.get("hf_upvotes") or 0) / 30.0, 1.0)
    return 0.5 * s.get("venue", 0) + 0.3 * h + 0.2 * hf
