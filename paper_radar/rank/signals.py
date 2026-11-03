"""Cheap quality signals that do not need an LLM.

Venue lines are parsed from arXiv comments (the reliable public acceptance
signal). OpenReview, when it answers, contributes a mean public review
rating — not a guessed accept/reject. Author h-index, Hugging Face daily
upvotes, a short institution list, and code/benchmark mentions fill the rest.
Any of these can fail; the run continues.
"""
from __future__ import annotations

import logging
import re
from datetime import date, timedelta

from ..http import request
from ..pdf import annotate_local, looks_like_pdf
from ..s2 import get_client

log = logging.getLogger("paper_radar")

TIER1 = ["CoRL", "RSS", "CVPR", "ICCV", "ECCV", "NeurIPS", "ICLR", "ICML", "T-RO", "TRO",
         "Transactions on Robotics", "Science Robotics", "IJRR", "International Journal of Robotics Research"]
TIER2 = ["ICRA", "IROS", "RA-L", "RAL", "Robotics and Automation Letters", "T-ITS",
         "Transactions on Intelligent Transportation", "T-IV", "Transactions on Intelligent Vehicles",
         "TCST", "Control Systems Technology", "Control Engineering Practice",
         "AAAI", "IJCAI", "TVT", "Transactions on Vehicular Technology",
         "TPAMI", "Pattern Analysis and Machine Intelligence"]
TIER3 = ["ITSC", "IV 20", "Intelligent Vehicles Symposium", "WACV", "ACC", "CDC", "L4DC", "Vehicle System Dynamics"]
ACCEPT = re.compile(r"accept|to appear|published|camera[- ]ready|in proceedings|oral|spotlight", re.I)
# On-road vehicles only. "robot", "planning", "MPC", and "control" alone are not enough:
# they also match warehouse fleets, manipulators, and legged robots, which then clear the
# embedding floor because those words sit in the topic profiles.
_STRONG_DRIVING = re.compile(
    r"(autonomous driving|self[- ]driving|on[- ]road|ego[- ]vehicle|"
    r"lane[- ](?:change|keeping|changing)|urban driving|highway driving|"
    r"traffic scenario|driving scenario|autonomous vehicle|"
    r"\b(?:nuplan|nuscenes|waymo|womd|argoverse|carla|commonroad|metadrive|bench2drive)\b|"
    r"\bhighway-env\b)",
    re.I,
)
_WEAK_VEHICLE = re.compile(
    r"\b(driving|driver|drivers|vehicle|vehicles|vehicular|automotive|automobile|"
    r"car|cars|truck|trucks|motorway|roadway|intersection|highway|lane|traffic)\b",
    re.I,
)
# Vehicle-control wording that often appears without the word "vehicle".
# Checked only after the off-vehicle robotics screen, so a manipulator paper
# that mentions yaw or MPC stays out.
_VEHICLE_CONTROL = re.compile(
    r"(path[- ]tracking|lane[- ]keeping|"
    r"\b(?:tire|tyre|tires|tyres)\b|\bsteering\b|"
    r"yaw[- ](?:rate|moment|stability|control|dynamics)|"
    r"\bsideslip\b|\bchassis\b|"
    r"bicycle model|single[- ]track|"
    r"adaptive cruise|\bcruise control\b|"
    r"\bwheel slip\b|\bundersteer\b|\boversteer\b|"
    r"automated vehicle|ground vehicle|"
    r"vehicle dynamics|electric vehicle)",
    re.I,
)
# Non-vehicle robotics. Kept only when the same text also has a strong driving phrase
# (the method is explicitly carried over to on-road vehicles).
_OFF_VEHICLE = re.compile(
    r"(robotic manipulator|robot manipulator|manipulator arm|robotic arm|robot arm|"
    r"\bgripper\b|\bgrasping\b|end[- ]effector|pick[- ]and[- ]place|bin picking|"
    r"\bwarehouse\b|fulfillment center|fulfilment centre|"
    r"\blegged\b|\bhumanoid\b|\bquadruped\b|\bbiped\b|"
    r"\bdrone\b|\buav\b|unmanned aerial|quadrotor|quadcopter|"
    r"surgical robot|household robot|domestic robot|mobile manipulator|"
    r"multi[- ]robot)",
    re.I,
)


def _domain_text(p) -> str:
    raw = f"{p.title}\n{p.abstract}"
    return re.sub(r"network traffic", " ", raw, flags=re.I)


def domain_label(p) -> str:
    """`vehicle` or `out`. Transfer into driving still counts as vehicle."""
    text = _domain_text(p)
    if _STRONG_DRIVING.search(text):
        return "vehicle"
    if _OFF_VEHICLE.search(text):
        return "out"
    if _WEAK_VEHICLE.search(text) or _VEHICLE_CONTROL.search(text):
        return "vehicle"
    return "out"


def in_domain(p) -> bool:
    return domain_label(p) == "vehicle"

# Substring match against OpenAlex affiliation strings. Short names are anchored.
_LABS = (
    "waymo", "nvidia", "tsinghua", "peking university", "baidu", "huawei", "motional",
    "waabi", "toyota research", "bosch", "mercedes-benz", "zoox", "shanghai ai laboratory",
    "shanghai artificial intelligence laboratory", "institute of automation",
    "horizon robotics", "sensetime", "megvii", "applied intuition", "tesla",
    "cruise llc", "argonne", "tencent", "alibaba", "didi", "momenta", "pony.ai", "ponyai",
)

RIGOR = {
    "benchmark": ["nuplan", "nuscenes", "waymo", "womd", "carla", "argoverse", "interaction dataset", "highd",
                  "bench2drive", "navsim", "metadrive", "highway-env", "sumo", "commonroad", "ind dataset"],
    "real": ["real vehicle", "real-vehicle", "real-world vehicle", "full-scale", "test vehicle", "on-road",
             "real car", "hardware experiment", "real-world experiment", "closed-loop"],
    "code": ["github.com", "code is available", "code will be", "open-source", "open source", "project page"],
}

_RATING = re.compile(r"(\d+(?:\.\d+)?)")


def venue_signal(p) -> tuple[float, str]:
    txt = f"{p.comments} {p.venue}"
    accepted = bool(ACCEPT.search(p.comments or "")) or bool(p.venue and "arxiv" not in p.venue.lower())
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
        s += 0.35
    if any(k in t for k in RIGOR["code"]) or p.code_url:
        s += 0.15
    return min(s, 1.0), hits


def institution_signal(p) -> tuple[float, str]:
    blob = " ; ".join(p.affiliations or []).lower()
    if not blob:
        return 0.0, ""
    for name in _LABS:
        if name in blob:
            return 0.8, name
    return 0.0, ""


def compute_basic(papers) -> None:
    for p in papers:
        annotate_local(p)
        vs, vname = venue_signal(p)
        rs, hits = rigor_signal(p)
        inst, iname = institution_signal(p)
        # Keep a stronger score already filled by OpenReview / an earlier pass.
        prev = float(p.signals.get("venue") or 0)
        if vs >= prev:
            p.signals["venue"] = vs
            if vname:
                p.signals["venue_name"] = vname
        elif vname and not p.signals.get("venue_name"):
            p.signals["venue_name"] = vname
        p.signals.update({
            "rigor": rs, "rigor_hits": hits[:6],
            "institution": iname, "institution_score": inst,
        })


def add_hf_upvotes(papers, days: int = 4) -> int:
    ups: dict[str, int] = {}
    for i in range(days):
        d = (date.today() - timedelta(days=i)).isoformat()
        r = request("GET", "https://huggingface.co/api/daily_papers", params={"date": d}, retries=2, timeout=25)
        if r is not None and r.status_code == 200:
            try:
                payload = r.json()
            except ValueError:
                continue
            for it in payload:
                paper = it.get("paper") or {}
                if paper.get("id"):
                    ups[paper["id"]] = paper.get("upvotes", 0)
        elif r is None or r.status_code >= 400:
            log.info("HF daily papers unavailable for %s", d)
            break
    n = 0
    for p in papers:
        if p.arxiv_id and p.arxiv_id in ups:
            p.signals["hf_upvotes"] = ups[p.arxiv_id]
            n += 1
    return n


def _hindexes(authors: list) -> list[int]:
    out = []
    for a in authors or []:
        if isinstance(a, dict) and isinstance(a.get("hIndex"), int):
            out.append(a["hIndex"])
    return out


def _apply_s2_meta(p, d: dict) -> None:
    hs = _hindexes(d.get("authors") or [])
    if hs:
        p.signals["max_hindex"] = max(hs)
    if d.get("venue") and not p.venue:
        p.venue = d["venue"]
    cites = d.get("citationCount")
    if isinstance(cites, int):
        p.citation_count = cites
        p.signals["citations"] = cites
    oa = d.get("openAccessPdf") or {}
    url = oa.get("url") if isinstance(oa, dict) else None
    if url and looks_like_pdf(url) and not (p.pdf_url and looks_like_pdf(p.pdf_url)):
        p.pdf_url = url


def add_author_hindex(papers, api_key: str = "", store=None) -> int:
    """One paper/batch for metadata. author/batch only if h-index was absent."""
    refs = list(dict.fromkeys(p.s2_ref for p in papers if p.s2_ref))
    if not refs:
        return 0
    client = get_client(api_key)
    rows = client.paper_batch(
        refs, "authors,authors.authorId,authors.hIndex,citationCount,venue,openAccessPdf",
    )
    by_ref = {ref: d for ref, d in zip(refs, rows) if d}
    need_authors: list[str] = []
    for d in by_ref.values():
        authors = d.get("authors") or []
        if _hindexes(authors):
            continue
        for a in authors:
            aid = a.get("authorId") if isinstance(a, dict) else None
            if aid:
                need_authors.append(aid)
    author_h: dict[str, int] = {}
    if need_authors:
        arows = client.author_batch(need_authors, "name,hIndex")
        for aid, row in zip(list(dict.fromkeys(need_authors)), arows):
            if isinstance(row, dict) and isinstance(row.get("hIndex"), int):
                author_h[aid] = row["hIndex"]
    n = 0
    for p in papers:
        d = by_ref.get(p.s2_ref or "")
        if not d:
            continue
        _apply_s2_meta(p, d)
        if "max_hindex" not in p.signals and author_h:
            hs = [author_h[a["authorId"]] for a in (d.get("authors") or [])
                  if isinstance(a, dict) and a.get("authorId") in author_h]
            if hs:
                p.signals["max_hindex"] = max(hs)
        if p.signals.get("max_hindex"):
            n += 1
    return n


def _rating_num(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    m = _RATING.search(str(value))
    if not m:
        return None
    num = float(m.group(1))
    if 0 < num <= 10:
        return num
    return None


def _openreview_summary(payload: dict) -> dict:
    notes = payload.get("notes") or []
    count = int(payload.get("count") or len(notes))
    if count > 80:
        # The query was not specific to one paper.
        return {"ratings": [], "venues": []}
    ratings, venues = [], []
    for note in notes:
        content = note.get("content") or {}
        rating = content.get("rating")
        if isinstance(rating, dict):
            rating = rating.get("value")
        num = _rating_num(rating)
        if num is not None:
            ratings.append(num)
        venue = content.get("venue")
        if isinstance(venue, dict):
            venue = venue.get("value")
        if isinstance(venue, str) and venue and "corr" not in venue.lower() and "arxiv" not in venue.lower():
            venues.append(venue)
    return {"ratings": ratings[:12], "venues": venues[:3]}


def add_openreview(papers, store=None, limit: int = 8) -> int:
    """Mean public OpenReview rating for arXiv ids. Stops on the first hard failure."""
    n = 0
    checked = 0
    for p in papers:
        if checked >= limit or not p.arxiv_id:
            continue
        key = f"openreview:{p.arxiv_id}"
        cached = store.cache_get(key) if store is not None else None
        if not isinstance(cached, dict):
            r = request(
                "GET", "https://api2.openreview.net/notes/search",
                params={"term": f"arxiv:{p.arxiv_id}", "limit": 20},
                retries=2, timeout=25,
            )
            if r is None or r.status_code != 200:
                log.info("OpenReview unavailable (%s); skip the rest", None if r is None else r.status_code)
                break
            try:
                cached = _openreview_summary(r.json())
            except ValueError:
                break
            if store is not None:
                store.cache_set(key, cached)
        checked += 1
        ratings = cached.get("ratings") or []
        if ratings:
            p.signals["openreview_rating"] = round(sum(ratings) / len(ratings), 2)
            p.signals["openreview_n"] = len(ratings)
            n += 1
        venues = cached.get("venues") or []
        if venues and not p.signals.get("venue_name"):
            p.signals["venue_name"] = venues[0]
            p.signals["venue"] = max(float(p.signals.get("venue") or 0), 0.75)
    return n


def quality_score(p) -> float:
    s = p.signals
    h = min((s.get("max_hindex") or 0) / 60.0, 1.0)
    hf = min((s.get("hf_upvotes") or 0) / 30.0, 1.0)
    inst = float(s.get("institution_score") or 0)
    orating = min((s.get("openreview_rating") or 0) / 10.0, 1.0)
    cites = s.get("citations") or p.citation_count or 0
    # Recent papers are usually uncited; this only nudges ones that already traveled.
    cite = min((cites or 0) / 40.0, 1.0)
    return (0.40 * float(s.get("venue") or 0) + 0.22 * h + 0.13 * hf
            + 0.12 * inst + 0.08 * orating + 0.05 * cite)
