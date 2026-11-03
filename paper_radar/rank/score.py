"""Per-topic interest profiles, a calibrated floor, MMR, and one exploration slot.

Each topic is a profile (name + description + seed text) plus up to a few
positive seed abstracts. A paper's topic score blends similarity to the
profile with the mean of its top seed matches, then subtracts negatives.
The paper keeps the max over topics (one strong direction is enough).
Scores are scaled by each topic's 95th percentile so a long profile does
not drown a short one. Papers under `min_emb` are dropped, but at least
`min_keep` in-domain papers survive so a quiet day still produces a digest.
Non-vehicle robotics never fill that floor or a direction slot.
"""
from __future__ import annotations

import logging
import re

import numpy as np

from .embed import Embedder

log = logging.getLogger("paper_radar")


def blend_scores(sim_profile: np.ndarray, seed_mean: np.ndarray, sim_neg: np.ndarray,
                 has_neg: np.ndarray, neg_w: float = 0.5) -> np.ndarray:
    """Combine profile / seed / negative similarities. NaN in seed_mean = no seeds."""
    blended = 0.55 * sim_profile + 0.45 * seed_mean
    sim = np.where(np.isnan(seed_mean), sim_profile, blended)
    return sim - float(neg_w) * sim_neg * has_neg.astype(float)


def calibrate(sim: np.ndarray) -> np.ndarray:
    """Divide each topic column by its 95th percentile of positive mass."""
    pos = np.maximum(sim, 0)
    scale = np.percentile(pos, 95, axis=0) + 1e-9
    return sim / scale


_CONTROL_SLOT = re.compile(
    r"(path[- ]tracking|trajectory[- ]tracking|"
    r"model predictive control|\bmpc\b|"
    r"\bcontrollers?\b|control law|"
    r"state estimation|"
    r"(?:dynamics|parameter|system) identification|"
    r"parameter (?:tuning|estimation)|"
    r"lateral control|longitudinal control|"
    r"learning-based mpc|tracking control)",
    re.I,
)
# Title-level planning / prediction. These do not fill the control slot even
# when the abstract mentions MPC or a controller in passing.
_NOT_CONTROL_TITLE = re.compile(
    r"(trajectory planning|motion planning|path planning|"
    r"lane[- ]change (?:trajectory|planning)|"
    r"trajectory prediction|motion prediction|behaviou?r prediction)",
    re.I,
)


def has_control_content(text: str) -> bool:
    """True for tracking, MPC, a controller, state estimation, or parameter tuning.

    A title that is trajectory planning or motion prediction does not qualify.
    The control minimum stays empty instead of borrowing those papers.
    """
    if not text or not _CONTROL_SLOT.search(text):
        return False
    title = text.split(". ", 1)[0]
    if _NOT_CONTROL_TITLE.search(title):
        return False
    return True


def prefer_control_topic(scores_row, directions: list[str], text: str, margin: float = 0.12) -> int:
    """If a vehicle-control paper is nearly tied, keep it on a control topic.

    Planning profiles also mention trajectories and MPC, so a path-tracking
    paper can lose the argmax by a hair and then never fill the control slot.
    Planning and prediction titles are not moved.
    """
    best = int(np.argmax(scores_row))
    if not has_control_content(text):
        return best
    ctrl = [j for j, d in enumerate(directions) if d == "运动控制"]
    if not ctrl:
        return best
    j = max(ctrl, key=lambda k: float(scores_row[k]))
    if float(scores_row[j]) >= float(scores_row[best]) - margin:
        return j
    return best


def apply_direction_quota(papers, scores: np.ndarray, chosen: list[int], *,
                          per_direction: int, floor: float, cap: int) -> list[int]:
    """Give each direction a few shortlist seats without admitting off-domain papers.

    Global top-k is dominated by planning. Control papers just under that cut
    still enter the LLM pool when they clear the softer per-direction floor.
    """
    if per_direction <= 0:
        return list(chosen)
    chosen = list(chosen)
    pools: dict[str, list[int]] = {}
    for i, p in enumerate(papers):
        if p.signals.get("off_domain"):
            continue
        if float(scores[i]) < floor or float(scores[i]) <= -1e8:
            continue
        d = p.signals.get("direction") or ""
        if d == "运动控制" and not has_control_content(p.text):
            continue
        if d:
            pools.setdefault(d, []).append(i)
    for pool in pools.values():
        pool.sort(key=lambda i: -float(scores[i]))
    for d, pool in pools.items():
        have = [i for i in chosen if (papers[i].signals.get("direction") or "") == d
                and not (d == "运动控制" and not has_control_content(papers[i].text))]
        need = per_direction - len(have)
        for i in pool:
            if need <= 0:
                break
            if i in chosen:
                continue
            if len(chosen) < cap:
                chosen.append(i)
                need -= 1
                continue
            donors = []
            for j in chosen:
                dj = papers[j].signals.get("direction") or ""
                n_here = sum(1 for k in chosen if (papers[k].signals.get("direction") or "") == dj)
                if n_here > per_direction:
                    donors.append(j)
            if not donors:
                break
            donor = min(donors, key=lambda j: float(scores[j]))
            chosen.remove(donor)
            chosen.append(i)
            need -= 1
    return chosen


def select_indices(scores: np.ndarray, top_k: int, min_emb: float, min_keep: int) -> list[int]:
    # Scores at or below -1e8 are off-domain sentinels and must not refill min_keep.
    order = [int(i) for i in np.argsort(-scores) if float(scores[i]) > -1e8]
    passed = [i for i in order if scores[i] >= min_emb]
    if len(passed) < min_keep:
        passed = order[:max(min_keep, 0)]
    return passed[:top_k]


def _seed_texts(topic, limit: int) -> list[str]:
    texts = [t.strip() for t in topic.positive_texts if t and len(t.strip()) > 40]
    texts = sorted(set(texts), key=len, reverse=True)
    return texts[:limit]


def prefilter(papers, topics, ecfg: dict, top_k: int, store=None,
              min_emb: float = 0.0, min_keep: int = 8,
              direction_keep: int = 0, direction_min_emb: float = 0.0):
    """Return (kept papers, their embedding rows)."""
    if not papers:
        prefilter.backend = "none"
        return [], None
    ecfg = ecfg or {}
    tlist = list(topics.values())
    emb = Embedder(ecfg, store)
    max_seeds = int(ecfg.get("seeds_per_topic", 6))
    profiles = [t.profile_text for t in tlist]
    # Every topic pushes away non-vehicle robotics, even when the yaml has no negatives.
    # Otherwise "MPC" and "planning" in the profiles pull manipulators and warehouse fleets
    # over min_emb.
    off_neg = (
        "robotic manipulator arm grasping gripper warehouse multi-robot "
        "humanoid quadruped legged locomotion drone UAV surgical robot"
    )
    negatives = [
        f"{' '.join(t.negative_texts)} {off_neg}".strip() if t.negative_texts else off_neg
        for t in tlist
    ]
    groups = [_seed_texts(t, max_seeds) for t in tlist]
    seed_flat = [s for g in groups for s in g]
    neg_rows = negatives
    texts = profiles + neg_rows + seed_flat + [p.text for p in papers]
    X = emb.encode(texts)
    nT = len(tlist)
    T = X[:nT]
    N = X[nT:2 * nT]
    S = X[2 * nT:2 * nT + len(seed_flat)]
    P = X[2 * nT + len(seed_flat):]
    sim_profile = P @ T.T
    sim_neg = P @ N.T
    seed_mean = np.full_like(sim_profile, np.nan)
    offset = 0
    for j, group in enumerate(groups):
        if not group:
            continue
        block = S[offset:offset + len(group)]
        seed_sim = P @ block.T
        k = min(3, seed_sim.shape[1])
        seed_mean[:, j] = np.sort(seed_sim, axis=1)[:, -k:].mean(axis=1)
        offset += len(group)
    has_neg = np.array([True] * len(negatives))
    sim = blend_scores(sim_profile, seed_mean, sim_neg, has_neg, float(ecfg.get("negative_weight", 0.5)))
    z = calibrate(sim)
    from .signals import in_domain
    directions = [t.direction for t in tlist]
    for i, p in enumerate(papers):
        j = prefer_control_topic(z[i], directions, p.text)
        if directions[j] == "运动控制" and not has_control_content(p.text):
            others = [k for k, d in enumerate(directions) if d != "运动控制"]
            if others:
                j = max(others, key=lambda k: float(z[i, k]))
        p.emb_scores = {t.id: round(float(sim[i, k]), 4) for k, t in enumerate(tlist)}
        p.best_topic = tlist[j].id
        p.signals["direction"] = tlist[j].direction
        p.signals["prefilter_topic"] = tlist[j].id
        p.emb_score = float(z[i, j])
        p.signals["off_domain"] = not in_domain(p)
    kept_scores = [p.emb_score for p in papers if not p.signals.get("off_domain")]
    if kept_scores:
        lo, hi = float(np.percentile(kept_scores, 5)), float(max(kept_scores))
    else:
        lo, hi = 0.0, 1.0
    for p in papers:
        if p.signals.get("off_domain"):
            p.signals["emb_norm"] = 0.0
        else:
            p.signals["emb_norm"] = float(np.clip((p.emb_score - lo) / (hi - lo + 1e-9), 0, 1))
    zs = np.array([-1e9 if p.signals.get("off_domain") else p.emb_score for p in papers], dtype=float)
    idx = select_indices(zs, int(top_k), float(min_emb), int(min_keep))
    idx = apply_direction_quota(
        papers, zs, idx, per_direction=int(direction_keep),
        floor=float(direction_min_emb), cap=max(int(top_k), int(direction_keep)),
    )
    prefilter.backend = emb.backend
    log.info("embedding backend=%s; prefilter %d -> %d (floor %.2f, keep≥%d, off-domain %d)",
             emb.backend, len(papers), len(idx), float(min_emb), int(min_keep),
             sum(1 for p in papers if p.signals.get("off_domain")))
    if not idx:
        return [], None
    return [papers[i] for i in idx], P[idx]


def final_scores(papers, wcfg: dict) -> None:
    from .signals import quality_score
    w = wcfg or {}
    for p in papers:
        q = quality_score(p)
        e = p.signals.get("emb_norm", 0)
        L = p.llm or {}
        if L and not L.get("heuristic"):
            p.final_score = (w.get("relevance", 0.4) * L["relevance"] / 10
                             + w.get("novelty", 0.15) * L["novelty"] / 10
                             + w.get("rigor", 0.15) * L["rigor"] / 10
                             + w.get("embedding", 0.15) * e
                             + w.get("quality", 0.15) * q)
            if L["relevance"] <= 3:
                p.final_score *= 0.5
        else:
            p.final_score = 0.55 * e + 0.25 * p.signals.get("rigor", 0) + 0.20 * q
        p.signals["quality"] = round(q, 3)


def _mmr(papers, vecs, n: int, lam: float, per_topic_max: int) -> list[int]:
    scores = np.array([p.final_score for p in papers], dtype=float)
    sims = vecs @ vecs.T if vecs is not None else np.zeros((len(papers), len(papers)))
    chosen: list[int] = []
    cand = list(range(len(papers)))
    cnt: dict[str, int] = {}
    while cand and len(chosen) < n:
        best, bval = None, -1e9
        for i in cand:
            if papers[i].signals.get("off_domain"):
                continue
            if cnt.get(papers[i].best_topic, 0) >= per_topic_max:
                continue
            red = max((float(sims[i, j]) for j in chosen), default=0.0)
            v = lam * float(scores[i]) - (1 - lam) * red
            if v > bval:
                best, bval = i, v
        if best is None:
            break
        chosen.append(best)
        cand.remove(best)
        cnt[papers[best].best_topic] = cnt.get(papers[best].best_topic, 0) + 1
    return chosen


def _direction(p) -> str:
    return p.signals.get("direction") or p.best_topic or ""


def _cover_directions(papers, chosen: list[int], swaps: int) -> list[int]:
    """If a whole research direction was crowded out, trade the weakest extra paper for it."""
    if swaps <= 0:
        return chosen
    chosen = list(chosen)
    present = {_direction(papers[i]) for i in chosen}
    missing = []
    for i, p in enumerate(papers):
        d = _direction(p)
        if d and d not in present and d not in missing:
            missing.append(d)
    for d in missing[:swaps]:
        pool = [i for i in range(len(papers))
                if i not in chosen and not papers[i].signals.get("off_domain") and _direction(papers[i]) == d]
        if not pool:
            continue
        pool.sort(key=lambda i: (-papers[i].final_score, -papers[i].emb_score))
        counts: dict[str, int] = {}
        for i in chosen:
            counts[_direction(papers[i])] = counts.get(_direction(papers[i]), 0) + 1
        donors = [i for i in chosen if counts.get(_direction(papers[i]), 0) >= 2]
        if not donors:
            break
        donor = min(donors, key=lambda i: (papers[i].final_score, papers[i].emb_score))
        chosen.remove(donor)
        papers[pool[0]].signals["explore"] = True
        chosen.append(pool[0])
        present.add(d)
    return chosen


def ensure_direction_minimum(papers, chosen: list[int], *, mins: dict | None, default_min: int,
                             n: int) -> list[int]:
    """Swap surplus papers so each direction with candidates keeps its minimum.

    Off-domain papers are never used to fill a slot. A direction with no
    surviving candidate stays empty. The list never grows past `n`.
    """
    chosen = list(chosen)
    mins = {str(k): int(v) for k, v in (mins or {}).items()}

    def eligible(i: int) -> bool:
        paper = papers[i]
        if paper.signals.get("off_domain") or not _direction(paper):
            return False
        if _direction(paper) == "运动控制" and not has_control_content(paper.text):
            return False
        return True

    def count(d: str) -> int:
        return sum(1 for i in chosen if eligible(i) and _direction(papers[i]) == d)

    dirs: list[str] = []
    for i, p in enumerate(papers):
        d = _direction(p)
        if eligible(i) and d not in dirs:
            dirs.append(d)
    targets = {d: max(0, int(mins.get(d, default_min))) for d in dirs}
    for d, want in targets.items():
        guard = 0
        while count(d) < want and guard < len(papers) + 2:
            guard += 1
            pool = [i for i in range(len(papers))
                    if i not in chosen and eligible(i) and _direction(papers[i]) == d]
            if not pool:
                break
            pool.sort(key=lambda i: (-papers[i].final_score, -papers[i].emb_score))
            pick = pool[0]
            if len(chosen) < n:
                papers[pick].signals["explore"] = True
                chosen.append(pick)
                continue
            donors = [i for i in chosen
                      if (not eligible(i)) or count(_direction(papers[i])) > targets.get(_direction(papers[i]), 0)]
            if not donors:
                break
            donor = min(donors, key=lambda i: (papers[i].final_score, papers[i].emb_score))
            chosen.remove(donor)
            papers[pick].signals["explore"] = True
            chosen.append(pick)
    return chosen[:n]


def lock_control_topics(papers, topics) -> int:
    """Drop planning/prediction papers that the LLM or the embedder parked on 运动控制."""
    moved = 0
    for paper in papers:
        topic = topics.get(paper.best_topic) if isinstance(topics, dict) else None
        direction = getattr(topic, "direction", "") or paper.signals.get("direction")
        if direction != "运动控制" or has_control_content(paper.text):
            if topic is not None:
                paper.signals["direction"] = topic.direction
            continue
        fallback = paper.signals.get("prefilter_topic") or ""
        other = topics.get(fallback) if fallback else None
        if other is None or getattr(other, "direction", "") == "运动控制":
            best_id, best_score = "", -1e9
            for key, score in (paper.emb_scores or {}).items():
                candidate = topics.get(key)
                if candidate is None or getattr(candidate, "direction", "") == "运动控制":
                    continue
                if float(score) > best_score:
                    best_id, best_score = key, float(score)
            other = topics.get(best_id) if best_id else None
        if other is not None:
            paper.best_topic = other.id
            paper.signals["direction"] = other.direction
        else:
            paper.signals["direction"] = ""
            paper.signals["control_rejected"] = True
        moved += 1
    return moved


def mmr_select(papers, vecs, n: int, lam: float = 0.7, per_topic_max: int = 4, exploration: int = 0,
               direction_mins: dict | None = None, per_direction_min: int = 0):
    """MMR under a per-topic cap, then slots so each direction keeps a minimum."""
    if not papers or n <= 0:
        return []
    explore_n = max(0, min(int(exploration), n - 1)) if n > 1 else 0
    chosen = _mmr(papers, vecs, n - explore_n, lam, per_topic_max)
    chosen = _cover_directions(papers, chosen, explore_n)
    if len(chosen) < n:
        counts: dict[str, int] = {}
        dir_counts: dict[str, int] = {}
        for i in chosen:
            counts[papers[i].best_topic] = counts.get(papers[i].best_topic, 0) + 1
            dir_counts[_direction(papers[i])] = dir_counts.get(_direction(papers[i]), 0) + 1
        pool = [i for i in range(len(papers)) if i not in set(chosen) and not papers[i].signals.get("off_domain")]
        pool.sort(key=lambda i: (
            dir_counts.get(_direction(papers[i]), 0),
            counts.get(papers[i].best_topic, 0),
            -papers[i].emb_score,
            -papers[i].final_score,
        ))
        for i in pool[:n - len(chosen)]:
            papers[i].signals["explore"] = True
            chosen.append(i)
    if direction_mins or per_direction_min:
        chosen = ensure_direction_minimum(
            papers, chosen, mins=direction_mins, default_min=int(per_direction_min), n=n,
        )
    return [papers[i] for i in chosen[:n]]
