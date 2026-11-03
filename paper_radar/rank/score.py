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
              min_emb: float = 0.0, min_keep: int = 8):
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
    best = z.argmax(axis=1)
    from .signals import in_domain
    for i, p in enumerate(papers):
        p.emb_scores = {t.id: round(float(sim[i, j]), 4) for j, t in enumerate(tlist)}
        p.best_topic = tlist[int(best[i])].id
        p.signals["direction"] = tlist[int(best[i])].direction
        p.emb_score = float(z[i, best[i]])
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


def mmr_select(papers, vecs, n: int, lam: float = 0.7, per_topic_max: int = 4, exploration: int = 0):
    """MMR under a per-topic cap, then a few slots for a missing direction or topic."""
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
    return [papers[i] for i in chosen]
