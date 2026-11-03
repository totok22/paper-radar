"""Per-topic MAX embedding similarity pre-filter, final scoring, MMR diversity."""
from __future__ import annotations

import logging

import numpy as np

from .embed import Embedder
from .signals import quality_score

log = logging.getLogger("paper_radar")


def prefilter(papers, topics, ecfg: dict, top_k: int):
    """Score each paper against every topic; keep MAX (not mean). Returns (kept, vectors)."""
    if not papers:
        return [], None
    tlist = list(topics.values())
    emb = Embedder(ecfg)
    neg_txt = [" ".join(t.negative_texts) for t in tlist]
    texts = [t.profile_text for t in tlist] + [n or "__none__" for n in neg_txt] + [p.text for p in papers]
    X = emb.encode(texts)
    T, N, P = X[:len(tlist)], X[len(tlist):2 * len(tlist)], X[2 * len(tlist):]
    sim = P @ T.T  # (papers, topics)
    nsim = P @ N.T
    has_neg = np.array([bool(n) for n in neg_txt], dtype=float)
    sim = sim - float(ecfg.get("negative_weight", 0.5)) * nsim * has_neg
    # scale-normalize per topic (divide by its 95th percentile) so topics with long/short
    # profiles are comparable, then take the MAX over topics (not the mean)
    z = sim / (np.percentile(sim, 95, axis=0) + 1e-9)
    best = z.argmax(1)
    for i, p in enumerate(papers):
        p.emb_scores = {t.id: round(float(sim[i, j]), 4) for j, t in enumerate(tlist)}
        p.best_topic = tlist[best[i]].id
        p.emb_score = float(z[i, best[i]])
    # map z to 0..1 for later combination
    zs = np.array([p.emb_score for p in papers])
    lo, hi = np.percentile(zs, 5), zs.max()
    for p in papers:
        p.signals["emb_norm"] = float(np.clip((p.emb_score - lo) / (hi - lo + 1e-9), 0, 1))
    order = np.argsort(-zs)[:top_k]
    log.info("embedding backend=%s; prefilter %d -> %d", emb.backend, len(papers), len(order))
    return [papers[i] for i in order], P[order]


def final_scores(papers, wcfg: dict) -> None:
    for p in papers:
        q = quality_score(p)
        e = p.signals.get("emb_norm", 0)
        if p.llm:
            L = p.llm
            p.final_score = (wcfg.get("relevance", 0.4) * L["relevance"] / 10
                             + wcfg.get("novelty", 0.15) * L["novelty"] / 10
                             + wcfg.get("rigor", 0.15) * L["rigor"] / 10
                             + wcfg.get("embedding", 0.15) * e
                             + wcfg.get("quality", 0.15) * q)
            if L["relevance"] <= 3:
                p.final_score *= 0.5
        else:  # heuristic fallback
            p.final_score = 0.6 * e + 0.2 * p.signals.get("rigor", 0) + 0.2 * q
        p.signals["quality"] = round(q, 3)


def mmr_select(papers, vecs, n: int, lam: float = 0.7, per_topic_max: int = 4):
    if not papers:
        return []
    scores = np.array([p.final_score for p in papers])
    sims = vecs @ vecs.T if vecs is not None else np.zeros((len(papers), len(papers)))
    chosen, cand, cnt = [], list(range(len(papers))), {}
    while cand and len(chosen) < n:
        best, bval = None, -1e9
        for i in cand:
            if cnt.get(papers[i].best_topic, 0) >= per_topic_max:
                continue
            red = max((sims[i, j] for j in chosen), default=0.0)
            v = lam * scores[i] - (1 - lam) * red
            if v > bval:
                best, bval = i, v
        if best is None:
            break
        chosen.append(best)
        cand.remove(best)
        cnt[papers[best].best_topic] = cnt.get(papers[best].best_topic, 0) + 1
    return [papers[i] for i in chosen]
