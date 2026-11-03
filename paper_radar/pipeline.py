from __future__ import annotations

import json
import logging
from collections import OrderedDict
from datetime import date

from .dedup import dedup
from .interests import build_topics
from .interests.seeds import enrich_seed_texts
from .push import feishu, report
from .rank import signals as sig
from .rank.llm import LLMReviewer
from .rank.score import final_scores, mmr_select, prefilter
from .sources import REGISTRY
from .store import Store

log = logging.getLogger("paper_radar")


def run(cfg: dict, dry_run: bool = False, no_mark: bool = False) -> dict:
    store = Store(cfg.get("state_db", "data/state.sqlite"))
    topics = build_topics(cfg, store)
    s2key = cfg["sources"].get("semantic_scholar", {}).get("api_key", "")
    n_seed = enrich_seed_texts(topics, store, s2key)
    log.info("topics=%d, seed abstracts=%d", len(topics), n_seed)

    # 1) multi-source recall
    raw, per_source = [], OrderedDict()
    for name, scfg in cfg["sources"].items():
        if not scfg.get("enabled", False) or name not in REGISTRY:
            continue
        try:
            got = REGISTRY[name]({**scfg}, cfg).fetch(topics)
        except Exception as e:
            log.exception("source %s crashed: %s", name, e)
            got = []
        per_source[name] = len(got)
        log.info("source %-16s %4d papers", name, len(got))
        raw += got

    # 2) normalize + dedup + history
    cands = dedup(raw, store.seen_keys(), store.seen_titles(), float(cfg.get("dedup_fuzzy", 0.93)))
    cands = [p for p in cands if len(p.abstract) > 200]  # need an abstract to judge
    log.info("after dedup/history/abstract filter: %d", len(cands))

    # 3) embedding pre-filter (MAX over topics)
    rk = cfg.get("ranking", {})
    kept, vecs = prefilter(cands, topics, cfg.get("embedding", {}), int(rk.get("top_k", 40)))
    n_pre = len(kept)
    from .rank.embed import Embedder
    embedder_name = Embedder(cfg.get("embedding", {})).backend

    # 4) quality signals + LLM review on top-K only
    sig.compute_basic(kept)
    q = cfg.get("quality", {})
    if q.get("hf_upvotes", True):
        log.info("HF upvotes matched: %d", sig.add_hf_upvotes(kept))
    if q.get("author_hindex", True):
        log.info("S2 h-index matched: %d", sig.add_author_hindex(kept, s2key, store))
    reviewer = LLMReviewer(cfg.get("llm", {}), topics)
    n_llm = reviewer.review(kept)
    final_scores(kept, rk.get("weights", {}))
    if n_llm:
        keep_idx = [i for i, p in enumerate(kept) if not p.llm or p.llm["relevance"] >= rk.get("min_relevance", 5)]
        kept = [kept[i] for i in keep_idx]
        vecs = vecs[keep_idx] if vecs is not None and keep_idx else vecs
    else:
        thr = float(rk.get("min_heuristic", 0.0))
        keep_idx = [i for i, p in enumerate(kept) if p.final_score >= thr]
        kept, vecs = [kept[i] for i in keep_idx], (vecs[keep_idx] if keep_idx else None)
    selected = mmr_select(kept, vecs, int(rk.get("top_n", 12)), float(rk.get("mmr_lambda", 0.7)),
                          int(rk.get("per_topic_max", 3)))

    # group by direction, keep config order
    dirs = list(dict.fromkeys(t.direction for t in topics.values()))
    grouped = OrderedDict((d, []) for d in dirs)
    for p in sorted(selected, key=lambda x: -x.final_score):
        grouped.setdefault(topics[p.best_topic].direction if p.best_topic in topics else "其他", []).append(p)

    stats = {"per_source": dict(per_source), "raw": len(raw), "candidates": len(cands),
             "prefiltered": n_pre, "selected": len(selected),
             "llm_reviewed": n_llm, "embedder": embedder_name,
             "scorer": f"LLM({cfg['llm']['model']})" if n_llm else "启发式(无LLM key)"}
    today = date.today().isoformat()
    mdp, hp = report.write_report(grouped, topics, today, stats, cfg.get("report_dir", "reports"))
    stats["report"] = str(mdp)

    # 5) push
    cards = feishu.build_cards(grouped, topics, today, stats)
    fs = cfg.get("feishu", {})
    if dry_run or not fs.get("webhook"):
        out = f"{cfg.get('report_dir', 'reports')}/{today}.feishu.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(cards, f, ensure_ascii=False, indent=1)
        log.info("dry-run / no webhook: Feishu cards written to %s", out)
        stats["pushed_cards"] = 0
    else:
        stats["pushed_cards"] = feishu.send(cards, fs["webhook"], fs.get("secret", ""))
    if selected and not dry_run and not no_mark:
        store.mark_pushed(selected)
    store.log_run(stats)
    return {"stats": stats, "grouped": grouped, "topics": topics}
