from __future__ import annotations

import json
import logging
from collections import OrderedDict

from .dedup import dedup
from .interests import build_topics
from .interests.seeds import enrich_seed_texts
from .pdf import resolve_pdfs
from .push import feishu
from .push import feishu_files
from .push import report
from .rank import signals as sig
from .rank.llm import LLMReviewer, fill_heuristic_text
from .rank.score import final_scores, lock_control_topics, mmr_select, prefilter
from .s2 import get_client
from .sources import REGISTRY
from .store import Store
from .util import shanghai_today
from .validate import ConfigError
from .weekly import maybe_weekly

log = logging.getLogger("paper_radar")


def run(cfg: dict, dry_run: bool = False, no_mark: bool = False) -> dict:
    warnings = cfg.get("_warnings") or []
    for w in warnings:
        log.warning("config: %s", w)
    if not cfg.get("topics"):
        raise ConfigError("config.topics 为空")

    store = Store(cfg.get("state_db", "data/state.sqlite"))
    topics = build_topics(cfg, store)
    if not topics:
        raise ConfigError("没有可用的兴趣子方向")
    s2cfg = (cfg.get("sources") or {}).get("semantic_scholar") or {}
    s2key = s2cfg.get("api_key", "")
    s2_client = get_client(
        s2key,
        min_interval=float(s2cfg.get("min_interval", 1.1)),
        max_requests=int(s2cfg.get("max_requests", 300)),
        batch_size=int(s2cfg.get("batch_size", 500)),
    )
    n_seed = enrich_seed_texts(topics, store, s2key)
    log.info("topics=%d, seed abstracts=%d", len(topics), n_seed)

    raw, per_source = [], OrderedDict()
    for name, scfg in (cfg.get("sources") or {}).items():
        if not isinstance(scfg, dict) or not scfg.get("enabled", False) or name not in REGISTRY:
            continue
        try:
            got = REGISTRY[name]({**scfg}, cfg).fetch(topics)
        except Exception as e:
            log.exception("source %s crashed: %s", name, e)
            got = []
        per_source[name] = len(got or [])
        log.info("source %-16s %4d papers", name, len(got or []))
        raw += got or []

    cands = dedup(raw, store.seen_keys(), store.seen_titles(), float(cfg.get("dedup_fuzzy", 0.93)))
    min_abs = int(cfg.get("min_abstract_chars", 80))
    cands = [p for p in cands if len(p.abstract or "") >= min_abs]
    log.info("after dedup/history/abstract filter: %d", len(cands))

    rk = cfg.get("ranking") or {}
    kept, vecs = prefilter(
        cands, topics, cfg.get("embedding") or {}, int(rk.get("top_k", 40)), store=store,
        min_emb=float(rk.get("min_emb", 0.35)), min_keep=int(rk.get("min_keep", 8)),
        direction_keep=int(rk.get("direction_keep", 6)),
        direction_min_emb=float(rk.get("direction_min_emb", 0.22)),
    )
    n_pre = len(kept)
    embedder_name = getattr(prefilter, "backend", "tfidf")

    sig.compute_basic(kept)
    q = cfg.get("quality") or {}
    if q.get("hf_upvotes", True) and kept:
        try:
            log.info("HF upvotes matched: %d", sig.add_hf_upvotes(kept))
        except Exception as e:
            log.warning("HF upvotes skipped: %s", e)
    if q.get("author_hindex", True) and kept:
        try:
            log.info("S2 h-index matched: %d", sig.add_author_hindex(kept, s2key, store))
        except Exception as e:
            log.warning("S2 h-index skipped: %s", e)
    if q.get("openreview", True) and kept:
        try:
            log.info("OpenReview ratings: %d", sig.add_openreview(kept, store, int(q.get("openreview_limit", 8))))
        except Exception as e:
            log.warning("OpenReview skipped: %s", e)
    # Recompute venue/institution after S2 may have filled venue, and pick up code links.
    sig.compute_basic(kept)

    reviewer = LLMReviewer(cfg.get("llm") or {}, topics)
    n_llm = reviewer.review(kept)
    final_scores(kept, rk.get("weights") or {})
    # Lexical gate first: a high LLM score cannot rescue a warehouse robot or a manipulator.
    # min_keep already refused to backfill those papers, this is the second check.
    if n_llm:
        floor = float(rk.get("min_relevance", 5))
        keep_idx = [
            i for i, p in enumerate(kept)
            if not p.signals.get("off_domain")
            and (not p.llm or p.llm.get("heuristic") or p.llm.get("relevance", 0) >= floor)
        ]
        kept = [kept[i] for i in keep_idx]
        vecs = vecs[keep_idx] if vecs is not None and keep_idx else None
    else:
        thr = float(rk.get("min_heuristic", 0.0))
        keep_idx = [i for i, p in enumerate(kept) if not p.signals.get("off_domain") and p.final_score >= thr]
        kept = [kept[i] for i in keep_idx]
        vecs = vecs[keep_idx] if vecs is not None and keep_idx else None
    lock_control_topics(kept, topics)
    selected = mmr_select(
        kept, vecs, int(rk.get("top_n", 12)), float(rk.get("mmr_lambda", 0.7)),
        int(rk.get("per_topic_max", 3)), int(rk.get("exploration_slots", 1)),
        direction_mins=rk.get("direction_mins") or {"运动控制": 2},
        per_direction_min=int(rk.get("per_direction_min", 1)),
    )
    fill_heuristic_text(selected, topics)
    try:
        resolve_pdfs(selected, cfg, store)
    except Exception as e:
        log.warning("PDF resolve skipped: %s", e)

    dirs = list(dict.fromkeys(t.direction for t in topics.values()))
    grouped = OrderedDict((d, []) for d in dirs)
    for p in sorted(selected, key=lambda x: -x.final_score):
        if p.signals.get("control_rejected"):
            direction = "其他"
        else:
            direction = topics[p.best_topic].direction if p.best_topic in topics else "其他"
        grouped.setdefault(direction, []).append(p)

    today = shanghai_today()
    stats = {
        "per_source": dict(per_source), "raw": len(raw), "candidates": len(cands),
        "prefiltered": n_pre, "selected": len(selected),
        "with_pdf": sum(1 for p in selected if p.pdf_link),
        "llm_reviewed": n_llm, "embedder": embedder_name,
        "s2_requests": s2_client.sent,
        "scorer": f"LLM({cfg.get('llm', {}).get('model', '')})" if n_llm else "启发式(无LLM key)",
    }
    mdp, _hp = report.write_report(grouped, topics, today, stats, cfg.get("report_dir", "reports"))
    stats["report"] = str(mdp)

    fs = cfg.get("feishu") or {}
    cards = feishu.build_cards(
        grouped, topics, today, stats, max_bytes=int(fs.get("max_card_bytes", 18000)),
    )
    if dry_run or not fs.get("webhook"):
        out = f"{cfg.get('report_dir', 'reports')}/{today}.feishu.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(cards, f, ensure_ascii=False, indent=1)
        log.info("dry-run / no webhook: Feishu cards written to %s", out)
        stats["pushed_cards"] = 0
    else:
        stats["pushed_cards"] = feishu.send(cards, fs["webhook"], fs.get("secret", ""))

    # File messages go through the self-built app, never the custom-bot webhook.
    # dry-run must not download or upload.
    try:
        stats["pdf_files"] = feishu_files.deliver(
            sorted(selected, key=lambda x: -x.final_score), fs, cfg.get("pdf") or {}, dry_run=dry_run,
        )
    except Exception as e:
        log.warning("Feishu PDF delivery skipped: %s", e)
        stats["pdf_files"] = {"enabled": feishu_files.enabled(fs), "sent": 0, "error": str(e)}

    if selected and not dry_run and not no_mark:
        store.mark_pushed(selected)
    try:
        weekly = maybe_weekly(cfg, store, topics, dry_run=dry_run)
        if weekly:
            stats["weekly"] = weekly
    except Exception as e:
        log.warning("weekly digest skipped: %s", e)
    store.log_run({k: v for k, v in stats.items() if k != "weekly"})
    return {"stats": stats, "grouped": grouped, "topics": topics}
