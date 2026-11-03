from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

from .config import load_config
from .store import Store


def _feedback(cfg, args):
    store = Store(cfg.get("state_db", "data/state.sqlite"))
    label = 1 if args.action == "like" else -1
    rows = store.find_pushed(args.target)
    if len(rows) == 1 or (rows and args.target.startswith(("arxiv:", "doi:"))):
        r = rows[0]
        store.add_feedback(r["key"], label, args.topic or r["topic"], r["title"], r["s2_ref"])
        print(f"{'👍' if label > 0 else '👎'} {r['title']}  (topic={args.topic or r['topic']})")
    elif len(rows) > 1:
        print("匹配到多篇，请用更精确的 key：")
        for r in rows[:10]:
            print(f"  {r['key']}\t{r['title'][:80]}")
        sys.exit(1)
    else:  # not pushed before: accept raw arXiv ID / DOI
        m = re.search(r"(\d{4}\.\d{4,5})", args.target)
        if m:
            key, ref = f"arxiv:{m.group(1)}", f"ARXIV:{m.group(1)}"
        elif args.target.lower().startswith(("doi:", "10.")):
            d = args.target.split(":", 1)[-1] if args.target.lower().startswith("doi:") else args.target
            key, ref = f"doi:{d.lower()}", f"DOI:{d}"
        else:
            print("未找到该论文；请给出 arXiv ID / DOI / 已推送论文的 key")
            sys.exit(1)
        store.add_feedback(key, label, args.topic or "", args.title or key, ref)
        print(f"{'👍' if label > 0 else '👎'} {key} (topic={args.topic or '全部子方向'})")


def _normalize_argv(argv):
    """Accept `-c` / `-v` either before or after the subcommand."""
    if not argv:
        return argv
    moved, rest = [], []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("-c", "--config") and i + 1 < len(argv):
            moved.extend((a, argv[i + 1]))
            i += 2
            continue
        if a.startswith("--config="):
            moved.append(a)
            i += 1
            continue
        if a in ("-v", "--verbose"):
            moved.append(a)
            i += 1
            continue
        rest.append(a)
        i += 1
    return moved + rest


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    argv = _normalize_argv(list(argv))
    ap = argparse.ArgumentParser(prog="paper_radar", description="每日高质量论文雷达 + 飞书推送")
    ap.add_argument("-c", "--config", default="config.yaml")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="运行一次完整流程")
    r.add_argument("--dry-run", action="store_true", help="不推送飞书、不写入已推送历史")
    r.add_argument("--no-mark", action="store_true", help="推送但不记录历史（调试用）")
    r.add_argument("--days", type=int, help="覆盖 lookback_days")
    f = sub.add_parser("feedback", help="标记喜欢/不喜欢，下次运行喂给 S2 推荐和兴趣画像")
    f.add_argument("action", choices=["like", "dislike"])
    f.add_argument("target", help="paper key（如 arxiv:2410.23262）、arXiv ID、DOI 或标题片段")
    f.add_argument("--topic", help="归入哪个子方向 id（默认沿用推送时的子方向）")
    f.add_argument("--title", help="未推送过的论文可附标题")
    sub.add_parser("recent", help="查看最近推送")
    sub.add_parser("topics", help="查看子方向配置")
    sub.add_parser("validate", help="检查 config.yaml")
    w = sub.add_parser("weekly", help="根据已推送历史生成周报")
    w.add_argument("--dry-run", action="store_true", help="只写报告，不推送飞书")
    w.add_argument("--days", type=int, default=None, help="覆盖 weekly.days")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    cfg = load_config(args.config)
    if args.cmd == "run":
        if args.days:
            cfg["lookback_days"] = args.days
        from .pipeline import run
        res = run(cfg, dry_run=args.dry_run, no_mark=args.no_mark)
        print(json.dumps(res["stats"], ensure_ascii=False, indent=1))
    elif args.cmd == "feedback":
        _feedback(cfg, args)
    elif args.cmd == "recent":
        for row in Store(cfg.get("state_db", "data/state.sqlite")).recent_pushed():
            print(f"{row['pushed_at'][:10]}  {row['score'] * 10:4.1f}  {row['key']:<24} {row['title'][:70]}")
    elif args.cmd == "topics":
        from .interests import build_topics
        for t in build_topics(cfg, Store(cfg.get("state_db", "data/state.sqlite"))).values():
            print(f"[{t.direction}] {t.id}: {t.name}  (+{len(t.positive_ids)} / -{len(t.negative_ids)} seeds)")
    elif args.cmd == "validate":
        for wmsg in cfg.get("_warnings") or []:
            print(f"警告: {wmsg}")
        print(f"配置可用：{len(cfg.get('topics') or [])} 个子方向，LLM 模型 {cfg['llm']['model']}")
    elif args.cmd == "weekly":
        from .interests import build_topics
        from .weekly import maybe_weekly
        store = Store(cfg.get("state_db", "data/state.sqlite"))
        if args.days:
            cfg.setdefault("weekly", {})["days"] = args.days
        # The subcommand always renders, regardless of which weekday it is.
        cfg.setdefault("weekly", {})["weekday"] = datetime.now(ZoneInfo("Asia/Shanghai")).weekday()
        stats = maybe_weekly(cfg, store, build_topics(cfg, store), dry_run=args.dry_run or not cfg.get("feishu", {}).get("webhook"))
        print(json.dumps(stats or {"papers": 0}, ensure_ascii=False, indent=1))
