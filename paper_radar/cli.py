from __future__ import annotations

import argparse
import json
import logging
import re
import sys

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


def main(argv=None):
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
