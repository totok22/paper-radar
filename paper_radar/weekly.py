"""A short Monday digest of what was already pushed this week."""
from __future__ import annotations

import json
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .push import feishu
from .util import shanghai_today


def _direction(row, topics) -> str:
    topic = topics.get(row["topic"]) if topics else None
    return topic.direction if topic else "其他"


def collect(store, topics, days: int = 7) -> "OrderedDict[str, list]":
    grouped: OrderedDict[str, list] = OrderedDict()
    if topics:
        for t in topics.values():
            grouped.setdefault(t.direction, [])
    for row in store.pushed_since(days):
        direction = _direction(row, topics)
        grouped.setdefault(direction, []).append(row)
    return grouped


def render_md(grouped, date_str: str, days: int) -> str:
    total = sum(len(v) for v in grouped.values())
    lines = [f"# 📡 Paper Radar 周报 {date_str}", "",
             f"> 近 {days} 天已推送 {total} 篇（按综合分排序）。", ""]
    for direction, rows in grouped.items():
        if not rows:
            continue
        lines.append(f"## {direction}（{len(rows)}）\n")
        for i, row in enumerate(rows, 1):
            lines.append(f"{i}. [{row['title']}]({row['link']}) · 分数 {row['score'] * 10:.1f} · `{row['key']}`")
        lines.append("")
    if total == 0:
        lines.append("这一周还没有推送记录。")
    return "\n".join(lines)


def build_card(grouped, date_str: str, days: int) -> dict | None:
    blocks = []
    total = 0
    for direction, rows in grouped.items():
        if not rows:
            continue
        total += len(rows)
        bits = [f"**{direction}（{len(rows)}）**"]
        for row in rows[:8]:
            bits.append(f"- [{row['title']}]({row['link']}) · {row['score'] * 10:.1f}")
        if len(rows) > 8:
            bits.append(f"- …另有 {len(rows) - 8} 篇，见周报文件")
        blocks.append("\n".join(bits))
    if total == 0:
        return None
    return {"msg_type": "interactive", "card": {
        "config": {"wide_screen_mode": True},
        "header": {"template": "purple",
                   "title": {"tag": "plain_text", "content": f"📚 Paper Radar 周报 {date_str}（{total}）"}},
        "elements": [
            {"tag": "markdown", "content": f"近 {days} 天推送，按方向汇总。\n\n" + "\n\n".join(blocks)},
        ],
    }}


def maybe_weekly(cfg: dict, store, topics, dry_run: bool) -> dict | None:
    wcfg = cfg.get("weekly") or {}
    weekday = int(wcfg.get("weekday", 0))
    if weekday < 0:
        return None
    today = datetime.now(ZoneInfo("Asia/Shanghai"))
    if today.weekday() != weekday:
        return None
    days = int(wcfg.get("days", 7))
    grouped = collect(store, topics, days)
    date_str = shanghai_today()
    text = render_md(grouped, date_str, days)
    outdir = Path(cfg.get("report_dir", "reports"))
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"weekly-{date_str}.md"
    path.write_text(text, encoding="utf-8")
    card = build_card(grouped, date_str, days)
    stats = {"report": str(path), "papers": sum(len(v) for v in grouped.values()), "pushed": 0}
    fs = cfg.get("feishu") or {}
    if card and not dry_run and fs.get("webhook"):
        stats["pushed"] = feishu.send([card], fs["webhook"], fs.get("secret", ""))
    elif card:
        side = outdir / f"weekly-{date_str}.feishu.json"
        side.write_text(json.dumps(card, ensure_ascii=False, indent=1), encoding="utf-8")
        stats["card"] = str(side)
    return stats
