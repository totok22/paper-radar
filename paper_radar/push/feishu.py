"""Feishu/Lark custom-bot webhook with signature, interactive card messages.

Signature (official): sign = base64(hmac_sha256(key=f"{timestamp}\\n{secret}", msg=b"")).
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import time

from ..http import request

log = logging.getLogger("paper_radar")
COLORS = {"运动规划": "blue", "运动控制": "green", "场景生成": "orange"}


def sign(secret: str, ts: int) -> str:
    key = f"{ts}\n{secret}".encode()
    return base64.b64encode(hmac.new(key, b"", hashlib.sha256).digest()).decode()


def paper_md(i: int, p, topics) -> str:
    t = topics.get(p.best_topic)
    L = p.llm
    tldr = L.get("tldr") or (p.abstract[:150] + "…")
    badges = []
    if p.signals.get("venue_name"):
        badges.append(f"🏆{p.signals['venue_name']}")
    if p.signals.get("rigor_hits"):
        badges.append("🧪" + "/".join(p.signals["rigor_hits"][:3]))
    if p.signals.get("hf_upvotes"):
        badges.append(f"👍HF {p.signals['hf_upvotes']}")
    if p.signals.get("max_hindex"):
        badges.append(f"h{p.signals['max_hindex']}")
    score = f"{p.final_score * 10:.1f}"
    sub = f"**分数 {score}**" + (f" · 相关{L['relevance']:.0f}/新颖{L['novelty']:.0f}/实验{L['rigor']:.0f}" if L else "")
    lines = [f"**{i}. [{p.title}]({p.link})**",
             f"<font color='grey'>{t.name if t else p.best_topic} · {p.published} · {', '.join(p.authors[:3])}{' 等' if len(p.authors) > 3 else ''}</font>",
             f"{sub}  {' '.join(badges)}",
             f"📝 {tldr}"]
    if L.get("reason"):
        lines.append(f"💡 {L['reason']}")
    lines.append(f"<font color='grey'>反馈: python -m paper_radar feedback like {p.key}</font>")
    return "\n".join(lines)


def build_cards(grouped: dict, topics, date_str: str, stats: dict) -> list[dict]:
    cards, idx = [], 1
    for direction, papers in grouped.items():
        if not papers:
            continue
        els = []
        for p in papers:
            els.append({"tag": "markdown", "content": paper_md(idx, p, topics)})
            els.append({"tag": "hr"})
            idx += 1
        els.append({"tag": "note", "elements": [{"tag": "plain_text", "content":
                    f"候选 {stats.get('candidates', 0)} 篇 → 预筛 {stats.get('prefiltered', 0)} → 推送 {stats.get('selected', 0)} · 评分: {stats.get('scorer', '')}"}]})
        cards.append({"msg_type": "interactive", "card": {
            "config": {"wide_screen_mode": True},
            "header": {"template": COLORS.get(direction, "purple"),
                       "title": {"tag": "plain_text", "content": f"📡 Paper Radar {date_str} · {direction}（{len(papers)}）"}},
            "elements": els}})
    return cards


def send(cards: list[dict], webhook: str, secret: str = "") -> int:
    ok = 0
    for c in cards:
        body = dict(c)
        if secret:
            ts = int(time.time())
            body.update({"timestamp": str(ts), "sign": sign(secret, ts)})
        r = request("POST", webhook, json=body, retries=2)
        js = r.json() if r is not None and r.headers.get("content-type", "").startswith("application/json") else {}
        if r is not None and r.status_code == 200 and js.get("code", js.get("StatusCode", 0)) == 0:
            ok += 1
        else:
            log.error("Feishu push failed: %s", js or (None if r is None else r.text[:200]))
        time.sleep(0.5)  # custom bot rate limit: 5/s, 100/min
    return ok
