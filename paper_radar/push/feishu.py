"""Feishu/Lark custom-bot webhook: signed interactive cards.

Signature (official): sign = base64(hmac_sha256(key=f"{timestamp}\\n{secret}", msg=b"")).

Cards are grouped by research direction. Each paper has a score line, a
Chinese TL;DR, a why-this-paper line, and buttons for the direct PDF plus
arXiv / DOI / code when those exist. A card is split before it nears the
custom-bot 20KB body limit.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import time

from ..http import request

log = logging.getLogger("paper_radar")
COLORS = {"运动规划": "blue", "运动控制": "green", "场景生成": "orange"}


def sign(secret: str, ts: int) -> str:
    key = f"{ts}\n{secret}".encode()
    return base64.b64encode(hmac.new(key, b"", hashlib.sha256).digest()).decode()


def _buttons(p) -> list[dict]:
    links = []
    if p.pdf_link:
        links.append(("PDF", p.pdf_link, "primary"))
    if p.arxiv_id:
        links.append(("arXiv", f"https://arxiv.org/abs/{p.arxiv_id}", "default"))
    if p.doi:
        links.append(("DOI", f"https://doi.org/{p.doi}", "default"))
    if p.code_url:
        links.append(("代码", p.code_url, "default"))
    actions = []
    for label, url, kind in links:
        actions.append({
            "tag": "button",
            "text": {"tag": "plain_text", "content": label},
            "type": kind,
            "url": url,
        })
    return actions


def paper_md(i: int, p, topics) -> str:
    t = topics.get(p.best_topic)
    L = p.llm or {}
    tldr = (L.get("tldr") or ((p.abstract or "")[:120] + "…"))[:80]
    score = p.final_score * 10
    badge = "🟢" if score >= 7 else "🔵" if score >= 5 else "⚪"
    flags = []
    if p.signals.get("venue_name"):
        flags.append(f"🏆{p.signals['venue_name']}")
    if p.signals.get("rigor_hits"):
        flags.append("🧪" + "/".join(p.signals["rigor_hits"][:3]))
    if p.signals.get("hf_upvotes"):
        flags.append(f"HF {p.signals['hf_upvotes']}")
    if p.signals.get("max_hindex"):
        flags.append(f"h{p.signals['max_hindex']}")
    if p.signals.get("institution"):
        flags.append(p.signals["institution"])
    if p.signals.get("openreview_rating"):
        flags.append(f"OR {p.signals['openreview_rating']}")
    if p.signals.get("explore"):
        flags.append("探索")
    dims = ""
    if L and not L.get("heuristic"):
        dims = f" · 相关{L['relevance']:.0f}/新颖{L['novelty']:.0f}/实验{L['rigor']:.0f}"
    topic = t.name if t else p.best_topic
    authors = ", ".join(p.authors[:3]) + (" 等" if len(p.authors) > 3 else "")
    lines = [
        f"{badge} **{i}. [{p.title}]({p.link})**",
        f"<font color='grey'>{topic} · {p.published} · {authors}</font>",
        f"**分数 {score:.1f}**{dims}  {' '.join(flags)}".rstrip(),
        f"📝 {tldr}",
    ]
    if L.get("reason"):
        lines.append(f"💡 {L['reason'][:120]}")
    if not p.pdf_link:
        lines.append("<font color='grey'>暂无开放获取 PDF</font>")
    lines.append(f"<font color='grey'>反馈 `python -m paper_radar feedback like {p.key}`</font>")
    return "\n".join(lines)


def _card(direction: str, elements: list, date_str: str, n_papers: int, part: str) -> dict:
    title = f"📡 Paper Radar {date_str} · {direction}（{n_papers}）{part}"
    return {"msg_type": "interactive", "card": {
        "config": {"wide_screen_mode": True},
        "header": {"template": COLORS.get(direction, "purple"),
                   "title": {"tag": "plain_text", "content": title[:120]}},
        "elements": elements,
    }}


def _note(stats: dict) -> dict:
    return {"tag": "note", "elements": [{"tag": "plain_text", "content":
            f"候选 {stats.get('candidates', 0)} → 预筛 {stats.get('prefiltered', 0)} → 推送 {stats.get('selected', 0)}"
            f" · {stats.get('scorer', '')}"}]}


def build_cards(grouped: dict, topics, date_str: str, stats: dict, max_bytes: int = 18000) -> list[dict]:
    cards = []
    idx = 1
    for direction, papers in grouped.items():
        if not papers:
            continue
        chunks: list[list] = []
        cur_els: list[dict] = []
        cur_n = 0
        for p in papers:
            block = [
                {"tag": "markdown", "content": paper_md(idx, p, topics)},
                {"tag": "action", "actions": _buttons(p)},
                {"tag": "hr"},
            ]
            idx += 1
            trial = cur_els + block
            blob = _card(direction, trial + [_note(stats)], date_str, cur_n + 1, "")
            if cur_els and len(json.dumps(blob, ensure_ascii=False).encode()) > max_bytes:
                chunks.append(cur_els)
                cur_els, cur_n = block, 1
            else:
                cur_els = trial
                cur_n += 1
        if cur_els:
            chunks.append(cur_els)
        n_parts = len(chunks)
        for pi, els in enumerate(chunks, 1):
            part = f" {pi}/{n_parts}" if n_parts > 1 else ""
            # recount papers in this chunk by hr tags
            n_here = sum(1 for e in els if e.get("tag") == "hr")
            cards.append(_card(direction, els + [_note(stats)], date_str, n_here, part))
    return cards


def send(cards: list[dict], webhook: str, secret: str = "") -> int:
    ok = 0
    for c in cards:
        body = dict(c)
        if secret:
            ts = int(time.time())
            body.update({"timestamp": str(ts), "sign": sign(secret, ts)})
        r = request("POST", webhook, json=body, retries=2, pace=0.3)
        js = {}
        if r is not None and "json" in r.headers.get("content-type", ""):
            try:
                js = r.json()
            except ValueError:
                js = {}
        if r is not None and r.status_code == 200 and js.get("code", js.get("StatusCode", 0)) == 0:
            ok += 1
        else:
            log.error("Feishu push failed: %s", js or (None if r is None else r.text[:200]))
    return ok
