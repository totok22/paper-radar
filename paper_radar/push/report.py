"""Markdown + HTML daily report."""
from __future__ import annotations

import html
from pathlib import Path


def _row(i, p, topics):
    t = topics.get(p.best_topic)
    L, s = p.llm, p.signals
    parts = [f"### {i}. [{p.title}]({p.link})",
             f"- 子方向：{t.direction + ' / ' + t.name if t else p.best_topic}　|　日期：{p.published}　|　来源：{', '.join(p.sources)}",
             f"- 作者：{', '.join(p.authors[:6])}{' 等' if len(p.authors) > 6 else ''}",
             f"- **综合分 {p.final_score * 10:.1f}**　嵌入相关 {s.get('emb_norm', 0):.2f}　质量 {s.get('quality', 0):.2f}"
             + (f"　LLM: 相关{L['relevance']:.0f} 新颖{L['novelty']:.0f} 实验{L['rigor']:.0f} 入门{L.get('beginner', 0):.0f}" if L else "")]
    sig = []
    if s.get("venue_name"):
        sig.append(f"刊会: {s['venue_name']}")
    if p.comments:
        sig.append(f"备注: {p.comments[:120]}")
    if s.get("rigor_hits"):
        sig.append("实验关键词: " + ", ".join(s["rigor_hits"]))
    if s.get("max_hindex"):
        sig.append(f"作者最高 h-index: {s['max_hindex']}")
    if s.get("hf_upvotes"):
        sig.append(f"HF upvotes: {s['hf_upvotes']}")
    if sig:
        parts.append("- " + "；".join(sig))
    if L.get("tldr"):
        parts.append(f"- 📝 TL;DR：{L['tldr']}")
    if L.get("reason"):
        parts.append(f"- 💡 理由：{L['reason']}")
    parts.append(f"- 摘要：{p.abstract[:600]}{'…' if len(p.abstract) > 600 else ''}")
    parts.append(f"- 反馈：`python -m paper_radar feedback like {p.key}` / `dislike`")
    return "\n".join(parts)


def write_report(grouped: dict, topics, date_str: str, stats: dict, outdir: str = "reports") -> tuple[Path, Path]:
    Path(outdir).mkdir(parents=True, exist_ok=True)
    md = [f"# 📡 Paper Radar 日报 {date_str}", "",
          f"> 来源召回：{stats.get('per_source')}　→ 去重后 {stats.get('candidates')}　→ 嵌入预筛 {stats.get('prefiltered')}"
          f"　→ 推送 {stats.get('selected')}　|　嵌入: {stats.get('embedder')}　评分: {stats.get('scorer')}", ""]
    i = 1
    for direction, papers in grouped.items():
        if not papers:
            continue
        md.append(f"## {direction}（{len(papers)}）\n")
        for p in papers:
            md.append(_row(i, p, topics) + "\n")
            i += 1
    text = "\n".join(md)
    mdp = Path(outdir) / f"{date_str}.md"
    mdp.write_text(text, encoding="utf-8")
    try:
        import markdown  # optional nicer rendering
        body = markdown.markdown(text)
    except ImportError:
        body = "<pre style='white-space:pre-wrap'>" + html.escape(text) + "</pre>"
    hp = Path(outdir) / f"{date_str}.html"
    hp.write_text(f"<!doctype html><meta charset='utf-8'><title>Paper Radar {date_str}</title>"
                  f"<body style='max-width:900px;margin:auto;font-family:sans-serif;line-height:1.6'>{body}</body>",
                  encoding="utf-8")
    return mdp, hp
