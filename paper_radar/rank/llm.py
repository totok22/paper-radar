"""LLM review (OpenAI-compatible, default DeepSeek). Falls back to a heuristic.

Papers are reviewed in small batches so a daily run stays inside the GitHub
Actions window. A batch that comes back malformed is retried, then split into
single calls. Papers the model never scores keep the embedding heuristic and
still get a short Chinese blurb for the card.
"""
from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor

from ..http import request
from ..util import openai_url

log = logging.getLogger("paper_radar")

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.I)

SINGLE = """你是自动驾驶方向（运动规划 / 运动控制 / 场景生成）的审稿人，正在帮一名大四学生筛每日论文。
兴趣子方向：
{topics}

论文：
标题：{title}
作者：{authors}
备注/录用信息：{comments}
摘要：{abstract}

只输出一个 JSON 对象，不要其他文字：
{{"topic": "<最相关的子方向 id，必须来自上面的列表>",
  "relevance": <1-10 整数，与该子方向的相关度>,
  "novelty": <1-10 整数>,
  "rigor": <1-10 整数，真车/闭环仿真/nuPlan/Waymo/CARLA 等基准、对比与消融>,
  "beginner": <1-10 整数，对入门者的可读性>,
  "tldr": "<一句中文 TL;DR，不超过 60 字>",
  "reason": "<一句中文，为什么推荐或不推荐，不超过 80 字>"}}
和所有子方向都无关时 relevance 给 1-3。"""

BATCH = """你是自动驾驶方向（运动规划 / 运动控制 / 场景生成）的审稿人，正在帮一名大四学生筛每日论文。
兴趣子方向：
{topics}

待评论文（JSON）：
{items}

只输出一个 JSON 对象，不要其他文字：
{{"reviews": [
  {{"i": <序号>, "topic": "<子方向 id>", "relevance": <1-10>, "novelty": <1-10>,
    "rigor": <1-10>, "beginner": <1-10>,
    "tldr": "<一句中文 TL;DR，不超过 60 字>",
    "reason": "<一句中文推荐理由，不超过 80 字>"}}
]}}
reviews 必须覆盖每一篇。和所有子方向都无关时 relevance 给 1-3。"""


def extract_json(txt: str):
    raw = _FENCE.sub("", (txt or "").strip())
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    for pat in (r"\{.*\}", r"\[.*\]"):
        m = re.search(pat, raw, re.S)
        if not m:
            continue
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
    raise ValueError("no json")


def normalize_review(d: dict, valid_topics: set[str]) -> dict | None:
    if not isinstance(d, dict):
        return None
    try:
        out = {
            "topic": str(d.get("topic") or ""),
            "relevance": max(1.0, min(10.0, float(d.get("relevance", 5)))),
            "novelty": max(1.0, min(10.0, float(d.get("novelty", 5)))),
            "rigor": max(1.0, min(10.0, float(d.get("rigor", 5)))),
            "beginner": max(1.0, min(10.0, float(d.get("beginner", 5)))),
            "tldr": str(d.get("tldr") or "").strip()[:80],
            "reason": str(d.get("reason") or "").strip()[:120],
        }
    except (TypeError, ValueError):
        return None
    if out["topic"] not in valid_topics:
        out["topic"] = ""
    if not out["tldr"]:
        return None
    return out


def heuristic_text(p, topics) -> dict:
    t = topics.get(p.best_topic)
    name = t.name if t else "综合"
    bits = []
    if p.signals.get("venue_name"):
        bits.append(f"刊会线索：{p.signals['venue_name']}")
    if p.code_url:
        bits.append("摘要里有代码链接")
    if p.signals.get("rigor_hits"):
        bits.append("实验线索：" + "、".join(p.signals["rigor_hits"][:3]))
    if p.signals.get("hf_upvotes"):
        bits.append(f"Hugging Face {p.signals['hf_upvotes']} 赞")
    if p.signals.get("institution"):
        bits.append(f"机构：{p.signals['institution']}")
    sent = re.split(r"(?<=[.!?])\s", (p.abstract or "").strip())
    first = (sent[0] if sent else "")[:90]
    reason = f"与「{name}」的兴趣画像最接近。" + (" ".join(bits) if bits else "未调用大模型，相关度来自嵌入预筛。")
    return {
        "topic": p.best_topic,
        "tldr": f"（启发式）{name}：{first}"[:80],
        "reason": reason[:120],
        "heuristic": True,
    }


def fill_heuristic_text(papers, topics) -> None:
    for p in papers:
        if not p.llm:
            p.llm = heuristic_text(p, topics)


class LLMReviewer:
    def __init__(self, lcfg: dict, topics: dict):
        self.cfg = lcfg or {}
        self.topics = topics
        self.enabled = bool(self.cfg.get("api_key"))
        self.valid = set(topics)
        self.topic_txt = "\n".join(
            f"- {t.id}（{t.direction}/{t.name}）：{t.description[:180]}" for t in topics.values()
        )

    def review(self, papers) -> int:
        if not self.enabled or not papers:
            if not self.enabled:
                log.info("LLM_API_KEY not set: heuristic scoring, no model TL;DR")
            return 0
        batch = max(1, int(self.cfg.get("batch_size", 4)))
        groups = [papers[i:i + batch] for i in range(0, len(papers), batch)]
        workers = max(1, int(self.cfg.get("concurrency", 2)))
        with ThreadPoolExecutor(workers) as ex:
            parts = list(ex.map(self._batch, groups))
        ok = 0
        for group, got in zip(groups, parts):
            for p, res in zip(group, got):
                if res:
                    p.llm = res
                    if res.get("topic") in self.topics:
                        p.best_topic = res["topic"]
                    ok += 1
        if ok == 0:
            log.warning("all LLM calls failed; heuristic scoring used")
        else:
            log.info("LLM reviewed %d/%d with %s", ok, len(papers), self.cfg.get("model"))
        return ok

    def _batch(self, papers) -> list[dict | None]:
        if len(papers) == 1:
            return [self._one(papers[0])]
        got = self._call_batch(papers)
        if got and any(got):
            missing = [p for p, r in zip(papers, got) if not r]
            if missing:
                extra = {id(p): self._one(p) for p in missing}
                got = [r or extra[id(p)] for p, r in zip(papers, got)]
            return got
        return [self._one(p) for p in papers]

    def _paper_item(self, i: int, p, abstract_chars: int) -> dict:
        return {
            "i": i,
            "title": p.title,
            "authors": ", ".join(p.authors[:6]),
            "comments": (p.comments or p.venue or "无")[:240],
            "abstract": (p.abstract or "")[:abstract_chars],
        }

    def _call_batch(self, papers) -> list[dict | None] | None:
        items = [self._paper_item(i + 1, p, 900) for i, p in enumerate(papers)]
        msg = BATCH.format(topics=self.topic_txt, items=json.dumps(items, ensure_ascii=False))
        txt = self._chat(msg, max_tokens=min(700 * len(papers), 3500))
        if not txt:
            return None
        try:
            data = extract_json(txt)
        except ValueError:
            return None
        rows = data.get("reviews") if isinstance(data, dict) else data
        if not isinstance(rows, list):
            return None
        by_i: dict[int, dict] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            try:
                idx = int(row.get("i"))
            except (TypeError, ValueError):
                continue
            norm = normalize_review(row, self.valid)
            if norm:
                by_i[idx] = norm
        return [by_i.get(i + 1) for i in range(len(papers))]

    def _one(self, p) -> dict | None:
        msg = SINGLE.format(
            topics=self.topic_txt, title=p.title, authors=", ".join(p.authors[:8]),
            comments=(p.comments or p.venue or "无")[:300], abstract=(p.abstract or "")[:1800],
        )
        txt = self._chat(msg, max_tokens=500)
        if not txt:
            return None
        try:
            return normalize_review(extract_json(txt), self.valid)
        except ValueError:
            return None

    def _chat(self, msg: str, max_tokens: int) -> str | None:
        url = openai_url(self.cfg.get("base_url") or "https://api.deepseek.com", "/chat/completions")
        body = {
            "model": self.cfg.get("model") or "deepseek-flash",
            "temperature": float(self.cfg.get("temperature", 0.1)),
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": msg}],
        }
        use_json = bool(self.cfg.get("json_mode", True))
        attempts = max(1, int(self.cfg.get("max_retries", 2)))
        for i in range(attempts):
            payload = dict(body)
            if use_json:
                payload["response_format"] = {"type": "json_object"}
            r = request(
                "POST", url, timeout=90, retries=2,
                headers={"Authorization": f"Bearer {self.cfg['api_key']}"},
                json=payload, pace=0,
            )
            if r is not None and r.status_code == 400 and use_json:
                log.info("LLM rejected json_mode; retrying without response_format")
                use_json = False
                continue
            if r is None or r.status_code != 200:
                log.debug("LLM error: %s", None if r is None else r.text[:200])
                continue
            try:
                return r.json()["choices"][0]["message"]["content"]
            except (KeyError, ValueError, IndexError, TypeError) as e:
                log.debug("LLM body parse error: %s", e)
        return None
