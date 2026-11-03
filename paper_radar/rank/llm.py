"""LLM reviewer (OpenAI-compatible chat API, default DeepSeek). Degrades to heuristics."""
from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor

from ..http import request

log = logging.getLogger("paper_radar")

PROMPT = """你是自动驾驶方向（运动规划/运动控制/场景生成）的资深审稿人，正在为一名刚入门的大四学生筛选每日论文。
课题组兴趣子方向：
{topics}

待评审论文：
标题：{title}
作者：{authors}
备注/录用信息：{comments}
摘要：{abstract}

请严格输出 JSON（不要其他文字）：
{{"topic": "<最相关的子方向 id>",
  "relevance": <1-10, 与该子方向的相关度>,
  "novelty": <1-10, 方法新颖性>,
  "rigor": <1-10, 实验扎实程度：真车/仿真闭环、nuPlan/Waymo/CARLA/nuScenes 等基准、对比与消融>,
  "beginner": <1-10, 对入门者的可读性/参考价值>,
  "tldr": "<一句话中文 TL;DR，≤60字>",
  "reason": "<中文推荐/不推荐理由，≤80字>"}}
与所有子方向都无关时 relevance 给 1-3。"""


class LLMReviewer:
    def __init__(self, lcfg: dict, topics: dict):
        self.cfg = lcfg
        self.topics = topics
        self.enabled = bool(lcfg.get("api_key"))
        self.topic_txt = "\n".join(f"- {t.id}（{t.direction}/{t.name}）：{t.description[:160]}" for t in topics.values())

    def review(self, papers) -> int:
        if not self.enabled:
            log.info("LLM_API_KEY not set: using heuristic scoring")
            return 0
        with ThreadPoolExecutor(int(self.cfg.get("concurrency", 4))) as ex:
            results = list(ex.map(self._one, papers))
        ok = 0
        for p, res in zip(papers, results):
            if res:
                p.llm = res
                if res.get("topic") in self.topics:
                    p.best_topic = res["topic"]
                ok += 1
        if ok == 0:
            log.warning("all LLM calls failed; heuristic scoring used")
        return ok

    def _one(self, p) -> dict | None:
        msg = PROMPT.format(topics=self.topic_txt, title=p.title, authors=", ".join(p.authors[:8]),
                            comments=p.comments or p.venue or "无", abstract=p.abstract[:2500])
        r = request("POST", self.cfg["base_url"].rstrip("/") + "/chat/completions", timeout=90, retries=2,
                    headers={"Authorization": f"Bearer {self.cfg['api_key']}"},
                    json={"model": self.cfg["model"], "temperature": 0.2,
                          "messages": [{"role": "user", "content": msg}],
                          **({"response_format": {"type": "json_object"}} if self.cfg.get("json_mode", True) else {})})
        if r is None or r.status_code != 200:
            log.debug("LLM error: %s", None if r is None else r.text[:200])
            return None
        try:
            txt = r.json()["choices"][0]["message"]["content"]
            m = re.search(r"\{.*\}", txt, re.S)
            d = json.loads(m.group(0) if m else txt)
            for k in ("relevance", "novelty", "rigor", "beginner"):
                d[k] = max(1.0, min(10.0, float(d.get(k, 5))))
            return d
        except Exception as e:
            log.debug("LLM parse error: %s", e)
            return None
