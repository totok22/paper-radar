"""Fail fast on a broken config; warn on softer issues."""
from __future__ import annotations

import re

_ISSN = re.compile(r"^\d{4}-?\d{3}[\dXx]$")


class ConfigError(ValueError):
    pass


def validate_config(cfg: dict) -> list[str]:
    """Return warnings. Raise ConfigError when the run cannot proceed."""
    warnings: list[str] = []
    topics = cfg.get("topics") or []
    if not isinstance(topics, list) or not topics:
        raise ConfigError("config.topics 为空：至少需要一个兴趣子方向")
    ids: list[str] = []
    for i, t in enumerate(topics):
        if not isinstance(t, dict):
            raise ConfigError(f"topics[{i}] 不是映射")
        for k in ("id", "name", "direction"):
            if not str(t.get(k) or "").strip():
                raise ConfigError(f"topics[{i}] 缺少 {k}")
        ids.append(t["id"])
        if not (t.get("description") or t.get("query")):
            warnings.append(f"子方向 {t['id']} 没有 description/query，检索和嵌入都会变弱")
    dup = {x for x in ids if ids.count(x) > 1}
    if dup:
        raise ConfigError(f"子方向 id 重复: {', '.join(sorted(dup))}")

    rk = cfg.get("ranking") or {}
    try:
        top_k = int(rk.get("top_k", 40))
        top_n = int(rk.get("top_n", 12))
    except (TypeError, ValueError) as e:
        raise ConfigError(f"ranking.top_k / top_n 必须是整数: {e}") from e
    if top_k < 1 or top_n < 1:
        raise ConfigError("ranking.top_k 和 top_n 必须 ≥ 1")
    if top_n > top_k:
        warnings.append(f"ranking.top_n ({top_n}) > top_k ({top_k})，推送数不会超过预筛数")
    weights = rk.get("weights") or {}
    if weights:
        try:
            total = sum(float(v) for v in weights.values())
        except (TypeError, ValueError) as e:
            raise ConfigError(f"ranking.weights 必须是数字: {e}") from e
        if abs(total - 1.0) > 0.08:
            warnings.append(f"ranking.weights 之和为 {total:.2f}（不是 1），分数仍会按这些权重相加")

    issns = ((cfg.get("sources") or {}).get("journals") or {}).get("issns") or {}
    for name, issn in issns.items():
        if issn and not _ISSN.match(str(issn).strip()):
            warnings.append(f"期刊 {name} 的 ISSN 看起来不合法: {issn}")

    days = cfg.get("lookback_days", 3)
    try:
        if int(days) < 1:
            raise ConfigError("lookback_days 必须 ≥ 1")
    except (TypeError, ValueError) as e:
        raise ConfigError(f"lookback_days 必须是整数: {e}") from e
    return warnings
