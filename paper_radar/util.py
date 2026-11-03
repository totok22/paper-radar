"""Small shared helpers."""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo


def shanghai_today() -> str:
    """Calendar date in China, so the 00:30 UTC run is labeled as that morning."""
    return datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat()


def openai_url(base: str, suffix: str) -> str:
    """Join an OpenAI-compatible base URL with `/chat/completions` or `/embeddings`.

    Accepts both `https://api.deepseek.com` and `https://api.deepseek.com/v1`,
    and a base that already ends with the suffix.
    """
    b = (base or "").rstrip("/")
    suf = suffix if suffix.startswith("/") else f"/{suffix}"
    if b.endswith(suf):
        return b
    return b + suf
