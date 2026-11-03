"""Pluggable interest sources.

An InterestSource contributes to Topic profiles: descriptions, positive /
negative seed papers (S2-compatible IDs such as "ARXIV:2310.03026" or
"DOI:10.xxx"), and free-text examples used by the embedding pre-filter.
New sources (Zotero, Notion, a BibTeX file...) only need to implement
`contribute(topics)`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger("paper_radar")


@dataclass
class Topic:
    id: str
    name: str
    direction: str
    description: str
    positive_ids: list[str] = field(default_factory=list)
    negative_ids: list[str] = field(default_factory=list)
    positive_texts: list[str] = field(default_factory=list)
    negative_texts: list[str] = field(default_factory=list)
    query: str = ""  # short keyword query for search APIs (OpenAlex)

    @property
    def profile_text(self) -> str:
        return " ".join([self.name, self.description, *self.positive_texts])


class InterestSource:
    name = "base"

    def contribute(self, topics: dict[str, Topic]) -> None:  # pragma: no cover
        raise NotImplementedError


def build_topics(cfg: dict, store=None) -> dict[str, Topic]:
    from .config_source import ConfigInterestSource
    from .feedback_source import FeedbackInterestSource
    from .zotero_source import ZoteroInterestSource

    topics: dict[str, Topic] = {}
    sources: list[InterestSource] = [ConfigInterestSource(cfg)]
    if store is not None:
        sources.append(FeedbackInterestSource(store))
    zcfg = cfg.get("zotero", {})
    if zcfg.get("enabled"):
        sources.append(ZoteroInterestSource(zcfg))
    for s in sources:
        try:
            s.contribute(topics)
        except Exception as e:  # never let one source kill the run
            log.warning("interest source %s failed: %s", s.name, e)
    return topics
