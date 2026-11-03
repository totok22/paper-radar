from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


def norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


@dataclass
class Paper:
    title: str
    abstract: str = ""
    authors: list[str] = field(default_factory=list)
    published: str = ""  # YYYY-MM-DD
    url: str = ""
    arxiv_id: Optional[str] = None
    doi: Optional[str] = None
    s2_id: Optional[str] = None
    venue: str = ""
    comments: str = ""  # arXiv comments / journal_ref
    categories: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    # ranking fields
    emb_scores: dict[str, float] = field(default_factory=dict)
    best_topic: str = ""
    emb_score: float = 0.0
    signals: dict = field(default_factory=dict)
    llm: dict = field(default_factory=dict)
    final_score: float = 0.0

    @property
    def key(self) -> str:
        if self.arxiv_id:
            return f"arxiv:{self.arxiv_id}"
        if self.doi:
            return f"doi:{self.doi.lower()}"
        return f"title:{norm_title(self.title)[:120]}"

    @property
    def s2_ref(self) -> Optional[str]:
        """ID usable with Semantic Scholar endpoints."""
        if self.s2_id:
            return self.s2_id
        if self.arxiv_id:
            return f"ARXIV:{self.arxiv_id}"
        if self.doi:
            return f"DOI:{self.doi}"
        return None

    @property
    def text(self) -> str:
        return f"{self.title}. {self.abstract}"

    @property
    def link(self) -> str:
        if self.arxiv_id:
            return f"https://arxiv.org/abs/{self.arxiv_id}"
        if self.doi:
            return f"https://doi.org/{self.doi}"
        return self.url
