from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

_ARXIV_VER = re.compile(r"v\d+$", re.I)


def norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def norm_arxiv(aid: str | None) -> str | None:
    if not aid:
        return None
    s = aid.strip()
    s = re.sub(r"^https?://arxiv\.org/(abs|pdf)/", "", s, flags=re.I)
    s = re.sub(r"^arxiv:", "", s, flags=re.I)
    s = s.split("?")[0].strip().strip("/")
    s = _ARXIV_VER.sub("", s)
    return s or None


def norm_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    d = doi.strip()
    d = re.sub(r"^https?://(dx\.)?doi\.org/", "", d, flags=re.I)
    d = re.sub(r"^doi:", "", d, flags=re.I)
    d = d.split("?")[0].strip()
    return d.lower() or None


def arxiv_from_doi(doi: str | None) -> str | None:
    d = norm_doi(doi) or ""
    m = re.search(r"arxiv\.(\d{4}\.\d{4,5})", d)
    return m.group(1) if m else None


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
    pdf_url: Optional[str] = None
    code_url: Optional[str] = None
    affiliations: list[str] = field(default_factory=list)
    citation_count: Optional[int] = None
    # ranking fields
    emb_scores: dict[str, float] = field(default_factory=dict)
    best_topic: str = ""
    emb_score: float = 0.0
    signals: dict = field(default_factory=dict)
    llm: dict = field(default_factory=dict)
    final_score: float = 0.0

    def __post_init__(self) -> None:
        self.arxiv_id = norm_arxiv(self.arxiv_id)
        self.doi = norm_doi(self.doi)
        if not self.arxiv_id:
            self.arxiv_id = arxiv_from_doi(self.doi)

    @property
    def key(self) -> str:
        if self.arxiv_id:
            return f"arxiv:{self.arxiv_id}"
        if self.doi:
            return f"doi:{self.doi}"
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

    @property
    def pdf_link(self) -> Optional[str]:
        """Direct PDF if we have one. arXiv always does."""
        if self.pdf_url:
            return self.pdf_url
        if self.arxiv_id:
            return f"https://arxiv.org/pdf/{self.arxiv_id}"
        return None
