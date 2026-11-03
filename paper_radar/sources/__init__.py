from .arxiv import ArxivSource
from .openalex import OpenAlexSource, OpenAlexJournalSource
from .semantic_scholar import S2RecommendSource
from .rss import RSSSource

REGISTRY = {
    "arxiv": ArxivSource,
    "openalex": OpenAlexSource,
    "journals": OpenAlexJournalSource,
    "semantic_scholar": S2RecommendSource,
    "rss": RSSSource,
}

__all__ = ["REGISTRY"]
