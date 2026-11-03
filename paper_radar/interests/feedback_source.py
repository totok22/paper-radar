from __future__ import annotations

from .base import InterestSource, Topic


class FeedbackInterestSource(InterestSource):
    """Feeds 👍/👎 feedback (from the SQLite store) back into topic profiles."""

    name = "feedback"

    def __init__(self, store):
        self.store = store

    def contribute(self, topics: dict[str, Topic]) -> None:
        for row in self.store.feedback_rows():
            t = topics.get(row["topic"]) if row["topic"] else None
            targets = [t] if t else list(topics.values())
            for tp in targets:
                if row["label"] > 0:
                    if row["s2_ref"]:
                        tp.positive_ids.append(row["s2_ref"])
                    if t:  # only enrich text of the specific topic
                        tp.positive_texts.append(row["title"])
                else:
                    if row["s2_ref"]:
                        tp.negative_ids.append(row["s2_ref"])
                    if t:
                        tp.negative_texts.append(row["title"])
