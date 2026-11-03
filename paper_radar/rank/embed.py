"""Embedding backends: OpenAI-compatible API -> sentence-transformers -> TF-IDF."""
from __future__ import annotations

import logging

import numpy as np

from ..http import request

log = logging.getLogger("paper_radar")


def _l2(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


class Embedder:
    def __init__(self, ecfg: dict):
        self.cfg = ecfg
        self.backend = "tfidf"
        if ecfg.get("api_key") and ecfg.get("base_url") and ecfg.get("model"):
            self.backend = "api"
        elif ecfg.get("local_model"):
            try:
                from sentence_transformers import SentenceTransformer  # noqa
                self.backend = "local"
            except ImportError:
                log.info("sentence-transformers not installed; using TF-IDF")

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode all texts jointly (TF-IDF needs a shared vocabulary)."""
        if self.backend == "api":
            try:
                return self._api(texts)
            except Exception as e:
                log.warning("embedding API failed (%s); falling back to TF-IDF", e)
                self.backend = "tfidf"
        if self.backend == "local":
            from sentence_transformers import SentenceTransformer
            m = SentenceTransformer(self.cfg["local_model"])
            return _l2(np.asarray(m.encode(texts, batch_size=32, show_progress_bar=False)))
        from sklearn.feature_extraction.text import TfidfVectorizer
        v = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True, min_df=1, max_df=0.6)
        return _l2(v.fit_transform(texts).toarray())

    def _api(self, texts: list[str]) -> np.ndarray:
        url = self.cfg["base_url"].rstrip("/") + "/embeddings"
        bs, vecs = int(self.cfg.get("batch_size", 32)), []
        for i in range(0, len(texts), bs):
            r = request("POST", url, json={"model": self.cfg["model"], "input": [t[:6000] for t in texts[i:i + bs]]},
                        headers={"Authorization": f"Bearer {self.cfg['api_key']}"})
            if r is None or r.status_code != 200:
                raise RuntimeError(None if r is None else f"{r.status_code} {r.text[:200]}")
            vecs += [d["embedding"] for d in sorted(r.json()["data"], key=lambda d: d["index"])]
        return _l2(np.asarray(vecs, dtype=float))
