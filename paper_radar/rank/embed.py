"""Embedding backends: OpenAI-compatible API -> sentence-transformers -> TF-IDF.

API vectors are cached in SQLite by (model, text hash). TF-IDF depends on
the day's joint vocabulary, so it is not cached.
"""
from __future__ import annotations

import hashlib
import logging

import numpy as np

from ..http import request
from ..util import openai_url

log = logging.getLogger("paper_radar")


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-9)


def _cache_key(model: str, text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:32]
    return f"emb:{model}:{digest}"


class Embedder:
    def __init__(self, ecfg: dict, store=None):
        self.cfg = ecfg or {}
        self.store = store
        self.backend = "tfidf"
        if ecfg.get("api_key") and ecfg.get("base_url") and ecfg.get("model"):
            self.backend = "api"
        elif ecfg.get("local_model"):
            try:
                from sentence_transformers import SentenceTransformer  # noqa: F401
                self.backend = "local"
            except ImportError:
                log.info("sentence-transformers not installed; using TF-IDF")

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode all texts jointly (TF-IDF needs a shared vocabulary)."""
        if not texts:
            return np.zeros((0, 1))
        if self.backend == "api":
            try:
                return self._api(texts)
            except Exception as e:
                log.warning("embedding API failed (%s); falling back to TF-IDF", e)
                self.backend = "tfidf"
        if self.backend == "local":
            from sentence_transformers import SentenceTransformer
            m = SentenceTransformer(self.cfg["local_model"])
            return _l2(np.asarray(m.encode(texts, batch_size=32, show_progress_bar=False), dtype=float))
        from sklearn.feature_extraction.text import TfidfVectorizer
        # max_df < 1 drops tokens that appear in almost every doc, which on a
        # tiny corpus (tests, a quiet day) deletes the whole vocabulary.
        max_df = 1.0 if len(texts) < 8 else 0.95
        v = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True,
                            min_df=1, max_df=max_df, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_]+\b")
        return _l2(v.fit_transform([(t or "empty") for t in texts]).toarray())

    def _api(self, texts: list[str]) -> np.ndarray:
        model = self.cfg["model"]
        clipped = [t[:6000] for t in texts]
        use_cache = bool(self.store is not None and self.cfg.get("cache", True))
        vecs: list[list[float] | None] = [None] * len(clipped)
        missing: list[int] = []
        if use_cache:
            for i, t in enumerate(clipped):
                hit = self.store.cache_get(_cache_key(model, t))
                if isinstance(hit, list) and hit:
                    vecs[i] = hit
                else:
                    missing.append(i)
        else:
            missing = list(range(len(clipped)))
        if missing:
            url = openai_url(self.cfg["base_url"], "/embeddings")
            bs = int(self.cfg.get("batch_size", 32))
            fresh: list[tuple[str, list]] = []
            for start in range(0, len(missing), bs):
                chunk = missing[start:start + bs]
                r = request("POST", url, json={"model": model, "input": [clipped[i] for i in chunk]},
                            headers={"Authorization": f"Bearer {self.cfg['api_key']}"}, timeout=60, retries=3)
                if r is None or r.status_code != 200:
                    raise RuntimeError(None if r is None else f"{r.status_code} {r.text[:200]}")
                data = sorted(r.json()["data"], key=lambda d: d["index"])
                if len(data) != len(chunk):
                    raise RuntimeError(f"embedding batch size mismatch: {len(data)} != {len(chunk)}")
                for i, d in zip(chunk, data):
                    emb = [round(float(x), 6) for x in d["embedding"]]
                    vecs[i] = emb
                    fresh.append((_cache_key(model, clipped[i]), emb))
            if use_cache and fresh:
                self.store.cache_set_many(fresh)
        if any(v is None for v in vecs):
            raise RuntimeError("embedding cache hole")
        log.info("embeddings model=%s cached=%d fetched=%d", model, len(texts) - len(missing), len(missing))
        return _l2(np.asarray(vecs, dtype=float))
