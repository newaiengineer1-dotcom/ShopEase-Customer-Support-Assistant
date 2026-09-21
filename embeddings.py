"""Embedders. Production = Sentence Transformers. 'hash' = offline deterministic stand-in for tests/CI."""
from __future__ import annotations

import hashlib

import numpy as np

from .utils import tokenize


class HashEmbedder:
    """Signed feature hashing of tokens. NOT semantic; only for offline tests."""

    def __init__(self, dim=2048):
        self.dim = dim

    def _vec(self, text):
        v = np.zeros(self.dim, dtype="float32")
        for tok in tokenize(text):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 64) & 1 else -1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode_documents(self, texts):
        return np.stack([self._vec(t) for t in texts]).astype("float32")

    def encode_query(self, text):
        return self._vec(text)[None, :]


class STEmbedder:
    def __init__(self, model, query_prefix="", batch_size=32):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model)
        self.query_prefix, self.batch_size = query_prefix, batch_size
        self.dim = int(self.model.encode(["dimension probe"]).shape[1])

    def encode_documents(self, texts):
        return self.model.encode(list(texts), batch_size=self.batch_size, normalize_embeddings=True,
                                 show_progress_bar=False).astype("float32")

    def encode_query(self, text):
        return self.model.encode([self.query_prefix + text], normalize_embeddings=True,
                                 show_progress_bar=False).astype("float32")


def get_embedder(cfg):
    e = cfg.embedding
    if e.provider == "hash":
        return HashEmbedder()
    return STEmbedder(e.model, e.query_prefix, e.batch_size)
