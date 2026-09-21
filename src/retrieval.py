"""FAISS + BM25 hybrid retrieval with permission and metadata gates."""
from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import faiss
import numpy as np

from .utils import chunk_embed_text, sha256_file, tokenize

ACCESS = {"customer": {"public"}, "staff": {"public", "internal"}}   # unknown roles fall back to customer
REQUIRED_META = ("document_id", "filename", "chunk_id", "section", "source", "version", "access_level", "status")


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.tf = [Counter(d) for d in docs]
        self.dl = [len(d) for d in docs]
        self.n = len(docs)
        self.avgdl = (sum(self.dl) / self.n) if self.n else 1.0
        self.df = Counter(t for d in docs for t in set(d))
        self.k1, self.b = k1, b

    def scores(self, query_tokens):
        out = np.zeros(self.n, dtype="float32")
        for t in set(query_tokens):
            df = self.df.get(t, 0)
            if not df:
                continue
            idf = math.log(1 + (self.n - df + 0.5) / (df + 0.5))
            for i, tf in enumerate(self.tf):
                f = tf.get(t, 0)
                if f:
                    out[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl))
        return out


@dataclass
class Hit:
    label: str      # S1, S2 ... (what the LLM cites)
    chunk: dict     # stored metadata + text (what the UI cites)
    dense: float
    bm25: float
    fused: float


@dataclass
class Retrieval:
    hits: list
    sufficient: bool
    stats: dict


def load_index(index_dir):
    d = Path(index_dir)
    index = faiss.read_index(str(d / "faiss.index"))
    chunks = [json.loads(x) for x in (d / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    info = json.loads((d / "index_manifest.json").read_text(encoding="utf-8"))
    return index, chunks, info


def validate_index(index, chunks, info, kb_dir, dim=None):
    """Returns a list of problems (empty list = index is consistent with the KB on disk)."""
    kb_dir, bad = Path(kb_dir), []
    if index.ntotal != len(chunks):
        bad.append(f"vector count {index.ntotal} != chunk count {len(chunks)}")
    if index.d != info.get("dim") or (dim and index.d != dim):
        bad.append(f"embedding dimension mismatch (index={index.d}, manifest={info.get('dim')}, embedder={dim}); rebuild the index")
    ids = [c.get("chunk_id") for c in chunks]
    if len(ids) != len(set(ids)):
        bad.append("duplicate chunk_id values")
    for c in chunks:
        missing = [k for k in REQUIRED_META if not c.get(k)]
        if missing:
            bad.append(f"{c.get('chunk_id')}: missing metadata {missing}")
    if len({c["document_id"] for c in chunks if c.get("document_id")}) != info.get("documents"):
        bad.append("document count differs from the index manifest")
    for name, digest in info.get("files", {}).items():
        p = kb_dir / name
        if not p.exists():
            bad.append(f"source file missing: {name}")
        elif sha256_file(p) != digest:
            bad.append(f"source file changed since indexing: {name}")
    for d in json.loads((kb_dir / "manifest.json").read_text(encoding="utf-8"))["documents"]:
        if (kb_dir / d["filename"]).exists() and d["filename"] not in info.get("files", {}):
            bad.append(f"new document not indexed yet: {d['filename']}")
    return bad


class KnowledgeBase:
    def __init__(self, cfg, embedder):
        self.cfg, self.embedder = cfg, embedder
        self.index, self.chunks, self.info = load_index(cfg.paths.index_dir)
        self.problems = validate_index(self.index, self.chunks, self.info, cfg.paths.kb_dir, embedder.dim)
        if self.problems:
            raise RuntimeError("Index validation failed: " + "; ".join(self.problems))
        self.bm25 = BM25([tokenize(chunk_embed_text(c)) for c in self.chunks])

    def search(self, query, role="customer", top_k=None, filters=None):
        r = self.cfg.retrieval
        top_k = top_k or r.top_k
        allowed = ACCESS.get(role, ACCESS["customer"])
        today = date.today().isoformat()

        # PERMISSION + METADATA gates run BEFORE ranking, so blocked chunks can never be retrieved.
        stats, ok = Counter(), []
        for i, c in enumerate(self.chunks):
            if c["access_level"] not in allowed:
                stats["blocked_permission"] += 1
            elif c["status"] != "active":
                stats["blocked_status"] += 1
            elif c.get("effective_date", "") > today:
                stats["blocked_not_yet_effective"] += 1
            elif filters and any(c.get(k) != v for k, v in filters.items()):
                stats["blocked_filter"] += 1
            else:
                ok.append(i)
        if not ok:
            return Retrieval([], False, dict(stats))

        # KB is small (<5k chunks): score everything exactly. At scale use IDSelector / Qdrant payload filters.
        scores, ids = self.index.search(self.embedder.encode_query(query), self.index.ntotal)
        dense = {int(i): float(s) for s, i in zip(scores[0], ids[0]) if i >= 0}
        stats["top_dense"] = round(max(dense[i] for i in ok), 3)
        dense_rank = sorted(ok, key=lambda i: -dense[i])[: r.candidates]
        lexical = self.bm25.scores(tokenize(query))
        lex_rank = [i for i in sorted(ok, key=lambda i: -lexical[i]) if lexical[i] > 0][: r.candidates]

        fused = Counter()                                    # reciprocal-rank fusion
        for ranking in (dense_rank, lex_rank if r.mode == "hybrid" else []):
            for rank, i in enumerate(ranking, 1):
                fused[i] += 1.0 / (60 + rank)
        ranked = [i for i, _ in fused.most_common() if dense[i] >= r.similarity_threshold]   # RELEVANT EVIDENCE gate
        stats["below_threshold"] = len(fused) - len(ranked)
        hits = [Hit(f"S{n}", self.chunks[i], dense[i], float(lexical[i]), fused[i]) for n, i in enumerate(ranked[:top_k], 1)]
        return Retrieval(hits, len(hits) >= r.min_evidence_chunks, dict(stats))
