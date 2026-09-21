"""Build the FAISS index from knowledge_base/manifest.json.     python -m src.ingest [--check]"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import faiss

from .config import load_settings
from .embeddings import get_embedder
from .retrieval import load_index, validate_index
from .utils import chunk_embed_text, sha256_file

MD_HEADING = re.compile(r"^#{1,4}\s+(.*\S)\s*$")
REQUIRED = ("document_id", "filename", "title", "category", "version", "effective_date", "access_level", "status", "source")


def _split_headings(text, txt_mode=False):
    section, buf = "Overview", []
    for line in text.splitlines():
        m = MD_HEADING.match(line)
        caps = txt_mode and 3 < len(line.strip()) < 70 and line.strip().isupper()   # ALL-CAPS lines = headings in .txt
        if m or caps:
            if buf:
                yield section, None, "\n".join(buf)
            section, buf = (m.group(1) if m else line.strip().title()), []
        else:
            buf.append(line)
    if buf:
        yield section, None, "\n".join(buf)


def load_units(path):
    """Yield (section, page, text) from a source file."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext in {".md", ".txt"}:
        yield from _split_headings(path.read_text(encoding="utf-8"), txt_mode=(ext == ".txt"))
    elif ext == ".pdf":
        from pypdf import PdfReader

        for n, page in enumerate(PdfReader(str(path)).pages, 1):
            text = page.extract_text() or ""
            if text.strip():
                yield f"Page {n}", n, text
    elif ext == ".docx":
        from docx import Document

        section, buf = "Overview", []
        for p in Document(str(path)).paragraphs:
            if p.style is not None and p.style.name.lower().startswith(("heading", "title")):
                if buf:
                    yield section, None, "\n".join(buf)
                section, buf = p.text.strip() or section, []
            elif p.text.strip():
                buf.append(p.text.strip())
        if buf:
            yield section, None, "\n".join(buf)
    else:
        raise ValueError(f"Unsupported file type: {path.name}")


def window(text, size, overlap):
    """Sentence-aware windows of ~size words with ~overlap words carried over."""
    pieces = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s and s.strip()]
    chunks, cur, n = [], [], 0
    for p in pieces:
        w = len(p.split())
        if cur and n + w > size:
            chunks.append(" ".join(cur))
            keep, m = [], 0
            for q in reversed(cur):
                qw = len(q.split())
                if m + qw > overlap:
                    break
                keep.insert(0, q)
                m += qw
            cur, n = keep, m
        cur.append(p)
        n += w
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def build_index(cfg, embedder=None):
    embedder = embedder or get_embedder(cfg)
    kb_dir, out = Path(cfg.paths.kb_dir), Path(cfg.paths.index_dir)
    man = json.loads((kb_dir / "manifest.json").read_text(encoding="utf-8"))
    docs = [{**man.get("defaults", {}), **d} for d in man["documents"]]   # only registered files are ever ingested
    chunks, files = [], {}
    for d in docs:
        for k in REQUIRED:
            if k not in d:
                raise ValueError(f"manifest entry '{d.get('filename')}' lacks '{k}'")
        path = kb_dir / d["filename"]
        if not path.exists():
            if d.get("optional"):
                continue
            raise FileNotFoundError(f"Registered document not found: {path}")
        files[d["filename"]] = sha256_file(path)
        n = 0
        for section, page, text in load_units(path):
            for piece in window(text, cfg.kb.chunk_size_words, cfg.kb.chunk_overlap_words):
                if len(piece.split()) < cfg.kb.min_chunk_words:
                    continue
                n += 1
                chunks.append({
                    "chunk_id": f"{d['document_id']}-C{n:03d}", "document_id": d["document_id"], "filename": d["filename"],
                    "title": d["title"], "section": section, "page": page, "source": d["source"], "version": d["version"],
                    "effective_date": d["effective_date"], "category": d["category"], "access_level": d["access_level"],
                    "status": d["status"], "text": piece})
    if not chunks:
        raise RuntimeError("No chunks produced; check knowledge_base/manifest.json")
    vecs = embedder.encode_documents([chunk_embed_text(c) for c in chunks])
    index = faiss.IndexFlatIP(int(vecs.shape[1]))     # vectors are L2-normalised, so inner product = cosine
    index.add(vecs)
    out.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out / "faiss.index"))
    (out / "chunks.jsonl").write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in chunks), encoding="utf-8")
    info = {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "embedding_model": "hash" if cfg.embedding.provider == "hash" else cfg.embedding.model,
            "dim": int(vecs.shape[1]), "documents": len({c["document_id"] for c in chunks}), "chunks": len(chunks), "files": files}
    (out / "index_manifest.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    problems = validate_index(index, chunks, info, kb_dir, embedder.dim)
    if problems:
        raise RuntimeError("Index validation failed: " + "; ".join(problems))
    return info


def ensure_index(cfg, embedder):
    """Reuse the index if it is valid, otherwise (re)build it. Handy on first deploy."""
    d = Path(cfg.paths.index_dir)
    if (d / "faiss.index").exists():
        try:
            idx, chunks, info = load_index(d)
            if not validate_index(idx, chunks, info, cfg.paths.kb_dir, embedder.dim):
                return
        except Exception:
            pass
    build_index(cfg, embedder)


if __name__ == "__main__":
    _cfg = load_settings()
    _emb = get_embedder(_cfg)
    if "--check" in sys.argv:
        _idx, _chunks, _info = load_index(_cfg.paths.index_dir)
        _bad = validate_index(_idx, _chunks, _info, _cfg.paths.kb_dir, _emb.dim)
        print("Index OK" if not _bad else "\n".join(_bad))
        sys.exit(1 if _bad else 0)
    print(json.dumps(build_index(_cfg, _emb), indent=2))
