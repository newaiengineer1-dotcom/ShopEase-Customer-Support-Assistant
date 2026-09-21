"""Calibrate similarity_threshold and run the golden set.    python scripts/evaluate.py [--full]"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agent import Session, SupportAgent  # noqa: E402
from src.config import load_settings  # noqa: E402
from src.embeddings import get_embedder  # noqa: E402
from src.ingest import ensure_index  # noqa: E402
from src.retrieval import KnowledgeBase  # noqa: E402
from src.tools import SupportTools  # noqa: E402

cfg = load_settings()
emb = get_embedder(cfg)
ensure_index(cfg, emb)
kb = KnowledgeBase(cfg, emb)
golden = json.loads((ROOT / "tests" / "golden_questions.json").read_text(encoding="utf-8"))

pos, neg = [], []
print(f"{'top cosine':>10}  {'doc@k':<6} question")
for g in golden:
    if g.get("customer"):
        continue
    r = kb.search(g["q"])
    top = r.stats.get("top_dense", 0.0)
    (neg if g.get("refuse") else pos).append(top)
    found = "-" if g.get("refuse") else ("yes" if any(h.chunk["document_id"] == g["doc"] for h in r.hits) else "NO")
    print(f"{top:10.3f}  {found:<6} {g['q']}")
if pos and neg:
    lo, hi = min(pos), max(neg)
    print(f"\nlowest in-scope score {lo:.3f} | highest out-of-scope score {hi:.3f}")
    print(f"suggested similarity_threshold ~ {(lo + hi) / 2:.2f}" if lo > hi
          else "WARNING: scores overlap; improve the KB/chunking before raising the threshold")

if "--full" in sys.argv:
    from src.llm import LLM

    agent = SupportAgent(cfg, kb, LLM(cfg), SupportTools(cfg))
    ok = 0
    for g in golden:
        res = agent.handle(g["q"], [], Session(g.get("customer")))
        refused = res.status == "refused"
        passed = refused if g.get("refuse") else (not refused and all(s.lower() in res.answer.lower() for s in g.get("must_contain", [])))
        ok += passed
        print("PASS" if passed else "FAIL", res.status, "|", g["q"])
        time.sleep(2)                                   # be gentle with free-tier rate limits
    print(f"\n{ok}/{len(golden)} passed")
