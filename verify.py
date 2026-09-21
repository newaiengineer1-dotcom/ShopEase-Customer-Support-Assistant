"""CLAIM/EVIDENCE CHECK -> CITATION CHECK -> POST-GENERATION VERIFICATION (deterministic first, optional LLM judge)."""
from __future__ import annotations

import re

from .utils import EMAIL_RE, PHONE_RE, tokenize

LABEL = re.compile(r"\[((?:S|T)\d+)\]")
NUM = re.compile(r"\d+(?:[.,]\d+)*")
URL = re.compile(r"https?://[^\s)\]]+|www\.[^\s)\]]+", re.I)
COURTESY = re.compile(
    r"^(sure|of course|certainly|thanks|thank you|i'm sorry|i am sorry|sorry|happy to|glad to|hello|hi|hey|is there anything|"
    r"let me know|would you like|do you want|please (let me know|confirm|sign in|press|use)|if you('d| would) like)\b", re.I)
ACTION_DONE = re.compile(
    r"\b(?:i|we)(?:'ve| have)\s+(?:just\s+|already\s+)?(?:created|submitted|cancel+ed|processed|issued|refunded|escalated|opened|initiated|updated|changed|arranged)\b"
    r"|\b(?:your|the)\s+(?:return|request|ticket|refund|cancellation)(?:\s+request)?\s+(?:has|have)\s+been\s+(?:created|submitted|cancel+ed|processed|issued|opened)\b",
    re.I)

VERIFIER_SYSTEM = """You are a strict fact-checker for a customer-support assistant.
Decide whether EVERY factual statement in ANSWER is fully supported by EVIDENCE.
- Use only EVIDENCE; no outside knowledge.
- Ignore courtesy phrases, questions and the bracketed labels such as [S1].
- A statement is unsupported if it adds, changes or contradicts a fact, number, date, condition or promise.
Return ONLY JSON: {"supported": true|false, "unsupported_claims": ["..."]}"""


def evidence_map(hits, results):
    m = {h.label: h.chunk["text"] for h in hits}
    m.update({r.label: r.evidence() for r in results})
    return m


def format_evidence(hits, results):
    blocks = [f"[{h.label}] {h.chunk['title']} > {h.chunk['section']} (v{h.chunk['version']})\n{h.chunk['text']}" for h in hits]
    blocks += [f"[{r.label}] system tool result\n{r.evidence()}" for r in results]
    return "\n\n".join(blocks)


def _norm(tok):
    f = float(tok.replace(",", ""))
    return str(int(f)) if f == int(f) else str(f)


def _numbers(text):
    text = re.sub(r"(?m)^\s*\d+[.)]\s+", "", LABEL.sub(" ", text))       # ignore citation labels and list numbering
    out = set()
    for tok in NUM.findall(text):
        try:
            out.add(_norm(tok))
        except ValueError:
            pass
    return out


def _coverage(body, support):
    words = {w for w in tokenize(body) if not w.isdigit()}
    return len(words & set(tokenize(support))) / len(words) if words else 1.0


def _sentences(text):
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+|\n+", text) if p and p.strip()]
    merged = []
    for p in parts:
        if merged and not LABEL.sub("", p).strip(" .-*•"):
            merged[-1] += " " + p                        # a dangling "[S1]" belongs to the previous sentence
        else:
            merged.append(p)
    return [m.lstrip("-*• ").strip() for m in merged]


def verify_answer(answer, hits, results, min_coverage=0.4):
    ev = evidence_map(hits, results)
    known = set(ev)
    cited = set(LABEL.findall(answer))
    issues = []
    if not cited:
        issues.append("no citations")
    for lab in sorted(cited - known):                                     # CITATION CHECK
        issues.append(f"citation [{lab}] does not exist in the retrieved evidence")
    for s in _sentences(answer):                                          # CLAIM/EVIDENCE CHECK
        body = LABEL.sub("", s).strip()
        if body.endswith(("?", ":")) or COURTESY.match(body) or len([w for w in tokenize(body) if not w.isdigit()]) <= 1:
            continue                                                      # question / courtesy / list intro: no claim
        labs = set(LABEL.findall(s)) & known
        if not labs:
            issues.append(f"uncited statement: '{body[:80]}'")
            continue
        support = " ".join(ev[label] for label in labs)
        missing = _numbers(body) - _numbers(support)
        if missing:
            issues.append(f"numbers {sorted(missing)} are not in the cited evidence: '{body[:80]}'")
        elif _coverage(body, support) < min_coverage:
            issues.append(f"statement not supported by its citation: '{body[:80]}'")
    all_ev = " ".join(ev.values())                                        # POST-GENERATION checks
    for kind, pat in (("URL", URL), ("email", EMAIL_RE), ("phone number", PHONE_RE)):
        for m in pat.findall(answer):
            if m.rstrip(".,") not in all_ev:
                issues.append(f"{kind} not present in evidence: {m}")
    if ACTION_DONE.search(answer):
        issues.append("claims an action was completed; actions only run after the customer confirms")
    return {"passed": not issues, "issues": issues, "cited": sorted(cited & known)}


def llm_verify(llm, answer, evidence_text):
    j = llm.chat_json(VERIFIER_SYSTEM, f"EVIDENCE:\n{evidence_text}\n\nANSWER:\n{answer}", fast=True)
    claims = [str(c) for c in (j.get("unsupported_claims") or [])][:3]
    return {"supported": str(j.get("supported")).lower() == "true" and not claims, "unsupported_claims": claims}
