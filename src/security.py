"""Input guards, PII redaction and the audit log."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from .utils import CARD_RE, EMAIL_RE, PHONE_RE

_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_INJECTION = re.compile(
    r"(ignore|disregard|forget|override)\s+(all\s+|any\s+|the\s+|your\s+)?(previous|prior|above|earlier|system)\s+(instructions?|prompts?|rules?|messages?)"
    r"|(reveal|show|print|repeat|leak)\s+(me\s+)?(your\s+|the\s+)?(system|hidden|developer|initial)\s+(prompt|instructions?|message)"
    r"|developer\s+mode|jailbreak|do\s+anything\s+now",
    re.I,
)


def sanitize_input(text, max_chars):
    text = re.sub(r"\s+", " ", _CTRL.sub("", text or "")).strip()
    if not text:
        return "", "Please type your question."
    if len(text) > max_chars:
        return "", f"Your message is too long (limit {max_chars} characters). Please shorten it."
    return text, None


def looks_like_injection(text):
    return bool(_INJECTION.search(text))


def redact_secrets(text):
    """Remove card-like numbers before anything reaches the LLM or the logs."""
    new, n = CARD_RE.subn("[REDACTED-CARD]", text)
    return new, n > 0


def redact_pii(text):
    text = CARD_RE.sub("[CARD]", text)
    text = EMAIL_RE.sub("[EMAIL]", text)
    return PHONE_RE.sub("[PHONE]", text)


def audit(path, event):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **event}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
