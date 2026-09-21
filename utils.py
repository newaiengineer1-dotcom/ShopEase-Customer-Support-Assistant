"""Shared text helpers."""
from __future__ import annotations

import hashlib
import re

STOPWORDS = set("""a an the is are was were be been am do does did i me my we our you your it its to of in on at for from by with
and or but if so as that this these those can could will would should may might must have has had not no how what when where which
who whom why many much any some there their they them he she his her than then also up out about into over after before per
please yes just""".split())

CARD_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
PHONE_RE = re.compile(r"(?:\+?\d{1,3}[ .-]?)?(?:\(\d{3}\)|\d{3})[ .-]\d{3}[ .-]\d{4}")


def _stem(t):
    if len(t) <= 3:
        return t
    if t.endswith("ies") and len(t) > 4:
        t = t[:-3] + "y"
    elif t.endswith(("sses", "shes", "ches", "xes")):
        t = t[:-2]
    elif t.endswith("s") and not t.endswith(("ss", "us", "is")):
        t = t[:-1]
    for suf in ("ing", "ed"):
        if t.endswith(suf) and len(t) - len(suf) >= 3:
            t = t[: -len(suf)]
            if len(t) > 3 and t[-1] == t[-2] and t[-1] not in "sz":
                t = t[:-1]
            break
    return t


def tokenize(text):
    """Lower-case, drop stop-words, light stemming. Shared by BM25 and the claim checker."""
    return [_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 1 and w not in STOPWORDS]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def chunk_embed_text(c):
    """Text that is embedded / BM25-indexed: title + section give each chunk its context."""
    return f"{c['title']}. {c['section']}. {c['text']}"
