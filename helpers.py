import re
from datetime import date

TODAY = date(2026, 9, 21)


class FakeLLM:
    """Scripted stand-in: one callable(system, user) per role (planner / generator / verifier)."""

    def __init__(self, planner, generator, verifier=None):
        self.planner, self.generator = planner, generator
        self.verifier = verifier or (lambda system, user: {"supported": True, "unsupported_claims": []})

    def chat_json(self, system, user, fast=False):
        if "planning module" in system:
            return self.planner(system, user)
        if "fact-checker" in system:
            return self.verifier(system, user)
        return self.generator(system, user)


def label_for(user_prompt, needle):
    """Label ([S#]/[T#]) of the evidence block that contains `needle`."""
    for m in re.finditer(r"\[([ST]\d+)\][^\n]*\n(.*?)(?=\n\n\[[ST]\d+\]|\Z)", user_prompt, re.S):
        if needle in m.group(2):
            return m.group(1)
    raise AssertionError(f"'{needle}' not found in the evidence")
