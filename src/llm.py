"""Provider-agnostic JSON chat over any OpenAI-compatible endpoint."""
from __future__ import annotations

import json
import os
import re

from openai import BadRequestError, OpenAI

PROVIDERS = {
    # Only the Groq defaults were verified (2026-09-21). For other providers set LLM_MODEL to a current model ID.
    "groq":   {"base_url": "https://api.groq.com/openai/v1", "key_env": "GROQ_API_KEY", "model": "openai/gpt-oss-120b", "fast": "openai/gpt-oss-20b"},
    "openai": {"base_url": "https://api.openai.com/v1", "key_env": "OPENAI_API_KEY", "model": None, "fast": None},
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/", "key_env": "GEMINI_API_KEY", "model": None, "fast": None},
    "ollama": {"base_url": "http://localhost:11434/v1", "key_env": "", "model": None, "fast": None},
}


def parse_json(text):
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise
        return json.loads(m.group(0))


class LLM:
    def __init__(self, cfg):
        name = os.getenv("LLM_PROVIDER", "groq").lower()
        if name not in PROVIDERS:
            raise RuntimeError(f"Unknown LLM_PROVIDER '{name}'. Use one of: {', '.join(PROVIDERS)}")
        p = PROVIDERS[name]
        key = os.getenv("LLM_API_KEY") or (os.getenv(p["key_env"]) if p["key_env"] else "ollama")
        if not key:
            raise RuntimeError(f"{p['key_env']} is not set. Copy .env.example to .env (or add it to your host's secrets).")
        self.model = os.getenv("LLM_MODEL") or p["model"]
        if not self.model:
            raise RuntimeError(f"Set LLM_MODEL for provider '{name}' (no verified default is shipped).")
        self.model_fast = os.getenv("LLM_MODEL_FAST") or p["fast"] or self.model
        self.provider, self.cfg = name, cfg.llm
        self.json_mode = os.getenv("LLM_JSON_MODE", "auto").lower()
        self.client = OpenAI(api_key=key, base_url=os.getenv("LLM_BASE_URL", p["base_url"]),
                             timeout=cfg.llm.timeout_s, max_retries=2)     # SDK backs off on 429/5xx
        self._skip = 0                                                       # remembers which request variant works

    def chat_json(self, system, user, fast=False):
        model = self.model_fast if fast else self.model
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        t, n = self.cfg.temperature, self.cfg.max_tokens
        reasoning = "gpt-oss" in model.lower()
        effort = os.getenv("LLM_REASONING_EFFORT", "low" if reasoning else "")
        extra = {"extra_body": {"reasoning_effort": effort}} if effort else {}
        js = {"response_format": {"type": "json_object"}} if (self.json_mode != "off" and not reasoning) else {}
        attempts = [
            {"temperature": t, "max_tokens": n, **js},
            {"temperature": t, "max_tokens": n},
            {"max_completion_tokens": n, **js},          # newer OpenAI models
            {"max_completion_tokens": n},
        ]
        last = None
        for i in range(self._skip, len(attempts)):
            try:
                r = self.client.chat.completions.create(model=model, messages=messages, **attempts[i], **extra)
            except BadRequestError as e:                 # parameter unsupported by this model/provider: try next variant
                last = e
                continue
            self._skip = i
            return parse_json(r.choices[0].message.content or "")
        raise last
