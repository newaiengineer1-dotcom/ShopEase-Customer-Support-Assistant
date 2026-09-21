"""Settings: tunables from config/settings.toml, secrets from .env / environment."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


def _ns(d):
    return SimpleNamespace(**{k: _ns(v) if isinstance(v, dict) else v for k, v in d.items()})


def load_settings(path=None):
    load_dotenv(ROOT / ".env")
    with open(path or ROOT / "config" / "settings.toml", "rb") as f:
        cfg = _ns(tomllib.load(f))
    for k, v in list(vars(cfg.paths).items()):
        setattr(cfg.paths, k, ROOT / v)
    return cfg
