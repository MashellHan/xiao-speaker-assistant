"""Bundled static assets for the Nomi Web Control Console."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_WEB_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def load_console_html() -> str:
    return (_WEB_DIR / "console.html").read_text(encoding="utf-8")
