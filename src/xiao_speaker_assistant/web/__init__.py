"""Bundled static assets for the Nomi Web Control Console.

Source files live cleanly separated as `index.html`, `console.css`, and `console.js`.
`load_console_html()` composes them into a single self-contained HTML document so
remote console tunnels (which allowlist `/`, `/health`, `/api/*`, and `/assets/*`)
work reliably even against older relay binaries without extra round-trips.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_WEB_DIR = Path(__file__).resolve().parent
_CSS_PLACEHOLDER = "/* INLINE_CSS_PLACEHOLDER */"
_JS_PLACEHOLDER = "// INLINE_JS_PLACEHOLDER"


@lru_cache(maxsize=1)
def load_console_css() -> str:
    return (_WEB_DIR / "console.css").read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def load_console_js() -> str:
    return (_WEB_DIR / "console.js").read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def load_console_html() -> str:
    template = (_WEB_DIR / "index.html").read_text(encoding="utf-8")
    return template.replace(_CSS_PLACEHOLDER, load_console_css(), 1).replace(
        _JS_PLACEHOLDER, load_console_js(), 1
    )
