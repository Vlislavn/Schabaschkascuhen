"""Tiny data-driven i18n. Locale catalogs live as JSON in schabasch/locales/*.json — translations
are DATA, not code: adding a language is dropping a `<code>.json` file, zero code change. stdlib only.

Keep the signature term «шабашка» literal in every locale (it's the product's brand word); the EN
catalog glosses it on first use. Everything else translates by key.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

DEFAULT_LANG = "en"
_LOCALES_DIR = Path(__file__).resolve().parent / "locales"


@lru_cache(maxsize=1)
def _catalogs() -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for p in sorted(_LOCALES_DIR.glob("*.json")):
        out[p.stem] = json.loads(p.read_text(encoding="utf-8"))
    return out


def available_langs() -> list[str]:
    """Languages discovered from the locales dir, DEFAULT_LANG first (drives the UI toggle order)."""
    langs = sorted(_catalogs())
    if DEFAULT_LANG in langs:
        return [DEFAULT_LANG] + [l for l in langs if l != DEFAULT_LANG]
    return langs


def normalize_lang(lang: str | None) -> str:
    """Coerce a (possibly None / unknown) lang to a valid available one, else DEFAULT_LANG."""
    return lang if lang in _catalogs() else DEFAULT_LANG


def t(lang: str, key: str, /, **kw) -> str:
    """Translate `key` for `lang`; fall back to DEFAULT_LANG, then to the key itself.
    `**kw` are `str.format` params (templates use `{name}` placeholders).
    `lang`/`key` are positional-only so a template can take a `{lang}` placeholder (e.g. for links)."""
    cats = _catalogs()
    s = cats.get(lang, {}).get(key)
    if s is None:
        s = cats.get(DEFAULT_LANG, {}).get(key, key)
    return s.format(**kw) if kw else s
