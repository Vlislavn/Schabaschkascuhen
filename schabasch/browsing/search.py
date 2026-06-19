"""Keyless web search adapter: ddgs (DuckDuckGo, no infra) primary; optional self-hosted SearXNG
JSON endpoint when `searxng_url` is set (higher quality, multi-engine). Returns a list of
{title, url, snippet} or [] — never raises. schabasch depends on `search()`, not the backend.

Ref: searxng/searxng (JSON API), the `ddgs` package (DuckDuckGo). Keyless; no API key.
"""
from __future__ import annotations

import logging

from ..llm import http_get_json

logger = logging.getLogger(__name__)


def _searxng(query: str, searxng_url: str, max_results: int, timeout_s: int) -> list[dict] | None:
    try:
        js = http_get_json(searxng_url.rstrip("/") + "/search", timeout_s=timeout_s,
                           params={"q": query, "format": "json"})
    except Exception as exc:
        logger.debug("searxng search failed: %s", exc)
        return None
    res = (js or {}).get("results") or []
    return [{"title": r.get("title") or "", "url": r.get("url") or "", "snippet": r.get("content") or ""}
            for r in res[:max_results]]


def _ddgs(query: str, max_results: int) -> list[dict] | None:
    try:
        from ddgs import DDGS
    except Exception:
        logger.debug("ddgs not installed — search backend unavailable")
        return None
    try:
        with DDGS() as d:
            hits = d.text(query, max_results=max_results)
        return [{"title": h.get("title") or "", "url": h.get("href") or h.get("url") or "",
                 "snippet": h.get("body") or h.get("snippet") or ""} for h in (hits or [])]
    except Exception as exc:
        logger.debug("ddgs search failed: %s", exc)
        return None


def search(query: str, *, max_results: int = 5, searxng_url: str | None = None,
           timeout_s: int = 15) -> list[dict]:
    """Keyless web search → [{title, url, snippet}]. Prefers a configured SearXNG instance, falls back
    to ddgs (DuckDuckGo). Returns [] on any failure (never raises)."""
    if not query or not query.strip():
        return []
    if searxng_url:
        out = _searxng(query, searxng_url, max_results, timeout_s)
        if out:
            return out   # SearXNG up → prefer it; else fall through to keyless ddgs
    return _ddgs(query, max_results) or []
