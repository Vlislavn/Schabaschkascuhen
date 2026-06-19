"""Keyless HTML→clean-markdown extraction via trafilatura (already a dependency).

Adapter so a fetched company page yields signal-dense text for the tight agent budget instead of raw
HTML/boilerplate. Graceful-degrade: returns None if trafilatura is absent or extraction fails.

Ref: adbar/trafilatura (Apache); heavier JS/anti-bot pages → unclecode/crawl4ai (optional extra).
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def clean(html_or_url: str, *, is_url: bool = False, timeout_s: int = 15) -> str | None:
    """Extract the main content of a page as markdown. `is_url=True` fetches the URL first (keyless).
    Returns None on any failure (missing lib / network / nothing extractable) — never raises."""
    if not html_or_url:
        return None
    try:
        import trafilatura
    except Exception:
        logger.debug("trafilatura not installed — extract.clean is a no-op")
        return None
    try:
        source = html_or_url
        if is_url:
            source = trafilatura.fetch_url(html_or_url)
            if not source:
                return None
        return trafilatura.extract(source, output_format="markdown", include_links=False,
                                   include_comments=False) or None
    except Exception as exc:
        logger.debug("trafilatura extract failed: %s", exc)
        return None
