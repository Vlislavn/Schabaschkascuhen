"""Keyless browsing adapters over IMPORTED packages — the stable boundary the rest of schabasch
depends on, not the underlying libs. Each adapter is keyless, time-bounded, and degrades to
``None``/empty on any failure (never raises — a flaky source must never break a tick). Swap a rotted
backend in one file without touching callers.

  entity.resolve(name)   → typed company entity via the Wikidata API (disambiguation fix)
  extract.clean(html|url) → clean markdown via trafilatura (signal-dense text for the tight agent)
  search.search(query)    → SearXNG / ddgs keyless web search            (added in a later step)
  registry.lookup(name)   → German company registry (legal form/status)  (added in a later step)

Imports are lazy per-module so importing this package stays cheap.
"""
