"""Publication-date freshness gate — drop stale postings at ingestion (root fix for the
Arbeitsagentur "veröffentlichtseit" flood: the API returns 4+ day-old, sometimes 14-month-old rows
that no client-side date filter caught).

ONE rule, shared by every source: a row whose *publication* date (date_posted) is older than the
window is dropped at the door. A missing/unparseable date passes — half of Indeed and all of
LinkedIn carry no date, and we never drop on a date we can't prove stale (recall-first).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone


def max_post_age_days(cfg: dict) -> int:
    """Window in days for the ingestion gate. Explicit `search.max_post_age_days`, else derived from
    the scrape window `search.hours_old` (ceil to whole days), else a safe default of 3."""
    search = cfg.get("search", {}) or {}
    v = search.get("max_post_age_days")
    if v is not None:
        return int(v)
    hours = search.get("hours_old")
    if hours:
        return max(1, -(-int(hours) // 24))  # ceil(hours/24), stdlib-only
    return 3


def too_old(date_posted: str | None, max_age_days: int) -> bool:
    """True ONLY when date_posted parses to a date older than now - max_age_days. None / blank /
    unparseable → False (can't prove stale → keep). max_age_days <= 0 disables the gate (always
    False), so an empty/zero config never silently drops everything."""
    if max_age_days <= 0 or not date_posted:
        return False
    raw = str(date_posted).strip()
    if not raw:
        return False
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        # tolerate a bare date or a trailing 'Z' that fromisoformat (pre-3.11) rejects
        try:
            dt = datetime.fromisoformat(raw[:10])
        except ValueError:
            return False
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    return dt < cutoff


def _demo() -> None:
    today = datetime.now(timezone.utc)
    iso = lambda d: (today - timedelta(days=d)).date().isoformat()
    assert too_old(None, 2) is False                  # no date → keep
    assert too_old("", 2) is False                    # blank → keep
    assert too_old("not-a-date", 2) is False          # unparseable → keep
    assert too_old(iso(0), 2) is False                # today → keep
    assert too_old(iso(1), 2) is False                # 1 day → keep
    assert too_old(iso(5), 2) is True                 # 5 days, window 2 → drop
    assert too_old("2025-04-01", 2) is True           # the 14-month AA job → drop
    assert too_old(iso(5), 0) is False                # window 0 → gate disabled
    # full ISO timestamp (AA aktuelleVeroeffentlichungsdatum can carry time)
    assert too_old((today - timedelta(days=10)).isoformat(), 7) is True
    print("freshness._demo OK")


if __name__ == "__main__":
    _demo()
