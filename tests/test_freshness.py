"""Publication-date freshness gate: too_old + max_post_age_days + the ingestion skip wiring
(regression for the Arbeitsagentur stale-flood — a 14-month-old job reaching the pool)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from schabasch import freshness


def _iso(days_ago: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).date().isoformat()


def test_too_old_keeps_unprovable():
    assert freshness.too_old(None, 2) is False        # no date → keep (LinkedIn / half of Indeed)
    assert freshness.too_old("", 2) is False
    assert freshness.too_old("not-a-date", 2) is False
    assert freshness.too_old(_iso(0), 2) is False     # today
    assert freshness.too_old(_iso(1), 2) is False     # within window
    assert freshness.too_old(_iso(5), 0) is False     # window 0 disables the gate (never drop all)


def test_too_old_drops_stale():
    assert freshness.too_old(_iso(5), 2) is True
    assert freshness.too_old("2025-04-01", 2) is True   # the 14-month AA job the user hit
    # full ISO timestamp (AA aktuelleVeroeffentlichungsdatum can carry time)
    ts10 = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
    assert freshness.too_old(ts10, 7) is True
    ts3 = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
    assert freshness.too_old(ts3, 7) is False


def test_max_post_age_days():
    assert freshness.max_post_age_days({"search": {"max_post_age_days": 5}}) == 5   # explicit
    assert freshness.max_post_age_days({"search": {"hours_old": 48}}) == 2          # ceil(48/24)
    assert freshness.max_post_age_days({"search": {"hours_old": 30}}) == 2          # ceil(30/24)
    assert freshness.max_post_age_days({"search": {}}) == 3                         # default
    assert freshness.max_post_age_days({}) == 3


def test_aa_ingestion_skips_stale(con, cfg, monkeypatch):
    """arbeitsagentur.search drops a stale row at the door and keeps the fresh one (the fix)."""
    from schabasch.sources import arbeitsagentur
    monkeypatch.setattr("time.sleep", lambda *a, **k: None)   # the source throttles 1s/page
    cfg = dict(cfg)
    cfg["search"] = dict(cfg["search"], max_post_age_days=2, cities=["Frankfurt am Main, Germany"])
    payload = {"maxErgebnisse": 2, "stellenangebote": [
        {"refnr": "FRESH1", "titel": "Fresh", "arbeitgeber": "Co",
         "arbeitsort": {"ort": "Frankfurt"}, "aktuelleVeroeffentlichungsdatum": _iso(0)},
        {"refnr": "STALE1", "titel": "Stale", "arbeitgeber": "Co",
         "arbeitsort": {"ort": "Frankfurt"}, "aktuelleVeroeffentlichungsdatum": "2025-04-01"},
    ]}
    calls = {"n": 0}

    def fake_get(url, **kw):
        calls["n"] += 1
        return payload if calls["n"] == 1 else {"stellenangebote": []}

    monkeypatch.setattr(arbeitsagentur, "http_get_json", fake_get)
    arbeitsagentur.search(cfg, con, queries=["aerospace"])
    urls = [r[0] for r in con.execute("SELECT url FROM vacancy").fetchall()]
    assert any("FRESH1" in u for u in urls)
    assert not any("STALE1" in u for u in urls)
    # the drop is observable, not silent
    skipped = con.execute(
        "SELECT count FROM funnel_log WHERE stage='ingest_stale_skip' AND source='arbeitsagentur'"
    ).fetchone()
    assert skipped and skipped[0] == 1
