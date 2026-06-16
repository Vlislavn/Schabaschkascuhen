"""Fuzzy dedup: guard regression tests (no network, no LLM).

Each guard has its own test per the aislop detector-loop discipline: one test per
disqualifier, one test for genuine near-dup, one for same-source skip.
"""
from __future__ import annotations

import json

import pytest

from schabasch import db
from schabasch.dedup import SIMILARITY_THRESHOLD, _disqualified, find_fuzzy_candidates, dedup_fuzzy
from schabasch.models import Status


def _insert(con, *, url, title, company, source="indeed", refnr=None, city="Frankfurt",
            card_json=None, description="x" * 300, status=Status.DESCRIBED):
    vid = db.upsert_vacancy(con, {"source": source, "url": url, "title": title,
                                  "company": company, "city": city, "refnr": refnr,
                                  "description": description})
    if card_json:
        db.set_status(con, vid, status, card_json=json.dumps(card_json))
    else:
        db.set_status(con, vid, status)
    return vid


# ─── unit tests for _disqualified guard stack ─────────────────────────────────

def test_disqualified_different_refnr():
    a = {"refnr": "REF-001", "city": "Frankfurt", "card_json": None}
    b = {"refnr": "REF-002", "city": "Frankfurt", "card_json": None}
    assert _disqualified(a, b) == "different_refnr"


def test_disqualified_different_city():
    a = {"refnr": None, "city": "Frankfurt", "card_json": None}
    b = {"refnr": None, "city": "Berlin", "card_json": None}
    assert _disqualified(a, b) == "different_city"


def test_disqualified_different_language_reality():
    card_de = json.dumps({"language_reality": "de"})
    card_en = json.dumps({"language_reality": "en"})
    a = {"refnr": None, "city": "Frankfurt", "card_json": card_de}
    b = {"refnr": None, "city": "Frankfurt", "card_json": card_en}
    assert _disqualified(a, b) == "different_language_reality"


def test_not_disqualified_same_city_same_lang():
    card = json.dumps({"language_reality": "en"})
    a = {"refnr": None, "city": "Frankfurt", "card_json": card}
    b = {"refnr": None, "city": "Frankfurt", "card_json": card}
    assert _disqualified(a, b) is None


def test_not_disqualified_unknown_city_passes():
    """Unknown city (empty string) should not trigger city guard — we can't confirm difference."""
    a = {"refnr": None, "city": "", "card_json": None}
    b = {"refnr": None, "city": "Frankfurt", "card_json": None}
    assert _disqualified(a, b) is None


# ─── integration tests via find_fuzzy_candidates ──────────────────────────────

def test_same_source_skipped(con, cfg):
    """Two indeed rows with the same title → NOT a candidate (same-source exact dedup handles it)."""
    _insert(con, url="u/1", title="Process Engineer", company="ACME GmbH", source="indeed")
    _insert(con, url="u/2", title="Process Engineer (m/w/d)", company="ACME GmbH", source="indeed")
    candidates = find_fuzzy_candidates(con, threshold=SIMILARITY_THRESHOLD)
    assert candidates == []


def test_cross_source_near_dup_logged(con, cfg):
    """Same company + near-identical title on indeed vs linkedin → candidate logged."""
    _insert(con, url="u/1", title="Senior Process Engineer", company="ACME GmbH", source="indeed",
            city="Frankfurt")
    _insert(con, url="u/2", title="Senior Process Engineer (m/w/d)", company="ACME GmbH",
            source="linkedin", city="Frankfurt")
    candidates = find_fuzzy_candidates(con, threshold=SIMILARITY_THRESHOLD)
    assert len(candidates) == 1
    assert candidates[0]["source_a"] != candidates[0]["source_b"]
    assert candidates[0]["similarity"] >= SIMILARITY_THRESHOLD


def test_different_refnr_guard_wins(con, cfg):
    """AA rows with different refnrs are never flagged even if titles match."""
    _insert(con, url="u/1", title="Ingenieur Maschinenbau", company="ACME GmbH",
            source="arbeitsagentur", refnr="REF-001", city="Frankfurt")
    _insert(con, url="u/2", title="Ingenieur Maschinenbau (m/w/d)", company="ACME GmbH",
            source="linkedin", refnr="REF-002", city="Frankfurt")
    candidates = find_fuzzy_candidates(con, threshold=SIMILARITY_THRESHOLD)
    assert candidates == []


def test_different_city_guard_wins(con, cfg):
    """Same company + title, different cities → not a near-dup (different locations)."""
    _insert(con, url="u/1", title="Data Engineer", company="SAP SE", source="indeed",
            city="Heidelberg")
    _insert(con, url="u/2", title="Data Engineer", company="SAP SE", source="linkedin",
            city="Berlin")
    candidates = find_fuzzy_candidates(con, threshold=SIMILARITY_THRESHOLD)
    assert candidates == []


def test_dedup_fuzzy_logs_to_funnel(con, cfg):
    """dedup_fuzzy() logs to funnel_log regardless of whether candidates found."""
    result = dedup_fuzzy(cfg, con)
    assert "candidates" in result
    row = con.execute("SELECT count FROM funnel_log WHERE stage='dedup_fuzzy'").fetchone()
    assert row is not None
    assert row["count"] == result["candidates"]


def test_dedup_fuzzy_no_status_mutation(con, cfg):
    """Logged candidates must NOT change vacancy status."""
    _insert(con, url="u/1", title="Senior ML Engineer", company="ACME GmbH", source="indeed",
            city="Frankfurt")
    _insert(con, url="u/2", title="Senior ML Engineer (f/m/d)", company="ACME GmbH",
            source="linkedin", city="Frankfurt")
    before = {r["url"]: r["status"] for r in
              con.execute("SELECT url, status FROM vacancy").fetchall()}
    dedup_fuzzy(cfg, con)
    after = {r["url"]: r["status"] for r in
             con.execute("SELECT url, status FROM vacancy").fetchall()}
    assert before == after  # no status mutations
