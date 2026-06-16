"""W4: authoritative company validation (Wikipedia keyless + German legal-suffix registry signal),
independent of the qwen agent. Wikipedia is mocked — no network."""
from __future__ import annotations

from schabasch import investigate as I


def test_german_rooted_from_legal_suffix(monkeypatch):
    monkeypatch.setattr(I, "_wikipedia_company", lambda name, lang="de": None)
    v = I.validate_company("Da Vinci Engineering GmbH", {}, cache={})
    assert v["german_rooted"] is True and v["validation_source"] == "legal-suffix"
    assert v["company_verified"] is True


def test_wikipedia_validation_and_description(monkeypatch):
    monkeypatch.setattr(I, "_wikipedia_company", lambda name, lang="de": {
        "found": True, "title": "Merck KGaA", "lang": "de", "url": "http://x",
        "extract": "Die Merck KGaA ist ein deutsches Unternehmen mit Sitz in Darmstadt."})
    v = I.validate_company("Merck KGaA", {}, cache={})
    assert v["company_verified"] is True and v["german_rooted"] is True   # "deutsches"/"Darmstadt"
    assert "Darmstadt" in v["company_description"] and v["validation_source"] == "wikipedia:de"


def test_us_company_verified_but_not_german_rooted(monkeypatch):
    monkeypatch.setattr(I, "_wikipedia_company", lambda name, lang="de": {
        "found": True, "title": "Westinghouse", "lang": "de", "url": None,
        "extract": "Die Westinghouse Electric Company ist ein US-amerikanischer Hersteller."})
    v = I.validate_company("Westinghouse Electric Company", {}, cache={})
    assert v["company_verified"] is True and v["german_rooted"] is False


def test_unknown_company_conservative_falls_back_to_agent(monkeypatch):
    monkeypatch.setattr(I, "_wikipedia_company", lambda name, lang="de": None)
    v = I.validate_company("Tiny Unknown Startup", {"company_description": "agent says X"}, cache={})
    assert v["company_verified"] is False and v["validation_source"] == "none"
    assert v["company_description"] == "agent says X"


def test_validation_cached_per_company(monkeypatch):
    calls = {"n": 0}
    def fake(name, lang="de"):
        calls["n"] += 1
        return None
    monkeypatch.setattr(I, "_wikipedia_company", fake)
    cache: dict = {}
    I.validate_company("ACME GmbH", {}, cache=cache)
    I.validate_company("ACME GmbH", {}, cache=cache)   # same normalized company → cache hit
    assert calls["n"] <= 2   # de+en at most once; second call hits the cache (0 new lookups)
