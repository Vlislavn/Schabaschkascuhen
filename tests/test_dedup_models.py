"""Дедуп-примитивы: нормализация компании/тайтла, ключ, content-hash."""
from __future__ import annotations

from schabasch.models import (Card, content_hash, dedup_key, normalize_company,
                              normalize_title)


def test_normalize_company_strips_legal_suffix():
    assert normalize_company("ACME GmbH") == "acme"
    assert normalize_company("Foo AG") == "foo"
    assert normalize_company("Bar Inc.") == "bar"
    assert normalize_company("Baz & Co. KG") == "baz"
    # одинаковая компания с/без суффикса → один ключ
    assert normalize_company("Siemens AG") == normalize_company("Siemens")


def test_normalize_title_strips_gender_tags():
    assert normalize_title("Process Engineer (m/w/d)") == "process engineer"
    assert normalize_title("Data Scientist (w/m/d)") == "data scientist"
    assert normalize_title("Pilot (all genders)") == "pilot"
    assert normalize_title("Engineer (m/f/d)") == "engineer"


def test_dedup_key_combines():
    k1 = dedup_key("ACME GmbH", "Process Engineer (m/w/d)")
    k2 = dedup_key("ACME", "Process Engineer")
    assert k1 == k2
    assert "::" in k1


def test_content_hash_stable_and_sensitive():
    h1 = content_hash("  Hello world  ")
    h2 = content_hash("Hello world")
    assert h1 == h2  # strip-инвариант
    assert content_hash("a") != content_hash("b")
    assert len(h1) == 16


def test_card_from_llm_json_coerces_and_validates():
    card = Card.from_llm_json({
        "role": "Eng", "company": "ACME", "domain": "aero", "city": "FRA",
        "work_mode": "HYBRID", "language_posting": "EN", "language_reality": "en",
        "integration_potential": 5, "summary_2lines": "a\nb",
        "slop_score": 250, "temp_agency_guess": False,
    })
    assert card.work_mode == "hybrid"           # lowercased
    assert card.integration_potential == 2      # clamped 0..2
    assert card.slop_score == 100               # clamped 0..100
    assert card.language_posting == "en"


def test_card_legacy_slop_flag_bool_maps_to_score():
    card = Card.from_llm_json({
        "role": "x", "company": "y", "domain": "z", "city": "c",
        "work_mode": "onsite", "language_posting": "de", "language_reality": "de",
        "integration_potential": 0, "summary_2lines": "a\nb",
        "slop_flag": True, "temp_agency_guess": True,
    })
    assert card.slop_score == 60  # bool True → density-tier ~60


def test_card_rejects_bad_language_reality():
    import pytest
    with pytest.raises(ValueError):
        Card.from_llm_json({
            "role": "x", "company": "y", "domain": "z", "city": "c",
            "work_mode": "onsite", "language_posting": "de", "language_reality": "fr",
            "integration_potential": 0, "summary_2lines": "a\nb",
            "slop_score": 0, "temp_agency_guess": False,
        })
