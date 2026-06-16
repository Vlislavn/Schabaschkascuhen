"""Tests for candidate.py — structured résumé intake + aspect extraction."""
from __future__ import annotations

import json

import pytest

from schabasch import candidate
from schabasch.llm import OllamaClient


def _profile_response():
    return {
        "skills": ["Python", "SQL", "Tableau", "Power BI", "AWS", "BPMN 2.0"],
        "experience": "Senior BA at BostonGene leading LIMS consolidation; ex-SWE at Nielsen.",
        "domains": ["digital transformation", "analytics", "lab software", "e-commerce"],
        "seniority": "senior",
        "years_experience": 6,
        "education": "Bachelor's in Business Informatics",
        "languages": {"de": "A2", "en": "C1", "ru": "C2"},
        "target_roles": ["Senior Business Analyst", "Product Manager", "Data Analyst"],
        "locations": ["Heidelberg", "Frankfurt"],
        "magnets": ["complex integration", "digital transformation", "data analytics"],
        "repellents": ["German-required roles", "pure lab science", "remote-only"],
        "summary": "Senior BA with SWE background seeking digital transformation roles in English environment.",
    }


def test_extract_candidate_persists_and_returns(cfg, con, monkeypatch):
    monkeypatch.setattr(OllamaClient, "chat_json", lambda self, s, u: _profile_response())
    result = candidate.extract_candidate(cfg, con, description="Alina, Senior BA, Heidelberg")

    assert result["seniority"] == "senior"
    assert "Python" in result["skills"]
    assert "aspect_texts" in result
    assert result["aspect_texts"]["skills_text"].startswith("Python")
    assert "full_doc" in result["aspect_texts"]


def test_aspect_texts_roundtrip(cfg, con, monkeypatch):
    monkeypatch.setattr(OllamaClient, "chat_json", lambda self, s, u: _profile_response())
    candidate.extract_candidate(cfg, con, description="test")

    texts = candidate.aspect_texts(con)
    assert texts is not None
    assert "Python" in texts["skills_text"]
    assert "Senior BA" in texts["experience_text"]
    assert "analytics" in texts["domains_text"]
    assert "Business Analyst" in texts["roles_text"]
    assert len(texts["full_doc"]) > 100


def test_candidate_doc_is_full_doc(cfg, con, monkeypatch):
    monkeypatch.setattr(OllamaClient, "chat_json", lambda self, s, u: _profile_response())
    candidate.extract_candidate(cfg, con, description="test")
    doc = candidate.candidate_doc(con)
    assert doc is not None
    assert "Skills:" in doc and "Experience:" in doc


def test_load_candidate_returns_latest(cfg, con, monkeypatch):
    # Insert two profiles; load_candidate returns the newer one
    monkeypatch.setattr(OllamaClient, "chat_json", lambda self, s, u: _profile_response())
    candidate.extract_candidate(cfg, con, description="first")

    second = dict(_profile_response())
    second["seniority"] = "lead"
    monkeypatch.setattr(OllamaClient, "chat_json", lambda self, s, u: second)
    candidate.extract_candidate(cfg, con, description="second")

    loaded = candidate.load_candidate(con)
    assert loaded["seniority"] == "lead"


def test_load_candidate_returns_none_when_empty(con):
    candidate._ensure_schema(con)
    assert candidate.load_candidate(con) is None
    assert candidate.aspect_texts(con) is None
    assert candidate.candidate_doc(con) is None


def test_extract_requires_description_or_cv_path(cfg, con):
    with pytest.raises(ValueError, match="description"):
        candidate.extract_candidate(cfg, con)


def test_missing_profile_key_raises(cfg, con, monkeypatch):
    bad = _profile_response()
    del bad["skills"]
    monkeypatch.setattr(OllamaClient, "chat_json", lambda self, s, u: bad)
    with pytest.raises(RuntimeError):
        candidate.extract_candidate(cfg, con, description="test")


def test_aspect_texts_content_hash_stable(cfg, con, monkeypatch):
    monkeypatch.setattr(OllamaClient, "chat_json", lambda self, s, u: _profile_response())
    r1 = candidate.extract_candidate(cfg, con, description="a")
    r2 = candidate.extract_candidate(cfg, con, description="a")
    # Same profile content → same doc_hash
    assert r1["doc_hash"] == r2["doc_hash"]


def test_validate_coerces_string_list_and_defaults_optionals():
    """Regression: a list field returned as a bare string must coerce to a list (not char-iterate),
    and a missing optional key (e.g. repellents) must not crash intake."""
    from schabasch.candidate import _validate
    d = _validate({"skills": "Python, SQL", "experience": "6y", "domains": "biotech"})
    assert d["skills"] == ["Python", "SQL"]
    assert d["domains"] == ["biotech"]          # not ['b','i','o',...]
    assert d["repellents"] == []                # missing optional → default-filled, no crash
    assert d["target_roles"] == []
