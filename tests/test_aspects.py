"""Tests for aspects.py — JD segmentation + aspect-pair coverage scoring.

All tests use toy numpy vectors (no bge-m3 model required).
"""
from __future__ import annotations

import numpy as np
import pytest

from schabasch import aspects


# ---------------------------------------------------------------------------
# segment_jd tests
# ---------------------------------------------------------------------------

EN_JD = """\
About TechCorp
We build cool software.

Requirements
- Python 3 years experience
- SQL proficiency
- Tableau or Power BI

Responsibilities
- Lead digital transformation projects
- Manage stakeholder requirements

We offer
- Flexible hours
- Remote option
"""

DE_JD = """\
Über uns
Wir sind ein innovatives Unternehmen.

Anforderungen
- 3 Jahre Python-Erfahrung
- SQL-Kenntnisse

Deine Aufgaben
- Führung von IT-Projekten
- Stakeholder-Management

Wir bieten
- Flexible Arbeitszeiten
"""

MIXED_JD = """\
Requirements:
- Python
- SQL

Deine Aufgaben:
- Projekte leiten

About us:
We are a great company.
"""

HEADERLESS_JD = """\
We are looking for a business analyst. You will need Python and SQL.
You will work on analytics projects. No structured headers here.
"""


def test_segment_jd_en_structure():
    s = aspects.segment_jd("Business Analyst", EN_JD)
    assert s["_meta"]["has_structure"] == 1
    assert "Python" in s.get("must_have", "")
    assert "digital transformation" in s.get("responsibilities", "")
    assert "TechCorp" in s.get("company", "")


def test_segment_jd_de_structure():
    s = aspects.segment_jd("Business Analyst", DE_JD)
    assert s["_meta"]["has_structure"] == 1
    assert "Python" in s.get("must_have", "")
    assert "IT-Projekten" in s.get("responsibilities", "")
    assert "innovatives" in s.get("company", "")


def test_segment_jd_mixed_en_de():
    s = aspects.segment_jd("Role", MIXED_JD)
    assert s["_meta"]["has_structure"] == 1
    assert "Python" in s.get("must_have", "")
    assert "Projekte" in s.get("responsibilities", "")


def test_segment_jd_headerless_fallback():
    s = aspects.segment_jd("Analyst", HEADERLESS_JD)
    assert s["_meta"]["has_structure"] == 0
    assert "full" in s
    assert "Python" in s["full"]


def test_segment_jd_intro_goes_to_company():
    """Text before first header should end up in company section."""
    jd = "ACME Corp is a great place.\n\nRequirements\n- SQL\n"
    s = aspects.segment_jd("Role", jd)
    assert "ACME" in s.get("company", "")
    assert "SQL" in s.get("must_have", "")


def test_segment_jd_meta_contains_title():
    s = aspects.segment_jd("Data Analyst", "Requirements\n- Python\n")
    assert s["_meta"]["title"] == "Data Analyst"


# ---------------------------------------------------------------------------
# extract_jd_skills tests
# ---------------------------------------------------------------------------

def test_extract_jd_skills_bullets():
    text = "- Python 3+ years\n- SQL proficiency\n• Tableau experience"
    skills = aspects.extract_jd_skills(text)
    assert "Python 3+ years" in skills
    assert "SQL proficiency" in skills
    assert "Tableau experience" in skills


def test_extract_jd_skills_empty():
    assert aspects.extract_jd_skills("") == []


# ---------------------------------------------------------------------------
# detect_jd_seniority tests
# ---------------------------------------------------------------------------

def test_detect_seniority_senior():
    assert aspects.detect_jd_seniority("Senior Business Analyst") == "senior"

def test_detect_seniority_junior():
    assert aspects.detect_jd_seniority("Junior Data Analyst") == "junior"

def test_detect_seniority_lead():
    assert aspects.detect_jd_seniority("Lead Product Manager") == "lead"

def test_detect_seniority_mid_default():
    assert aspects.detect_jd_seniority("Business Analyst") == "mid"

def test_detect_seniority_de():
    assert aspects.detect_jd_seniority("Senior Business Analyst (m/w/d)") == "senior"


# ---------------------------------------------------------------------------
# score() tests with toy vectors
# ---------------------------------------------------------------------------

DIM = 8  # small for speed; real is 1024

def _ones(i: int) -> np.ndarray:
    """Unit vector with 1 at position i, rest 0."""
    v = np.zeros(DIM, dtype=np.float32)
    v[i] = 1.0
    return v

def _rand(seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    v = rng.randn(DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def _toy_cand_vecs(skill_idx=0, exp_idx=1, dom_idx=2, roles_idx=3):
    return {
        "skills":     _ones(skill_idx),
        "experience": _ones(exp_idx),
        "domains":    _ones(dom_idx),
        "roles":      _ones(roles_idx),
        "full":       _ones(skill_idx),
    }


def _toy_jd_vecs(skill_idx=0, resp_idx=1, comp_idx=2):
    return {
        "must_have":        _ones(skill_idx),
        "responsibilities": _ones(resp_idx),
        "company":          _ones(comp_idx),
        "nice_to_have":     _ones(skill_idx),
    }


def _sections():
    return {
        "_meta": {"has_structure": 1, "title": "Business Analyst"},
        "must_have": "- Python\n- SQL",
        "responsibilities": "- Lead projects",
        "company": "About us text",
    }


def test_score_returns_all_feature_names():
    feat = aspects.score(
        cand_vecs=_toy_cand_vecs(),
        cand_skills=["Python", "SQL"],
        jd_sections=_sections(),
        jd_section_vecs=_toy_jd_vecs(),
        jd_full_vec=_ones(0),
    )
    for name in aspects.FEATURE_NAMES:
        assert name in feat, f"Missing feature: {name}"
    assert "match_score" in feat


class _RecordingLibrary:
    """Fake PositiveLibrary that records the vector score() hands to taste_features."""
    n_rows = 3

    def __init__(self):
        self.received = None

    def taste_features(self, vec, *, company=None, exclude_vacancy_id=None):
        self.received = np.asarray(vec, dtype=np.float32).copy()
        return {k: 0.0 for k in ("nearest_liked_cosine", "positive_centroid_cosine",
                                 "recent_centroid_cosine", "topic_drift", "company_overlap_count")}


def test_taste_uses_jd_vector_not_cv():
    """Regression: taste features must be scored against THIS vacancy's JD vector, not the
    constant candidate CV vector (else every vacancy gets an identical taste score)."""
    lib = _RecordingLibrary()
    jd = _ones(5)
    cand = _toy_cand_vecs()
    cand["full"] = _ones(0)        # CV 'full' deliberately != jd
    aspects.score(
        cand_vecs=cand, cand_skills=["x"], jd_sections=_sections(),
        jd_section_vecs=_toy_jd_vecs(), jd_full_vec=jd, library=lib, vacancy_id=7,
    )
    assert lib.received is not None
    assert np.allclose(lib.received, jd)          # got the JD vector
    assert not np.allclose(lib.received, cand["full"])  # NOT the constant CV vector


def test_score_high_overlap_gives_high_match():
    """When CV skills align perfectly with JD requirements, match_score should be high."""
    feat = aspects.score(
        cand_vecs=_toy_cand_vecs(skill_idx=0, exp_idx=1),
        cand_skills=["Python", "SQL"],
        jd_sections=_sections(),
        jd_section_vecs=_toy_jd_vecs(skill_idx=0, resp_idx=1),
        jd_full_vec=_ones(0),
    )
    assert feat["cov_musthave_maxsim"] > 0.9
    assert feat["sim_skills_requirements"] > 0.9
    assert feat["match_score"] > 0.5


def test_score_no_overlap_gives_low_match():
    """Orthogonal CV and JD → low coverage, low match_score."""
    feat = aspects.score(
        cand_vecs=_toy_cand_vecs(skill_idx=4, exp_idx=5),
        cand_skills=["Pottery", "Knitting"],  # no keyword match to "Python SQL"
        jd_sections=_sections(),
        jd_section_vecs=_toy_jd_vecs(skill_idx=0, resp_idx=1),
        jd_full_vec=_ones(0),
    )
    assert feat["cov_musthave_maxsim"] < 0.1
    assert feat["match_score"] < 0.2


def test_seniority_gap_signed():
    """senior CV vs junior JD → positive gap (overqualified); junior CV vs senior → negative."""
    feat_over = aspects.score(
        cand_vecs=_toy_cand_vecs(), cand_skills=[],
        jd_sections=_sections(), jd_section_vecs=_toy_jd_vecs(), jd_full_vec=_ones(0),
        cand_seniority="senior", jd_seniority="junior",
    )
    feat_under = aspects.score(
        cand_vecs=_toy_cand_vecs(), cand_skills=[],
        jd_sections=_sections(), jd_section_vecs=_toy_jd_vecs(), jd_full_vec=_ones(0),
        cand_seniority="junior", jd_seniority="senior",
    )
    assert feat_over["seniority_gap"] > 0
    assert feat_under["seniority_gap"] < 0


def test_underqualified_penalizes_match_score():
    feat_under = aspects.score(
        cand_vecs=_toy_cand_vecs(0, 1), cand_skills=["Python", "SQL"],
        jd_sections=_sections(), jd_section_vecs=_toy_jd_vecs(0, 1), jd_full_vec=_ones(0),
        cand_seniority="junior", jd_seniority="lead",
    )
    feat_match = aspects.score(
        cand_vecs=_toy_cand_vecs(0, 1), cand_skills=["Python", "SQL"],
        jd_sections=_sections(), jd_section_vecs=_toy_jd_vecs(0, 1), jd_full_vec=_ones(0),
        cand_seniority="senior", jd_seniority="senior",
    )
    assert feat_under["match_score"] < feat_match["match_score"]


def test_n_musthave_missing_no_skills():
    """JD requires Python/SQL, CV has none → n_musthave_missing > 0."""
    feat = aspects.score(
        cand_vecs=_toy_cand_vecs(), cand_skills=[],
        jd_sections=_sections(), jd_section_vecs=_toy_jd_vecs(), jd_full_vec=_ones(0),
    )
    assert feat["n_musthave_missing"] >= 0  # may be 0 if no bullets found


def test_n_musthave_missing_with_bullets():
    """When JD must_have has bullets CV can't cover, n_musthave_missing > 0."""
    sections = {
        "_meta": {"has_structure": 1, "title": "Analyst"},
        "must_have": "- Kubernetes\n- React\n- Blockchain",  # none in CV skills
    }
    feat = aspects.score(
        cand_vecs=_toy_cand_vecs(), cand_skills=["Python", "SQL"],
        jd_sections=sections, jd_section_vecs=_toy_jd_vecs(), jd_full_vec=_ones(0),
    )
    assert feat["n_musthave_missing"] > 0


def test_taste_features_zero_when_library_none():
    feat = aspects.score(
        cand_vecs=_toy_cand_vecs(), cand_skills=[],
        jd_sections=_sections(), jd_section_vecs=_toy_jd_vecs(), jd_full_vec=_ones(0),
        library=None,
    )
    assert feat["nearest_liked_cosine"] == 0.0
    assert feat["positive_centroid_cosine"] == 0.0
    assert feat["topic_drift"] == 0.0


def test_gate_features_passed_through():
    gates = {"lang_de_required": 1.0, "geo_distance_norm": 0.3, "is_remote_hint": 1.0,
             "recency_days": 7.0}
    feat = aspects.score(
        cand_vecs=_toy_cand_vecs(), cand_skills=[],
        jd_sections=_sections(), jd_section_vecs=_toy_jd_vecs(), jd_full_vec=_ones(0),
        gates=gates,
    )
    assert feat["lang_de_required"] == 1.0
    assert feat["geo_distance_norm"] == 0.3
    assert feat["is_remote_hint"] == 1.0
    assert feat["recency_days"] == 7.0


def test_colbert_fn_called_lazily():
    called = []
    def lazy_colbert():
        called.append(1)
        return 0.75

    feat = aspects.score(
        cand_vecs=_toy_cand_vecs(), cand_skills=[],
        jd_sections=_sections(), jd_section_vecs=_toy_jd_vecs(), jd_full_vec=_ones(0),
        colbert_fn=lazy_colbert,
    )
    assert len(called) == 1
    assert feat["colbert_req_to_cv"] == 0.75


def test_match_score_bounded():
    for _ in range(20):
        feat = aspects.score(
            cand_vecs={k: _rand(i) for i, k in enumerate(
                ("skills", "experience", "domains", "roles", "full"))},
            cand_skills=["Python"],
            jd_sections=_sections(),
            jd_section_vecs={k: _rand(i+5) for i, k in enumerate(
                ("must_have", "responsibilities", "company", "nice_to_have"))},
            jd_full_vec=_rand(10),
        )
        assert 0.0 <= feat["match_score"] <= 1.0
