"""Deterministic role-kind classifier (schabasch/role_kind.py)."""
from __future__ import annotations

from schabasch import role_kind as rk
from schabasch import role_feedback
from tests.conftest import seed_scored


def test_multiplier_behaviour_preserving_when_learning_off(cfg, con):
    """P2 gate: con supplied but learning disabled (default) → exactly the static default."""
    assert rk.multiplier("hands_on_engineer", cfg, con) == rk._DEFAULT_MULT["hands_on_engineer"]
    assert rk.multiplier("hands_on_engineer", cfg, None) == rk._DEFAULT_MULT["hands_on_engineer"]


def test_multiplier_learns_from_wrong_role_votes(con):
    """P2: ≥ n_min golden '🙅 wrong role' votes pull the engineer multiplier below the 0.7 default
    toward the floor; a kind voted all-fits stays at its default. Below n_min → static default."""
    cfg = {"slate": {"role_kind_mult": {"hands_on_engineer": 0.7},   # the user's own penalty
                     "role_kind_learn": {"enabled": True, "n_min": 5, "alpha": 3.0, "mult_floor": 0.5}}}
    assert rk.multiplier("hands_on_engineer", cfg, con) == 0.7          # no votes → configured default

    for i in range(5):                                                   # 5 wrong-role engineer votes
        v = seed_scored(con, f"e/{i}", score=4, company="C", title="Software Engineer")
        role_feedback.record(con, v, "hands_on_engineer", False, source="slate")
    m = rk.multiplier("hands_on_engineer", cfg, con)
    assert 0.5 <= m < 0.7                                                # learned DOWN from data

    for i in range(5):                                                   # 5 fits votes on lead
        v = seed_scored(con, f"l/{i}", score=4, company="C", title="Principal Engineer")
        role_feedback.record(con, v, "lead", True, source="slate")
    assert rk.multiplier("lead", cfg, con) == 1.0                        # all-fits → stays at default 1.0


def test_classify_engineer_vs_lead_vs_junior():
    assert rk.classify("Data Processing System Engineer") == "hands_on_engineer"
    assert rk.classify("ML Optimization Engineer") == "hands_on_engineer"
    assert rk.classify("Softwareentwickler (m/w/d)") == "hands_on_engineer"
    # a LEAD/principal engineering role is head-not-hands work the user likes → not a repellent
    assert rk.classify("Principal Software Engineer") == "lead"
    assert rk.classify("Lead Systems Engineer") == "lead"
    # intern/working-student floor wins over everything
    assert rk.classify("Working Student AI Engineer") == "junior"
    assert rk.classify("Praktikum Data Science") == "junior"
    # roles that fit the user's → neutral (no penalty, no flag)
    assert rk.classify("Senior Business Analyst") == ""
    assert rk.classify("Business Process Owner") == ""
    assert rk.classify("Program Manager") == ""


def test_multiplier_soft_downrank_never_zero():
    # de-personalization 2026-07-03: code defaults are NEUTRAL; the penalty is the user's config
    assert rk.multiplier("hands_on_engineer") == 1.0
    assert rk.multiplier("junior") == 1.0
    taste = {"slate": {"role_kind_mult": {"hands_on_engineer": 0.45, "junior": 0.5}}}
    assert 0 < rk.multiplier("hands_on_engineer", taste) < 1.0   # penalized, never zeroed
    assert 0 < rk.multiplier("junior", taste) < 1.0
    assert rk.multiplier("lead", taste) == 1.0
    assert rk.multiplier("", taste) == 1.0
    # config override
    cfg = {"slate": {"role_kind_mult": {"hands_on_engineer": 0.9}}}
    assert rk.multiplier("hands_on_engineer", cfg) == 0.9


def test_flag_only_for_repellent_kinds():
    assert "hands-on" in rk.flag("hands_on_engineer")
    assert rk.flag("junior")
    assert rk.flag("lead") == "" and rk.flag("") == ""
