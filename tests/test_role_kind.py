"""Deterministic role-kind classifier (schabasch/role_kind.py)."""
from __future__ import annotations

from schabasch import role_kind as rk


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
    assert rk.multiplier("hands_on_engineer") < 1.0
    assert rk.multiplier("junior") < rk.multiplier("hands_on_engineer")   # interns sink hardest
    assert rk.multiplier("lead") == 1.0
    assert rk.multiplier("") == 1.0
    # config override
    cfg = {"slate": {"role_kind_mult": {"hands_on_engineer": 0.9}}}
    assert rk.multiplier("hands_on_engineer", cfg) == 0.9


def test_flag_only_for_repellent_kinds():
    assert "hands-on" in rk.flag("hands_on_engineer")
    assert rk.flag("junior")
    assert rk.flag("lead") == "" and rk.flag("") == ""
