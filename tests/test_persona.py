"""Multi-user de-personalization (2026-07-03 audit): the judge persona, hard drops, and role
penalties come from cfg.profile — user #1's taste must not leak into user #2's pipeline."""
from __future__ import annotations

import json

import pytest

from schabasch import hardfilters, judge, role_kind
from schabasch.models import WHY_TAGS, Card, FilterReason, Status


def _cfg_with_profile(cfg, **profile):
    out = dict(cfg)
    out["profile"] = {**cfg.get("profile", {}), **profile}
    return out


# ── judge persona from cfg ────────────────────────────────────────────────────────────────────

def test_judge_prompt_uses_user_magnets_and_repellents(cfg):
    c = _cfg_with_profile(cfg, magnets=["clinical-ai", "oncology"], repellents=["slop-text"],
                          taste_rules=["Hands-on разработка — это ПЛЮС."])
    prompt = judge.build_system_prompt(c, fewshot="")
    assert "МАГНИТЫ (тянут к 4–5): clinical-ai, oncology" in prompt
    assert "ОТТАЛКИВАТЕЛИ (тянут к 1–2): slop-text" in prompt
    assert "Hands-on разработка — это ПЛЮС." in prompt
    # user #1's hardcoded taste must be GONE from the code path
    assert "работать ГОЛОВОЙ" not in prompt
    assert "animals > space" not in prompt
    assert "биотех" not in prompt
    # example answer anchors to the USER's first magnet
    assert '"why_tag": "clinical-ai"' in prompt


def test_judge_prompt_falls_back_to_why_tags(cfg):
    c = dict(cfg)
    c["profile"] = {"summary": "x", "scale": {}}   # no magnets/repellents/taste_rules
    prompt = judge.build_system_prompt(c, fewshot="")
    assert ", ".join(WHY_TAGS[:6]) in prompt       # legacy vocabulary as the default
    assert ", ".join(WHY_TAGS[6:]) in prompt


def test_judge_taste_rules_render_from_base_yaml_shape(cfg):
    """The default user's rules moved verbatim from judge.py into her profile.yaml — the prompt
    must carry them again when the config provides them (behavior preservation)."""
    rules = ["Пользователь хочет работать ГОЛОВОЙ, не руками: чисто hands-on dev/инженер-роль — минус."]
    prompt = judge.build_system_prompt(_cfg_with_profile(cfg, taste_rules=rules), fewshot="")
    assert "работать ГОЛОВОЙ" in prompt


# ── hard drops gated on repellents ────────────────────────────────────────────────────────────

def _seed_described(con, url, desc="x" * 300, temp=0):
    from schabasch import db
    vid = db.upsert_vacancy(con, {"source": "indeed", "url": url, "title": f"T {url}",
                                  "company": "C", "city": "Berlin", "description": desc,
                                  "is_temp_agency": temp})
    db.set_status(con, vid, Status.DESCRIBED)
    return vid


GERMAN_DESC = "You will need fließend Deutsch (C1) for daily work. " + "x" * 300


def test_hardfilters_german_drop_gated_on_repellent(cfg, con):
    from schabasch import db
    # user WITHOUT hidden-german repellent → German-required job survives
    vid = _seed_described(con, "u/de1", desc=GERMAN_DESC)
    c = _cfg_with_profile(cfg, repellents=["slop-text"])
    hardfilters.apply_hard_filters(c, con)
    assert con.execute("SELECT status FROM vacancy WHERE id=?", (vid,)).fetchone()[0] == \
        Status.DESCRIBED.value
    # user WITH it (the default profile) → filtered as before
    db.set_status(con, vid, Status.DESCRIBED)
    c2 = _cfg_with_profile(cfg, repellents=["hidden-german"])
    hardfilters.apply_hard_filters(c2, con)
    assert con.execute("SELECT status, filter_reason FROM vacancy WHERE id=?", (vid,)).fetchone()[
        "filter_reason"] == FilterReason.LANGUAGE_DE.value


def test_hardfilters_default_without_profile_preserves_old_behavior(cfg, con):
    """No profile.repellents in cfg at all → the original full drop set (single-user compat)."""
    vid = _seed_described(con, "u/de2", desc=GERMAN_DESC)
    c = dict(cfg)
    c.pop("profile", None)
    hardfilters.apply_hard_filters(c, con)
    assert con.execute("SELECT status FROM vacancy WHERE id=?", (vid,)).fetchone()[0] == \
        Status.FILTERED.value


def test_normalize_remote_drop_gated_on_repellent(cfg, con):
    from schabasch.normalize import _filter_card
    card = Card.from_llm_json(dict(role="r", company="C", domain="ai", city="Berlin",
                                   work_mode="remote", language_posting="en",
                                   language_reality="en", integration_potential=1,
                                   summary_2lines="a\nb", slop_score=5, temp_agency_guess=False))
    vid = _seed_described(con, "u/rem1")
    # remote-only NOT a repellent (user #2) → job survives normalization
    res = _filter_card(con, vid, card, _cfg_with_profile(cfg, repellents=["hidden-german"]))
    assert res == "normalized"
    # remote-only IS a repellent (default profile) → filtered
    vid2 = _seed_described(con, "u/rem2")
    res = _filter_card(con, vid2, card, _cfg_with_profile(cfg, repellents=["remote-only"]))
    assert res == "filtered"


# ── role_kind neutral defaults ────────────────────────────────────────────────────────────────

def test_role_kind_default_is_neutral_and_cfg_penalty_applies():
    assert role_kind.multiplier("hands_on_engineer", {}) == 1.0    # neutral without config
    assert role_kind.multiplier("junior", {}) == 1.0
    alina = {"slate": {"role_kind_mult": {"hands_on_engineer": 0.45, "junior": 0.5}}}
    assert role_kind.multiplier("hands_on_engineer", alina) == 0.45
    assert role_kind.multiplier("junior", alina) == 0.5


def test_role_flag_follows_penalty():
    neutral = {"slate": {"role_kind_mult": {"hands_on_engineer": 1.0}}}
    penalizing = {"slate": {"role_kind_mult": {"hands_on_engineer": 0.45}}}
    assert role_kind.flag("hands_on_engineer", neutral) == ""            # no penalty → no «не твоё»
    assert "hands-on" in role_kind.flag("hands_on_engineer", penalizing)
    assert "hands-on" in role_kind.flag("hands_on_engineer")             # legacy no-cfg call


# ── enrichment injects the user profile ───────────────────────────────────────────────────────

def test_enrichment_prompt_injects_user_profile(cfg):
    from schabasch.enrichment import _system_prompt
    c = _cfg_with_profile(cfg, summary="Oncologist moving into Clinical AI.",
                          repellents=["slop-text", "temp-agency"])
    s = _system_prompt(c)
    assert "Oncologist moving into Clinical AI." in s
    assert "slop-text, temp-agency" in s
    assert "мастер/PhD" not in s          # user #1's hardcoded cons are gone
    assert "скрытый немецкий, мастер" not in s
