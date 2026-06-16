"""Hard-qualification / eligibility gate — the layer that catches PhD-without-Master, clearance,
German-C1, etc. Pure logic + a mocked-qwen extraction round-trip. All offline."""
from __future__ import annotations

import json

from schabasch import db, eligibility as E


# --- education normalization ---------------------------------------------------------------

def test_normalize_education_terms():
    assert E.normalize_education("bachelor") == 1
    assert E.normalize_education("BSc Computer Science") == 1
    assert E.normalize_education("Master") == 2
    assert E.normalize_education("Diplom (Univ.)") == 2     # German Diplom ≈ Master (EQF 7)
    assert E.normalize_education("Promotion / Dr.") == 3
    assert E.normalize_education("PhD") == 3
    assert E.normalize_education("highest degree and field") is None   # extraction placeholder
    assert E.normalize_education("") is None and E.normalize_education(None) is None


def test_candidate_quals():
    q = E.candidate_quals({"education_level": "bachelor", "years_experience": 6,
                           "languages": {"de": "A2", "en": "C1"}})
    assert q["education_ordinal"] == 1 and q["education_known"] is True
    assert q["languages"] == {"de": 2, "en": 5} and q["years"] == 6
    # unknown education → known=False, ordinal defaults 0 (gate must treat as no-penalty)
    qu = E.candidate_quals({"education": "highest degree and field"})
    assert qu["education_known"] is False


# --- the gate ------------------------------------------------------------------------------

BACHELOR = {"education_ordinal": 1, "education_known": True, "years": 6,
            "credentials": [], "languages": {"de": 2, "en": 5}}
MASTER = {**BACHELOR, "education_ordinal": 2}
UNKNOWN = {"education_ordinal": 0, "education_known": False, "years": 6,
           "credentials": [], "languages": {}}


def _req(**kw):
    base = E._empty_req()
    base.update(kw)
    return base


def test_phd_position_gates_bachelor_to_floor():
    g, note, sev = E.eligibility_gate(_req(is_phd_or_doctoral_position=True), BACHELOR)
    assert g == 0.35 and "Master" in note            # the exact bug the user caught
    assert sev == "structural"                        # a PhD position is the one red ⛔ STOP


def test_master_required_bachelor_is_mid():
    g, _, sev = E.eligibility_gate(_req(education_required="master", education_is_hard=True), BACHELOR)
    assert g == 0.6                                   # one-step gap → softer down-rank
    assert sev == "soft"                              # prose degree = amber, not red


def test_phd_prose_2step_is_structural_not_lifted():
    # prose "PhD required" for a Bachelor = a 2-step gap → STRUCTURAL (red ⛔), and NOT lifted by a
    # high fit (unlike a 1-step master gap) — a Bachelor genuinely can't satisfy a hard PhD-degree req.
    g, _, sev = E.eligibility_gate(_req(education_required="phd", education_is_hard=True), BACHELOR,
                                   fit_score=0.95, soft_lift_threshold=0.55)
    assert g == 0.6 and sev == "structural"
    # contrast: a 1-step master gap at the same high fit IS soft + lifted to 1.0
    g2, _, sev2 = E.eligibility_gate(_req(education_required="master", education_is_hard=True),
                                     BACHELOR, fit_score=0.95, soft_lift_threshold=0.55)
    assert g2 == 1.0 and sev2 == "soft"


def test_high_fit_lifts_soft_degree_to_1():
    # WS1c / her SCHOTT ask: a strong-fit job she'd apply to despite a prose degree line is NOT sunk
    # — the soft factor is lifted to 1.0 (note still emitted, severity still soft → amber).
    g, note, sev = E.eligibility_gate(_req(education_required="master", education_is_hard=True),
                                      BACHELOR, fit_score=0.70, soft_lift_threshold=0.55)
    assert g == 1.0 and sev == "soft" and "master" in note.lower()
    # below the threshold it still down-ranks
    g2, _, _ = E.eligibility_gate(_req(education_required="master", education_is_hard=True),
                                  BACHELOR, fit_score=0.40, soft_lift_threshold=0.55)
    assert g2 == 0.6


def test_master_required_master_is_eligible():
    g, note, _ = E.eligibility_gate(_req(education_required="master", education_is_hard=True), MASTER)
    assert g == 1.0 and note == ""


def test_overqualified_not_penalized():
    g, _, _ = E.eligibility_gate(_req(education_required="bachelor", education_is_hard=True), MASTER)
    assert g == 1.0                                   # minimum is a floor, not a window


def test_or_equivalent_does_not_gate():
    # "Master's OR equivalent experience" → education_is_hard=False → no penalty
    g, _, _ = E.eligibility_gate(_req(education_required="master", education_is_hard=False), BACHELOR)
    assert g == 1.0


def test_unknown_candidate_education_no_penalty():
    g, _, _ = E.eligibility_gate(_req(education_required="master", education_is_hard=True), UNKNOWN)
    assert g == 1.0                                   # never gate on what we don't know


def test_mandatory_credentials_do_NOT_gate():
    # credentials are DROPPED from the gate (qwen mislabels skills as credentials → over-fired).
    g, _, _ = E.eligibility_gate(_req(mandatory_credentials=["security clearance", "Python"]), BACHELOR)
    assert g == 1.0


def test_language_hard_below_is_mid():
    g, _, sev = E.eligibility_gate(
        _req(language_required=[{"lang": "German", "cefr": "C1", "is_hard": True}]), BACHELOR)
    assert g == 0.6                                   # cand de=A2(2) < C1(5)
    assert sev == "structural"                        # hard non-EN language is a real STOP → red ⛔


def test_language_hard_not_lifted_by_high_fit():
    # structural blockers are NEVER lifted by fit (unlike the soft degree gap)
    g, _, sev = E.eligibility_gate(
        _req(language_required=[{"lang": "German", "cefr": "C1", "is_hard": True}]), BACHELOR,
        fit_score=0.95, soft_lift_threshold=0.55)
    assert g == 0.6 and sev == "structural"


def test_language_unknown_no_penalty():
    g, _, _ = E.eligibility_gate(
        _req(language_required=[{"lang": "French", "cefr": "C1", "is_hard": True}]), BACHELOR)
    assert g == 1.0                                   # she has no French level → don't gate


# --- WS1b: "Master Data" / "Scrum Master" ≠ a master's DEGREE (the Merz-439 false positive) ----

def test_master_data_is_not_a_degree():
    assert not E._master_is_degree("Business Process Owner – Master Data & Artwork Coordination\n"
                                   "Ownership of global master data across SAP objects.")
    assert not E._master_is_degree("We are hiring a Scrum Master for an agile team.")
    assert E._master_is_degree("Master's degree in Engineering is required.")
    assert E._master_is_degree("academic degree (Masters or PhD) in computer science")
    # 1164-style: both a non-degree 'Master Data' AND a real 'Master's degree' → degree present
    assert E._master_is_degree("Interfaces with Master Data Management.\nMaster's degree in Eng.")


def test_merz_439_master_data_no_degree_gate(con):
    """Merz-439: qwen wrongly extracts education_required='master' from the title 'Master Data'.
    The negative-context guard must NULL it so her #1 job is no longer falsely gated."""
    class _C:
        def chat_json(self, system, user):
            return {"education_required": "master", "education_is_hard": True,
                    "is_phd_or_doctoral_position": False, "reason": "phantom"}
    jd = ("Business Process Owner – Master Data & Artwork Coordination (m/f/d)\n"
          "Own global master data across all SAP objects; experience in master data management.")
    req = E.extract_requirements(con, content_hash="merz439", jd_text=jd, client=_C())
    assert req["education_required"] is None and req["education_is_hard"] is False
    g, note, _ = E.eligibility_gate(req, BACHELOR)
    assert g == 1.0 and note == ""                    # no phantom master gate


def test_real_master_degree_still_gates(con):
    """A genuine 'Master's degree' JD keeps the (soft) gate — the guard must not over-suppress."""
    class _C:
        def chat_json(self, system, user):
            return {"education_required": "master", "education_is_hard": True,
                    "is_phd_or_doctoral_position": False, "reason": "Master required"}
    jd = "Senior Risk Analyst\nWho You Are: Master's degree in a relevant field."
    req = E.extract_requirements(con, content_hash="real_master", jd_text=jd, client=_C())
    assert req["education_required"] == "master" and req["education_is_hard"] is True


# --- extraction round-trip (qwen mocked, cached, null-on-failure) ---------------------------

def test_title_guard_distinguishes_position_from_mention():
    # a doctoral/student POSITION (title) gates; a regular role that merely MENTIONS PhD does not
    assert E._title_is_student_or_doctoral("PhD – Agentic AI and Multi-Agent Systems")
    assert E._title_is_student_or_doctoral("Masterarbeit: Applied Statistics")
    assert E._title_is_student_or_doctoral("Werkstudent (m/w/d) Data")
    assert not E._title_is_student_or_doctoral("Equity Platform AI Engineer (f/m/d)")
    assert not E._title_is_student_or_doctoral("Manager Strategy & Projects")


def test_soft_degree_guard():
    assert E._degree_requirement_is_soft("PhD preferred (or advanced degree) in CS")
    assert E._degree_requirement_is_soft("Master oder gleichwertige Erfahrung")
    assert not E._degree_requirement_is_soft("Requirements: MSc/PhD in Computer Science")


def test_extract_overrides_recover_phd_preferred(con):
    """A regular role that says 'PhD preferred' must NOT be flagged a doctoral position, and its
    degree requirement must be softened — the exact false-positive that nuked good ML jobs."""
    class _C:
        def chat_json(self, system, user):   # qwen (wrongly) flags it doctoral + hard
            return {"education_required": "phd", "education_is_hard": True,
                    "is_phd_or_doctoral_position": True}
    jd = "Equity Platform AI Engineer\nWhat you bring: PhD preferred (or advanced degree) in CS"
    req = E.extract_requirements(con, content_hash="eq1", jd_text=jd, client=_C())
    assert req["is_phd_or_doctoral_position"] is False   # title guard overrode qwen
    assert req["education_is_hard"] is False             # 'preferred' softened it
    g, _, _ = E.eligibility_gate(req, BACHELOR)
    assert g == 1.0                                      # → not gated, recovered


def test_slate_downranks_ineligible_even_with_high_fit(con, cfg):
    """The PhD-without-Master bug: a judge-5, high-SKILL-fit job sinks below a lower job once it
    fails the ELIGIBILITY gate — but is not hidden (down-rank)."""
    from datetime import date
    from schabasch import slate, features
    from schabasch.models import Status
    from tests.conftest import make_card

    def seed(url, score, fit, elig, note="", title="T", company="C"):
        vid = db.upsert_vacancy(con, {"source": "indeed", "url": url, "title": title,
                                      "company": company, "city": "Frankfurt", "description": "x" * 200})
        db.set_status(con, vid, Status.DESCRIBED, card_json=json.dumps(make_card()))
        db.insert_judge_score(con, vid, {"score": score, "why_tag": None, "why_freetext": None,
                                         "explanation": "x", "model": "q", "model_digest": "d",
                                         "rubric_version": cfg["judge"]["rubric_version"], "fewshot_hash": "h"})
        db.set_status(con, vid, Status.SCORED)
        features._ensure_schema(con)
        feat = {"match_score": 0.5, "fit_score": fit, "fit_hyre": fit,
                "xenc_full": fit, "xenc_musthave": fit, "elig_score": elig, "elig_note": note}
        con.execute("INSERT OR REPLACE INTO vacancy_feature (vacancy_id, match_score, feature_json, computed_at)"
                    " VALUES (?,?,?,datetime('now'))", (vid, 0.5, json.dumps(feat)))
        con.commit()
        return vid

    phd = seed("u/phd", 5, 0.85, 0.35, "PhD требует Master",   # great skills, NOT eligible
               title="PhD Researcher Agentic AI", company="DoctoralCo")
    ml = seed("u/ml", 4, 0.85, 1.0,                            # great skills, eligible
              title="Senior Data Analyst", company="EligibleCo")
    items = slate.build_slate(cfg, con, date.today().isoformat())
    order = [it["vacancy_id"] for it in items]
    assert order.index(ml) < order.index(phd)                  # eligible outranks ineligible
    assert phd in order                                        # but not hidden
    html = slate.render_html(items, date.today().isoformat())
    assert "⛔" in html and "Master" in html                   # eligibility warning rendered


def test_extract_requirements_cached_and_robust(con, monkeypatch):
    calls = {"n": 0}

    class _C:
        def chat_json(self, system, user):
            calls["n"] += 1
            return {"education_required": "master", "education_is_hard": True,
                    "is_phd_or_doctoral_position": True, "reason": "нужен Master"}
    req = E.extract_requirements(con, content_hash="h1", jd_text="PhD position …", client=_C())
    assert req["is_phd_or_doctoral_position"] is True and req["education_required"] == "master"
    assert calls["n"] == 1
    E.extract_requirements(con, content_hash="h1", jd_text="PhD position …", client=_C())
    assert calls["n"] == 1                            # cache hit, no second call

    class _Boom:
        def chat_json(self, system, user):
            raise RuntimeError("ollama down")
    req2 = E.extract_requirements(con, content_hash="h2", jd_text="x", client=_Boom())
    assert req2 == E._empty_req()                     # failure → all-empty = no constraint
