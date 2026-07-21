"""Skill-gap stats + slate freshness ceiling + dates fallback + themed score badge."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from schabasch import candidate, db, features as _features, gaps, slate
from schabasch.models import Status
from tests.conftest import make_card


def _seed(con, url, *, score, label=None, applied=0, reqs=None, last_seen=None):
    vid = db.upsert_vacancy(con, {"source": "indeed", "url": url, "title": f"T{url}",
                                  "company": "ACME", "city": "Frankfurt", "description": "x" * 400})
    db.set_status(con, vid, Status.SCORED, card_json=json.dumps(make_card()))
    db.insert_judge_score(con, vid, {"score": score, "why_tag": None, "why_freetext": None,
                                     "explanation": "e", "model": "qwen3:8b", "model_digest": "d",
                                     "rubric_version": "test-v1", "fewshot_hash": "h"})
    if reqs is not None:
        _features._ensure_schema(con)
        feat = {"fit_score": 0.5, "llm_cov": 0.5, "llm_cov_reqs": reqs}
        con.execute("INSERT OR REPLACE INTO vacancy_feature (vacancy_id, match_score, feature_json,"
                    " computed_at) VALUES (?,?,?,?)",
                    (vid, 0.5, json.dumps(feat), datetime.now(timezone.utc).isoformat()))
    if label is not None or applied:
        db.insert_label(con, vid, {"score_1_5": label, "applied": applied, "source": "slate"})
    if last_seen:
        con.execute("UPDATE vacancy SET last_seen=? WHERE id=?", (last_seen, vid))
    con.commit()
    return vid


# --------------------------------------------------------------------------- skill-gap report

def test_gap_report_aggregates_wanted_only(con, cfg):
    _seed(con, "u/a", score=5, label=5, reqs=[  # wanted (⭐)
        {"requirement": "BPMN", "verdict": "present"},
        {"requirement": "Kubernetes", "verdict": "missing"},
        {"requirement": "ML", "verdict": "partial"}])
    _seed(con, "u/b", score=3, applied=1, reqs=[  # wanted (applied) even though score=3
        {"requirement": "Kubernetes", "verdict": "missing"},
        {"requirement": "Power BI", "verdict": "present"}])
    _seed(con, "u/c", score=2, label=2, reqs=[   # NOT wanted (👎)
        {"requirement": "Kubernetes", "verdict": "missing"}])
    rep = gaps.gap_report(cfg, con)
    assert rep["n_wanted"] == 2 and rep["n_jobs_with_reqs"] == 2
    by_req = {r["requirement"]: r for r in rep["rows"]}
    assert by_req["Kubernetes"]["missing"] == 2 and by_req["Kubernetes"]["jobs"] == 2  # a+b, not c
    assert by_req["ML"]["partial"] == 1 and by_req["BPMN"]["present"] == 1
    assert rep["rows"][0]["requirement"] == "Kubernetes"  # worst gap ranks first


def test_gap_report_empty_when_no_wanted(con, cfg):
    _seed(con, "u/x", score=2, label=2, reqs=[{"requirement": "X", "verdict": "missing"}])
    rep = gaps.gap_report(cfg, con)
    assert rep["n_wanted"] == 0 and rep["rows"] == [] and rep["reliable"] is False


def test_render_gaps_html(con, cfg):
    rep = {"n_wanted": 4, "n_jobs_with_reqs": 4, "reliable": True,
           "rows": [{"requirement": "Kubernetes", "missing": 3, "partial": 1, "present": 0, "jobs": 4}]}
    html = slate.render_gaps_html(rep)
    assert "Skill gaps" in html and "Kubernetes" in html and "3 ✗" in html
    empty = slate.render_gaps_html({"n_wanted": 0, "rows": []})
    assert "Mark" in empty and "/annotate" in empty


# ---------------------------------------------------- reconcile false gaps against ground truth

def _seed_candidate(con, *, education_level="bachelor", languages=None, skills=None):
    candidate._ensure_schema(con)
    prof = {"skills": skills or ["Tech stack: Python, SQL, Excel (Advanced), Power BI, Tableau"],
            "education_level": education_level,
            "education": "Bachelor’s Degree in Business Informatics",
            "languages": languages or {"en": "C1", "de": "A2", "ru": "C2"},
            "years_experience": 6}
    con.execute(
        "INSERT INTO candidate_profile (created_at, raw_input, profile_json, aspect_texts, doc_hash)"
        " VALUES (?,?,?,?,?)",
        (datetime.now(timezone.utc).isoformat(), "t", json.dumps(prof),
         json.dumps({"full_doc": "bachelor business informatics excel power bi"}), "h16"))
    con.commit()


def test_gap_report_reconciles_education_and_language(con, cfg):
    """qwen marks the candidate's real degree/languages 'missing' inconsistently; the /gaps page must
    not surface a requirement the structured profile provably meets (education ordinal / language CEFR)."""
    _seed_candidate(con)
    _seed(con, "u/1", score=5, label=5, reqs=[
        {"requirement": "Bachelor’s degree or PMI accreditation", "verdict": "missing"},   # has bachelor
        {"requirement": "Fluency in English", "verdict": "missing"},                        # en=C1
        {"requirement": "Knowledge of MS Office (Excel and PowerPoint)", "verdict": "missing"},  # skill (residual)
        {"requirement": "Master's degree in Computer Science", "verdict": "missing"},        # real: master>bachelor
        {"requirement": "Strong understanding of satellite systems", "verdict": "missing"}]) # real
    _seed(con, "u/2", score=4, label=4, reqs=[
        {"requirement": "Fluent in English", "verdict": "present"},
        {"requirement": "Fluency in English and Mandarin", "verdict": "missing"}])           # real: no Mandarin
    _seed(con, "u/3", score=3, applied=1, reqs=[
        {"requirement": "German C1 (written and spoken)", "verdict": "missing"},             # real: de=A2<C1
        {"requirement": "Fluency in English and Russian", "verdict": "present"}])            # en+ru held
    rep = gaps.gap_report(cfg, con)
    shown = {r["requirement"] for r in rep["rows"]}
    # reconciled away — candidate provably meets these
    for gone in ("Bachelor’s degree or PMI accreditation", "Fluency in English",
                 "Fluent in English", "Fluency in English and Russian"):
        assert gone not in shown, f"{gone!r} should be reconciled out"
    # genuine gaps kept — veto is precise, not a blanket suppressor
    for kept in ("Master's degree in Computer Science", "Strong understanding of satellite systems",
                 "German C1 (written and spoken)", "Fluency in English and Mandarin",
                 "Knowledge of MS Office (Excel and PowerPoint)"):  # skill = documented residual noise
        assert kept in shown, f"{kept!r} is a real gap and must stay"
    assert rep["n_reconciled"] >= 3


def test_gap_report_no_reconcile_without_profile(con, cfg):
    """No candidate profile → education unknown, no languages → nothing is reconciled (safe default)."""
    _seed(con, "u/1", score=5, label=5, reqs=[
        {"requirement": "Bachelor’s degree", "verdict": "missing"},
        {"requirement": "Fluency in English", "verdict": "missing"}])
    rep = gaps.gap_report(cfg, con)
    shown = {r["requirement"] for r in rep["rows"]}
    assert "Bachelor’s degree" in shown and "Fluency in English" in shown
    assert rep["n_reconciled"] == 0


def test_render_gaps_html_hides_reconciled(con, cfg):
    _seed_candidate(con)
    _seed(con, "u/1", score=5, label=5, reqs=[
        {"requirement": "Fluency in English", "verdict": "missing"},
        {"requirement": "Strong understanding of satellite systems", "verdict": "missing"}])
    _seed(con, "u/2", score=4, label=4, reqs=[{"requirement": "Data partnerships", "verdict": "missing"}])
    _seed(con, "u/3", score=4, label=4, reqs=[{"requirement": "Space-sector network", "verdict": "missing"}])
    html = slate.render_gaps_html(gaps.gap_report(cfg, con))
    assert "satellite" in html and "Data partnerships" in html
    assert "Fluency in English" not in html


# --------------------------------------------------------------------------- dates fallback

def test_posted_ago_first_seen_fallback():
    iso = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
    assert slate._posted_ago(iso, None).startswith("posted")
    # no posting date → fall back to first_seen, labelled "found"
    assert slate._posted_ago(None, iso).startswith("found")
    assert "3" in slate._posted_ago(None, iso)
    assert slate._posted_ago(None, None) == ""   # nothing → empty (no crash)


# --------------------------------------------------------------------------- themed score badge

def test_score_badge_themed_with_gradient():
    b1 = slate._score_badge(1)
    assert "💻🐀" in b1 and "linear-gradient" in b1 and "office mouse" in b1
    b5 = slate._score_badge(5)
    assert "👸✨🧚" in b5 and "linear-gradient" in b5 and "шабашка" in b5
    assert slate._score_badge(None) == ""


# --------------------------------------------------------------------------- freshness ceiling

def test_slate_freshness_ceiling_bounds_slate_not_annotate(con, cfg):
    old = (datetime.now(timezone.utc) - timedelta(days=20)).isoformat()
    fresh = _seed(con, "u/fresh", score=5)
    stale = _seed(con, "u/stale", score=5, last_seen=old)
    # daily slate: 14-day ceiling drops the 20-day-old job
    ids_slate = {it["vacancy_id"] for it in slate._load_scored(con, max_age_days=14)}
    assert fresh in ids_slate and stale not in ids_slate
    # /annotate (no ceiling) keeps the full backlog
    ids_all = {it["vacancy_id"] for it in slate._load_scored(con, max_age_days=None)}
    assert fresh in ids_all and stale in ids_all
