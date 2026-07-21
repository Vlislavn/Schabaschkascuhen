"""Tests for agent_discovery.py and investigate.py (mocked agent).

kl_agent_builder is not installed in the test env — the import path is mocked.
"""
from __future__ import annotations

import json

import pytest

from schabasch import db, investigate
from schabasch.models import Status
from tests.conftest import make_card


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed_scored(con, url: str, company: str = "ACME", score: int = 4) -> int:
    vid = db.upsert_vacancy(con, {
        "source": "indeed", "url": url,
        "title": "Senior Business Analyst", "company": company,
        "city": "Frankfurt", "description": "x" * 600,
    })
    card = make_card(company=company)
    db.set_status(con, vid, Status.DESCRIBED, card_json=json.dumps(card))
    db.insert_judge_score(con, vid, {
        "score": score, "why_tag": None, "why_freetext": None,
        "explanation": "good", "model": "qwen3:8b",
        "model_digest": "d", "rubric_version": "v1", "fewshot_hash": "h",
    })
    db.set_status(con, vid, Status.SCORED)
    return vid


# ---------------------------------------------------------------------------
# agent_discovery — mocked agent
# ---------------------------------------------------------------------------

def test_agent_discovery_no_kl_agent_builder(con, cfg, monkeypatch):
    """When kl_agent_builder is not installed, scrape() returns graceful error dict."""
    from schabasch.sources import agent_discovery

    def _fail(*_, **__):
        raise ImportError("kl_agent_builder not installed")

    monkeypatch.setattr("schabasch.agent_runtime.build_agent", _fail)
    result = agent_discovery.scrape(cfg, con)
    assert result.get("upserted", 0) == 0
    assert result.get("errors", 0) >= 1


def test_agent_discovery_with_mocked_agent(con, cfg, monkeypatch):
    """Mocked agent returns a JSON posting list → vacancies upserted with source='agent'."""
    from schabasch.sources import agent_discovery

    FAKE_POSTINGS = [
        {"title": "BA Lead", "company": "TechCorp", "url": "https://techcorp.de/jobs/1",
         "city": "Heidelberg", "description": "We need a senior BA."},
        {"title": "Product Manager", "company": "DataCo", "url": "https://dataco.com/jobs/2",
         "city": "Frankfurt", "description": "Digital transformation PM role."},
    ]

    def _fake_agent(cfg, *, system_prompt, max_turns=8):
        def _run(task: str, context=None) -> str:
            return json.dumps(FAKE_POSTINGS)
        return _run

    monkeypatch.setattr("schabasch.agent_runtime.build_agent", _fake_agent)
    result = agent_discovery.scrape(cfg, con, max_results=10)
    assert result["found"] == 2
    assert result["upserted"] == 2
    assert result["errors"] == 0

    # Verify source="agent" rows exist
    rows = con.execute(
        "SELECT url, source FROM vacancy WHERE source = 'agent'"
    ).fetchall()
    assert len(rows) == 2


def test_agent_discovery_skips_empty_url(con, cfg, monkeypatch):
    """Postings without a URL should be counted as errors, not upserted."""
    from schabasch.sources import agent_discovery

    FAKE_POSTINGS = [
        {"title": "BA", "company": "ACME", "url": "", "city": "Frankfurt"},
        {"title": "PM", "url": "https://example.com/1", "company": "X", "city": "HH"},
    ]

    def _fake_agent(cfg, *, system_prompt, max_turns=8):
        def _run(task: str, context=None) -> str:
            return json.dumps(FAKE_POSTINGS)
        return _run

    monkeypatch.setattr("schabasch.agent_runtime.build_agent", _fake_agent)
    result = agent_discovery.scrape(cfg, con, max_results=10)
    assert result["upserted"] == 1
    assert result["errors"] == 1


def test_agent_discovery_bad_json_graceful(con, cfg, monkeypatch):
    """Malformed agent output → returns errors=1, no vacancies upserted."""
    from schabasch.sources import agent_discovery

    def _fake_agent(cfg, *, system_prompt, max_turns=8):
        def _run(task: str, context=None) -> str:
            return "not valid json [[[{"
        return _run

    monkeypatch.setattr("schabasch.agent_runtime.build_agent", _fake_agent)
    result = agent_discovery.scrape(cfg, con, max_results=10)
    assert result["upserted"] == 0
    assert result["errors"] >= 1


# ---------------------------------------------------------------------------
# investigate.py
# ---------------------------------------------------------------------------

def test_investigate_schema_idempotent(con):
    investigate._ensure_schema(con)
    investigate._ensure_schema(con)
    tables = {r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    assert "investigation" in tables


def test_investigate_no_scored_vacancies(con, cfg, monkeypatch):
    """With no SCORED vacancies, investigate returns zeros."""
    # No need to even call build_agent
    result = investigate.investigate_top(cfg, con, slate_date="2026-06-14")
    assert result == {"investigated": 0, "ok": 0, "stale": 0, "suspect": 0, "errors": 0}


def test_investigate_kl_agent_builder_missing(con, cfg, monkeypatch):
    """When kl_agent_builder not installed, returns errors=1."""
    _seed_scored(con, "u/i1")

    def _fail(*_, **__):
        raise ImportError("kl_agent_builder not installed")

    monkeypatch.setattr("schabasch.agent_runtime.build_agent", _fail)
    result = investigate.investigate_top(cfg, con, slate_date="2026-06-14")
    assert result["errors"] >= 1
    assert result["investigated"] == 0


def test_investigate_mocked_agent_populates_sidecar(con, cfg, monkeypatch):
    """Mocked agent → investigation row written with correct verdict."""
    vid = _seed_scored(con, "u/inv1", company="TechGmbH", score=5)

    enrichment = {
        "is_temp_agency": False, "company_known": True,
        "company_size": "large", "verified_requirements": "Python, SQL, BPMN",
        "salary_eur_min": 70000, "salary_eur_max": 90000,
        "english_team_signal": True, "verdict": "ok",
        "notes": "Great role at TechGmbH.",
    }

    def _fake_agent(cfg, *, system_prompt, max_turns=8):
        def _run(task: str, context=None) -> str:
            return json.dumps(enrichment)
        return _run

    monkeypatch.setattr("schabasch.agent_runtime.build_agent", _fake_agent)
    # still_open comes from the DETERMINISTIC check now, not the agent — mock it (no network in tests)
    monkeypatch.setattr("schabasch.investigate._check_still_open", lambda url, source, refnr: True)
    result = investigate.investigate_top(cfg, con, slate_date="2026-06-14", top_n=3)
    assert result["investigated"] == 1
    assert result.get("ok", 0) == 1

    row = con.execute(
        "SELECT * FROM investigation WHERE vacancy_id = ?", (vid,)
    ).fetchone()
    assert row is not None
    assert row["verdict"] == "ok"
    loaded = json.loads(row["enrichment_json"])
    assert loaded["still_open"] is True   # set by the deterministic override, not the agent


def test_investigate_deterministic_closed_not_a_guess(con, cfg, monkeypatch):
    """A deterministically-gone listing (still_open=False) is recorded; the agent no longer guesses
    closure, so the 'stale' verdict is gone — closure is the HTTP/AA check (_check_still_open)."""
    vid = _seed_scored(con, "u/inv2", score=4)
    enrichment = {"verdict": "ok", "notes": "Real role.", "company_known": True}

    def _fake_agent(cfg, *, system_prompt, max_turns=8):
        def _run(task: str, context=None) -> str:
            return json.dumps(enrichment)
        return _run

    monkeypatch.setattr("schabasch.agent_runtime.build_agent", _fake_agent)
    monkeypatch.setattr("schabasch.investigate._check_still_open", lambda url, source, refnr: False)
    result = investigate.investigate_top(cfg, con, slate_date="2026-06-14", top_n=1)
    assert result["investigated"] == 1
    loaded = json.loads(con.execute(
        "SELECT enrichment_json FROM investigation WHERE vacancy_id = ?", (vid,)).fetchone()[0])
    assert loaded["still_open"] is False and loaded["verdict"] == "ok"


def test_investigate_agent_error_increments_errors(con, cfg, monkeypatch):
    _seed_scored(con, "u/inv3", score=3)

    def _fake_agent(cfg, *, system_prompt, max_turns=8):
        def _run(task: str, context=None) -> str:
            return "not json"
        return _run

    monkeypatch.setattr("schabasch.agent_runtime.build_agent", _fake_agent)
    result = investigate.investigate_top(cfg, con, slate_date="2026-06-14", top_n=1)
    assert result["errors"] >= 1
    assert result["investigated"] == 0


def test_patch_feature_row_requirements_verified_from_contract_key(con, cfg):
    """Regression: _patch_feature_row must read the contract key 'verified_requirements'
    (NOT 'requirements' and NOT 'still_open') to set requirements_verified — otherwise the
    triage feature is hardwired to a constant and the gate learns nothing from investigation."""
    import json as _json
    from schabasch import features, investigate
    features._ensure_schema(con)
    vid = _seed_scored(con, "u/patch1")
    con.execute(
        "INSERT OR REPLACE INTO vacancy_feature (vacancy_id, match_score, feature_json, computed_at) "
        "VALUES (?, ?, ?, datetime('now'))",
        (vid, 0.5, _json.dumps({"requirements_verified": 0.0, "company_known": 0.0})),
    )
    con.commit()

    # verified_requirements present → 1.0, even though still_open is False. company_known now reflects
    # INDEPENDENT verification (W4: validate_company sets company_verified), not the agent's self-claim.
    investigate._patch_feature_row(con, vid, {
        "verified_requirements": "Python, SQL, BPMN", "still_open": False,
        "company_verified": True, "german_rooted": True})
    feat = _json.loads(con.execute(
        "SELECT feature_json FROM vacancy_feature WHERE vacancy_id=?", (vid,)).fetchone()[0])
    assert feat["requirements_verified"] == 1.0
    assert feat["company_known"] == 1.0       # company_verified → company_known feature
    assert feat["german_rooted"] == 1.0       # W4: integration signal patched

    # verified_requirements absent → 0.0, even though still_open is True
    investigate._patch_feature_row(con, vid, {"still_open": True})
    feat = _json.loads(con.execute(
        "SELECT feature_json FROM vacancy_feature WHERE vacancy_id=?", (vid,)).fetchone()[0])
    assert feat["requirements_verified"] == 0.0


def test_quality_gate_rejects_search_pages_and_placeholders():
    """The discovery quality gate must reject aggregator search/listing URLs + placeholder
    company names (the actual noise qwen3:8b produced), while passing real direct postings."""
    from schabasch.sources.agent_discovery import _is_search_or_aggregator_url, _is_placeholder_company
    # real noise observed live → must be rejected
    bad_urls = [
        "https://www.stepstone.de/jobs/machine-learning-engineer/in-heidelberg",
        "https://www.glassdoor.com/Job/heidelberg-process-engineer-jobs-SRCH_IL.0,10_IC2563483.htm",
        "https://www.linkedin.com/jobs/system-engineer-jobs-frankfurt",
        "https://englishjobs.de/in/frankfurt-am-main/systems_engineer",
        "https://example.com/search?q=engineer",
        "https://board.de/suche?keywords=ml",
    ]
    for u in bad_urls:
        assert _is_search_or_aggregator_url(u), f"should reject: {u}"
    # real direct postings → must pass
    good_urls = [
        "https://de.indeed.com/viewjob?jk=89e82f633d8e516a",
        "https://careers.airbus.com/en/jobs/ml-engineer-12345",
        "https://www.rheinmetall.com/de/karriere/stellenangebot/system-engineer-9981",
    ]
    for u in good_urls:
        assert not _is_search_or_aggregator_url(u), f"should pass: {u}"
    # placeholder employer names → rejected; real ones → pass
    for c in ("LinkedIn Employer", "Glassdoor Employer", "Stepstone Partner", "Indeed", "Glassdoor"):
        assert _is_placeholder_company(c), f"should reject company: {c}"
    for c in ("Airbus", "Rheinmetall", "OHB SE", "Bosch"):
        assert not _is_placeholder_company(c), f"should pass company: {c}"


def test_check_still_open_is_deterministic_never_a_guess(monkeypatch):
    """_check_still_open maps HTTP status / AA re-query → open/closed/UNKNOWN; a blocked or
    timed-out link is UNKNOWN (None), never a false 'closed'."""
    from schabasch import investigate
    monkeypatch.setattr("schabasch.investigate.http_get_status", lambda url, **kw: 200)
    assert investigate._check_still_open("http://x", "indeed", None) is True
    monkeypatch.setattr("schabasch.investigate.http_get_status", lambda url, **kw: 404)
    assert investigate._check_still_open("http://x", "indeed", None) is False
    monkeypatch.setattr("schabasch.investigate.http_get_status", lambda url, **kw: 403)
    assert investigate._check_still_open("http://x", "linkedin", None) is None   # blocked → unknown
    monkeypatch.setattr("schabasch.investigate.http_get_status", lambda url, **kw: None)
    assert investigate._check_still_open("http://x", "indeed", None) is None     # timeout → unknown
    assert investigate._check_still_open("", "indeed", None) is None             # no url → unknown
    import schabasch.sources.arbeitsagentur as aa
    monkeypatch.setattr(aa, "check_open", lambda refnr: False)
    assert investigate._check_still_open("http://x", "arbeitsagentur", "R123") is False  # routed to API


def test_arbeitsagentur_check_open(monkeypatch):
    """AA re-query by refnr: 200→open, 404/410(permanent)→closed, timeout/other→unknown."""
    from schabasch.llm import LLMError
    from schabasch.models import ErrorClass
    from schabasch.sources import arbeitsagentur as aa
    monkeypatch.setattr(aa, "http_get_json", lambda *a, **k: {"refnr": "R1"})
    assert aa.check_open("R1") is True
    monkeypatch.setattr(aa, "http_get_json",
                        lambda *a, **k: (_ for _ in ()).throw(LLMError(ErrorClass.HTTP_ERROR, "HTTP 404 (permanent)")))
    assert aa.check_open("R1") is False
    monkeypatch.setattr(aa, "http_get_json",
                        lambda *a, **k: (_ for _ in ()).throw(LLMError(ErrorClass.TIMEOUT, "timeout")))
    assert aa.check_open("R1") is None
    assert aa.check_open("") is None
