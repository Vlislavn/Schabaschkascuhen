"""investigate_one: idempotent single-card entrypoint (progressive path). No agent invoked here —
only the cache-hit and missing-vacancy branches, which must not load any model."""
from __future__ import annotations

from datetime import datetime, timezone

from schabasch import db, investigate
from tests.conftest import seed_scored


def test_investigate_one_cached(cfg, con):
    vid = seed_scored(con, "u/inv", score=5, company="Co")
    investigate._ensure_schema(con)
    con.execute(
        "INSERT INTO investigation (vacancy_id, enrichment_json, verdict, investigated_at) "
        "VALUES (?, ?, ?, ?)",
        (vid, "{}", "ok", datetime.now(timezone.utc).isoformat()),
    )
    con.commit()
    # already investigated → returns 'cached' without building/running the agent
    assert investigate.investigate_one(cfg, con, vid) == "cached"


def test_investigate_one_missing_vacancy(cfg, con):
    investigate._ensure_schema(con)
    assert investigate.investigate_one(cfg, con, 999999) == "missing"


# ---------------------------------------------------------------------------
# Redundant-work fixes: investigate_top skip + persistent employer knowledge base
# ---------------------------------------------------------------------------

def _fake_agent_json(**over):
    import json
    base = {"is_temp_agency": False, "company_known": True, "company_size": "mid",
            "company_description": "A real company.", "german_rooted": True,
            "verified_requirements": "Python, SQL", "salary_eur_min": None, "salary_eur_max": None,
            "english_team_signal": True, "verdict": "ok", "notes": "ok"}
    base.update(over)
    return json.dumps(base)


def _mock_agent(monkeypatch, tasks):
    """Replace the real ReAct agent with a counter/recorder — no model, no network."""
    from schabasch import agent_runtime
    from schabasch.browsing import entity as _entity
    monkeypatch.setattr(_entity, "resolve", lambda name, **kw: None)          # no Wikidata network
    monkeypatch.setattr(investigate, "_wikipedia_company", lambda name, lang="de": None)  # no network
    monkeypatch.setattr(investigate, "_check_still_open", lambda url, src, refnr: True)
    monkeypatch.setattr(agent_runtime, "build_agent",
                        lambda cfg, system_prompt, max_turns=None: object())
    monkeypatch.setattr(agent_runtime, "run_agent",
                        lambda agent_fn, task: (tasks.append(task) or _fake_agent_json()))


def test_investigate_top_skips_already_investigated(cfg, con, monkeypatch):
    """① The agent (the most expensive op) must NOT re-run on a vacancy already in `investigation`."""
    investigate._ensure_schema(con)
    done = seed_scored(con, "u/done", score=5, company="DoneCo")
    fresh = seed_scored(con, "u/fresh", score=4, company="FreshCo")
    con.execute("INSERT INTO investigation (vacancy_id, enrichment_json, verdict, investigated_at) "
                "VALUES (?,?,?,?)", (done, "{}", "ok", datetime.now(timezone.utc).isoformat()))
    con.commit()
    tasks: list = []
    _mock_agent(monkeypatch, tasks)
    res = investigate.investigate_top(cfg, con, slate_date="2026-06-17", top_n=6)
    assert res["investigated"] == 1            # only the un-investigated vacancy
    assert len(tasks) == 1                       # agent ran ONCE, not on the already-done job
    assert con.execute("SELECT 1 FROM investigation WHERE vacancy_id=?", (fresh,)).fetchone()


def test_company_researched_once_then_reused(cfg, con, monkeypatch):
    """Persistent employer DB + hard-before-soft: two vacancies at the SAME company → the keyless
    identity is resolved ONCE; the 2nd reuses the stored facts and the agent is fed the verified
    identity pre-run (one company_knowledge row, resolve called once)."""
    from schabasch.browsing import entity as _entity
    seed_scored(con, "u/a", score=5, company="SAP SE")
    seed_scored(con, "u/b", score=4, company="SAP SE")   # same employer, lower judge score
    tasks: list = []
    _mock_agent(monkeypatch, tasks)
    calls = {"n": 0}

    def fake_resolve(name, **kw):
        calls["n"] += 1
        return {"qid": "Q1", "label": "SAP SE", "description": "German software company",
                "official_site": "https://www.sap.com/", "country": "Germany",
                "employees": None, "inception": None, "wikidata_url": "https://www.wikidata.org/wiki/Q1"}

    monkeypatch.setattr(_entity, "resolve", fake_resolve)   # overrides the None set by _mock_agent
    investigate.investigate_top(cfg, con, slate_date="2026-06-17", top_n=6)
    n = con.execute("SELECT COUNT(*) c FROM company_knowledge").fetchone()["c"]
    assert n == 1                                            # employer stored once, not per-vacancy
    assert calls["n"] == 1                                   # identity resolved ONCE (2nd reused the KB)
    assert any("VERIFIED EMPLOYER" in t for t in tasks)     # the resolved identity is fed to the agent


def test_name_matches_rejects_fuzzy_wrong_company():
    """Root-cause guard for bare-opensearch wrong matches: every significant query token must appear
    in the article title; near-but-different names are rejected (the live agent fills them instead)."""
    m = investigate._name_matches
    assert m("Terma Group", "Therme Group") is False                       # spa, not the defense firm
    assert m("Phoenix Medical", "Phoenix Media/Communications Group") is False
    assert m("Merz Therapeutics", "Merz Pharma") is False                  # conservative: 2nd token differs
    assert m("ABB", "ABB Group") is True
    assert m("Air Liquide", "Air Liquide") is True
    assert m("Heidelberg Materials AG", "Heidelberg Materials") is True
    assert m("IQVIA", "IQVIA") is True
    assert m("", "Anything") is True                                       # empty query never blocks


def test_wikipedia_company_rejects_wrong_title(monkeypatch):
    """The guard fires inside _wikipedia_company: a fuzzy wrong opensearch title → None, and the
    summary fetch is never made (no false validation, no wasted call)."""
    calls = {"n": 0}

    def fake_get(url, **kw):
        calls["n"] += 1                                                    # opensearch: [q,[titles],[descs],[urls]]
        return ["Terma Group", ["Therme Group"], ["a spa company"],
                ["https://de.wikipedia.org/wiki/Therme_Group"]]

    monkeypatch.setattr(investigate, "http_get_json", fake_get)
    assert investigate._wikipedia_company("Terma Group", lang="de") is None
    assert calls["n"] == 1                                                 # rejected post-opensearch; no summary call
