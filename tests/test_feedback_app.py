"""Страница фидбека (FastAPI TestClient) на временной файловой БД."""
from __future__ import annotations

import json

import pytest

from schabasch import db
from schabasch.models import Status


@pytest.fixture
def file_cfg(cfg, tmp_path):
    cfg = dict(cfg)
    cfg["paths"] = dict(cfg["paths"])
    cfg["paths"]["db"] = str(tmp_path / "t.sqlite3")
    return cfg


def _seed_scored(con, url, *, score, company="ACME"):
    vid = db.upsert_vacancy(con, {"source": "indeed", "url": url, "title": f"T {url}",
                                  "company": company, "city": "Frankfurt", "description": "x" * 500})
    card = dict(role="r", company=company, domain="aerospace", city="Frankfurt",
                work_mode="hybrid", language_posting="en", language_reality="en",
                integration_potential=2, summary_2lines="a\nb", slop_score=5, temp_agency_guess=False)
    db.set_status(con, vid, Status.SCORED, card_json=json.dumps(card, ensure_ascii=False))
    db.insert_judge_score(con, vid, {"score": score, "model": "qwen3:8b", "rubric_version": "v1",
                                     "explanation": "e"})
    return vid


def _client(file_cfg):
    from fastapi.testclient import TestClient
    from schabasch import feedback_app
    return TestClient(feedback_app.create_app(file_cfg))


def test_index_renders_slate(file_cfg):
    con = db.connect(file_cfg["paths"]["db"])
    for i in range(5):
        _seed_scored(con, f"u/{i}", score=4 + (i % 2), company=f"Co{i}")
    con.close()
    r = _client(file_cfg).get("/")
    assert r.status_code == 200
    assert "Slate" in r.text and "/feedback" in r.text


def test_feedback_good_writes_label(file_cfg):
    con = db.connect(file_cfg["paths"]["db"])
    vid = _seed_scored(con, "u/fb", score=5)
    con.close()
    client = _client(file_cfg)
    client.get("/")  # построить slate
    r = client.post("/feedback", json={"vacancy_id": vid, "action": "good"})
    assert r.status_code == 200 and r.json()["ok"] is True
    con = db.connect(file_cfg["paths"]["db"])
    row = con.execute("SELECT score_1_5, applied, source FROM label WHERE vacancy_id=?",
                      (vid,)).fetchone()
    assert row["score_1_5"] == 4 and row["applied"] == 0 and row["source"] == "slate"
    assert con.execute("SELECT status FROM vacancy WHERE id=?", (vid,)).fetchone()[0] == \
        Status.LABELED.value
    con.close()


def test_feedback_applied_is_flag_not_fabricated_score(file_cfg):
    """'applied' is a flag ON TOP of the score (USE_CASE), not a 5-star rating. With no prior
    rating, score_1_5 stays NULL — we never fabricate a 5 for an unrated job."""
    con = db.connect(file_cfg["paths"]["db"])
    vid = _seed_scored(con, "u/ap", score=5)
    con.close()
    client = _client(file_cfg)
    client.get("/")
    r = client.post("/feedback", json={"vacancy_id": vid, "action": "applied"})
    assert r.status_code == 200
    con = db.connect(file_cfg["paths"]["db"])
    row = con.execute("SELECT score_1_5, applied FROM label WHERE vacancy_id=?", (vid,)).fetchone()
    assert row["applied"] == 1 and row["score_1_5"] is None
    con.close()


def test_feedback_applied_preserves_prior_rating(file_cfg):
    """Rate first (👍=4), then click applied → applied=1 AND the 4 is preserved (not clobbered)."""
    con = db.connect(file_cfg["paths"]["db"])
    vid = _seed_scored(con, "u/ap2", score=3)
    con.close()
    client = _client(file_cfg)
    client.post("/feedback", json={"vacancy_id": vid, "action": "good"})
    client.post("/feedback", json={"vacancy_id": vid, "action": "applied"})
    con = db.connect(file_cfg["paths"]["db"])
    row = con.execute("SELECT score_1_5, applied FROM label WHERE vacancy_id=?", (vid,)).fetchone()
    assert row["applied"] == 1 and row["score_1_5"] == 4
    con.close()


def test_feedback_note_roundtrips_to_why_freetext(file_cfg):
    """WS2: a free-text note on a rating lands in label.why_freetext (→ judge few-shot)."""
    con = db.connect(file_cfg["paths"]["db"])
    vid = _seed_scored(con, "u/note", score=5)
    con.close()
    client = _client(file_cfg)
    client.get("/")
    r = client.post("/feedback", json={"vacancy_id": vid, "action": "bad",
                                       "note": "Master Data ≠ degree, неверная интерпретация"})
    assert r.status_code == 200
    con = db.connect(file_cfg["paths"]["db"])
    row = con.execute("SELECT score_1_5, why_freetext FROM label WHERE vacancy_id=?", (vid,)).fetchone()
    assert row["score_1_5"] == 2 and "Master Data" in row["why_freetext"]
    con.close()


def test_feedback_applied_note_keeps_prior_score(file_cfg):
    """WS2: applied + note keeps the prior rating (COALESCE) and stores the note."""
    con = db.connect(file_cfg["paths"]["db"])
    vid = _seed_scored(con, "u/an", score=3)
    con.close()
    client = _client(file_cfg)
    client.post("/feedback", json={"vacancy_id": vid, "action": "good"})       # → score 4
    client.post("/feedback", json={"vacancy_id": vid, "action": "applied", "note": "люблю lead"})
    con = db.connect(file_cfg["paths"]["db"])
    row = con.execute("SELECT score_1_5, applied, why_freetext FROM label WHERE vacancy_id=?",
                      (vid,)).fetchone()
    assert row["applied"] == 1 and row["score_1_5"] == 4 and "lead" in row["why_freetext"]
    con.close()


def test_feedback_empty_note_does_not_wipe_prior(file_cfg):
    """An empty note on a later click must not erase a previously-saved note (COALESCE)."""
    con = db.connect(file_cfg["paths"]["db"])
    vid = _seed_scored(con, "u/keep", score=3)
    con.close()
    client = _client(file_cfg)
    client.post("/feedback", json={"vacancy_id": vid, "action": "good", "note": "важная заметка"})
    client.post("/feedback", json={"vacancy_id": vid, "action": "star"})   # no note
    con = db.connect(file_cfg["paths"]["db"])
    row = con.execute("SELECT score_1_5, why_freetext FROM label WHERE vacancy_id=?", (vid,)).fetchone()
    assert row["score_1_5"] == 5 and row["why_freetext"] == "важная заметка"
    con.close()


def test_feedback_unknown_vacancy_404(file_cfg):
    """Feedback for a non-existent vacancy is rejected (FKs off → would corrupt golden table)."""
    con = db.connect(file_cfg["paths"]["db"])
    _seed_scored(con, "u/exists", score=3)
    con.close()
    r = _client(file_cfg).post("/feedback", json={"vacancy_id": 999999, "action": "good"})
    assert r.status_code == 404


def test_feedback_bad_action_400(file_cfg):
    con = db.connect(file_cfg["paths"]["db"])
    _seed_scored(con, "u/x", score=3)
    con.close()
    r = _client(file_cfg).post("/feedback", json={"vacancy_id": 1, "action": "nonsense"})
    assert r.status_code == 400


def test_fetch_single_flight_409(file_cfg):
    """POST /fetch while a fetch is running → 409 (single-flight; prevents double model-load)."""
    import schabasch.feedback_app as fa
    client = _client(file_cfg)
    fa._FETCH_LOCK.acquire()   # simulate a fetch already running
    try:
        r = client.post("/fetch")
        assert r.status_code == 409 and r.json()["ok"] is False
    finally:
        fa._FETCH_LOCK.release()


def test_fetch_runs_tick_async(file_cfg, monkeypatch):
    """POST /fetch returns immediately, runs the (mocked) tick in a background thread, releases lock."""
    import time
    import schabasch.pipeline as pipe
    calls = {"n": 0}
    monkeypatch.setattr(pipe, "nightly_tick",
                        lambda cfg, con, **kw: calls.__setitem__("n", calls["n"] + 1) or {"slate": {"size": 1}})
    client = _client(file_cfg)
    r = client.post("/fetch")
    assert r.status_code == 200 and r.json()["ok"] is True
    for _ in range(60):                      # wait for the bg worker
        if not client.get("/fetch-status").json()["running"]:
            break
        time.sleep(0.05)
    assert calls["n"] == 1
    import schabasch.feedback_app as fa
    assert fa._FETCH_LOCK.acquire(blocking=False)   # lock released after the run
    fa._FETCH_LOCK.release()


def test_fetch_releases_lock_on_worker_error(file_cfg, monkeypatch):
    """A failure in the background worker (incl. connect/import before the run) must release the
    single-flight lock + clear `running` — else /fetch wedges into a permanent 409."""
    import time
    import schabasch.feedback_app as fa
    import schabasch.pipeline as pipe

    def _boom(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(pipe, "nightly_tick", _boom)
    client = _client(file_cfg)
    assert client.post("/fetch").status_code == 200
    s = {"running": True}
    for _ in range(80):
        s = client.get("/fetch-status").json()
        if not s["running"]:
            break
        time.sleep(0.05)
    assert s["running"] is False and s["error"] and "boom" in s["error"]   # surfaced, not hung
    assert fa._FETCH_LOCK.acquire(blocking=False)   # lock released → not wedged
    fa._FETCH_LOCK.release()
    assert client.post("/fetch").status_code == 200   # a new fetch is possible (no permanent 409)
    for _ in range(80):
        if not client.get("/fetch-status").json()["running"]:
            break
        time.sleep(0.05)


def test_funnel_surfaces_dedup(file_cfg):
    """/funnel parses the latest dedup_fuzzy funnel detail into a visible candidate list."""
    con = db.connect(file_cfg["paths"]["db"])
    db.log_funnel(con, "dedup_fuzzy", 2,
                  detail='[{"a":1,"sa":"indeed","b":2,"sb":"linkedin","sim":100.0,"co":"X","ta":"T","tb":"T"}]')
    con.close()
    body = _client(file_cfg).get("/funnel").json()
    assert body["dedup_count"] == 2 and len(body["dedup_candidates"]) == 1
    assert body["dedup_candidates"][0]["sim"] == 100.0


def test_funnel_endpoint(file_cfg):
    con = db.connect(file_cfg["paths"]["db"])
    _seed_scored(con, "u/f", score=4)
    db.log_funnel(con, "scrape", 5, "indeed")
    db.log_canary(con, "indeed", "ok", 5)
    con.close()
    r = _client(file_cfg).get("/funnel")
    assert r.status_code == 200
    body = r.json()
    assert "funnel" in body and "canaries" in body and "status_counts" in body
