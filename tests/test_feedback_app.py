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


def test_feedback_direction_boosts_domain(file_cfg):
    """🧭 direction: a low score removes THIS vacancy from the slate, while a positive role-fit row
    (label_role fits=1) is the domain/direction boost signal — the 'wrong specifics, right direction' fix."""
    con = db.connect(file_cfg["paths"]["db"])
    vid = _seed_scored(con, "u/dir", score=4, company="SpaceCo")
    con.close()
    client = _client(file_cfg)
    client.get("/")
    r = client.post("/feedback", json={"vacancy_id": vid, "action": "direction"})
    assert r.status_code == 200 and r.json()["ok"] is True
    con = db.connect(file_cfg["paths"]["db"])
    lab = con.execute("SELECT score_1_5, source FROM label WHERE vacancy_id=?", (vid,)).fetchone()
    assert lab["score_1_5"] == 2 and lab["source"] == "slate"        # not-a-match → excluded from re-show
    rf = con.execute("SELECT fits FROM label_role WHERE vacancy_id=? AND source='slate'", (vid,)).fetchone()
    assert rf is not None and rf["fits"] == 1                         # positive direction signal (boost domain)
    assert con.execute("SELECT status FROM vacancy WHERE id=?", (vid,)).fetchone()[0] == Status.LABELED.value
    con.close()


def test_backlog_json_returns_pool(file_cfg):
    """/backlog.json = the judged-but-unrated pool the bot's /more walks (beyond the ≤10 daily slate)."""
    file_cfg = {**file_cfg, "judge": {**file_cfg.get("judge", {}), "rubric_version": "v1"}}  # match _seed
    con = db.connect(file_cfg["paths"]["db"])
    for i in range(3):
        _seed_scored(con, f"u/bk{i}", score=4, company=f"Bk{i}")
    con.close()
    r = _client(file_cfg).get("/backlog.json")
    assert r.status_code == 200
    body = r.json()
    assert "cards" in body and "total" in body and body["total"] >= 3


def test_feedback_dry_acks_but_does_not_persist(file_cfg):
    """`serve --dry`: /feedback returns ok+dry but writes NOTHING to the label table (golden safe)."""
    from fastapi.testclient import TestClient
    from schabasch import feedback_app
    con = db.connect(file_cfg["paths"]["db"])
    vid = _seed_scored(con, "u/dry", score=5)
    con.close()
    client = TestClient(feedback_app.create_app(file_cfg, dry=True))
    r = client.post("/feedback", json={"vacancy_id": vid, "action": "good", "note": "x"})
    assert r.status_code == 200 and r.json().get("dry") is True
    # bad action is still rejected in dry mode (validation runs before the dry short-circuit)
    assert client.post("/feedback", json={"vacancy_id": vid, "action": "nope"}).status_code == 400
    con = db.connect(file_cfg["paths"]["db"])
    assert con.execute("SELECT COUNT(*) FROM label WHERE vacancy_id=?", (vid,)).fetchone()[0] == 0
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


def test_refetch_guard_skips_when_fresh_fetches_when_stale(file_cfg):
    """Startup fetch is skipped when a fetch (the 'slate' funnel stage) completed < refetch_after_hours
    ago; runs when stale or never; --refetch forces."""
    from datetime import datetime, timedelta, timezone
    from schabasch import feedback_app
    con = db.connect(file_cfg["paths"]["db"])
    cfg = dict(file_cfg)
    cfg["serve"] = {"refetch_after_hours": 12}

    assert feedback_app._refetch_guard(con, cfg, force=False)[0] is False   # never fetched → fetch

    def _log_slate(hours_ago):
        ts = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat(timespec="seconds")
        con.execute("INSERT INTO funnel_log (run_at, stage, count) VALUES (?,?,?)", (ts, "slate", 9))
        con.commit()

    _log_slate(1)                                                            # fresh (1h ago)…
    # …but an EMPTY DB never skips: the stamp can come from a mere GET /slate.json on a
    # never-fetched user (live 2026-07-03 — a new user's first fetch was wrongly skipped)
    assert feedback_app._refetch_guard(con, cfg, force=False)[0] is False
    _seed_scored(con, "u/guard", score=4)                                    # a real fetch leaves data
    skip, msg = feedback_app._refetch_guard(con, cfg, force=False)
    assert skip is True and "skipping startup fetch" in msg
    assert feedback_app._refetch_guard(con, cfg, force=True)[0] is False     # --refetch overrides

    con.execute("DELETE FROM funnel_log")                                    # only a stale entry remains
    con.commit()
    _log_slate(20)                                                           # last fetch 20h ago → stale
    assert feedback_app._refetch_guard(con, cfg, force=False)[0] is False
    con.close()


def test_fetch_status_and_index_surface_skipped_refetch(file_cfg):
    """When the startup fetch was skipped (data fresh), /fetch-status exposes it and the index page
    shows the 'data not updated — refetch?' alert."""
    import schabasch.feedback_app as fa
    from tests.conftest import seed_scored
    con = db.connect(file_cfg["paths"]["db"])
    seed_scored(con, "u/x", score=5, company="Co")
    con.close()
    fa._fetch_state().update(fetch_skipped=True, data_age_hours=3.0)
    try:
        client = _client(file_cfg)
        st = client.get("/fetch-status").json()
        assert st["fetch_skipped"] is True and st["data_age_hours"] == 3.0
        html = client.get("/").text
        assert "не обновлялись" in html and "обновления" in html      # staleness + refetch hint
    finally:
        fa._fetch_state().update(fetch_skipped=False, data_age_hours=None)


def test_role_feedback_writes_sidecar(file_cfg):
    """POST /role-feedback classifies the role server-side and records it in label_role (golden)."""
    from tests.conftest import seed_scored
    from schabasch import role_feedback
    con = db.connect(file_cfg["paths"]["db"])
    vid = seed_scored(con, "u/role", score=4, company="Co", title="Senior Software Engineer")
    con.close()
    r = _client(file_cfg).post("/role-feedback", json={"vacancy_id": vid, "fits": False})
    assert r.status_code == 200 and r.json()["ok"] is True
    assert r.json()["role_kind"] == "hands_on_engineer"   # classified server-side
    con = db.connect(file_cfg["paths"]["db"])
    assert role_feedback.veto_map(con).get(vid) == 0      # wrong role recorded
    con.close()


def test_feedback_note_abgelaufen_expires(file_cfg):
    """5b: a note flagging the posting as expired EXPIRES it (hard removal), even with a 👍."""
    con = db.connect(file_cfg["paths"]["db"])
    vid = _seed_scored(con, "u/exp", score=4)
    con.close()
    r = _client(file_cfg).post("/feedback", json={
        "vacancy_id": vid, "action": "good",
        "note": "Diese Stellenanzeige ist auf Indeed abgelaufen, не могу податься"})
    assert r.status_code == 200 and r.json()["ok"] is True
    con = db.connect(file_cfg["paths"]["db"])
    st = con.execute("SELECT status FROM vacancy WHERE id=?", (vid,)).fetchone()[0]
    con.close()
    assert st == Status.EXPIRED.value


def test_feedback_note_no_false_expire(file_cfg):
    """A non-expiry note (false-positive guard) leaves the vacancy LABELED, not EXPIRED."""
    con = db.connect(file_cfg["paths"]["db"])
    vid = _seed_scored(con, "u/keepexp", score=2)
    con.close()
    r = _client(file_cfg).post("/feedback", json={
        "vacancy_id": vid, "action": "bad", "note": "salt water, an alternative role"})
    assert r.status_code == 200
    con = db.connect(file_cfg["paths"]["db"])
    st = con.execute("SELECT status FROM vacancy WHERE id=?", (vid,)).fetchone()[0]
    con.close()
    assert st == Status.LABELED.value


def test_role_feedback_dry_does_not_persist(file_cfg):
    """--dry: role feedback is acked but NOT written (same firewall as /feedback)."""
    from fastapi.testclient import TestClient
    from schabasch import feedback_app, role_feedback
    from tests.conftest import seed_scored
    con = db.connect(file_cfg["paths"]["db"])
    vid = seed_scored(con, "u/roledry", score=4, company="Co", title="Backend Engineer")
    con.close()
    r = TestClient(feedback_app.create_app(file_cfg, dry=True)).post(
        "/role-feedback", json={"vacancy_id": vid, "fits": False})
    assert r.status_code == 200 and r.json().get("dry") is True
    con = db.connect(file_cfg["paths"]["db"])
    assert role_feedback.veto_map(con) == {}              # nothing persisted
    con.close()


def test_role_feedback_unknown_vacancy_404(file_cfg):
    assert _client(file_cfg).post("/role-feedback", json={"vacancy_id": 999999, "fits": True}).status_code == 404


def test_startup_pipeline_retrains_fetches_then_progresses(file_cfg, monkeypatch):
    """serve-start orchestration (heavy stages mocked): retrain → fetch core (run_investigate=False)
    → flip slate_ready after the seed → investigate the rest top→down → release the lock."""
    import schabasch.feedback_app as fa
    import schabasch.investigate as inv
    import schabasch.pipeline as pipe
    import schabasch.slate as sl
    import schabasch.triage as tri
    calls = {"retrain": 0, "tick_kwargs": None, "investigated": []}
    monkeypatch.setattr(tri, "retrain_checkpointed",
                        lambda cfg, con: calls.update(retrain=calls["retrain"] + 1) or {"skipped": True})
    monkeypatch.setattr(pipe, "nightly_tick",
                        lambda cfg, con, **kw: calls.update(tick_kwargs=kw) or {"date": "d", "slate": {"size": 4}})
    monkeypatch.setattr(sl, "build_slate", lambda cfg, con, d: [{"vacancy_id": v} for v in (10, 11, 12, 13)])
    monkeypatch.setattr(inv, "investigate_one", lambda cfg, con, vid: calls["investigated"].append(vid) or "ok")
    monkeypatch.setattr(fa.memory_guard, "require_headroom", lambda *a, **k: None)
    monkeypatch.setattr(fa.memory_guard, "start_watchdog", lambda: None)
    monkeypatch.setattr(fa.memory_guard, "configure_from_cfg", lambda cfg: None)
    fa._fetch_state().update(slate_ready=False, running=False)

    fa._run_startup_pipeline(file_cfg, seed=2)

    assert calls["retrain"] == 1                                # retrained (checkpointed) on start
    assert calls["tick_kwargs"]["run_investigate"] is False     # fetch core only; agent batch deferred
    assert calls["investigated"] == [10, 11, 12, 13]            # seed-2 + the rest, top→down
    assert fa._fetch_state()["slate_ready"] is True               # greet trigger flipped
    assert fa._fetch_state()["running"] is False                  # finished
    assert fa._FETCH_LOCK.acquire(blocking=False)               # lock released (not wedged)
    fa._FETCH_LOCK.release()


def test_startup_pipeline_marks_existing_slate_ready_before_fetch_finishes(file_cfg, monkeypatch):
    """A slow startup fetch must not block the bot/UI when an already-scored slate exists."""
    import threading

    import schabasch.feedback_app as fa
    import schabasch.investigate as inv
    import schabasch.pipeline as pipe
    import schabasch.slate as sl
    import schabasch.triage as tri

    entered_tick = threading.Event()
    release_tick = threading.Event()
    con = db.connect(file_cfg["paths"]["db"])
    _seed_scored(con, "u/ready", score=5)
    con.close()

    monkeypatch.setattr(tri, "retrain_checkpointed", lambda cfg, con: {"skipped": True})
    monkeypatch.setattr(sl, "build_slate", lambda cfg, con, d: [{"vacancy_id": 10}])
    monkeypatch.setattr(inv, "investigate_one", lambda cfg, con, vid: "ok")
    monkeypatch.setattr(fa.memory_guard, "require_headroom", lambda *a, **k: None)
    monkeypatch.setattr(fa.memory_guard, "start_watchdog", lambda: None)
    monkeypatch.setattr(fa.memory_guard, "configure_from_cfg", lambda cfg: None)

    def _slow_tick(cfg, con, **kw):
        entered_tick.set()
        assert release_tick.wait(timeout=2)
        return {"date": "d", "slate": {"size": 1}}

    monkeypatch.setattr(pipe, "nightly_tick", _slow_tick)
    fa._fetch_state().update(slate_ready=False, running=False)

    t = threading.Thread(target=fa._run_startup_pipeline, args=(file_cfg,),
                         kwargs={"seed": 0, "quiet": True})
    t.start()
    try:
        assert entered_tick.wait(timeout=2)
        assert fa._fetch_state()["running"] is True
        assert fa._fetch_state()["slate_ready"] is True
    finally:
        release_tick.set()
        t.join(timeout=2)
    assert fa._fetch_state()["running"] is False


def test_funnel_surfaces_dedup(file_cfg):
    """/funnel parses the latest dedup_fuzzy funnel detail into a visible candidate list."""
    con = db.connect(file_cfg["paths"]["db"])
    db.log_funnel(con, "dedup_fuzzy", 2,
                  detail='[{"a":1,"sa":"indeed","b":2,"sb":"linkedin","sim":100.0,"co":"X","ta":"T","tb":"T"}]')
    con.close()
    body = _client(file_cfg).get("/funnel").json()
    assert body["dedup_count"] == 2 and len(body["dedup_candidates"]) == 1
    assert body["dedup_candidates"][0]["sim"] == 100.0


def test_slate_json_returns_serializable_cards(file_cfg):
    """/slate.json returns today's slate as JSON (the Telegram bot's only read) — same cards as the
    HTML index, must JSON-serialize (the default=float shim handles any numpy fit_score).
    Uses conftest.seed_scored (rubric matches the cfg, so cards actually land in build_slate)."""
    from tests.conftest import seed_scored
    con = db.connect(file_cfg["paths"]["db"])
    vids = [seed_scored(con, f"u/json{i}", score=5, company=f"Co{i}") for i in range(3)]
    con.close()
    r = _client(file_cfg).get("/slate.json")
    assert r.status_code == 200
    body = r.json()                       # would raise if the response weren't valid JSON
    assert isinstance(body, list) and body
    assert set(c["vacancy_id"] for c in body) & set(vids)


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
