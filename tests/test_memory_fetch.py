"""W2: transparent fetch progress (human stage map) + memory-safe worker/pipeline gating."""
from __future__ import annotations

import time

import pytest

from schabasch import db, memory_guard


@pytest.fixture
def file_cfg(cfg, tmp_path):
    cfg = dict(cfg)
    cfg["paths"] = dict(cfg["paths"])
    cfg["paths"]["db"] = str(tmp_path / "t.sqlite3")
    return cfg


def _client(file_cfg):
    from fastapi.testclient import TestClient
    from schabasch import feedback_app
    return TestClient(feedback_app.create_app(file_cfg))


def test_fetch_status_maps_stage_to_human(file_cfg):
    """A bare 'normalize' funnel stage surfaces a plain-RU description + heavy flag + checklist."""
    con = db.connect(file_cfg["paths"]["db"])
    db.log_funnel(con, "normalize", 12, detail="x")
    con.close()
    s = _client(file_cfg).get("/fetch-status").json()
    assert s["stage"] == "normalize"
    assert "qwen3:8b" in s["stage_human"] and s["heavy"] is True
    assert s["stage_index"] >= 0 and s["n_stages"] == len(s["stages"]) >= 10
    assert any(st["code"] == "scrape" for st in s["stages"])


def test_fetch_status_memory_skip_stage(file_cfg):
    con = db.connect(file_cfg["paths"]["db"])
    db.log_funnel(con, "memory_skip", 0, source="judge", detail="low")
    con.close()
    s = _client(file_cfg).get("/fetch-status").json()   # default render = English
    assert "memory" in s["stage_human"] and s["heavy"] is False


def test_fetch_blocked_when_low_memory_releases_lock(file_cfg, monkeypatch):
    """require_headroom raising at worker entry → friendly error, lock released, no permanent 409."""
    import schabasch.feedback_app as fa

    def _no_headroom(ctx):
        raise memory_guard.MemoryHeadroomError("only 5% RAM free")
    monkeypatch.setattr(memory_guard, "require_headroom", _no_headroom)
    monkeypatch.setattr(memory_guard, "start_watchdog", lambda *a, **k: False)
    client = _client(file_cfg)
    assert client.post("/fetch").status_code == 200
    s = {"running": True}
    for _ in range(80):
        s = client.get("/fetch-status").json()
        if not s["running"]:
            break
        time.sleep(0.05)
    assert s["running"] is False and s["error"] and "memory" in s["error"]
    assert fa._FETCH_LOCK.acquire(blocking=False)   # lock released → not wedged
    fa._FETCH_LOCK.release()


def test_heavy_step_skips_on_low_memory(file_cfg, monkeypatch):
    """pipeline._heavy_step skips its fn + logs a memory_skip funnel row when headroom is refused."""
    from schabasch import pipeline
    con = db.connect(file_cfg["paths"]["db"])
    called = {"n": 0}

    def _no_headroom(ctx):
        raise memory_guard.MemoryHeadroomError("low")
    monkeypatch.setattr(memory_guard, "require_headroom", _no_headroom)
    summary: dict = {}
    pipeline._heavy_step(con, "normalize", lambda: called.__setitem__("n", 1), summary, "qwen")
    assert called["n"] == 0   # fn NOT called
    assert "skipped_low_memory" in summary["normalize"]
    row = con.execute("SELECT stage, source FROM funnel_log ORDER BY id DESC LIMIT 1").fetchone()
    assert row["stage"] == "memory_skip" and row["source"] == "normalize"
    con.close()


def test_heavy_step_runs_fn_with_headroom(file_cfg, monkeypatch):
    from schabasch import pipeline
    con = db.connect(file_cfg["paths"]["db"])
    monkeypatch.setattr(memory_guard, "require_headroom", lambda ctx: None)   # plenty of RAM
    summary: dict = {}
    pipeline._heavy_step(con, "judge", lambda: {"scored": 7}, summary, "qwen")
    assert summary["judge"] == {"scored": 7}
    con.close()
