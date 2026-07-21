"""Comment→task tracker (schabasch/tasks.py) + /tasks, /task-status endpoints."""
from __future__ import annotations

import json

import pytest

from schabasch import db, tasks
from schabasch.models import Status


@pytest.fixture
def file_cfg(cfg, tmp_path):
    cfg = dict(cfg)
    cfg["paths"] = dict(cfg["paths"])
    cfg["paths"]["db"] = str(tmp_path / "t.sqlite3")
    return cfg


def _seed_labeled(con, url, *, score, note, company="ACME", title="T"):
    vid = db.upsert_vacancy(con, {"source": "indeed", "url": url, "title": title,
                                  "company": company, "city": "Frankfurt", "description": "x" * 300})
    db.insert_label(con, vid, {"score_1_5": score, "applied": 0, "source": "slate",
                               "why_freetext": note})
    return vid


def test_theme_for_disambiguates_master_data_and_ni_slova():
    assert tasks.theme_for("Ни слова про master как degree — только Master Data") == "degree-misread"
    assert tasks.theme_for("Ничего не понятно, какой-то AI-слоп текст") == "jd-slop"
    assert tasks.theme_for("никаких инженеров") == "engineer-repellent"
    assert tasks.theme_for("требуется стажёр, не подходит") == "junior-floor"
    assert tasks.theme_for("не подходит из-за PhD") == "degree-gap"
    assert tasks.theme_for("какой-то нейтральный текст") == "other"


def test_ingest_is_idempotent_and_null_safe(file_cfg):
    con = db.connect(file_cfg["paths"]["db"])
    _seed_labeled(con, "u/1", score=2, note="никаких инженеров")
    _seed_labeled(con, "u/2", score=5, note="Master Data ≠ degree")
    extra = [{"comment_text": "Люблю lead", "theme": "pref"},
             {"comment_text": "Peraton: плохой мэтч", "company": "Peraton", "theme": "gap-too-big"}]
    r1 = tasks.ingest_from_db(con, extra=extra)
    assert r1["ingested"] == 4
    n1 = len(tasks.all_tasks(con))
    # re-run: the vacancy_id=NULL extras must NOT duplicate (SQLite NULL-in-UNIQUE trap)
    tasks.ingest_from_db(con, extra=extra)
    assert len(tasks.all_tasks(con)) == n1 == 4
    con.close()


def test_ingest_preserves_manual_status(file_cfg):
    con = db.connect(file_cfg["paths"]["db"])
    _seed_labeled(con, "u/1", score=2, note="инженеры отстой")
    tasks.ingest_from_db(con)
    tid = tasks.all_tasks(con)[0]["id"]
    assert tasks.set_status(con, tid, "wontfix")
    tasks.ingest_from_db(con)   # re-ingest must NOT reset the user's verdict
    assert con.execute("SELECT task_status FROM session_comment_task WHERE id=?",
                       (tid,)).fetchone()[0] == "wontfix"
    con.close()


def test_set_status_rejects_bad_status(file_cfg):
    con = db.connect(file_cfg["paths"]["db"])
    _seed_labeled(con, "u/1", score=2, note="инженер")
    tasks.ingest_from_db(con)
    tid = tasks.all_tasks(con)[0]["id"]
    assert tasks.set_status(con, tid, "nonsense") is False
    assert tasks.set_status(con, 999999, "accounted") is False   # missing id
    con.close()


def _client(file_cfg):
    from fastapi.testclient import TestClient
    from schabasch import feedback_app
    return TestClient(feedback_app.create_app(file_cfg))


def test_tasks_page_renders(file_cfg):
    con = db.connect(file_cfg["paths"]["db"])
    _seed_labeled(con, "u/1", score=2, note="никаких инженеров", title="ML Engineer")
    tasks.ingest_from_db(con)
    con.close()
    r = _client(file_cfg).get("/tasks")
    assert r.status_code == 200
    assert "Engineering" in r.text and "никаких инженеров" in r.text and "taskStatus" in r.text


def test_task_status_endpoint_flips(file_cfg):
    con = db.connect(file_cfg["paths"]["db"])
    _seed_labeled(con, "u/1", score=2, note="инженер")
    tasks.ingest_from_db(con)
    tid = tasks.all_tasks(con)[0]["id"]
    con.close()
    client = _client(file_cfg)
    assert client.post("/task-status", json={"task_id": tid, "status": "wontfix"}).json()["ok"]
    assert client.post("/task-status", json={"task_id": tid, "status": "bad"}).status_code == 400
    assert client.post("/task-status", json={"task_id": 999, "status": "open"}).status_code == 404
    con = db.connect(file_cfg["paths"]["db"])
    assert con.execute("SELECT task_status FROM session_comment_task WHERE id=?",
                       (tid,)).fetchone()[0] == "wontfix"
    con.close()


def test_engineer_role_downranks_in_slate(file_cfg):
    """W1 acted-gap: a hands-on-engineer card sinks below an equal-fit analyst card via role_kind."""
    from schabasch import role_kind as rk
    # equal everything except the role title → only role_kind differs
    eng = {"title": "Data Processing System Engineer", "summary": "", "fit_score": 0.6,
           "score": 4, "elig_score": 1.0}
    ana = {"title": "Senior Business Analyst", "summary": "", "fit_score": 0.6,
           "score": 4, "elig_score": 1.0}
    cfg = {"slate": {"role_kind_mult": {"hands_on_engineer": 0.45}}}  # her taste, explicit
    eff = lambda x: x["fit_score"] * x["elig_score"] * rk.multiplier(rk.classify(x["title"]), cfg)
    assert eff(eng) < eff(ana)   # engineer down-ranked, analyst untouched
