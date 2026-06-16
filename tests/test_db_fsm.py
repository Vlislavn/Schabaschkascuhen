"""FSM-хранилище: идемпотентный upsert, статусы, конфликт-апдейт меток, grader-tuple."""
from __future__ import annotations

from schabasch import db
from schabasch.models import FilterReason, Status


def test_upsert_idempotent_same_url(con):
    v = {"source": "indeed", "url": "https://x/1", "title": "Eng", "company": "ACME"}
    id1 = db.upsert_vacancy(con, v)
    id2 = db.upsert_vacancy(con, v)
    assert id1 == id2
    n = con.execute("SELECT COUNT(*) FROM vacancy").fetchone()[0]
    assert n == 1


def test_upsert_adds_description_promotes_to_described(con):
    vid = db.upsert_vacancy(con, {"source": "indeed", "url": "https://x/2", "title": "Eng"})
    row = con.execute("SELECT status, description FROM vacancy WHERE id=?", (vid,)).fetchone()
    assert row["status"] == Status.NEW.value
    # повторный upsert с описанием → DESCRIBED + desc_hash
    db.upsert_vacancy(con, {"source": "indeed", "url": "https://x/2", "title": "Eng",
                            "description": "full text " * 20})
    row = con.execute("SELECT status, desc_hash FROM vacancy WHERE id=?", (vid,)).fetchone()
    assert row["status"] == Status.DESCRIBED.value
    assert row["desc_hash"]


def test_set_status_with_filter_reason(con):
    vid = db.upsert_vacancy(con, {"source": "indeed", "url": "https://x/3", "title": "Eng"})
    db.set_status(con, vid, Status.FILTERED, filter_reason=FilterReason.LANGUAGE_DE)
    row = con.execute("SELECT status, filter_reason FROM vacancy WHERE id=?", (vid,)).fetchone()
    assert row["status"] == Status.FILTERED.value
    assert row["filter_reason"] == FilterReason.LANGUAGE_DE.value


def test_insert_label_conflict_updates_and_keeps_applied_max(con):
    vid = db.upsert_vacancy(con, {"source": "indeed", "url": "https://x/4", "title": "Eng"})
    db.insert_label(con, vid, {"score_1_5": 4, "applied": 1, "source": "slate"})
    db.insert_label(con, vid, {"score_1_5": 2, "applied": 0, "source": "slate"})
    row = con.execute("SELECT score_1_5, applied FROM label WHERE vacancy_id=?", (vid,)).fetchone()
    assert row["score_1_5"] == 2          # перезаписано
    assert row["applied"] == 1            # applied не понижается (MAX)
    # одна строка на (vacancy, source)
    assert con.execute("SELECT COUNT(*) FROM label WHERE vacancy_id=?", (vid,)).fetchone()[0] == 1


def test_insert_label_sets_labeled_status(con):
    vid = db.upsert_vacancy(con, {"source": "indeed", "url": "https://x/5", "title": "Eng"})
    db.insert_label(con, vid, {"score_1_5": 5, "applied": 0, "source": "bootstrap"})
    row = con.execute("SELECT status FROM vacancy WHERE id=?", (vid,)).fetchone()
    assert row["status"] == Status.LABELED.value


def test_judge_score_persists_full_grader_tuple(con):
    vid = db.upsert_vacancy(con, {"source": "indeed", "url": "https://x/6", "title": "Eng"})
    db.insert_judge_score(con, vid, {"score": 5, "model": "qwen3:8b", "model_digest": "sha:abc",
                                     "rubric_version": "v1", "fewshot_hash": "ffff",
                                     "explanation": "ok"})
    row = con.execute("SELECT model, model_digest, rubric_version, fewshot_hash FROM judge_score "
                      "WHERE vacancy_id=?", (vid,)).fetchone()
    assert row["model"] == "qwen3:8b"
    assert row["model_digest"] == "sha:abc"
    assert row["rubric_version"] == "v1"
    assert row["fewshot_hash"] == "ffff"


def test_card_by_hash_short_circuit(con):
    vid = db.upsert_vacancy(con, {"source": "indeed", "url": "https://x/7", "title": "Eng",
                                  "description": "repost body " * 30})
    h = con.execute("SELECT desc_hash FROM vacancy WHERE id=?", (vid,)).fetchone()["desc_hash"]
    assert db.card_by_hash(con, h) is None  # ещё нет карточки
    db.set_status(con, vid, Status.NORMALIZED, card_json='{"role":"x"}')
    assert db.card_by_hash(con, h) == '{"role":"x"}'


def test_upsert_captures_date_posted(con):
    vid = db.upsert_vacancy(con, {"source": "indeed", "url": "https://x/dp", "title": "T",
                                  "date_posted": "2026-06-10"})
    dp = con.execute("SELECT date_posted FROM vacancy WHERE id=?", (vid,)).fetchone()[0]
    assert dp == "2026-06-10"


def test_migrate_adds_date_posted_to_old_db(tmp_path):
    """An older DB whose vacancy table predates date_posted gets the column added on connect()."""
    import sqlite3
    p = tmp_path / "old.sqlite3"
    # Build the full current schema, then drop date_posted to simulate a pre-migration DB.
    raw = sqlite3.connect(p)
    raw.executescript(db._SCHEMA)
    raw.execute("ALTER TABLE vacancy DROP COLUMN date_posted")
    raw.commit(); raw.close()
    assert "date_posted" not in {r[1] for r in sqlite3.connect(p).execute("PRAGMA table_info(vacancy)")}
    con = db.connect(str(p))   # runs _SCHEMA (no-op on existing table) then _migrate
    cols = {r[1] for r in con.execute("PRAGMA table_info(vacancy)")}
    assert "date_posted" in cols
    con.close()
