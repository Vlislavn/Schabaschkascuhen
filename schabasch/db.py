"""SQLite-хранилище: схема, FSM-переходы, error envelope, лог воронки.

stdlib sqlite3, без ORM. Все записи идемпотентны (INSERT OR IGNORE / UPSERT),
перезапуск pipeline безопасен. Файл БД: data/schabasch.sqlite3.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .models import ErrorClass, FilterReason, Status

DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "schabasch.sqlite3"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS vacancy (
    id            INTEGER PRIMARY KEY,
    source        TEXT NOT NULL,             -- indeed | linkedin | arbeitsagentur | arbeitnow | gtj
    url           TEXT NOT NULL UNIQUE,
    refnr         TEXT,                      -- стабильный ключ Arbeitsagentur
    title         TEXT NOT NULL,
    company       TEXT,
    city          TEXT,
    is_remote_hint INTEGER,                  -- подсказка борда, НЕ истина
    is_temp_agency INTEGER,                  -- AA istArbeitnehmerUeberlassung; NULL = unknown
    description   TEXT,
    desc_hash     TEXT,                      -- content-hash: short-circuit повторной нормализации
    card_json     TEXT,
    status        TEXT NOT NULL DEFAULT 'new',
    filter_reason TEXT,                      -- закрытый enum FilterReason
    last_error_class   TEXT,                 -- закрытый enum ErrorClass
    last_error_details TEXT,                 -- lossless подробности
    dedup_key     TEXT,
    query_term    TEXT,
    query_city    TEXT,
    first_seen    TEXT NOT NULL,
    last_seen     TEXT NOT NULL,
    date_posted   TEXT                       -- дата ПУБЛИКАЦИИ вакансии в источнике (не скрейпа)
);
CREATE INDEX IF NOT EXISTS idx_vacancy_status ON vacancy(status);
CREATE INDEX IF NOT EXISTS idx_vacancy_dedup  ON vacancy(dedup_key);
CREATE INDEX IF NOT EXISTS idx_vacancy_hash   ON vacancy(desc_hash);

CREATE TABLE IF NOT EXISTS judge_score (
    id             INTEGER PRIMARY KEY,
    vacancy_id     INTEGER NOT NULL REFERENCES vacancy(id),
    score          INTEGER NOT NULL,         -- 1..5
    why_tag        TEXT,
    why_freetext   TEXT,
    explanation    TEXT,
    -- полный grader-tuple (паттерн are/pinned-judge-model):
    model          TEXT NOT NULL,
    model_digest   TEXT,
    rubric_version TEXT NOT NULL,
    fewshot_hash   TEXT,
    created_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_judge_vacancy ON judge_score(vacancy_id);

CREATE TABLE IF NOT EXISTS label (
    id           INTEGER PRIMARY KEY,
    vacancy_id   INTEGER NOT NULL REFERENCES vacancy(id),
    score_1_5    INTEGER,
    why_tag      TEXT,
    why_freetext TEXT,
    interview    INTEGER,                    -- признак 3 бутстрапа
    applied      INTEGER NOT NULL DEFAULT 0,
    source       TEXT NOT NULL,              -- bootstrap | slate
    created_at   TEXT NOT NULL,
    UNIQUE (vacancy_id, source)
);

CREATE TABLE IF NOT EXISTS slate_entry (
    id          INTEGER PRIMARY KEY,
    slate_date  TEXT NOT NULL,
    vacancy_id  INTEGER NOT NULL REFERENCES vacancy(id),
    rank        INTEGER NOT NULL,
    slot_type   TEXT NOT NULL,               -- exploit | explore
    feedback    TEXT,                        -- bad | good | star | applied (applied поверх good/star)
    UNIQUE (slate_date, vacancy_id)
);

CREATE TABLE IF NOT EXISTS funnel_log (
    id        INTEGER PRIMARY KEY,
    run_at    TEXT NOT NULL,
    stage     TEXT NOT NULL,
    source    TEXT,
    count     INTEGER NOT NULL,
    detail    TEXT
);

CREATE TABLE IF NOT EXISTS canary_log (
    id        INTEGER PRIMARY KEY,
    run_at    TEXT NOT NULL,
    source    TEXT NOT NULL,
    verdict   TEXT NOT NULL,                 -- enum CanaryVerdict
    rows      INTEGER NOT NULL,
    detail    TEXT
);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(db_path: Path | str = DEFAULT_DB) -> sqlite3.Connection:
    p = Path(db_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(p)
    con.row_factory = sqlite3.Row
    # busy_timeout: the always-on feedback server shares the DB with the nightly tick. Without
    # this, a click during a tick write raised HTTP 500 and was silently lost; now writers wait.
    con.execute("PRAGMA busy_timeout = 5000")
    con.executescript(_SCHEMA)
    _migrate(con)
    return con


def _migrate(con: sqlite3.Connection) -> None:
    """Idempotent additive migrations for DBs created before a column existed
    (CREATE TABLE IF NOT EXISTS never alters an existing table)."""
    cols = {r[1] for r in con.execute("PRAGMA table_info(vacancy)")}
    if "date_posted" not in cols:
        con.execute("ALTER TABLE vacancy ADD COLUMN date_posted TEXT")
    con.commit()


def upsert_vacancy(con: sqlite3.Connection, v: dict[str, Any]) -> int:
    """Идемпотентная запись вакансии. Возвращает vacancy.id.

    Обязательные ключи v: source, url, title. Опциональные: company, city, refnr,
    is_remote_hint, is_temp_agency, description, dedup_key, query_term, query_city, date_posted.
    Повторная встреча того же url обновляет last_seen (и description, если пришёл новый).
    """
    ts = now_iso()
    cur = con.execute("SELECT id, description FROM vacancy WHERE url = ?", (v["url"],))
    row = cur.fetchone()
    if row:
        sets, args = ["last_seen = ?"], [ts]
        if v.get("description") and not row["description"]:
            sets += ["description = ?", "desc_hash = ?", "status = ?"]
            from .models import content_hash
            args += [v["description"], content_hash(v["description"]), Status.DESCRIBED.value]
        con.execute(f"UPDATE vacancy SET {', '.join(sets)} WHERE id = ?", (*args, row["id"]))
        con.commit()
        return int(row["id"])
    from .models import content_hash, dedup_key as mk_key
    desc = v.get("description")
    cur = con.execute(
        """INSERT INTO vacancy (source, url, refnr, title, company, city, is_remote_hint,
               is_temp_agency, description, desc_hash, status, dedup_key, query_term,
               query_city, first_seen, last_seen, date_posted)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            v["source"], v["url"], v.get("refnr"), v["title"], v.get("company"),
            v.get("city"), v.get("is_remote_hint"), v.get("is_temp_agency"),
            desc, content_hash(desc) if desc else None,
            (Status.DESCRIBED if desc else Status.NEW).value,
            v.get("dedup_key") or mk_key(v.get("company") or "", v["title"]),
            v.get("query_term"), v.get("query_city"), ts, ts, v.get("date_posted"),
        ),
    )
    con.commit()
    return int(cur.lastrowid)


def set_status(
    con: sqlite3.Connection,
    vacancy_id: int,
    status: Status,
    *,
    filter_reason: FilterReason | None = None,
    card_json: str | None = None,
) -> None:
    sets, args = ["status = ?"], [status.value]
    if filter_reason is not None:
        sets.append("filter_reason = ?"); args.append(filter_reason.value)
    if card_json is not None:
        sets.append("card_json = ?"); args.append(card_json)
    con.execute(f"UPDATE vacancy SET {', '.join(sets)} WHERE id = ?", (*args, vacancy_id))
    con.commit()


def set_error(con: sqlite3.Connection, vacancy_id: int, err: ErrorClass, details: str) -> None:
    con.execute(
        "UPDATE vacancy SET last_error_class = ?, last_error_details = ? WHERE id = ?",
        (err.value, details[:2000], vacancy_id),
    )
    con.commit()


def by_status(con: sqlite3.Connection, status: Status, limit: int = 10_000) -> list[sqlite3.Row]:
    return con.execute(
        "SELECT * FROM vacancy WHERE status = ? ORDER BY id LIMIT ?", (status.value, limit)
    ).fetchall()


def card_by_hash(con: sqlite3.Connection, desc_hash: str) -> str | None:
    """Short-circuit: готовая карточка для идентичного описания (репост)."""
    row = con.execute(
        "SELECT card_json FROM vacancy WHERE desc_hash = ? AND card_json IS NOT NULL LIMIT 1",
        (desc_hash,),
    ).fetchone()
    return row["card_json"] if row else None


def insert_judge_score(con: sqlite3.Connection, vacancy_id: int, score: dict[str, Any]) -> None:
    con.execute(
        """INSERT INTO judge_score (vacancy_id, score, why_tag, why_freetext, explanation,
               model, model_digest, rubric_version, fewshot_hash, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (
            vacancy_id, int(score["score"]), score.get("why_tag"), score.get("why_freetext"),
            score.get("explanation"), score["model"], score.get("model_digest"),
            score["rubric_version"], score.get("fewshot_hash"), now_iso(),
        ),
    )
    con.commit()


def latest_scores(con: sqlite3.Connection) -> dict[int, sqlite3.Row]:
    """Последняя оценка судьи на вакансию."""
    rows = con.execute(
        """SELECT js.* FROM judge_score js
           JOIN (SELECT vacancy_id, MAX(id) mid FROM judge_score GROUP BY vacancy_id) m
             ON js.id = m.mid"""
    ).fetchall()
    return {int(r["vacancy_id"]): r for r in rows}


def insert_label(con: sqlite3.Connection, vacancy_id: int, lab: dict[str, Any]) -> None:
    con.execute(
        """INSERT INTO label (vacancy_id, score_1_5, why_tag, why_freetext, interview,
               applied, source, created_at)
           VALUES (?,?,?,?,?,?,?,?)
           ON CONFLICT (vacancy_id, source) DO UPDATE SET
               -- COALESCE: a flag-only update (e.g. 'applied' with score_1_5=NULL) must NOT
               -- wipe an existing rating/tag — keep the prior value when the new one is NULL.
               score_1_5 = COALESCE(excluded.score_1_5, label.score_1_5),
               why_tag = COALESCE(excluded.why_tag, label.why_tag),
               why_freetext = COALESCE(excluded.why_freetext, label.why_freetext),
               interview = COALESCE(excluded.interview, label.interview),
               applied = MAX(label.applied, excluded.applied),
               created_at = excluded.created_at""",
        (
            vacancy_id, lab.get("score_1_5"), lab.get("why_tag"), lab.get("why_freetext"),
            lab.get("interview"), int(lab.get("applied", 0)), lab["source"], now_iso(),
        ),
    )
    con.execute("UPDATE vacancy SET status = ? WHERE id = ?", (Status.LABELED.value, vacancy_id))
    con.commit()


def log_funnel(con: sqlite3.Connection, stage: str, count: int, source: str | None = None,
               detail: str | None = None) -> None:
    con.execute(
        "INSERT INTO funnel_log (run_at, stage, source, count, detail) VALUES (?,?,?,?,?)",
        (now_iso(), stage, source, count, detail),
    )
    con.commit()


def log_canary(con: sqlite3.Connection, source: str, verdict: str, rows: int,
               detail: str = "") -> None:
    con.execute(
        "INSERT INTO canary_log (run_at, source, verdict, rows, detail) VALUES (?,?,?,?,?)",
        (now_iso(), source, verdict, rows, detail),
    )
    con.commit()


def export_golden_csv(con: sqlite3.Connection, out_path: Path) -> int:
    """Экспорт golden dataset: метки + карточки + метаданные."""
    import csv
    rows = con.execute(
        """SELECT l.vacancy_id, v.source, v.title, v.company, v.city, v.url,
                  l.score_1_5, l.why_tag, l.why_freetext, l.interview, l.applied,
                  l.source AS label_source, l.created_at, v.card_json
           FROM label l JOIN vacancy v ON v.id = l.vacancy_id ORDER BY l.id"""
    ).fetchall()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([d[0] for d in con.execute("SELECT 1").description] if False else
                   ["vacancy_id", "source", "title", "company", "city", "url", "score_1_5",
                    "why_tag", "why_freetext", "interview", "applied", "label_source",
                    "created_at", "card_json"])
        for r in rows:
            w.writerow(list(r))
    return len(rows)
