"""scripts.backfill_session_notes — maps the 15 June session feedback onto labelled vacancies,
filling only label.why_freetext (never the score), idempotently, and feeding judge few-shot."""
from __future__ import annotations

import json

from schabasch import db, judge
from schabasch.models import Status
from tests.conftest import make_card
from scripts import backfill_session_notes as B


def _seed_label(con, vid, *, score, company="Co", title="T"):
    db.upsert_vacancy(con, {"source": "indeed", "url": f"u/{vid}", "title": title,
                            "company": company, "city": "Frankfurt", "description": "x" * 300})
    # the script keys vacancy_id directly, so force this row's id to the mapped vid
    con.execute("UPDATE vacancy SET id = ? WHERE url = ?", (vid, f"u/{vid}"))
    # build_fewshot requires a card_json on the vacancy
    db.set_status(con, vid, Status.SCORED, card_json=json.dumps(make_card(company=company)))
    db.insert_label(con, vid, {"score_1_5": score, "applied": 0, "source": "slate",
                               "interview": None, "why_freetext": None})


def test_backfill_fills_note_keeps_score_idempotent(con):
    # seed two of the mapped vids (a positive 439 and a negative 1071) with their real scores
    _seed_label(con, 439, score=5, company="Merz", title="Business Process Owner – Master Data")
    _seed_label(con, 1071, score=2, company="EUMETSAT", title="Data Processing System Engineer")
    res = B.backfill(con)
    assert res["written"] == 2 and res["skipped_no_label"] == 5   # the other 5 vids aren't seeded here

    r439 = con.execute("SELECT score_1_5, why_freetext FROM label WHERE vacancy_id=439").fetchone()
    assert r439["score_1_5"] == 5                       # score untouched
    assert "Master Data" in r439["why_freetext"]        # the user's note landed
    r1071 = con.execute("SELECT score_1_5, why_freetext FROM label WHERE vacancy_id=1071").fetchone()
    assert r1071["score_1_5"] == 2 and "руками" in r1071["why_freetext"]

    # idempotent: a second run writes nothing new
    res2 = B.backfill(con)
    assert res2["written"] == 0 and res2["already"] == 2


def test_backfilled_notes_reach_judge_fewshot(con):
    # 439 (score 5) and 1071 (score 2) are both in build_fewshot's range (score<=2 OR =5) → NOTE: lines
    _seed_label(con, 439, score=5, company="Merz", title="BPO Master Data")
    _seed_label(con, 1071, score=2, company="EUMETSAT", title="Data Processing System Engineer")
    B.backfill(con)
    fewshot, _ = judge.build_fewshot(con, 6)
    assert "NOTE:" in fewshot
    assert "Master Data" in fewshot and "руками" in fewshot
