"""Единая веб-поверхность разметки (xlsx-пакет ретайрнут).

Очередь разметки = оценённые судьёй, но ещё не размеченные вакансии (тот же источник, что и
дневной slate). Та же карточка, кнопки 👎/👍/⭐ (без 'applied'), запись в ту же label-таблицу
через /feedback. Плюс регрессия: few-shot судьи должен подхватывать 👎=2 (нижний якорь),
иначе единственный веб-сигнал негатива никогда не учит судью отталкивателям.
"""
from __future__ import annotations

import json

from schabasch import db, judge, slate
from schabasch.models import Status, WHY_TAGS
from tests.conftest import make_card


def _seed_scored(con, url, *, score, company="ACME", title="Engineer", card=True):
    vid = db.upsert_vacancy(con, {"source": "indeed", "url": url, "title": title,
                                  "company": company, "city": "Frankfurt", "description": "x" * 500})
    cj = json.dumps(make_card(company=company)) if card else None
    db.set_status(con, vid, Status.SCORED, card_json=cj)
    db.insert_judge_score(con, vid, {"score": score, "why_tag": None, "why_freetext": None,
                                     "explanation": "e", "model": "qwen3:8b", "model_digest": "d",
                                     "rubric_version": "test-v1", "fewshot_hash": "h"})
    return vid


# --------------------------------------------------------------------------- очередь разметки

def test_annotation_batch_is_unlabeled_scored(con, cfg):
    a = _seed_scored(con, "u/1", score=5)
    b = _seed_scored(con, "u/2", score=2)
    items, total = slate.annotation_batch(cfg, con, "2026-06-15")
    assert total == 2 and {it["vacancy_id"] for it in items} == {a, b}
    # размеченная вакансия выпадает из очереди (status → LABELED) — Goal-Gradient к нулю
    db.insert_label(con, a, {"score_1_5": 5, "source": "slate"})
    items2, total2 = slate.annotation_batch(cfg, con, "2026-06-15")
    assert total2 == 1 and {it["vacancy_id"] for it in items2} == {b}


def test_annotation_batch_caps_at_config_n(con, cfg):
    cfg = {**cfg, "slate": {**cfg["slate"], "annotate_batch": 2}}
    for i in range(5):
        _seed_scored(con, f"u/{i}", score=3 + (i % 3))
    items, total = slate.annotation_batch(cfg, con, "2026-06-15")
    assert total == 5 and len(items) == 2


# --------------------------------------------------------------------------- рендер

def test_render_annotate_has_rating_buttons_no_applied(con, cfg):
    _seed_scored(con, "u/r", score=4, title="ML Engineer")
    items, total = slate.annotation_batch(cfg, con, "2026-06-15")
    html = slate.render_annotate_html(items, "2026-06-15", total_pending=total)
    assert "Annotate" in html and "/feedback" in html
    assert "fb(" in html and "'bad'" in html and "'good'" in html and "'star'" in html
    # 'applied' нечем кликать на случайной вакансии из очереди → кнопки нет
    assert "'applied'" not in html
    # прогресс-счётчик (Goal-Gradient)
    assert 'id="progress"' in html and "Marked 0/" in html


def test_render_annotate_empty_queue(con, cfg):
    html = slate.render_annotate_html([], "2026-06-15", total_pending=0)
    assert "annotation queue is empty" in html


# --------------------------------------------------------------------------- HTTP route

def test_annotate_route_renders(cfg, tmp_path):
    from fastapi.testclient import TestClient

    from schabasch import feedback_app
    fcfg = {**cfg, "paths": {**cfg["paths"], "db": str(tmp_path / "t.sqlite3")}}
    con = db.connect(fcfg["paths"]["db"])
    _seed_scored(con, "u/x", score=5, company="SpaceCo", title="Orbital Engineer")
    con.close()
    r = TestClient(feedback_app.create_app(fcfg)).get("/annotate")
    assert r.status_code == 200 and "Annotate" in r.text and "Orbital Engineer" in r.text


# --------------------------------------------------------------------------- few-shot нижний якорь

def test_fewshot_picks_up_downvote_low_anchor(con):
    """👎=2 (единственный веб-сигнал негатива) должен попасть в few-shot. До фикса WHERE был
    score IN (1,5) → литеральная 1 в UI отсутствует, значит 👎 никогда не учил судью."""
    pos = _seed_scored(con, "u/pos", score=5)
    neg = _seed_scored(con, "u/neg", score=1)  # judge score irrelevant; label drives few-shot
    db.insert_label(con, pos, {"score_1_5": 5, "why_tag": "space", "source": "slate"})
    db.insert_label(con, neg, {"score_1_5": 2, "why_tag": "temp-agency", "source": "slate"})
    fewshot, digest = judge.build_fewshot(con, 6)
    assert fewshot.strip() and digest
    assert "SCORE: 2" in fewshot, "👎=2 dropped from few-shot — low-anchor regression"
    assert "SCORE: 5" in fewshot
    assert "temp-agency" in fewshot  # repellent principle rendered for the score-2 anchor


def test_fewshot_validates_tag_in_vocab(con):
    """why_tag вне WHY_TAGS не попадает (наследие чужой рубрики); NULL допускается."""
    assert "temp-agency" in WHY_TAGS  # guard: the fixture tag is real
