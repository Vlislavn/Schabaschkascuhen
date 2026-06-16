"""Live match-quality validation against REAL labels (schabasch.validation → /eval).

Gold = the user's actual score_1_5 (applied→5); metrics update as the user rates in /annotate. Fit signals
(fit_score/xenc/llm_cov/elig) are leak-free (clean=True); judge/effective/triage train on labels
(clean=False). Reuses schabasch.metrics (shared with the CLI eval/match_eval.py).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from schabasch import db, features as _features, metrics, validation
from schabasch.models import Status
from tests.conftest import make_card


def _seed(con, url, *, label, fit, judge=None):
    """Seed a SCORED vacancy with a stored fit signal, a judge score, and the user's label."""
    vid = db.upsert_vacancy(con, {"source": "indeed", "url": url, "title": f"T{url}",
                                  "company": "ACME", "city": "Frankfurt", "description": "x" * 500})
    db.set_status(con, vid, Status.SCORED, card_json=json.dumps(make_card()))
    db.insert_judge_score(con, vid, {"score": judge if judge is not None else label,
                                     "why_tag": None, "why_freetext": None, "explanation": "e",
                                     "model": "qwen3:8b", "model_digest": "d",
                                     "rubric_version": "test-v1", "fewshot_hash": "h"})
    _features._ensure_schema(con)
    feat = {"fit_score": fit, "xenc_full": fit, "llm_cov": fit, "elig_score": 1.0}
    con.execute("INSERT OR REPLACE INTO vacancy_feature (vacancy_id, match_score, feature_json,"
                " computed_at) VALUES (?,?,?,?)",
                (vid, fit, json.dumps(feat), datetime.now(timezone.utc).isoformat()))
    db.insert_label(con, vid, {"score_1_5": label, "source": "slate"})
    con.commit()
    return vid


# --------------------------------------------------------------------------- gold from real labels

def test_label_gold_uses_score_and_applied(con):
    a = db.upsert_vacancy(con, {"source": "indeed", "url": "u/a", "title": "A", "company": "C",
                                "city": "F", "description": "x" * 200})
    b = db.upsert_vacancy(con, {"source": "indeed", "url": "u/b", "title": "B", "company": "C",
                                "city": "F", "description": "x" * 200})
    db.insert_label(con, a, {"score_1_5": 5, "source": "slate"})
    db.insert_label(con, b, {"score_1_5": None, "applied": 1, "source": "slate"})  # applied → 5
    assert validation.label_gold(con) == {a: 5, b: 5}


# --------------------------------------------------------------------------- report

def test_eval_report_fit_tracks_labels(con, cfg):
    # higher label → higher fit → fit_score must order every pair correctly (pairwise = 1.0)
    _seed(con, "u/1", label=1, fit=0.1)
    _seed(con, "u/2", label=2, fit=0.4)
    _seed(con, "u/4", label=4, fit=0.7)
    _seed(con, "u/5", label=5, fit=0.95)
    rep = validation.eval_report({**cfg, "slate": {**cfg["slate"], "eval_min_pairs": 1}}, con)
    assert rep["n_labels"] == 4
    assert rep["headline"]["name"] == "fit_score" and rep["headline"]["pairwise_acc"] == 1.0
    assert rep["n_comparable_pairs"] == 6 and rep["reliable"] is True
    clean = {r["name"]: r["clean"] for r in rep["rows"]}
    assert clean["fit_score"] is True and clean["xenc_full"] is True and clean["llm_cov"] is True
    assert clean["judge_only"] is False and clean["effective"] is False  # train on labels → flagged


def test_eval_report_reliable_gate_below_threshold(con, cfg):
    _seed(con, "u/1", label=1, fit=0.1)
    _seed(con, "u/5", label=5, fit=0.9)   # exactly 1 comparable pair
    rep = validation.eval_report(cfg, con)  # default eval_min_pairs = 15
    assert rep["n_comparable_pairs"] == 1 and rep["reliable"] is False


def test_eval_report_empty(con, cfg):
    rep = validation.eval_report(cfg, con)
    assert rep["n_labels"] == 0 and rep["reliable"] is False


# --------------------------------------------------------------------------- HTTP route

def test_eval_route_renders_with_labels(cfg, tmp_path):
    from fastapi.testclient import TestClient

    from schabasch import feedback_app
    fcfg = {**cfg, "paths": {**cfg["paths"], "db": str(tmp_path / "t.sqlite3")}}
    con = db.connect(fcfg["paths"]["db"])
    _seed(con, "u/1", label=1, fit=0.1)
    _seed(con, "u/5", label=5, fit=0.9)
    con.close()
    r = TestClient(feedback_app.create_app(fcfg)).get("/eval")
    assert r.status_code == 200
    assert "Match validation" in r.text and "/annotate" in r.text
    assert "trains on labels" in r.text   # leaky signals flagged


def test_eval_route_zero_labels(cfg, tmp_path):
    from fastapi.testclient import TestClient

    from schabasch import feedback_app
    fcfg = {**cfg, "paths": {**cfg["paths"], "db": str(tmp_path / "t.sqlite3")}}
    db.connect(fcfg["paths"]["db"]).close()  # schema only, no labels
    r = TestClient(feedback_app.create_app(fcfg)).get("/eval")
    assert r.status_code == 200 and "No ratings yet" in r.text


# --------------------------------------------------------------------------- metrics parity

def test_metrics_evaluate_parity():
    scores = {1: 0.9, 2: 0.1, 3: 0.5}
    gold = {1: 5, 2: 1, 3: 3}
    m = metrics.evaluate(scores, gold, name="x")
    # comparable ordered pairs (1>2),(1>3),(3>2) — all ranked correctly
    assert m["pairwise_acc"] == 1.0 and m["n"] == 3 and m["n_pairs"] == 3
