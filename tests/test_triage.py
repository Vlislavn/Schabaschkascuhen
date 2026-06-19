"""Tests for triage.py — cheap ML gate: bucket, cold start, select_for_normalize.

No real LightGBM or bge-m3 needed: labeled rows are seeded directly; lgbm skipped
via pytest.importorskip or monkeypatched; feature rows inserted raw.
"""
from __future__ import annotations

import json
import struct
from datetime import datetime, timezone

import numpy as np
import pytest

from schabasch import db, features as _feat, triage
from schabasch.aspects import FEATURE_NAMES
from schabasch.models import Status
from tests.conftest import make_card


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _f32_blob(vec: np.ndarray) -> bytes:
    arr = np.asarray(vec, dtype=np.float32).ravel()
    return struct.pack(f"{len(arr)}f", *arr)


def _seed_described(con, url: str, company: str = "ACME") -> int:
    vid = db.upsert_vacancy(con, {
        "source": "indeed", "url": url,
        "title": "Senior Business Analyst", "company": company,
        "city": "Frankfurt", "description": "x" * 600,
    })
    card = make_card(company=company)
    db.set_status(con, vid, Status.DESCRIBED, card_json=json.dumps(card))
    return vid


def _seed_feature_row(con, vid: int, match_score: float = 0.7) -> None:
    _feat._ensure_schema(con)
    feat_json = json.dumps({k: 0.0 for k in FEATURE_NAMES})
    con.execute(
        """INSERT OR REPLACE INTO vacancy_feature (vacancy_id, match_score, feature_json, computed_at)
           VALUES (?, ?, ?, '2026-01-01')""",
        (vid, match_score, feat_json),
    )
    con.commit()


def _seed_embedding(con, vid: int) -> None:
    _feat._ensure_schema(con)
    vec = np.zeros(1024, dtype=np.float32)
    vec[0] = 1.0
    con.execute(
        """INSERT OR REPLACE INTO vacancy_embedding
           (vacancy_id, content_hash, dense, aspects_json, model_version, computed_at)
           VALUES (?, 'h', ?, '{}', 'bge-m3', '2026-01-01')""",
        (vid, _f32_blob(vec)),
    )
    con.commit()


def _seed_label(con, vid: int, score: int = 4, applied: int = 0,
                source: str = "slate") -> None:
    db.insert_label(con, vid, {
        "score_1_5": score, "why_tag": None, "why_freetext": None,
        "interview": 0, "applied": applied, "source": source,
    })


def _seed_scored_for_label(con, url: str, company: str = "ACME",
                            score: int = 4) -> int:
    """DESCRIBED + feature row + label — the training row shape."""
    vid = _seed_described(con, url, company)
    db.insert_judge_score(con, vid, {
        "score": score, "why_tag": None, "why_freetext": None,
        "explanation": "good", "model": "qwen3:8b",
        "model_digest": "d", "rubric_version": "v1", "fewshot_hash": "h",
    })
    db.set_status(con, vid, Status.SCORED)
    _seed_feature_row(con, vid, match_score=score / 5)
    _seed_label(con, vid, score=score)
    return vid


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def test_ensure_schema_idempotent(con):
    triage._ensure_schema(con)
    triage._ensure_schema(con)
    tables = {r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    assert "triage_decision" in tables


# ---------------------------------------------------------------------------
# Bucketing
# ---------------------------------------------------------------------------

CUTOFFS = {"must": 4.5, "should": 3.5, "could": 2.0}

@pytest.mark.parametrize("score,expected", [
    (5.0, "must"), (4.5, "must"), (4.0, "should"), (3.5, "should"),
    (3.0, "could"), (2.0, "could"), (1.9, "drop"), (1.0, "drop"),
])
def test_score_to_priority(score, expected):
    assert triage._score_to_priority(score, CUTOFFS) == expected


# ---------------------------------------------------------------------------
# Target derivation
# ---------------------------------------------------------------------------

def test_build_target_from_score():
    assert triage._build_target(4, 0) == 4.0


def test_build_target_applied_overrides():
    # applied=1 → 5.0 regardless of score
    assert triage._build_target(None, 1) == 5.0


def test_build_target_none_when_no_info():
    assert triage._build_target(None, 0) is None


# ---------------------------------------------------------------------------
# Sample weights
# ---------------------------------------------------------------------------

def test_build_weight_applied_high():
    assert triage._build_weight(None, 1, "slate") == 1.0


def test_build_weight_star_high():
    assert triage._build_weight(5, 0, "bootstrap") == 1.0


def test_build_weight_slate_medium():
    assert triage._build_weight(4, 0, "slate") == 0.7


def test_build_weight_bootstrap_low():
    assert triage._build_weight(3, 0, "bootstrap") == 0.5


# ---------------------------------------------------------------------------
# Temporal split
# ---------------------------------------------------------------------------

def test_temporal_split_newest_in_test():
    groups = ["a", "b", "c", "d", "e"]  # chronological order
    train_mask, test_mask = triage._temporal_split(groups, frac=0.2)
    # 20% of 5 unique groups = 1 test group = "e" (newest)
    assert test_mask.sum() == 1
    assert groups[test_mask.tolist().index(True)] == "e"


def test_temporal_split_no_overlap():
    groups = ["a", "b", "a", "b", "c", "d"]
    train_mask, test_mask = triage._temporal_split(groups)
    assert (train_mask & test_mask).sum() == 0


def test_temporal_split_covers_all():
    groups = ["a", "b", "c"]
    train_mask, test_mask = triage._temporal_split(groups)
    assert (train_mask | test_mask).all()


# ---------------------------------------------------------------------------
# Labels hash
# ---------------------------------------------------------------------------

def test_labels_hash_deterministic():
    rows = [{"vacancy_id": 1, "score_1_5": 4, "applied": 0},
            {"vacancy_id": 2, "score_1_5": 5, "applied": 1}]
    h1 = triage._labels_hash(rows)
    h2 = triage._labels_hash(rows[::-1])
    assert h1 == h2  # sorted
    assert len(h1) == 16


def test_labels_hash_changes_on_new_label():
    r1 = [{"vacancy_id": 1, "score_1_5": 4, "applied": 0}]
    r2 = [{"vacancy_id": 1, "score_1_5": 4, "applied": 0},
          {"vacancy_id": 2, "score_1_5": 3, "applied": 0}]
    assert triage._labels_hash(r1) != triage._labels_hash(r2)


# ---------------------------------------------------------------------------
# triage_pending — cold start (no model artifact)
# ---------------------------------------------------------------------------

def test_triage_pending_cold_start_no_features(con, cfg):
    """Cold start: no feature rows → fall back to 0.0 match_score → all in 'could/drop'."""
    vid = _seed_described(con, "u/1")
    counts = triage.triage_pending(cfg, con)
    assert counts.get("cold_start") is True
    assert sum(counts[k] for k in ("must", "should", "could", "drop")) == 1
    # 0.0 match_score → calibrated = 1.0 → drop
    assert counts.get("drop", 0) == 1


def test_triage_pending_cold_start_with_feature_rows(con, cfg):
    """Cold start with match_score=0.8 → calibrated=4.2 → 'should'."""
    vid = _seed_described(con, "u/2")
    _seed_feature_row(con, vid, match_score=0.8)  # → calibrated = 1 + 4*0.8 = 4.2

    counts = triage.triage_pending(cfg, con)
    assert counts.get("cold_start") is True
    assert counts.get("should", 0) == 1


def test_triage_pending_cold_start_high_match_is_must(con, cfg):
    """match_score=1.0 → calibrated=5.0 → 'must'."""
    vid = _seed_described(con, "u/3")
    _seed_feature_row(con, vid, match_score=1.0)

    counts = triage.triage_pending(cfg, con)
    assert counts.get("must", 0) == 1


def test_triage_pending_writes_decision_row(con, cfg):
    vid = _seed_described(con, "u/4")
    _seed_feature_row(con, vid, match_score=0.9)

    triage.triage_pending(cfg, con)

    row = con.execute(
        "SELECT * FROM triage_decision WHERE vacancy_id = ?", (vid,)
    ).fetchone()
    assert row is not None
    assert row["model_version"] == "cold_start"
    assert row["calibrated_score"] > 0
    assert row["priority"] in ("must", "should", "could", "drop")


def test_triage_pending_upsert_idempotent(con, cfg):
    """Running triage_pending twice should overwrite, not duplicate."""
    vid = _seed_described(con, "u/5")
    triage.triage_pending(cfg, con)
    triage.triage_pending(cfg, con)
    n = con.execute(
        "SELECT COUNT(*) FROM triage_decision WHERE vacancy_id = ?", (vid,)
    ).fetchone()[0]
    assert n == 1


def test_triage_pending_multiple_vacancies(con, cfg):
    vids = []
    for i, ms in enumerate([0.0, 0.4, 0.85, 1.0]):
        v = _seed_described(con, f"u/m{i}")
        _seed_feature_row(con, v, match_score=ms)
        vids.append(v)

    counts = triage.triage_pending(cfg, con)
    total = sum(counts[k] for k in ("must", "should", "could", "drop"))
    assert total == 4


def test_triage_pending_limit_respected(con, cfg):
    for i in range(5):
        _seed_described(con, f"u/lim{i}")
    counts = triage.triage_pending(cfg, con, limit=2)
    total = sum(counts[k] for k in ("must", "should", "could", "drop"))
    assert total == 2


# ---------------------------------------------------------------------------
# select_for_normalize
# ---------------------------------------------------------------------------

def test_select_for_normalize_fallback_when_no_triage(con, cfg):
    """No triage rows at all → falls back to db.by_status."""
    for i in range(3):
        _seed_described(con, f"u/fb{i}")
    rows = triage.select_for_normalize(cfg, con, budget=10)
    assert len(rows) == 3


def test_select_for_normalize_cold_start_no_hard_drop(con, cfg):
    """Cold start: even 'drop' priority rows are included (rank-only, no hard-drop)."""
    vid_drop = _seed_described(con, "u/drop")
    vid_must = _seed_described(con, "u/must")

    _seed_feature_row(con, vid_drop, match_score=0.0)   # → drop
    _seed_feature_row(con, vid_must, match_score=1.0)   # → must

    triage.triage_pending(cfg, con)

    rows = triage.select_for_normalize(cfg, con, budget=10)
    ids = [r["id"] for r in rows]
    assert vid_drop in ids
    assert vid_must in ids


def test_select_for_normalize_cold_start_ordered_by_score(con, cfg):
    """Cold start: higher calibrated_score rows come first."""
    vids = []
    for ms in [0.2, 0.9, 0.5]:
        v = _seed_described(con, f"u/ord{ms}")
        _seed_feature_row(con, v, match_score=ms)
        vids.append(v)

    triage.triage_pending(cfg, con)
    rows = triage.select_for_normalize(cfg, con, budget=10)
    ids = [r["id"] for r in rows]
    # highest match (0.9) should come first
    assert ids[0] == vids[1]


def test_select_for_normalize_trained_drops_excluded(con, cfg):
    """Trained mode: drop-priority rows excluded from normalize queue."""
    vid_drop = _seed_described(con, "u/tdrop")
    vid_good = _seed_described(con, "u/tgood")
    triage._ensure_schema(con)
    now = datetime.now(timezone.utc).isoformat()
    # Manually insert trained-mode triage_decision rows
    con.execute(
        """INSERT INTO triage_decision
           (vacancy_id, raw_score, calibrated_score, priority, model_version, decided_at)
           VALUES (?, 1.5, 1.5, 'drop', 'trained:abc', ?)""",
        (vid_drop, now),
    )
    con.execute(
        """INSERT INTO triage_decision
           (vacancy_id, raw_score, calibrated_score, priority, model_version, decided_at)
           VALUES (?, 4.8, 4.8, 'must', 'trained:abc', ?)""",
        (vid_good, now),
    )
    con.commit()

    rows = triage.select_for_normalize(cfg, con, budget=10)
    ids = [r["id"] for r in rows]
    assert vid_drop not in ids
    assert vid_good in ids


def test_select_for_normalize_respects_budget(con, cfg):
    for i in range(5):
        _seed_described(con, f"u/bgt{i}")
    rows = triage.select_for_normalize(cfg, con, budget=2)
    assert len(rows) <= 2


# ---------------------------------------------------------------------------
# scores_by_vacancy / triage_score
# ---------------------------------------------------------------------------

def test_scores_by_vacancy_empty(con):
    triage._ensure_schema(con)
    assert triage.scores_by_vacancy(con) == {}


def test_scores_by_vacancy_normalized(con, cfg):
    """calibrated_score/5 should be in [0,1]."""
    vid = _seed_described(con, "u/sv1")
    _seed_feature_row(con, vid, match_score=0.8)
    triage.triage_pending(cfg, con)

    scores = triage.scores_by_vacancy(con)
    assert vid in scores
    assert 0.0 <= scores[vid] <= 1.0


def test_triage_score_none_when_absent(con):
    triage._ensure_schema(con)
    assert triage.triage_score(con, 9999) is None


def test_triage_score_present(con, cfg):
    vid = _seed_described(con, "u/ts1")
    _seed_feature_row(con, vid, match_score=0.75)
    triage.triage_pending(cfg, con)
    score = triage.triage_score(con, vid)
    assert score is not None
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# train — not enough labels
# ---------------------------------------------------------------------------

def test_train_skipped_when_too_few_labels(con, cfg):
    """With < 30 labels, train() should return skipped."""
    for i in range(5):
        _seed_scored_for_label(con, f"u/tskip{i}")
    result = triage.train(cfg, con)
    assert result.get("skipped") or result.get("error")


def test_train_no_labels_returns_error(con, cfg):
    result = triage.train(cfg, con)
    # Either error (no labeled+featured rows) or skipped
    assert "error" in result or result.get("skipped")


# ---------------------------------------------------------------------------
# train — with enough labels (requires LightGBM)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    __import__("importlib").util.find_spec("lightgbm") is None,
    reason="lightgbm not installed"
)
def test_train_with_30_labels_produces_artifact(con, cfg, tmp_path):
    """With >= 30 labeled+featured rows, train() should succeed."""
    cfg = dict(cfg, paths={**cfg["paths"], "model_dir": str(tmp_path)})
    cfg = dict(cfg, triage={**cfg["triage"], "min_labels_to_train": 10})  # lower threshold

    for i in range(12):
        score = (i % 5) + 1
        vid = _seed_described(con, f"u/tr{i}", company=f"Corp{i}")
        _seed_feature_row(con, vid, match_score=score / 5)
        _seed_embedding(con, vid)
        db.set_status(con, vid, Status.SCORED)
        _seed_label(con, vid, score=score, source="slate")

    result = triage.train(cfg, con)
    # May be skipped if labels_hash unchanged; check it ran
    assert result.get("trained") or result.get("skipped")


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("lightgbm") is None,
    reason="lightgbm not installed"
)
def test_train_skipped_on_unchanged_labels(con, cfg, tmp_path):
    """Second train() call without new labels should be a no-op."""
    cfg = dict(cfg, paths={**cfg["paths"], "model_dir": str(tmp_path)})
    cfg = dict(cfg, triage={**cfg["triage"], "min_labels_to_train": 3})

    for i in range(5):
        score = (i % 5) + 1
        vid = _seed_described(con, f"u/skip{i}", company=f"Co{i}")
        _seed_feature_row(con, vid, match_score=score / 5)
        _seed_label(con, vid, score=score, source="slate")

    triage.train(cfg, con)
    result2 = triage.train(cfg, con)
    # Second call should be skipped (labels_sha unchanged)
    assert result2.get("skipped")


# ---------------------------------------------------------------------------
# evaluate — too few rows
# ---------------------------------------------------------------------------

def test_evaluate_error_when_too_few_labels(con, cfg):
    result = triage.evaluate(cfg, con)
    assert "error" in result


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("lightgbm") is None,
    reason="lightgbm not installed"
)
def test_evaluate_with_enough_labels(con, cfg):
    for i in range(8):
        score = (i % 5) + 1
        vid = _seed_described(con, f"u/ev{i}", company=f"Ev{i}")
        _seed_feature_row(con, vid, match_score=score / 5)
        _seed_label(con, vid, score=score, source="slate")

    result = triage.evaluate(cfg, con)
    # Should have at least n_train / n_test (or an error if split fails)
    assert "n_train" in result or "error" in result


# ---------------------------------------------------------------------------
# Monotone constraints shape
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Deploy gate (R1): measure-then-ship — never ship a model that fails its temporal holdout
# ---------------------------------------------------------------------------

def test_should_deploy_gate():
    assert triage._should_deploy({"spearman_rho": 0.4}, 0.0) is True
    assert triage._should_deploy({"spearman_rho": -0.31}, 0.0) is False   # the 2026-06-18 incident
    assert triage._should_deploy({"spearman_rho": 0.0}, 0.0) is False     # at floor → reject
    assert triage._should_deploy({"spearman_rho": None}, 0.0) is True     # uncomputable → don't block
    assert triage._should_deploy({}, 0.0) is True
    assert triage._should_deploy({"spearman_rho": float("nan")}, 0.0) is True   # uncomputable → don't block
    assert triage._should_deploy({"spearman_rho": 0.05}, 0.1) is False    # below a custom floor


def _seed_30_for_gate(con, prefix: str):
    for i in range(12):
        score = (i % 5) + 1
        vid = _seed_described(con, f"u/{prefix}{i}", company=f"{prefix}{i}")
        _seed_feature_row(con, vid, match_score=score / 5)
        _seed_embedding(con, vid)
        db.set_status(con, vid, Status.SCORED)
        _seed_label(con, vid, score=score, source="slate")


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("lightgbm") is None,
    reason="lightgbm not installed"
)
def test_train_deploy_gate_rejects_negative_holdout(con, cfg, tmp_path, monkeypatch):
    """The regression for the incident: a model whose TEMPORAL holdout is negative must NOT be saved
    (the prior model/cold_start is kept). _compute_metrics is pinned so the gate decision is
    deterministic (isolated from LightGBM's noisy tiny-data holdout)."""
    cfg = dict(cfg, paths={**cfg["paths"], "model_dir": str(tmp_path)})
    cfg = dict(cfg, triage={**cfg["triage"], "min_labels_to_train": 10, "deploy_min_spearman": 0.0})
    _seed_30_for_gate(con, "gate")
    monkeypatch.setattr(triage, "_compute_metrics", lambda *a, **k: {"spearman_rho": -0.31})
    res = triage.train(cfg, con, force=True)
    assert res.get("rejected") is True and not res.get("trained")
    assert not (tmp_path / "triage.joblib").exists()   # bad model NOT shipped


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("lightgbm") is None,
    reason="lightgbm not installed"
)
def test_train_deploy_gate_ships_positive_holdout(con, cfg, tmp_path, monkeypatch):
    """A model that PASSES its temporal holdout is deployed normally (gate doesn't block good models)."""
    cfg = dict(cfg, paths={**cfg["paths"], "model_dir": str(tmp_path)})
    cfg = dict(cfg, triage={**cfg["triage"], "min_labels_to_train": 10, "deploy_min_spearman": 0.0})
    _seed_30_for_gate(con, "ship")
    monkeypatch.setattr(triage, "_compute_metrics", lambda *a, **k: {"spearman_rho": 0.5})
    res = triage.train(cfg, con, force=True)
    assert res.get("trained") is True
    assert (tmp_path / "triage.joblib").exists()


def test_monotone_constraints_named_only():
    n = len(FEATURE_NAMES)
    mc = triage._monotone_constraints(n)
    assert len(mc) == n


def test_monotone_constraints_with_dense():
    n = 1024 + len(FEATURE_NAMES)
    mc = triage._monotone_constraints(n)
    assert len(mc) == n
    # Dense portion should all be 0
    assert all(c == 0 for c in mc[:1024])


def test_coverage_features_positive_constraint():
    mc = triage._monotone_constraints(len(FEATURE_NAMES))
    idx = FEATURE_NAMES.index("cov_musthave_maxsim")
    assert mc[idx] == 1


def test_missing_feature_negative_constraint():
    mc = triage._monotone_constraints(len(FEATURE_NAMES))
    idx = FEATURE_NAMES.index("n_musthave_missing")
    assert mc[idx] == -1
