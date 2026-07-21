"""Tests for features.rerank_scored — cross-encoder re-rank on top-K SCORED vacancies.

FlagReranker is monkeypatched — no real model needed.
"""
from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from schabasch import db, features
from schabasch.models import Status
from tests.conftest import make_card


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _f32_blob(vec: np.ndarray) -> bytes:
    arr = np.asarray(vec, dtype=np.float32).ravel()
    return struct.pack(f"{len(arr)}f", *arr)


def _seed_scored(con, url: str, company: str = "ACME", score: int = 4) -> int:
    vid = db.upsert_vacancy(con, {
        "source": "indeed", "url": url,
        "title": "Senior Business Analyst", "company": company,
        "city": "Frankfurt",
        "description": "Requirements: Python, SQL\nResponsibilities: Lead projects",
    })
    card = make_card(company=company)
    db.set_status(con, vid, Status.DESCRIBED, card_json=json.dumps(card))
    db.insert_judge_score(con, vid, {
        "score": score, "why_tag": None, "why_freetext": None,
        "explanation": "good", "model": "qwen3:8b",
        "model_digest": "d", "rubric_version": "v1", "fewshot_hash": "h",
    })
    db.set_status(con, vid, Status.SCORED)
    return vid


def _seed_candidate(con) -> None:
    from schabasch.candidate import _ensure_schema as _cs, _build_aspect_texts
    import hashlib, json as _json
    from datetime import datetime, timezone
    features._ensure_schema(con)
    _cs(con)
    profile = {
        "skills": ["Python", "SQL", "Tableau"],
        "experience": "6 years BA at BostonGene and Nielsen.",
        "domains": ["digital transformation"],
        "seniority": "senior", "years_experience": 6, "education": "BSc",
        "languages": {"en": "C1", "de": "A2"},
        "target_roles": ["Senior Business Analyst"],
        "locations": ["Heidelberg"], "magnets": ["data-analytics"],
        "repellents": ["lab-science"],
        "summary": "Senior BA seeking digital transformation roles.",
    }
    at = _build_aspect_texts(profile)
    doc_hash = hashlib.sha256(_json.dumps(at, sort_keys=True).encode()).hexdigest()[:16]
    con.execute(
        """INSERT INTO candidate_profile
           (created_at, raw_input, profile_json, aspect_texts, doc_hash)
           VALUES (?, ?, ?, ?, ?)""",
        (datetime.now(timezone.utc).isoformat(), "test",
         _json.dumps(profile), _json.dumps(at), doc_hash),
    )
    con.commit()


class FakeReranker:
    """Returns deterministic scores based on text lengths (no real model)."""
    def compute_score(self, pairs, normalize=True):
        scores = []
        for pair in pairs:
            # Score = fraction of CV tokens appearing in JD (simple proxy)
            cv_tokens = set(pair[0].lower().split())
            jd_tokens = set(pair[1].lower().split())
            overlap = len(cv_tokens & jd_tokens)
            total = max(1, len(cv_tokens))
            scores.append(float(overlap) / float(total))
        return scores


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_rerank_scored_no_candidate(con, cfg, monkeypatch):
    """No candidate profile → returns skipped, no crash."""
    _seed_scored(con, "u/r1")
    result = features.rerank_scored(cfg, con)
    assert result.get("skipped") == "no_candidate"
    assert result.get("reranked", 0) == 0


def test_rerank_scored_no_scored_vacancies(con, cfg, monkeypatch):
    """No SCORED vacancies → reranked=0."""
    _seed_candidate(con)
    monkeypatch.setattr(features, "_load_reranker", lambda name: FakeReranker())
    result = features.rerank_scored(cfg, con)
    assert result.get("reranked", 0) == 0


def test_rerank_scored_reranker_unavailable(con, cfg, monkeypatch):
    """Reranker import fails → degraded gracefully."""
    _seed_candidate(con)
    _seed_scored(con, "u/r2")

    def _fail(name):
        raise ImportError("FlagEmbedding not installed")

    monkeypatch.setattr(features, "_load_reranker", _fail)
    result = features.rerank_scored(cfg, con)
    assert result.get("skipped") == "reranker_unavailable"
    assert result.get("reranked", 0) == 0


def test_rerank_scored_writes_xenc_features(con, cfg, monkeypatch):
    """Mocked reranker → xenc_full + xenc_musthave written into vacancy_feature."""
    _seed_candidate(con)
    vid = _seed_scored(con, "u/r3", score=5)

    monkeypatch.setattr(features, "_load_reranker", lambda name: FakeReranker())
    result = features.rerank_scored(cfg, con)
    assert result.get("reranked", 0) == 1

    feat = features.feature_row(con, vid)
    assert feat is not None
    assert "xenc_full" in feat
    assert "xenc_musthave" in feat
    assert 0.0 <= feat["xenc_full"] <= 1.0
    assert 0.0 <= feat["xenc_musthave"] <= 1.0


def test_rerank_scored_top_k_respected(con, cfg, monkeypatch):
    """top_k=1 should only rerank the top scorer, not all SCORED."""
    _seed_candidate(con)
    for i, score in enumerate([5, 4, 3]):
        _seed_scored(con, f"u/rk{i}", score=score)

    monkeypatch.setattr(features, "_load_reranker", lambda name: FakeReranker())
    result = features.rerank_scored(cfg, con, top_k=1)
    assert result.get("reranked", 0) == 1


def test_rerank_drains_unprocessed_below_top_score(con, cfg, monkeypatch):
    """Regression — cold-start starvation (the «куда деваются вакансии» bug). The top-by-judge
    candidate already carries xenc; a LOWER-scored one does not. The OLD code did
    `ORDER BY score DESC LIMIT k` then POST-filtered fresh rows, so with k=1 the fresh top-1
    drained to 'all_fresh' and the unprocessed job NEVER got reranked → fit_score stuck at 0 →
    couldn't fill an exploit slot. The WHERE-clause freshness filter now selects the top UNPROCESSED."""
    _seed_candidate(con)
    hi = _seed_scored(con, "u/hi", score=5)   # old gem: already reranked
    lo = _seed_scored(con, "u/lo", score=3)   # newer: extract_features ran, rerank never did
    con.execute(
        "INSERT OR REPLACE INTO vacancy_feature (vacancy_id, match_score, feature_json, computed_at)"
        " VALUES (?,?,?,datetime('now'))",
        (hi, 0.7, json.dumps({"match_score": 0.7, "xenc_full": 0.8, "fit_score": 0.8})),
    )
    con.execute(
        "INSERT OR REPLACE INTO vacancy_feature (vacancy_id, match_score, feature_json, computed_at)"
        " VALUES (?,?,?,datetime('now'))",
        (lo, 0.3, json.dumps({"match_score": 0.3})),   # partial row, NO xenc — the real case
    )
    con.commit()
    monkeypatch.setattr(features, "_load_reranker", lambda name: FakeReranker())
    out = features.rerank_scored(cfg, con, top_k=1)
    assert out.get("reranked") == 1                       # processed lo, did NOT short-circuit 'all_fresh'
    lo_feat = features.feature_row(con, lo)
    assert lo_feat.get("xenc_full") is not None           # lo finally reranked → fit can exceed 0
    assert lo_feat.get("fit_score") is not None


def test_rerank_prioritizes_fresh_over_old_within_budget(con, cfg, monkeypatch):
    """Regression — among judge-score ties the bounded top_k must go to the FRESHEST unprocessed
    (last_seen DESC), the cards the daily slate actually shows (last_seen within fresh_days). The old
    `ORDER BY js.score DESC` alone broke ties by id ASC → oldest-first → reranked stale judge-2 cards
    while the fresh slate pool starved at fit=0. With top_k=1 the fresh job must win."""
    _seed_candidate(con)
    old = _seed_scored(con, "u/old", score=2)
    new = _seed_scored(con, "u/new", score=2)
    con.execute("UPDATE vacancy SET last_seen='2026-06-01T00:00:00+00:00' WHERE id=?", (old,))
    con.execute("UPDATE vacancy SET last_seen='2026-06-30T00:00:00+00:00' WHERE id=?", (new,))
    con.commit()
    monkeypatch.setattr(features, "_load_reranker", lambda name: FakeReranker())
    out = features.rerank_scored(cfg, con, top_k=1)
    assert out.get("reranked") == 1
    assert features.feature_row(con, new).get("xenc_full") is not None    # fresh got the budget
    old_feat = features.feature_row(con, old)
    assert old_feat is None or old_feat.get("xenc_full") is None          # stale did NOT


def test_rerank_scored_multiple_vacancies(con, cfg, monkeypatch):
    """Reranks all SCORED within top_k."""
    _seed_candidate(con)
    for i in range(3):
        _seed_scored(con, f"u/rm{i}", score=5 - i)

    monkeypatch.setattr(features, "_load_reranker", lambda name: FakeReranker())
    result = features.rerank_scored(cfg, con, top_k=10)
    assert result.get("reranked", 0) == 3


def test_rerank_scored_idempotent(con, cfg, monkeypatch):
    """Running rerank_scored twice should update, not duplicate rows."""
    _seed_candidate(con)
    vid = _seed_scored(con, "u/ridem", score=4)
    monkeypatch.setattr(features, "_load_reranker", lambda name: FakeReranker())

    features.rerank_scored(cfg, con)
    features.rerank_scored(cfg, con)

    n = con.execute(
        "SELECT COUNT(*) FROM vacancy_feature WHERE vacancy_id = ?", (vid,)
    ).fetchone()[0]
    assert n == 1  # INSERT OR REPLACE ensures no duplicates


def test_real_reranker_loads_and_scores():
    """Smoke: _DirectReranker loads the real model and returns floats in [0,1].

    Skipped when the model isn't cached locally (CI / first run without model).
    Run manually after `schabasch rerank` to confirm the fix.
    """
    import os
    from pathlib import Path

    cache_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    model_dir = cache_root / "hub" / "models--BAAI--bge-reranker-v2-m3"
    if not model_dir.exists():
        pytest.skip("bge-reranker-v2-m3 not cached locally — run `schabasch rerank` first")

    reranker = features._DirectReranker("BAAI/bge-reranker-v2-m3")
    pairs = [
        ["Senior ML engineer, 6 yrs Python, PyTorch, NLP", "We need a senior ML engineer with Python and NLP."],
        ["Senior ML engineer, 6 yrs Python, PyTorch, NLP", "Looking for a junior front-end developer in React."],
    ]
    scores = reranker.compute_score(pairs, normalize=True)
    assert len(scores) == 2, f"expected 2 scores, got {scores}"
    assert all(isinstance(s, float) for s in scores), f"non-float scores: {scores}"
    assert all(0.0 <= s <= 1.0 for s in scores), f"scores out of [0,1]: {scores}"
    assert scores[0] > scores[1], f"relevant pair should outscore irrelevant: {scores}"


def test_slate_reads_xenc_score(con, cfg, monkeypatch):
    """slate.build_slate reads xenc_full from vacancy_feature (no crash when absent = 0)."""
    from schabasch import slate
    from datetime import date

    # Seed scored vacancy without feature row
    vid = _seed_scored(con, "u/slate_xenc", score=3)
    d = date.today().isoformat()
    items = slate.build_slate(cfg, con, d)
    # Should not crash; xenc_score defaults to 0.0 when no feature row
    assert isinstance(items, list)
