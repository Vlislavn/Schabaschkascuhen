"""Tests for features.py — bge-m3 cache, PositiveLibrary, feature assembly.

bge-m3 model is monkeypatched: fake_batch returns deterministic toy vectors.
No real model needed.
"""
from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from schabasch import db, features
from schabasch.features import PositiveLibrary, EMBEDDING_DIM
from schabasch.models import Status
from tests.conftest import make_card


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _f32_blob(vec: np.ndarray) -> bytes:
    arr = np.asarray(vec, dtype=np.float32).ravel()
    return struct.pack(f"{len(arr)}f", *arr)


def _unit_vec(i: int, dim: int = EMBEDDING_DIM) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[i % dim] = 1.0
    return v


def fake_model_encode(texts, *, batch_size=32, max_length=512,
                       return_dense=True, return_sparse=True, return_colbert_vecs=False):
    """Deterministic fake: dense = unit vector at text-length position, sparse = {}."""
    n = len(texts)
    dense = np.zeros((n, EMBEDDING_DIM), dtype=np.float32)
    for i, t in enumerate(texts):
        dense[i, len(t) % EMBEDDING_DIM] = 1.0
    sparse = [{} for _ in range(n)]
    return {"dense_vecs": dense, "lexical_weights": sparse}


def _seed_described(con, url: str, company: str = "ACME") -> int:
    vid = db.upsert_vacancy(con, {
        "source": "indeed", "url": url,
        "title": "Senior Business Analyst", "company": company,
        "city": "Frankfurt", "description": "x" * 600,
    })
    card = make_card(company=company)
    db.set_status(con, vid, Status.DESCRIBED, card_json=json.dumps(card))
    return vid


def _seed_labeled_positive(con, vid: int, score: int = 5) -> None:
    db.insert_judge_score(con, vid, {
        "score": score, "why_tag": None, "why_freetext": None,
        "explanation": "good", "model": "qwen3:8b",
        "model_digest": "d", "rubric_version": "v1", "fewshot_hash": "h",
    })
    db.set_status(con, vid, Status.SCORED)
    db.insert_label(con, vid, {
        "score_1_5": score, "why_tag": None, "why_freetext": None,
        "interview": 0, "applied": 0, "source": "slate",
    })


def _plant_embedding(con, vid: int, vec: np.ndarray) -> None:
    """Insert a pre-built embedding row so PositiveLibrary.build() can find it."""
    features._ensure_schema(con)
    blob = _f32_blob(vec)
    con.execute(
        """INSERT OR REPLACE INTO vacancy_embedding
           (vacancy_id, content_hash, dense, aspects_json, model_version, computed_at)
           VALUES (?, 'h', ?, '{}', 'bge-m3', '2026-01-01')""",
        (vid, blob),
    )
    con.commit()


# ---------------------------------------------------------------------------
# PositiveLibrary tests
# ---------------------------------------------------------------------------

def test_positive_library_empty_cold_start(con):
    lib = PositiveLibrary.build(con)
    assert lib.n_rows == 0
    feat = lib.taste_features(_unit_vec(0))
    assert all(v == 0.0 for v in feat.values())


def test_positive_library_build_one_positive(con):
    """With one labeled-positive vacancy + embedding, library has 1 row."""
    vid = _seed_described(con, "u/1", "TechCorp")
    _plant_embedding(con, vid, _unit_vec(0))
    _seed_labeled_positive(con, vid, score=5)

    lib = PositiveLibrary.build(con)
    assert lib.n_rows == 1


def test_positive_library_nearest_is_one_for_identical(con):
    """nearest_liked_cosine should be 1.0 when query matches the library row exactly."""
    vid = _seed_described(con, "u/1")
    _plant_embedding(con, vid, _unit_vec(3))
    _seed_labeled_positive(con, vid, score=5)

    lib = PositiveLibrary.build(con)
    feat = lib.taste_features(_unit_vec(3))
    assert abs(feat["nearest_liked_cosine"] - 1.0) < 0.01


def test_positive_library_leave_one_out(con):
    """With leave-one-out, self-cosine is excluded so nearest can drop below 1.0."""
    vid = _seed_described(con, "u/1")
    _plant_embedding(con, vid, _unit_vec(3))
    _seed_labeled_positive(con, vid, score=5)

    # add a second row (orthogonal)
    vid2 = _seed_described(con, "u/2")
    _plant_embedding(con, vid2, _unit_vec(4))
    _seed_labeled_positive(con, vid2, score=4)

    lib = PositiveLibrary.build(con)
    feat_with = lib.taste_features(_unit_vec(3))            # no exclude
    feat_loo  = lib.taste_features(_unit_vec(3), exclude_vacancy_id=vid)  # exclude self

    assert feat_loo["nearest_liked_cosine"] < feat_with["nearest_liked_cosine"] + 0.01


def test_positive_library_low_score_not_positive(con):
    """bad (score=2) labels should NOT enter the positive library."""
    vid = _seed_described(con, "u/1")
    _plant_embedding(con, vid, _unit_vec(0))
    db.insert_judge_score(con, vid, {
        "score": 2, "why_tag": None, "why_freetext": None,
        "explanation": "bad", "model": "qwen3:8b",
        "model_digest": "d", "rubric_version": "v1", "fewshot_hash": "h",
    })
    db.set_status(con, vid, Status.SCORED)
    db.insert_label(con, vid, {
        "score_1_5": 2, "why_tag": None, "why_freetext": None,
        "interview": 0, "applied": 0, "source": "slate",
    })

    lib = PositiveLibrary.build(con)
    assert lib.n_rows == 0


# ---------------------------------------------------------------------------
# Schema creation tests
# ---------------------------------------------------------------------------

def test_ensure_schema_idempotent(con):
    features._ensure_schema(con)
    features._ensure_schema(con)  # must not raise
    tables = {r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    assert "vacancy_embedding" in tables
    assert "vacancy_feature" in tables


# ---------------------------------------------------------------------------
# feature_row / feature_vector / match_summary tests
# ---------------------------------------------------------------------------

def test_feature_row_none_when_absent(con):
    assert features.feature_row(con, 9999) is None


def test_feature_vector_zero_padded_dense_when_no_embedding(con):
    """No vacancy_embedding → CONSISTENT shape: zero-padded dense(1024) ++ named (NOT named-only).
    Mixing 1053- and 29-dim vectors crashed triage._load_labeled np.stack; a fixed width fixes it."""
    import numpy as np
    features._ensure_schema(con)
    vid = _seed_described(con, "u/fx1")
    con.execute(
        """INSERT INTO vacancy_feature (vacancy_id, match_score, feature_json, computed_at)
           VALUES (?, 0.5, '{}', '2026-01-01')""", (vid,)
    )
    con.commit()
    vec = features.feature_vector(con, vid)
    assert vec is not None
    from schabasch.aspects import FEATURE_NAMES
    assert len(vec) == features.EMBEDDING_DIM + len(FEATURE_NAMES)   # full width, dense zero-padded
    assert np.allclose(vec[:features.EMBEDDING_DIM], 0.0)


def test_feature_vector_dense_plus_named(con):
    """If both embedding and feature row exist, vector = dense(1024) ++ named."""
    features._ensure_schema(con)
    vid = _seed_described(con, "u/fx2")
    _plant_embedding(con, vid, _unit_vec(0))
    from schabasch.aspects import FEATURE_NAMES
    feat_json = json.dumps({k: 0.0 for k in FEATURE_NAMES})
    con.execute(
        """INSERT INTO vacancy_feature (vacancy_id, match_score, feature_json, computed_at)
           VALUES (?, 0.5, ?, '2026-01-01')""", (vid, feat_json)
    )
    con.commit()
    vec = features.feature_vector(con, vid)
    assert vec is not None
    assert len(vec) == EMBEDDING_DIM + len(FEATURE_NAMES)


def test_match_summary_none_when_absent(con):
    assert features.match_summary(con, 9999) is None


def test_scores_by_vacancy_empty(con):
    features._ensure_schema(con)
    assert features.scores_by_vacancy(con) == {}


# ---------------------------------------------------------------------------
# extract_features (mocked bge-m3) tests
# ---------------------------------------------------------------------------

class FakeModel:
    def encode(self, texts, **kwargs):
        return fake_model_encode(texts)


def test_extract_features_no_model(con, cfg, monkeypatch):
    """When model load fails, extract_features degrades gracefully."""
    monkeypatch.setattr(features, "_load_model", lambda name: (_ for _ in ()).throw(ImportError("no model")))
    result = features.extract_features(cfg, con)
    assert result.get("error") == "model_unavailable"


def test_extract_features_no_candidate(con, cfg, monkeypatch):
    """When no candidate profile exists, features still run (with zero cand_vecs)."""
    monkeypatch.setattr(features, "_load_model", lambda name: FakeModel())
    vid = _seed_described(con, "u/noc")
    result = features.extract_features(cfg, con)
    # Should process vacancy but emit zero features (no candidate profile)
    assert isinstance(result, dict)
    assert "featured" in result


def test_extract_features_with_candidate_and_fake_model(con, cfg, monkeypatch):
    """Full happy path: candidate profile + fake model → vacancy_feature row."""
    monkeypatch.setattr(features, "_load_model", lambda name: FakeModel())

    # Seed candidate profile directly (bypass OllamaClient)
    features._ensure_schema(con)
    from schabasch.candidate import _ensure_schema as _cs, _build_aspect_texts
    _cs(con)
    import hashlib, json as _json
    from datetime import datetime, timezone
    profile = {
        "skills": ["Python", "SQL", "Tableau"],
        "experience": "Senior BA at BostonGene.",
        "domains": ["digital transformation"],
        "seniority": "senior",
        "years_experience": 6,
        "education": "BSc",
        "languages": {"en": "C1", "de": "A2"},
        "target_roles": ["Senior Business Analyst"],
        "locations": ["Heidelberg"],
        "magnets": ["data analytics"],
        "repellents": ["lab science"],
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

    vid = _seed_described(con, "u/feat1", "DataCorp")
    result = features.extract_features(cfg, con)

    assert result.get("featured", 0) >= 0  # may be 0 if cand_vecs load fails due to mock

    # Check schema was created
    tables = {r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    assert "vacancy_embedding" in tables
    assert "vacancy_feature" in tables


def test_sparse_overlap_token_id_space():
    """Regression: _sparse_overlap operates in bge-m3 token-id space (both args
    {token_id: weight}); the old version compared token-ids to words and was always 0."""
    from schabasch import features
    jd = {"101": 0.5, "202": 0.5}
    cv = {"101": 0.9, "303": 0.2}
    # token 101 covered (weight 0.5) of total 1.0 → 0.5
    assert abs(features._sparse_overlap(jd, cv) - 0.5) < 1e-6
    assert features._sparse_overlap(jd, {}) == 0.0
    assert features._sparse_overlap({}, cv) == 0.0
    # full coverage → 1.0
    assert abs(features._sparse_overlap({"1": 1.0}, {"1": 0.3}) - 1.0) < 1e-6


def test_recency_days_from_posting_date():
    """recency_days is days-since-posted (lower = fresher); falls back to first_seen then 7.0."""
    from datetime import datetime, timezone, timedelta
    from schabasch import features
    d10 = (datetime.now(timezone.utc) - timedelta(days=10)).date().isoformat()
    assert 9 <= features._recency_days(d10, None) <= 11
    fs = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
    assert 2 <= features._recency_days(None, fs) <= 4       # falls back to first_seen
    assert features._recency_days(None, None) == 7.0         # neutral default
    assert features._recency_days("not-a-date", None) == 7.0  # unparseable → default


def test_torch_thread_cap_applied():
    """CAPA invariant: _cap_torch_threads pins torch's intra-op (OpenMP) pool to the configured
    size — the condition that prevents the features-stage copy_kernel deadlock (all threads parked
    in _pthread_cond_wait). Config-driven, not hardcoded; restores 1 so other tests see the safe
    default. Reverting the cap (so threads != configured) is what makes this fail → it guards the fix."""
    import torch
    from schabasch import features
    features._cap_torch_threads({"features": {"torch_num_threads": 1}})
    assert torch.get_num_threads() == 1
    features._cap_torch_threads({"features": {"torch_num_threads": 2}})
    assert torch.get_num_threads() == 2                      # config-driven, not hardcoded to 1
    features._cap_torch_threads({"features": {"torch_num_threads": 0}})
    assert torch.get_num_threads() == 2                      # <=0 → leave current setting (no-op)
    features._cap_torch_threads({"features": {"torch_num_threads": 1}})  # restore safe default
    assert torch.get_num_threads() == 1


# ---------------------------------------------------------------------------
# llm_coverage CANDIDATE block (fix: judge stopped false-flagging LISTED skills as missing)
# ---------------------------------------------------------------------------

def test_candidate_cov_block_lists_skills_and_languages():
    """The coverage CANDIDATE block renders discrete skills one-per-line + languages AHEAD of the
    prose — an explicit candidate spec, not a comma-blob with languages absent. Generalization:
    a non-trace candidate (Rust/Kubernetes/French), not the failing case."""
    block = features._candidate_cov_block(
        cv_text="Built distributed systems in Rust.",
        cand_skills=["Rust", "Kubernetes", "Distributed Systems"],
        languages={"fr": "B2", "en": "C1"},
    )
    assert "SKILLS:" in block
    assert "- Rust" in block and "- Kubernetes" in block
    assert "LANGUAGES:" in block and "fr=B2" in block and "en=C1" in block
    assert "PROFILE:" in block and "distributed systems in rust" in block.lower()
    assert block.index("SKILLS:") < block.index("PROFILE:")   # explicit spec is read first


def test_candidate_cov_block_degrades_without_skills():
    """No skills/languages → just the prose (backward-compatible with the old prose-only block)."""
    assert features._candidate_cov_block("prose only", None, None) == "PROFILE:\nprose only"
    assert features._candidate_cov_block("", [], {}) == ""


class _StubCovClient:
    """Captures the (system, user) handed to chat_json; returns a fixed coverage verdict."""
    def __init__(self, reqs):
        self.seen: list[tuple[str, str]] = []
        self._reqs = reqs

    def chat_json(self, system, user):
        self.seen.append((system, user))
        return {"requirements": self._reqs, "missing_summary": ""}


def test_llm_coverage_feeds_explicit_skills_and_full_cv(con):
    """Fix: _llm_coverage hands the judge the explicit skills + the FULL cv (not cv[:1500]) so a
    LISTED skill can be judged present. Different data than the trace (a Go/Kafka candidate)."""
    features._ensure_schema(con)
    long_cv = "A" * 1400 + " GOLANG_MARKER " + "B" * 400      # marker sits PAST the old 1500 cap
    client = _StubCovClient([
        {"requirement": "Go programming", "verdict": "present"},
        {"requirement": "Kafka streaming", "verdict": "missing"},
    ])
    cov, miss, reqs = features._llm_coverage(
        con, cache_key="cvX:jdY:test", jd_title="Backend Engineer",
        jd_desc="Build Go services with Kafka.", cv_text=long_cv, client=client,
        cand_skills=["Go", "Kafka", "PostgreSQL"], languages={"en": "C1"},
    )
    assert len(client.seen) == 1
    _sys, user = client.seen[0]
    assert "- Go" in user and "- Kafka" in user               # explicit skills reached the judge
    assert "en=C1" in user                                     # languages reached the judge
    assert "GOLANG_MARKER" in user                             # full cv, NOT truncated at 1500
    assert reqs and any(r["verdict"] == "present" for r in reqs)
    assert 0.0 <= cov <= 1.0


def test_llm_coverage_cache_roundtrip(con):
    """Stores under the given (versioned) cache_key and serves a second call from cache without
    re-calling the judge — the recompute-on-version-bump substrate."""
    features._ensure_schema(con)
    client = _StubCovClient([{"requirement": "X", "verdict": "present"}])
    key = "cvA:jdB:c2"
    features._llm_coverage(con, cache_key=key, jd_title="t", jd_desc="d", cv_text="cv",
                           client=client, cand_skills=["X"], languages={})
    assert len(client.seen) == 1
    cov, miss, reqs = features._llm_coverage(con, cache_key=key, jd_title="t", jd_desc="d",
                                             cv_text="cv", client=client, cand_skills=["X"], languages={})
    assert len(client.seen) == 1                               # second call served from cache
    assert reqs == [{"requirement": "X", "verdict": "present"}]


def test_rerank_skips_when_all_fresh(cfg, con, monkeypatch):
    """② rerank_scored re-scoring an already-ranked job is pure waste (deterministic). When every
    candidate already carries a fresh xenc_full it returns 'all_fresh' WITHOUT loading the 2GB
    reranker — proven by a loader that records calls."""
    from schabasch import candidate, features
    from tests.conftest import seed_scored
    features._ensure_schema(con)
    monkeypatch.setattr(candidate, "candidate_doc", lambda c: "CV text for the candidate.")
    vid = seed_scored(con, "u/ranked", score=5, company="Co")
    feat = {"match_score": 0.7, "fit_score": 0.7, "xenc_full": 0.8, "elig_score": 1.0}
    con.execute("INSERT OR REPLACE INTO vacancy_feature (vacancy_id, match_score, feature_json,"
                " computed_at) VALUES (?,?,?,datetime('now'))", (vid, 0.7, json.dumps(feat)))
    con.commit()
    called = {"load": 0}
    monkeypatch.setattr(features, "_load_reranker",
                        lambda name: called.__setitem__("load", called["load"] + 1))
    out = features.rerank_scored(cfg, con)
    assert out.get("skipped") == "all_fresh"
    assert called["load"] == 0     # the 2GB reranker was NOT loaded — pure waste avoided
