"""Tier-1 SOTA matching fix: genuine fit signal (cross-encoder + HyRE) + de-conflated ranking.

All offline: bge-m3 + reranker + ollama are mocked. References:
- HyRE: ConFit-v2/src/utils/convert_by_llm.py:44-55 (ideal-résumé), metrics.py:301-330 ((dot+1)/2)
- de-conflate: DualOptimization_jobrec/stage_2/tau_0.01.py:627-637 (qualification gates preference)
"""
from __future__ import annotations

import json
from datetime import date, datetime, timezone

import numpy as np

from schabasch import db, features
from schabasch.aspects import FEATURE_NAMES
from schabasch.llm import OllamaClient
from schabasch.models import Status
from tests.conftest import make_card


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------

def test_cosine01_range_and_identity():
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    assert abs(features._cosine01(a, a) - 1.0) < 1e-6        # identical → 1.0
    assert abs(features._cosine01(a, -a) - 0.0) < 1e-6       # opposite → 0.0
    b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    assert abs(features._cosine01(a, b) - 0.5) < 1e-6        # orthogonal → 0.5
    assert features._cosine01(a, np.zeros(3, dtype=np.float32)) == 0.0  # zero-norm safe


def test_blend_fit_renormalizes_over_present_signals():
    w = {"xenc": 0.5, "llm_cov": 0.5}
    # both present
    assert abs(features._blend_fit(xenc_full=0.8, llm_cov=0.4, weights=w)
               - (0.5*0.8 + 0.5*0.4)) < 1e-6
    # llm_cov missing → renormalize over xenc alone → 0.8
    assert abs(features._blend_fit(xenc_full=0.8, llm_cov=None, weights=w) - 0.8) < 1e-6
    # nothing present → 0
    assert features._blend_fit(xenc_full=None, llm_cov=None, weights=w) == 0.0


def test_hybrid_sparse_blend_and_fit_from_feature():
    """SOTA hybrid (2026-06-15 PM): fit = 0.7·hyre + 0.3·sparse_norm, sparse normalized by a fixed
    divisor (per-vacancy, no set-relative min-max), graceful fallback to HyRE when sparse is absent."""
    w = {"hyre": 0.7, "sparse": 0.3}
    # _blend_fit fuses hyre + sparse_norm
    assert abs(features._blend_fit(xenc_full=None, llm_cov=None, fit_hyre=0.8, sparse_norm=0.5,
                                   weights=w) - (0.7 * 0.8 + 0.3 * 0.5)) < 1e-6
    # fit_from_feature normalizes raw bgem3_sparse by sparse_scale and clamps to 1.0
    feat = {"fit_hyre": 0.8, "bgem3_sparse": 0.45}
    assert abs(features.fit_from_feature(feat, w, sparse_scale=0.45)
               - (0.7 * 0.8 + 0.3 * 1.0)) < 1e-6        # 0.45/0.45 = 1.0 (clamped)
    feat2 = {"fit_hyre": 0.8, "bgem3_sparse": 0.225}
    assert abs(features.fit_from_feature(feat2, w, sparse_scale=0.45)
               - (0.7 * 0.8 + 0.3 * 0.5)) < 1e-6        # 0.225/0.45 = 0.5
    # no sparse component → renormalize over HyRE alone (pure-HyRE fallback during rollout)
    assert abs(features.fit_from_feature({"fit_hyre": 0.8}, w) - 0.8) < 1e-6


def test_fit_from_feature_treats_zero_hyre_as_absent():
    """Regression (2026-06-22): a LABELED gold job that never enters the rerank pool keeps the
    extract_features fit_hyre=0.0 default. Since HyRE carries weight 0.7, that phantom zero used to
    drag fit to ~0 (the under-scoring bug for jobs 3321/3069 the user rated 4). A real HyRE cosine is
    (cos+1)/2, never exactly 0.0 → 0.0 means 'not computed' and must fall back to the sparse signal."""
    w = {"hyre": 0.7, "sparse": 0.3}
    feat = {"fit_hyre": 0.0, "bgem3_sparse": 0.225}      # sparse_norm = 0.225/0.45 = 0.5
    # BEFORE the fix: (0.7·0.0 + 0.3·0.5)/1.0 = 0.15 (heaviest signal zeroes the score)
    # AFTER  the fix: hyre dropped → renormalize over sparse alone → 0.5
    assert abs(features.fit_from_feature(feat, w, sparse_scale=0.45) - 0.5) < 1e-6
    # a genuine (non-zero) HyRE is still blended normally
    feat2 = {"fit_hyre": 0.8, "bgem3_sparse": 0.225}
    assert abs(features.fit_from_feature(feat2, w, sparse_scale=0.45) - (0.7 * 0.8 + 0.3 * 0.5)) < 1e-6
    # both signals zero/absent → 0.0 (no rescue from nothing)
    assert features.fit_from_feature({"fit_hyre": 0.0}, w, sparse_scale=0.45) == 0.0


def test_feature_vector_consistent_shape_with_and_without_embedding(con, cfg):
    """Regression: feature_vector must return the SAME width whether or not a bge-m3 embedding exists
    (zero-padded dense), else triage._load_labeled np.stack crashes on mixed 1053/29-dim rows and a
    trained model can't score an un-embedded vacancy."""
    from schabasch.aspects import FEATURE_NAMES
    features._ensure_schema(con)
    # vacancy A: feature row WITH a dense embedding
    a = db.upsert_vacancy(con, {"source": "indeed", "url": "u/a", "title": "A", "company": "C",
                                "city": "Frankfurt", "description": "x" * 200})
    b = db.upsert_vacancy(con, {"source": "indeed", "url": "u/b", "title": "B", "company": "C",
                                "city": "Frankfurt", "description": "y" * 200})
    feat = {k: 0.1 for k in FEATURE_NAMES}
    for vid in (a, b):
        con.execute("INSERT OR REPLACE INTO vacancy_feature (vacancy_id, match_score, feature_json,"
                    " computed_at) VALUES (?,?,?,datetime('now'))", (vid, 0.5, json.dumps(feat)))
    dense = np.ones(features.EMBEDDING_DIM, dtype=np.float32)
    con.execute("INSERT OR REPLACE INTO vacancy_embedding (vacancy_id, content_hash, dense,"
                " aspects_json, model_version, computed_at) VALUES (?,?,?,?,?,datetime('now'))",
                (a, "h", features._vec_to_blob(dense), "{}", "bge-m3"))
    con.commit()
    va = features.feature_vector(con, a)   # has embedding
    vb = features.feature_vector(con, b)   # no embedding → zero-padded dense
    assert va.shape == vb.shape == (features.EMBEDDING_DIM + len(FEATURE_NAMES),)
    assert np.allclose(vb[:features.EMBEDDING_DIM], 0.0)   # missing dense → zeros, not dropped
    assert np.allclose(va[:features.EMBEDDING_DIM], 1.0)


def test_bgem3_sparse_scores_helper():
    """bgem3_sparse_scores batches CV↔JD pairs through model.compute_score and returns the sparse list."""
    class _M:
        def compute_score(self, pairs, **kw):
            assert all(len(p) == 2 for p in pairs)        # [cv, jd] pairs
            return {"sparse": [0.3 * (i + 1) for i in range(len(pairs))],
                    "dense": [0.0] * len(pairs), "colbert": [0.0] * len(pairs)}
    out = features.bgem3_sparse_scores(_M(), "cv text", ["jd one", "jd two"])
    assert out == [0.3, 0.6]
    assert features.bgem3_sparse_scores(_M(), "cv", []) == []


def test_fit_features_registered():
    assert "fit_hyre" in FEATURE_NAMES and "fit_score" in FEATURE_NAMES
    from schabasch.triage import _NAMED_CONSTRAINTS
    assert _NAMED_CONSTRAINTS["fit_hyre"] == 1 and _NAMED_CONSTRAINTS["fit_score"] == 1


# ---------------------------------------------------------------------------
# rerank → fit_score (cross-encoder path, HyRE off)
# ---------------------------------------------------------------------------

class _FakeReranker:
    """xenc score = fraction of CV tokens present in the JD text (deterministic)."""
    def compute_score(self, pairs, normalize=True):
        out = []
        for cv, jd in pairs:
            cvt = set(cv.lower().split())
            out.append(len(cvt & set(jd.lower().split())) / max(1, len(cvt)))
        return out


def _seed_candidate(con):
    from schabasch.candidate import _ensure_schema as _cs
    features._ensure_schema(con); _cs(con)
    at = {"skills_text": "python sql", "experience_text": "x", "domains_text": "d",
          "roles_text": "analyst", "full_doc": "python sql analyst data"}
    con.execute(
        "INSERT INTO candidate_profile (created_at, raw_input, profile_json, aspect_texts, doc_hash)"
        " VALUES (?,?,?,?,?)",
        (datetime.now(timezone.utc).isoformat(), "t",
         json.dumps({"skills": ["python", "sql"]}), json.dumps(at), "hash16"),
    )
    con.commit()


def _seed_scored(con, url, *, score, company="ACME", title="Engineer", desc="python sql role"):
    vid = db.upsert_vacancy(con, {"source": "indeed", "url": url, "title": title,
                                  "company": company, "city": "Frankfurt", "description": desc})
    db.set_status(con, vid, Status.DESCRIBED, card_json=json.dumps(make_card(company=company)))
    db.insert_judge_score(con, vid, {"score": score, "why_tag": None, "why_freetext": None,
                                     "explanation": "x", "model": "qwen3:8b", "model_digest": "d",
                                     "rubric_version": "test-v1", "fewshot_hash": "h"})
    db.set_status(con, vid, Status.SCORED)
    return vid


def test_rerank_computes_fit_score_xenc_only(con, cfg, monkeypatch):
    _seed_candidate(con)
    vid = _seed_scored(con, "u/f1", score=5)
    monkeypatch.setattr(features, "_load_reranker", lambda name: _FakeReranker())
    res = features.rerank_scored(cfg, con)              # cfg has hyre=False
    assert res["reranked"] == 1 and res.get("hyre", 0) == 0
    feat = features.feature_row(con, vid)
    assert "fit_score" in feat and 0.0 <= feat["fit_score"] <= 1.0
    # with HyRE off and no coverage feature, fit_score == xenc_musthave
    assert abs(feat["fit_score"] - feat["xenc_musthave"]) < 1e-6
    assert "fit_note" in feat


def test_hyre_degrades_when_model_unavailable(con, cfg, monkeypatch):
    """HyRE on, but bge-m3/ollama unavailable → rerank must NOT crash and still produce fit_score
    from the cross-encoder (hyre falls back to None). The one previously-untested shipped path."""
    _seed_candidate(con)
    vid = _seed_scored(con, "u/degrade", score=5)
    cfg = {**cfg, "features": {**cfg["features"], "hyre": True}}

    def _boom(name=None):
        raise RuntimeError("bge-m3 unavailable")
    monkeypatch.setattr(features, "_load_model", _boom)
    monkeypatch.setattr(features, "_load_reranker", lambda name: _FakeReranker())

    res = features.rerank_scored(cfg, con)
    assert res["reranked"] == 1 and res["hyre"] == 0      # degraded: no HyRE, no crash
    feat = features.feature_row(con, vid)
    assert "fit_score" in feat and "fit_hyre" not in feat  # fit from xenc only
    assert abs(feat["fit_score"] - feat["xenc_musthave"]) < 1e-6


def test_hyre_generated_then_cached(con, cfg, monkeypatch):
    """HyRE: qwen writes an ideal résumé once per JD (cached by content_hash); fit_hyre computed."""
    _seed_candidate(con)
    vid = _seed_scored(con, "u/h1", score=5)
    cfg = {**cfg, "features": {**cfg["features"], "hyre": True}}

    calls = {"chat": 0}

    def _chat(self, system, user):
        calls["chat"] += 1
        return {"ideal_resume": "python sql data analyst with 6 years experience"}
    monkeypatch.setattr(OllamaClient, "chat_json", _chat)

    class _FakeM3:
        def encode(self, texts, **kw):
            d = np.zeros((len(texts), features.EMBEDDING_DIM), dtype=np.float32)
            d[:, 0] = 1.0   # all same direction → cosine01 = 1.0
            return {"dense_vecs": d, "lexical_weights": [{} for _ in texts]}
    monkeypatch.setattr(features, "_load_model", lambda name=None: _FakeM3())
    monkeypatch.setattr(features, "_load_reranker", lambda name: _FakeReranker())

    r1 = features.rerank_scored(cfg, con)
    assert r1["hyre"] == 1
    feat = features.feature_row(con, vid)
    assert feat.get("fit_hyre") == 1.0
    assert calls["chat"] == 1
    # second run → cache hit, no new chat call
    features.rerank_scored(cfg, con)
    assert calls["chat"] == 1


# ---------------------------------------------------------------------------
# de-conflate: fit gates the magnet (down-rank, not hard-drop)
# ---------------------------------------------------------------------------

def _write_feat(con, vid, *, fit_score, match_score=0.3):
    features._ensure_schema(con)
    feat = {"match_score": match_score, "fit_score": fit_score, "fit_hyre": fit_score,
            "fit_note": ("крупный разрыв по навыкам" if fit_score < 0.4 else ""),
            "xenc_full": fit_score, "xenc_musthave": fit_score}
    con.execute("INSERT OR REPLACE INTO vacancy_feature (vacancy_id, match_score, feature_json, computed_at)"
                " VALUES (?,?,?,?)", (vid, match_score, json.dumps(feat),
                                      datetime.now(timezone.utc).isoformat()))
    con.commit()


def test_deconflate_downranks_high_judge_low_fit(con, cfg):
    from schabasch import slate
    # A: magnet job, judge 5 but the user can't do it (fit 0.2). B: judge 4, genuine fit 0.95.
    a = _seed_scored(con, "u/space", score=5, company="SpaceCo", title="Orbital Engineer")
    b = _seed_scored(con, "u/data", score=4, company="DataCo", title="Data Analyst")
    _write_feat(con, a, fit_score=0.2)
    _write_feat(con, b, fit_score=0.95)
    items = slate.build_slate(cfg, con, date.today().isoformat())
    order = [it["vacancy_id"] for it in items]
    assert order.index(b) < order.index(a), "genuine-fit job must outrank the unqualified magnet job"
    # but the magnet job is NOT hidden (down-rank, not hard-drop)
    assert a in order


def test_skill_gap_note_renders(con, cfg):
    from schabasch import slate
    a = _seed_scored(con, "u/gap", score=5, company="SpaceCo")
    _write_feat(con, a, fit_score=0.2)
    items = slate.build_slate(cfg, con, date.today().isoformat())
    html = slate.render_html(items, date.today().isoformat())
    assert "⚠" in html and "разрыв по навыкам" in html


def test_llm_coverage_caches_and_scores(con):
    """LLM per-requirement coverage = (present + 0.5*partial)/total, cached, neutral-on-failure."""
    from schabasch import features
    features._ensure_schema(con)
    calls = {"n": 0}

    class _C:
        def chat_json(self, system, user):
            calls["n"] += 1
            return {"requirements": [{"requirement": "Python", "verdict": "present"},
                                     {"requirement": "MATLAB", "verdict": "missing"},
                                     {"requirement": "ML", "verdict": "partial"}],
                    "missing_summary": "MATLAB"}
    cov, miss, reqs = features._llm_coverage(con, cache_key="k1", jd_title="T", jd_desc="d",
                                             cv_text="cv", client=_C())
    assert abs(cov - (1 + 0.5) / 3) < 1e-6 and miss == "MATLAB" and calls["n"] == 1
    # the per-requirement list (for the card breakdown) is returned + persisted
    assert [r["verdict"] for r in reqs] == ["present", "missing", "partial"]
    cov_c, miss_c, reqs_c = features._llm_coverage(con, cache_key="k1", jd_title="T", jd_desc="d",
                                                   cv_text="cv", client=_C())
    assert calls["n"] == 1 and len(reqs_c) == 3  # cache hit returns persisted reqs, no second call

    class _Boom:
        def chat_json(self, s, u):
            raise RuntimeError("ollama down")
    cov2, miss2, reqs2 = features._llm_coverage(con, cache_key="k2", jd_title="T", jd_desc="d",
                                                cv_text="cv", client=_Boom())
    assert cov2 == 0.5 and miss2 == "" and reqs2 == []   # neutral on failure (no boost, no gate)


def test_skill_breakdown_renders_and_supersedes_fit_note():
    """Collapsible per-skill block: headline llm_cov% + ✓/◐/✗ list; supersedes the 1-line fit ⚠."""
    from schabasch import slate
    h = slate._skills_html({"llm_cov": 0.75, "llm_cov_reqs": [
        {"requirement": "Business analysis", "verdict": "present"},
        {"requirement": "BPMN", "verdict": "present"},
        {"requirement": "Kubernetes", "verdict": "missing"},
        {"requirement": "ML", "verdict": "partial"}]})
    assert "🎯 Skills 75%" in h and "2 ✓" in h and "1 ◐" in h and "1 ✗" in h
    assert "<details" in h and "✗ Kubernetes" in h and "✓ Business analysis" in h
    # no per-requirement data → empty (graceful: llm_cov off / pre-rerank)
    assert slate._skills_html({"llm_cov": None, "llm_cov_reqs": []}) == ""
    assert slate._skills_html({"llm_cov": 0.5, "llm_cov_reqs": []}) == ""
    # in a card: the skill block present → the one-line fit ⚠ note is suppressed (Occam)
    card = slate._card_block({"vacancy_id": 1, "title": "X", "llm_cov": 0.3,
                              "llm_cov_reqs": [{"requirement": "BPMN", "verdict": "missing"}],
                              "fit_score": 0.2, "fit_note": "крупный разрыв"})
    assert "🎯 Skills" in card and "крупный разрыв" not in card
    # no reqs → falls back to the fit ⚠ gap note
    card2 = slate._card_block({"vacancy_id": 2, "title": "Y", "fit_score": 0.2, "fit_note": "крупный разрыв"})
    assert "крупный разрыв" in card2
