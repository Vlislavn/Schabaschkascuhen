"""bge-m3 multi-vector embedding cache + aspect feature assembly.

Owns: bge-m3 model (lazy, cached), vacancy_embedding sidecar,
      vacancy_feature sidecar, PositiveLibrary.
Delegates: segment_jd / score math → aspects.py.

bge-m3 API (verified):
  BGEM3FlagModel("BAAI/bge-m3", use_fp16=False, devices="cpu")
  model.encode(texts, batch_size=32, max_length=512,
               return_dense=True, return_sparse=True, return_colbert_vecs=True)
  → {"dense_vecs": (N,1024) f32 L2-normed, "lexical_weights": list[dict tokid→w],
     "colbert_vecs": list[(toks,1024)]}
  NOTE: `devices` (plural) NOT `device`; use_fp16=False on CPU/Mac.
"""
from __future__ import annotations

import base64
import hashlib
import json
import struct
from datetime import datetime, timezone
from typing import Any

import numpy as np

from . import aspects as _asp
from . import eligibility as _elig
from . import geo as _geo
from . import hardfilters as _hard
from .candidate import aspect_texts as _cand_aspect_texts, load_candidate
from .models import Status, normalize_company

EMBEDDING_DIM = 1024
_MODEL_CACHE: dict[str, Any] = {}

# ---------------------------------------------------------------------------
# Sidecar schemas
# ---------------------------------------------------------------------------

_SCHEMA_EMBEDDING = """
CREATE TABLE IF NOT EXISTS vacancy_embedding (
    vacancy_id    INTEGER PRIMARY KEY,
    content_hash  TEXT NOT NULL,
    dense         BLOB NOT NULL,
    aspects_json  TEXT,
    model_version TEXT NOT NULL,
    computed_at   TEXT NOT NULL
)
"""

_SCHEMA_FEATURE = """
CREATE TABLE IF NOT EXISTS vacancy_feature (
    vacancy_id   INTEGER PRIMARY KEY,
    match_score  REAL,
    feature_json TEXT NOT NULL,
    computed_at  TEXT NOT NULL
)
"""

# HyRE (ConFit-v2): cache the LLM-generated "ideal résumé" + its bge-m3 vector per JD content-hash,
# so the qwen call + embed happen once per vacancy (the cosine vs the live CV is recomputed cheaply).
_SCHEMA_HYRE = """
CREATE TABLE IF NOT EXISTS hyre_cache (
    content_hash TEXT PRIMARY KEY,
    ideal_resume TEXT NOT NULL,
    ideal_vec    BLOB NOT NULL,
    computed_at  TEXT NOT NULL
)
"""

# LLM per-requirement coverage (ConFit-v3 "non-negotiable requirements"): qwen lists each must-have
# and judges present/partial/missing vs the CV → covered/total. Cached per (CV-hash, JD-hash).
# Validated as the strongest single fit signal on the gold set (pairwise 0.80 vs cross-encoder 0.71).
_SCHEMA_LLMCOV = """
CREATE TABLE IF NOT EXISTS llmcov_cache (
    cache_key    TEXT PRIMARY KEY,
    coverage     REAL NOT NULL,
    missing      TEXT,
    requirements TEXT,            -- JSON [{requirement, verdict}] for the per-skill card breakdown
    computed_at  TEXT NOT NULL
)
"""


def _ensure_schema(con) -> None:
    con.execute(_SCHEMA_EMBEDDING)
    con.execute(_SCHEMA_FEATURE)
    con.execute(_SCHEMA_HYRE)
    con.execute(_SCHEMA_LLMCOV)
    # additive migration (CREATE TABLE IF NOT EXISTS never alters an existing table) — same
    # PRAGMA pattern as db._migrate. Old cache rows lack the per-requirement list → NULL → recompute.
    cols = {r[1] for r in con.execute("PRAGMA table_info(llmcov_cache)")}
    if "requirements" not in cols:
        con.execute("ALTER TABLE llmcov_cache ADD COLUMN requirements TEXT")
    con.commit()


# ---------------------------------------------------------------------------
# bge-m3 model (lazy load, cached per process)
# ---------------------------------------------------------------------------

def _load_model(model_name: str = "BAAI/bge-m3"):
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]
    from FlagEmbedding import BGEM3FlagModel  # type: ignore
    model = BGEM3FlagModel(model_name, use_fp16=False, devices="cpu")
    _MODEL_CACHE[model_name] = model
    return model


class _DirectReranker:
    """Calls bge-reranker-v2-m3 directly via transformers, bypassing FlagReranker.

    FlagReranker.compute_score calls tokenizer.prepare_for_model which was
    removed in transformers >=4.40; this wrapper hits the model directly so
    the transformers version doesn't matter.
    """

    def __init__(self, model_name: str) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self._model.eval()
        self._torch = torch

    def compute_score(self, pairs: list[list[str]], normalize: bool = True) -> list[float]:
        torch = self._torch
        results: list[float] = []
        for i in range(0, len(pairs), 16):
            batch = pairs[i : i + 16]
            enc = self._tok(
                [p[0] for p in batch],
                [p[1] for p in batch],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            with torch.no_grad():
                logits = self._model(**enc).logits.squeeze(-1)
            s = (torch.sigmoid(logits) if normalize else logits).tolist()
            results.extend([s] if isinstance(s, float) else s)
        return results


def _load_reranker(model_name: str = "BAAI/bge-reranker-v2-m3"):
    key = f"reranker:{model_name}"
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    reranker = _DirectReranker(model_name)
    _MODEL_CACHE[key] = reranker
    return reranker


# ---------------------------------------------------------------------------
# HyRE (Hypothetical Resume Embedding) — ConFit-v2 pattern.
# Ref: ~/code/from GH/ConFit-v2/src/utils/convert_by_llm.py:44-55 (ideal-résumé prompt),
#      src/evaluation/metrics.py:301-330 (normalized cosine (dot+1)/2).
# CV-vs-JD cosine is genre-mismatched (~0.4–0.6 for any pair); comparing the CV to an
# LLM-written *ideal résumé for this JD* (same genre) is genuinely discriminative.
# ---------------------------------------------------------------------------

_HYRE_SYSTEM = (
    "You write the résumé of the IDEAL candidate for a given job posting. "
    "Given the job below, output ONLY JSON: {\"ideal_resume\": \"<plain-text résumé>\"}. "
    "The résumé must describe the skills, years of experience, education and domain background "
    "of a person who clearly and obviously qualifies for THIS exact job. Plain prose, no markdown, "
    "no headers, ~120 words."
)


def _cosine01(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized cosine in [0,1] = (cos+1)/2 (ConFit-v2 DotProductMetric)."""
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float((float(np.dot(a, b) / (na * nb)) + 1.0) / 2.0)


def _hyre_ideal_vec(con, *, content_hash: str, jd_text: str, model, client) -> np.ndarray | None:
    """Return the bge-m3 dense vec of the ideal-résumé for this JD, cached per content_hash.
    Misses cost one qwen call + one embed; hits are free. None on any failure (graceful)."""
    row = con.execute(
        "SELECT ideal_vec FROM hyre_cache WHERE content_hash = ?", (content_hash,)
    ).fetchone()
    if row and row["ideal_vec"]:
        return _blob_to_vec(row["ideal_vec"])
    try:
        obj = client.chat_json(_HYRE_SYSTEM, jd_text[:4000])
        ideal = str(obj.get("ideal_resume") or "").strip()
        if not ideal:
            return None
        vec = _encode_batch(model, [ideal])["dense_vecs"][0]
        con.execute(
            "INSERT OR REPLACE INTO hyre_cache (content_hash, ideal_resume, ideal_vec, computed_at) "
            "VALUES (?, ?, ?, ?)",
            (content_hash, ideal, _vec_to_blob(vec), datetime.now(timezone.utc).isoformat()),
        )
        con.commit()
        return np.asarray(vec, dtype=np.float32)
    except Exception:
        return None


_LLMCOV_SYSTEM = (
    "You assess whether a CANDIDATE can DO a job. From the JOB, extract the 3-8 genuine MUST-HAVE "
    "requirements (core skills, tools, experience, qualifications needed to perform the role — "
    "IGNORE nice-to-haves, perks, and company boilerplate). For EACH, judge strictly from the "
    "CANDIDATE only: present | partial | missing. Be honest — a skill not evidenced is missing. "
    'Return ONLY JSON: {"requirements":[{"requirement": str, "verdict": "present|partial|missing"}], '
    '"missing_summary": "<short Russian phrase naming the key missing must-haves, or empty>"}'
)


def _llm_coverage(con, *, cache_key: str, jd_title: str, jd_desc: str, cv_text: str,
                  client) -> tuple[float, str, list[dict]]:
    """LLM per-requirement coverage in [0,1] = (present + 0.5*partial)/total, the missing-summary,
    AND the per-requirement list [{requirement, verdict}] (for the card's skill breakdown). Cached
    per (CV,JD). The strongest single fit signal in offline eval. (0.5, '', []) on failure."""
    row = con.execute("SELECT coverage, missing, requirements FROM llmcov_cache WHERE cache_key = ?",
                      (cache_key,)).fetchone()
    if row is not None and row["requirements"] is not None:
        return float(row["coverage"]), (row["missing"] or ""), json.loads(row["requirements"])
    cov, miss, reqs = 0.5, "", []
    try:
        user = f"CANDIDATE:\n{cv_text[:1500]}\n\nJOB:\n{jd_title}\n{(jd_desc or '')[:3000]}"
        obj = client.chat_json(_LLMCOV_SYSTEM, user)
        for r in (obj.get("requirements") or []):
            if isinstance(r, dict):
                v = str(r.get("verdict", "")).lower()
                if v in ("present", "partial", "missing"):
                    reqs.append({"requirement": str(r.get("requirement", ""))[:120], "verdict": v})
        verdicts = [r["verdict"] for r in reqs]
        if verdicts:
            cov = (verdicts.count("present") + 0.5 * verdicts.count("partial")) / len(verdicts)
        miss = str(obj.get("missing_summary") or "")
    except Exception:
        return 0.5, "", []   # neutral on failure (don't boost or gate)
    con.execute("INSERT OR REPLACE INTO llmcov_cache "
                "(cache_key, coverage, missing, requirements, computed_at) VALUES (?, ?, ?, ?, ?)",
                (cache_key, cov, miss, json.dumps(reqs, ensure_ascii=False),
                 datetime.now(timezone.utc).isoformat()))
    con.commit()
    return cov, miss, reqs


# bge-m3 sparse-lexical scores (compute_score['sparse']) for Alina's CV↔JD pairs sit ≈ [0.03, 0.45];
# divide by this to map ≈[0,1] (clamped) so the 'sparse' blend weight is comparable to HyRE's [0,1].
# Calibrated on the labeled set; the hybrid win is robust to the exact divisor (eval/hybrid_measure).
_SPARSE_NORM_SCALE = 0.45


def _blend_fit(*, xenc_full: float | None, llm_cov: float | None, fit_hyre: float | None = None,
               sparse_norm: float | None = None, weights: dict) -> float:
    """Headline fit = weighted blend of HyRE (ideal-résumé cosine), bge-m3 SPARSE-lexical (CV↔JD),
    cross-encoder relevance (xenc_full) and LLM per-requirement coverage (llm_cov), all in [0,1].
    Renormalized over the signals actually present (graceful when a vacancy lacks one).

    RE-TUNED on Alina's 37 REAL labels. HyRE is the best single signal (0.803); on 2026-06-15 PM the
    bge-m3 SPARSE hybrid (the deferred SOTA upgrade — bge-m3 is a native dense+sparse+ColBERT model)
    was measured and ADDED: `0.7·hyre + 0.3·sparse_norm` beats HyRE-only on ALL THREE metrics
    (pairwise 0.803→0.814, ndcg@10 0.539→0.584, spearman 0.408→0.427) AND adds a DETERMINISTIC signal
    (sparse-lexical has no qwen run-variance, unlike HyRE). ColBERT was measured and did NOT help.
    llm_cov stays out of the headline (card breakdown only). See eval/hybrid_measure.py for the sweep."""
    parts = [("xenc", xenc_full), ("llm_cov", llm_cov), ("hyre", fit_hyre), ("sparse", sparse_norm)]
    num = sum(float(weights.get(k, 0.0)) * v for k, v in parts if v is not None)
    den = sum(float(weights.get(k, 0.0)) for k, v in parts if v is not None)
    return float(num / den) if den > 1e-9 else 0.0


def fit_from_feature(feat: dict, weights: dict, *, sparse_scale: float = _SPARSE_NORM_SCALE) -> float:
    """Recompute the headline fit_score from a stored feature_json's components (fit_hyre / bgem3_sparse
    / xenc_full / llm_cov) under the given fit_weights. Single source of the blend math, shared by
    rerank_scored (compute+store), slate.build_slate (recompute live so a weight change takes effect
    WITHOUT a heavy model re-run), and the eval harness. bge-m3 sparse is normalized by sparse_scale
    (a fixed divisor, so fit stays per-vacancy/local — no set-relative min-max). A vacancy missing a
    component just drops it from the renormalized blend (sparse-less vacancies fall back to HyRE)."""
    def _g(k):
        v = feat.get(k)
        return float(v) if v is not None else None
    raw_sparse = _g("bgem3_sparse")
    sparse_norm = (min(1.0, raw_sparse / sparse_scale) if (raw_sparse is not None and sparse_scale)
                   else None)
    return _blend_fit(xenc_full=_g("xenc_full"), llm_cov=_g("llm_cov"),
                      fit_hyre=_g("fit_hyre"), sparse_norm=sparse_norm, weights=weights)


def bgem3_sparse_scores(model, cv_text: str, jd_texts: list[str]) -> list[float]:
    """bge-m3 SPARSE-lexical relevance (CV ↔ each JD) via the native compute_score (the canonical
    bge-m3 hybrid component; FlagEmbedding .../encoder_only/m3.py:686-699). Returns one score per JD,
    parallel to jd_texts. Shared by rerank_scored (forward path) + the backfill. Needs a loaded
    BGEM3FlagModel; do NOT call without a coordinated, foreground model load (memory-heavy)."""
    if not jd_texts:
        return []
    pairs = [[cv_text[:3000], (jd or "")[:4000]] for jd in jd_texts]
    out = model.compute_score(pairs, weights_for_different_modes=[0, 1, 0],
                              max_query_length=512, max_passage_length=512)
    sp = out["sparse"]
    return [float(x) for x in (sp if hasattr(sp, "__len__") else [sp])]


# ---------------------------------------------------------------------------
# Embedding cache helpers
# ---------------------------------------------------------------------------

def _content_hash(title: str, description: str) -> str:
    return hashlib.sha1(f"{title}|||{description}|||bge-m3".encode()).hexdigest()[:20]


def _vec_to_blob(vec: np.ndarray) -> bytes:
    arr = np.asarray(vec, dtype=np.float32).ravel()
    return struct.pack(f"{len(arr)}f", *arr)


def _blob_to_vec(blob: bytes) -> np.ndarray:
    n = len(blob) // 4
    return np.array(struct.unpack(f"{n}f", blob), dtype=np.float32)


def _encode_batch(model, texts: list[str]) -> dict[str, Any]:
    """Encode a batch of texts; returns dense_vecs + lexical_weights (ColBERT lazy)."""
    out = model.encode(
        texts, batch_size=32, max_length=512,
        return_dense=True, return_sparse=True, return_colbert_vecs=False,
    )
    return {
        "dense_vecs": out["dense_vecs"],         # (N, 1024) f32 L2-normed
        "lexical_weights": out["lexical_weights"],  # list[dict tokid→w]
    }


def _encode_colbert_pair(model, query: str, doc: str) -> float:
    """Compute asymmetric ColBERT MaxSim score for a single (query, doc) pair.

    ColBERT score is asymmetric: normalized by query token count → coverage signal.
    Returns float in [0, 1] approximately.
    """
    try:
        from FlagEmbedding import BGEM3FlagModel  # type: ignore
        out = model.encode([query, doc], return_dense=False, return_sparse=False,
                           return_colbert_vecs=True)
        q_vecs = out["colbert_vecs"][0]   # (q_toks, 1024)
        d_vecs = out["colbert_vecs"][1]   # (d_toks, 1024)
        # MaxSim: for each query token, max cosine to any doc token
        sims = np.matmul(q_vecs, d_vecs.T)  # (q_toks, d_toks)
        return float(sims.max(axis=1).mean())
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# PositiveLibrary: matrix of positively-labeled vacancy embeddings
# ---------------------------------------------------------------------------

class PositiveLibrary:
    """Dense embedding matrix of vacancies the candidate liked / applied to.

    Cold start (n_rows == 0) → all taste features = 0.0 (no false positives).
    """

    def __init__(self, matrix: np.ndarray, vacancy_ids: list[int],
                 company_tokens: list[set[str]]):
        self.matrix = matrix              # (N, 1024) L2-normalized
        self.vacancy_ids = vacancy_ids    # parallel to matrix rows
        self.company_tokens = company_tokens
        self.n_rows = len(vacancy_ids)

    @classmethod
    def empty(cls) -> "PositiveLibrary":
        return cls(np.zeros((0, EMBEDDING_DIM), dtype=np.float32), [], [])

    @classmethod
    def build(cls, con) -> "PositiveLibrary":
        """Build from positively-labeled vacancies in the DB."""
        _ensure_schema(con)
        rows = con.execute(
            """SELECT ve.vacancy_id, ve.dense, v.company
               FROM vacancy_embedding ve
               JOIN vacancy v ON v.id = ve.vacancy_id
               JOIN label l   ON l.vacancy_id = ve.vacancy_id
               WHERE l.score_1_5 >= 4 OR l.applied = 1 OR l.interview = 1"""
        ).fetchall()
        if not rows:
            return cls.empty()

        vecs, ids, tokens = [], [], []
        for r in rows:
            vecs.append(_blob_to_vec(r["dense"]))
            ids.append(int(r["vacancy_id"]))
            comp_norm = normalize_company(r["company"] or "")
            tokens.append({t for t in comp_norm.split() if len(t) > 2})

        matrix = np.stack(vecs, axis=0).astype(np.float32)
        # L2-normalize
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms < 1e-9, 1.0, norms)
        matrix = matrix / norms
        return cls(matrix, ids, tokens)

    def taste_features(self, vec: np.ndarray, *, company: str | None = None,
                       exclude_vacancy_id: int | None = None) -> dict[str, float]:
        """Compute 5 taste features (nearest, centroid_cos, recent_cos, drift, company_overlap).

        Cold start → all 0.0. leave-one-out via exclude_vacancy_id.
        """
        if self.n_rows == 0:
            return {k: 0.0 for k in ("nearest_liked_cosine", "positive_centroid_cosine",
                                      "recent_centroid_cosine", "topic_drift", "company_overlap_count")}

        vec_f = np.asarray(vec, dtype=np.float32).ravel()
        norm = np.linalg.norm(vec_f)
        if norm < 1e-9:
            return {k: 0.0 for k in ("nearest_liked_cosine", "positive_centroid_cosine",
                                      "recent_centroid_cosine", "topic_drift", "company_overlap_count")}
        vec_f = vec_f / norm

        # build mask: exclude self (leave-one-out) to prevent cosine≈1 self-match
        mask = np.ones(self.n_rows, dtype=bool)
        if exclude_vacancy_id is not None and exclude_vacancy_id in self.vacancy_ids:
            idx = self.vacancy_ids.index(exclude_vacancy_id)
            mask[idx] = False

        mat = self.matrix[mask]
        if len(mat) == 0:
            return {k: 0.0 for k in ("nearest_liked_cosine", "positive_centroid_cosine",
                                      "recent_centroid_cosine", "topic_drift", "company_overlap_count")}

        cosines = mat @ vec_f                           # (N,)
        nearest = float(cosines.max())

        centroid = mat.mean(axis=0)
        cn = np.linalg.norm(centroid)
        centroid_cos = float(centroid @ vec_f / cn) if cn > 1e-9 else 0.0

        # recent = last 20% of rows (chronologically ordered by build-time row order)
        recent_n = max(1, len(mat) // 5)
        recent_centroid = mat[-recent_n:].mean(axis=0)
        rcn = np.linalg.norm(recent_centroid)
        recent_cos = float(recent_centroid @ vec_f / rcn) if rcn > 1e-9 else 0.0

        # topic drift: distance between recent and full centroid (library diversity)
        if cn > 1e-9 and rcn > 1e-9:
            drift = float(1.0 - (centroid / cn) @ (recent_centroid / rcn))
        else:
            drift = 0.0

        # company overlap: fraction of P-library rows sharing company tokens
        comp_toks = {t for t in normalize_company(company or "").split() if len(t) > 2}
        if comp_toks:
            masked_tokens = [self.company_tokens[i] for i in range(self.n_rows) if mask[i]]
            company_overlap = sum(1 for t in masked_tokens if t & comp_toks)
        else:
            company_overlap = 0

        return {
            "nearest_liked_cosine":    nearest,
            "positive_centroid_cosine": centroid_cos,
            "recent_centroid_cosine":  recent_cos,
            "topic_drift":             drift,
            "company_overlap_count":   float(company_overlap),
        }


# ---------------------------------------------------------------------------
# Core: extract features for DESCRIBED vacancies
# ---------------------------------------------------------------------------

def extract_features(cfg: dict, con, *, limit: int | None = None) -> dict[str, int]:
    """Embed DESCRIBED vacancies + compute aspect features.

    For each vacancy without a fresh vacancy_feature row:
      1. Load JD text, segment into sections (aspects.segment_jd)
      2. Embed whole-JD + section texts with bge-m3 (dense+sparse cached)
      3. Load candidate aspect_vecs + PositiveLibrary
      4. Call aspects.score → named feature dict
      5. Write vacancy_embedding + vacancy_feature

    Returns {"featured": n, "cached": n}.
    """
    _ensure_schema(con)

    model_name = cfg.get("features", {}).get("model", "BAAI/bge-m3")

    # Check model availability; degrade gracefully
    try:
        model = _load_model(model_name)
    except Exception:
        return {"featured": 0, "cached": 0, "error": "model_unavailable"}

    # Load candidate aspect vecs
    cand = load_candidate(con)
    cand_texts = _cand_aspect_texts(con) if cand else None
    cand_vecs = _load_candidate_vecs(model, cand_texts, con) if cand_texts else None
    cand_skills = (cand.get("skills") or []) if cand else []
    cand_seniority = (cand.get("seniority") or "senior") if cand else "senior"

    # CV lexical weights (bge-m3 token-id space), computed once — used by _sparse_overlap so
    # JD must-have keyword coverage is measured in the SAME id space (the old version compared
    # token-ids against word strings and was always 0).
    cand_sparse: dict[str, float] = {}
    if cand_skills:
        try:
            _cv_enc = _encode_batch(model, [" ".join(cand_skills)])
            _lw = _cv_enc.get("lexical_weights")
            if _lw:
                cand_sparse = {str(k): float(v) for k, v in _lw[0].items()}
        except Exception:
            cand_sparse = {}

    library = PositiveLibrary.build(con)

    # Vacancies needing features (DESCRIBED, no vacancy_feature row yet)
    sql = """SELECT v.id, v.title, v.company, v.city, v.description, v.is_remote_hint,
                    v.date_posted, v.first_seen,
                    ve.dense, ve.aspects_json, ve.content_hash
             FROM vacancy v
             LEFT JOIN vacancy_embedding ve ON ve.vacancy_id = v.id
             LEFT JOIN vacancy_feature   vf ON vf.vacancy_id = v.id
             WHERE v.status = ? AND vf.vacancy_id IS NULL"""
    args = [Status.DESCRIBED.value]
    if limit:
        sql += " LIMIT ?"
        args.append(limit)
    rows = con.execute(sql, args).fetchall()

    n_featured = n_cached = 0

    # Texts to embed: collect misses, then batch-encode
    to_embed: list[tuple[int, str, str, str, dict]] = []   # (id, title, desc, ch, sections)
    for row in rows:
        desc = row["description"] or ""
        ch = _content_hash(row["title"] or "", desc)
        sections = _asp.segment_jd(row["title"] or "", desc)
        if row["dense"] and row["content_hash"] == ch:
            # Cache hit: embedding already stored
            to_embed.append((row["id"], row["title"], desc, ch, sections, True))
        else:
            to_embed.append((row["id"], row["title"], desc, ch, sections, False))

    # Batch-encode cache misses
    miss_indices = [i for i, t in enumerate(to_embed) if not t[5]]
    if miss_indices:
        miss_texts = []
        miss_section_texts: list[list[str]] = []
        for i in miss_indices:
            vid, title, desc, ch, sections, _ = to_embed[i]
            miss_texts.append(desc[:8000])
            sec_texts = [sections.get(k, "") for k in
                         ("must_have", "responsibilities", "company", "nice_to_have")]
            miss_section_texts.append([t[:2000] for t in sec_texts])

        # Flatten: [desc0, sec0_0, sec0_1, sec0_2, sec0_3, desc1, ...]
        flat_texts: list[str] = []
        for i, idx in enumerate(miss_indices):
            flat_texts.append(miss_texts[i])
            flat_texts.extend(miss_section_texts[i])

        enc = _encode_batch(model, flat_texts)
        dense_all = enc["dense_vecs"]   # (len(flat_texts), 1024)
        sparse_all = enc["lexical_weights"]

        # Store back
        per_vacancy = {}
        stride = 5  # 1 full-JD + 4 sections
        for k, i in enumerate(miss_indices):
            vid = to_embed[i][0]
            base = k * stride
            jd_dense = dense_all[base]
            sec_dense = {
                "must_have":       _encode_b64(dense_all[base + 1]),
                "responsibilities": _encode_b64(dense_all[base + 2]),
                "company":          _encode_b64(dense_all[base + 3]),
                "nice_to_have":     _encode_b64(dense_all[base + 4]),
            }
            # Convert token-id keys to str and weights to float for JSON serialization
            sec_sparse = {
                "must_have": {str(k): float(v) for k, v in sparse_all[base + 1].items()},
            }
            per_vacancy[i] = (jd_dense, sec_dense, sec_sparse)

    now = datetime.now(timezone.utc).isoformat()

    for idx, entry in enumerate(to_embed):
        if len(entry) == 6:
            vid, title, desc, ch, sections, is_hit = entry
        else:
            continue
        row = rows[idx]

        if is_hit:
            n_cached += 1
            jd_dense = _blob_to_vec(row["dense"])
            aspects_j = json.loads(row["aspects_json"]) if row["aspects_json"] else {}
            sec_dense = {k: _decode_b64(v) for k, v in aspects_j.items() if k != "sparse"}
            sec_sparse = aspects_j.get("sparse", {})
        else:
            if idx not in per_vacancy:
                continue
            jd_dense, sec_dense_b64, sec_sparse = per_vacancy[idx]

            # Persist embedding
            aspects_j = {**sec_dense_b64, "sparse": sec_sparse}
            con.execute(
                """INSERT OR REPLACE INTO vacancy_embedding
                   (vacancy_id, content_hash, dense, aspects_json, model_version, computed_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (vid, ch, _vec_to_blob(jd_dense),
                 json.dumps(aspects_j, ensure_ascii=False),
                 model_name, now),
            )
            sec_dense = {k: _decode_b64(v) for k, v in sec_dense_b64.items()}

        # Compute aspect features
        if cand_vecs is None:
            feat = _zero_features()
        else:
            jd_section_vecs = {k: sec_dense.get(k) for k in
                               ("must_have", "responsibilities", "company", "nice_to_have")}
            # sparse keyword coverage: JD must-have lexical weights ∩ CV lexical weights, both
            # in bge-m3 token-id space (cand_sparse computed once per process below).
            sparse_score = _sparse_overlap(
                sec_sparse.get("must_have", {}),
                cand_sparse,
            )
            # Real gate signals (were hardcoded to neutral constants → location/language never
            # influenced the score, which Alina's hybrid/Heidelberg-Frankfurt requirement needs).
            _in_radius, _dist_km = _geo.geo_check(row["city"], cfg)
            geo_norm = 0.5 if _dist_km is None else min(1.0, float(_dist_km) / 100.0)
            gates = {
                "is_remote_hint": float(row["is_remote_hint"] or 0),
                "lang_de_required": 1.0 if _hard.german_required(row["description"] or "") else 0.0,
                "geo_distance_norm": geo_norm,
                # real posting age (lower = fresher; triage constraint −1). Was hardcoded 1.0.
                "recency_days": _recency_days(row["date_posted"], row["first_seen"]),
                "requirements_verified": 0.0,
                "company_known": 0.0,
                "salary_vs_target_gap": 0.0,
            }
            feat = _asp.score(
                cand_vecs=cand_vecs,
                cand_skills=cand_skills,
                cand_seniority=cand_seniority,
                jd_sections=sections,
                jd_section_vecs=jd_section_vecs,
                jd_full_vec=jd_dense,
                sparse_score=sparse_score,
                library=library,
                gates=gates,
                vacancy_id=vid,
                company=row["company"],
            )

        con.execute(
            """INSERT OR REPLACE INTO vacancy_feature
               (vacancy_id, match_score, feature_json, computed_at)
               VALUES (?, ?, ?, ?)""",
            (vid, feat.get("match_score"), json.dumps(feat, ensure_ascii=False), now),
        )
        n_featured += 1

    con.commit()
    return {"featured": n_featured, "cached": n_cached}


def rerank_scored(cfg: dict, con, *, top_k: int | None = None) -> dict:
    """Cross-encoder re-rank top-K SCORED vacancies with bge-reranker-v2-m3.

    Computes xenc_full (CV × full JD) and xenc_musthave (CV × must-have section).
    Stores both into vacancy_feature.feature_json for the slate sort blend.
    Degrades gracefully: reranker unavailable → {"skipped": reason}.
    """
    from .candidate import candidate_doc as _cand_doc

    _ensure_schema(con)
    reranker_name = cfg.get("features", {}).get("reranker", "BAAI/bge-reranker-v2-m3")
    k = top_k if top_k is not None else int(cfg.get("features", {}).get("rerank_top_k", 30))

    cv_text = _cand_doc(con)
    if not cv_text:
        return {"reranked": 0, "skipped": "no_candidate"}

    try:
        reranker = _load_reranker(reranker_name)
    except Exception:
        return {"reranked": 0, "skipped": "reranker_unavailable"}

    # Score the FULL slate-candidate pool (SCORED + SLATED). The slate re-admits unviewed SLATED
    # jobs (UC 9a); if they aren't scored here they'd get fit=0 and be wrongly de-ranked off the
    # slate for missing data rather than genuine low fit.
    rows = con.execute(
        """SELECT v.id, v.title, v.description
           FROM vacancy v
           JOIN (SELECT vacancy_id, MAX(id) mid FROM judge_score GROUP BY vacancy_id) m
             ON m.vacancy_id = v.id
           JOIN judge_score js ON js.id = m.mid
           WHERE v.status IN (?, ?)
           ORDER BY js.score DESC
           LIMIT ?""",
        (Status.SCORED.value, Status.SLATED.value, k),
    ).fetchall()

    if not rows:
        return {"reranked": 0}

    cv = cv_text[:3000]
    pairs_full: list[list[str]] = []
    pairs_mh: list[list[str]] = []
    for r in rows:
        desc = (r["description"] or "")[:4000]
        sections = _asp.segment_jd(r["title"] or "", r["description"] or "")
        mh = sections.get("must_have", sections.get("full", desc))[:2000]
        pairs_full.append([cv, desc])
        pairs_mh.append([cv, mh])

    try:
        scores_full = reranker.compute_score(pairs_full, normalize=True)
        scores_mh = reranker.compute_score(pairs_mh, normalize=True)
    except Exception as exc:
        return {"reranked": 0, "skipped": f"compute_score_error: {exc}"}

    # Handle single-item case where reranker returns a scalar
    if not hasattr(scores_full, "__len__"):
        scores_full = [scores_full]
        scores_mh = [scores_mh]

    f_cfg = cfg.get("features", {})
    fit_w = f_cfg.get("fit_weights", {"xenc": 0.5, "llm_cov": 0.5})
    sparse_norm_scale = float(f_cfg.get("sparse_norm", _SPARSE_NORM_SCALE))
    warn_t = float(cfg.get("slate", {}).get("fit_warn_threshold", 0.4))
    hyre_on = bool(f_cfg.get("hyre", True))
    llmcov_on = bool(f_cfg.get("llm_cov", True))
    elig_on = bool(f_cfg.get("eligibility", True))
    e_cfg = cfg.get("eligibility", {})
    elig_floor = float(e_cfg.get("floor", 0.35))
    elig_mid = float(e_cfg.get("mid", 0.6))
    elig_soft_lift = float(e_cfg.get("soft_lift_threshold", 0.55))
    cand_quals = _elig.candidate_quals(load_candidate(con))   # education/years/langs (once)

    # One qwen client shared by HyRE, LLM-coverage, and eligibility.
    qwen = None
    if hyre_on or elig_on or llmcov_on:
        try:
            from .llm import OllamaClient
            _llm = cfg.get("llm", {})
            qwen = OllamaClient(model=_llm.get("judge_model", "qwen3:8b"),
                                num_ctx=int(_llm.get("num_ctx", 8192)), temperature=0)
        except Exception:
            qwen = None
    # bge-m3 is needed for HyRE (dense ideal-résumé cosine) AND the SOTA SPARSE-lexical hybrid signal.
    # Sparse does NOT need qwen, so load the model whenever HyRE or sparse is on.
    sparse_on = bool(f_cfg.get("hybrid_sparse", True))
    cv_vec = bge_model = None
    if (hyre_on and qwen is not None) or sparse_on:
        try:
            bge_model = _load_model(f_cfg.get("model", "BAAI/bge-m3"))
        except Exception:
            bge_model = None
    if hyre_on and qwen is not None and bge_model is not None:
        try:
            cv_vec = _encode_batch(bge_model, [cv])["dense_vecs"][0]
        except Exception:
            cv_vec = None
    hyre_model = bge_model   # HyRE reuses the same loaded model

    # bge-m3 SPARSE-lexical (CV ↔ JD), batched once — the deferred SOTA hybrid signal (measured to
    # beat HyRE-only on all 3 metrics when fused). Reuses the model already loaded above; no qwen.
    sparse_by_vid: dict[int, float] = {}
    if sparse_on and bge_model is not None:
        try:
            jds = [f"{r['title'] or ''}\n{r['description'] or ''}" for r in rows]
            sp = bgem3_sparse_scores(bge_model, cv, jds)
            sparse_by_vid = {rows[i]["id"]: sp[i] for i in range(min(len(rows), len(sp)))}
        except Exception:
            sparse_by_vid = {}

    cv_hash = _content_hash("cv", cv)   # part of the llm_cov cache key (CV changes → recompute)
    now = datetime.now(timezone.utc).isoformat()
    n = n_hyre = n_elig = n_llm = 0
    for i, row in enumerate(rows):
        vid = row["id"]
        feat = feature_row(con, vid) or {}
        feat["xenc_full"] = float(scores_full[i])
        feat["xenc_musthave"] = float(scores_mh[i])
        ch = _content_hash(row["title"] or "", row["description"] or "")
        jd_text = f"{row['title'] or ''}\n{row['description'] or ''}"

        # HyRE kept as a stored feature (for the future trained gate); NOT in the headline fit blend.
        if cv_vec is not None and qwen is not None:
            ivec = _hyre_ideal_vec(con, content_hash=ch, jd_text=jd_text,
                                   model=hyre_model, client=qwen)
            if ivec is not None:
                feat["fit_hyre"] = _cosine01(cv_vec, ivec)
                n_hyre += 1

        # LLM per-requirement coverage — the validated co-headline fit signal.
        llm_cov = None
        miss = ""
        if llmcov_on and qwen is not None:
            llm_cov, miss, llm_reqs = _llm_coverage(con, cache_key=f"{cv_hash}:{ch}",
                                                    jd_title=row["title"] or "",
                                                    jd_desc=row["description"] or "",
                                                    cv_text=cv, client=qwen)
            feat["llm_cov"] = llm_cov
            feat["llm_cov_missing"] = miss
            feat["llm_cov_reqs"] = llm_reqs   # [{requirement, verdict}] → card skill breakdown
            n_llm += 1

        # bge-m3 SPARSE-lexical (the SOTA hybrid signal) — stored raw; normalized + blended below.
        if vid in sparse_by_vid:
            feat["bgem3_sparse"] = sparse_by_vid[vid]

        # headline SKILL fit = HyRE + bge-m3 sparse hybrid (real-label tuned). fit_from_feature is the
        # single blend source (normalizes sparse by _SPARSE_NORM_SCALE, renormalizes over present).
        fit = fit_from_feature(feat, fit_w, sparse_scale=sparse_norm_scale)
        feat["fit_score"] = fit
        # prefer the interpretable missing-must-haves note; fall back to a generic gap note
        feat["fit_note"] = (miss or "крупный разрыв по навыкам") if fit < warn_t else ""

        # ELIGIBILITY gate (hard qualifications: education/PhD-position/credentials/language).
        # Separate from skill fit — catches "great skills but not allowed to apply" (e.g. a PhD
        # role needing a Master's she lacks). Down-rank via elig_score, never drop.
        if elig_on and qwen is not None:
            req = _elig.extract_requirements(con, content_hash=ch, jd_text=jd_text, client=qwen)
            eg, en, sev = _elig.eligibility_gate(req, cand_quals, floor=elig_floor, mid=elig_mid,
                                                 fit_score=fit, soft_lift_threshold=elig_soft_lift)
            feat["elig_score"] = eg
            feat["elig_note"] = en
            feat["elig_severity"] = sev   # "structural" (red ⛔) | "soft" (amber, never sinks high-fit)
            if eg < 1.0:
                n_elig += 1

        con.execute(
            """INSERT OR REPLACE INTO vacancy_feature
               (vacancy_id, match_score, feature_json, computed_at)
               VALUES (?, ?, ?, ?)""",
            (vid, feat.get("match_score"), json.dumps(feat, ensure_ascii=False), now),
        )
        n += 1

    con.commit()
    return {"reranked": n, "hyre": n_hyre, "llm_cov": n_llm, "elig_flagged": n_elig}


# ---------------------------------------------------------------------------
# Public read helpers
# ---------------------------------------------------------------------------

def feature_row(con, vacancy_id: int) -> dict | None:
    """Return named feature dict for a vacancy (from vacancy_feature)."""
    _ensure_schema(con)
    row = con.execute(
        "SELECT feature_json FROM vacancy_feature WHERE vacancy_id = ?", (vacancy_id,)
    ).fetchone()
    return json.loads(row["feature_json"]) if row else None


def recompute_live(con, vacancy_id: int, cfg: dict, *, cand_quals: dict | None = None) -> dict:
    """Recompute the headline fit_score AND the eligibility gate LIVE from stored caches (no model
    load): fit from the stored xenc/hyre/llm_cov components under the current fit_weights, eligibility
    from the cached requirement record under the current gate logic (Master-Data guard + high-fit
    soft lift). This makes a fit-weight re-tune and an eligibility-logic fix take effect on already-
    stored data WITHOUT the memory-heavy rerank re-run; the next coordinated tick persists the same
    values. Returns {fit_score, fit_note, elig_score, elig_note, elig_severity}; falls back to the
    persisted values when a cache is missing.

    cand_quals (eligibility.candidate_quals(load_candidate(con))) is computed once by the caller and
    passed in to avoid re-loading the CV per card."""
    feat = feature_row(con, vacancy_id) or {}
    f_cfg = cfg.get("features", {})
    fit_w = f_cfg.get("fit_weights", {"hyre": 0.5, "xenc": 0.5})
    warn_t = float(cfg.get("slate", {}).get("fit_warn_threshold", 0.4))
    fit = fit_from_feature(feat, fit_w, sparse_scale=float(f_cfg.get("sparse_norm", _SPARSE_NORM_SCALE)))
    miss = feat.get("llm_cov_missing") or ""
    out = {
        "fit_score": fit,
        "fit_note": (miss or "крупный разрыв по навыкам") if fit < warn_t else "",
        "elig_score": float(feat["elig_score"]) if feat.get("elig_score") is not None else 1.0,
        "elig_note": feat.get("elig_note") or "",
        "elig_severity": feat.get("elig_severity") or "structural",
    }
    if not bool(f_cfg.get("eligibility", True)) or cand_quals is None:
        return out
    row = con.execute("SELECT title, description FROM vacancy WHERE id = ?", (vacancy_id,)).fetchone()
    if not row:
        return out
    ch = _content_hash(row["title"] or "", row["description"] or "")
    jd_text = f"{row['title'] or ''}\n{row['description'] or ''}"
    req = _elig.req_from_cache(con, content_hash=ch, jd_text=jd_text)
    if req is None:
        return out
    e_cfg = cfg.get("eligibility", {})
    eg, en, sev = _elig.eligibility_gate(
        req, cand_quals, floor=float(e_cfg.get("floor", 0.35)), mid=float(e_cfg.get("mid", 0.6)),
        fit_score=fit, soft_lift_threshold=float(e_cfg.get("soft_lift_threshold", 0.55)))
    out["elig_score"], out["elig_note"], out["elig_severity"] = eg, en, sev
    return out


def feature_vector(con, vacancy_id: int) -> np.ndarray | None:
    """Return concatenated [dense_1024 ++ named_extras] feature vector for LGBM."""
    _ensure_schema(con)
    emb_row = con.execute(
        "SELECT dense FROM vacancy_embedding WHERE vacancy_id = ?", (vacancy_id,)
    ).fetchone()
    feat = feature_row(con, vacancy_id)
    if feat is None:
        return None

    named = np.array([feat.get(k, 0.0) for k in _asp.FEATURE_NAMES], dtype=np.float32)

    # CONSISTENT shape: always [dense(1024) ++ named]. A vacancy without a bge-m3 embedding gets a
    # ZERO dense block (not a short named-only vector) so train (triage._load_labeled) and score share
    # the same feature width — a model trained on 1053 dims can't score a 29-dim vector.
    dense = _blob_to_vec(emb_row["dense"]) if emb_row else np.zeros(EMBEDDING_DIM, dtype=np.float32)
    return np.concatenate([dense, named], axis=0)


def match_summary(con, vacancy_id: int) -> dict | None:
    """Return interpretable match summary (coverage + missing skills) for the slate."""
    feat = feature_row(con, vacancy_id)
    if feat is None:
        return None
    return {
        "match_score":              feat.get("match_score"),
        "cov_musthave_maxsim":     feat.get("cov_musthave_maxsim"),
        "n_musthave_missing":      feat.get("n_musthave_missing"),
        "sim_experience_resp":     feat.get("sim_experience_responsibilities"),
        "seniority_gap":           feat.get("seniority_gap"),
        "nearest_liked_cosine":    feat.get("nearest_liked_cosine"),
    }


def scores_by_vacancy(con) -> dict[int, float]:
    """Return {vacancy_id: match_score} for all vacancies with a feature row."""
    _ensure_schema(con)
    rows = con.execute("SELECT vacancy_id, match_score FROM vacancy_feature").fetchall()
    return {int(r["vacancy_id"]): float(r["match_score"] or 0.0) for r in rows}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _load_candidate_vecs(model, cand_texts: dict[str, str], con) -> dict[str, np.ndarray] | None:
    """Load or compute bge-m3 embeddings for each candidate aspect."""
    from .candidate import _ensure_schema as _cand_schema
    _cand_schema(con)
    row = con.execute(
        "SELECT id, doc_hash, aspect_vecs FROM candidate_profile ORDER BY id DESC LIMIT 1"
    ).fetchone()
    # Cache invalidation: only reuse stored vecs if they were computed for the CURRENT aspect
    # texts. Without this, editing the profile silently reused stale embeddings.
    cur_hash = hashlib.sha256(
        json.dumps(cand_texts, sort_keys=True).encode()).hexdigest()[:16]
    if row and row["aspect_vecs"] and row["doc_hash"] == cur_hash:
        blob = row["aspect_vecs"]
        n_aspects = 5
        chunk = EMBEDDING_DIM * 4  # float32
        if len(blob) == n_aspects * chunk:
            keys = ("skills", "experience", "domains", "roles", "full")
            return {k: _blob_to_vec(blob[i*chunk:(i+1)*chunk]) for i, k in enumerate(keys)}

    # Compute and cache
    texts = [
        cand_texts.get("skills_text", ""),
        cand_texts.get("experience_text", ""),
        cand_texts.get("domains_text", ""),
        cand_texts.get("roles_text", ""),
        cand_texts.get("full_doc", ""),
    ]
    try:
        enc = _encode_batch(model, texts)
        dense = enc["dense_vecs"]  # (5, 1024)
        blob = b"".join(_vec_to_blob(dense[i]) for i in range(5))
        if row:
            con.execute("UPDATE candidate_profile SET aspect_vecs = ?, doc_hash = ? WHERE id = ?",
                        (blob, cur_hash, row["id"]))
            con.commit()
        keys = ("skills", "experience", "domains", "roles", "full")
        return {k: np.array(dense[i], dtype=np.float32) for i, k in enumerate(keys)}
    except Exception:
        return None


def _encode_b64(vec: np.ndarray) -> str:
    return base64.b64encode(_vec_to_blob(vec)).decode()


def _decode_b64(s: str) -> np.ndarray | None:
    if not s:
        return None
    try:
        return _blob_to_vec(base64.b64decode(s))
    except Exception:
        return None


def _sparse_overlap(jd_sparse: dict, cv_sparse: dict) -> float:
    """Lexical overlap in bge-m3 token-id space: fraction of JD must-have token WEIGHT whose
    token-id also appears in the CV's lexical weights. Both args are {token_id_str: weight}.

    (The previous version compared bge-m3 token-IDs against CV word strings and therefore always
    returned 0.0 — a permanently dead feature.)"""
    if not jd_sparse or not cv_sparse:
        return 0.0
    cv_ids = {str(k) for k in cv_sparse}
    total_w = sum(float(w) for w in jd_sparse.values())
    if total_w < 1e-9:
        return 0.0
    covered = sum(float(w) for tid, w in jd_sparse.items() if str(tid) in cv_ids)
    return float(min(1.0, covered / total_w))


def _recency_days(*candidates: str | None) -> float:
    """Days since the job was POSTED (lower = fresher; triage monotone constraint is −1). Tries
    each candidate (date_posted, then first_seen) and returns days-ago; neutral 7.0 if none parse."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    for val in candidates:
        if not val:
            continue
        try:
            dt = datetime.fromisoformat(str(val))   # handles 'YYYY-MM-DD' and full ISO timestamps
        except ValueError:
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return float(max(0, (now - dt).days))
    return 7.0


def _zero_features() -> dict[str, float]:
    return {k: 0.0 for k in _asp.FEATURE_NAMES + ["match_score"]}
