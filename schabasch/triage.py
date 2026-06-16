"""Cheap ML triage gate: score every DESCRIBED vacancy → bucket → drop bottom before LLM.

Transplanted from zotero-summarizer's production triage (classifier_fit / band_calibration /
classifier_training / _gate). Adapted for job-search context.

Cold start (< min_labels_to_train labels):
  calibrated_score = features.match_score (coverage combo)  — no hard-drop, rank only.
Trained (≥ min_labels_to_train labels):
  LightGBM regressor → isotonic band-calibration → bucket → hard-drop bottom.

All new state lives in the `triage_decision` sidecar table (self-created).
normalize.py calls select_for_normalize() instead of db.by_status().
slate.py blends calibrated_score/5 into the sort key.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from . import db, features as _feat
from .aspects import FEATURE_NAMES
from .models import Status

_SCHEMA = """
CREATE TABLE IF NOT EXISTS triage_decision (
    vacancy_id       INTEGER PRIMARY KEY,
    raw_score        REAL NOT NULL,
    calibrated_score REAL NOT NULL,
    priority         TEXT NOT NULL,
    model_version    TEXT NOT NULL,
    decided_at       TEXT NOT NULL
)
"""

# Named feature monotone constraints (+1 = higher is better, -1 = lower is better, 0 = free)
_NAMED_CONSTRAINTS: dict[str, int] = {
    "cov_musthave_maxsim":           1,
    "cov_musthave_frac_above_tau":   1,
    "n_musthave_missing":           -1,
    "cov_nicetohave_maxsim":         1,
    "colbert_req_to_cv":             1,
    "sparse_req_vs_cv":              1,
    "sim_skills_requirements":       1,
    "sim_experience_responsibilities": 1,
    "sim_domain_company":            1,
    "sim_title_role":                1,
    "sim_fullcv_fulljd":             1,
    "seniority_gap":                 0,   # not monotone (over/under both bad)
    "nearest_liked_cosine":          1,
    "positive_centroid_cosine":      1,
    "recent_centroid_cosine":        1,
    "topic_drift":                   0,
    "company_overlap_count":         0,
    "lang_de_required":             -1,
    "geo_distance_norm":            -1,
    "is_remote_hint":               -1,
    "recency_days":                 -1,
    "has_structure":                 1,
    "title_log_len":                 0,
    "desc_log_len":                  1,
    "requirements_verified":         1,
    "company_known":                 1,
    "salary_vs_target_gap":         -1,
    "fit_hyre":                      1,   # genuine fit (CV vs ideal résumé) — higher is better
    "fit_score":                     1,   # blended cross-encoder + HyRE + coverage
}


def _ensure_schema(con) -> None:
    con.execute(_SCHEMA)
    con.commit()


# ---------------------------------------------------------------------------
# Bucketing
# ---------------------------------------------------------------------------

def _score_to_priority(score: float, cutoffs: dict[str, float]) -> str:
    if score >= cutoffs.get("must", 4.5):
        return "must"
    if score >= cutoffs.get("should", 3.5):
        return "should"
    if score >= cutoffs.get("could", 2.0):
        return "could"
    return "drop"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _build_target(score_1_5, applied: int) -> float | None:
    """Derive continuous y ∈ [1,5] from label table row."""
    if applied:
        return 5.0
    if score_1_5 is not None:
        return float(score_1_5)
    return None


def _build_weight(score_1_5, applied: int, label_source: str) -> float:
    """Sample weight ∈ [0.2, 1.0]."""
    if applied or score_1_5 == 5:
        return 1.0
    if label_source == "slate":
        return 0.7
    return 0.5


def _labels_hash(rows: list) -> str:
    """SHA256 of sorted (vacancy_id, score_1_5, applied) — retrain-on-drift key."""
    data = sorted((r["vacancy_id"], r["score_1_5"], r["applied"]) for r in rows)
    return hashlib.sha256(json.dumps(data).encode()).hexdigest()[:16]


def _load_labeled(con) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    """Return (X, y, weights, group_keys, vacancy_ids) for all labeled+featured vacancies."""
    _feat._ensure_schema(con)   # ensure vacancy_feature/embedding exist (train on a fresh DB)
    rows = con.execute(
        """SELECT l.vacancy_id, l.score_1_5, l.applied, l.source as label_source,
                  v.dedup_key, vf.feature_json, vf.match_score,
                  ve.dense
           FROM label l
           JOIN vacancy v       ON v.id = l.vacancy_id
           JOIN vacancy_feature vf ON vf.vacancy_id = l.vacancy_id
           LEFT JOIN vacancy_embedding ve ON ve.vacancy_id = l.vacancy_id
           WHERE l.score_1_5 IS NOT NULL OR l.applied = 1"""
    ).fetchall()

    Xs, ys, ws, groups, vids = [], [], [], [], []
    for r in rows:
        y = _build_target(r["score_1_5"], r["applied"])
        if y is None:
            continue
        feat = json.loads(r["feature_json"])
        named = np.array([feat.get(k, 0.0) for k in FEATURE_NAMES], dtype=np.float32)
        # CONSISTENT feature dimensionality: always [dense(1024) ++ named]. A labeled row missing its
        # bge-m3 embedding (labelled before it was embedded) gets a ZERO dense block instead of a
        # short named-only vector — otherwise np.stack mixes 1053- and 29-dim rows and crashes, and a
        # model trained on one shape can't score the other. (best-practice: fixed feature width.)
        from .features import _blob_to_vec, EMBEDDING_DIM
        dense = _blob_to_vec(r["dense"]) if r["dense"] else np.zeros(EMBEDDING_DIM, dtype=np.float32)
        x = np.concatenate([dense, named], axis=0)
        Xs.append(x)
        ys.append(y)
        ws.append(_build_weight(r["score_1_5"], r["applied"], r["label_source"] or ""))
        groups.append(r["dedup_key"] or str(r["vacancy_id"]))
        vids.append(str(r["vacancy_id"]))

    if not Xs:
        from .features import EMBEDDING_DIM
        return (np.zeros((0, EMBEDDING_DIM + len(FEATURE_NAMES)), dtype=np.float32),
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=np.float32),
                [], [])

    X = np.stack(Xs, axis=0)
    y = np.array(ys, dtype=np.float32)
    w = np.array(ws, dtype=np.float32)
    return X, y, w, groups, vids


def _monotone_constraints(n_features: int) -> list[int]:
    """Build monotone constraint list. Dense embedding features = 0."""
    n_named = len(FEATURE_NAMES)
    n_dense = n_features - n_named
    dense_c = [0] * max(0, n_dense)
    named_c = [_NAMED_CONSTRAINTS.get(k, 0) for k in FEATURE_NAMES]
    return dense_c + named_c


# ---------------------------------------------------------------------------
# Isotonic band calibration
# ---------------------------------------------------------------------------

def _fit_calibrator(raw_oof: np.ndarray, y_true: np.ndarray, cutoffs: dict[str, float]):
    """Fit isotonic regression on OOF predictions.

    Kept only if it improves top-band macro-F1; else identity (None).
    Monotone → ranking preserved.
    """
    try:
        from sklearn.isotonic import IsotonicRegression  # type: ignore
        from sklearn.metrics import f1_score  # type: ignore
    except ImportError:
        return None

    cal = IsotonicRegression(y_min=1.0, y_max=5.0, out_of_bounds="clip")
    cal.fit(raw_oof, y_true)
    cal_pred = cal.predict(raw_oof)

    # Compare top-band F1 (score ≥ 4.5 = "must")
    threshold = cutoffs.get("must", 4.5)
    raw_top = (raw_oof >= threshold).astype(int)
    cal_top = (cal_pred >= threshold).astype(int)
    true_top = (y_true >= threshold).astype(int)

    def _f1(pred):
        if pred.sum() == 0:
            return 0.0
        return float(f1_score(true_top, pred, zero_division=0))

    f1_raw = _f1(raw_top)
    f1_cal = _f1(cal_top)
    return cal if f1_cal > f1_raw else None


# ---------------------------------------------------------------------------
# Temporal holdout
# ---------------------------------------------------------------------------

def _temporal_split(groups: list[str], frac: float = 0.2):
    """Hold out newest ~frac of unique groups (by insertion order as proxy for time).

    Returns (train_mask, test_mask) boolean arrays.
    """
    unique_groups = list(dict.fromkeys(groups))  # preserve insertion order = chronological
    n_test = max(1, int(len(unique_groups) * frac))
    test_groups = set(unique_groups[-n_test:])   # newest groups → test
    mask = np.array([g in test_groups for g in groups])
    return ~mask, mask


# ---------------------------------------------------------------------------
# Artifact persistence
# ---------------------------------------------------------------------------

def _artifact_path(cfg: dict) -> Path:
    model_dir = Path(cfg.get("paths", {}).get("model_dir", "data/models"))
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir / "triage.joblib"


def _save_artifact(cfg: dict, regressor, calibrator, n_features: int,
                   labels_sha: str, metrics: dict) -> str:
    try:
        import joblib  # type: ignore
    except ImportError:
        return ""
    path = _artifact_path(cfg)
    payload = {
        "regressor": regressor,
        "calibrator": calibrator,
        "n_features": n_features,
        "labels_sha": labels_sha,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
    }
    joblib.dump(payload, path)
    # JSON twin for human inspection
    meta = {k: v for k, v in payload.items() if k not in ("regressor", "calibrator")}
    path.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def _load_artifact(cfg: dict) -> dict | None:
    try:
        import joblib  # type: ignore
    except ImportError:
        return None
    path = _artifact_path(cfg)
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public: train
# ---------------------------------------------------------------------------

def train(cfg: dict, con, *, force: bool = False) -> dict:
    """Fit LightGBM triage gate on labeled+featured vacancies.

    Skips (no-op) when labels_sha unchanged, unless force=True.
    Returns OOF + temporal metrics.
    """
    _ensure_schema(con)

    try:
        import lightgbm as lgb  # type: ignore
        from sklearn.model_selection import GroupKFold  # type: ignore
    except ImportError:
        return {"error": "lightgbm or scikit-learn not installed"}

    triage_cfg = cfg.get("triage", {})
    cutoffs = triage_cfg.get("cutoffs", {"must": 4.5, "should": 3.5, "could": 2.0})
    min_labels = int(triage_cfg.get("min_labels_to_train", 30))

    X, y, w, groups, vids = _load_labeled(con)
    if len(y) == 0:
        return {"error": "no labeled vacancies with feature rows"}
    if len(y) < min_labels:
        return {"skipped": True, "reason": f"only {len(y)} labels (need {min_labels})"}

    # Check retrain-on-drift
    label_rows = con.execute(
        "SELECT vacancy_id, score_1_5, applied FROM label"
    ).fetchall()
    labels_sha = _labels_hash(label_rows)
    artifact = _load_artifact(cfg)
    if not force and artifact and artifact.get("labels_sha") == labels_sha:
        return {"skipped": True, "reason": "labels unchanged", "labels_sha": labels_sha}

    n_features = X.shape[1]
    mc = _monotone_constraints(n_features)

    defaults = {
        "objective": "regression", "n_estimators": 200, "num_leaves": 15,
        "max_depth": 4, "learning_rate": 0.05, "min_child_samples": 10,
        "reg_lambda": 1.0, "verbose": -1, "random_state": 42,
        "n_jobs": 1, "num_threads": 1,
        "monotone_constraints": mc,
    }
    reg = lgb.LGBMRegressor(**defaults)

    # OOF for calibration. GroupKFold needs n_splits <= n_unique_groups; with <2 groups it
    # raises ("n_splits greater than number of groups"), so fall back to in-sample preds.
    n_groups = len(set(groups))
    oof_pred = np.zeros_like(y)
    if n_groups < 2:
        reg.fit(X, y, sample_weight=w)
        oof_pred = reg.predict(X)
    else:
        kf = GroupKFold(n_splits=min(5, n_groups))
        for train_idx, val_idx in kf.split(X, groups=groups):
            reg.fit(X[train_idx], y[train_idx], sample_weight=w[train_idx])
            oof_pred[val_idx] = reg.predict(X[val_idx])

    # Full fit
    reg.fit(X, y, sample_weight=w)

    # Band calibration
    calibrator = _fit_calibrator(oof_pred, y, cutoffs)

    # Temporal holdout evaluation
    train_mask, test_mask = _temporal_split(groups)
    metrics: dict = {}
    if test_mask.sum() > 0:
        reg_t = lgb.LGBMRegressor(**defaults)
        reg_t.fit(X[train_mask], y[train_mask], sample_weight=w[train_mask])
        pred_test = reg_t.predict(X[test_mask])
        if calibrator:
            pred_test = calibrator.predict(pred_test)
        metrics = _compute_metrics(y[test_mask], pred_test, cutoffs)

    sha = _save_artifact(cfg, reg, calibrator, n_features, labels_sha, metrics)
    return {"trained": True, "n_rows": len(y), "labels_sha": labels_sha,
            "artifact_sha": sha, "metrics": metrics, "calibrated": calibrator is not None}


# ---------------------------------------------------------------------------
# Public: triage_pending
# ---------------------------------------------------------------------------

def triage_pending(cfg: dict, con, *, limit: int | None = None) -> dict[str, int]:
    """Score DESCRIBED vacancies → write triage_decision rows.

    Cold start (no artifact or < min_labels): use match_score, no hard-drop.
    Returns bucket counts + cold_start flag.
    """
    _ensure_schema(con)
    triage_cfg = cfg.get("triage", {})
    cutoffs = triage_cfg.get("cutoffs", {"must": 4.5, "should": 3.5, "could": 2.0})
    min_labels = int(triage_cfg.get("min_labels_to_train", 30))

    artifact = _load_artifact(cfg)
    n_labeled = con.execute("SELECT COUNT(*) FROM label").fetchone()[0]
    cold_start = artifact is None or n_labeled < min_labels

    sql = "SELECT id FROM vacancy WHERE status = ?"
    args = [Status.DESCRIBED.value]
    if limit:
        sql += " LIMIT ?"
        args.append(limit)
    vacancy_ids = [r[0] for r in con.execute(sql, args).fetchall()]

    counts = {"must": 0, "should": 0, "could": 0, "drop": 0}
    now = datetime.now(timezone.utc).isoformat()

    # If cold start: use match_score (scale [0,1] → [1,5])
    if cold_start:
        try:
            feat_rows = con.execute(
                "SELECT vacancy_id, match_score FROM vacancy_feature"
            ).fetchall()
        except Exception:
            feat_rows = []
        feat_map = {r[0]: r[1] or 0.0 for r in feat_rows}

        for vid in vacancy_ids:
            ms = feat_map.get(vid, 0.0)
            calibrated = 1.0 + 4.0 * float(ms)  # [0,1] → [1,5]
            priority = _score_to_priority(calibrated, cutoffs)
            counts[priority] += 1
            con.execute(
                """INSERT OR REPLACE INTO triage_decision
                   (vacancy_id, raw_score, calibrated_score, priority, model_version, decided_at)
                   VALUES (?, ?, ?, ?, 'cold_start', ?)""",
                (vid, ms, calibrated, priority, now),
            )
    else:
        reg = artifact["regressor"]
        cal = artifact.get("calibrator")
        n_feat = artifact.get("n_features", len(FEATURE_NAMES))

        for vid in vacancy_ids:
            vec = _feat.feature_vector(con, vid)
            if vec is None:
                # No feature row: use match_score fallback
                ms = _feat.feature_row(con, vid)
                ms = (ms or {}).get("match_score", 0.0) or 0.0
                raw = 1.0 + 4.0 * float(ms)
                calibrated = raw
                model_v = "cold_start"
            else:
                # Pad/trim to artifact's expected n_features
                if len(vec) < n_feat:
                    vec = np.pad(vec, (0, n_feat - len(vec)))
                elif len(vec) > n_feat:
                    vec = vec[:n_feat]
                raw = float(reg.predict(vec.reshape(1, -1))[0])
                raw = float(np.clip(raw, 1.0, 5.0))
                calibrated = float(cal.predict([raw])[0]) if cal else raw
                model_v = artifact.get("labels_sha", "trained")

            priority = _score_to_priority(calibrated, cutoffs)
            counts[priority] += 1
            con.execute(
                """INSERT OR REPLACE INTO triage_decision
                   (vacancy_id, raw_score, calibrated_score, priority, model_version, decided_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (vid, raw if "raw" in dir() else calibrated, calibrated, priority, model_v, now),
            )

    con.commit()
    counts["cold_start"] = cold_start
    return counts


# ---------------------------------------------------------------------------
# Public: select_for_normalize (replacement for db.by_status in normalize.py)
# ---------------------------------------------------------------------------

def _geo_near_first(rows: list, cfg: dict) -> list:
    """Stable-partition rows so near/unknown cities precede far-but-in-Germany ones, preserving the
    score order within each group. Far-DE jobs are no longer geo-dropped (geo.geo_class → kept +
    marked); this keeps the capped normalize budget from being starved by them — near jobs first."""
    from .geo import geo_class
    near, far = [], []
    for r in rows:
        (far if geo_class(r["city"], cfg) == "far" else near).append(r)
    return near + far


def select_for_normalize(cfg: dict, con, *, budget: int) -> list:
    """Return DESCRIBED vacancies eligible for normalize, ordered by calibrated_score DESC and then
    NEAR-cities-first (so far-but-in-Germany jobs, now kept not dropped, don't starve the budget).

    Cold start: all DESCRIBED included (no hard-drop), ordered by match_score.
    Trained: drop_priorities excluded, ranked by calibrated_score.
    Falls back to db.by_status when no triage_decision rows exist.
    """
    _ensure_schema(con)
    triage_cfg = cfg.get("triage", {})
    drop_set = set(triage_cfg.get("drop_priorities", ["drop"]))

    # Check if any triage decisions exist
    n_triage = con.execute("SELECT COUNT(*) FROM triage_decision").fetchone()[0]
    if n_triage == 0:
        rows = db.by_status(con, Status.DESCRIBED)
        return _geo_near_first(list(rows), cfg)[:budget]

    # Determine if we're in cold-start mode
    trained_row = con.execute(
        "SELECT 1 FROM triage_decision WHERE model_version != 'cold_start' LIMIT 1"
    ).fetchone()
    cold_start = trained_row is None

    # Fetch ALL eligible rows score-ordered (the DESCRIBED set is bounded by expire_stale), then
    # apply near-first + truncate to budget in Python (geo class needs the offline city table).
    if cold_start:
        # Rank only; no hard-drop
        rows = con.execute(
            """SELECT v.* FROM vacancy v
               LEFT JOIN triage_decision td ON td.vacancy_id = v.id
               WHERE v.status = ?
               ORDER BY COALESCE(td.calibrated_score, 1.0) DESC""",
            (Status.DESCRIBED.value,),
        ).fetchall()
    else:
        # Hard-drop enabled
        placeholders = ",".join("?" * len(drop_set))
        rows = con.execute(
            f"""SELECT v.* FROM vacancy v
                LEFT JOIN triage_decision td ON td.vacancy_id = v.id
                WHERE v.status = ?
                  AND (td.vacancy_id IS NULL OR td.priority NOT IN ({placeholders}))
                ORDER BY COALESCE(td.calibrated_score, 1.0) DESC""",
            (Status.DESCRIBED.value, *list(drop_set)),
        ).fetchall()

    return _geo_near_first(list(rows), cfg)[:budget]


# ---------------------------------------------------------------------------
# Public: read helpers
# ---------------------------------------------------------------------------

def scores_by_vacancy(con) -> dict[int, float]:
    """Return {vacancy_id: calibrated_score/5} for the slate blend."""
    _ensure_schema(con)
    rows = con.execute(
        "SELECT vacancy_id, calibrated_score FROM triage_decision"
    ).fetchall()
    return {int(r[0]): float(r[1]) / 5.0 for r in rows}


def triage_score(con, vacancy_id: int) -> float | None:
    """Return calibrated_score/5 for a single vacancy, or None."""
    _ensure_schema(con)
    row = con.execute(
        "SELECT calibrated_score FROM triage_decision WHERE vacancy_id = ?", (vacancy_id,)
    ).fetchone()
    return float(row[0]) / 5.0 if row else None


# ---------------------------------------------------------------------------
# Public: evaluate
# ---------------------------------------------------------------------------

def evaluate(cfg: dict, con) -> dict:
    """Temporal holdout evaluation: train on oldest 80% → predict newest 20% → metrics."""
    _ensure_schema(con)
    try:
        import lightgbm as lgb  # type: ignore
        from sklearn.model_selection import GroupKFold  # type: ignore
    except ImportError:
        return {"error": "lightgbm/sklearn not installed"}

    triage_cfg = cfg.get("triage", {})
    cutoffs = triage_cfg.get("cutoffs", {"must": 4.5, "should": 3.5, "could": 2.0})

    X, y, w, groups, vids = _load_labeled(con)
    if len(y) < 5:
        return {"error": f"too few labeled rows: {len(y)}"}

    train_mask, test_mask = _temporal_split(groups)
    if test_mask.sum() == 0:
        return {"error": "temporal split produced empty test set"}

    n_features = X.shape[1]
    mc = _monotone_constraints(n_features)
    defaults = {
        "objective": "regression", "n_estimators": 200, "num_leaves": 15,
        "max_depth": 4, "learning_rate": 0.05, "min_child_samples": 5,
        "reg_lambda": 1.0, "verbose": -1, "random_state": 42,
        "n_jobs": 1, "num_threads": 1,
        "monotone_constraints": mc,
    }
    reg = lgb.LGBMRegressor(**defaults)
    reg.fit(X[train_mask], y[train_mask], sample_weight=w[train_mask])
    pred_test = reg.predict(X[test_mask])

    metrics = _compute_metrics(y[test_mask], pred_test, cutoffs)
    metrics["n_train"] = int(train_mask.sum())
    metrics["n_test"]  = int(test_mask.sum())
    return metrics


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                      cutoffs: dict[str, float]) -> dict:
    metrics: dict = {}
    # scipy/sklearn are optional ([v2] extra). Catch ONLY the import: an absent library → record
    # the metric as None (honest "unavailable"); a real computation error is a bug and propagates.
    try:
        from scipy.stats import kendalltau, spearmanr  # type: ignore
    except ImportError:
        metrics["spearman_rho"] = metrics["kendall_tau"] = None
    else:
        metrics["spearman_rho"] = float(spearmanr(y_pred, y_true).statistic or 0)
        metrics["kendall_tau"]  = float(kendalltau(y_pred, y_true).statistic or 0)

    try:
        from sklearn.metrics import cohen_kappa_score, mean_absolute_error, ndcg_score  # type: ignore
    except ImportError:
        metrics["ndcg_10"] = metrics["mae"] = metrics["cohens_kappa"] = None
        return metrics
    n = len(y_true)
    if n > 1:
        metrics["ndcg_10"] = float(ndcg_score(
            y_true.reshape(1, -1), y_pred.reshape(1, -1), k=min(10, n)
        ))
    metrics["mae"] = float(mean_absolute_error(y_true, y_pred))

    def _bucket(scores):
        return [_score_to_priority(float(s), cutoffs) for s in scores]

    labels = ["must", "should", "could", "drop"]
    true_b = _bucket(y_true)
    pred_b = _bucket(y_pred)
    # cohen_kappa is undefined on degenerate (single-label) input → record None, don't crash.
    try:
        metrics["cohens_kappa"] = float(cohen_kappa_score(true_b, pred_b, labels=labels))
    except ValueError:
        metrics["cohens_kappa"] = None

    return metrics
