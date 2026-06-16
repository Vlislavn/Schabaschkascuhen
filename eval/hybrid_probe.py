"""Read-only HYBRID-SIGNAL probe over the gold set — NO model load, ALREADY-CACHED vectors only.

Mirrors eval/experiment.py's style (config-free CLI, --real-labels default-on, reuses
schabasch.metrics + validation + features helpers) but instead of sweeping fit BLENDS over stored
feature_json, it builds CANDIDATE SOTA signals from the cached BLOBS and measures whether any beats
the shipped fit_hyre (real pairwise 0.803 / ndcg@10 0.539) or improves the effective (0.793/0.483):

  (1) LIKED-SIMILARITY  — features.PositiveLibrary.build(con) (loads cached dense vecs of liked
      jobs; NO model). Per labeled vacancy, leave-one-out lib.taste_features(jd_dense, company=...,
      exclude_vacancy_id=vid). Evaluate nearest_liked / positive_centroid / recent_centroid cosine.
  (2) DENSE CV<->JD cosine — features._cosine01(cv_full_dense, jd_dense), both from cache.
  (3) SPARSE lexical overlap — DEFERRED: needs a model. JD must_have sparse is cached, but there is
      NO cached CV sparse/lexical weight map (candidate_profile has no sparse column, no side table),
      so _sparse_overlap(jd, cv) is not reconstructable read-only. Reported as deferred, NOT run.
  (4) FUSION — RRF(k=60) and convex blends of {fit_hyre, xenc_full, nearest_liked, centroid_liked}
      (all in [0,1]; min-max where a signal isn't already bounded). Each evaluated as a raw ranker
      AND as an effective base (fused * elig_stored, β=0 — FIT-led, matches shipped slate).

MEMORY SAFETY: reads vacancy_embedding.dense, candidate_profile.aspect_vecs slices, hyre_cache,
vacancy_feature ONLY. Never calls extract_features / rerank_scored / OllamaClient / _load_model /
_load_reranker / model.encode / _load_candidate_vecs (which can recompute on a hash mismatch).
"""
from __future__ import annotations

import sys

import numpy as np

from schabasch import config, db, features as _features
from schabasch.metrics import evaluate


# --- cached-blob loaders (NO model) -----------------------------------------

def _jd_dense(con) -> dict[int, np.ndarray]:
    """{vacancy_id: cached whole-JD bge-m3 dense vec} from vacancy_embedding.dense."""
    out: dict[int, np.ndarray] = {}
    for r in con.execute("SELECT vacancy_id, dense FROM vacancy_embedding WHERE dense IS NOT NULL"):
        out[int(r[0])] = _features._blob_to_vec(r[1])
    return out


def _cv_full_vec(con) -> np.ndarray | None:
    """The cached CV 'full' dense vec = 5th 1024-float slice of candidate_profile.aspect_vecs.
    Read the blob DIRECTLY (do NOT call _load_candidate_vecs — it may recompute via the model)."""
    row = con.execute(
        "SELECT aspect_vecs FROM candidate_profile ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if not row or not row["aspect_vecs"]:
        return None
    blob = row["aspect_vecs"]
    chunk = _features.EMBEDDING_DIM * 4   # float32 bytes per aspect vec
    if len(blob) != 5 * chunk:            # keys order: skills,experience,domains,roles,full
        return None
    return _features._blob_to_vec(blob[4 * chunk:5 * chunk])   # idx 4 = "full"


def _vac_companies(con, ids) -> dict[int, str]:
    out = {}
    for i in ids:
        r = con.execute("SELECT company FROM vacancy WHERE id=?", (i,)).fetchone()
        out[i] = (r["company"] if r and r["company"] else "")
    return out


def _minmax(d: dict[int, float]) -> dict[int, float]:
    """Min-max to [0,1] over the labeled set (RRF/convex inputs that aren't already bounded)."""
    if not d:
        return d
    vals = list(d.values())
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return {i: 0.0 for i in d}
    return {i: (v - lo) / (hi - lo) for i, v in d.items()}


def _rrf(signals: list[dict[int, float]], ids, k: int = 60) -> dict[int, float]:
    """Reciprocal-rank fusion: sum 1/(k+rank) over each signal's descending rank (1-based)."""
    out = {i: 0.0 for i in ids}
    for sig in signals:
        ranked = sorted(ids, key=lambda i: -sig.get(i, 0.0))
        for rank, i in enumerate(ranked, start=1):
            out[i] += 1.0 / (k + rank)
    return out


def main():
    real = "--real-labels" in sys.argv or not any(a == "--synthetic" for a in sys.argv)
    cfg = config.load()
    con = db.connect(cfg["paths"]["db"])

    if real:
        from schabasch import validation
        gold = validation.label_gold(con)
        gold_src = "REAL labels"
    else:
        from eval.match_eval import GOLD
        gold = {i: v[0] for i, v in GOLD.items()}
        gold_src = "SYNTHETIC gold"

    # stored feature components (fit_hyre / xenc_full / fit_score / elig_score) — read-only.
    import json
    feats: dict[int, dict] = {}
    for r in con.execute("SELECT vacancy_id, feature_json FROM vacancy_feature"):
        try:
            feats[int(r[0])] = json.loads(r[1])
        except Exception:
            pass

    jd_dense = _jd_dense(con)
    cv_full = _cv_full_vec(con)

    # eligible id set: labeled AND has a feature row (for fit_hyre) AND a cached dense (for taste/CV).
    # We report n per-signal; the FUSION set is the intersection so every component is real per id.
    feat_ids = [i for i in gold if i in feats]
    dense_ids = [i for i in gold if i in jd_dense]
    fuse_ids = [i for i in gold if i in feats and i in jd_dense]

    def fcol(name: str, ids) -> dict[int, float]:
        return {i: float(feats[i].get(name)) for i in ids
                if feats[i].get(name) is not None}

    # ---- shipped baselines (sanity check) ----
    fit_hyre = fcol("fit_hyre", feat_ids)
    xenc_full = fcol("xenc_full", feat_ids)
    fit_score = fcol("fit_score", feat_ids)
    elig_stored = {i: float(feats[i]["elig_score"]) if feats[i].get("elig_score") is not None else 1.0
                   for i in feat_ids}

    # ---- (1) LIKED-SIMILARITY via PositiveLibrary (NO model) ----
    lib = _features.PositiveLibrary.build(con)   # loads cached dense vecs of liked jobs only
    companies = _vac_companies(con, dense_ids)
    nearest_liked, centroid_liked, recent_liked = {}, {}, {}
    for i in dense_ids:
        tf = lib.taste_features(jd_dense[i], company=companies.get(i),
                                exclude_vacancy_id=i)   # leave-one-out
        nearest_liked[i] = tf["nearest_liked_cosine"]
        centroid_liked[i] = tf["positive_centroid_cosine"]
        recent_liked[i] = tf["recent_centroid_cosine"]

    # ---- (2) DENSE CV<->JD cosine (NO model) ----
    cv_jd_cos: dict[int, float] = {}
    if cv_full is not None:
        for i in dense_ids:
            cv_jd_cos[i] = _features._cosine01(cv_full, jd_dense[i])

    # ---- (3) SPARSE lexical overlap — DEFERRED (no cached CV sparse map) ----
    sparse_note = ("DEFERRED — JD aspects_json['sparse']['must_have'] is cached (non-empty on 12/30 "
                   "labeled vacancies) but there is NO cached CV sparse/lexical-weight map "
                   "(candidate_profile has no sparse column and no side table). _sparse_overlap needs "
                   "BOTH sides; reconstructing the CV side requires encoding the CV with bge-m3 "
                   "(model.encode return_sparse) — a model load. NOT run per memory-safety guard.")

    # ---- evaluate single signals ----
    print(f"=== HYBRID-SIGNAL probe on {gold_src} — beat fit_hyre (real .803/.539) "
          f"& effective (real .793/.483) ===")
    print(f"    feat_ids n={len(feat_ids)}  dense_ids n={len(dense_ids)}  fuse_ids n={len(fuse_ids)}")
    print(f"    PositiveLibrary liked rows (cached dense, score>=4|applied|interview): "
          f"n={lib.n_rows}  ids={lib.vacancy_ids}")

    def row(name, sc):
        m = evaluate(sc, gold, name=name)
        print(f"  {name:34s} pairwise={m['pairwise_acc']:.3f}  ndcg@10={m['ndcg@10']:.3f}  "
              f"spearman={m['spearman']:.3f}  n={m['n']}  pairs={m['n_pairs']}")
        return m

    print("\n--- BASELINES (sanity: fit_hyre must reproduce ~0.803) ---")
    row("fit_hyre (SHIPPED)", fit_hyre)
    row("xenc_full", xenc_full)
    row("fit_score (stored blend)", fit_score)

    print("\n--- (1) LIKED-SIMILARITY (small-sample caveat: liked n above) ---")
    row("nearest_liked_cosine", nearest_liked)
    row("positive_centroid_cosine", centroid_liked)
    row("recent_centroid_cosine", recent_liked)

    print("\n--- (2) DENSE CV<->JD cosine ---")
    if cv_jd_cos:
        row("cv_full<->jd dense cosine", cv_jd_cos)
    else:
        print("  cv_full<->jd dense cosine    SKIPPED — no readable cached CV 'full' vec")

    print("\n--- (3) SPARSE lexical overlap ---")
    print(f"  {sparse_note}")

    # ---- (4) FUSION on the intersection set (every component real per id) ----
    print(f"\n--- (4) FUSION (on fuse_ids n={len(fuse_ids)}; components min-maxed for RRF/convex) ---")
    fh = {i: fit_hyre[i] for i in fuse_ids if i in fit_hyre}
    xf = {i: xenc_full[i] for i in fuse_ids if i in xenc_full}
    nl = {i: nearest_liked[i] for i in fuse_ids}
    cl = {i: centroid_liked[i] for i in fuse_ids}
    # min-max the fusion inputs (cosines are bounded but compress better after min-max on n~30)
    fh_n, xf_n, nl_n, cl_n = _minmax(fh), _minmax(xf), _minmax(nl), _minmax(cl)

    fusions: dict[int, dict] = {}
    fusions["RRF(fit_hyre,xenc)"] = _rrf([fh, xf], fuse_ids)
    fusions["RRF(fit_hyre,xenc,nearest_liked)"] = _rrf([fh, xf, nl], fuse_ids)
    fusions["RRF(fit_hyre,nearest_liked,centroid)"] = _rrf([fh, nl, cl], fuse_ids)
    g = lambda d, i: d.get(i, 0.0)   # missing component (e.g. a vacancy lacking xenc_full) → 0.0
    fusions["0.8fit_hyre+0.2nearest"] = {
        i: 0.8 * g(fh_n, i) + 0.2 * g(nl_n, i) for i in fuse_ids}
    fusions["0.7fit_hyre+0.3nearest"] = {
        i: 0.7 * g(fh_n, i) + 0.3 * g(nl_n, i) for i in fuse_ids}
    fusions["0.6fit_hyre+0.2xenc+0.2nearest"] = {
        i: 0.6 * g(fh_n, i) + 0.2 * g(xf_n, i) + 0.2 * g(nl_n, i) for i in fuse_ids}
    fusions["0.5fit_hyre+0.25nearest+0.25centroid"] = {
        i: 0.5 * g(fh_n, i) + 0.25 * g(nl_n, i) + 0.25 * g(cl_n, i) for i in fuse_ids}

    fuse_results = {}
    for nm, sc in fusions.items():
        fuse_results[nm] = sc
        row(nm, sc)

    # ---- fused as the EFFECTIVE base: fused * elig_stored (β=0, FIT-led, shipped form) ----
    print("\n--- (4b) FUSION as EFFECTIVE base = fused * elig_stored (β=0) — beat effective .793/.483 ---")
    # reference: shipped effective on the SAME fuse_ids (fit_hyre * elig_stored, the real prod base is
    # fit_score*(1+0*judge)*elig = fit_score*elig; here we show fit_hyre*elig for a like-for-like base)
    ref_eff = {i: fit_hyre[i] * elig_stored[i] for i in fuse_ids if i in fit_hyre}
    row("EFF fit_hyre*elig (ref)", ref_eff)
    ref_eff_fs = {i: fit_score[i] * elig_stored[i] for i in fuse_ids if i in fit_score}
    row("EFF fit_score*elig (shipped)", ref_eff_fs)
    for nm, sc in fuse_results.items():
        eff = {i: sc[i] * elig_stored[i] for i in fuse_ids}
        row(f"EFF {nm}", eff)

    # ---- verdict (SAME-SET: every candidate lives on fuse_ids/dense_ids — the 6 gold vacancies
    # lacking a cached dense vec drop out — so compare against fit_hyre RESTRICTED to fuse_ids,
    # NOT the full-36 fit_hyre=0.803. The 0.803 is the cross-set headline used only for sanity.) ----
    fh_fuse = {i: fit_hyre[i] for i in fuse_ids if i in fit_hyre}
    base_pa = evaluate(fh_fuse, gold)["pairwise_acc"]
    base_nd = evaluate(fh_fuse, gold)["ndcg@10"]
    cand = {**{"nearest_liked": nearest_liked, "positive_centroid": centroid_liked,
               "recent_centroid": recent_liked},
            **({"cv_jd_cosine": cv_jd_cos} if cv_jd_cos else {}),
            **fuse_results}
    print(f"\n=== VERDICT — SAME-SET (fit_hyre on fuse_ids n={len(fh_fuse)}: "
          f"pairwise={base_pa:.3f} ndcg@10={base_nd:.3f}; full-36 fit_hyre=0.803/0.539 for sanity) ===")
    beat = [(nm, evaluate(sc, gold)["pairwise_acc"], evaluate(sc, gold)["ndcg@10"])
            for nm, sc in cand.items()]
    beat.sort(key=lambda t: -t[1])
    for nm, pa, nd in beat:
        flag = "BEATS" if pa > base_pa else ("ties " if pa == base_pa else "below")
        print(f"  {flag} fit_hyre  {nm:36s} pairwise={pa:.3f} (Δ{pa-base_pa:+.3f})  ndcg@10={nd:.3f}")


if __name__ == "__main__":
    main()
