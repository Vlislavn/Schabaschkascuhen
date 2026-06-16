"""Measure the REAL bge-m3 HYBRID (dense + sparse-lexical + ColBERT) on the user's 37 real labels —
the SOTA upgrade the matcher was missing (it ships DENSE-ONLY via HyRE; sparse cache was empty +
ColBERT disabled).

bge-m3 is a native hybrid retrieval model: `BGEM3FlagModel.compute_score(pairs,
weights_for_different_modes=[w_dense,w_sparse,w_colbert])` returns dense / sparse / colbert and their
convex blends in one call (FlagEmbedding .../inference/embedder/encoder_only/m3.py:686-699). This probe
asks: does the sparse/ColBERT hybrid beat or complement HyRE (0.803 pairwise / 0.539 ndcg@10) on the user's
REAL clicks — BEFORE any production wiring (measure-then-ship).

ONE foreground bge-m3 load; everything else is read-only over the DB. Gate on `sysctl vm.swapusage`
before running. Run: .venv/bin/python -m eval.hybrid_measure --real-labels
"""
from __future__ import annotations

import json

from schabasch import aspects as _asp
from schabasch import config, db, features as _features, metrics, validation
from schabasch.candidate import candidate_doc, load_candidate


def main():
    cfg = config.load()
    con = db.connect(cfg["paths"]["db"])
    gold = validation.label_gold(con)

    feats: dict[int, dict] = {}
    for r in con.execute("SELECT vacancy_id, feature_json FROM vacancy_feature"):
        try:
            feats[int(r[0])] = json.loads(r[1])
        except Exception:
            pass

    cv_full = (candidate_doc(con) or "")[:3000]
    cand = load_candidate(con) or {}
    cv_skills = (" ".join(cand.get("skills") or [])[:1500]) or cv_full[:1500]

    rows: dict[int, tuple[str, str]] = {}
    for vid in gold:
        r = con.execute("SELECT title, description FROM vacancy WHERE id = ?", (vid,)).fetchone()
        if r and (r["description"] or "").strip():
            rows[vid] = (r["title"] or "", r["description"] or "")
    ids = [i for i in gold if i in rows]
    print(f"labeled vacancies with a JD: {len(ids)} / {len(gold)}")

    pairs_full, pairs_mh = [], []
    for i in ids:
        title, desc = rows[i]
        pairs_full.append([cv_full, f"{title}\n{desc}"[:4000]])
        mh = _asp.segment_jd(title, desc).get("must_have") or desc
        pairs_mh.append([cv_skills, mh[:2000]])

    print("loading bge-m3 (foreground, single-instance) …")
    model = _features._load_model(cfg.get("features", {}).get("model", "BAAI/bge-m3"))
    print("computing native hybrid scores (dense/sparse/colbert + blends) …")
    sc_full = model.compute_score(pairs_full, weights_for_different_modes=[1, 1, 1],
                                  max_query_length=512, max_passage_length=512)
    sc_mh = model.compute_score(pairs_mh, weights_for_different_modes=[1, 1, 1],
                                max_query_length=512, max_passage_length=512)

    def to_dict(score_list) -> dict[int, float]:
        return {ids[k]: float(score_list[k]) for k in range(len(ids))}

    sig: dict[str, dict[int, float]] = {}
    for mode in ("dense", "sparse", "colbert", "sparse+dense", "colbert+sparse+dense"):
        sig[f"full:{mode}"] = to_dict(sc_full[mode])
        sig[f"mh:{mode}"] = to_dict(sc_mh[mode])
    sig["fit_hyre (SHIPPED)"] = {i: float(feats[i].get("fit_hyre") or 0.0) for i in ids if i in feats}
    sig["xenc_full"] = {i: float(feats[i].get("xenc_full") or 0.0) for i in ids if i in feats}
    sig["llm_cov"] = {i: float(feats[i].get("llm_cov") or 0.0) for i in ids if i in feats}

    print(f"\n=== bge-m3 native hybrid on REAL labels (n={len(ids)}) — beat fit_hyre 0.803/0.539 ===")
    for nm, s in sig.items():
        print(f"  {nm:28s}", metrics.evaluate(s, gold, name=""))

    hy = sig["fit_hyre (SHIPPED)"]

    def mm(d):
        vals = [d[i] for i in ids if i in d]
        lo, hi = min(vals), max(vals)
        rng = (hi - lo) or 1.0
        return {i: (d[i] - lo) / rng for i in d}

    def rrf(sigs, k=60):
        out = {i: 0.0 for i in ids}
        for s in sigs:
            present = [i for i in ids if i in s]
            rank = {i: r for r, i in enumerate(sorted(present, key=lambda x: -s[x]))}
            for i in present:
                out[i] += 1.0 / (k + rank[i] + 1)
        return out

    print("\n=== fusions with fit_hyre (does hybrid COMPLEMENT the dense HyRE?) ===")
    fuse_modes = {
        "full:colbert+sparse+dense": sig["full:colbert+sparse+dense"],
        "mh:colbert+sparse+dense": sig["mh:colbert+sparse+dense"],
        "full:sparse": sig["full:sparse"],
        "mh:colbert": sig["mh:colbert"],
        "mh:sparse": sig["mh:sparse"],
    }
    hymm = mm(hy)
    for nm, s in fuse_modes.items():
        print(f"  RRF(hyre, {nm:24s})", metrics.evaluate(rrf([hy, s]), gold, name=""))
        smm = mm(s)
        conv = {i: 0.6 * hymm[i] + 0.4 * smm[i] for i in ids if i in smm and i in hymm}
        print(f"  0.6hyre+0.4*{nm:21s}", metrics.evaluate(conv, gold, name=""))

    # --- production-realizability: full:sparse raw range + FIXED-scale (per-vacancy) blends ---
    sp = sig["full:sparse"]
    spv = sorted(sp[i] for i in ids)
    import statistics as _st
    p = lambda q: spv[min(len(spv) - 1, int(q * len(spv)))]
    print(f"\n=== full:sparse RAW range — min {spv[0]:.4f} p50 {p(0.5):.4f} p90 {p(0.9):.4f} "
          f"max {spv[-1]:.4f} mean {_st.mean(spv):.4f} ===")
    print("  (min-max win must survive a FIXED per-vacancy normalizer to ship cleanly)")
    for label, scale in (("/p90", p(0.9)), ("/max", spv[-1]), ("/mean", _st.mean(spv))):
        spn = {i: min(1.0, sp[i] / scale) if scale else 0.0 for i in ids}
        for w in (0.3, 0.4, 0.5):
            conv = {i: (1 - w) * hy[i] + w * spn[i] for i in ids if i in hy}
            print(f"  {(1-w):.1f}hyre+{w:.1f}*sparse{label:6s}", metrics.evaluate(conv, gold, name=""))


if __name__ == "__main__":
    main()
