"""Fast signal-combination search over the gold set — no model re-runs (uses stored feature_json).
Lets me find the best fit blend / effective formula and PROVE it beats baseline before shipping.

Two gold sources (Receipts over assertions):
  default        → synthetic Opus GOLD (eval.match_eval.GOLD) — a dev-floor for ranking regressions.
  --real-labels  → Alina's REAL label table (validation.label_gold) — what actually tracks her taste.

The 2026-06-15 re-tune was measured here on --real-labels: on real clicks HyRE is the BEST single
signal and llm_cov the WEAKEST (the inverse of the synthetic ranking), so the headline fit blend is
now HyRE-led and the effective sort is FIT-led (de-conflate card: fit must be able to lead, the
magnet judge only differentiates among comparable-fit jobs).
"""
from __future__ import annotations

import json
import sys

from schabasch import config, db, eligibility as _elig, features as _features
from schabasch.candidate import load_candidate
from eval.match_eval import GOLD, evaluate, top_bottom


def _load(con):
    feats, judge = {}, {}
    for r in con.execute("SELECT vacancy_id, feature_json FROM vacancy_feature"):
        try:
            feats[int(r[0])] = json.loads(r[1])
        except Exception:
            pass
    for r in con.execute("SELECT vacancy_id, MAX(score) FROM judge_score GROUP BY vacancy_id"):
        judge[int(r[0])] = float(r[1] or 0)
    return feats, judge


def main():
    real = "--real-labels" in sys.argv
    cfg = config.load()
    con = db.connect(cfg["paths"]["db"])
    feats, judge = _load(con)
    if real:
        from schabasch import validation
        gold = validation.label_gold(con)
    else:
        gold = {i: v[0] for i, v in GOLD.items()}
    IDS = [i for i in gold if i in feats]

    def col(name, default=0.0):
        return {i: float(feats[i].get(name) if feats[i].get(name) is not None else default)
                for i in IDS}

    elig_stored = {i: float(feats[i]["elig_score"]) if feats[i].get("elig_score") is not None else 1.0
                   for i in IDS}
    jud = {i: judge.get(i, 0.0) for i in IDS}

    # LIVE eligibility recompute (Master-Data guard + high-fit soft lift), per vacancy, given a fit.
    # Cached req + candidate quals only — no model load. elig depends on fit (the lift), so it must be
    # recomputed per blend, not read from the stale stored elig_score.
    cand_quals = _elig.candidate_quals(load_candidate(con))
    e_cfg = cfg.get("eligibility", {})
    _vac = {i: con.execute("SELECT title, description FROM vacancy WHERE id=?", (i,)).fetchone()
            for i in IDS}
    _reqcache = {}
    for i in IDS:
        r = _vac[i]
        if not r:
            _reqcache[i] = None; continue
        ch = _features._content_hash(r["title"] or "", r["description"] or "")
        jd = f"{r['title'] or ''}\n{r['description'] or ''}"
        _reqcache[i] = _elig.req_from_cache(con, content_hash=ch, jd_text=jd)

    def elig_live(fit_dict):
        out = {}
        for i in IDS:
            req = _reqcache[i]
            if req is None:
                out[i] = elig_stored[i]; continue
            eg, _, _ = _elig.eligibility_gate(
                req, cand_quals, floor=float(e_cfg.get("floor", 0.35)),
                mid=float(e_cfg.get("mid", 0.6)), fit_score=fit_dict[i],
                soft_lift_threshold=float(e_cfg.get("soft_lift_threshold", 0.55)))
            out[i] = eg
        return out

    # --- candidate RAW-fit blends (production-realizable: all components in [0,1], no z-norm needed).
    # weights are {xenc, llm_cov, hyre}; fit_from_feature renormalizes over present signals.
    blends = {
        "BASE fit_score (stored)":          {"_stored": True},
        "xenc_full only":                   {"xenc": 1.0},
        "fit_hyre only":                    {"hyre": 1.0},
        "llm_cov only":                     {"llm_cov": 1.0},
        "0.7hyre+0.3xenc":                  {"hyre": 0.7, "xenc": 0.3},
        "0.6hyre+0.4xenc":                  {"hyre": 0.6, "xenc": 0.4},
        "0.5hyre+0.5xenc":                  {"hyre": 0.5, "xenc": 0.5},
        "0.8hyre+0.2xenc":                  {"hyre": 0.8, "xenc": 0.2},
        "0.6hyre+0.2xenc+0.2cov":           {"hyre": 0.6, "xenc": 0.2, "llm_cov": 0.2},
        "0.5hyre+0.3xenc+0.2cov":           {"hyre": 0.5, "xenc": 0.3, "llm_cov": 0.2},
        "0.5hyre+0.25xenc+0.25cov":         {"hyre": 0.5, "xenc": 0.25, "llm_cov": 0.25},
    }

    def fit_of(w):
        if w.get("_stored"):
            return col("fit_score")
        return {i: _features.fit_from_feature(feats[i], w) for i in IDS}

    print(f"=== FIT signal search on {'REAL labels' if real else 'SYNTHETIC gold'} "
          f"(n={len(IDS)}) — beat BASE fit_score (real pairwise .644 / ndcg .276) ===")
    fit_results = {}
    for nm, w in blends.items():
        sc = fit_of(w)
        fit_results[nm] = sc
        print(f"  {nm:28s}", evaluate(sc, gold, name=""))

    # --- EFFECTIVE = fit01 · (1 + β·judge_norm) · elig_LIVE  (FIT-led; magnet judge differentiates).
    # judge_norm = (judge-1)/4 in [0,1]; β small. elig recomputed live per blend (guard + lift).
    # Beats the OLD judge-led effective (real .564). Both stored-elig and live-elig shown for β=0.
    print("\n=== EFFECTIVE (fit · (1+β·judge_norm) · elig_live) — beat OLD effective (real pairwise .564 / ndcg .247) ===")
    jn = {i: max(0.0, (jud[i] - 1.0) / 4.0) for i in IDS}
    print("  --- β=0, STORED elig (the buggy gate, for contrast) ---")
    for nm, sc in fit_results.items():
        eff = {i: sc[i] * elig_stored[i] for i in IDS}
        print(f"    {nm:28s}", evaluate(eff, gold, name=""))
    for beta in (0.0, 0.15, 0.3):
        print(f"  --- β={beta}, LIVE elig (fixed gate) ---")
        for nm, sc in fit_results.items():
            el = elig_live(sc)
            eff = {i: sc[i] * (1.0 + beta * jn[i]) * el[i] for i in IDS}
            print(f"    {nm:28s}", evaluate(eff, gold, name=""))

    # reference: the magnet judge alone (the signal the OLD effective let lead)
    print("\n  judge_only (the old lead signal):", evaluate(jud, gold, name=""))

    best = max(fit_results.items(),
               key=lambda kv: evaluate(kv[1], gold)["pairwise_acc"])
    print(f"\nbest raw-fit blend: {best[0]}")
    if not real:
        print(top_bottom(best[1], gold))


if __name__ == "__main__":
    main()
