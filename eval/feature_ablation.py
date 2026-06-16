"""Per-feature ablation over stored `feature_json` vs the user's real labels — MODEL-FREE.

Run:  python -m eval.feature_ablation [--real-labels] [--boot=500]

Answers, per feature, the three questions the review's gap #2 raised:
  MODE 1  STANDALONE ranking power   — does the feature, alone, rank the labels at all? (oriented by
                                       spearman sign; pairwise/ndcg/spearman + bootstrap 95% CI)
  MODE 2  LEAVE-ONE-OUT of prod fit  — drop each component of the production fit blend → Δpairwise
                                       → what the matcher currently RELIES on
  MODE 3  ADD-ONE-IN (5-fold HELD-OUT)— blend(prod_fit, feature) with α picked on train folds, eval on
                                       held-out fold → does the feature ADD orthogonal signal? ← the
                                       real "earns its place" test (mirrors eval/canonical_jd_experiment)

Reads cached `vacancy_feature.feature_json` only — NO LLM / bge-m3 / 35B, runs in seconds, deterministic.
Decision rule: a feature earns a place only if MODE-3 held-out Δ > 0. At n≈50 CIs are wide — single-feature
Δ < ~0.05 is noise; the honest takeaway is usually "grow the label set", not "ship this feature".
"""
from __future__ import annotations
import statistics, sys

import numpy as np

from schabasch import config, db, metrics, validation, features as F, aspects
from eval.experiment import _load  # reuse the feature_json loader (no model runs)

_RNG = np.random.default_rng(42)
_BOOT = int(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--boot=")), "500"))

_LLM = {"fit_hyre", "llm_cov", "requirements_verified", "company_known", "salary_vs_target_gap"}
_TASTE = {"nearest_liked_cosine", "positive_centroid_cosine", "recent_centroid_cosine",
          "topic_drift", "company_overlap_count"}


def _ftype(name: str) -> str:
    return "LLM" if name in _LLM else ("taste" if name in _TASTE else "det")


def _bootstrap_ci(scores: dict, gold: dict, n: int = _BOOT) -> tuple[float, float]:
    """95% CI for pairwise accuracy, resampling labeled vacancies with replacement."""
    ids = [i for i in gold if i in scores]
    if len(ids) < 4:
        return (float("nan"), float("nan"))
    accs = []
    for _ in range(n):
        s = set(_RNG.choice(ids, size=len(ids), replace=True))
        if len(s) >= 4:
            a, _ = metrics.pairwise_accuracy({i: scores[i] for i in s}, {i: gold[i] for i in s})
            accs.append(a)
    return (float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))) if accs else (float("nan"), float("nan"))


def _kfold_blend(fit: dict, feat: dict, gold: dict, k: int = 5) -> float:
    """Mean HELD-OUT pairwise of blend(α·fit+(1−α)·feat), α chosen per train-fold (no fit-on-all)."""
    ids = [i for i in gold if i in fit and i in feat]
    rng = np.random.default_rng(7)
    rng.shuffle(ids)
    folds = [ids[i::k] for i in range(k)]
    accs = []
    for fi in range(k):
        test = set(folds[fi])
        train = [i for i in ids if i not in test]
        best_a, best_acc = 1.0, -1.0
        for a in (j / 10 for j in range(11)):
            sc = {i: a * fit[i] + (1 - a) * feat[i] for i in train}
            acc, _ = metrics.pairwise_accuracy(sc, {i: gold[i] for i in train})
            if acc > best_acc:
                best_acc, best_a = acc, a
        sc = {i: best_a * fit[i] + (1 - best_a) * feat[i] for i in test}
        acc, _ = metrics.pairwise_accuracy(sc, {i: gold[i] for i in test})
        accs.append(acc)
    return statistics.mean(accs) if accs else float("nan")


def _norm01(d: dict) -> dict:
    vals = list(d.values())
    lo, hi = min(vals), max(vals)
    return {i: ((v - lo) / (hi - lo) if hi - lo > 1e-9 else 0.5) for i, v in d.items()}


def main():
    real = "--real-labels" in sys.argv
    cfg = config.load()
    con = db.connect(cfg["paths"]["db"])
    feats, _ = _load(con)
    if real:
        gold = validation.label_gold(con)
    else:
        from eval.match_eval import GOLD
        gold = {i: v[0] for i, v in GOLD.items()}
    IDS = [i for i in gold if i in feats]
    con.close()

    def col(name: str) -> dict:
        return {i: float(feats[i][name]) for i in IDS if isinstance(feats[i].get(name), (int, float, bool))}

    cand = list(dict.fromkeys(list(aspects.FEATURE_NAMES) + ["bgem3_sparse", "xenc_full", "xenc_musthave", "llm_cov"]))
    feat_scores = {}
    for name in cand:
        if name == "fit_score":  # the blend target, not a feature
            continue
        sc = col(name)
        if len(sc) >= max(10, len(IDS) // 2):  # enough coverage to be meaningful
            feat_scores[name] = sc

    prod_w = (cfg.get("features", {}) or {}).get("fit_weights") or {"hyre": 0.7, "sparse": 0.3}
    fit = {i: F.fit_from_feature(feats[i], prod_w) for i in IDS}

    # ── MODE 1 — standalone ranking power ──────────────────────────────────────
    print(f"\n=== MODE 1 — STANDALONE ranking power vs {'REAL' if real else 'SYNTH'} labels "
          f"(n={len(IDS)}, boot={_BOOT}, model-free) ===")
    print(f"{'feature':32}{'type':6}{'pairwise':>9}{'  95% CI':>17}{'ndcg':>7}{'spear':>7}{'cov':>5}")
    rows = []
    for name, sc in feat_scores.items():
        ev = metrics.evaluate(sc, gold)
        orient = 1.0
        if ev["spearman"] < 0:  # orient so higher=better; flag the natural direction
            sc = {i: -v for i, v in sc.items()}
            ev = metrics.evaluate(sc, gold)
            orient = -1.0
        lo, hi = _bootstrap_ci(sc, gold)
        rows.append((ev["pairwise_acc"], name, ev, (lo, hi), orient, len(sc)))
    for pw, name, ev, (lo, hi), orient, cov in sorted(rows, reverse=True):
        d = " (↓better)" if orient < 0 else ""
        print(f"{name:32}{_ftype(name):6}{pw:>9.3f}  [{lo:.3f},{hi:.3f}]{ev['ndcg@10']:>7.3f}"
              f"{ev['spearman']:>7.3f}{cov:>5}{d}")

    # ── MODE 2 — leave-one-out of the production fit blend ─────────────────────
    print(f"\n=== MODE 2 — LEAVE-ONE-OUT of production fit {prod_w} ===")
    base = metrics.evaluate(fit, gold)["pairwise_acc"]
    blo, bhi = _bootstrap_ci(fit, gold)
    print(f"  production fit              pairwise={base:.3f}  [{blo:.3f},{bhi:.3f}]")
    for k in list(prod_w):
        w2 = {kk: v for kk, v in prod_w.items() if kk != k}
        if not w2:
            continue
        sc = {i: F.fit_from_feature(feats[i], w2) for i in IDS}
        pw = metrics.evaluate(sc, gold)["pairwise_acc"]
        print(f"  − drop {k:18} pairwise={pw:.3f}  Δ={base - pw:+.3f}  (← contribution of {k})")

    # ── MODE 3 — add-one-in, 5-fold held-out (the earns-its-place test) ────────
    # Δ vs fit-alone is computed on the SAME id-subset + SAME folds per feature (each feature has its
    # own coverage), so the delta isolates the feature's contribution — not a subset artifact.
    print("\n=== MODE 3 — ADD-ONE-IN to production fit (5-fold HELD-OUT, same-subset Δ) — earns-its-place ===")
    addin = []
    for name, sc in feat_scores.items():
        if metrics.evaluate(sc, gold)["spearman"] < 0:
            sc = {i: -v for i, v in sc.items()}
        f01 = _norm01(sc)
        ids = [i for i in f01 if i in fit]
        fit_sub = {i: fit[i] for i in ids}
        gold_sub = {i: gold[i] for i in ids}
        m_feat = _kfold_blend(fit_sub, f01, gold_sub)
        m_base = _kfold_blend(fit_sub, fit_sub, gold_sub)   # fit-alone on the SAME ids/folds
        addin.append((m_feat - m_base, name, m_feat, m_base, len(ids)))
    for d, name, m, b, cov in sorted(addin, reverse=True):
        flag = "  ← ADDS" if d > 1e-9 else ("  ~noise" if d >= -0.02 else "  hurts")
        print(f"  + {name:30}({_ftype(name):5}) held-out={m:.3f} vs fit {b:.3f}  Δ={d:+.3f} (n={cov}){flag}")

    print(f"\nDECISION: a feature earns a place only if MODE-3 Δ>0 (adds beyond production fit). "
          f"At n={len(IDS)} CIs are wide → treat |Δ|<~0.05 as noise; grow the label set to confirm.")


if __name__ == "__main__":
    main()
