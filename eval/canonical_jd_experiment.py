"""Frontier experiment — does an LLM-canonicalized JD, embedded, beat raw-JD embedding for matching?

Tests the purest form of "LLM-assists-ML": the LLM extracts a CANONICAL skills/requirements
representation of each JD (boilerplate/slop stripped); we embed THAT with bge-m3 and measure the
CV↔JD similarity against Alina's real labels — vs the deterministic raw-JD baseline
(`sim_skills_requirements`), the current production `fit_score`, and `fit_hyre`/`bgem3_sparse`.

Phase-separated so the heavy 35B extractor never co-loads with the embedder:
  MODE=extract ARM=qwen|35b   → LLM-extract canonical text per labeled JD → cache to JSON (LLM only)
  MODE=eval                   → load bge-m3 once, embed cached texts (both arms) + comparators from
                                feature_json → evaluate every signal with bootstrap 95% CIs + a
                                5-fold held-out blend test → print the comparison table (no LLM)

Reuses (no rebuild): features._load_model/_encode_batch/_cosine01/_content_hash/bgem3_sparse_scores/
fit_from_feature/feature_row/_load_candidate_vecs, candidate.aspect_texts, validation.label_gold,
metrics.{pairwise_accuracy,ndcg_at_k,spearman,evaluate}, llm_clients.make_llm_client.

DS rigor: bootstrap CIs (n≈50 → mandatory), random + raw-JD baselines, held-out (5-fold) blend
selection (never fit-on-all-then-report), mean+median, pre-registered decision rule.
Run on a DB COPY. Memory-safe: 35B arm exclusive (evict ollama; serve_mlx); eval = bge-m3 only.
"""
from __future__ import annotations
import json, os, sys, statistics, time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/Users/vladnikulin/code/personal/Schabaschkascuhen")
from schabasch import config, db, features as F, metrics, validation, candidate, memory_guard
from schabasch.llm_clients import make_llm_client

MODE = os.environ.get("MODE", "eval")
ARM = os.environ.get("ARM", "qwen")                 # qwen | 35b  (extract mode)
DB = os.environ.get("PROBE_DB", config.load()["paths"]["db"])
CACHE_PATH = Path("/tmp/schabasch_capa/canon_cache.json")
BOOT_N = int(os.environ.get("BOOT_N", "1000"))
_RNG = np.random.default_rng(42)                    # deterministic resampling

_CANON_SYSTEM = (
    "You extract a CANONICAL, normalized requirements profile from a job posting. From the JOB, list "
    "ONLY the genuine must-have skills, tools, technologies, methods and qualifications needed to DO "
    "the role. STRIP all company boilerplate, perks, benefits, marketing fluff, EVP and culture-speak. "
    "Use short canonical phrases (e.g. 'SQL', 'process modelling (BPMN)', 'stakeholder management', "
    "'Bachelor in business/IT'). Merge synonyms. "
    'Return ONLY JSON: {"canonical_skills": ["...", "..."]}'
)

_ARM_ROLE = {
    "qwen": {"client": "ollama", "model": "qwen3:8b"},
    "35b": {"client": "openai", "provider": "openai",
            "model": "/Users/vladnikulin/models/mlx/Qwen3.6-35B-OptiQ-4bit",
            "base_url": "http://localhost:8082/v1", "api_key": "mlx"},
}


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())
    return {}


def _save_cache(c: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(c, ensure_ascii=False, indent=0))


def _labeled_rows(con) -> list[dict]:
    gold = validation.label_gold(con)
    rows = []
    for vid in gold:
        r = con.execute("SELECT title, description FROM vacancy WHERE id=?", (vid,)).fetchone()
        if r and (r["description"] or r["title"]):
            rows.append({"id": vid, "title": r["title"] or "", "desc": r["description"] or "",
                         "ch": F._content_hash(r["title"] or "", r["description"] or "")})
    return rows


# ---------------------------------------------------------------- extract phase (LLM only)
def run_extract(con):
    cfg = config.load()
    cfg.setdefault("llm", {}).setdefault("roles", {})["canon"] = _ARM_ROLE[ARM]
    memory_guard.require_headroom(f"canonical-extract {ARM}")
    client = make_llm_client(cfg, "canon")
    # qwen3:8b is a thinking model — without /no_think a one-shot extraction can loop/stall on a
    # pathological JD (same lesson as the agent). The 35B serves enable_thinking:false already.
    system = ("/no_think\n" + _CANON_SYSTEM) if ARM == "qwen" else _CANON_SYSTEM
    cache = _load_cache()
    rows = _labeled_rows(con)
    n_new, n_fail = 0, 0
    t0 = time.time()
    for i, row in enumerate(rows):
        key = f"{ARM}:{row['ch']}"
        if key in cache:
            continue
        jd = f"{row['title']}\n{row['desc'][:4000]}"
        try:
            obj = client.chat_json(system, jd)
            skills = [str(s).strip() for s in (obj.get("canonical_skills") or []) if str(s).strip()]
            cache[key] = ", ".join(skills)
            n_new += 1
        except Exception as e:
            n_fail += 1
            print(f"  [{i}] vid={row['id']} extract failed: {type(e).__name__}: {str(e)[:60]}")
        if n_new and n_new % 5 == 0:
            _save_cache(cache)
    _save_cache(cache)
    print(f"ARM={ARM} extract: +{n_new} new, {n_fail} fail, {len(rows)} labeled, {time.time()-t0:.0f}s")


# ---------------------------------------------------------------- eval phase (bge-m3 only, no LLM)
def _bootstrap_pairwise_ci(scores: dict, gold: dict, n: int = BOOT_N) -> tuple[float, float]:
    """95% CI for pairwise accuracy by resampling labeled vacancies with replacement."""
    ids = [i for i in gold if i in scores]
    if len(ids) < 4:
        return (float("nan"), float("nan"))
    accs = []
    for _ in range(n):
        samp = list(_RNG.choice(ids, size=len(ids), replace=True))
        # de-dup-free resample: pairwise over the multiset's distinct ids weighted by draw is complex;
        # use the resampled id list to subset gold/scores (distinct ids present in the draw).
        sub = {i: gold[i] for i in set(samp)}
        ssc = {i: scores[i] for i in set(samp)}
        if len(sub) >= 4:
            acc, _ = metrics.pairwise_accuracy(ssc, sub)
            accs.append(acc)
    if not accs:
        return (float("nan"), float("nan"))
    return (float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5)))


def _kfold_blend(fit: dict, canon: dict, gold: dict, k: int = 5) -> tuple[float, float]:
    """5-fold HELD-OUT blend(α·fit+(1-α)·canon): pick α on train folds, eval on the held-out fold.
    Returns (mean held-out pairwise, last-fold α). Honest — no fit-on-all-then-report."""
    ids = [i for i in gold if i in fit and i in canon]
    rng = np.random.default_rng(7)
    rng.shuffle(ids)
    folds = [ids[i::k] for i in range(k)]
    alphas = [i / 10 for i in range(11)]
    test_accs, last_a = [], 0.0
    for fi in range(k):
        test = set(folds[fi])
        train = [i for i in ids if i not in test]
        best_a, best_acc = 1.0, -1.0
        for a in alphas:
            sc = {i: a * fit[i] + (1 - a) * canon[i] for i in train}
            acc, _ = metrics.pairwise_accuracy(sc, {i: gold[i] for i in train})
            if acc > best_acc:
                best_acc, best_a = acc, a
        sc = {i: best_a * fit[i] + (1 - best_a) * canon[i] for i in test}
        acc, _ = metrics.pairwise_accuracy(sc, {i: gold[i] for i in test})
        test_accs.append(acc)
        last_a = best_a
    return (statistics.mean(test_accs) if test_accs else float("nan"), last_a)


def run_eval(con):
    cfg = config.load()
    gold = validation.label_gold(con)
    cache = _load_cache()
    model = F._load_model()
    cand_texts = candidate.aspect_texts(con)
    if not cand_texts:
        print("NO candidate profile — run `schabasch candidate` first"); return
    cv_vecs = F._load_candidate_vecs(model, cand_texts, con) or {}
    cv_skills_text = cand_texts.get("skills_text", "")
    cv_skills_vec, cv_full_vec = cv_vecs.get("skills"), cv_vecs.get("full")
    fit_weights = (cfg.get("features", {}) or {}).get("fit_weights") or {"hyre": 0.7, "sparse": 0.3}

    rows = _labeled_rows(con)
    signals: dict[str, dict] = {}

    # --- comparators from stored feature_json (no model re-run) ---
    def _put(name, vid, val):
        signals.setdefault(name, {})[vid] = val
    for row in rows:
        vid = row["id"]
        feat = F.feature_row(con, vid) or {}
        if feat.get("sim_skills_requirements") is not None:
            _put("sim_skills_requirements (raw-JD baseline)", vid, float(feat["sim_skills_requirements"]))
        if feat.get("fit_hyre") is not None:
            _put("fit_hyre", vid, float(feat["fit_hyre"]))
        rs = feat.get("bgem3_sparse")
        if rs is not None:
            _put("bgem3_sparse(norm)", vid, min(1.0, float(rs) / F._SPARSE_NORM_SCALE))
        if feat.get("xenc_full") is not None:
            _put("xenc_full", vid, float(feat["xenc_full"]))
        if feat:
            _put("fit_score (production)", vid, F.fit_from_feature(feat, fit_weights))
        _put("random baseline", vid, float(_RNG.random()))

    # --- canonical-JD signals per arm (embed cached texts; bge-m3 only) ---
    for arm in ("qwen", "35b"):
        keyed = [(row["id"], cache.get(f"{arm}:{row['ch']}")) for row in rows]
        present = [(vid, txt) for vid, txt in keyed if txt]
        if not present:
            continue
        vids = [v for v, _ in present]
        texts = [t for _, t in present]
        dense = F._encode_batch(model, texts)["dense_vecs"]
        sparse = F.bgem3_sparse_scores(model, cv_skills_text, texts)
        for j, vid in enumerate(vids):
            if cv_skills_vec is not None:
                _put(f"canon→cv_skills [{arm}]", vid, F._cosine01(dense[j], cv_skills_vec))
            if cv_full_vec is not None:
                _put(f"canon→cv_full [{arm}]", vid, F._cosine01(dense[j], cv_full_vec))
            _put(f"canon_sparse→cv [{arm}]", vid, min(1.0, float(sparse[j]) / F._SPARSE_NORM_SCALE))

    # --- evaluate every signal vs gold + bootstrap CI ---
    print(f"\n=== n_labels={len(gold)}  fit_weights={fit_weights}  bootstrap_n={BOOT_N} ===")
    print(f"{'signal':40} {'pairwise':>9} {'95% CI':>16} {'ndcg@10':>8} {'spear':>7} {'n':>4}")
    rank = []
    for name, sc in signals.items():
        ev = metrics.evaluate(sc, gold, name=name)
        lo, hi = _bootstrap_pairwise_ci(sc, gold)
        rank.append((ev["pairwise_acc"], name, ev, (lo, hi)))
        print(f"{name:40} {ev['pairwise_acc']:>9.3f} [{lo:.3f},{hi:.3f}] "
              f"{ev['ndcg@10']:>8.3f} {ev['spearman']:>7.3f} {ev['n']:>4}")

    # --- held-out blend: does the best canonical signal ADD on top of production fit? ---
    fit_sc = signals.get("fit_score (production)", {})
    print("\n=== 5-fold HELD-OUT blend(α·fit + (1−α)·canon) vs fit-alone ===")
    base_mean, _ = _kfold_blend(fit_sc, fit_sc, gold)  # fit blended with itself = fit-alone held-out
    print(f"  fit-alone (held-out)              pairwise={base_mean:.3f}")
    for name, sc in signals.items():
        if name.startswith("canon") and fit_sc:
            m, a = _kfold_blend(fit_sc, sc, gold)
            flag = "  ← ADDS" if m > base_mean + 1e-9 else ""
            print(f"  fit + {name:30} pairwise={m:.3f}  (α_fit≈{a:.1f}){flag}")

    # --- pre-registered decision ---
    base = signals.get("sim_skills_requirements (raw-JD baseline)", {})
    base_ev = metrics.evaluate(base, gold) if base else None
    print("\n=== DECISION (pre-registered) ===")
    if base_ev:
        b_lo, b_hi = _bootstrap_pairwise_ci(base, gold)
        print(f"  raw-JD baseline pairwise={base_ev['pairwise_acc']:.3f} CI[{b_lo:.3f},{b_hi:.3f}] — the bar to beat")
    for pw, name, ev, (lo, hi) in sorted(rank, reverse=True):
        if name.startswith("canon"):
            verdict = "CLEARS bar (CI-lo > baseline-pt)" if (base_ev and lo > base_ev["pairwise_acc"]) \
                      else "within noise / does NOT clear"
            print(f"  {name:36} pairwise={pw:.3f} CI-lo={lo:.3f} → {verdict}")


def main():
    con = db.connect(DB)
    print(f"MODE={MODE} ARM={ARM} db={os.path.basename(DB)}")
    if MODE == "extract":
        run_extract(con)
    else:
        run_eval(con)
    con.close()


if __name__ == "__main__":
    main()
