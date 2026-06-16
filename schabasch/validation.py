"""Live match-quality validation against Alina's REAL labels (the `label` table).

The `/eval` page calls `eval_report`. Gold = her actual score_1_5 ratings (applied→5), so the
numbers update automatically as she rates jobs in `/annotate` — no manual code re-point. Reuses
the metric helpers in `schabasch.metrics` (shared with the CLI `eval/match_eval.py`).

Leakage honesty: the fit signals (fit_score / xenc_full / llm_cov / elig_score) are computed from
CV↔JD only and never see labels → scoring them against labels is leak-free ("clean"). The judge
(few-shot trains on labels), the effective ranking (uses the judge), and triage (LightGBM trains
on labels) are OPTIMISTIC against the same labels → flagged clean=False.
"""
from __future__ import annotations

import json

from . import eligibility as _elig, features as _features, metrics, triage as _triage
from .candidate import load_candidate

# CV↔JD-only signals: never trained on labels, so evaluating them against labels is leak-free.
_CLEAN_SIGNALS = {
    "fit_score": "fit_score — итоговый матч CV↔вакансия",
    "xenc_full": "cross-encoder (CV↔полная вакансия)",
    "llm_cov": "покрытие требований (LLM)",
    "elig_score": "eligibility-гейт",
}


def label_gold(con) -> dict[int, int]:
    """Gold relevance from REAL labels: {vacancy_id: score_1_5}, with applied=1 → 5 fallback
    (matches triage._build_target). Both label sources count; if a vacancy is labelled by two
    sources, keep the higher score."""
    rows = con.execute(
        "SELECT vacancy_id, score_1_5, applied FROM label WHERE score_1_5 IS NOT NULL OR applied = 1"
    ).fetchall()
    gold: dict[int, int] = {}
    for r in rows:
        score = r["score_1_5"]
        val = int(score) if score is not None else (5 if r["applied"] else None)
        if val is None:
            continue
        vid = int(r["vacancy_id"])
        gold[vid] = max(gold.get(vid, val), val)
    return gold


def _latest_judge(con, rubric_version: str | None = None) -> dict[int, float]:
    """{vacancy_id: latest judge score} restricted to the active rubric (same pattern as
    slate._load_scored — stale-rubric scores never leak in)."""
    mid_sql = "SELECT vacancy_id, MAX(id) mid FROM judge_score"
    params: list = []
    if rubric_version is not None:
        mid_sql += " WHERE rubric_version = ?"
        params.append(rubric_version)
    mid_sql += " GROUP BY vacancy_id"
    rows = con.execute(
        f"SELECT m.vacancy_id, js.score FROM ({mid_sql}) m JOIN judge_score js ON js.id = m.mid",
        params,
    ).fetchall()
    return {int(r["vacancy_id"]): float(r["score"]) for r in rows}


def eval_report(cfg: dict, con) -> dict:
    """Compute ranking-quality of each matcher signal against Alina's real labels. Returns
    {n_labels, n_comparable_pairs, min_pairs, reliable, headline, rows[{name,label,clean,
    pairwise_acc,ndcg@10,spearman,n,n_pairs}]}. Empty/near-empty is handled by the caller."""
    _features._ensure_schema(con)   # /eval may be hit before any tick — guarantee the table exists
    gold = label_gold(con)
    feats: dict[int, dict] = {}
    for r in con.execute("SELECT vacancy_id, feature_json FROM vacancy_feature"):
        try:
            feats[int(r[0])] = json.loads(r[1])
        except (TypeError, ValueError, json.JSONDecodeError):
            continue  # skip a single malformed feature row, don't abort the whole report
    rubric = cfg.get("judge", {}).get("rubric_version")
    judge = _latest_judge(con, rubric)
    triage = _triage.scores_by_vacancy(con)

    # fit_score + eligibility recomputed LIVE (current fit_weights + Master-Data/high-fit-lift gate)
    # so /eval reflects production WITHOUT a heavy rerank — same source as slate.build_slate.
    cand_quals = _elig.candidate_quals(load_candidate(con))
    live = {i: _features.recompute_live(con, i, cfg, cand_quals=cand_quals)
            for i in gold if i in feats}

    def sig(name: str) -> dict[int, float]:
        if name in ("fit_score", "elig_score"):   # use the live-recomputed value
            return {i: float(live[i][name]) for i in gold if i in live}
        return {i: float(feats[i].get(name) or 0.0) for i in gold if i in feats}

    rows: list[dict] = []
    for name, label in _CLEAN_SIGNALS.items():
        m = metrics.evaluate(sig(name), gold, name=name)
        m["label"], m["clean"] = label, True
        rows.append(m)

    # leaky (train on labels → optimistic against the same labels)
    jscore = {i: float(judge.get(i, 0)) for i in gold if i in judge}
    mj = metrics.evaluate(jscore, gold, name="judge_only")
    mj["label"], mj["clean"] = "оценка судьи (магнит)", False
    rows.append(mj)

    # PRODUCTION effective ranking (mirrors slate._effective): FIT LEADS, magnet judge is a small
    # bounded differentiator. effective = fit · (1 + β·judge_norm) · elig (β = slate.judge_blend_beta).
    beta = float(cfg.get("slate", {}).get("judge_blend_beta", 0.0))
    eff: dict[int, float] = {}
    for i in gold:
        if i in live:
            j = float(judge.get(i, 0))
            judge_norm = max(0.0, (j - 1.0) / 4.0)
            eff[i] = live[i]["fit_score"] * (1.0 + beta * judge_norm) * live[i]["elig_score"]
    me = metrics.evaluate(eff, gold, name="effective")
    me["label"], me["clean"] = "итоговое ранжирование (fit·judge·elig)", False
    rows.append(me)

    tscore = {i: float(triage.get(i, 0.0)) for i in gold if i in triage}
    if tscore:
        mt = metrics.evaluate(tscore, gold, name="triage")
        mt["label"], mt["clean"] = "ML-гейт (triage)", False
        rows.append(mt)

    headline = rows[0]  # fit_score — the clean, label-independent headline
    min_pairs = int(cfg.get("slate", {}).get("eval_min_pairs", 15))
    return {"n_labels": len(gold), "n_comparable_pairs": headline["n_pairs"],
            "min_pairs": min_pairs, "reliable": headline["n_pairs"] >= min_pairs,
            "headline": headline, "rows": rows}
