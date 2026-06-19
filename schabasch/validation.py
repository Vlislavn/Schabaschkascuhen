"""Live match-quality validation against the user's REAL labels (the `label` table).

The `/eval` page calls `eval_report`. Gold = the user's actual score_1_5 ratings (applied→5), so the
numbers update automatically as the user rates jobs in `/annotate` — no manual code re-point. Reuses
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
    """Compute ranking-quality of each matcher signal against the user's real labels. Returns
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
    _triage._ensure_schema(con)   # ensure triage_decision exists; the ML-gate row queries it directly

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

    # PRODUCTION effective ranking — uses the SHARED slate.effective_score so /eval can never drift
    # from the slate (fit_eff = (1-λ)·fit + λ·llm_cov, × judge × elig × role-kind). Needs title (rk)
    # + llm_cov (coverage blend) per vacancy; summary is title-only here (role_kind keys on the title).
    from . import slate as _slate
    titles = {int(r[0]): (r[1] or "") for r in con.execute(
        "SELECT id, title FROM vacancy WHERE id IN (%s)" % ",".join("?" * len(gold)),
        tuple(gold))} if gold else {}
    eff: dict[int, float] = {}
    for i in gold:
        if i in live:
            eff[i] = _slate.effective_score(
                {"fit_score": live[i]["fit_score"], "elig_score": live[i]["elig_score"],
                 "llm_cov": feats.get(i, {}).get("llm_cov"), "score": judge.get(i),
                 "title": titles.get(i, ""), "summary": ""}, cfg, con)
    me = metrics.evaluate(eff, gold, name="effective")
    me["label"], me["clean"] = "итоговое ранжирование (fit·judge·elig)", False
    rows.append(me)

    # P1 — veto-aware DIRECTIONAL gold: the raw label masked by a "🙅 wrong role" vote (fits=0 → 0),
    # i.e. "what the rank SHOULD agree with once role is separated from domain". Rewards a role-aware
    # rank, but is SELF-CONFIRMING (built from the same vetoes the learned multiplier trains on) → it is
    # DIRECTIONAL ONLY: it may veto a ship, never justify one. The raw-label `effective` row above stays
    # the decisive guardrail (Delphi panel). Absent until any golden role vote exists.
    from . import role_feedback as _rolefb
    veto = _rolefb.veto_map(con, source="slate")
    if veto:
        gold_eff = {i: (0.0 if veto.get(i) == 0 else float(gold[i])) for i in gold}
        mre = metrics.evaluate({i: eff[i] for i in eff if i in gold_eff}, gold_eff, name="effective_role")
        mre["label"], mre["clean"] = "role-aware gold (directional — НЕ gate)", False
        rows.append(mre)

    # ML-гейт (triage): a DROP-filter, not a CV↔JD matcher. Measure ONLY actual ML-model decisions
    # (model_version != 'cold_start') — the cold_start match_score sediment dominated the stored scores
    # and made this read like a broken matcher (the 42%/−0.14 on a 2/3-cold_start mix). Deploy is now
    # gated on the model's temporal holdout (triage._should_deploy), so a deployed model has passed a
    # forward test. Omit the row when too few ML-scored jobs exist (honest "no signal" > a misleading number).
    triage_ml = {int(r[0]): float(r[1]) / 5.0 for r in con.execute(
        "SELECT vacancy_id, calibrated_score FROM triage_decision WHERE model_version != 'cold_start'")}
    tscore = {i: triage_ml[i] for i in gold if i in triage_ml}
    if len(tscore) >= 10:
        mt = metrics.evaluate(tscore, gold, name="triage")
        mt["label"], mt["clean"] = "ML-гейт (triage, drop-фильтр)", False
        rows.append(mt)

    headline = rows[0]  # fit_score — the clean, label-independent headline
    min_pairs = int(cfg.get("slate", {}).get("eval_min_pairs", 15))
    return {"n_labels": len(gold), "n_comparable_pairs": headline["n_pairs"],
            "min_pairs": min_pairs, "reliable": headline["n_pairs"] >= min_pairs,
            "headline": headline, "rows": rows}
