"""Offline match-quality eval harness for the user's matcher.

Why: "make matches SOTA" is meaningless without a metric. This is an EXPERT GOLD SET — each of
the user's currently-scored jobs hand-labeled 0..3 for genuine fit (skill match × eligibility × interest),
grounded in the user's REAL CV: **Senior Business Analyst** (ex-Software Engineer), Bachelor in Business
Informatics (no master), German A2; skills = business analysis (requirements, BPMN/UML, process
design, target operating models, UAT), delivery/PM, Python/SQL/Tableau/Power BI/AWS, SWE background.
The user is NOT an ML engineer. Magnets (space/animals/military/public-sector/complex/new-domain) are
ASPIRATIONAL pivot domains, not the user's skill base. A method is GOOD iff it ranks high-gold above low.

Label rubric (CV-grounded): 3=strong (the user's core: business-analyst / process-owner / data-BI /
program-PM / consulting, eligible, ideally a magnet/pivot); 2=good (those skills clearly apply,
eligible); 1=adjacent/weak (tangential, OR aspiration-domain-but-wrong-skill, OR eligibility-
limited); 0=poor (not the user's skills — pure ML-research/sales/finance/procurement/web-dev — OR
ineligible: PhD/master/student positions).

HONEST CAVEAT (2026-06-15): GOLD was RE-LABELED from the real CV after discovering the stored
profile was wrong ("ML-engineer" → actually Senior BA), which had INVERTED the labels (the user's data/
process/PM/consulting strengths were scored low, pure-ML high). Still an Opus proxy (not the user's
own labels; the live `label` table holds ~1 real label) — a dev floor for ranking regressions,
refine after reading every JD + the qwen re-run on the real CV. Re-point to the `label` table
once ~30+ real labels accrue via /annotate.
"""
from __future__ import annotations

import json

# Metric helpers live in schabasch.metrics (shared with the live /eval page); re-export for
# back-compat with anything importing them from here.
from schabasch.metrics import (  # noqa: F401
    evaluate, ndcg_at_k, pairwise_accuracy, spearman, top_bottom,
)

# vacancy_id -> (gold_label, rationale)
GOLD: dict[int, tuple[int, str]] = {
    # 3 — the user's core skill (BA / process / data-BI / program-PM / consulting) + eligible + magnet/pivot
    166: (3, "PM Laser-Driven Radiation Sources — program/project MANAGEMENT (the user's core) + complex-project magnet, eligible"),
    # 2 — the user's skills clearly apply, eligible
    904: (2, "Senior Data Scientist Marketing Intelligence — data/marketing analytics = the user's Nielsen background"),
    67:  (2, "EMEA Demand Planner Aerospace — forecasting + data + process (the user's Nielsen work) + aerospace magnet"),
    455: (2, "GRO Data Analytics & Reporting — data analysis + Power BI + process improvement (exact fit); minus: internship"),
    906: (2, "GRO Data Analytics & Reporting — same (data/BI/process + nuclear/complex magnet)"),
    946: (2, "Inhouse Consulting (Merck) — consulting/process/transformation/PM (the user's core); minus: pharma-adjacent"),
    953: (2, "Inhouse Consulting Commercial (Merck) — same"),
    797: (2, "Senior Manager IT Controlling — ROI/cost-benefit + reporting/BI + process optimization (strong BA/analyst overlap)"),
    826: (2, "Product Owner Vulnerability Mgmt — PO/backlog/requirements (the user's BA/PO skills) + security magnet; minus: biotech co"),
    # 1 — adjacent/weak: tangential, aspiration-domain-but-wrong-skill, or eligibility-limited
    354: (1, "Senior ML/AI Engineer (CTI) — security magnet but ML-ENGINEER core the user lacks"),
    341: (1, "AI Engineer (Allianz) — GenAI/ML engineering, not the user's core (some Python overlap)"),
    345: (1, "Equity Platform AI Engineer — AI/ML engineering, not the user's core"),
    895: (1, "AI Automation Engineer — ML/automation engineering, not the user's core"),
    48:  (1, "BDM Space — space magnet but business-DEVELOPMENT/sales, not the user's"),
    198: (1, "SWE Full-Stack/Web @ VisionSpace — space/defense magnet but WEB dev, not the user's"),
    746: (1, "Senior Simulation Engineer Tire Wear — simulation/vehicle-dynamics core the user lacks"),
    363: (1, "Junior Consultant — consulting fits but junior/for-students (the user is senior — mismatch)"),
    432: (1, "Finance Associate Management Reporting — reporting/BI overlap but junior + finance domain"),
    439: (1, "Business Process Owner Master Data — BPO = the user's skill BUT master-required (elig) + pharma-adjacent"),
    597: (1, "BPO Third-Party Risk (Merck) — BPO/process/risk = the user's skill BUT master-required + pharma-adjacent"),
    451: (1, "Oil & Gas Project Manager — PM skill fits but O&G domain-specific knowledge the user lacks"),
    # 0 — not the user's skills, or ineligible
    361: (0, "Expert ML Optimization (self-driving) — deep ML + requires MSc/PhD (the user is Bachelor)"),
    42:  (0, "Sales Manager — pure sales, not the user's"),
    138: (0, "PhD Agentic AI — doctoral position, no master (ineligible)"),
    139: (0, "PhD Agentic AI — doctoral position, ineligible"),
    317: (0, "Cyber Threat Analyst — US TS/SCI clearance + not the user's skills"),
    449: (0, "Procurement Specialist — procurement, not the user's"),
    763: (0, "Purchasing Specialist — purchasing, not the user's"),
    891: (0, "Diploma Student AI Methods — student/thesis position, ineligible"),
    893: (0, "Masterarbeit Applied Stats — Master's-thesis position, ineligible"),
}


def _gold_scores() -> dict[int, int]:
    return {i: v[0] for i, v in GOLD.items()}


def _gold_rationales() -> dict[int, str]:
    return {i: v[1] for i, v in GOLD.items()}


if __name__ == "__main__":
    import sys

    from schabasch import config, db, eligibility as _elig, features as _features, validation
    from schabasch.candidate import load_candidate
    cfg = config.load()
    con = db.connect(cfg["paths"]["db"])
    # --real-labels: score against the user's real label table (live, the same gold the /eval page
    # uses) instead of the synthetic dev-floor GOLD. Default stays synthetic for regression checks.
    real = "--real-labels" in sys.argv
    gold = validation.label_gold(con) if real else _gold_scores()
    rationales = {} if real else _gold_rationales()
    feats = {}
    for r in con.execute("SELECT vacancy_id, feature_json FROM vacancy_feature"):
        try:
            feats[int(r[0])] = json.loads(r[1])
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
    judge = {int(r[0]): r[1] for r in con.execute(
        "SELECT vacancy_id, MAX(score) FROM judge_score GROUP BY vacancy_id")}

    # fit_score + elig recomputed LIVE (current fit_weights + fixed gate) — matches production.
    cand_quals = _elig.candidate_quals(load_candidate(con))
    live = {i: _features.recompute_live(con, i, cfg, cand_quals=cand_quals)
            for i in gold if i in feats}

    def sig(name):
        if name in ("fit_score", "elig_score"):
            return {i: float(live[i][name]) for i in gold if i in live}
        return {i: float(feats[i].get(name) or 0.0) for i in gold if i in feats}

    print(f"=== match-quality on {'REAL labels' if real else 'SYNTHETIC gold'} (n={len(gold)}) ===")
    for name in ("xenc_musthave", "xenc_full", "fit_hyre", "llm_cov", "fit_score"):
        print(" ", evaluate(sig(name), gold, name=name))
    # judge alone (the magnet signal) — expected to be a POOR fit-ranker
    jscore = {i: float(judge.get(i, 0)) for i in gold if i in judge}
    print(" ", evaluate(jscore, gold, name="judge_only"))
    # PRODUCTION effective (mirrors slate._effective + validation): FIT LEADS, magnet judge a small
    # bounded differentiator. effective = fit · (1 + β·judge_norm) · elig (β = slate.judge_blend_beta).
    beta = float(cfg.get("slate", {}).get("judge_blend_beta", 0.0))
    eff = {}
    for i in gold:
        if i in live:
            j = float(judge.get(i, 0))
            jn = max(0.0, (j - 1.0) / 4.0)
            eff[i] = live[i]["fit_score"] * (1.0 + beta * jn) * live[i]["elig_score"]
    print(" ", evaluate(eff, gold, name="effective(fit*judge*elig)"))
    print("\nfit_score (live) ranking:")
    print(top_bottom(sig("fit_score"), gold, rationales=rationales))
