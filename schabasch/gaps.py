"""Skill-gap analytics: across the jobs the user WANTS (😎/👸✨🧚/applied), which requirements recur as
✗ missing / ◐ partial — the "jobs I'd want but keep lacking skill X" report.

Reuses the per-requirement verdicts (ConFit-v3 "non-negotiable requirements") already stored in
`feature_json.llm_cov_reqs` by features._llm_coverage — pure aggregation, no new LLM calls. Surfaced
at `/gaps` (slate.render_gaps_html) and the `gaps` CLI command.

The qwen coverage judge is NOISY: it marks the candidate's real degree / languages "missing" in some
jobs and "present" in others (project memory `llmcov-false-missing-fix`). So before ranking a
requirement as a gap we RECONCILE each one against the candidate's STRUCTURED profile using the
existing eligibility ordinals (ISCED education level, CEFR language level) — a deterministic
ground-truth check, NOT CV-token matching. A requirement the candidate provably meets is dropped from
the gap list. Skill-level noise (e.g. "MS Office") has no structured ground truth and is left as-is —
the only further lever there is a stronger judge.
"""
from __future__ import annotations

import re

from . import candidate, eligibility as _elig, features as _features


def _norm(req: str) -> str:
    """Coarse grouping key for a requirement phrase — qwen names the same skill many ways
    ('Power BI' / 'Power BI dashboards' / 'Microsoft Power BI'), so clustering is approximate."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s/+#.-]", "", str(req).lower())).strip()


def _have_via(req: str, cand: dict) -> str:
    """If the candidate's STRUCTURED profile provably meets `req`, name the ground-truth signal
    ('education' | 'language'); else ''. Reuses eligibility.py's qualification matchers (ISCED level /
    CEFR) — the ONE place that logic lives — NOT CV-token matching (rejected as overfit; see memory)."""
    if _elig.meets_education(req, cand):
        return "education"
    if _elig.meets_language(req, cand):
        return "language"
    return ""


def gap_report(cfg: dict, con) -> dict:
    """Aggregate missing/partial/present requirement verdicts across WANTED vacancies
    (label.score_1_5 >= 4 OR applied = 1), using each vacancy's stored llm_cov_reqs, then RECONCILE
    away requirements the candidate's structured profile provably meets (education / language).

    Returns {n_wanted, n_jobs_with_reqs, rows:[{requirement, missing, partial, present, jobs}],
    reliable, candidate_skills, n_reconciled, reconciled:[{requirement, have_via}]}; genuine-gap rows
    sorted by (missing + 0.5*partial) desc (the worst gaps first).
    """
    wanted = [int(r[0]) for r in con.execute(
        "SELECT DISTINCT vacancy_id FROM label WHERE score_1_5 >= 4 OR applied = 1").fetchall()]
    agg: dict[str, dict] = {}
    n_jobs_with_reqs = 0
    for vid in wanted:
        feat = _features.feature_row(con, vid) or {}
        reqs = feat.get("llm_cov_reqs") or []
        if reqs:
            n_jobs_with_reqs += 1
        for r in reqs:
            if not isinstance(r, dict):
                continue
            key = _norm(r.get("requirement", ""))
            verdict = str(r.get("verdict", "")).lower()
            if not key or verdict not in ("missing", "partial", "present"):
                continue
            a = agg.setdefault(key, {"requirement": r.get("requirement", ""),
                                     "missing": 0, "partial": 0, "present": 0, "_jobs": set()})
            a[verdict] += 1
            a["_jobs"].add(vid)
    prof = candidate.load_candidate(con)
    cand = _elig.candidate_quals(prof)
    rows: list[dict] = []
    reconciled: list[dict] = []
    for a in agg.values():
        a["jobs"] = len(a.pop("_jobs"))
        via = _have_via(a["requirement"], cand)
        if via:
            reconciled.append({"requirement": a["requirement"], "have_via": via})
        else:
            rows.append(a)
    rows.sort(key=lambda a: -(a["missing"] + 0.5 * a["partial"]))
    return {"n_wanted": len(wanted), "n_jobs_with_reqs": n_jobs_with_reqs, "rows": rows,
            "reliable": len(wanted) >= 3, "candidate_skills": (prof or {}).get("skills") or [],
            "n_reconciled": len(reconciled), "reconciled": reconciled}
