"""Deterministic role-kind classifier — drives the engineer/junior down-rank + card flags (W1).

The user's review comments are emphatic and repeated: «никаких инженеров / не хочу руками, хочу
головой» (≥4 jobs) and intern/working-student rejections. This classifier turns the role TITLE
(+summary) into a coarse kind so the slate can softly down-rank hands-on-engineer and junior roles
and flag them — recall-first (never a hard drop: the user rated one engineer role 4, «правда прикольный,
но инженер»). It's title-keyword based (generalizable, no per-vacancy hardcode); the multiplier is
config-driven and MEASURED on real labels before being trusted (eval/experiment.py).

Kinds: "lead" (principal/lead/head — positive signal → no penalty), "junior" (intern/working-
student/diploma), "hands_on_engineer" (engineer/developer NOT also lead), "" (neutral / analyst /
manager / owner — roles that fit the profile).
"""
from __future__ import annotations

import re

_LEAD_RE = re.compile(r"\b(lead|principal|head\s+of|chief|director|vp|staff)\b", re.IGNORECASE)
# `\bintern(ship|s)?\b` matches intern/interns/internship but NOT international/internal (more
# letters → boundary fails). Stems like praktik\w* / werkstudent\w* catch Praktikum/Werkstudenten.
_JUNIOR_RE = re.compile(
    r"\b(intern(ship|s)?|praktik\w*|werkstudent\w*|working\s+student|trainee|diplomand|"
    r"diploma\s+student|studentische\w*|graduate\s+program|junior|entry[- ]level|ausbildung)\b",
    re.IGNORECASE)
_ENGINEER_RE = re.compile(
    r"\b(engineer|entwickler|developer|programmer|programmierer|sde|softwareentwickl)\w*\b",
    re.IGNORECASE)

# Default multipliers are NEUTRAL (multi-user fix 2026-07-03): the engineer/junior penalty is a
# PERSONAL taste, not a product default — it now lives in the user's config
# (slate.role_kind_mult; user #1's measured 0.45/0.5 moved into her profile.yaml explicitly).
_DEFAULT_MULT = {"hands_on_engineer": 1.0, "junior": 1.0, "lead": 1.0, "": 1.0}

_FLAG = {
    "hands_on_engineer": "🛠 hands-on — не твоё",
    "junior": "🎓 стажёр/junior",
    "lead": "",   # positive — no warning flag
    "": "",
}


def classify(title: str | None, summary: str | None = None) -> str:
    """Return the role kind from the title (summary as a weak secondary signal)."""
    t = f"{title or ''} {summary or ''}"
    is_lead = bool(_LEAD_RE.search(t))
    if _JUNIOR_RE.search(t):
        return "junior"           # intern/working-student floor wins (even 'Lead Intern' is junior)
    if _ENGINEER_RE.search(t):
        # A lead/principal engineering ROLE is the head-not-hands work the user likes → not a repellent.
        return "lead" if is_lead else "hands_on_engineer"
    if is_lead:
        return "lead"
    return ""


def multiplier(kind: str, cfg: dict | None = None, con=None) -> float:
    """Multiplicative down-rank factor for the slate effective score (1.0 = no change).

    Default: the static `role_kind_mult` table (config) over `_DEFAULT_MULT`. When P2 learning is
    enabled (`slate.role_kind_learn.enabled`) AND `con` is supplied AND ≥ n_min golden role-fit votes
    exist for this kind, the factor is the **Beta-smoothed empirical role-fit rate** from `label_role`
    instead of the hardcoded constant — so a kind the user keeps tagging '🙅 wrong role' sinks toward
    `mult_floor` from data, not a magic number. Below n_min it falls back to the static value
    (graceful, behaviour-preserving). con=None or learning-off ⇒ exactly today's behaviour."""
    table = dict(_DEFAULT_MULT)
    if cfg:
        table.update((cfg.get("slate", {}) or {}).get("role_kind_mult", {}) or {})
    default = float(table.get(kind, 1.0))
    learn = ((cfg or {}).get("slate", {}) or {}).get("role_kind_learn", {}) or {}
    if con is None or not learn.get("enabled"):
        return default
    return _learned_multiplier(kind, default, learn, con)


def _learned_multiplier(kind: str, default: float, learn: dict, con) -> float:
    from . import role_feedback   # local import: avoid a cycle (role_feedback → db only)
    n_min = int(learn.get("n_min", 5))
    alpha = float(learn.get("alpha", 3.0))
    floor = float(learn.get("mult_floor", 0.5))
    n, fits = role_feedback.fit_counts(con, source="slate").get(kind, (0, 0))
    if n < n_min:
        return default                      # starved → the documented static fallback
    # prior anchored so a kind whose votes match the prior reproduces `default` (continuity at n_min).
    prior_p = (default - floor) / (1.0 - floor) if floor < 1.0 else 1.0
    p_fit = (fits + alpha * prior_p) / (n + alpha)        # Beta / α-shrink to the prior
    return floor + (1.0 - floor) * max(0.0, min(1.0, p_fit))


def flag(kind: str, cfg: dict | None = None) -> str:
    """Short card flag for a repellent role kind ('' for lead/neutral). The flag FOLLOWS the
    penalty: a user who doesn't down-rank this kind (multiplier ≥ 1.0) gets no «не твоё» label —
    it's user #1's verdict, not a fact about the job (multi-user fix)."""
    if cfg is not None and multiplier(kind, cfg) >= 1.0:
        return ""
    return _FLAG.get(kind, "")
