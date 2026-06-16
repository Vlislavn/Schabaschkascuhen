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

# Default multipliers (config: slate.role_kind_mult). Soft, recall-first — measured on real labels.
_DEFAULT_MULT = {"hands_on_engineer": 0.7, "junior": 0.5, "lead": 1.0, "": 1.0}

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


def multiplier(kind: str, cfg: dict | None = None) -> float:
    """Multiplicative down-rank factor for the slate effective score (1.0 = no change)."""
    table = dict(_DEFAULT_MULT)
    if cfg:
        table.update((cfg.get("slate", {}) or {}).get("role_kind_mult", {}) or {})
    return float(table.get(kind, 1.0))


def flag(kind: str) -> str:
    """Short card flag for a repellent role kind ('' for lead/neutral — no flag)."""
    return _FLAG.get(kind, "")
