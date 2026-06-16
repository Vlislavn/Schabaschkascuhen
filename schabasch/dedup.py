"""Fuzzy cross-board dedup: RapidFuzz token_set_ratio + exoneration-first guard stack.

Pattern: JobFunnel filters.py (MIT, archived — patten copied, lib imported).
Logged-not-merged contract: near-dup candidates are logged to funnel_log for diagnostics;
NO vacancy status is mutated, NO automatic merge. Asymmetry is deliberate — a false merge
silently kills a live vacancy, which is worse than leaving a duplicate in the pipeline.
Same-source exact dedup is handled by the url UNIQUE constraint in upsert_vacancy.
"""
from __future__ import annotations

import json
from collections import defaultdict
from itertools import combinations

from rapidfuzz import fuzz

from . import db
from .models import Status, normalize_title

SIMILARITY_THRESHOLD = 88  # token_set_ratio ≥ this → near-dup candidate
_ACTIVE = (Status.NEW.value, Status.DESCRIBED.value, Status.NORMALIZED.value,
           Status.SCORED.value)


def _language_reality(card_json: str | None) -> str | None:
    if not card_json:
        return None
    try:
        return json.loads(card_json).get("language_reality")
    except (ValueError, TypeError):
        return None


def _disqualified(a: dict, b: dict) -> str | None:
    """Exoneration-first: return disqualifier reason if pair must NOT be flagged; else None.
    Guards checked cheapest-first. A surviving pair is logged, never auto-merged."""
    # Different AA refnr → distinct postings in the official system
    ra, rb = a.get("refnr"), b.get("refnr")
    if ra and rb and ra != rb:
        return "different_refnr"
    # Different city → different posting
    ca = (a.get("city") or "").lower().strip()
    cb = (b.get("city") or "").lower().strip()
    if ca and cb and ca != cb:
        return "different_city"
    # Different known language_reality → distinct roles
    lra = _language_reality(a.get("card_json"))
    lrb = _language_reality(b.get("card_json"))
    if lra and lrb and lra != lrb:
        return "different_language_reality"
    return None


def find_fuzzy_candidates(con, *, threshold: int = SIMILARITY_THRESHOLD) -> list[dict]:
    """Scan active vacancies for cross-source near-dup pairs. Returns candidate list.
    Does NOT mutate any row."""
    placeholders = ",".join("?" * len(_ACTIVE))
    rows = con.execute(
        f"SELECT id, source, refnr, title, company, city, card_json, dedup_key "
        f"FROM vacancy WHERE status IN ({placeholders})",
        _ACTIVE,
    ).fetchall()

    # Block by normalized company (first segment of dedup_key before '::')
    by_company: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        key_str = r["dedup_key"] or "::"
        co_key = key_str.split("::")[0]
        by_company[co_key].append(dict(r))

    candidates: list[dict] = []
    for co_key, group in by_company.items():
        if len(group) < 2:
            continue
        for a, b in combinations(group, 2):
            if a["source"] == b["source"]:
                continue  # same-source dupes handled by exact URL dedup
            t_a = normalize_title(a["title"] or "")
            t_b = normalize_title(b["title"] or "")
            if not t_a or not t_b:
                continue
            sim = fuzz.token_set_ratio(t_a, t_b)
            if sim < threshold:
                continue
            disq = _disqualified(a, b)
            if disq:
                continue
            candidates.append({
                "id_a": a["id"], "source_a": a["source"],
                "id_b": b["id"], "source_b": b["source"],
                "title_a": a["title"], "title_b": b["title"],
                "company": a["company"], "similarity": sim,
            })
    return candidates


def dedup_fuzzy(cfg: dict, con, *, threshold: int = SIMILARITY_THRESHOLD) -> dict:
    """Main entry: find + log near-dup candidates across sources. NO row mutations."""
    candidates = find_fuzzy_candidates(con, threshold=threshold)
    if candidates:
        detail = json.dumps(
            [{"a": c["id_a"], "sa": c["source_a"], "b": c["id_b"], "sb": c["source_b"],
              "sim": c["similarity"], "co": (c["company"] or "")[:40],
              "ta": c["title_a"][:50], "tb": c["title_b"][:50]}
             for c in candidates[:25]],
            ensure_ascii=False,
        )
    else:
        detail = "no near-dups found"
    db.log_funnel(con, "dedup_fuzzy", len(candidates), detail=detail)
    return {"candidates": len(candidates), "threshold": threshold}
