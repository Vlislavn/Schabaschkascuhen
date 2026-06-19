"""Backfill / refresh the persistent employer DB (`company_knowledge`) using the PRODUCTION grounding
ladder (Wikidata typed entity → Wikipedia REST, name-guarded → legal-suffix).

Every per-vacancy `investigation` row already carries the agent's durable employer signals
(company_size / english_team_signal / is_temp_agency). This one-shot lifts those and re-grounds each
employer's IDENTITY through `investigate.validate_company` — so the partial employers gain canonical,
correctly-disambiguated facts (Terma → Danish defence company, not "Therme"; rich official_site /
country / employees / inception from Wikidata). Idempotent (upsert ON CONFLICT), additive (only the
new sidecar), keyless, no model. Back up the DB first.

Run: .venv/bin/python -m scripts.backfill_company_knowledge
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

from schabasch import config, db, investigate
from schabasch.models import normalize_company

# Agent durable employer signals to keep alongside the re-grounded identity facts.
_AGENT_DURABLE = ("company_size", "english_team_signal", "is_temp_agency")
_POLITE_S = 1.0   # pause between employers — be kind to the keyless Wikidata/Wikipedia APIs


def main() -> None:
    cfg = config.load()
    con = db.connect(cfg["paths"]["db"])
    investigate._ensure_schema(con)
    now = datetime.now(timezone.utc).isoformat()   # facts validated NOW → TTL counts from here

    rows = con.execute(
        """SELECT v.company AS company, i.enrichment_json AS enrichment_json
           FROM investigation i JOIN vacancy v ON v.id = i.vacancy_id
           WHERE v.company IS NOT NULL AND TRIM(v.company) != ''
           ORDER BY i.investigated_at DESC""",   # most-recent first → most-recent agent signals win
    ).fetchall()

    cache: dict = {}
    seen: set[str] = set()
    verified = partial = skipped_dup = 0
    for r in rows:
        company = r["company"]
        key = normalize_company(company)
        if not key or key in seen:
            skipped_dup += 1
            continue
        seen.add(key)
        try:
            enr = json.loads(r["enrichment_json"]) if r["enrichment_json"] else {}
        except (TypeError, json.JSONDecodeError):
            enr = {}
        agent_facts = {k: enr[k] for k in _AGENT_DURABLE if enr.get(k) is not None}
        val = investigate.validate_company(company, enr, cache=cache)   # Wikidata-first keyless ladder
        facts = {**agent_facts, **{k: v for k, v in val.items() if v is not None}}
        if not facts:
            continue
        investigate.upsert_company_facts(con, key, company, facts, now)
        src = str(facts.get("validation_source") or "")
        if facts.get("company_description") and src.startswith("wikidata"):
            verified += 1
        else:
            partial += 1
        time.sleep(_POLITE_S)

    total = con.execute("SELECT COUNT(*) n FROM company_knowledge").fetchone()["n"]
    print(f"re-grounded employers: {verified + partial} "
          f"(canonical Wikidata identity: {verified}; partial/fallback: {partial}) | "
          f"dup-key skipped: {skipped_dup}")
    print(f"company_knowledge now holds {total} employers")
    con.close()


if __name__ == "__main__":
    main()
