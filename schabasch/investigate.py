"""ReAct top-match investigator: deep-dives the top-N SCORED vacancies.

Verifies:  still open? real employer (not Zeitarbeit)? culture signals?
Extracts:  verified requirements text, company facts, salary.
Enriches:  re-runs aspects.score coverage on agent-verified requirements,
           sets requirements_verified=1/company_known in vacancy_feature.

Sidecar table: investigation (self-created).
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import re

from . import agent_runtime, db
from .llm import http_get_json, http_get_status
from .models import normalize_company

logger = logging.getLogger(__name__)

# German legal entity suffixes — a strong, deterministic "rooted in Germany" signal (matters for her
# integration score: укоренённые в Германии — да; чисто американские стартапы — нет).
_GERMAN_LEGAL = re.compile(r"\b(gmbh|mbh|\bag\b|\bse\b|\bkg\b|kgaa|\bohg\b|\bug\b|gbr|e\.?\s?v|ev\b)\b", re.I)
_GERMAN_HINT = re.compile(r"(deutschland|deutsch\w*|german\w*|germany|münchen|munich|berlin|"
                          r"frankfurt|hamburg|stuttgart|köln|cologne|düsseldorf|mannheim|heidelberg)", re.I)


# Wikimedia blocks requests without a descriptive User-Agent (HTTP 403) — set one per their policy.
_WIKI_UA = {"User-Agent": "Schabaschkascuhen-jobmatcher/1.0 (personal job-search tool; local use)"}
# A bare/short company name (e.g. "SCHOTT") fuzzy-matches the wrong article ("Schottland"=Scotland).
# Accept a Wikipedia hit only if its intro reads like a COMPANY/ORG — else it's a place/person/etc.
_ORG_INDICATOR = re.compile(
    r"(unternehmen|konzern|\bgmbh\b|\bag\b|\bse\b|\bkg\b|hersteller|dienstleister|firma|corporation|"
    r"\bcompany\b|\bfirm\b|manufacturer|provider|organi[sz]ation|\bagency\b|betreibt|gegründet|"
    r"founded|startup|\bbank\b|versicherun|holding|software|technolog|conglomerate)", re.I)


def _wikipedia_company(name: str, *, lang: str = "de") -> dict | None:
    """Keyless authoritative cross-check on a KNOWN site (Wikipedia). opensearch → article title →
    REST summary intro. Returns {found, title, extract, url, lang} or None (no article / network /
    any HTTP error — degrades gracefully so a Wikipedia hiccup never aborts the investigate run)."""
    base = f"https://{lang}.wikipedia.org"
    try:
        js = http_get_json(base + "/w/api.php", headers=_WIKI_UA,
                           params={"action": "opensearch", "search": name, "limit": 1,
                                   "namespace": 0, "format": "json"}, timeout_s=8)
        if not isinstance(js, list) or len(js) < 2 or not js[1]:
            return None
        title = js[1][0]
        desc = js[2][0] if len(js) > 2 and js[2] else ""
        sm = http_get_json(base + "/api/rest_v1/page/summary/" + title.replace(" ", "_"),
                           headers=_WIKI_UA, timeout_s=8)
    except Exception as exc:   # network / HTTP error → no validation, not a crash
        logger.debug("wikipedia lookup failed for %r (%s): %s", name, lang, exc)
        return None
    if (sm or {}).get("type") == "disambiguation":
        return None   # ambiguous bare name ("Bosch" → a disambiguation list) → not a clean match
    extract = (sm or {}).get("extract") or desc
    if not extract or not _ORG_INDICATOR.search(extract):
        return None   # no article, or the match is a place/person/etc. (not a company) → don't trust it
    url = ((sm or {}).get("content_urls", {}).get("desktop", {}) or {}).get("page")
    return {"found": True, "title": title, "extract": extract[:600], "url": url, "lang": lang}


def validate_company(name: str, agent_enrichment: dict, *, cache: dict | None = None) -> dict:
    """Cross-validate the company on KNOWN sites (Wikipedia, keyless; + the legal-suffix registry
    signal) — independent of the qwen agent's claims. Returns {company_description, german_rooted,
    company_verified, validation_source}. Cached per normalized company name within a run."""
    cache = cache if cache is not None else {}
    key = normalize_company(name)
    if key in cache:
        return cache[key]
    # deterministic german-rooted: a German legal-entity suffix in the posted company name.
    german_rooted = bool(name and _GERMAN_LEGAL.search(name))
    wiki = _wikipedia_company(name, lang="de") or _wikipedia_company(name, lang="en") if name else None
    if wiki:
        german_rooted = german_rooted or bool(_GERMAN_HINT.search(wiki["extract"]))
    # authoritative description: prefer Wikipedia (a known site) over the agent's free text.
    description = (wiki["extract"] if wiki else
                  str(agent_enrichment.get("company_description") or "").strip())
    out = {
        "company_description": description,
        "german_rooted": german_rooted,
        # verified = an independent KNOWN source confirms the employer exists (Wikipedia) OR it has a
        # German legal entity (registry-style signal). Absence isn't proof of fakeness (small firms).
        "company_verified": bool(wiki) or german_rooted,
        "validation_source": ("wikipedia:" + wiki["lang"]) if wiki else ("legal-suffix" if german_rooted else "none"),
        "wiki_url": (wiki or {}).get("url"),
    }
    cache[key] = out
    return out


def _check_still_open(url: str, source: str, refnr: str | None) -> bool | None:
    """DETERMINISTIC open/closed/unknown — never a model guess (the qwen agent can't see anti-bot
    -blocked links and wrongly reports 'closed'). Arbeitsagentur: re-query the API by refnr. Boards:
    HTTP status — 404/410 → closed (False); 2xx/3xx → open (True); 403/429/5xx/timeout → unknown
    (None, 'couldn't verify', NOT closed)."""
    if source == "arbeitsagentur" and refnr:
        from .sources import arbeitsagentur
        return arbeitsagentur.check_open(refnr)
    if not url:
        return None
    status = http_get_status(url)
    if status is None:
        return None
    if status in (404, 410):
        return False
    if 200 <= status < 400:
        return True
    return None

_SCHEMA = """
CREATE TABLE IF NOT EXISTS investigation (
    vacancy_id       INTEGER PRIMARY KEY,
    enrichment_json  TEXT NOT NULL,
    verdict          TEXT,
    investigated_at  TEXT NOT NULL
)
"""

# NOTE: the agent does NOT judge whether the posting is still open — that's checked
# deterministically (HTTP status / Arbeitsagentur API re-query, see _check_still_open) because a
# blocked link made the model guess "closed" and emit false "no active listing" notes. The agent
# only does qualitative enrichment (employer/skills/culture); still_open is overridden after.
_SYSTEM_PROMPT = """You are a job-search research agent. Given a job posting, your task is to:
1. Visit the company's career page / official website AND search a KNOWN reference (Wikipedia; for a
   German company, the Handelsregister / company registry) to confirm the employer is real.
2. Verify: Is it a real employer (not a temp agency / Zeitarbeit)? Is it ROOTED IN GERMANY
   (German-headquartered / German legal entity), or a foreign company just hiring here?
3. Extract: the exact required skills/qualifications (must-haves), company size/type, salary range,
   and a 1-2 sentence factual DESCRIPTION of what the company does (from the official site / Wikipedia).
4. Assess culture: any signals of English-language work environment, international team, remote/hybrid?

Do NOT judge whether the job is still open — that is verified separately.
Return a single JSON object (no prose, no fences):
{
  "is_temp_agency": true/false,
  "company_known": true/false,
  "company_size": "startup|mid|large|unknown",
  "company_description": "1-2 sentence factual description of the company (from its site / Wikipedia)",
  "german_rooted": true/false,
  "verified_requirements": "exact text of must-have requirements from the actual posting",
  "salary_eur_min": null or integer,
  "salary_eur_max": null or integer,
  "english_team_signal": true/false,
  "verdict": "ok" | "suspect",
  "notes": "one-line summary (note if the official site / Wikipedia disagrees with the posting)"
}
"""


def _ensure_schema(con) -> None:
    con.execute(_SCHEMA)
    con.commit()


def investigate_top(cfg: dict, con, *, slate_date: str, top_n: int = 5) -> dict[str, int]:
    """Deep-dive the top-N SCORED vacancies, store investigation results.

    Also patches vacancy_feature with requirements_verified / company_known
    so aspects.score can use the richer data next time features are recomputed.

    Returns {"investigated": n, "ok": n, "stale": n, "suspect": n, "errors": n}.
    """
    _ensure_schema(con)

    # Load top-N by latest judge score (SCORED status, highest score first)
    rows = con.execute(
        """SELECT v.id, v.title, v.company, v.url, v.description, v.refnr, v.source,
                  js.score
           FROM vacancy v
           JOIN (SELECT vacancy_id, MAX(id) mid FROM judge_score GROUP BY vacancy_id) m
             ON m.vacancy_id = v.id
           JOIN judge_score js ON js.id = m.mid
           WHERE v.status = ?
           ORDER BY js.score DESC
           LIMIT ?""",
        (db.Status.SCORED.value if hasattr(db, "Status") else "SCORED", top_n),
    ).fetchall()

    if not rows:
        return {"investigated": 0, "ok": 0, "stale": 0, "suspect": 0, "errors": 0}

    try:
        agent_fn = agent_runtime.build_agent(
            cfg, system_prompt=_SYSTEM_PROMPT
        )
    except ImportError as e:
        logger.warning("investigate: kl_agent_builder not installed: %s", e)
        return {"investigated": 0, "ok": 0, "stale": 0, "suspect": 0, "errors": 1}

    counts = {"investigated": 0, "ok": 0, "stale": 0, "suspect": 0, "errors": 0}
    now = datetime.now(timezone.utc).isoformat()
    company_cache: dict = {}   # validate each company once per run (Wikipedia/registry)

    for row in rows:
        vid = row["id"]
        url = row["url"] or ""
        title = row["title"] or ""
        company = row["company"] or ""
        desc_snippet = (row["description"] or "")[:500]

        task = (
            f"Investigate this job posting:\n"
            f"Title: {title}\nCompany: {company}\nURL: {url}\n\n"
            f"Description preview:\n{desc_snippet}\n\n"
            f"Fetch the URL and company career page, then return the JSON object."
        )

        # qwen3:8b occasionally exhausts the turn budget without finalizing — retry ONCE
        # (2 attempts). Each run is hard-bounded (turns/tool-calls/wall-time), so a job that won't
        # finalize is skipped quickly instead of burning ~3 long runs.
        enrichment = None
        last_err = "no output"
        for _attempt in range(2):
            try:
                raw = agent_runtime.run_agent(agent_fn, task)
                parsed = agent_runtime.parse_json_output(raw)
                if isinstance(parsed, dict):
                    enrichment = parsed
                    break
                last_err = f"expected dict, got {type(parsed).__name__}"
            except Exception as exc:
                last_err = str(exc)
            logger.warning("investigate: vacancy %s attempt %d: %s", vid, _attempt + 1, last_err)
        if enrichment is None:
            counts["errors"] += 1
            continue

        # deterministic open/closed/unknown (never a model guess) — overrides any agent field.
        enrichment["still_open"] = _check_still_open(url, row["source"], row["refnr"])
        # AUTHORITATIVE company validation on a KNOWN site (Wikipedia, keyless) + the legal-suffix
        # registry signal — independent of the agent's claims; fills the deeper description + the
        # german_rooted integration signal, and confirms the employer exists.
        val = validate_company(company, enrichment, cache=company_cache)
        enrichment.update({k: val[k] for k in
                           ("company_description", "german_rooted", "company_verified",
                            "validation_source", "wiki_url")})
        verdict = enrichment.get("verdict", "ok")
        if verdict not in ("ok", "suspect"):
            verdict = "ok"   # closure is deterministic now (still_open), not a 'stale' agent guess
        con.execute(
            """INSERT OR REPLACE INTO investigation
               (vacancy_id, enrichment_json, verdict, investigated_at)
               VALUES (?, ?, ?, ?)""",
            (vid, json.dumps(enrichment, ensure_ascii=False), verdict, now),
        )

        # Patch vacancy_feature with enrichment signals
        _patch_feature_row(con, vid, enrichment)

        counts["investigated"] += 1
        counts[verdict] = counts.get(verdict, 0) + 1

    con.commit()
    return counts


def _patch_feature_row(con, vacancy_id: int, enrichment: dict) -> None:
    """Update vacancy_feature with verified signals from investigation."""
    try:
        row = con.execute(
            "SELECT feature_json FROM vacancy_feature WHERE vacancy_id = ?",
            (vacancy_id,),
        ).fetchone()
        if not row:
            return
        feat = json.loads(row["feature_json"])
        # requirements_verified = the agent actually returned verified requirements text — NOT
        # whether the posting is still open (orthogonal signals; conflating them mislabels the
        # gate's training feature). The contract key is 'verified_requirements' (system prompt
        # above); still_open is captured by the investigation verdict instead.
        feat["requirements_verified"] = (
            1.0 if str(enrichment.get("verified_requirements") or "").strip() else 0.0)
        # company_known = independently verified (Wikipedia/registry), not just the agent's claim.
        feat["company_known"] = 1.0 if enrichment.get("company_verified") else 0.0
        # german_rooted → integration signal (укоренена в Германии = путь к гражданству/языку).
        feat["german_rooted"] = 1.0 if enrichment.get("german_rooted") else 0.0
        salary_min = enrichment.get("salary_eur_min")
        if salary_min is not None:
            feat["salary_vs_target_gap"] = 0.0  # placeholder; real gap = (target - salary) / target
        con.execute(
            "UPDATE vacancy_feature SET feature_json = ? WHERE vacancy_id = ?",
            (json.dumps(feat), vacancy_id),
        )
    except (sqlite3.Error, TypeError, ValueError, json.JSONDecodeError) as exc:
        # best-effort enrichment patch: a malformed feature row must not abort the investigate
        # run, but the skip is RECORDED (funnel), not silently dropped.
        logger.debug("investigate: patch feature failed for %s: %s", vacancy_id, exc)
        db.log_funnel(con, "investigate_patch_failed", 1, detail=f"{vacancy_id}: {str(exc)[:160]}")
