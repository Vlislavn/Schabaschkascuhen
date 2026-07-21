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
from .browsing import entity as _entity
from .llm import http_get_json, http_get_status
from .models import normalize_company

logger = logging.getLogger(__name__)

# German legal entity suffixes — a strong, deterministic "rooted in Germany" signal for the
# integration score: укоренённые в Германии — да; чисто американские стартапы — нет.
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


# Name-match guard for the Wikipedia-REST fallback lives in the browsing layer now (re-exported here
# so investigate._name_matches and the _wikipedia_company guard keep working). Wikidata's typed entity
# resolution (browsing.entity.resolve) is the PRIMARY disambiguation; this guards the fallback path.
_name_matches = _entity._name_matches
_significant_tokens = _entity._significant_tokens


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
        if not _name_matches(name, title):
            return None   # opensearch fuzzy-matched a DIFFERENT entity (Terma→Therme) → reject early
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
    """Cross-validate the company on KNOWN keyless sources — independent of the qwen agent's claims.
    Ladder (hard-before-soft): (1) Wikidata typed entity resolution (SOTA disambiguation — a typed,
    name-matched entity can't return "Therme" for "Terma"); (2) Wikipedia REST + name-match guard;
    (3) German legal-suffix signal. Returns {company_description, german_rooted, company_verified,
    validation_source, wiki_url} plus the richer Wikidata facts (official_site, country, employees,
    inception, wikidata_qid) when available. Cached per normalized company name within a run."""
    cache = cache if cache is not None else {}
    key = normalize_company(name)
    if key in cache:
        return cache[key]
    # deterministic german-rooted: a German legal-entity suffix in the posted company name.
    german_rooted = bool(name and _GERMAN_LEGAL.search(name))

    # (1) Wikidata typed entity resolution — the authoritative, disambiguated source.
    ent = _entity.resolve(name) if name else None
    if ent:
        country = ent.get("country") or ""
        german_rooted = german_rooted or "german" in country.lower() or country.lower() == "germany"
        out = {
            "company_description": ent.get("description")
                or str(agent_enrichment.get("company_description") or "").strip(),
            "german_rooted": german_rooted,
            "company_verified": True,
            "validation_source": f"wikidata:{ent['qid']}",
            "wiki_url": ent.get("wikidata_url"),
            "official_site": ent.get("official_site"),
            "country": country or None,
            "employees": ent.get("employees"),
            "inception": ent.get("inception"),
            "wikidata_qid": ent.get("qid"),
        }
        cache[key] = out
        return out

    # (2) Wikipedia REST fallback (guarded by _name_matches) → (3) legal-suffix signal.
    wiki = _wikipedia_company(name, lang="de") or _wikipedia_company(name, lang="en") if name else None
    if wiki:
        german_rooted = german_rooted or bool(_GERMAN_HINT.search(wiki["extract"]))
    description = (wiki["extract"] if wiki else
                  str(agent_enrichment.get("company_description") or "").strip())
    out = {
        "company_description": description,
        "german_rooted": german_rooted,
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

# Persistent EMPLOYER knowledge base (keyed by normalize_company). Research a company ONCE and reuse
# the DURABLE facts on every future vacancy + tick instead of re-running the agent per posting; only
# the FRESH signals (recent news / funding / reputation) are re-fetched past a TTL. Additive sidecar.
_SCHEMA_COMPANY = """
CREATE TABLE IF NOT EXISTS company_knowledge (
    company_key   TEXT PRIMARY KEY,
    display_name  TEXT,
    facts_json    TEXT NOT NULL,
    facts_at      TEXT NOT NULL,
    news_json     TEXT,
    news_at       TEXT
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
    con.execute(_SCHEMA_COMPANY)
    con.commit()


def _age_days(ts: str | None, now: str) -> float:
    """Days between ISO `ts` and ISO `now`; +inf when `ts` is missing/unparseable (→ treat as stale,
    so a missing record always refreshes)."""
    if not ts:
        return float("inf")
    try:
        a, b = datetime.fromisoformat(ts), datetime.fromisoformat(now)
    except ValueError:
        return float("inf")
    if a.tzinfo is None:
        a = a.replace(tzinfo=timezone.utc)
    if b.tzinfo is None:
        b = b.replace(tzinfo=timezone.utc)
    return (b - a).total_seconds() / 86400.0


def get_company_knowledge(con, company_key: str) -> dict | None:
    """Durable employer record (facts + cached fresh news) by normalized company key, or None."""
    _ensure_schema(con)
    row = con.execute(
        "SELECT company_key, display_name, facts_json, facts_at, news_json, news_at "
        "FROM company_knowledge WHERE company_key = ?", (company_key,),
    ).fetchone()
    if not row:
        return None
    try:
        facts = json.loads(row["facts_json"])
    except (TypeError, json.JSONDecodeError):
        facts = {}
    try:
        news = json.loads(row["news_json"]) if row["news_json"] else None
    except (TypeError, json.JSONDecodeError):
        news = None
    return {"company_key": row["company_key"], "display_name": row["display_name"],
            "facts": facts, "facts_at": row["facts_at"], "news": news, "news_at": row["news_at"]}


def upsert_company_facts(con, company_key: str, display_name: str, facts: dict, now: str) -> None:
    """Store/refresh the DURABLE employer facts (researched once, reused across vacancies + ticks)."""
    con.execute(
        """INSERT INTO company_knowledge (company_key, display_name, facts_json, facts_at)
           VALUES (?, ?, ?, ?)
           ON CONFLICT (company_key) DO UPDATE SET
               display_name = excluded.display_name,
               facts_json   = excluded.facts_json,
               facts_at     = excluded.facts_at""",
        (company_key, display_name, json.dumps(facts, ensure_ascii=False), now),
    )
    con.commit()


def upsert_company_news(con, company_key: str, news: dict, now: str) -> None:
    """Store/refresh the FRESH employer signals (recent news / funding / reputation). Requires the
    facts row to exist first (the caller upserts facts before news)."""
    con.execute(
        "UPDATE company_knowledge SET news_json = ?, news_at = ? WHERE company_key = ?",
        (json.dumps(news, ensure_ascii=False), now, company_key),
    )
    con.commit()


def investigate_top(cfg: dict, con, *, slate_date: str, top_n: int = 5) -> dict[str, int]:
    """Deep-dive the top-N SCORED vacancies, store investigation results.

    Also patches vacancy_feature with requirements_verified / company_known
    so aspects.score can use the richer data next time features are recomputed.

    Returns {"investigated": n, "ok": n, "stale": n, "suspect": n, "errors": n}.
    """
    _ensure_schema(con)

    # Load top-N by latest judge score (SCORED status, highest score first) that are NOT yet
    # investigated — the agent is the single most expensive op, so it must never re-run on a vacancy
    # already in the `investigation` table (mirrors investigate_one's idempotency guard). A closed
    # job already has an investigation row, so this also stops re-investigating closed listings.
    rows = con.execute(
        """SELECT v.id, v.title, v.company, v.url, v.description, v.refnr, v.source,
                  js.score
           FROM vacancy v
           JOIN (SELECT vacancy_id, MAX(id) mid FROM judge_score GROUP BY vacancy_id) m
             ON m.vacancy_id = v.id
           JOIN judge_score js ON js.id = m.mid
           LEFT JOIN investigation inv ON inv.vacancy_id = v.id
           WHERE v.status = ? AND inv.vacancy_id IS NULL
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
        verdict = _investigate_row(con, row, cfg=cfg, agent_fn=agent_fn,
                                   company_cache=company_cache, now=now)
        if verdict is None:
            counts["errors"] += 1
            continue
        counts["investigated"] += 1
        counts[verdict] = counts.get(verdict, 0) + 1

    con.commit()
    return counts


def _investigate_row(con, row, *, cfg: dict | None, agent_fn, company_cache: dict, now: str) -> str | None:
    """Run the ReAct agent on one vacancy row, store the result. Returns the verdict or None on
    failure. Shared by investigate_top (batch) and investigate_one (progressive/on-demand).

    Reuses the persistent employer knowledge base: a company researched within company_facts_ttl_days
    is NOT re-researched (the agent is told to skip company research and the durable facts are pulled
    from `company_knowledge`); only the FRESH news/reputation signal is re-asked past its own TTL."""
    vid = row["id"]
    url = row["url"] or ""
    title = row["title"] or ""
    company = row["company"] or ""
    desc_snippet = (row["description"] or "")[:500]

    inv_cfg = (cfg or {}).get("investigate", {})
    key = normalize_company(company) if company else ""
    ck = get_company_knowledge(con, key) if company else None
    # "Known" = a FRESH record that actually carries validated facts (a description or verification).
    # A thin agent-signals-only record (e.g. a backfilled niche employer) does NOT suppress research —
    # it gets a proper, cross-checked description from this run instead.
    facts_fresh = (ck is not None
                   and _age_days(ck["facts_at"], now) <= float(inv_cfg.get("company_facts_ttl_days", 90))
                   and bool(ck["facts"].get("company_description") or ck["facts"].get("company_verified")))
    news_enabled = bool(inv_cfg.get("company_news_enabled", True))
    news_due = bool(company) and news_enabled and _age_days((ck or {}).get("news_at"), now) > float(
        inv_cfg.get("company_news_refresh_days", 30))

    # HARD-BEFORE-SOFT: ground the employer IDENTITY with the keyless ladder (Wikidata typed entity →
    # Wikipedia REST → legal-suffix) BEFORE the agent and feed it in — the agent stops GUESSING who the
    # company is and spends its tight budget on THIS posting + fresh signals. Reuse the KB when fresh.
    if facts_fresh:
        ground = ck["facts"]
    else:
        ground = validate_company(company, {}, cache=company_cache)   # authoritative-only, pre-agent
        if company:
            upsert_company_facts(con, key, company, ground, now)
    ident = str(ground.get("company_description") or "").strip()

    # Free-text CONTENT (the job-description preview, resolved employer facts) rides in `context`, NOT
    # in `task`. kl's facet-coverage gate (runtime/loop/react.py::_check_facet_coverage) scans ONLY
    # `task`, and an incidental word there — 'current'/'today'/'now' or a bare year — is mis-extracted
    # as a required research facet ('Fetch current data'/'Fetch data for 2024') that no observation
    # covers, so it SILENTLY BLOCKS an already-valid final_answer and the run grinds to
    # max_turns_exhausted. MEASURED 2026-06-21: all 5 of the day's failing slate cards carried
    # 'today'/'currently' in the description preview; the 35B emitted a correct final_answer on turn 5
    # that the gate discarded. Keep `task` to terse INSTRUCTIONS; put prose in `context`.
    context: dict = {"job_description_preview": desc_snippet}
    parts = ["Investigate this job posting (the description preview is in the Context below):",
             f"Title: {title}\nCompany: {company}\nURL: {url}\n"]
    if ident or ground.get("official_site"):   # we have an authoritative identity → don't re-research it
        context["verified_employer"] = {"name": company, **{k: ground[k] for k in
            ("company_description", "official_site", "country") if ground.get(k)}}
        parts.append(
            "VERIFIED EMPLOYER (authoritative, already resolved — see Context.verified_employer; do "
            "NOT re-research the company; CONFIRM this posting matches it, then focus on the posting's "
            "must-have requirements and English/remote signals).")
    if news_due:      # fold the fresh-signal ask into THIS single agent call (no second run)
        parts.append(
            'ALSO add to your JSON: "recent_news" (1-2 sentences on recent funding/layoffs/branch '
            'news, or empty), "reputation" (loved|mixed|disliked|unknown), "reputation_note" '
            '(1 sentence on employee sentiment, e.g. Kununu/Glassdoor, or empty).')
    parts.append("Fetch the URL and company career page, then return the JSON object.")
    task = "\n\n".join(parts)

    # qwen3:8b occasionally exhausts the turn budget without finalizing — retry ONCE
    # (2 attempts). Each run is hard-bounded (turns/tool-calls/wall-time), so a job that won't
    # finalize is skipped quickly instead of burning ~3 long runs.
    enrichment = None
    last_err = "no output"
    for _attempt in range(2):
        try:
            raw = agent_runtime.run_agent(agent_fn, task, context)
            parsed = agent_runtime.parse_json_output(raw)
            if isinstance(parsed, dict):
                enrichment = parsed
                break
            last_err = f"expected dict, got {type(parsed).__name__}"
        except Exception as exc:
            last_err = str(exc)
        logger.warning("investigate: vacancy %s attempt %d: %s", vid, _attempt + 1, last_err)
    if enrichment is None:
        return None

    # deterministic open/closed/unknown (never a model guess) — overrides any agent field.
    enrichment["still_open"] = _check_still_open(url, row["source"], row["refnr"])
    # Durable identity was resolved hard-before-soft (above); merge it into the per-vacancy record.
    # An empty authoritative description does NOT clobber the agent's card text (niche employers off
    # Wikidata keep the agent's prose for display, while company_knowledge stays authoritative-only).
    val = ground
    enrichment.update({k: val.get(k) for k in
                       ("company_description", "german_rooted", "company_verified", "validation_source",
                        "wiki_url", "official_site", "country", "employees", "inception", "wikidata_qid")
                       if val.get(k) not in (None, "")})
    # FRESH employer signals (news/reputation): persist when re-asked, else reuse the cached record.
    if news_due:
        news = {k: enrichment[k] for k in ("recent_news", "reputation", "reputation_note")
                if enrichment.get(k) not in (None, "")}
        if news:
            upsert_company_news(con, key, news, now)
            enrichment["company_news"] = news
    elif ck and ck.get("news"):
        enrichment["company_news"] = ck["news"]
    verdict = enrichment.get("verdict", "ok")
    if verdict not in ("ok", "suspect"):
        verdict = "ok"   # closure is deterministic now (still_open), not a 'stale' agent guess
    con.execute(
        """INSERT OR REPLACE INTO investigation
           (vacancy_id, enrichment_json, verdict, investigated_at)
           VALUES (?, ?, ?, ?)""",
        (vid, json.dumps(enrichment, ensure_ascii=False), verdict, now),
    )
    _patch_feature_row(con, vid, enrichment)   # propagate verified signals to vacancy_feature
    return verdict


def investigate_one(cfg: dict, con, vacancy_id: int, *, agent_fn=None, force: bool = False) -> str:
    """Investigate ONE vacancy (progressive / on-demand). Idempotent: returns 'cached' when an
    investigation row already exists (so the serve-start seed-2-then-rest pass never re-does a card).
    Builds the agent lazily if not supplied. Returns the verdict | 'cached' | 'missing' | 'error'."""
    _ensure_schema(con)
    if not force and con.execute(
            "SELECT 1 FROM investigation WHERE vacancy_id = ?", (vacancy_id,)).fetchone():
        return "cached"
    row = con.execute(
        "SELECT id, title, company, url, description, refnr, source FROM vacancy WHERE id = ?",
        (vacancy_id,),
    ).fetchone()
    if not row:
        return "missing"
    if agent_fn is None:
        try:
            agent_fn = agent_runtime.build_agent(cfg, system_prompt=_SYSTEM_PROMPT)
        except ImportError as e:
            logger.warning("investigate_one: kl_agent_builder not installed: %s", e)
            return "error"
    verdict = _investigate_row(con, row, cfg=cfg, agent_fn=agent_fn, company_cache={},
                               now=datetime.now(timezone.utc).isoformat())
    con.commit()
    return verdict or "error"


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
