"""ReAct discovery crawler: finds jobs beyond fixed boards via kl_agent_builder agent.

Source contract: scrape(cfg, con, *, queries=None, max_results=20) -> dict[str,int].
Discovered postings enter the same funnel as board scrapes (DESCRIBED status → geo →
hardfilters → dedup → features → triage → normalize → judge → slate).

Requires kl_agent_builder + local ollama (qwen3:8b). Gracefully degrades on ImportError.
"""
from __future__ import annotations

import json
import logging
import re

from .. import agent_runtime, db
from ..candidate import load_candidate

logger = logging.getLogger(__name__)

# Aggregator SEARCH/LISTING URL shapes — NOT individual postings. qwen3:8b tends to return these
# (glassdoor SRCH_, linkedin '/jobs/role-jobs-city', stepstone '/jobs/role/in-city', etc.), which
# are dead-ends on the slate. Reject them so discovery only ever contributes real openings (or none).
_SEARCH_URL_RE = re.compile(
    r"SRCH_"
    r"|/jobs/[^/?]+-jobs-"
    r"|/jobs/[^/?]+/in-"
    r"|/in/[^/?]+/[^/?]+/?(?:$|\?)"
    r"|[?&](?:q|query|keywords?|suche)="   # NB: not bare 'k=' (would hit Indeed '?jk=')
    r"|/(?:search|suche|job-search)(?:/|\b)",
    re.I,
)
# 'company' values that are aggregator placeholders, not real employers
_PLACEHOLDER_CO_RE = re.compile(
    r"^\s*(?:linkedin|glassdoor|stepstone|indeed|xing|monster)\b"
    r"|\b(?:employer|partner)\s*$",
    re.I,
)


def _is_search_or_aggregator_url(url: str) -> bool:
    return bool(_SEARCH_URL_RE.search(url or ""))


def _is_placeholder_company(name: str) -> bool:
    return bool(_PLACEHOLDER_CO_RE.search(name or ""))

# German-rooted employers aligned to the candidate's magnets (space / defense-security / public /
# complex-systems). Targeting a real company's OWN careers page yields direct postings; free-text
# job search only returns aggregator listing pages (which the quality gate discards).
_TARGET_EMPLOYERS = (
    "Airbus, OHB SE, Rheinmetall, Hensoldt, Diehl Defence, DLR (Deutsches Zentrum für Luft- und "
    "Raumfahrt), Fraunhofer, Tesat-Spacecom, Jena-Optronik, Deutsche Bahn, Siemens, Bosch, ZEISS"
)

_SYSTEM_PROMPT = """You are a job-search agent for the following candidate:
{summary}
Your task: find currently-open postings on REAL EMPLOYERS' OWN careers pages. Target roles:
{target_roles}
Relevant skills/background: {skills_preview}
Locations: {locations}

How to work: pick 2–3 German-rooted employers relevant to the candidate (e.g. {employers}),
search for each one's official careers/jobs page (e.g. "Airbus careers", "OHB Stellenangebote"),
open it, and read off the actual open roles with their DIRECT posting links. For each, return a
JSON array:
[
  {{
    "title": "...",
    "company": "...",
    "url": "...",
    "city": "...",
    "description": "..."
  }},
  ...
]

Return ONLY the JSON array as your final answer. No prose, no markdown fences.
Find up to {max_results} distinct postings from different companies. Prefer companies
with English-language teams in Heidelberg / Frankfurt / hybrid.

URL RULES (strict — bad URLs are discarded):
- "url" MUST be a DIRECT link to ONE specific job posting (a page describing a single role you
  could click "apply" on), e.g. a company careers page like acme.com/careers/ml-engineer-1234.
- NEVER return a search-results or listing page. Reject URLs that contain: "SRCH_",
  "/jobs/<role>-jobs-<city>", "/jobs/<role>/in-<city>", "/in/<city>/<role>", "?q=", "?keywords=",
  "/search", "/suche". Those are dead-ends, not jobs.
- "company" MUST be the real employer's name (e.g. "Airbus", "Rheinmetall") — NEVER "LinkedIn",
  "Glassdoor", "Stepstone", "Indeed", or any "...Employer"/"...Partner" placeholder.
- If you cannot find a real direct posting, return an empty array []. Quality over quantity.

CRITICAL — finalize fast and stay cheap: use AT MOST ~2 searches and ~2 page fetches. After
you've opened a careers page or two, you MUST emit the final_answer with the JSON array — do not
keep browsing. If you found no real direct postings, return an empty array []. Never end without
a final_answer.
"""


def scrape(cfg: dict, con, *, queries: list[str] | None = None,
           max_results: int = 20) -> dict[str, int]:
    """Discover jobs via ReAct agent, upsert as source='agent' vacancies.

    Returns {"found": n, "upserted": n, "errors": n}.
    """
    # Build system prompt from candidate profile
    profile = load_candidate(con)
    prof_cfg = cfg.get("profile", {})
    if profile:
        summary = str(profile.get("summary") or "").strip()
        target_roles = ", ".join((profile.get("target_roles") or [])[:5])
        skills_preview = ", ".join((profile.get("skills") or [])[:8])
        locations = ", ".join((profile.get("locations") or ["Heidelberg", "Frankfurt"]))
    else:
        # Fall back to the configured profile/search, NOT a hardcoded persona.
        search_cfg = cfg.get("search", {})
        summary = str(prof_cfg.get("summary") or "").strip()
        target_roles = ", ".join((search_cfg.get("queries_en") or [])[:6])
        skills_preview = ", ".join(prof_cfg.get("magnets") or [])
        locations = ", ".join((search_cfg.get("cities") or ["Heidelberg", "Frankfurt"])[:3])

    # Allow overriding the target employers via config; else use the magnet-aligned default list.
    employers = ", ".join(cfg.get("search", {}).get("target_companies") or []) or _TARGET_EMPLOYERS

    system_prompt = _SYSTEM_PROMPT.format(
        summary=summary or "(see roles/skills below)",
        target_roles=target_roles or "(any relevant role)",
        skills_preview=skills_preview or "(see summary)",
        locations=locations,
        employers=employers,
        max_results=max_results,
    )

    task = (
        f"Pick 2–3 German-rooted employers relevant to the candidate (e.g. {employers}). For each, "
        f"open its official careers page and list up to {max_results} currently-open roles matching "
        f"({target_roles or 'the profile'}), near {locations}. Return a JSON array of postings, "
        f"each with the DIRECT url to that individual job. Real postings only — no listing pages."
    )

    try:
        agent_fn = agent_runtime.build_agent(cfg, system_prompt=system_prompt)
    except ImportError as e:
        logger.warning("kl_agent_builder not installed: %s", e)
        return {"found": 0, "upserted": 0, "errors": 1,
                "error": f"kl_agent_builder not installed: {e}"}

    # qwen3:8b ReAct occasionally ends a run with empty/non-JSON content. Retry ONCE (2 attempts)
    # before giving up — each attempt is now hard-bounded (turns/tool-calls/wall-time), so a flaky
    # empty doesn't waste the pass and a wanderer can't run away.
    postings = None
    last_err = "no output"
    for _attempt in range(2):
        try:
            raw = agent_runtime.run_agent(agent_fn, task)
            parsed = agent_runtime.parse_json_output(raw)
            if isinstance(parsed, list):
                postings = parsed
                break
            last_err = f"expected list, got {type(parsed).__name__}"
        except Exception as exc:
            last_err = str(exc)
        logger.warning("agent_discovery: attempt %d failed: %s", _attempt + 1, last_err)
    if postings is None:
        return {"found": 0, "upserted": 0, "errors": 1, "error": last_err}

    found = len(postings)
    upserted = 0
    errors = 0
    rejected = 0
    for posting in postings[:max_results]:
        if not isinstance(posting, dict):
            errors += 1
            continue
        url = (posting.get("url") or "").strip()
        company = (posting.get("company") or "").strip()
        if not url:
            errors += 1
            continue
        # QUALITY GATE: drop aggregator search/listing pages and placeholder employer names so the
        # slate only ever gets real openings from discovery. Zero good > five dead search pages.
        if _is_search_or_aggregator_url(url) or _is_placeholder_company(company):
            rejected += 1
            logger.info("agent_discovery: rejected non-posting url=%s company=%r", url, company)
            continue
        try:
            db.upsert_vacancy(con, {
                "source": "agent",
                "url": url,
                "title": str(posting.get("title") or "")[:200],
                "company": company[:200] or None,
                "city": str(posting.get("city") or "")[:100] or None,
                "description": str(posting.get("description") or "") or None,
            })
            upserted += 1
        except Exception as exc:
            logger.warning("agent_discovery: upsert failed for %s: %s", url, exc)
            errors += 1

    return {"found": found, "upserted": upserted, "rejected": rejected, "errors": errors}
