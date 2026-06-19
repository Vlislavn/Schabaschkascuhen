"""Keyless company ENTITY RESOLUTION via the official Wikidata API (wbsearchentities + wbgetentities).

Adapter over Wikidata's public JSON API. Typed entities + aliases resolve a company NAME to its
canonical entity far more reliably than Wikipedia opensearch, which fuzzy-matched ambiguous names to
the wrong article ("Terma Group" → "Therme Group" spa; "Phoenix Medical" → "Phoenix Media"). Wikidata
search ranks by label+aliases, and a one-line `description` reliably states the entity TYPE, so we
keep only hits that are a company/organisation AND whose label matches the query. Returns rich,
structured facts (official site, country, employees, inception) for the employer knowledge base.

Keyless, graceful-degrade (``None`` on any failure — never raises). `schabasch` depends on
``resolve()``, not on the API shape.

Ref: Wikidata API action=wbsearchentities / wbgetentities; inspire cwrc/wikidata-entity-lookup,
salmon-kg/llm-sparql.
"""
from __future__ import annotations

import logging
import re

from ..llm import http_get_json

logger = logging.getLogger(__name__)

_WD_API = "https://www.wikidata.org/w/api.php"
# Wikidata/Wikimedia returns 403 without a descriptive User-Agent (per their API policy).
_WD_UA = {"User-Agent": "Schabaschkascuhen-jobmatcher/1.0 (personal job-search tool; local use)"}

# A Wikidata one-line description that names the right KIND of entity (company/organisation), not a
# place/person/brand/product. wbsearchentities descriptions are concise and reliably state the type.
_ORG_DESC = re.compile(
    r"(compan|business|enterprise|corporation|\bfirm\b|manufacturer|producer|provider|"
    r"organi[sz]ation|organisation|\bagency\b|institut|conglomerate|\bgroup\b|holding|startup|"
    r"\bbank\b|insurer|insurance|consultanc|consulting|laborator|foundation|"
    r"unternehmen|konzern|hersteller|dienstleister|gesellschaft|stiftung|behörde)", re.I)

# Generic / legal tokens that don't identify a specific employer — ignored when matching a candidate
# label to the queried name (so "Terma Group" must match on "terma", not the shared "group").
_GENERIC_NAME_TOKENS = frozenset({
    "group", "gmbh", "mbh", "ag", "se", "kg", "kgaa", "ohg", "ug", "gbr", "ev", "co", "inc", "corp",
    "corporation", "company", "llc", "ltd", "plc", "holding", "the", "und", "and", "sa", "spa", "nv",
    "bv", "oy", "ab", "international", "global"})


def _significant_tokens(name: str) -> list[str]:
    toks = re.findall(r"[a-z0-9]+", (name or "").lower())
    sig = [t for t in toks if t not in _GENERIC_NAME_TOKENS and len(t) >= 2]
    return sig or toks   # all-generic name → fall back to every token (don't over-block)


def _name_matches(query: str, title: str) -> bool:
    """True when a candidate title/label plausibly names the queried EMPLOYER — every significant
    query token must appear in the title (exact word, or a ≥4-char substring either way). Rejects the
    fuzzy wrong-matches (Terma→"Therme", Phoenix Medical→"Phoenix Media"). Conservative by design: a
    near-but-different name fails here and the live agent fills the description instead."""
    qt = _significant_tokens(query)
    if not qt:
        return True
    tt = _significant_tokens(title)

    def _hit(t: str) -> bool:
        return any(t == w or (len(t) >= 4 and len(w) >= 4 and (t in w or w in t)) for w in tt)

    return all(_hit(t) for t in qt)


def _claim_value(claims: dict, pid: str):
    """First usable value of a Wikidata claim P-id across its datatypes (string/url, quantity, time
    →year, wikibase-entityid →qid). None if absent/unparseable."""
    for snak in (claims.get(pid) or []):
        try:
            dv = snak["mainsnak"]["datavalue"]
        except (KeyError, TypeError):
            continue
        val, t = dv.get("value"), dv.get("type")
        if t == "string":
            return val
        if t == "quantity":
            try:
                return int(float(str(val.get("amount")).lstrip("+")))
            except (TypeError, ValueError, AttributeError):
                return None
        if t == "time":
            m = re.search(r"([+-]?\d{4})", str((val or {}).get("time") or ""))
            return m.group(1).lstrip("+") if m else None
        if t == "wikibase-entityid":
            return (val or {}).get("id")
    return None


def _qid_label(qid: str, *, lang: str, timeout_s: int) -> str | None:
    try:
        js = http_get_json(_WD_API, headers=_WD_UA, timeout_s=timeout_s, params={
            "action": "wbgetentities", "ids": qid, "props": "labels",
            "languages": f"{lang}|en|de", "format": "json"})
    except Exception:   # keyless cross-check must never break the caller
        return None
    labels = (((js or {}).get("entities") or {}).get(qid) or {}).get("labels") or {}
    for la in (lang, "en", "de"):
        if la in labels:
            return labels[la].get("value")
    return None


def resolve(name: str, *, lang: str = "en", timeout_s: int = 8) -> dict | None:
    """Resolve a company NAME to its canonical Wikidata entity (typed + name-matched), or None.

    Returns {qid, label, description, official_site, country, employees, inception, wikidata_url}.
    None when no organisation-typed, name-matching entity is found (caller falls back to Wikipedia
    REST / legal-suffix). Keyless; ~2-3 cached API calls; never raises.
    """
    if not name or not name.strip():
        return None
    try:
        js = http_get_json(_WD_API, headers=_WD_UA, timeout_s=timeout_s, params={
            "action": "wbsearchentities", "search": name, "language": lang, "uselang": lang,
            "type": "item", "limit": 7, "format": "json"})
    except Exception as exc:
        logger.debug("wikidata search failed for %r: %s", name, exc)
        return None
    pick = None
    for c in ((js or {}).get("search") or []):
        if _ORG_DESC.search(c.get("description") or "") and _name_matches(name, c.get("label") or ""):
            pick = c
            break
    if not pick:
        return None   # no organisation-typed, name-matching candidate → don't guess

    qid = pick["id"]
    try:
        ent = http_get_json(_WD_API, headers=_WD_UA, timeout_s=timeout_s, params={
            "action": "wbgetentities", "ids": qid, "props": "claims", "format": "json"})
    except Exception as exc:
        logger.debug("wikidata getentities failed for %s: %s", qid, exc)
        ent = None
    claims = (((ent or {}).get("entities") or {}).get(qid) or {}).get("claims") or {}
    country_qid = _claim_value(claims, "P17")   # country
    return {
        "qid": qid,
        "label": pick.get("label") or name,
        "description": pick.get("description") or "",
        "official_site": _claim_value(claims, "P856"),
        "country": _qid_label(country_qid, lang=lang, timeout_s=timeout_s) if country_qid else None,
        "employees": _claim_value(claims, "P1128"),
        "inception": _claim_value(claims, "P571"),
        "wikidata_url": f"https://www.wikidata.org/wiki/{qid}",
    }
