"""Прямое перечисление публичных ATS-бордов — keyless JSON/XML, без anti-bot войны и без LLM.

Восемь паттернов (все проверены live 2026-07-03; SmartRecruiters-keyless подтверждён curl'ом —
research-источники противоречили, live-проба решила):
  greenhouse       boards-api.greenhouse.io/v1/boards/{slug}/jobs?content=true
  lever            api.lever.co/v0/postings/{slug}?mode=json   (+ lever-eu: api.eu.lever.co)
  ashby            api.ashbyhq.com/posting-api/job-board/{slug}
  personio         {slug}.jobs.personio.de/xml                  (немецкий ATS — DE SME)
  workable         apply.workable.com/api/v1/widget/accounts/{slug}
  recruitee        {slug}.recruitee.com/api/offers/
  smartrecruiters  api.smartrecruiters.com/v1/companies/{slug}/postings (+ detail для DE/remote)

Slug-discovery: пробы кандидатов от имени компании (search.ats_companies, фолбэк
target_companies); хиты кэшируются в sidecar-таблицу ats_board (аддитивно, frozen-схема
не тронута), fetch каждый тик идёт по кэшу, re-probe промахов раз в _PROBE_DAYS.

Freshness-гейт НЕ применяется: присутствие на живом борде компании = доказательство liveness
(в отличие от AA-флуда старых строк); date_posted пишется честно для recency-re-rank.

ponytail: slug-коллизия (чужая компания под тем же slug) теоретически возможна — токен борда
создаёт сама компания-клиент, так что вероятность мала; где ответ несёт имя (workable) —
сверяем RapidFuzz'ом. Апгрейд при первом реальном промахе: careers-page крауль за embed-ссылкой.
"""
from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timedelta, timezone

from rapidfuzz import fuzz

from .. import db, geo
from ..llm import LLMError, http_get_json

logger = logging.getLogger(__name__)

_PROBE_DAYS = 30
_THROTTLE_S = 0.3

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ats_board (
  company    TEXT NOT NULL,
  ats        TEXT NOT NULL,
  slug       TEXT,
  last_ok    TEXT,
  last_probe TEXT,
  PRIMARY KEY (company, ats)
);
"""


def ensure_tables(con) -> None:
    con.executescript(_SCHEMA)
    con.commit()


def _slugs(name: str) -> list[str]:
    flat = re.sub(r"[^a-z0-9]", "", name.lower())
    dashed = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return [s for s in dict.fromkeys([flat, dashed]) if s]


def _keep(city: str | None, location_raw: str, cfg: dict) -> tuple[bool, int | None]:
    """(брать?, is_remote_hint). Борды глобальные → как tertiary: только ИЗВЕСТНЫЙ город в
    радиусе; явный remote проходит с хинтом."""
    if re.search(r"(?i)\bremote\b", location_raw or ""):
        return True, 1
    in_radius, dist = geo.geo_check(city, cfg)
    return (in_radius and dist is not None), None


def _ms_to_iso(ms) -> str | None:
    try:
        return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).isoformat()
    except (TypeError, ValueError, OSError):
        return None


# ------------------------------------------------------- fetchers: ats → list[posting-dict]
# Каждый возвращает [{url,title,city,location_raw,description,date_posted,company_name?}]
# или бросает LLMError (проба мимо / борд лёг).

def _greenhouse(slug: str) -> list[dict]:
    data = http_get_json(f"https://boards-api.greenhouse.io/v1/boards/{slug}/jobs",
                         params={"content": "true"}, attempts=1)
    out = []
    for j in data.get("jobs") or []:
        loc = ((j.get("location") or {}).get("name")) or ""
        out.append({"url": j.get("absolute_url"), "title": j.get("title") or "",
                    "city": loc.split(",")[0].strip() or None, "location_raw": loc,
                    "description": j.get("content"), "date_posted": j.get("updated_at")})
    return out


def _lever(slug: str, host: str = "api.lever.co") -> list[dict]:
    data = http_get_json(f"https://{host}/v0/postings/{slug}", params={"mode": "json"}, attempts=1)
    if not isinstance(data, list):
        _schema_err(data)
    out = []
    for j in data:
        loc = ((j.get("categories") or {}).get("location")) or ""
        out.append({"url": j.get("hostedUrl"), "title": j.get("text") or "",
                    "city": loc.split(",")[0].strip() or None, "location_raw": loc,
                    "description": j.get("descriptionPlain"),
                    "date_posted": _ms_to_iso(j.get("createdAt"))})
    return out


def _lever_eu(slug: str) -> list[dict]:
    return _lever(slug, host="api.eu.lever.co")


def _ashby(slug: str) -> list[dict]:
    data = http_get_json(f"https://api.ashbyhq.com/posting-api/job-board/{slug}", attempts=1)
    out = []
    for j in data.get("jobs") or []:
        loc = j.get("location") or ""
        out.append({"url": j.get("jobUrl") or j.get("applyUrl"), "title": j.get("title") or "",
                    "city": loc.split(",")[0].strip() or None, "location_raw": loc,
                    "description": j.get("descriptionPlain") or j.get("descriptionHtml"),
                    "date_posted": j.get("publishedAt")})
    return out


def _personio(slug: str) -> list[dict]:
    import xml.etree.ElementTree as ET

    import requests

    from ..models import ErrorClass
    try:
        r = requests.get(f"https://{slug}.jobs.personio.de/xml", params={"language": "en"},
                         timeout=30, headers={"User-Agent": "schabasch/0.1"})
        r.raise_for_status()
        root = ET.fromstring(r.content)
    except (requests.RequestException, ET.ParseError) as e:
        raise LLMError(ErrorClass.HTTP_ERROR, f"personio {slug}: {e}") from e
    out = []
    for pos in root.iter("position"):
        pid = (pos.findtext("id") or "").strip()
        if not pid:
            continue
        city = (pos.findtext("office") or "").split(",")[0].strip() or None
        desc = "\n\n".join(v.strip() for v in
                           (d.findtext("value") or "" for d in pos.iter("jobDescription"))
                           if v and v.strip()) or None
        out.append({"url": f"https://{slug}.jobs.personio.de/job/{pid}",
                    "title": (pos.findtext("name") or "").strip(),
                    "city": city, "location_raw": pos.findtext("office") or "",
                    "description": desc, "date_posted": None})
    return out


def _workable(slug: str) -> list[dict]:
    data = http_get_json(f"https://apply.workable.com/api/v1/widget/accounts/{slug}", attempts=1)
    out = []
    for j in data.get("jobs") or []:
        loc = ", ".join(x for x in (j.get("city"), j.get("country")) if x)
        out.append({"url": j.get("url"), "title": j.get("title") or "",
                    "city": (j.get("city") or "").strip() or None, "location_raw": loc,
                    "description": j.get("description"), "date_posted": j.get("published_on"),
                    "company_name": data.get("name")})
    return out


def _recruitee(slug: str) -> list[dict]:
    data = http_get_json(f"https://{slug}.recruitee.com/api/offers/", attempts=1)
    out = []
    for j in data.get("offers") or []:
        loc = ", ".join(x for x in (j.get("city"), j.get("country")) if x)
        out.append({"url": j.get("careers_url"), "title": j.get("title") or "",
                    "city": (j.get("city") or "").strip() or None, "location_raw": loc,
                    "description": j.get("description"), "date_posted": j.get("created_at")})
    return out


def _smartrecruiters(slug: str) -> list[dict]:
    data = http_get_json(f"https://api.smartrecruiters.com/v1/companies/{slug}/postings",
                         params={"limit": 100}, attempts=1)
    out = []
    for j in data.get("content") or []:
        loc = j.get("location") or {}
        country = (loc.get("country") or "").lower()
        remote = bool(loc.get("remote"))
        desc = None
        # detail (N+1) только для потенциально релевантных — DE или remote; остальным хватит NEW
        if j.get("id") and (country == "de" or remote):
            try:
                detail = http_get_json(
                    f"https://api.smartrecruiters.com/v1/companies/{slug}/postings/{j['id']}",
                    attempts=1)
                sections = ((detail.get("jobAd") or {}).get("sections") or {})
                desc = "\n\n".join(t for t in (
                    (sections.get(k) or {}).get("text") for k in
                    ("companyDescription", "jobDescription", "qualifications",
                     "additionalInformation")) if t) or None
            except LLMError as e:
                logger.debug("smartrecruiters detail fetch failed for %s/%s: %s",
                             slug, j.get("id"), e)
                desc = None  # fallback: список остаётся источником; постинг идёт как NEW без detail
        out.append({"url": f"https://jobs.smartrecruiters.com/{slug}/{j.get('id')}",
                    "title": j.get("name") or "",
                    "city": (loc.get("city") or "").split(",")[0].strip() or None,
                    "location_raw": loc.get("fullLocation") or "",
                    "description": desc, "date_posted": j.get("releasedDate"),
                    "company_name": (j.get("company") or {}).get("name")})
    return out


_ATS = {"greenhouse": _greenhouse, "lever": _lever, "lever-eu": _lever_eu, "ashby": _ashby,
        "personio": _personio, "workable": _workable, "recruitee": _recruitee,
        "smartrecruiters": _smartrecruiters}


def _schema_err(data):
    from ..models import ErrorClass
    raise LLMError(ErrorClass.SCHEMA_VIOLATION, f"unexpected shape: {str(data)[:100]}")


# ------------------------------------------------------------------ probe + fetch

def _probe(cfg: dict, con, companies: list[str]) -> int:
    """Найти борды для компаний без известного ATS; хиты → ats_board. Возвращает число хитов."""
    now = datetime.now(timezone.utc).isoformat()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=_PROBE_DAYS)).isoformat()
    hits = 0
    for company in companies:
        known = con.execute("SELECT 1 FROM ats_board WHERE company = ? AND last_ok IS NOT NULL",
                            (company,)).fetchone()
        probed = con.execute("SELECT 1 FROM ats_board WHERE company = ? AND last_probe > ?",
                             (company, cutoff)).fetchone()
        if known or probed:
            continue
        found = False
        for ats, fetch in _ATS.items():
            for slug in _slugs(company):
                try:
                    postings = fetch(slug)
                except LLMError:
                    time.sleep(_THROTTLE_S)
                    continue
                name = next((p.get("company_name") for p in postings if p.get("company_name")),
                            None)
                if name and fuzz.token_set_ratio(name.lower(), company.lower()) < 80:
                    continue  # slug-коллизия: борд отвечает, но это другая компания
                con.execute("INSERT OR REPLACE INTO ats_board VALUES (?, ?, ?, ?, ?)",
                            (company, ats, slug, now, now))
                hits += 1
                found = True
                break
            if found:
                break
        if not found:  # запомнить промах, чтобы не долбить те же 404 каждый тик
            con.execute("INSERT OR REPLACE INTO ats_board VALUES (?, 'none', NULL, NULL, ?)",
                        (company, now))
    con.commit()
    return hits


def scrape(cfg: dict, con) -> dict[str, int]:
    """Пробы (по кэшу) + fetch всех известных бордов. Возвращает новые вакансии по ATS."""
    ensure_tables(con)
    search = cfg.get("search", {}) or {}
    companies = search.get("ats_companies") or search.get("target_companies") or []
    probed = _probe(cfg, con, [c for c in companies if c])
    counts: dict[str, int] = {}
    rows = con.execute(
        "SELECT company, ats, slug FROM ats_board WHERE last_ok IS NOT NULL").fetchall()
    now = datetime.now(timezone.utc).isoformat()
    for r in rows:
        try:
            postings = _ATS[r["ats"]](r["slug"])
        except (LLMError, KeyError) as e:
            db.log_funnel(con, "scrape", 0, f"ats:{r['ats']}",
                          detail=f"{r['company']}: {e}")
            continue
        new = 0
        for p in postings:
            url, title = p.get("url"), p.get("title")
            if not url or not title:
                continue
            keep, remote = _keep(p.get("city"), p.get("location_raw") or "", cfg)
            if not keep:
                continue
            before = con.execute("SELECT 1 FROM vacancy WHERE url = ?", (url,)).fetchone()
            desc = p.get("description")
            db.upsert_vacancy(con, {
                "source": f"ats:{r['ats']}", "url": url, "title": title,
                "company": r["company"], "city": p.get("city"),
                "is_remote_hint": remote,
                "description": desc if desc and desc.strip() else None,
                "date_posted": p.get("date_posted"),
            })
            if before is None:
                new += 1
        con.execute("UPDATE ats_board SET last_ok = ? WHERE company = ? AND ats = ?",
                    (now, r["company"], r["ats"]))
        counts[r["ats"]] = counts.get(r["ats"], 0) + new
        time.sleep(_THROTTLE_S)
    con.commit()
    total = sum(counts.values())
    db.log_funnel(con, "scrape", total, "ats",
                  detail=f"boards={len(rows)} probed_new={probed} {counts}")
    return counts or {"ats": 0}
