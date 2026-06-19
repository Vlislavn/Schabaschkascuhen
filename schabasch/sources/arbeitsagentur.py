"""Нативный клиент Bundesagentur für Arbeit Jobsuche API.

Co-primary источник (найден спайком, 298/298 HTTP 200 @0.93 req/s, ноль 403/429).
search: /pc/v4/jobs (пагинация size=50). details: /pc/v4/jobdetails/{base64(refnr)}.
Статический ключ X-API-Key: jobboerse-jobsuche (без регистрации). Throttle 1 req/s.
Сетевые вызовы через llm.http_get_json (центральный retry-враппер; 404/410 → EXPIRED).
Import-list item #2: мигрировали с v3 на v4 jobdetails по bundesAPI/jobsuche-api README
(live-probe 2026-06-14: v3 и v4 возвращают идентичные описания, оба HTTP 200).
"""
from __future__ import annotations

import base64
import math
import time

from .. import db, freshness
from ..llm import LLMError, http_get_json
from ..models import ErrorClass, Status

API = "https://rest.arbeitsagentur.de/jobboerse/jobsuche-service"
HEADERS = {"X-API-Key": "jobboerse-jobsuche", "User-Agent": "schabasch/0.1"}
THROTTLE_S = 1.0
PAGE_SIZE = 50
MAX_PAGES = 10  # страховка от бесконечной пагинации


def _city_param(city: str) -> str:
    """'Frankfurt am Main, Germany' → 'Frankfurt am Main' (API ждёт голый город)."""
    return str(city).split(",")[0].strip()


def search(cfg: dict, con, *, queries: list[str] | None = None) -> int:
    """GET /pc/v4/jobs по матрице (query × city) с пагинацией. Возвращает число новых."""
    search_cfg = cfg.get("search", {})
    queries = queries if queries is not None else search_cfg.get("queries_en", [])
    cities = [_city_param(c) for c in search_cfg.get("cities", [])]
    umkreis = int(cfg.get("geo", {}).get("umkreis_km", 50))
    # Свежесть: hours_old → veröffentlichtseit (дни), если задано (инкрементальный сбор).
    hours_old = search_cfg.get("hours_old")
    veroeff_days = math.ceil(hours_old / 24) if hours_old else None
    # CLIENT-SIDE date gate: the AA API honours veroeffentlichtseit loosely (returns 4+ day-old, even
    # 14-month-old rows by aktuelleVeroeffentlichungsdatum). Drop anything older than the window here
    # so stale postings never enter the pool. Null/unparseable dates pass (freshness.too_old).
    max_age = freshness.max_post_age_days(cfg)

    new_count = 0
    stale_skipped = 0
    for city in cities:
        for q in queries:
            page = 1
            while page <= MAX_PAGES:
                params = {"was": q, "wo": city, "umkreis": umkreis,
                          "size": PAGE_SIZE, "page": page}
                if veroeff_days:
                    params["veroeffentlichtseit"] = veroeff_days
                try:
                    data = http_get_json(f"{API}/pc/v4/jobs", headers=HEADERS, params=params)
                except LLMError:
                    break  # источник недоступен на этом запросе — следующий
                if not isinstance(data, dict):
                    break
                jobs = data.get("stellenangebote") or []
                if not jobs:
                    break
                for j in jobs:
                    refnr = j.get("refnr")
                    if not refnr:
                        continue
                    # реальная дата публикации вакансии (не скрейпа)
                    posted = j.get("aktuelleVeroeffentlichungsdatum")
                    if freshness.too_old(posted, max_age):
                        stale_skipped += 1
                        continue
                    ort = j.get("arbeitsort") or {}
                    ext = j.get("externeUrl")
                    url = ext or f"https://www.arbeitsagentur.de/jobsuche/jobdetail/{refnr}"
                    before = con.execute(
                        "SELECT 1 FROM vacancy WHERE url = ?", (url,)
                    ).fetchone()
                    db.upsert_vacancy(con, {
                        "source": "arbeitsagentur",
                        "url": url,
                        "refnr": refnr,
                        "title": j.get("titel") or "",
                        "company": j.get("arbeitgeber"),
                        "city": ort.get("ort"),
                        "query_term": q,
                        "query_city": city,
                        "date_posted": posted,
                    })
                    if before is None:
                        new_count += 1
                total = data.get("maxErgebnisse") or 0
                if page * PAGE_SIZE >= total or len(jobs) < PAGE_SIZE:
                    break
                page += 1
                time.sleep(THROTTLE_S)
            time.sleep(THROTTLE_S)
    if stale_skipped:
        db.log_funnel(con, "ingest_stale_skip", stale_skipped, "arbeitsagentur",
                      detail=f"older than {max_age}d by date_posted")
    db.log_funnel(con, "scrape", new_count, "arbeitsagentur")
    return new_count


def check_open(refnr: str) -> bool | None:
    """Deterministic still-open check for an Arbeitsagentur job: re-query jobdetails by refnr.
    True = alive, False = gone (404/410 'permanent'), None = couldn't verify (timeout/endpoint
    error) — never guess 'closed' from a failed fetch. Same v4 endpoint + base64(refnr) as
    fetch_details; one attempt (a liveness probe, not a retried fetch)."""
    if not refnr:
        return None
    b64 = base64.b64encode(str(refnr).encode("utf-8")).decode("ascii")
    try:
        http_get_json(f"{API}/pc/v4/jobdetails/{b64}", headers=HEADERS, attempts=1)
        return True
    except LLMError as e:
        return False if "permanent" in (e.details or "") else None


def fetch_details(cfg: dict, con, *, limit: int = 400) -> int:
    """Добыча полных описаний AA через /pc/v3/jobdetails. Возвращает число описанных."""
    rows = con.execute(
        "SELECT id, refnr FROM vacancy WHERE source = 'arbeitsagentur' AND status = ? "
        "AND refnr IS NOT NULL ORDER BY id LIMIT ?",
        (Status.NEW.value, limit),
    ).fetchall()
    described = 0
    for row in rows:
        refnr = row["refnr"]
        b64 = base64.b64encode(refnr.encode("utf-8")).decode("ascii")
        try:
            data = http_get_json(f"{API}/pc/v4/jobdetails/{b64}", headers=HEADERS, attempts=2)
        except LLMError as e:
            if "permanent" in e.details:  # 404/410 — вакансия исчезла
                db.set_status(con, row["id"], Status.EXPIRED)
            else:
                db.set_error(con, row["id"], e.error_class, e.details)
            time.sleep(THROTTLE_S)
            continue
        if not isinstance(data, dict):
            db.set_error(con, row["id"], ErrorClass.SCHEMA_VIOLATION, "details not an object")
            time.sleep(THROTTLE_S)
            continue
        desc = data.get("stellenangebotsBeschreibung") or ""
        loks = data.get("stellenlokationen") or [{}]
        adresse = (loks[0] or {}).get("adresse") or {}
        is_temp = data.get("istArbeitnehmerUeberlassung")
        is_temp_int = None if is_temp is None else (1 if is_temp else 0)
        # publication date from details (backfill only if search() didn't already set it)
        posted = data.get("ersteVeroeffentlichungsdatum") or data.get("datumErsteVeroeffentlichung")
        if desc.strip():
            from ..models import content_hash
            con.execute(
                "UPDATE vacancy SET title = COALESCE(?, title), company = COALESCE(?, company), "
                "city = COALESCE(?, city), description = ?, desc_hash = ?, "
                "is_temp_agency = ?, date_posted = COALESCE(date_posted, ?), status = ? WHERE id = ?",
                (data.get("stellenangebotsTitel"), data.get("firma"), adresse.get("ort"),
                 desc, content_hash(desc), is_temp_int, posted, Status.DESCRIBED.value, row["id"]),
            )
            con.commit()
            described += 1
        else:
            # Empty description = transient (AA details endpoint precedent: v2 died silently).
            # Leave status=NEW so tomorrow's fetch retries it (USE_CASE 4a) — was PREFILTERED,
            # which is terminal and dropped the vacancy permanently after one empty response.
            db.set_error(con, row["id"], ErrorClass.EMPTY_OUTPUT, "empty AA description")
        time.sleep(THROTTLE_S)
    db.log_funnel(con, "details", described, "arbeitsagentur")
    return described
