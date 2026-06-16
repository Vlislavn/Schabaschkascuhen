"""Indeed + LinkedIn через форк Vlislavn/JobSpy (editable install в venv).

Паттерн пейсинга/стопа из spike/scripts/a1_runner.py: LinkedIn — fetch_description=True,
5 с между запросами, стоп источника после 2 сбоев подряд. Каждая строка DataFrame →
db.upsert_vacancy. Канарейка: 1 запрос на источник, min-row assertion отличает мёртвый
скрейпер (0 строк, без исключения — сигнатура «тихого Google») от пустого рынка.
"""
from __future__ import annotations

import time
from typing import Any

from .. import db
from ..models import CanaryVerdict

# Между LinkedIn-запросами (страховка поверх встроенного пейсинга jobspy ~0.64 req/s).
LINKEDIN_SLEEP_S = 5.0
INDEED_SLEEP_S = 1.0
# Стоп источника после стольких подряд пустых/упавших запросов (a1_runner.py).
MAX_CONSEC_FAILURES = 2


def _scrape_jobs(**kwargs: Any):
    """Тонкая обёртка над jobspy.scrape_jobs — точка мока в тестах."""
    from jobspy import scrape_jobs  # импорт внутри: форк ставится editable отдельно

    return scrape_jobs(**kwargs)


def _row_to_vacancy(source: str, r: dict, query_term: str, query_city: str) -> dict[str, Any]:
    """Строка DataFrame jobspy → словарь для db.upsert_vacancy."""
    def g(*keys, default=None):
        for k in keys:
            v = r.get(k)
            if v is not None and str(v).strip() and str(v).lower() != "nan":
                return v
        return default

    loc = g("location", default="") or ""
    city = str(loc).split(",")[0].strip() if loc else None
    is_remote = g("is_remote")
    desc = g("description")
    dp = g("date_posted")   # jobspy returns the real publication date (YYYY-MM-DD)
    return {
        "source": source,
        "url": str(g("job_url", "job_url_direct")),
        "title": str(g("title", default="")),
        "company": g("company"),
        "city": city,
        "is_remote_hint": 1 if is_remote is True or str(is_remote).lower() == "true" else
                          (0 if is_remote is False or str(is_remote).lower() == "false" else None),
        "description": str(desc) if desc else None,
        "query_term": query_term,
        "query_city": query_city,
        "date_posted": str(dp)[:10] if dp else None,
    }


def _scrape_one(source: str, term: str, city: str, *, results_wanted: int,
                hours_old: int | None) -> tuple[int, bool]:
    """Один (source, term, city). Возвращает (вставлено_строк, было_исключение)."""
    kw: dict[str, Any] = dict(
        site_name=[source],
        search_term=term,
        location=city,
        results_wanted=results_wanted,
        country_indeed="germany",
        description_format="markdown",
        hours_old=hours_old,
        verbose=0,
    )
    if source == "linkedin":
        kw["linkedin_fetch_description"] = True
    try:
        df = _scrape_jobs(**kw)
    except Exception:
        return 0, True
    if df is None or len(df) == 0:
        return 0, False
    return df, False  # type: ignore[return-value]


def scrape(cfg: dict, con, *, queries: list[str] | None = None,
           sources: list[str] | None = None, hours_old: int | None = None) -> dict[str, int]:
    """Скрейп Indeed/LinkedIn по матрице (query × city). Возвращает {source: inserted}."""
    search = cfg.get("search", {})
    queries = queries if queries is not None else search.get("queries_en", [])
    cities = search.get("cities", [])
    sources = sources if sources is not None else [
        s for s in search.get("sources", []) if s in ("indeed", "linkedin")
    ]
    if hours_old is None:
        hours_old = search.get("hours_old")
    results_wanted = int(search.get("results_wanted", 25))

    inserted: dict[str, int] = {}
    for source in sources:
        sleep_s = LINKEDIN_SLEEP_S if source == "linkedin" else INDEED_SLEEP_S
        consec_fail = 0
        count = 0
        pairs = [(t, c) for t in queries for c in cities]
        for i, (term, city) in enumerate(pairs):
            res, exc = _scrape_one(source, term, city, results_wanted=results_wanted,
                                   hours_old=hours_old)
            rows = 0
            if not exc and not isinstance(res, int):
                df = res
                for rec in df.to_dict("records"):
                    vac = _row_to_vacancy(source, rec, term, city)
                    if not vac["url"] or vac["url"].lower() == "nan":
                        continue
                    db.upsert_vacancy(con, vac)
                    rows += 1
            count += rows
            # Break the source ONLY on consecutive real FAILURES (exceptions / blocks like 429),
            # NOT on empty results. A niche magnet query (space, animals, defense) legitimately
            # returns 0 some nights; stopping there silently killed the remaining productive
            # queries and starved exactly the niche magnets the user cares about.
            if exc:
                consec_fail += 1
                if consec_fail >= MAX_CONSEC_FAILURES:
                    db.log_funnel(con, "scrape_stop", count, source,
                                  f"stop after {consec_fail} consecutive failures")
                    break
            else:
                consec_fail = 0
            if i < len(pairs) - 1:
                time.sleep(sleep_s)
        inserted[source] = count
        db.log_funnel(con, "scrape", count, source)
    return inserted


def canary(cfg: dict, con) -> dict[str, str]:
    """1 канарный запрос на источник. min-row=0 без исключения → DEAD_SCRAPER (кейс Google)."""
    search = cfg.get("search", {})
    sources = [s for s in search.get("sources", []) if s in ("indeed", "linkedin")]
    city = (search.get("cities") or ["Frankfurt am Main, Germany"])[0]
    verdicts: dict[str, str] = {}
    for source in sources:
        try:
            res, exc = _scrape_one(source, "machine learning", city,
                                   results_wanted=5, hours_old=None)
        except Exception as e:  # pragma: no cover - defensive
            db.log_canary(con, source, CanaryVerdict.DEAD_SCRAPER.value, 0, str(e)[:300])
            verdicts[source] = CanaryVerdict.DEAD_SCRAPER.value
            continue
        if exc:
            verdict, rows = CanaryVerdict.DEGRADED.value, 0
        else:
            rows = 0 if isinstance(res, int) else len(res)
            # 0 строк И отсутствие исключения = сигнатура тихой смерти скрейпера.
            verdict = CanaryVerdict.DEAD_SCRAPER.value if rows == 0 else CanaryVerdict.OK.value
        db.log_canary(con, source, verdict, rows)
        verdicts[source] = verdict
    return verdicts
