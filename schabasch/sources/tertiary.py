"""Третичные фетчеры (Should): Arbeitnow JSON API + GermanTechJobs RSS.

Легально чистые, 0% пересечения с основными бордами (измерено спайком). Фильтр по региону
на входе через geo.geo_check — в БД попадают только вакансии в радиусе якорей (иначе пул
засоряется общенемецкой выдачей). Arbeitnow отдаёт ПОЛНЫЕ описания → сразу DESCRIBED и
готовы к нормализации. GTJ RSS — только title+link (описаний нет) → NEW, ценность маргинальна
без desc-фетчера их сайта (вне скоупа v1); оставлено как лёгкий ingest по плану.
"""
from __future__ import annotations

import re

from .. import db, geo
from ..llm import LLMError, http_get_json

ARBEITNOW_API = "https://www.arbeitnow.com/api/job-board-api"
GTJ_RSS = "https://germantechjobs.de/rss"


def _in_region(city: str | None, cfg: dict) -> bool:
    """СТРОГО: только ИЗВЕСТНЫЙ город в радиусе. Третичные фиды общенемецкие — в отличие от
    таргетированного поиска бордов, здесь «неизвестный город» (dist=None) НЕ пропускаем,
    иначе фид заливает пул нерелевантной общенемецкой выдачей (Ramstein/Magstadt/Pörnbach…)."""
    in_radius, dist = geo.geo_check(city, cfg)
    return in_radius and dist is not None


def fetch_arbeitnow(cfg: dict, con, *, max_pages: int = 3) -> int:
    """Arbeitnow JSON API (пагинация). Полные описания → DESCRIBED. Возвращает число новых."""
    new_count = 0
    for page in range(1, max_pages + 1):
        try:
            data = http_get_json(ARBEITNOW_API, params={"page": page}, attempts=2)
        except LLMError:
            break
        jobs = data.get("data") if isinstance(data, dict) else None
        if not jobs:
            break
        for j in jobs:
            url = j.get("url")
            if not url:
                continue
            city = str(j.get("location") or "").split(",")[0].strip()
            if not _in_region(city, cfg):
                continue
            remote = j.get("remote")
            is_remote = 1 if remote is True or str(remote).lower() == "true" else 0
            desc = j.get("description") or ""
            before = con.execute("SELECT 1 FROM vacancy WHERE url = ?", (url,)).fetchone()
            db.upsert_vacancy(con, {
                "source": "arbeitnow", "url": url, "title": j.get("title") or "",
                "company": j.get("company_name"), "city": city,
                "is_remote_hint": is_remote,
                "description": desc if desc.strip() else None,
                "query_term": ",".join(j.get("tags") or []) or None, "query_city": None,
            })
            if before is None:
                new_count += 1
    db.log_funnel(con, "scrape", new_count, "arbeitnow")
    return new_count


# Сепаратор города — РОВНО ' - ' с пробелами (иначе ловит дефис в 'IT-Support'); en-dash '–'
# в названии роли не мешает. Город — последний ' - ...' перед ' @ Company [salary]'.
_GTJ_TITLE = re.compile(r"^(?P<title>.*?)\s+-\s+(?P<city>[^@\[]+?)\s*@\s*(?P<company>[^\[]+?)"
                        r"(?:\s*\[[^\]]*\])?\s*$")


def _parse_gtj_title(raw: str) -> tuple[str, str | None, str | None]:
    """'Role - City @ Company [salary]' → (role, city, company). Грубо, с фоллбеком."""
    m = _GTJ_TITLE.match(raw.strip())
    if m:
        return m.group("title").strip(), m.group("city").strip(), m.group("company").strip()
    return raw.strip(), None, None


def fetch_germantechjobs(cfg: dict, con) -> int:
    """GermanTechJobs RSS → NEW (без описаний). Регион-фильтр по распарсенному городу."""
    import xml.etree.ElementTree as ET

    import requests

    from ..models import ErrorClass
    try:
        r = requests.get(GTJ_RSS, timeout=30, headers={"User-Agent": "schabasch/0.1"})
        r.raise_for_status()
        root = ET.fromstring(r.content)
    except (requests.RequestException, ET.ParseError) as e:
        db.log_funnel(con, "scrape", 0, "gtj", detail=f"{ErrorClass.HTTP_ERROR.value}: {e}")
        return 0
    new_count = 0
    for item in root.iter("item"):
        link = (item.findtext("link") or "").strip()
        raw_title = (item.findtext("title") or "").strip()
        if not link or not raw_title:
            continue
        title, city, company = _parse_gtj_title(raw_title)
        # GTJ без описаний и общенемецкий → берём ТОЛЬКО уверенно-региональные (известный город
        # в радиусе). Нераспарсенный/неизвестный/далёкий город — дроп (иначе 1200+ мусорных NEW).
        if not _in_region(city, cfg):
            continue
        before = con.execute("SELECT 1 FROM vacancy WHERE url = ?", (link,)).fetchone()
        db.upsert_vacancy(con, {"source": "gtj", "url": link, "title": title,
                                "company": company, "city": city, "query_term": None})
        if before is None:
            new_count += 1
    db.log_funnel(con, "scrape", new_count, "gtj")
    return new_count
