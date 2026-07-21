"""HN «Ask HN: Who is hiring?» — месячный тред через keyless Algolia API (2 запроса/тик).

Структурно «посты от людей»: правило треда запрещает агентства и борды («please only post if
you personally are part of the hiring company»). Top-level comment = одна вакансия; формат
де-факто «Company | Role | Location | …». Выдача global/dev-skewed → режем по регион-термам
(города из cfg + Germany/EU/Remote); остальное структурирует normalizer.

Freshness: тред МЕСЯЧНЫЙ, глобальное окно max_post_age_days (≈2 дня, под каденс бордов)
опустошило бы источник — здесь своё окно = жизнь треда (35 дней), дата поста пишется честно.
"""
from __future__ import annotations

import re

from .. import db, freshness
from ..llm import LLMError, http_get_json
from .social import _mk_title, _strip_html

HN_SEARCH = "https://hn.algolia.com/api/v1/search_by_date"
HN_ITEM = "https://hn.algolia.com/api/v1/items/{id}"
_THREAD_MAX_AGE_DAYS = 35


def _region_re(cfg: dict) -> re.Pattern:
    cities = [c.split(",")[0].strip() for c in (cfg.get("search", {}) or {}).get("cities", [])]
    terms = [re.escape(t) for t in cities if t] + \
            ["Germany", "Deutschland", "Europe", "European", "EU", "Remote"]
    return re.compile(r"(?i)\b(" + "|".join(terms) + r")\b")


def _company_from_title(first_line: str) -> str | None:
    """«Company | Role | Location» → Company; мусорные первые сегменты не берём."""
    if "|" not in first_line:
        return None
    seg = first_line.split("|", 1)[0].strip()
    return seg if 0 < len(seg) <= 60 and "hiring" not in seg.lower() else None


def scrape(cfg: dict, con) -> dict[str, int]:
    try:
        found = http_get_json(HN_SEARCH, params={"tags": "story,author_whoishiring",
                                                 "hitsPerPage": 10}, attempts=2)
    except LLMError as e:
        db.log_funnel(con, "scrape", 0, "hn", detail=f"search: {e}")
        return {"hn": 0}
    hit = next((h for h in (found.get("hits") or [])
                if "who is hiring" in (h.get("title") or "").lower()), None)
    if hit is None:
        db.log_funnel(con, "scrape", 0, "hn", detail="no whoishiring thread found")
        return {"hn": 0}
    try:
        thread = http_get_json(HN_ITEM.format(id=hit["objectID"]), attempts=2)
    except LLMError as e:
        db.log_funnel(con, "scrape", 0, "hn", detail=f"items: {e}")
        return {"hn": 0}

    region = _region_re(cfg)
    new = 0
    for child in thread.get("children") or []:
        cid = child.get("id")
        text = _strip_html_keep_lines(child.get("text") or "")
        if not cid or not text.strip():  # deleted/dead коммент
            continue
        if not region.search(text):
            continue
        if freshness.too_old(child.get("created_at"), _THREAD_MAX_AGE_DAYS):
            continue
        url = f"https://news.ycombinator.com/item?id={cid}"
        title = _mk_title(text, limit=160)
        before = con.execute("SELECT 1 FROM vacancy WHERE url = ?", (url,)).fetchone()
        db.upsert_vacancy(con, {
            "source": "hn", "url": url, "title": title,
            "company": _company_from_title(title), "city": None,
            "is_remote_hint": 1 if re.search(r"(?i)\bremote\b", text) else None,
            "description": f"Posted by {child.get('author') or '?'} in HN Who is hiring "
                           f"({hit.get('title')}):\n\n{text}",
            "date_posted": child.get("created_at"),
        })
        if before is None:
            new += 1
    db.log_funnel(con, "scrape", new, "hn", detail=f"thread={hit.get('title')}")
    return {"hn": new}


def _strip_html_keep_lines(html: str) -> str:
    """HN text — HTML с <p>-абзацами; первая строка нужна как title → режем по <p>, внутри
    абзаца инлайн-теги склеиваются пробелом (обычный _strip_html)."""
    parts = re.split(r"(?i)<p[^>]*>", html or "")
    return "\n".join(p for p in (_strip_html(x) for x in parts) if p)
