"""Person-post mining: Bluesky + Mastodon + Telegram public channels — все keyless.

Гипотеза: объявления «мы ищем» в постах ЛЮДЕЙ (hiring manager / founder / тиммейт), а не на
корпоративных бордах, — самый ценный канал. Twitter/X keyless в 2026 нежизнеспособен (free tier
закрыт 02/2026, Nitter мёртв), Bluesky — прямая замена; Reddit .json заблокирован с 05/2026.

Ладдер hard-before-soft: пост проходит ДЕШЁВЫЕ ворота до upsert (длина → hiring-лексикон →
релевантность запросам → freshness), LLM не тратится на очевидный шум; выжившие идут полным
текстом в description → DESCRIBED → normalize структурирует free-text в Card сам.

Bluesky без auth отдаёт ТОЛЬКО первую страницу поиска (cursor на публичном AppView 403-ит,
bluesky-social/atproto#3583) — кап логируется в funnel (no silent caps).
"""
from __future__ import annotations

import re
import time

from .. import db, freshness
from ..llm import LLMError, http_get_json
from ..models import ErrorClass

BLUESKY_SEARCH = "https://api.bsky.app/xrpc/app.bsky.feed.searchPosts"
DEFAULT_INSTANCES = ["mastodon.social", "hachyderm.io", "mastodon.online"]
DEFAULT_HASHTAGS = ["hiring", "fedihire", "getfedihired"]
_UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) schabasch/0.1"}
_THROTTLE_S = 0.3

# ponytail: naive keyword gate, поднять до qwen-триажа если замеренный шум останется высоким.
# DE-фразы не валидированы по корпусу (см. план) — замеряются на живой выборке.
_HIRING_RE = re.compile(
    r"(?i)(\bhiring\b|we'?re looking for|join (our|the|my) team|open (role|position)s?\b"
    r"|job opening|\bvacanc(y|ies)\b|wir suchen|stellenangebot|stellenausschreibung"
    r"|verst[äa]rkung|bewirb dich|\(m/w/d\))"
)
_MIN_LEN = 80  # короче — не может нести реальное объявление (отсекает "hiring!" + ссылка-мусор)


def _strip_html(html: str) -> str:
    from bs4 import BeautifulSoup
    return BeautifulSoup(html or "", "html.parser").get_text(" ", strip=True)


def _mk_title(text: str, limit: int = 120) -> str:
    first = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    return re.sub(r"\s+", " ", first)[:limit]


def _query_tokens(cfg: dict) -> set[str]:
    search = cfg.get("search", {}) or {}
    toks: set[str] = set()
    for q in (search.get("queries_en") or []) + (search.get("queries_de") or []):
        toks.update(re.findall(r"\w{4,}", q.lower()))
    return toks


def _ingest(cfg: dict, con, *, source: str, url: str, text: str, author: str,
            created_at: str | None, max_age: int, tokens: set[str] | None) -> bool:
    """Ворота + upsert одного поста. True — новая вакансия (url не виден раньше)."""
    if len(text) < _MIN_LEN or not _HIRING_RE.search(text):
        return False
    if tokens is not None and not any(t in text.lower() for t in tokens):
        return False  # хэштег/канальные фиды глобальны по профессиям — режем по токенам запросов
    if freshness.too_old(created_at, max_age):
        return False
    before = con.execute("SELECT 1 FROM vacancy WHERE url = ?", (url,)).fetchone()
    db.upsert_vacancy(con, {
        "source": source, "url": url, "title": _mk_title(text),
        # company не заполняем хэндлом автора — dedup_key(company,title) стал бы мусорным;
        # normalizer извлечёт работодателя из текста. Провенанс — в description.
        "company": None, "city": None,
        "is_remote_hint": 1 if re.search(r"(?i)\bremote\b", text) else None,
        "description": f"Posted by {author} on {source}:\n\n{text}",
        "date_posted": created_at,
    })
    return before is None


# ---------------------------------------------------------------- bluesky

def _bluesky(cfg: dict, con, max_age: int) -> int:
    search = cfg.get("search", {}) or {}
    new = 0
    capped = False
    # Суффикс-якорь повышает precision единственной доступной страницы (100 постов):
    # en → "hiring", de → "suchen" («wir suchen Projektmanager»).
    plans = [(q, "en", f"{q} hiring") for q in (search.get("queries_en") or [])] + \
            [(q, "de", f"{q} suchen") for q in (search.get("queries_de") or [])]
    params_base: dict = {"sort": "latest", "limit": 100}
    if max_age > 0:
        from datetime import datetime, timedelta, timezone
        params_base["since"] = (datetime.now(timezone.utc) -
                                timedelta(days=max_age)).strftime("%Y-%m-%dT%H:%M:%SZ")
    for query, lang, q in plans:
        try:
            data = http_get_json(BLUESKY_SEARCH, params={**params_base, "q": q, "lang": lang},
                                 attempts=2)
        except LLMError:
            continue  # один упавший запрос не валит остальные
        posts = data.get("posts") if isinstance(data, dict) else None
        if not posts:
            continue
        capped = capped or len(posts) >= 100
        for p in posts:
            rec = p.get("record") or {}
            handle = (p.get("author") or {}).get("handle") or ""
            uri = p.get("uri") or ""
            if not handle or not uri:
                continue
            url = f"https://bsky.app/profile/{handle}/post/{uri.rsplit('/', 1)[-1]}"
            if _ingest(cfg, con, source="bluesky", url=url, text=rec.get("text") or "",
                       author=f"@{handle}", created_at=rec.get("createdAt"),
                       max_age=max_age, tokens=None):  # выдача уже query-скоупнута поиском
                new += 1
        time.sleep(_THROTTLE_S)
    db.log_funnel(con, "scrape", new, "bluesky",
                  detail="page-capped (unauth = 1st page only)" if capped else None)
    return new


# ---------------------------------------------------------------- mastodon

def _mastodon(cfg: dict, con, max_age: int, tokens: set[str]) -> int:
    search = cfg.get("search", {}) or {}
    instances = search.get("social_instances") or DEFAULT_INSTANCES
    hashtags = search.get("social_hashtags") or DEFAULT_HASHTAGS
    new = 0
    for inst in instances:
        for tag in hashtags:
            try:
                statuses = http_get_json(f"https://{inst}/api/v1/timelines/tag/{tag}",
                                         params={"limit": 40}, attempts=1)
            except LLMError:
                continue  # инстанс может требовать auth / лежать — идём дальше
            if not isinstance(statuses, list):
                continue
            for s in statuses:
                url = s.get("url")
                if not url:
                    continue
                if _ingest(cfg, con, source="mastodon", url=url,
                           text=_strip_html(s.get("content") or ""),
                           author=f"@{(s.get('account') or {}).get('acct') or '?'}",
                           created_at=s.get("created_at"), max_age=max_age, tokens=tokens):
                    new += 1
            time.sleep(_THROTTLE_S)
    db.log_funnel(con, "scrape", new, "mastodon")
    return new


# ---------------------------------------------------------------- telegram

def _telegram(cfg: dict, con, max_age: int, tokens: set[str]) -> int:
    """Публичные каналы через no-auth HTML-preview t.me/s/<channel>. 302/пустая страница =
    preview выключен или канала нет → warn в funnel, канал скипается."""
    import requests
    from bs4 import BeautifulSoup

    channels = (cfg.get("search", {}) or {}).get("telegram_channels") or []
    new = 0
    for ch in channels:
        try:
            r = requests.get(f"https://t.me/s/{ch}", headers=_UA, timeout=30)
            r.raise_for_status()
        except requests.RequestException as e:
            db.log_funnel(con, "scrape", 0, "telegram",
                          detail=f"{ErrorClass.HTTP_ERROR.value}: {ch}: {e}")
            continue
        msgs = BeautifulSoup(r.text, "html.parser").select(".tgme_widget_message")
        if not msgs:
            db.log_funnel(con, "scrape", 0, "telegram", detail=f"no preview: {ch}")
            continue
        for m in msgs:
            post = m.get("data-post")  # "channel/123"
            text_el = m.select_one(".tgme_widget_message_text")
            if not post or text_el is None:
                continue
            time_el = m.select_one("time[datetime]")
            if _ingest(cfg, con, source="telegram", url=f"https://t.me/{post}",
                       text=text_el.get_text("\n", strip=True), author=f"t.me/{ch}",
                       created_at=time_el.get("datetime") if time_el else None,
                       max_age=max_age, tokens=tokens):
                new += 1
        time.sleep(_THROTTLE_S)
    db.log_funnel(con, "scrape", new, "telegram")
    return new


def scrape(cfg: dict, con) -> dict[str, int]:
    """Все три person-post канала за один шаг тика. Возвращает новые вакансии по каналам."""
    max_age = freshness.max_post_age_days(cfg)
    tokens = _query_tokens(cfg)
    return {
        "bluesky": _bluesky(cfg, con, max_age),
        "mastodon": _mastodon(cfg, con, max_age, tokens),
        "telegram": _telegram(cfg, con, max_age, tokens),
    }
