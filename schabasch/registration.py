"""Self-service user registration (Telegram bot → POST /register).

Orchestrates: cap/duplicate checks → resolve city offline → create the overlay yaml + DB
(users.create_user) → extract the CV into the new user's own DB (candidate.extract_candidate,
reused as-is) → derive search queries + judge summary from the extraction (Tesler: the system,
not the user, turns a CV into config) → rewrite the overlay with the final values.
Extraction failure rolls the user back completely (no half-registered state).

Overlay key shapes that MUST hold (latent-bug fixes, pinned by tests):
- search.queries_en / queries_de (nothing reads a bare `queries`)
- geo.anchors is a DICT {City: {lat, lon, radius_km}} — geo.geo_check iterates .values()
"""
from __future__ import annotations

from rapidfuzz import process as _fuzz_process

from . import candidate, db, geo, users

_DEFAULT_RADIUS_KM = 40
_SUGGESTIONS = 3
_FUZZY_FLOOR = 70  # rapidfuzz WRatio below this → no suggestion (garbage input)

# RU→DE aliases for cities RU-speaking users type in Cyrillic (live finding: «Франкфурт» was
# rejected 4× — CITY_COORDS keys are Latin). Curated, normalized-lowercase → CITY_COORDS key.
_CITY_ALIASES = {
    "берлин": "berlin", "мюнхен": "münchen", "гамбург": "hamburg", "кёльн": "köln",
    "кельн": "köln", "франкфурт": "frankfurt", "франкфурт-на-майне": "frankfurt am main",
    "штутгарт": "stuttgart", "дюссельдорф": "düsseldorf", "гейдельберг": "heidelberg",
    "хайдельберг": "heidelberg", "мангейм": "mannheim", "манхайм": "mannheim",
    "дармштадт": "darmstadt", "майнц": "mainz", "висбаден": "wiesbaden",
    "карлсруэ": "karlsruhe", "нюрнберг": "nürnberg", "лейпциг": "leipzig",
    "дрезден": "dresden", "ганновер": "hannover", "бремен": "bremen", "эссен": "essen",
    "дортмунд": "dortmund", "бонн": "bonn", "фрайбург": "freiburg", "фрейбург": "freiburg",
    "кайзерслаутерн": "kaiserslautern", "кобленц": "koblenz",
}
# «вся Германия» / remote — no geo restriction (live finding: the user's FIRST attempt)
_NATIONWIDE = {"вся германия", "германия", "germany", "deutschland", "везде", "вся страна",
               "remote", "удалённо", "удаленно", "anywhere", "all germany"}


class RegistrationError(Exception):
    """Typed registration failure; `status` maps to the HTTP code the route returns."""

    def __init__(self, message: str, *, status: int, suggestions: list[str] | None = None):
        super().__init__(message)
        self.status = status
        self.suggestions = suggestions or []


def resolve_city(name: str) -> dict:
    """Offline city → coords via geo.CITY_COORDS (~90 German cities). Exact (normalized) match
    wins, then RU→DE aliases, then rapidfuzz suggestions (Postel: typo-tolerant input).
    Nationwide sentinels («вся Германия»/remote/…) → {ok, nationwide: True}."""
    norm = geo._normalize_city(name)
    if str(name).strip().lower() in _NATIONWIDE or norm in _NATIONWIDE:
        return {"ok": True, "nationwide": True, "name": "Germany"}
    norm = _CITY_ALIASES.get(norm, norm)
    coords = geo.CITY_COORDS.get(norm)
    if coords:
        return {"ok": True, "name": norm.title(), "lat": coords[0], "lon": coords[1]}
    # fuzzy against Latin keys AND Cyrillic aliases (a typo'd «Франкфрут» still suggests)
    pool = list(geo.CITY_COORDS.keys()) + list(_CITY_ALIASES.keys())
    matches = _fuzz_process.extract(norm, pool, limit=_SUGGESTIONS) if norm else []
    sugg = []
    for m, score, _ in matches:
        if score >= _FUZZY_FLOOR:
            latin = _CITY_ALIASES.get(m, m).title()
            if latin not in sugg:
                sugg.append(latin)
    return {"ok": False, "name": name, "suggestions": sugg}


def resolve_cities(text: str) -> dict:
    """Multi-city input (live finding: «Франкфурт, Мюнхен» was rejected): split on commas/«и»/
    slashes, resolve each. Any nationwide token → nationwide. Returns
    {ok, nationwide, cities: [{name, lat, lon}]} or {ok: False, failed, suggestions}."""
    import re as _re
    parts = [p.strip() for p in _re.split(r"[,;/]|\bи\b|\band\b", text) if p.strip()]
    if not parts:
        return {"ok": False, "failed": text, "suggestions": []}
    cities, nationwide = [], False
    for p in parts:
        loc = resolve_city(p)
        if not loc["ok"]:
            return {"ok": False, "failed": p, "suggestions": loc["suggestions"]}
        if loc.get("nationwide"):
            nationwide = True
        else:
            cities.append(loc)
    return {"ok": True, "nationwide": nationwide, "cities": cities}


def _derive_queries(profile: dict) -> list[str]:
    """Search queries from the extracted CV: target_roles first, domains as filler — deduped,
    max 4 (each query is a full scrape pass across all boards; more ≠ better)."""
    seen, out = set(), []
    for q in (profile.get("target_roles") or []) + (profile.get("domains") or []):
        q = str(q).strip()
        if q and q.lower() not in seen:
            seen.add(q.lower())
            out.append(q)
    return out[:4]


def register_user(base_cfg: dict, *, chat_id: int, name: str, city: str,
                  cv_text: str | None = None, cv_path: str | None = None,
                  queries: list[str] | None = None) -> dict:
    """The whole registration. Returns {user, queries_en, cities, summary} or raises
    RegistrationError(status=409|422|500)."""
    if not chat_id:
        raise RegistrationError("chat_id is required", status=422)
    if not (cv_text or cv_path):
        raise RegistrationError("provide cv_text or cv_path", status=422)
    if users.by_telegram_id(chat_id):
        raise RegistrationError("this Telegram account is already registered", status=409)
    if len(users.list_users()) >= users.MAX_USERS:
        raise RegistrationError(f"user cap reached ({users.MAX_USERS})", status=409)

    res = resolve_cities(city)
    if not res["ok"]:
        raise RegistrationError(f"unknown city {res['failed']!r}", status=422,
                                suggestions=res["suggestions"])
    if res["nationwide"] or not res["cities"]:
        # «вся Германия»/remote → country-wide search, NO geo anchors (falsy anchors = no geo
        # preference in geo_check/geo_mark; nothing gets the 📍 far tag)
        cities_yaml, anchors = ["Germany"], None
    else:
        cities_yaml = [f"{c['name']}, Germany" for c in res["cities"]]
        anchors = {c["name"]: {"lat": c["lat"], "lon": c["lon"],
                               "radius_km": _DEFAULT_RADIUS_KM} for c in res["cities"]}

    key = users.derive_key(name, chat_id)
    overlay = {
        "telegram": {"chat_id": int(chat_id)},
        # Persona keys are written EXPLICITLY so the new user never inherits the base profile's
        # taste via deep-merge (audit 2026-07-03: user #2 was judged on user #1's magnets/scale
        # and penalized by her role_kind_mult). magnets/repellents get filled from the CV below.
        "profile": {"summary": "", "taste_rules": [],
                    "scale": {"5": "Шабашка: сильное совпадение с целями и магнитами пользователя",
                              "4": "Почти: хороший домен или сильная команда, один минус",
                              "3": "Просто работа: нейтрально, без изюминки",
                              "2": "С киллером: явный отталкиватель",
                              "1": "Совсем мимо: нерелевантно и скучно"}},
        "judge": {"rubric_version": f"v1-{key}"},
        "slate": {"role_kind_mult": {"hands_on_engineer": 1.0, "junior": 1.0}},
        "search": {"queries_en": list(queries or []), "queries_de": list(queries or []),
                   "cities": cities_yaml},  # copies → no yaml &/* aliases
        "geo": {"anchors": anchors},
    }
    users.create_user(key, overlay)

    # CV file + free-text addendum can BOTH arrive (live finding: the document caption carried
    # «…ищу biotech, pharma, AI agents» and was lost). extract_candidate takes exactly one input,
    # so merge: read the file here and append the addendum.
    if cv_path and cv_text:
        try:
            cv_text = candidate._read_cv(cv_path) + "\n\n" + cv_text
            cv_path = None
        except Exception as e:
            users.delete_user(key)
            raise RegistrationError(f"CV extraction failed: {e}", status=500) from e

    # CV → structured profile, into the NEW user's own DB. Any failure → full rollback:
    # a user without a candidate profile would silently match on the base user's CV vectors.
    try:
        ucfg = users.load(key)
        con = db.connect(ucfg["paths"]["db"])
        try:
            profile = candidate.extract_candidate(ucfg, con, description=cv_text, cv_path=cv_path)
        finally:
            con.close()
    except Exception as e:
        users.delete_user(key)
        raise RegistrationError(f"CV extraction failed: {e}", status=500) from e

    final_queries = queries or _derive_queries(profile)
    if not final_queries:
        users.delete_user(key)
        raise RegistrationError("could not derive search queries from the CV — "
                                "pass queries explicitly", status=422)
    summary = str(profile.get("summary") or "").strip()
    # Magnets/repellents from THEIR CV extraction (candidate._SYSTEM already extracts both);
    # the safety repellents are always on for a fresh user (editable in the overlay later).
    magnets = [str(m) for m in (profile.get("magnets") or [])][:6] or ["new-domain"]
    repellents = [str(r) for r in (profile.get("repellents") or [])][:4]
    for must in ("hidden-german", "slop-text", "temp-agency"):
        if must not in repellents:
            repellents.append(must)
    users.update_overlay(key, {
        "profile": {"summary": summary, "magnets": magnets, "repellents": repellents},
        "search": {"queries_en": list(final_queries), "queries_de": list(final_queries)},
    })
    return {"user": key, "queries_en": final_queries,
            "cities": overlay["search"]["cities"], "summary": summary}
