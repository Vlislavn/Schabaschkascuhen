"""Геокодированный гео-фильтр: офлайн-таблица городов → реальное расстояние до якорей.

Радиусу борда не верить (измерено: Stuttgart ~85 км в выдаче Heidelberg@50 — утечка миль).
Город неизвестен → НЕ режем (возврат (True, None)): ложный отказ дороже лишней карточки;
авторитетный фильтр по карточке (work_mode/language) ловит остаток. Паттерн hard-before-soft:
детерминированный дешёвый чекер ДО LLM.
"""
from __future__ import annotations

import math

from . import db
from .models import Status

# Офлайн-таблица: города/общины BW + Hessen + RLP вокруг Heidelberg/Frankfurt +
# дальние якоря для отрицательных проверок. {нормализованное имя: (lat, lon)}.
CITY_COORDS: dict[str, tuple[float, float]] = {
    # --- Rhein-Neckar / вокруг Heidelberg ---
    "heidelberg": (49.3988, 8.6724),
    "mannheim": (49.4875, 8.4660),
    "ludwigshafen": (49.4774, 8.4452),
    "ludwigshafen am rhein": (49.4774, 8.4452),
    "weinheim": (49.5450, 8.6700),
    "schwetzingen": (49.3830, 8.5700),
    "wiesloch": (49.2940, 8.6980),
    "walldorf": (49.3050, 8.6420),
    "leimen": (49.3470, 8.6880),
    "eppelheim": (49.4030, 8.6390),
    "sandhausen": (49.3400, 8.6560),
    "hockenheim": (49.3210, 8.5490),
    "sinsheim": (49.2520, 8.8780),
    "bruchsal": (49.1240, 8.5980),
    "speyer": (49.3170, 8.4310),
    "worms": (49.6340, 8.3590),
    "frankenthal": (49.5350, 8.3540),
    "neustadt": (49.3500, 8.1390),
    "landau": (49.1980, 8.1180),
    "bensheim": (49.6810, 8.6230),
    "heppenheim": (49.6420, 8.6360),
    "lampertheim": (49.5980, 8.4700),
    "viernheim": (49.5400, 8.5780),
    "hemsbach": (49.5910, 8.6450),
    "ladenburg": (49.4720, 8.6090),
    "neckargemünd": (49.3920, 8.7980),
    "mosbach": (49.3530, 9.1440),
    "buchen": (49.5220, 9.3220),
    "tauberbischofsheim": (49.6230, 9.6630),
    "karlsruhe": (49.0069, 8.4037),
    "pforzheim": (48.8920, 8.6940),
    "ettlingen": (48.9410, 8.4070),
    "rastatt": (48.8590, 8.2090),
    "baden-baden": (48.7610, 8.2410),
    # --- Rhein-Main / вокруг Frankfurt ---
    "frankfurt": (50.1109, 8.6821),
    "frankfurt am main": (50.1109, 8.6821),
    "offenbach": (50.1055, 8.7610),
    "offenbach am main": (50.1055, 8.7610),
    "hanau": (50.1330, 8.9160),
    "darmstadt": (49.8728, 8.6512),
    "wiesbaden": (50.0826, 8.2400),
    "mainz": (49.9929, 8.2473),
    "rüsselsheim": (49.9930, 8.4130),
    "rüsselsheim am main": (49.9930, 8.4130),
    "bad homburg": (50.2270, 8.6180),
    "oberursel": (50.2010, 8.5770),
    "kelkheim": (50.1370, 8.4500),
    "hofheim": (50.0900, 8.4490),
    "eschborn": (50.1430, 8.5700),
    "kronberg": (50.1830, 8.5210),
    "friedberg": (50.3380, 8.7560),
    "bad nauheim": (50.3650, 8.7390),
    "gießen": (50.5840, 8.6784),
    "wetzlar": (50.5630, 8.5010),
    "aschaffenburg": (49.9740, 9.1480),
    "dieburg": (49.8980, 8.8410),
    "rödermark": (49.9870, 8.8190),
    "dreieich": (50.0240, 8.6960),
    "neu-isenburg": (50.0490, 8.6940),
    "langen": (49.9930, 8.6660),
    "mörfelden-walldorf": (49.9940, 8.5840),
    "groß-gerau": (49.9220, 8.4810),
    "bad vilbel": (50.1780, 8.7370),
    "maintal": (50.1480, 8.8360),
    "rodgau": (50.0190, 8.8850),
    "limburg": (50.3840, 8.0640),
    "marburg": (50.8090, 8.7700),
    "fulda": (50.5550, 9.6800),
    "kaiserslautern": (49.4400, 7.7490),
    "koblenz": (50.3569, 7.5890),
    # --- дальние якоря (для отрицательных проверок) ---
    "stuttgart": (48.7758, 9.1829),
    "münchen": (48.1351, 11.5820),
    "munich": (48.1351, 11.5820),
    "berlin": (52.5200, 13.4050),
    "hamburg": (53.5511, 9.9937),
    "köln": (50.9375, 6.9603),
    "cologne": (50.9375, 6.9603),
    "düsseldorf": (51.2277, 6.7735),
    "nürnberg": (49.4521, 11.0767),
    "leipzig": (51.3397, 12.3731),
    "dresden": (51.0504, 13.7373),
    "hannover": (52.3759, 9.7320),
    "bremen": (53.0793, 8.8017),
    "essen": (51.4556, 7.0116),
    "dortmund": (51.5136, 7.4653),
    "freiburg": (47.9990, 7.8421),
    "ulm": (48.4011, 9.9876),
    "würzburg": (49.7913, 9.9534),
}


def _normalize_city(city: str | None) -> str:
    """'Frankfurt am Main, HE, DE' → 'frankfurt am main'; срезает регион/страну после запятой."""
    if not city:
        return ""
    s = str(city).split(",")[0].strip().lower()
    # убрать индекс PLZ в начале ('69115 heidelberg' → 'heidelberg')
    parts = s.split()
    if parts and parts[0].isdigit():
        parts = parts[1:]
    return " ".join(parts).strip()


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Расстояние между двумя точками по большой окружности, км."""
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def geo_check(city: str | None, cfg: dict) -> tuple[bool, float | None]:
    """(в радиусе?, расстояние_км до ближайшего якоря). Неизвестный город → (True, None)."""
    name = _normalize_city(city)
    coords = CITY_COORDS.get(name)
    if coords is None:
        return True, None  # не знаем — не режем, помечаем
    anchors = cfg.get("geo", {}).get("anchors", {})
    best: float | None = None
    in_radius = False
    for a in anchors.values():
        d = haversine_km(coords[0], coords[1], a["lat"], a["lon"])
        if best is None or d < best:
            best = d
        if d <= float(a.get("radius_km", 40)):
            in_radius = True
    return in_radius, (round(best, 1) if best is not None else None)


def geo_class(city: str | None, cfg: dict) -> str:
    """Classify a city as 'near' (within an anchor radius), 'far' (a KNOWN German city out of every
    radius), or 'unknown' (not in the offline table). The table is all-German, so a known city out
    of radius = far-but-in-Germany → SHOW IT MARKED + route to explore, not drop (the user's geo
    ask). Unknown stays kept-unmarked (recall > precision; the card-level filter catches the rest)."""
    in_radius, dist = geo_check(city, cfg)
    if _normalize_city(city) not in CITY_COORDS:
        return "unknown"
    return "near" if in_radius else "far"


def geo_mark(city: str | None, cfg: dict) -> dict:
    """Slate-display geo info: {far, dist_km, anchor}. far = a KNOWN German city out of every anchor
    radius (→ quiet 📍 'далеко' tag + explore routing). dist_km + anchor name = distance to the
    nearest anchor. unknown/near → far=False (no tag)."""
    name = _normalize_city(city)
    coords = CITY_COORDS.get(name)
    if coords is None:
        return {"far": False, "dist_km": None, "anchor": None}
    anchors = cfg.get("geo", {}).get("anchors", {})
    best_name: str | None = None
    best_d: float | None = None
    in_radius = False
    for an, a in anchors.items():
        d = haversine_km(coords[0], coords[1], a["lat"], a["lon"])
        if best_d is None or d < best_d:
            best_d, best_name = d, an
        if d <= float(a.get("radius_km", 40)):
            in_radius = True
    return {"far": not in_radius, "dist_km": round(best_d, 1) if best_d is not None else None,
            "anchor": best_name}


def prefilter(cfg: dict, con) -> dict[str, int]:
    """Грубый гео-маркер для NEW+DESCRIBED без карточки.

    Far-but-in-Germany cities are NO LONGER dropped (the user wants them shown + marked + routed to
    explore — a München/Berlin role she might consider for a strong magnet). Nothing is prefiltered
    out here anymore; the funnel just records near/far/unknown counts. near jobs are returned first
    so the capped nightly normalize budget isn't starved by far ones (see pipeline)."""
    rows = con.execute(
        "SELECT id, city FROM vacancy WHERE status IN (?, ?) AND card_json IS NULL",
        (Status.NEW.value, Status.DESCRIBED.value),
    ).fetchall()
    near = far = unknown = 0
    for row in rows:
        cls = geo_class(row["city"], cfg)
        near += cls == "near"
        far += cls == "far"
        unknown += cls == "unknown"
    db.log_funnel(con, "prefilter", 0, detail=f"near={near} far={far} unknown={unknown} (none dropped)")
    return {"kept": near + far + unknown, "prefiltered": 0,
            "near": near, "far": far, "unknown": unknown}
