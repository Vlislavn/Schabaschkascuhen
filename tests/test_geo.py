"""Гео: класс near/far/unknown. Far-but-in-Germany больше НЕ режется — показывается + помечается
(WS4). prefilter ничего не дропает, только считает near/far/unknown."""
from __future__ import annotations

from schabasch import db, geo
from schabasch.models import Status


def test_stuttgart_outside_radius(cfg):
    in_radius, dist = geo.geo_check("Stuttgart", cfg)
    assert in_radius is False
    assert dist is not None and dist > 40


def test_mannheim_inside_radius(cfg):
    in_radius, dist = geo.geo_check("Mannheim", cfg)
    assert in_radius is True
    assert dist is not None and dist < 40


def test_frankfurt_and_heidelberg_inside(cfg):
    assert geo.geo_check("Frankfurt am Main, HE, DE", cfg)[0] is True
    assert geo.geo_check("Heidelberg", cfg)[0] is True


def test_unknown_city_not_cut(cfg):
    in_radius, dist = geo.geo_check("Kleinkleckersdorf", cfg)
    assert in_radius is True      # неизвестный — НЕ режем
    assert dist is None           # помечаем неизвестностью


def test_normalization_handles_plz_and_region(cfg):
    assert geo.geo_check("69115 Heidelberg", cfg)[0] is True
    assert geo.geo_check("München, BY, DE", cfg)[0] is False


def test_geo_class(cfg):
    # WS4: München/Berlin = far-but-in-Germany (kept+marked), Heidelberg/Mannheim = near, unknown.
    assert geo.geo_class("München, BY, DE", cfg) == "far"
    assert geo.geo_class("Berlin", cfg) == "far"
    assert geo.geo_class("Heidelberg", cfg) == "near"
    assert geo.geo_class("Mannheim", cfg) == "near"
    assert geo.geo_class("Kleinkleckersdorf", cfg) == "unknown"


def test_geo_mark(cfg):
    m = geo.geo_mark("München, BY, DE", cfg)
    assert m["far"] is True and m["dist_km"] is not None and m["dist_km"] > 200
    assert m["anchor"] in ("Heidelberg", "Frankfurt")
    near = geo.geo_mark("Heidelberg", cfg)
    assert near["far"] is False
    unk = geo.geo_mark("Nowhereville", cfg)
    assert unk["far"] is False and unk["dist_km"] is None


def test_prefilter_keeps_far_marked_not_dropped(cfg, con):
    """WS4: far-but-in-Germany jobs are NO LONGER dropped — prefilter keeps everything and only
    counts near/far/unknown (the geo signal becomes a mark, not a cut)."""
    near = db.upsert_vacancy(con, {"source": "indeed", "url": "u/near", "title": "T",
                                   "city": "Mannheim", "description": "d" * 300})
    far = db.upsert_vacancy(con, {"source": "indeed", "url": "u/far", "title": "T",
                                  "city": "Berlin", "description": "d" * 300})
    db.upsert_vacancy(con, {"source": "indeed", "url": "u/unk", "title": "T",
                            "city": "Nowhereville", "description": "d" * 300})
    res = geo.prefilter(cfg, con)
    assert res["prefiltered"] == 0          # nothing dropped
    assert res["far"] == 1 and res["near"] == 1 and res["unknown"] == 1
    far_row = con.execute("SELECT status FROM vacancy WHERE id=?", (far,)).fetchone()
    assert far_row["status"] == Status.DESCRIBED.value   # far kept, NOT prefiltered
    near_row = con.execute("SELECT status FROM vacancy WHERE id=?", (near,)).fetchone()
    assert near_row["status"] == Status.DESCRIBED.value
