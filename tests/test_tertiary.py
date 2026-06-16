"""Третичные фетчеры: парсер GTJ-тайтла + регион-фильтр (без сети)."""
from __future__ import annotations

from schabasch.sources.tertiary import _parse_gtj_title, fetch_arbeitnow
from schabasch import db
from schabasch.models import Status


def test_parse_gtj_title_full():
    role, city, company = _parse_gtj_title(
        "Mitarbeiter IT-Support (m/w/d) - Berlin @ Epikur Software GmbH [39.000 €]")
    assert role.startswith("Mitarbeiter IT-Support")
    assert city == "Berlin"
    assert company == "Epikur Software GmbH"


def test_parse_gtj_title_no_salary():
    role, city, company = _parse_gtj_title("Backend Engineer - München @ ACME GmbH")
    assert role == "Backend Engineer" and city == "München" and company == "ACME GmbH"


def test_parse_gtj_title_fallback():
    role, city, company = _parse_gtj_title("Some weird title without structure")
    assert role == "Some weird title without structure"
    assert city is None and company is None


def test_arbeitnow_region_filter_and_described(cfg, con, monkeypatch):
    # мок http_get_json: одна вакансия в регионе (Frankfurt, с описанием), одна далеко (Berlin)
    page = {"data": [
        {"url": "an/1", "title": "Eng", "company_name": "ACME", "location": "Frankfurt, Hessen, Germany",
         "remote": False, "description": "full desc " * 30, "tags": ["python"]},
        {"url": "an/2", "title": "Eng2", "company_name": "BCO", "location": "Berlin, Germany",
         "remote": False, "description": "x" * 200, "tags": []},
    ]}
    calls = {"n": 0}
    def fake(url, params=None, attempts=3, **kw):
        calls["n"] += 1
        return page if calls["n"] == 1 else {"data": []}
    monkeypatch.setattr("schabasch.sources.tertiary.http_get_json", fake)
    n = fetch_arbeitnow(cfg, con, max_pages=2)
    assert n == 1  # только франкфуртская прошла регион-фильтр
    row = con.execute("SELECT city, status FROM vacancy WHERE url='an/1'").fetchone()
    assert row["city"] == "Frankfurt" and row["status"] == Status.DESCRIBED.value
    assert con.execute("SELECT 1 FROM vacancy WHERE url='an/2'").fetchone() is None  # Berlin отрезан
