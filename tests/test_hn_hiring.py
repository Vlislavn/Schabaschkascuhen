"""HN Who-is-hiring: выбор треда, регион-фильтр, парсинг компании из первой строки (без сети)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from schabasch.models import Status
from schabasch.sources import hn_hiring


def _recent(days: int = 1) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _fake_api(search_hits, children):
    def fake(url, params=None, attempts=3, **kw):
        if "search_by_date" in url:
            return {"hits": search_hits}
        return {"children": children}
    return fake


def test_scrape_filters_and_parses(cfg, con, monkeypatch):
    hits = [
        {"title": "Ask HN: Who wants to be hired? (July 2026)", "objectID": "1"},
        {"title": "Ask HN: Who is hiring? (July 2026)", "objectID": "2"},
    ]
    children = [
        {"id": 101, "author": "founder1", "created_at": _recent(),
         "text": "<p>ACME GmbH | Business Analyst | Berlin, Germany | Hybrid</p>"
                 "<p>We build process tooling. Apply at acme.example</p>"},
        {"id": 102, "author": "founder2", "created_at": _recent(),
         "text": "<p>USCorp | Backend Engineer | New York, onsite only</p>"},   # вне региона
        {"id": 103, "author": "founder3", "created_at": _recent(days=60),
         "text": "<p>OldCo | Analyst | Remote (Europe)</p>"},                    # старше окна треда
        {"id": 104, "author": None, "text": None, "created_at": _recent()},      # deleted
    ]
    monkeypatch.setattr(hn_hiring, "http_get_json", _fake_api(hits, children))
    counts = hn_hiring.scrape(cfg, con)
    assert counts == {"hn": 1}
    row = con.execute("SELECT source, title, company, status, description FROM vacancy "
                      "WHERE url = 'https://news.ycombinator.com/item?id=101'").fetchone()
    assert row is not None
    assert row["source"] == "hn" and row["status"] == Status.DESCRIBED.value
    assert row["company"] == "ACME GmbH"
    assert row["title"].startswith("ACME GmbH | Business Analyst")
    assert "founder1" in row["description"]
    assert con.execute("SELECT 1 FROM vacancy WHERE url LIKE '%id=102'").fetchone() is None


def test_scrape_no_thread(cfg, con, monkeypatch):
    monkeypatch.setattr(hn_hiring, "http_get_json", _fake_api([], []))
    assert hn_hiring.scrape(cfg, con) == {"hn": 0}
    d = con.execute("SELECT detail FROM funnel_log WHERE source='hn'").fetchone()
    assert d is not None and "no whoishiring" in d["detail"]


def test_company_from_title():
    assert hn_hiring._company_from_title("ACME GmbH | Role | Loc") == "ACME GmbH"
    assert hn_hiring._company_from_title("We are hiring | Role") is None
    assert hn_hiring._company_from_title("no pipes here") is None


def test_region_re_uses_cfg_cities(cfg):
    r = hn_hiring._region_re(cfg)
    assert r.search("office in Heidelberg")
    assert r.search("Remote (EU)")
    assert not r.search("Reutlingen only")  # 'eu' внутри слова не матчится
