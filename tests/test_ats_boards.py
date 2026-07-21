"""ATS-борды: slug-probe c кэшем в sidecar ats_board, регион-фильтр, идемпотентность (без сети)."""
from __future__ import annotations

from schabasch.models import ErrorClass, Status
from schabasch.llm import LLMError
from schabasch.sources import ats_boards


_GH_JOBS = {"jobs": [
    {"absolute_url": "https://boards.greenhouse.io/acmegmbh/jobs/1", "title": "Business Analyst",
     "location": {"name": "Frankfurt, Germany"}, "updated_at": "2026-07-01T00:00:00Z",
     "content": "Long description " * 20},
    {"absolute_url": "https://boards.greenhouse.io/acmegmbh/jobs/2", "title": "BA Berlin",
     "location": {"name": "Berlin, Germany"}, "updated_at": "2026-07-01T00:00:00Z",
     "content": "x" * 100},                                        # вне радиуса → дроп
    {"absolute_url": "https://boards.greenhouse.io/acmegmbh/jobs/3", "title": "Remote BA",
     "location": {"name": "Remote - Germany"}, "updated_at": "2026-07-01T00:00:00Z",
     "content": "y" * 100},                                        # remote → берём с хинтом
]}


def _router(calls):
    def fake(url, params=None, attempts=3, **kw):
        calls.append(url)
        if "boards-api.greenhouse.io/v1/boards/acmegmbh/jobs" in url:
            return _GH_JOBS
        raise LLMError(ErrorClass.HTTP_ERROR, "HTTP 404 (permanent)")
    return fake


def test_probe_fetch_cache(cfg, con, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(ats_boards, "http_get_json", _router(calls))
    monkeypatch.setattr("time.sleep", lambda s: None)
    # personio идёт через requests.get — пусть тоже промахивается
    import requests
    monkeypatch.setattr("requests.get",
                        lambda *a, **kw: (_ for _ in ()).throw(requests.ConnectionError("down")))
    cfg["search"]["ats_companies"] = ["ACME GmbH", "Nowhere Inc"]

    counts = ats_boards.scrape(cfg, con)
    assert counts == {"greenhouse": 2}
    row = con.execute("SELECT company, ats, slug, last_ok FROM ats_board "
                      "WHERE company = 'ACME GmbH'").fetchone()
    assert row["ats"] == "greenhouse" and row["slug"] == "acmegmbh" and row["last_ok"]
    miss = con.execute("SELECT ats, last_ok FROM ats_board WHERE company = 'Nowhere Inc'").fetchone()
    assert miss["ats"] == "none" and miss["last_ok"] is None

    v = con.execute("SELECT source, company, city, status FROM vacancy "
                    "WHERE url LIKE '%jobs/1'").fetchone()
    assert v["source"] == "ats:greenhouse" and v["company"] == "ACME GmbH"
    assert v["city"] == "Frankfurt" and v["status"] == Status.DESCRIBED.value
    assert con.execute("SELECT 1 FROM vacancy WHERE url LIKE '%jobs/2'").fetchone() is None
    remote = con.execute("SELECT is_remote_hint FROM vacancy WHERE url LIKE '%jobs/3'").fetchone()
    assert remote["is_remote_hint"] == 1

    # второй тик: probe закэширован (ни одного нового probe-вызова к чужим ATS), всё reseen
    n_calls = len(calls)
    counts2 = ats_boards.scrape(cfg, con)
    assert counts2 == {"greenhouse": 0}
    probe_calls = [c for c in calls[n_calls:] if "acmegmbh/jobs" not in c]
    assert probe_calls == []  # только fetch известного борда, никаких повторных проб


def test_slugs():
    assert ats_boards._slugs("ACME GmbH") == ["acmegmbh", "acme-gmbh"]
    assert ats_boards._slugs("Ada") == ["ada"]


def test_smartrecruiters_mapping(monkeypatch):
    listing = {"content": [
        {"id": "111", "name": "BA", "company": {"name": "ACME GmbH"},
         "releasedDate": "2026-07-01T00:00:00Z",
         "location": {"city": "Frankfurt", "country": "de", "remote": False,
                      "fullLocation": "Frankfurt, Germany"}},
        {"id": "222", "name": "PM", "company": {"name": "ACME GmbH"},
         "releasedDate": "2026-07-01T00:00:00Z",
         "location": {"city": "New York", "country": "us", "remote": False,
                      "fullLocation": "New York, USA"}},
    ]}
    detail = {"jobAd": {"sections": {"jobDescription": {"text": "desc text"},
                                     "qualifications": {"text": "quals"}}}}
    monkeypatch.setattr(ats_boards, "http_get_json",
                        lambda url, params=None, attempts=1, **kw:
                        detail if url.endswith("/111") else listing)
    out = ats_boards._smartrecruiters("acme")
    de = next(p for p in out if p["url"].endswith("/111"))
    us = next(p for p in out if p["url"].endswith("/222"))
    assert de["description"] == "desc text\n\nquals" and de["city"] == "Frankfurt"
    assert us["description"] is None  # detail (N+1) не дёргается вне DE/remote
    assert de["company_name"] == "ACME GmbH" and de["date_posted"].startswith("2026-07-01")
