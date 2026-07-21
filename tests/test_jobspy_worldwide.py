"""search.remote_worldwide: the extra fully-remote worldwide LinkedIn pass (per-user knob)."""
from __future__ import annotations

from schabasch.sources import jobspy_source


def _run(cfg, con, monkeypatch):
    calls = []

    def fake(**kw):
        calls.append(kw)
        return None   # 0 rows — we only assert the scrape matrix

    monkeypatch.setattr(jobspy_source, "_scrape_jobs", fake)
    jobspy_source.scrape(cfg, con)
    return calls


def test_worldwide_remote_pass_linkedin_only(cfg, con, monkeypatch):
    c = dict(cfg)
    c["search"] = {**cfg["search"], "cities": ["Frankfurt, Germany"],
                   "queries_en": ["agentic AI"], "sources": ["indeed", "linkedin"],
                   "remote_worldwide": True}
    calls = _run(c, con, monkeypatch)
    ww = [k for k in calls if k.get("location") == "Worldwide"]
    assert ww and all(k["site_name"] == ["linkedin"] and k.get("is_remote") is True for k in ww)
    # indeed never gets a worldwide pass (country_indeed is per-country)
    assert all(k["site_name"] != ["indeed"] for k in ww)
    # the normal local passes carry NO is_remote filter
    local = [k for k in calls if k.get("location") != "Worldwide"]
    assert local and all("is_remote" not in k for k in local)


def test_worldwide_off_by_default(cfg, con, monkeypatch):
    c = dict(cfg)
    c["search"] = {**cfg["search"], "cities": ["Frankfurt, Germany"],
                   "queries_en": ["agentic AI"], "sources": ["linkedin"]}
    calls = _run(c, con, monkeypatch)
    assert all(k.get("location") != "Worldwide" for k in calls)
