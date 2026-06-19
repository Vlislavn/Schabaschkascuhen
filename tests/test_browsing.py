"""Keyless browsing adapters — entity resolution (Wikidata, mocked), name-match guard, extraction.
All hermetic: the Wikidata API is monkeypatched, so no network. Memory-safe (no model)."""
from __future__ import annotations

from schabasch.browsing import entity, extract, registry, search


def _fake_wikidata(search_results, *, country="Denmark", site="https://www.terma.com/"):
    """Canned http_get_json: wbsearchentities → results; wbgetentities → claims / country label."""
    def fake(url, *, headers=None, params=None, timeout_s=8, **kw):
        action = params.get("action")
        if action == "wbsearchentities":
            return {"search": search_results}
        if action == "wbgetentities":
            qid = params.get("ids")
            if params.get("props") == "labels":
                return {"entities": {qid: {"labels": {"en": {"value": country}}}}}
            return {"entities": {qid: {"claims": {
                "P17":  [{"mainsnak": {"datavalue": {"type": "wikibase-entityid", "value": {"id": "Q35"}}}}],
                "P856": [{"mainsnak": {"datavalue": {"type": "string", "value": site}}}],
                "P1128": [{"mainsnak": {"datavalue": {"type": "quantity", "value": {"amount": "+4200"}}}}],
                "P571": [{"mainsnak": {"datavalue": {"type": "time", "value": {"time": "+1949-00-00T00:00:00Z"}}}}],
            }}}}
        return {}
    return fake


def test_entity_resolve_picks_org_typed_name_match(monkeypatch):
    """The defense firm (org-typed + name-matched) wins over the spa — the exact Terma→Therme fix."""
    monkeypatch.setattr(entity, "http_get_json", _fake_wikidata([
        {"id": "Q1", "label": "Therme Group", "description": "thermal spa operator"},   # wrong: not org-typed
        {"id": "Q4405539", "label": "Terma A/S", "description": "Danish company"},       # right
    ]))
    e = entity.resolve("Terma Group")
    assert e is not None
    assert e["qid"] == "Q4405539" and e["label"] == "Terma A/S"
    assert e["country"] == "Denmark" and e["official_site"] == "https://www.terma.com/"
    assert e["employees"] == 4200 and e["inception"] == "1949"


def test_entity_resolve_rejects_wrong_only_candidate(monkeypatch):
    """If the ONLY hit is a wrong-type/wrong-name entity, resolve returns None (no confident-wrong)."""
    monkeypatch.setattr(entity, "http_get_json", _fake_wikidata([
        {"id": "Q1", "label": "Therme Group", "description": "thermal spa operator"},
    ]))
    assert entity.resolve("Terma Group") is None


def test_entity_resolve_graceful_on_network_error(monkeypatch):
    """A Wikidata hiccup degrades to None — never raises (a flaky source must not break a tick)."""
    def boom(*a, **k):
        raise RuntimeError("network down")
    monkeypatch.setattr(entity, "http_get_json", boom)
    assert entity.resolve("Anything GmbH") is None
    assert entity.resolve("") is None


def test_name_matches_guard():
    """Moved to the browsing layer; same guard the Wikipedia fallback + Wikidata label-check use."""
    m = entity._name_matches
    assert m("Terma Group", "Therme Group") is False
    assert m("Phoenix Medical", "Phoenix Media/Communications Group") is False
    assert m("ABB", "ABB Group") is True
    assert m("Heidelberg Materials AG", "Heidelberg Materials") is True
    assert m("", "Anything") is True


def test_extract_clean_degrades_and_extracts():
    """trafilatura adapter: empty in → None; real HTML → markdown text (never raises)."""
    assert extract.clean("") is None
    html = ("<html><body><article><h1>Acme GmbH</h1>"
            "<p>Acme builds reliable widgets for the European market since 1990.</p>"
            "</article></body></html>")
    out = extract.clean(html)
    assert out is None or "widgets" in out   # trafilatura present → extracts; absent → None (both ok)


def test_search_normalizes_and_empty(monkeypatch):
    """search() → [{title,url,snippet}]; empty query → []; ddgs backend normalized."""
    monkeypatch.setattr(search, "_ddgs", lambda q, n: [{"title": "T", "url": "u", "snippet": "s"}])
    out = search.search("Boehringer Kununu reviews")
    assert out and out[0]["url"] == "u" and set(out[0]) == {"title", "url", "snippet"}
    assert search.search("") == []


def test_search_prefers_searxng_then_falls_back(monkeypatch):
    """A configured SearXNG wins; if it's down the adapter falls back to keyless ddgs."""
    monkeypatch.setattr(search, "_ddgs", lambda q, n: [{"title": "D", "url": "dd", "snippet": ""}])
    monkeypatch.setattr(search, "_searxng", lambda q, url, n, t: [{"title": "S", "url": "sx", "snippet": ""}])
    assert search.search("x", searxng_url="http://localhost:8080")[0]["url"] == "sx"
    monkeypatch.setattr(search, "_searxng", lambda q, url, n, t: None)   # SearXNG down
    assert search.search("x", searxng_url="http://localhost:8080")[0]["url"] == "dd"


def test_registry_degrades_without_backend():
    """The registry backend (deutschland) is rejected-for-cause (breaks the ML stack) → graceful no-op."""
    assert registry.lookup("") is None
    assert registry.lookup("Some Company GmbH") is None   # no compatible backend installed → None
