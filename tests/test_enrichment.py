"""W3: Zotero-style card enrichment (schabasch/enrichment.py) + render."""
from __future__ import annotations

import json

import pytest

from schabasch import db, enrichment as E, slate
from schabasch.models import Status


@pytest.fixture
def file_cfg(cfg, tmp_path):
    cfg = dict(cfg)
    cfg["paths"] = dict(cfg["paths"])
    cfg["paths"]["db"] = str(tmp_path / "t.sqlite3")
    return cfg


class _FakeRR:
    """Cross-encoder stub: scores a sentence by whether it carries a 'signal' keyword."""
    def compute_score(self, pairs, normalize=True):
        return [0.9 if any(k in p[1] for k in ("Python", "Master", "German", "hybrid"))
                else 0.2 for p in pairs]


def test_split_keeps_terse_requirement_lines():
    sents = E._split_sentences("We need an analyst. Master degree required. Fluent German needed here.")
    assert any("Master degree required" in s for s in sents)


def test_extract_snippets_picks_relevant_per_goal():
    jd = ("Senior Analyst role in aerospace. Requires Python and SQL skills. "
          "Master degree required for this position. Hybrid work in Frankfurt with English team. "
          "Apply through our careers portal as soon as possible please.")
    snips = E.extract_snippets(_FakeRR(), jd, per_goal=1, min_score=0.1)
    assert snips and all("goal" in s and "snippet" in s for s in snips)
    joined = " ".join(s["snippet"] for s in snips)
    assert "Master" in joined or "Python" in joined or "German" in joined


def test_deep_chain_is_single_small_tier_never_sota():
    """Escalation stays WITHIN small models: the cascade is the single small `deep_reasoning` tier +
    the always-available ollama `normalizer` fallback. `sota` is NEVER in the chain (the heavy MLX 35B
    is reserved for the supervised agent, never co-loaded), even if `enable_sota` is left True."""
    from schabasch.llm_clients import client_label
    cfg = {"llm": {"roles": {
        "deep_reasoning": {"client": "ollama", "model": "qwen3.5:4b"},
        "sota": {"client": "openai", "base_url": "https://api.kather.ai/v1", "model": "sota", "api_key": "k"},
        "normalizer": {"client": "ollama", "model": "qwen3:8b"}}},
        "deep": {"enable_sota": True}}
    assert [client_label(c) for c in E._deep_chain(cfg)] == ["qwen3.5:4b", "qwen3:8b"]
    cfg["deep"]["enable_sota"] = False
    assert [client_label(c) for c in E._deep_chain(cfg)] == ["qwen3.5:4b", "qwen3:8b"]
    # no deep_reasoning role configured → just the always-available normalizer
    cfg["llm"]["roles"].pop("deep_reasoning")
    assert [client_label(c) for c in E._deep_chain(cfg)] == ["qwen3:8b"]


def _seed(con, *, slop=10, desc="Senior Analyst. Requires Python. Master degree required here please."):
    vid = db.upsert_vacancy(con, {"source": "indeed", "url": "u/1", "title": "Senior Analyst",
                                  "company": "ACME", "city": "Frankfurt", "description": desc})
    card = {"summary_2lines": "a\nb", "slop_score": slop, "work_mode": "hybrid"}
    db.set_status(con, vid, Status.SCORED, card_json=json.dumps(card))
    return vid


def test_enrich_one_stores_and_is_cached(file_cfg, monkeypatch):
    con = db.connect(file_cfg["paths"]["db"])
    vid = _seed(con)
    monkeypatch.setattr(E, "abstractive", lambda *a, **k: (
        {"pros": ["космос"], "cons": ["нужен мастер"], "company_review": "крупная компания",
         "clean_summary": ""}, "qwen3:8b"))
    assert E.enrich_one(file_cfg, con, vid, reranker=_FakeRR()) == "enriched"
    em = E.enrichments(con)
    assert vid in em and em[vid]["pros"] == ["космос"] and em[vid]["model_used"] == "qwen3:8b"
    assert em[vid]["key_snippets"]   # snippets stored
    # second call with unchanged content_hash → cached (no recompute)
    assert E.enrich_one(file_cfg, con, vid, reranker=_FakeRR()) == "cached"
    con.close()


def test_enrich_one_degrades_without_abstractive(file_cfg, monkeypatch):
    """No abstractive tier reachable → snippets still stored (deterministic), block still renders."""
    con = db.connect(file_cfg["paths"]["db"])
    vid = _seed(con)
    monkeypatch.setattr(E, "abstractive", lambda *a, **k: (None, None))
    assert E.enrich_one(file_cfg, con, vid, reranker=_FakeRR()) == "enriched"
    em = E.enrichments(con)[vid]
    assert em["key_snippets"] and not em["pros"] and em["model_used"] is None
    con.close()


def test_enrich_slate_runs_on_slate_entries(file_cfg, monkeypatch):
    con = db.connect(file_cfg["paths"]["db"])
    vid = _seed(con)
    con.execute("INSERT INTO slate_entry (slate_date, vacancy_id, rank, slot_type) VALUES (?,?,?,?)",
                ("2026-06-16", vid, 1, "exploit"))
    con.commit()
    monkeypatch.setattr(E, "_load_reranker", lambda *a, **k: _FakeRR())
    monkeypatch.setattr(E, "abstractive", lambda *a, **k: ({"pros": ["x"], "cons": [],
                        "company_review": "", "clean_summary": ""}, "M35"))
    out = E.enrich_slate(file_cfg, con, slate_date="2026-06-16")
    assert out["enriched"] == 1
    assert E.enrichments(con)[vid]["model_used"] == "M35"
    con.close()


def test_enrichment_html_renders_block():
    html = slate._enrichment_html({
        "key_snippets": [{"goal": "требования", "snippet": "Master degree required"}],
        "pros": ["космос-домен"], "cons": ["нужен мастер"],
        "company_review": "крупный аэрокосмический игрок", "clean_summary": "аналитик в космосе",
        "model_used": "sota"})
    assert "Deep dive" in html and "Pros" in html and "Cons" in html
    assert "From the description" in html and "sota" in html and "аналитик в космосе" in html


def test_enrichment_html_empty_when_no_data():
    assert slate._enrichment_html(None) == ""
    assert slate._enrichment_html({"key_snippets": [], "pros": [], "cons": []}) == ""
