"""Normalizer + Judge с мок-ollama (llm.OllamaClient.chat_json): без сети, без модели."""
from __future__ import annotations

import json

import pytest

from schabasch import db, judge, normalize
from schabasch.llm import LLMError, OllamaClient
from schabasch.models import ErrorClass, FilterReason, Status


def _card_json(**over):
    base = dict(role="Process Engineer", company="ACME", domain="aerospace", city="Frankfurt",
                work_mode="hybrid", language_posting="en", language_reality="en",
                integration_potential=2, summary_2lines="Инженер процессов.\nНужен Lean.",
                slop_score=10, temp_agency_guess=False)
    base.update(over)
    return base


def _seed_described(con, url="u/1", desc="x" * 500):
    return db.upsert_vacancy(con, {"source": "indeed", "url": url, "title": "Eng",
                                   "company": "ACME", "city": "Frankfurt", "description": desc})


# ---------------- Normalizer ----------------

def test_normalize_builds_card_and_normalizes(cfg, con, monkeypatch):
    _seed_described(con)
    monkeypatch.setattr(OllamaClient, "chat_json", lambda self, s, u: _card_json())
    out = normalize.normalize_pending(cfg, con)
    assert out["normalized"] == 1
    row = con.execute("SELECT status, card_json FROM vacancy").fetchone()
    assert row["status"] == Status.NORMALIZED.value
    assert json.loads(row["card_json"])["role"] == "Process Engineer"


def test_normalize_filters_remote(cfg, con, monkeypatch):
    _seed_described(con, url="u/remote")
    monkeypatch.setattr(OllamaClient, "chat_json", lambda self, s, u: _card_json(work_mode="remote"))
    out = normalize.normalize_pending(cfg, con)
    assert out["filtered"] == 1
    row = con.execute("SELECT status, filter_reason FROM vacancy").fetchone()
    assert row["status"] == Status.FILTERED.value
    assert row["filter_reason"] == FilterReason.REMOTE_ONLY.value


def test_normalize_filters_language_de(cfg, con, monkeypatch):
    _seed_described(con, url="u/de")
    monkeypatch.setattr(OllamaClient, "chat_json",
                        lambda self, s, u: _card_json(language_reality="de"))
    out = normalize.normalize_pending(cfg, con)
    assert out["filtered"] == 1
    assert con.execute("SELECT filter_reason FROM vacancy").fetchone()[0] == \
        FilterReason.LANGUAGE_DE.value


def test_normalize_filters_temp_agency_guess(cfg, con, monkeypatch):
    """W2: the normalizer's temp_agency_guess (Zeitarbeit) is now a hard drop (was ignored) — catches
    temp agencies the scraped is_temp_agency flag missed."""
    _seed_described(con, url="u/temp")
    monkeypatch.setattr(OllamaClient, "chat_json",
                        lambda self, s, u: _card_json(temp_agency_guess=True))
    out = normalize.normalize_pending(cfg, con)
    assert out["filtered"] == 1
    assert con.execute("SELECT filter_reason FROM vacancy").fetchone()[0] == \
        FilterReason.TEMP_AGENCY.value


def test_normalize_content_hash_short_circuit(cfg, con, monkeypatch):
    # две вакансии с ИДЕНТИЧНЫМ описанием — вторая берёт карточку из кэша (репост)
    same = "identical repost body " * 30
    _seed_described(con, url="u/a", desc=same)
    _seed_described(con, url="u/b", desc=same)
    calls = {"n": 0}

    def fake(self, s, u):
        calls["n"] += 1
        return _card_json()

    monkeypatch.setattr(OllamaClient, "chat_json", fake)
    out = normalize.normalize_pending(cfg, con)
    assert calls["n"] == 1            # LLM вызван ровно раз
    assert out["normalized"] == 1 and out["cached"] == 1


def test_normalize_min_length_floor(cfg, con, monkeypatch):
    _seed_described(con, url="u/short", desc="too short")
    monkeypatch.setattr(OllamaClient, "chat_json",
                        lambda self, s, u: (_ for _ in ()).throw(AssertionError("LLM must not be called")))
    out = normalize.normalize_pending(cfg, con)
    assert out["normalized"] == 0
    assert out["filtered"] == 1  # short-desc drop is now counted, not silent
    row = con.execute("SELECT status, filter_reason FROM vacancy").fetchone()
    # Terminal card-stage FILTERED (not PREFILTERED, which means a pre-description geo cut).
    assert row["status"] == Status.FILTERED.value
    assert row["filter_reason"] == FilterReason.NO_DESCRIPTION.value


def test_normalize_llm_error_keeps_described(cfg, con, monkeypatch):
    _seed_described(con, url="u/err")
    def boom(self, s, u):
        raise LLMError(ErrorClass.INVALID_JSON, "bad json")
    monkeypatch.setattr(OllamaClient, "chat_json", boom)
    out = normalize.normalize_pending(cfg, con)
    assert out["errors"] == 1
    row = con.execute("SELECT status, last_error_class FROM vacancy").fetchone()
    assert row["status"] == Status.DESCRIBED.value     # остаётся, подхватится завтра
    assert row["last_error_class"] == ErrorClass.INVALID_JSON.value


# ---------------- Judge ----------------

def _seed_normalized(con, url, **card_over):
    vid = db.upsert_vacancy(con, {"source": "indeed", "url": url, "title": "Eng",
                                  "company": "ACME", "description": "x" * 500})
    db.set_status(con, vid, Status.NORMALIZED,
                  card_json=json.dumps(_card_json(**card_over), ensure_ascii=False))
    return vid


def test_judge_scores_and_pins_grader_tuple(cfg, con, monkeypatch):
    _seed_normalized(con, "u/j1")
    monkeypatch.setattr(OllamaClient, "model_digest", lambda self: "sha:deadbeef")
    monkeypatch.setattr(OllamaClient, "chat_json", lambda self, s, u: {
        "score": 5, "why_tag": "space", "why_freetext": None, "explanation": "магнит"})
    out = judge.judge_pending(cfg, con)
    assert out["scored"] == 1
    row = con.execute("SELECT score, why_tag, model, model_digest, rubric_version, fewshot_hash "
                      "FROM judge_score").fetchone()
    assert row["score"] == 5 and row["why_tag"] == "space"
    assert row["model"] == "qwen3:8b" and row["model_digest"] == "sha:deadbeef"
    assert row["rubric_version"] == "test-v1" and row["fewshot_hash"]
    assert con.execute("SELECT status FROM vacancy").fetchone()[0] == Status.SCORED.value


def test_judge_invalid_tag_moves_to_freetext(cfg, con, monkeypatch):
    _seed_normalized(con, "u/j2")
    monkeypatch.setattr(OllamaClient, "model_digest", lambda self: None)
    monkeypatch.setattr(OllamaClient, "chat_json", lambda self, s, u: {
        "score": 4, "why_tag": "не-из-словаря", "why_freetext": "x", "explanation": "y"})
    judge.judge_pending(cfg, con)
    row = con.execute("SELECT why_tag, why_freetext FROM judge_score").fetchone()
    assert row["why_tag"] is None                      # тег вне словаря обнулён
    assert "[не-из-словаря]" in row["why_freetext"]     # перенесён в freetext


def test_judge_invalid_json_marks_unjudgeable(cfg, con, monkeypatch):
    _seed_normalized(con, "u/j3")
    monkeypatch.setattr(OllamaClient, "model_digest", lambda self: None)
    def boom(self, s, u):
        raise LLMError(ErrorClass.INVALID_JSON, "nope")
    monkeypatch.setattr(OllamaClient, "chat_json", boom)
    out = judge.judge_pending(cfg, con)
    assert out["errors"] == 1 and out["scored"] == 0
    assert con.execute("SELECT status FROM vacancy").fetchone()[0] == Status.NORMALIZED.value


def test_build_fewshot_from_extreme_labels(cfg, con):
    # метка score=5 + карточка с тегом из текущего словаря → попадает в few-shot
    vid = db.upsert_vacancy(con, {"source": "indeed", "url": "u/lab", "title": "T",
                                  "description": "x" * 500})
    db.set_status(con, vid, Status.NORMALIZED,
                  card_json=json.dumps(_card_json(domain="space"), ensure_ascii=False))
    db.insert_label(con, vid, {"score_1_5": 5, "why_tag": "space", "source": "bootstrap"})
    text, h = judge.build_fewshot(con, 6)
    assert "<example>" in text and "space" in text and len(h) == 16


def test_build_fewshot_excludes_stale_persona_tags(cfg, con):
    """Метка с тегом вне текущего WHY_TAGS (наследие прошлой персоны) НЕ попадает в few-shot."""
    vid = db.upsert_vacancy(con, {"source": "indeed", "url": "u/stale", "title": "T",
                                  "description": "x" * 500})
    db.set_status(con, vid, Status.NORMALIZED,
                  card_json=json.dumps(_card_json(domain="x"), ensure_ascii=False))
    db.insert_label(con, vid, {"score_1_5": 5, "why_tag": "космос", "source": "bootstrap"})
    text, _ = judge.build_fewshot(con, 6)
    assert "космос" not in text and "<example>" not in text
