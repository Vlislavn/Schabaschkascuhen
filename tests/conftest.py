"""Общие фикстуры: in-memory SQLite, минимальный cfg, фабрика карточек."""
from __future__ import annotations

import json

import pytest

from schabasch import db
from schabasch.models import Card, Status


@pytest.fixture
def con():
    c = db.connect(":memory:")
    yield c
    c.close()


@pytest.fixture
def cfg():
    return {
        "profile": {
            "summary": "ML-инженер, Heidelberg, ищет новую область.",
            "scale": {"5": "шабашка", "1": "офисная мышь"},
            # full base-profile repellent set: the hard drops are gated per-user on these
            # (multi-user de-personalization 2026-07-03) — tests exercise the drops ON
            "magnets": ["space"],
            "repellents": ["hidden-german", "biotech", "slop-text", "boring-role",
                           "remote-only", "temp-agency"],
        },
        "search": {
            "queries_en": ["aerospace"], "queries_de": ["Raumfahrt"],
            "cities": ["Heidelberg, Germany", "Frankfurt am Main, Germany"],
            "hours_old": 48, "results_wanted": 5, "sources": ["indeed", "linkedin", "arbeitsagentur"],
        },
        "geo": {"anchors": {
            "Heidelberg": {"lat": 49.3988, "lon": 8.6724, "radius_km": 40},
            "Frankfurt": {"lat": 50.1109, "lon": 8.6821, "radius_km": 40},
        }},
        "llm": {"normalizer_model": "qwen3:8b", "judge_model": "qwen3:8b", "num_ctx": 8192,
                "temperature": 0.1, "desc_truncate_chars": 6000, "nightly_normalize_budget": 150},
        "judge": {"rubric_version": "test-v1", "fewshot_max": 6},
        "slate": {"exploit": 8, "explore": 2, "max_per_company": 3, "port": 8787,
                  "triage_blend": 0.5, "xenc_blend": 0.3,
                  "fit_gate_floor": 0.5, "hyre_blend": 0.1, "fit_warn_threshold": 0.4},
        "triage": {
            "cutoffs": {"must": 4.5, "should": 3.5, "could": 2.0},
            "drop_priorities": ["drop"],
            "min_labels_to_train": 30,
            "audit_sample_per_tick": 1,
        },
        "features": {
            "model": "BAAI/bge-m3",
            "reranker": "BAAI/bge-reranker-v2-m3",
            "rerank_top_k": 30,
            "coverage_tau": 0.6,
            "esco_csv": None,
            "hyre": False,   # off in unit tests (no ollama/bge-m3); HyRE test enables + mocks it
            "llm_cov": False,   # off in unit tests; mocked where needed
            "eligibility": False,   # off in unit tests; eligibility test enables + mocks the qwen call
            "fit_weights": {"xenc": 0.6, "llm_cov": 0.4},
        },
        "eligibility": {"floor": 0.35, "mid": 0.6},
        "paths": {"db": ":memory:", "golden_csv": "/tmp/golden.csv", "slate_dir": "/tmp/slates",
                  "model_dir": "/tmp/schabasch_models", "agent_workdir": "/tmp/schabasch_agent"},
    }


def make_card(**over) -> dict:
    base = dict(role="Process Engineer", company="ACME GmbH", domain="aerospace",
                city="Frankfurt", work_mode="hybrid", language_posting="en",
                language_reality="en", integration_potential=2,
                summary_2lines="Строка1\nСтрока2", slop_score=10, temp_agency_guess=False)
    base.update(over)
    return base


def seed_scored(con, vacancy_id_url: str, *, score: int, company: str, work_mode="hybrid",
                integration=2, slop=10, status=Status.SCORED, title=None, city="Frankfurt") -> int:
    """Создать вакансию + карточку + judge_score со статусом SCORED. Возвращает vacancy id."""
    vid = db.upsert_vacancy(con, {"source": "indeed", "url": vacancy_id_url,
                                  "title": title or f"Job {vacancy_id_url}", "company": company,
                                  "city": city, "description": "x" * 500})
    card = make_card(company=company, city=city, work_mode=work_mode,
                     integration_potential=integration, slop_score=slop)
    db.set_status(con, vid, status, card_json=json.dumps(card, ensure_ascii=False))
    db.insert_judge_score(con, vid, {"score": score, "why_tag": None, "why_freetext": None,
                                     "explanation": "test", "model": "qwen3:8b",
                                     "model_digest": "d", "rubric_version": "test-v1",
                                     "fewshot_hash": "h"})
    return vid
