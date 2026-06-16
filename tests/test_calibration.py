"""CV-калибровка судьи с мок-ollama: проверяем честный расчёт (folds, SEM, gate-логика)."""
from __future__ import annotations

import json

from schabasch import calibration, db
from schabasch.llm import OllamaClient
from schabasch.models import Status


def _seed_label(con, url, *, score, intent, domain):
    vid = db.upsert_vacancy(con, {"source": "indeed", "url": url, "title": "T",
                                  "company": f"Co{url}", "description": "x" * 500})
    card = dict(role="r", company="c", domain=domain, city="Frankfurt", work_mode="hybrid",
                language_posting="en", language_reality="en", integration_potential=2,
                summary_2lines="a\nb", slop_score=10, temp_agency_guess=False)
    db.set_status(con, vid, Status.NORMALIZED, card_json=json.dumps(card, ensure_ascii=False))
    db.insert_label(con, vid, {"score_1_5": score, "interview": intent, "source": "bootstrap",
                               "why_tag": "космос" if score == 5 else "скучная-роль"})


def test_cv_perfect_agreement_passes_gate(cfg, con, monkeypatch):
    # 6 позитивов (космос, intent=1), 6 негативов (boring, intent=0)
    for i in range(6):
        _seed_label(con, f"pos/{i}", score=5, intent=1, domain="космос")
    for i in range(6):
        _seed_label(con, f"neg/{i}", score=1, intent=0, domain="boring")
    # судья: космос → 5, иначе 1 → идеальное согласие
    monkeypatch.setattr(OllamaClient, "model_digest", lambda self: None)
    monkeypatch.setattr(OllamaClient, "chat_json",
                        lambda self, s, u: {"score": 5 if "космос" in u else 1})
    res = calibration.cross_validate(cfg, con, folds=3, runs=1)
    assert res["mean_agreement"] == 1.0
    assert res["gate_pass"] is True
    assert res["verdict_stability"] == 1.0
    assert res["unjudgeable"] == 0
    assert len(res["fold_agreements"]) == 3


def test_cv_majority_baseline_blocks_trivial(cfg, con, monkeypatch):
    # перекос: 10 позитивов, 2 негатива; судья всегда говорит 5 → agreement ~0.83, но
    # majority baseline тоже высокий → gate по baseline+10pp не проходит
    for i in range(10):
        _seed_label(con, f"pos/{i}", score=5, intent=1, domain="космос")
    for i in range(2):
        _seed_label(con, f"neg/{i}", score=1, intent=0, domain="boring")
    monkeypatch.setattr(OllamaClient, "model_digest", lambda self: None)
    monkeypatch.setattr(OllamaClient, "chat_json", lambda self, s, u: {"score": 5})
    res = calibration.cross_validate(cfg, con, folds=2, runs=1)
    assert res["majority_baseline"] >= 0.8
    # agreement (~0.83) не превышает baseline+0.10 → гейт по baseline валит
    assert res["gate_detail"]["agreement>=baseline+0.10"] is False
    assert res["gate_pass"] is False


def test_cv_unjudgeable_excluded(cfg, con, monkeypatch):
    from schabasch.llm import LLMError
    from schabasch.models import ErrorClass
    for i in range(6):
        _seed_label(con, f"pos/{i}", score=5, intent=1, domain="космос")
    for i in range(6):
        _seed_label(con, f"neg/{i}", score=1, intent=0, domain="boring")
    # каждая третья карточка — невалидный JSON
    state = {"n": 0}
    def flaky(self, s, u):
        state["n"] += 1
        if state["n"] % 3 == 0:
            raise LLMError(ErrorClass.INVALID_JSON, "x")
        return {"score": 5 if "космос" in u else 1}
    monkeypatch.setattr(OllamaClient, "model_digest", lambda self: None)
    monkeypatch.setattr(OllamaClient, "chat_json", flaky)
    res = calibration.cross_validate(cfg, con, folds=3, runs=1)
    assert res["unjudgeable"] >= 1
    assert res["n_labels"] == 12


def test_cv_too_few_labels(cfg, con):
    _seed_label(con, "x/1", score=5, intent=1, domain="космос")
    res = calibration.cross_validate(cfg, con, folds=5, runs=1)
    assert "error" in res
