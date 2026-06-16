"""Normalizer: сырое описание → единая карточка (локальный qwen3:8b по пилоту TBC-7).

Дисциплина дешёвого выхода (are/hard-before-soft-judge): content-hash short-circuit
репостов ДО оплаты LLM. slop_score — density-smoothed 0–100 с тирами (aislop/slop-scoring),
не бинарный флаг. Промпт собран в коде с литеральным few-shot примером карточки (фикс
формата «2 строки» из пилота). После нормализации — фильтр по карточке (remote / language_de).
"""
from __future__ import annotations

from . import db
from .llm import LLMError, OllamaClient
from .models import Card, ErrorClass, FilterReason, Status

# Минимальная длина описания — ниже неё карточку не строим (огрызок → undescribable).
MIN_DESC_CHARS = 200

SYSTEM_PROMPT = """You are a job-vacancy normalizer for a personal job search in Germany \
(Heidelberg / Frankfurt area). You receive one job posting and return ONLY a JSON object with \
EXACTLY these keys and no others:
{
 "role": str,                       // normalized job title, concise
 "company": str,
 "domain": str,                     // industry/domain (e.g. "aerospace", "public sector", "biotech")
 "city": str,                       // primary work city
 "work_mode": "remote"|"hybrid"|"onsite"|"unknown",
 "language_posting": "de"|"en",     // language the posting TEXT is written in
 "language_reality": "de"|"en"|"unknown",  // language actually NEEDED TO WORK. An English posting \
that demands fluent/native German => "de". A "German nice to have / von Vorteil / is a plus" is NOT \
a requirement => keep "en". A CONDITIONAL phrasing like "local language, depending on location" or \
"German if the role is based in Germany" for a role located IN Germany => "de" (it is de-facto \
required). Judge from explicit requirements, not the posting language alone.
 "integration_potential": 0|1|2,    // help for a newcomer integrating in Germany. 0 = none, \
1 = some, 2 = strong (international team, relocation/visa support, English-friendly, path to staying)
 "summary_2lines": str,             // EXACTLY two lines separated by a single \\n, IN RUSSIAN. \
Line 1 = what the job is. Line 2 = the key requirement or catch. No more than two lines.
 "slop_score": int,                 // 0..100 how much the ad reads like generic AI/boilerplate \
slop. 0 = concrete, specific, human. ~30 = some buzzword filler. ~70+ = vague buzzword soup, \
decorative structure, says nothing concrete. Judge density, not length.
 "temp_agency_guess": bool          // true if this looks like a temp/staffing agency (Zeitarbeit, \
Arbeitnehmerüberlassung, Personaldienstleister) posting
}
No markdown, no comments, no extra keys. Output must be a single valid JSON object.

Example of a correct summary_2lines value (note: exactly two Russian lines):
"Инженер по процессам на производстве аэрокосмических компонентов под Франкфуртом.\\nНужен опыт \
Lean/Six Sigma; английского достаточно, немецкий — плюс."
"""


def _user_msg(row, truncate: int) -> str:
    desc = (row["description"] or "")[:truncate]
    remote_hint = {1: "true", 0: "false"}.get(row["is_remote_hint"], "unknown")
    return (
        f"TITLE: {row['title']}\nCOMPANY: {row['company'] or ''}\nLOCATION: {row['city'] or ''}\n"
        f"SOURCE: {row['source']}\nIS_REMOTE_HINT: {remote_hint}\n\nDESCRIPTION:\n{desc}"
    )


def _filter_card(con, vacancy_id: int, card: Card, cfg: dict | None = None) -> str:
    """Авторитетный фильтр по карточке. Возвращает 'normalized' | 'filtered'."""
    if card.work_mode == "remote":
        db.set_status(con, vacancy_id, Status.FILTERED, filter_reason=FilterReason.REMOTE_ONLY,
                      card_json=card.to_json())
        return "filtered"
    # GEO no longer DROPS here. Far-but-in-Germany onsite/hybrid jobs (a München/Berlin role the user
    # might take for a strong magnet) are KEPT and MARKED 📍 at slate-build time + routed to the
    # explore slots (geo.geo_class), per the user's "show far jobs, don't drop them" ask. Remote-only
    # and a German-language reality remain hard drops (real repellents).
    if card.language_reality == "de":
        db.set_status(con, vacancy_id, Status.FILTERED, filter_reason=FilterReason.LANGUAGE_DE,
                      card_json=card.to_json())
        return "filtered"
    # Temp-agency (Zeitarbeit/Personaldienstleister) is a hard repellent. hardfilters drops the
    # SCRAPED is_temp_agency flag; this catches the ones only the normalizer's read reveals
    # (temp_agency_guess). slop is NOT hard-dropped here — it's noisy + handled by the judge's graded
    # slop penalty (recall-first; a strong job with a sloppy ad shouldn't vanish).
    if card.temp_agency_guess:
        db.set_status(con, vacancy_id, Status.FILTERED, filter_reason=FilterReason.TEMP_AGENCY,
                      card_json=card.to_json())
        return "filtered"
    db.set_status(con, vacancy_id, Status.NORMALIZED, card_json=card.to_json())
    return "normalized"


def _make_client(cfg: dict):
    # Role-routed (llm.roles.normalizer) — defaults to ollama qwen3:8b, so behaviour is unchanged
    # unless the role is explicitly pointed at a stronger model. See schabasch/llm_clients.py.
    from .llm_clients import make_llm_client
    return make_llm_client(cfg, "normalizer")


def _normalize_row(con, client: OllamaClient, row, cfg: dict, truncate: int,
                   out: dict[str, int]) -> None:
    """Построить карточку для ОДНОЙ строки vacancy и применить фильтр. Мутирует счётчики out.

    Общее ядро normalize_pending (FSM-очередь DESCRIBED) и normalize_ids (явный список id).
    """
    desc = row["description"] or ""
    if len(desc.strip()) < MIN_DESC_CHARS:
        # Too short to build a card. Terminal card-stage FILTERED (not PREFILTERED, which means a
        # pre-description geo cut) + counted in the funnel, so these drops are visible, not silent.
        db.set_status(con, row["id"], Status.FILTERED,
                      filter_reason=FilterReason.NO_DESCRIPTION)
        out["filtered"] += 1
        return
    # 1) short-circuit: готовая карточка для идентичного описания (репост под новым URL).
    cached = db.card_by_hash(con, row["desc_hash"]) if row["desc_hash"] else None
    if cached:
        try:
            card = Card.from_llm_json(__import__("json").loads(cached))
        except ValueError:
            card = None  # битый кэш — пересчитать
        if card is not None:
            res = _filter_card(con, row["id"], card, cfg)
            out["cached"] += 1            # cache-hit считается отдельно от normalized
            if res == "filtered":
                out["filtered"] += 1
            return
    # 2) платный вызов LLM.
    try:
        obj = client.chat_json(SYSTEM_PROMPT, _user_msg(row, truncate))
        card = Card.from_llm_json(obj)
    except LLMError as e:
        db.set_error(con, row["id"], e.error_class, e.details)
        out["errors"] += 1
        return
    except ValueError as e:  # schema violation из Card.from_llm_json
        db.set_error(con, row["id"], ErrorClass.SCHEMA_VIOLATION, str(e))
        out["errors"] += 1
        return
    res = _filter_card(con, row["id"], card, cfg)
    if res == "filtered":
        out["filtered"] += 1
    else:
        out["normalized"] += 1


def normalize_pending(cfg: dict, con, *, budget: int | None = None) -> dict[str, int]:
    """DESCRIBED → карточка (+ фильтр). Бюджет K/ночь; overflow → следующая ночь."""
    llm_cfg = cfg.get("llm", {})
    if budget is None:
        budget = int(llm_cfg.get("nightly_normalize_budget", 150))
    truncate = int(llm_cfg.get("desc_truncate_chars", 6000))
    client = _make_client(cfg)

    from . import triage as _triage
    rows = _triage.select_for_normalize(cfg, con, budget=budget)
    out = {"normalized": 0, "cached": 0, "filtered": 0, "errors": 0}
    for row in rows:
        _normalize_row(con, client, row, cfg, truncate, out)

    db.log_funnel(con, "normalize", out["normalized"] + out["cached"],
                  detail=f"normalized={out['normalized']} cached={out['cached']} "
                         f"filtered={out['filtered']} errors={out['errors']}")
    return out
