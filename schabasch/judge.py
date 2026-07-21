"""Judge: карточка → оценка 1–5 + why-тег/free-text + объяснение (локальный qwen3:8b).

Промпт собран в коде (claude-code/tool-prompts-fewshot): рубрика из профиля + блок
<example>+<commentary> из СВЕЖИХ крайних меток Влада (1 и 5), словарь тегов
интерполируется из models.WHY_TAGS (примеры не дрейфуют от схемы). Полный grader-tuple
пинуется в каждой строке JudgeScore (are/pinned-judge-model): model+digest+rubric+fewshot_hash.

Шкала 1–5 калибруется якорными примерами. Если CV-гейт ≥75% не берётся за 3 итерации —
Plan B (setwise/best-worst, llm-rankers) по решению Влада (см. ROADMAP Phase 2).
"""
from __future__ import annotations

import hashlib
import json

from . import db
from .llm import LLMError, OllamaClient
from .models import WHY_TAGS, Status

# Fallback single-user vocabulary (the original persona). Per-user magnets/repellents come from
# cfg.profile via _persona() — WHY_TAGS is only the default when a profile doesn't define them.
MAGNETS = WHY_TAGS[:6]
REPELLENTS = WHY_TAGS[6:]


def _persona(cfg: dict) -> tuple[list[str], list[str]]:
    """Per-user (magnets, repellents) tag lists from cfg.profile — the multi-user fix: the judge
    used to read the frozen WHY_TAGS for every user, so user #2 was scored against user #1's
    taste (e.g. `biotech` as a repellent for someone whose target domain IS biotech)."""
    p = (cfg or {}).get("profile") or {}
    magnets = [str(t) for t in (p.get("magnets") or [])] or list(MAGNETS)
    repellents = [str(t) for t in (p.get("repellents") or [])] or list(REPELLENTS)
    return magnets, repellents


def _card_brief(card: dict) -> str:
    return (
        f"role: {card.get('role','')}\ndomain: {card.get('domain','')}\n"
        f"company: {card.get('company','')}\ncity: {card.get('city','')}\n"
        f"work_mode: {card.get('work_mode','')}\n"
        f"language_reality: {card.get('language_reality','')}\n"
        f"integration_potential: {card.get('integration_potential','')}\n"
        f"slop_score: {card.get('slop_score','')}\n"
        f"summary: {card.get('summary_2lines','')}"
    )


def render_fewshot(rows) -> tuple[str, str]:
    """Собрать блок <example>+<commentary> из переданных строк меток (+ их card_json).

    Каждая строка должна иметь поля: score_1_5, why_tag, why_freetext, card_json.
    Возвращает (fewshot_text, fewshot_hash). Выделено отдельно для fold-restricted CV.
    """
    blocks: list[str] = []
    for r in rows:
        try:
            card = json.loads(r["card_json"])
        except (TypeError, json.JSONDecodeError):
            continue
        score = int(r["score_1_5"])
        tag = r["why_tag"] or ""
        if score == 5:
            principle = (f"магнит «{tag}» → шабашка (5)" if tag
                         else "сильный магнит + локация + интеграция → шабашка (5)")
        else:
            principle = (f"отталкиватель «{tag}» → офисная мышь (1)" if tag
                         else "скука/стоп-фактор, ничего не даёт → офисная мышь (1)")
        ft = (r["why_freetext"] or "").strip()
        blocks.append(
            "<example>\n"
            f"CARD:\n{_card_brief(card)}\n"
            f"SCORE: {score}\n"
            f"WHY_TAG: {tag}\n"
            + (f"NOTE: {ft}\n" if ft else "")
            + f"<commentary>{principle}</commentary>\n"
            "</example>"
        )
    text = "\n".join(blocks)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return text, digest


def build_fewshot(con, max_n: int, vocab: list[str] | None = None) -> tuple[str, str]:
    """(fewshot_text, fewshot_hash) из СВЕЖАЙШИХ крайних меток, newest-first.

    Крайние = негативный якорь (score<=2: 👎) ∪ сильный позитив (score=5: ⭐). Раньше брали
    ровно {1,5}, но единственная веб-кнопка для негатива даёт 👎=2 (литеральная 1 в UI нет) —
    значит 👎-фидбек НИКОГДА не попадал в few-shot и судья не учился отталкивателям с реальных
    меток. <=2 чинит это; render_fewshot мапит score==5→магнит, иначе→отталкиватель.
    Метки с why_tag вне текущего словаря (наследие прошлой персоны/рубрики) исключаются,
    чтобы few-shot не учил судью чужими тегами; why_tag=NULL допускается. `vocab` — словарь тегов
    ЭТОГО пользователя (магниты+отталкиватели из cfg.profile); None → WHY_TAGS (legacy)."""
    vocab = list(vocab) if vocab else list(WHY_TAGS)
    placeholders = ",".join("?" for _ in vocab)
    rows = con.execute(
        f"""SELECT l.score_1_5, l.why_tag, l.why_freetext, v.card_json, l.created_at
           FROM label l JOIN vacancy v ON v.id = l.vacancy_id
           WHERE (l.score_1_5 <= 2 OR l.score_1_5 = 5) AND v.card_json IS NOT NULL
             AND (l.why_tag IS NULL OR l.why_tag IN ({placeholders}))
           ORDER BY l.created_at DESC, l.id DESC LIMIT ?""",
        (*vocab, max_n),
    ).fetchall()
    return render_fewshot(rows)


def build_system_prompt(cfg: dict, fewshot: str) -> str:
    """Per-user judge persona: magnets/repellents + free-text taste rules all come from
    cfg.profile (multi-user fix — they used to be hardcoded to user #1's taste; her rules now
    live verbatim in HER profile.yaml `taste_rules`)."""
    p = cfg.get("profile", {})
    scale = p.get("scale", {})
    scale_txt = "\n".join(f"  {k} = {v}" for k, v in sorted(scale.items(), reverse=True))
    magnets, repellents = _persona(cfg)
    vocab = magnets + repellents
    # Per-user free-text judging rules (profile.taste_rules: list[str]); empty = generic judge.
    taste = "".join(f"- {str(r).strip()}\n" for r in (p.get("taste_rules") or []) if str(r).strip())
    return (
        "Ты — персональный судья вакансий для пользователя. Оцени ОДНУ карточку по рубрике и верни "
        "ТОЛЬКО JSON-объект:\n"
        '{"score": 1..5, "why_tag": str|null, "why_freetext": str|null, "explanation": str}\n\n'
        f"ПРОФИЛЬ ПОЛЬЗОВАТЕЛЯ:\n{p.get('summary','').strip()}\n\n"
        f"ШКАЛА:\n{scale_txt}\n\n"
        f"МАГНИТЫ (тянут к 4–5): {', '.join(magnets)}\n"
        f"ОТТАЛКИВАТЕЛИ (тянут к 1–2): {', '.join(repellents)}\n\n"
        "ПРАВИЛА:\n"
        f"- why_tag — РОВНО один из словаря: {', '.join(vocab)} (или null, если ни один не главный).\n"
        "- Главный драйвер оценки → why_tag; нюансы → why_freetext (по-русски, кратко).\n"
        "- explanation — одно предложение по-русски, почему именно эта оценка.\n"
        "- Часть отталкивателей уже частично отфильтрована до судьи — но если видишь стоп-фактор "
        "из списка отталкивателей в карточке, понижай оценку и ставь соответствующий тег.\n"
        "- slop_score — плотность AI-слопа/буллшита (0 = конкретно, 70+ = вода без конкретики). "
        "Градуированно: slop_score ≥ 45 → мягкий минус; ≥ 60 → текст пустой/слоп, понижай сильнее и "
        "ставь тег slop-text (пример сигнала: «какой-то AI слоп текст»).\n"
        + taste +
        "- Не выдумывай факты сверх карточки; при нехватке сигнала ставь 3.\n\n"
        + (f"ПРИМЕРЫ (из пользовательских меток):\n{fewshot}\n\n" if fewshot else "")
        + (f'Пример валидного ответа: {{"score": 4, "why_tag": "{magnets[0]}", '
           f'"why_freetext": "сильный магнит-домен + подходящий формат", "explanation": '
           f'"Домен-магнит и подходящие условия — почти шабашка."}}')
    )


def score_card(client: OllamaClient, system: str, card: dict) -> int:
    """Один вызов судьи по карточке → score 1..5. Бросает LLMError/ValueError (unjudgeable)."""
    obj = client.chat_json(system, f"CARD:\n{_card_brief(card)}")
    return max(1, min(5, int(obj["score"])))


def judge_pending(cfg: dict, con) -> dict[str, int]:
    """NORMALIZED → оценка судьи. Полный grader-tuple в каждой строке. Возвращает счётчики."""
    llm_cfg = cfg.get("llm", {})
    judge_cfg = cfg.get("judge", {})
    from .llm_clients import make_llm_client
    client = make_llm_client(cfg, "judge")  # role-routed; defaults to ollama judge_model
    model = getattr(client, "model", llm_cfg.get("judge_model", "qwen3:8b"))
    digest = client.model_digest()
    rubric_version = judge_cfg.get("rubric_version", "v1")
    magnets, repellents = _persona(cfg)
    vocab = magnets + repellents
    fewshot, fewshot_hash = build_fewshot(con, int(judge_cfg.get("fewshot_max", 6)), vocab)
    system = build_system_prompt(cfg, fewshot)

    rows = db.by_status(con, Status.NORMALIZED)
    out = {"scored": 0, "errors": 0}
    for row in rows:
        try:
            card = json.loads(row["card_json"]) if row["card_json"] else {}
        except (TypeError, json.JSONDecodeError):
            card = {}
        user = f"CARD:\n{_card_brief(card)}"
        try:
            obj = client.chat_json(system, user)
            raw_score = int(obj["score"])
            if not 1 <= raw_score <= 5:
                # Out-of-range = schema violation, NOT a 5. Clamping floated hallucinated
                # high scores to the top of the slate; treat as unjudgeable → retry next night.
                raise ValueError(f"score out of range 1..5: {raw_score}")
            score = raw_score
        except (LLMError, KeyError, TypeError, ValueError) as e:
            from .models import ErrorClass
            ec = e.error_class if isinstance(e, LLMError) else ErrorClass.SCHEMA_VIOLATION
            db.set_error(con, row["id"], ec, str(e))
            out["errors"] += 1
            continue
        why_tag = obj.get("why_tag")
        why_freetext = obj.get("why_freetext")
        if why_tag is not None and why_tag not in vocab:
            # тег вне словаря → не врём схеме: переносим в freetext, тег обнуляем.
            why_freetext = (f"[{why_tag}] " + (why_freetext or "")).strip()
            why_tag = None
        db.insert_judge_score(con, row["id"], {
            "score": score, "why_tag": why_tag, "why_freetext": why_freetext,
            "explanation": obj.get("explanation"), "model": model, "model_digest": digest,
            "rubric_version": rubric_version, "fewshot_hash": fewshot_hash,
        })
        db.set_status(con, row["id"], Status.SCORED)
        out["scored"] += 1

    db.log_funnel(con, "judge", out["scored"],
                  detail=f"scored={out['scored']} errors={out['errors']} "
                         f"rubric={rubric_version} fewshot={fewshot_hash}")
    return out
