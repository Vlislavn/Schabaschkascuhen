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

MAGNETS = WHY_TAGS[:6]
REPELLENTS = WHY_TAGS[6:]


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


def build_fewshot(con, max_n: int) -> tuple[str, str]:
    """(fewshot_text, fewshot_hash) из СВЕЖАЙШИХ крайних меток, newest-first.

    Крайние = негативный якорь (score<=2: 👎) ∪ сильный позитив (score=5: ⭐). Раньше брали
    ровно {1,5}, но единственная веб-кнопка для негатива даёт 👎=2 (литеральная 1 в UI нет) —
    значит 👎-фидбек НИКОГДА не попадал в few-shot и судья не учился отталкивателям с реальных
    меток. <=2 чинит это; render_fewshot мапит score==5→магнит, иначе→отталкиватель.
    Метки с why_tag вне текущего словаря WHY_TAGS (наследие прошлой персоны/рубрики) исключаются,
    чтобы few-shot не учил судью чужими тегами; why_tag=NULL допускается."""
    placeholders = ",".join("?" for _ in WHY_TAGS)
    rows = con.execute(
        f"""SELECT l.score_1_5, l.why_tag, l.why_freetext, v.card_json, l.created_at
           FROM label l JOIN vacancy v ON v.id = l.vacancy_id
           WHERE (l.score_1_5 <= 2 OR l.score_1_5 = 5) AND v.card_json IS NOT NULL
             AND (l.why_tag IS NULL OR l.why_tag IN ({placeholders}))
           ORDER BY l.created_at DESC, l.id DESC LIMIT ?""",
        (*WHY_TAGS, max_n),
    ).fetchall()
    return render_fewshot(rows)


def build_system_prompt(cfg: dict, fewshot: str) -> str:
    p = cfg.get("profile", {})
    scale = p.get("scale", {})
    scale_txt = "\n".join(f"  {k} = {v}" for k, v in sorted(scale.items(), reverse=True))
    return (
        "Ты — персональный судья вакансий для Алины. Оцени ОДНУ карточку по её рубрике и верни "
        "ТОЛЬКО JSON-объект:\n"
        '{"score": 1..5, "why_tag": str|null, "why_freetext": str|null, "explanation": str}\n\n'
        f"ПРОФИЛЬ АЛИНЫ:\n{p.get('summary','').strip()}\n\n"
        f"ШКАЛА:\n{scale_txt}\n\n"
        f"МАГНИТЫ (тянут к 4–5): {', '.join(MAGNETS)}\n"
        f"ОТТАЛКИВАТЕЛИ (тянут к 1–2): {', '.join(REPELLENTS)}\n\n"
        "ПРАВИЛА:\n"
        f"- why_tag — РОВНО один из словаря: {', '.join(WHY_TAGS)} (или null, если ни один не главный).\n"
        "- Главный драйвер оценки → why_tag; нюансы → why_freetext (по-русски, кратко).\n"
        "- explanation — одно предложение по-русски, почему именно эта оценка.\n"
        "- Скрытый немецкий, remote-only, биотех, slop-текст уже частично отфильтрованы — но если "
        "видишь стоп-фактор в карточке, понижай оценку и ставь соответствующий тег.\n"
        "- slop_score — плотность AI-слопа/буллшита (0 = конкретно, 70+ = вода без конкретики). "
        "Градуированно: slop_score ≥ 45 → мягкий минус; ≥ 60 → текст пустой/слоп, понижай сильнее и "
        "ставь тег slop-text (Алина: «какой-то AI слоп текст»).\n"
        "- Алина хочет работать ГОЛОВОЙ, не руками: чисто hands-on dev/инженер-роль («сама пишу код "
        "целыми днями») — мягкий отталкиватель (снизь на 1, она может работать С разработчиками, но "
        "не быть одним из них). Роли lead/principal/ownership — наоборот, тяни вверх. Monthly/жёсткие "
        "дедлайны/рутинная операционка — мягкий минус (стресс, рутина).\n"
        "- Скучная зарегулированная операционка (GMP/GxP, чистый комплаенс/качество, рутинный "
        "контроллинг без изюминки) → тег boring-role, мягкий минус (Алина: «GMP/GxP кажется скучным»).\n"
        "- work_mode='onsite' без упоминания hybrid/remote-option → мягкий отталкиватель: "
        "снизь оценку на 1. Алина ищет гибридный формат, не чисто офисный.\n"
        "- Алина ищет новую область. Приоритет магнитов по убыванию: "
        "animals > space > military-security > public-sector > complex-projects > new-domain. "
        "Если подходит более конкретный магнит, не используй complex-projects.\n"
        "- complex-projects — ТОЛЬКО для реально крупных системных / инфраструктурных проектов "
        "(спутниковые системы, боевые комплексы, критическая госинфраструктура). "
        "Обычная ML, аналитика, веб-разработка — НЕ complex-projects.\n"
        "- Не выдумывай факты сверх карточки; при нехватке сигнала ставь 3.\n\n"
        + (f"ПРИМЕРЫ (из меток Алины):\n{fewshot}\n\n" if fewshot else "")
        + 'Пример валидного ответа: {"score": 4, "why_tag": "space", '
          '"why_freetext": "аэрокосмический домен + гибрид под Франкфуртом", "explanation": '
          '"Космос-магнит и подходящая локация — почти шабашка."}'
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
    fewshot, fewshot_hash = build_fewshot(con, int(judge_cfg.get("fewshot_max", 6)))
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
        if why_tag is not None and why_tag not in WHY_TAGS:
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
