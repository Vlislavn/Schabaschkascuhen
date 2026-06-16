"""Калибровка судьи: 5-fold CV agreement против меток Влада (gate запуска daily).

Честный расчёт по are/multi-run-variance:
- fold-restricted few-shot (тест-фолд не видит своих примеров — нет утечки),
- бинарный интент: score≥4 ↔ interview=да; agreement = доля совпадений,
- mean ± SEM по FOLD-level средним (std ddof=1, guard n≤1 → 0),
- N=3 прогона судьи на карточку → verdict-stability (доля карточек без флипа интента),
- unjudgeable (LLM invalid после ретраев) — ВНЕ знаменателя agreement,
- majority-baseline = max(p, 1−p) истинного интента; gate = agreement ≥75% И ≥ baseline+10pp,
- Spearman репортится диагностически (не порог).

Plan B (если gate не берётся за 3 итерации рубрики/few-shot) — setwise/best-worst, решение Влада.
"""
from __future__ import annotations

import json
import math
import statistics

from .judge import build_system_prompt, render_fewshot, score_card
from .llm import LLMError, OllamaClient


def _load_labeled(con) -> list[dict]:
    """Метки (bootstrap+slate) с интентом и карточкой. interview даёт истинный бинарный интент."""
    rows = con.execute(
        """SELECT l.vacancy_id, l.score_1_5, l.why_tag, l.why_freetext, l.interview, v.card_json
           FROM label l JOIN vacancy v ON v.id = l.vacancy_id
           WHERE l.score_1_5 IS NOT NULL AND v.card_json IS NOT NULL
           ORDER BY l.vacancy_id"""
    ).fetchall()
    out = []
    for r in rows:
        try:
            card = json.loads(r["card_json"])
        except (TypeError, json.JSONDecodeError):
            continue
        score = int(r["score_1_5"])
        # истинный интент: явный interview если задан, иначе score≥4
        if r["interview"] in (0, 1):
            intent = int(r["interview"])
        else:
            intent = 1 if score >= 4 else 0
        out.append({"vacancy_id": int(r["vacancy_id"]), "score_1_5": score,
                    "why_tag": r["why_tag"], "why_freetext": r["why_freetext"],
                    "card": card, "intent": intent})
    return out


def _folds(items: list[dict], k: int, seed: int = 42) -> list[list[int]]:
    import random
    idx = list(range(len(items)))
    random.Random(seed).shuffle(idx)
    return [idx[i::k] for i in range(k)]  # stratified-ish round-robin assignment


def _sem(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return statistics.stdev(values) / math.sqrt(len(values))


def cross_validate(cfg: dict, con, *, folds: int = 5, runs: int = 3,
                   client: OllamaClient | None = None) -> dict:
    """5-fold CV agreement судья↔метки с N прогонов. Возвращает diagnostics + вердикт гейта."""
    items = _load_labeled(con)
    n = len(items)
    if n < folds:
        return {"error": f"too few labels ({n}) for {folds}-fold CV", "n_labels": n}

    llm_cfg = cfg.get("llm", {})
    if client is None:
        client = OllamaClient(model=llm_cfg.get("judge_model", "qwen3:8b"),
                              num_ctx=int(llm_cfg.get("num_ctx", 8192)),
                              temperature=float(llm_cfg.get("temperature", 0.1)))
    fewshot_max = int(cfg.get("judge", {}).get("fewshot_max", 6))

    fold_assign = _folds(items, folds)
    fold_agreements: list[float] = []
    # для verdict-stability: на карточку собираем предсказанные интенты по N прогонам
    pred_intents: dict[int, list[int]] = {}
    unjudgeable = 0
    per_card_correct: list[tuple[int, int]] = []  # (true_intent, majority_pred) для spearman/диагностики
    judged_scores: list[tuple[int, int]] = []     # (true_score, pred_score) для Spearman

    for fi in range(folds):
        test_idx = set(fold_assign[fi])
        train = [items[i] for i in range(n) if i not in test_idx]
        # fold-restricted few-shot: крайние примеры (1 и 5) только из train
        extreme = [t for t in train if t["score_1_5"] in (1, 5)]
        extreme.sort(key=lambda t: abs(t["score_1_5"] - 3), reverse=True)
        fs_rows = [{"score_1_5": t["score_1_5"], "why_tag": t["why_tag"],
                    "why_freetext": t["why_freetext"],
                    "card_json": json.dumps(t["card"], ensure_ascii=False)}
                   for t in extreme[:fewshot_max]]
        fewshot, _ = render_fewshot(fs_rows)
        system = build_system_prompt(cfg, fewshot)

        correct = 0
        denom = 0
        for i in fold_assign[fi]:
            it = items[i]
            run_intents = []
            for _ in range(runs):
                try:
                    s = score_card(client, system, it["card"])
                except (LLMError, KeyError, TypeError, ValueError):
                    continue
                run_intents.append(1 if s >= 4 else 0)
                judged_scores.append((it["score_1_5"], s))
            if not run_intents:
                unjudgeable += 1
                continue
            pred_intents[it["vacancy_id"]] = run_intents
            # majority-vote интент по N прогонам
            maj = 1 if sum(run_intents) * 2 >= len(run_intents) else 0
            denom += 1
            if maj == it["intent"]:
                correct += 1
            per_card_correct.append((it["intent"], maj))
        fold_agreements.append(correct / denom if denom else 0.0)

    mean_agreement = statistics.mean(fold_agreements) if fold_agreements else 0.0
    sem = _sem(fold_agreements)

    # verdict-stability: доля карточек, у которых интент не флипнул между прогонами
    stable = sum(1 for v in pred_intents.values() if len(set(v)) == 1)
    stability = stable / len(pred_intents) if pred_intents else 0.0

    # majority-baseline истинного интента
    intents = [it["intent"] for it in items]
    p = sum(intents) / len(intents)
    majority_baseline = max(p, 1 - p)

    # Spearman (диагностика) истинный score vs предсказанный
    spearman = _spearman([a for a, _ in judged_scores], [b for _, b in judged_scores])

    gate_75 = mean_agreement >= 0.75
    gate_baseline = mean_agreement >= majority_baseline + 0.10
    return {
        "n_labels": n, "folds": folds, "runs": runs,
        "fold_agreements": [round(a, 3) for a in fold_agreements],
        "mean_agreement": round(mean_agreement, 3),
        "sem": round(sem, 3),
        "verdict_stability": round(stability, 3),
        "unjudgeable": unjudgeable,
        "majority_baseline": round(majority_baseline, 3),
        "spearman_diag": None if spearman is None else round(spearman, 3),
        "gate_pass": bool(gate_75 and gate_baseline),
        "gate_detail": {"agreement>=0.75": gate_75,
                        "agreement>=baseline+0.10": gate_baseline},
        "model": getattr(client, "model", None),
    }


def _spearman(a: list[int], b: list[int]) -> float | None:
    if len(a) < 2:
        return None

    def rank(xs):
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        r = [0.0] * len(xs)
        i = 0
        while i < len(xs):
            j = i
            while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    ra, rb = rank(a), rank(b)
    ma, mb = statistics.mean(ra), statistics.mean(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = math.sqrt(sum((x - ma) ** 2 for x in ra))
    vb = math.sqrt(sum((y - mb) ** 2 for y in rb))
    if va == 0 or vb == 0:
        return None
    return cov / (va * vb)
