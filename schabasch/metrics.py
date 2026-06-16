"""Ranking-quality metrics for the matcher — shared by the CLI eval harness (eval/match_eval.py)
and the live web validation page (schabasch/validation.py → /eval).

A scoring method is GOOD iff it ranks high-gold jobs above low-gold ones. `gold` is always an
explicit {vacancy_id: relevance} dict (synthetic GOLD for the CLI dev-floor; the user's real
score_1_5 labels for the UI). scipy is optional — `spearman` degrades to 0.0 if it's absent.
"""
from __future__ import annotations

import math


def pairwise_accuracy(scores: dict[int, float], gold: dict[int, int]) -> tuple[float, int]:
    """% of comparable (higher-gold, lower-gold) pairs the method orders correctly. Ties in gold
    are skipped. Returns (accuracy, n_pairs)."""
    ids = [i for i in gold if i in scores]
    correct = total = 0
    for a in ids:
        for b in ids:
            if gold[a] <= gold[b]:
                continue
            total += 1
            if scores[a] > scores[b]:
                correct += 1
            elif scores[a] == scores[b]:
                correct += 0.5
    return (correct / total if total else 0.0), total


def ndcg_at_k(scores: dict[int, float], gold: dict[int, int], k: int = 10) -> float:
    ranked = sorted([i for i in gold if i in scores], key=lambda i: -scores[i])[:k]
    dcg = sum((2 ** gold[i] - 1) / math.log2(rank + 2) for rank, i in enumerate(ranked))
    ideal = sorted(gold.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(rank + 2) for rank, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def spearman(scores: dict[int, float], gold: dict[int, int]) -> float:
    ids = [i for i in gold if i in scores]
    try:
        from scipy.stats import spearmanr
        rho, _ = spearmanr([scores[i] for i in ids], [gold[i] for i in ids])
        return float(rho) if rho == rho else 0.0
    except Exception:
        return 0.0


def evaluate(scores: dict[int, float], gold: dict[int, int], *, name: str = "") -> dict:
    """{name, pairwise_acc, ndcg@10, spearman, n, n_pairs} for one signal against an explicit gold."""
    pa, npairs = pairwise_accuracy(scores, gold)
    return {"name": name, "pairwise_acc": round(pa, 3), "ndcg@10": round(ndcg_at_k(scores, gold), 3),
            "spearman": round(spearman(scores, gold), 3),
            "n": len([i for i in gold if i in scores]), "n_pairs": npairs}


def top_bottom(scores: dict[int, float], gold: dict[int, int], *,
               rationales: dict[int, str] | None = None, k: int = 8) -> str:
    """Top-k by score with their gold label (+ optional rationale) — a human spot-check."""
    rationales = rationales or {}
    ranked = sorted([i for i in scores if i in gold], key=lambda i: -scores[i])
    out = ["  TOP:"]
    for i in ranked[:k]:
        out.append(f"    {scores[i]:.3f} gold={gold[i]}  {rationales.get(i, '')[:60]}")
    return "\n".join(out)
