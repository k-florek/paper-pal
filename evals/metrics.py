from __future__ import annotations

import math
from typing import Iterable


def accuracy(predictions: Iterable[str], labels: Iterable[str]) -> float:
    preds = list(predictions)
    gold = list(labels)
    if not preds or len(preds) != len(gold):
        return 0.0
    correct = sum(int(p == g) for p, g in zip(preds, gold))
    return correct / len(gold)


def precision_at_k(relevance_flags: list[int], k: int = 5) -> float:
    if k <= 0:
        return 0.0
    topk = relevance_flags[:k]
    if not topk:
        return 0.0
    return sum(topk) / len(topk)


def ndcg_at_k(gains: list[float], k: int = 10) -> float:
    if k <= 0:
        return 0.0
    actual = gains[:k]
    if not actual:
        return 0.0

    def dcg(values: list[float]) -> float:
        return sum(v / math.log2(i + 2) for i, v in enumerate(values))

    ideal = sorted(actual, reverse=True)
    best = dcg(ideal)
    if best == 0:
        return 0.0
    return dcg(actual) / best
