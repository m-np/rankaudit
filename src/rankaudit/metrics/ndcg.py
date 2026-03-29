"""Standard IR metrics: NDCG, Precision, MAP, MRR."""

from __future__ import annotations

import math


def _dcg(ranked_ids: list[str], rel_map: dict[str, float], k: int) -> float:
    total = 0.0
    for i, doc_id in enumerate(ranked_ids[:k], start=1):
        rel = rel_map.get(doc_id, 0.0)
        total += (2**rel - 1) / math.log2(i + 1)
    return total


def compute_ndcg(ranked_ids: list[str], rel_map: dict[str, float], k: int) -> float:
    """Normalised Discounted Cumulative Gain at cut-off k."""
    dcg = _dcg(ranked_ids, rel_map, k)
    ideal_ids = sorted(rel_map.keys(), key=lambda d: rel_map[d], reverse=True)
    idcg = _dcg(ideal_ids, rel_map, k)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def compute_precision(ranked_ids: list[str], rel_map: dict[str, float], k: int) -> float:
    """Precision at cut-off k (relevance > 0 treated as relevant)."""
    relevant = sum(1 for doc_id in ranked_ids[:k] if rel_map.get(doc_id, 0.0) > 0)
    return relevant / k if k > 0 else 0.0


def compute_map(ranked_ids: list[str], rel_map: dict[str, float]) -> float:
    """Mean Average Precision."""
    n_relevant = sum(1 for v in rel_map.values() if v > 0)
    if n_relevant == 0:
        return 0.0
    ap = 0.0
    hits = 0
    for i, doc_id in enumerate(ranked_ids, start=1):
        if rel_map.get(doc_id, 0.0) > 0:
            hits += 1
            ap += hits / i
    return ap / n_relevant


def compute_mrr(ranked_ids: list[str], rel_map: dict[str, float]) -> float:
    """Mean Reciprocal Rank."""
    for i, doc_id in enumerate(ranked_ids, start=1):
        if rel_map.get(doc_id, 0.0) > 0:
            return 1.0 / i
    return 0.0