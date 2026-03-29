"""
Bias and fairness detector for ranked lists.

Computes:
- Exposure bias: are protected groups systematically ranked lower?
- Position bias: is there a systematic skew in which rank positions
  documents land in, controlling for relevance?
- Demographic parity: do different groups receive proportional exposure
  at top-k positions?
"""

from __future__ import annotations

import math
from collections import defaultdict

from ..core.report import BiasResult
from ..core.types import QueryDocPair, QueryResult


_LOG2_DISCOUNT = [1.0 / math.log2(i + 2) for i in range(1000)]


def _exposure(rank: int) -> float:
    """Logarithmic exposure weight at a given rank (1-indexed)."""
    if rank <= 0:
        return 0.0
    idx = rank - 1
    if idx < len(_LOG2_DISCOUNT):
        return _LOG2_DISCOUNT[idx]
    return 1.0 / math.log2(rank + 1)


class BiasDetector:
    """
    Analyse a QueryResult for bias indicators.

    Reads group membership from `metadata["group"]` on each QueryDocPair
    (if present).  Falls back to position-bias-only analysis when group
    labels are absent.

    Parameters
    ----------
    top_k:
        Cut-off for demographic parity checks.
    """

    def __init__(self, top_k: int = 10) -> None:
        self.top_k = top_k

    def analyse(
        self, query_result: QueryResult, pairs: list[QueryDocPair]
    ) -> BiasResult:
        pair_map = {p.doc_id: p for p in pairs}
        result = BiasResult(query_id=query_result.query_id)

        # --- position bias ---
        result.position_bias = self._position_bias(query_result, pair_map)

        # --- group-level metrics (only if group labels exist) ---
        groups = {
            p.doc_id: p.metadata.get("group")
            for p in pairs
            if p.metadata.get("group") is not None
        }
        if groups:
            result.exposure_bias = self._exposure_bias(query_result, groups)
            result.demographic_parity = self._demographic_parity(
                query_result, groups, self.top_k
            )
            self._add_notes(result)

        return result

    # ------------------------------------------------------------------
    # Metric implementations
    # ------------------------------------------------------------------

    def _position_bias(
        self, qr: QueryResult, pair_map: dict[str, QueryDocPair]
    ) -> float:
        """
        Compute the Spearman rank correlation between relevance labels and
        actual rank positions.  A value close to -1 indicates strong
        position bias (lower relevance docs ranked higher).
        Returns 0 if no relevance labels are available.
        """
        labeled = [
            (r.rank, pair_map[r.doc_id].relevance)
            for r in qr.results
            if r.doc_id in pair_map and pair_map[r.doc_id].relevance is not None
        ]
        if len(labeled) < 2:
            return 0.0

        n = len(labeled)
        ranks = [x[0] for x in labeled]
        rels = [x[1] for x in labeled]
        mean_rank = sum(ranks) / n
        mean_rel = sum(rels) / n

        cov = sum((r - mean_rank) * (v - mean_rel) for r, v in zip(ranks, rels)) / n
        std_rank = math.sqrt(sum((r - mean_rank) ** 2 for r in ranks) / n)
        std_rel = math.sqrt(sum((v - mean_rel) ** 2 for v in rels) / n)

        if std_rank == 0 or std_rel == 0:
            return 0.0
        return cov / (std_rank * std_rel)

    def _exposure_bias(
        self, qr: QueryResult, groups: dict[str, str]
    ) -> dict[str, float]:
        """
        Total logarithmic exposure per group, normalised by group size.
        A ratio far from 1.0 between groups signals exposure bias.
        """
        group_exposure: dict[str, float] = defaultdict(float)
        group_count: dict[str, int] = defaultdict(int)

        for result in qr.results:
            g = groups.get(result.doc_id)
            if g is not None:
                group_exposure[g] += _exposure(result.rank)
                group_count[g] += 1

        return {
            g: group_exposure[g] / group_count[g]
            for g in group_exposure
            if group_count[g] > 0
        }

    def _demographic_parity(
        self, qr: QueryResult, groups: dict[str, str], k: int
    ) -> dict[str, float]:
        """
        Fraction of top-k results belonging to each group.
        """
        top_k_results = [r for r in qr.results if r.rank <= k]
        group_count: dict[str, int] = defaultdict(int)
        for r in top_k_results:
            g = groups.get(r.doc_id)
            if g is not None:
                group_count[g] += 1

        total = len(top_k_results) or 1
        return {g: count / total for g, count in group_count.items()}

    def _add_notes(self, result: BiasResult) -> None:
        """Attach human-readable notes when bias thresholds are exceeded."""
        if result.exposure_bias:
            exposures = list(result.exposure_bias.values())
            if len(exposures) >= 2:
                ratio = max(exposures) / (min(exposures) + 1e-9)
                if ratio > 2.0:
                    result.notes.append(
                        f"Exposure ratio between groups is {ratio:.2f}x — "
                        "significant disparity detected."
                    )

        if result.demographic_parity:
            fracs = list(result.demographic_parity.values())
            if len(fracs) >= 2:
                disparity = max(fracs) - min(fracs)
                if disparity > 0.4:
                    result.notes.append(
                        f"Demographic parity gap is {disparity:.2f} — "
                        "one group dominates top-k results."
                    )