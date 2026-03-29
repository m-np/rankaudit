"""
Counterfactual ranking engine.

Given a query result, finds the minimal feature perturbation needed to flip
the ranking order of two adjacent documents (or a specified pair).

This is original IP: standard counterfactual explanation tools (DiCE, CARLA)
are not ranking-aware and do not model the score gap between two documents
in the context of a list.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..core.report import CounterfactualResult
from ..core.types import QueryDocPair, QueryResult

if TYPE_CHECKING:
    from ..core.pipeline import Ranker


class CounterfactualEngine:
    """
    Generate counterfactual flip explanations for ranked document pairs.

    Strategy
    --------
    For each pair (doc_a ranked above doc_b), we perform coordinate-wise
    gradient ascent on doc_b's features — perturbing one feature at a time
    by small increments and checking whether doc_b's score exceeds doc_a's.
    We record the first feature set that achieves the flip.

    Parameters
    ----------
    ranker:
        A Ranker-compatible object.
    max_pairs:
        How many adjacent pairs to generate counterfactuals for per query.
    step_size:
        Fractional increment per feature perturbation step.
    max_steps:
        Maximum perturbation steps before giving up on a pair.
    """

    def __init__(
        self,
        ranker: "Ranker",
        max_pairs: int = 3,
        step_size: float = 0.05,
        max_steps: int = 100,
    ) -> None:
        self.ranker = ranker
        self.max_pairs = max_pairs
        self.step_size = step_size
        self.max_steps = max_steps

    def generate(
        self,
        query_result: QueryResult,
        pairs: list[QueryDocPair],
    ) -> list[CounterfactualResult]:
        results = []
        pair_map = {p.doc_id: p for p in pairs}
        ranked = query_result.results

        # consider only adjacent pairs in the top-k
        candidates = list(zip(ranked[:-1], ranked[1:]))[: self.max_pairs]

        for doc_a, doc_b in candidates:
            cf = self._flip(
                query_result.query_id,
                pair_map.get(doc_a.doc_id),
                pair_map.get(doc_b.doc_id),
                doc_a,
                doc_b,
                pairs,
            )
            if cf is not None:
                results.append(cf)

        return results

    def _flip(
        self,
        query_id: str,
        pair_a: QueryDocPair | None,
        pair_b: QueryDocPair | None,
        ranked_a,
        ranked_b,
        all_pairs: list[QueryDocPair],
    ) -> CounterfactualResult | None:
        if pair_a is None or pair_b is None:
            return None
        if not pair_b.features:
            return None  # no features to perturb

        feature_names = sorted(pair_b.features.keys())
        current_features = {f: pair_b.features.get(f, 0.0) for f in feature_names}
        delta_score = ranked_a.score - ranked_b.score
        flipping_changes: dict[str, tuple[float, float]] = {}

        modified_b = QueryDocPair(
            query_id=pair_b.query_id,
            query_text=pair_b.query_text,
            doc_id=pair_b.doc_id,
            doc_text=pair_b.doc_text,
            features=dict(current_features),
            relevance=pair_b.relevance,
        )

        for _ in range(self.max_steps):
            # try perturbing each feature
            for feat in feature_names:
                trial_features = dict(modified_b.features)
                increment = max(abs(trial_features.get(feat, 0.0)) * self.step_size, self.step_size)
                trial_features[feat] = trial_features.get(feat, 0.0) + increment

                trial_b = QueryDocPair(
                    query_id=pair_b.query_id,
                    query_text=pair_b.query_text,
                    doc_id=pair_b.doc_id,
                    doc_text=pair_b.doc_text,
                    features=trial_features,
                    relevance=pair_b.relevance,
                )

                scored = dict(self.ranker.score([pair_a, trial_b]))
                new_score_b = scored.get(pair_b.doc_id, 0.0)
                new_score_a = scored.get(pair_a.doc_id, ranked_a.score)

                if new_score_b > new_score_a:
                    # found the flip — record the change
                    flipping_changes[feat] = (
                        pair_b.features.get(feat, 0.0),
                        trial_features[feat],
                    )
                    return CounterfactualResult(
                        query_id=query_id,
                        doc_a_id=pair_a.doc_id,
                        doc_b_id=pair_b.doc_id,
                        original_rank_a=ranked_a.rank,
                        original_rank_b=ranked_b.rank,
                        flipping_changes=flipping_changes,
                        delta_score=delta_score,
                    )

                # keep the best-improving perturbation
                if new_score_b > scored.get(pair_b.doc_id, 0.0):
                    modified_b.features[feat] = trial_features[feat]
                    flipping_changes[feat] = (
                        pair_b.features.get(feat, 0.0),
                        trial_features[feat],
                    )

        # could not flip within budget — return partial result
        if flipping_changes:
            return CounterfactualResult(
                query_id=query_id,
                doc_a_id=pair_a.doc_id,
                doc_b_id=pair_b.doc_id,
                original_rank_a=ranked_a.rank,
                original_rank_b=ranked_b.rank,
                flipping_changes=flipping_changes,
                delta_score=delta_score,
            )
        return None