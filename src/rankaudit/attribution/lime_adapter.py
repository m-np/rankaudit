"""
LIME adapter for ranking models.

Uses LIME's tabular explainer on the feature vector of each document,
treating the ranker's score as the prediction function.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

from ..core.report import AttributionResult
from ..core.types import QueryDocPair, QueryResult

if TYPE_CHECKING:
    from ..core.pipeline import Ranker


class LIMERankingAdapter:
    """
    LIME-based feature attribution for a ranking model.

    Parameters
    ----------
    ranker:
        A Ranker-compatible object.
    num_samples:
        Number of perturbation samples per document.  Default 300.
    """

    def __init__(self, ranker: "Ranker", num_samples: int = 300) -> None:
        self.ranker = ranker
        self.num_samples = num_samples

    def explain_query(
        self,
        query_result: QueryResult,
        pairs: list[QueryDocPair],
    ) -> list[AttributionResult]:
        try:
            from lime.lime_tabular import LimeTabularExplainer
        except ImportError:
            warnings.warn(
                "lime is not installed. Run `pip install lime` to enable LIME attribution. "
                "Returning empty attributions.",
                stacklevel=2,
            )
            return []

        feature_names = self._get_feature_names(pairs)
        if not feature_names:
            return []

        X = self._build_feature_matrix(pairs, feature_names)
        explainer = LimeTabularExplainer(
            X,
            feature_names=feature_names,
            mode="regression",
            verbose=False,
        )

        score_fn = self._make_score_fn(pairs, feature_names)
        results = []
        for i, pair in enumerate(pairs):
            ranked = query_result.get_doc(pair.doc_id)
            exp = explainer.explain_instance(
                X[i], score_fn, num_features=len(feature_names),
                num_samples=self.num_samples,
            )
            importances = dict(exp.as_list())
            results.append(
                AttributionResult(
                    doc_id=pair.doc_id,
                    query_id=pair.query_id,
                    feature_importances=importances,
                    base_score=float(exp.intercept[1]) if hasattr(exp, "intercept") else 0.0,
                    predicted_score=ranked.score if ranked else 0.0,
                    method="lime",
                )
            )
        return results

    def _get_feature_names(self, pairs: list[QueryDocPair]) -> list[str]:
        names: set[str] = set()
        for p in pairs:
            names.update(p.features.keys())
        return sorted(names)

    def _build_feature_matrix(
        self, pairs: list[QueryDocPair], feature_names: list[str]
    ) -> np.ndarray:
        return np.array(
            [[p.features.get(f, 0.0) for f in feature_names] for p in pairs],
            dtype=float,
        )

    def _make_score_fn(
        self, pairs: list[QueryDocPair], feature_names: list[str]
    ) -> callable:
        template_pairs = pairs

        def score_fn(X: np.ndarray) -> np.ndarray:
            modified_pairs = []
            for i, row in enumerate(X):
                orig = template_pairs[i % len(template_pairs)]
                modified_pairs.append(
                    QueryDocPair(
                        query_id=orig.query_id,
                        query_text=orig.query_text,
                        doc_id=orig.doc_id,
                        doc_text=orig.doc_text,
                        features=dict(zip(feature_names, row.tolist())),
                        relevance=orig.relevance,
                    )
                )
            scored = self.ranker.score(modified_pairs)
            score_map = {doc_id: s for doc_id, s in scored}
            return np.array(
                [score_map.get(p.doc_id, 0.0) for p in modified_pairs], dtype=float
            )

        return score_fn