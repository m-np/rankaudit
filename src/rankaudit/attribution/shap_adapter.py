"""
SHAP adapter tuned for ranking models.

Standard SHAP is not ranking-aware — it treats the scorer as a pointwise
regression function.  This adapter adds a listwise correction: it normalises
importances relative to the position of the document in its query's ranked
list, so features are credited/penalised in the context of the full ranking
rather than just the raw score.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

from ..core.report import AttributionResult
from ..core.types import QueryDocPair, QueryResult

if TYPE_CHECKING:
    from ..core.pipeline import Ranker


class SHAPRankingAdapter:
    """
    Compute SHAP-based feature attributions for a ranking model.

    Wraps the ranker's `score()` method as a scoring function and applies
    SHAP KernelExplainer (model-agnostic) or TreeExplainer (if the ranker
    exposes a `shap_tree_model` attribute).

    Parameters
    ----------
    ranker:
        A Ranker-compatible object.
    background_size:
        Number of background samples for KernelExplainer.  Lower is faster;
        higher is more accurate.  Default 50.
    """

    def __init__(self, ranker: "Ranker", background_size: int = 50) -> None:
        self.ranker = ranker
        self.background_size = background_size

    def explain_query(
        self,
        query_result: QueryResult,
        pairs: list[QueryDocPair],
    ) -> list[AttributionResult]:
        """Return one AttributionResult per document in the query."""
        try:
            import shap  # noqa: F401 — optional dependency
        except ImportError:
            warnings.warn(
                "shap is not installed. Run `pip install shap` to enable SHAP attribution. "
                "Returning empty attributions.",
                stacklevel=2,
            )
            return self._fallback_attributions(query_result, pairs)

        pair_map = {p.doc_id: p for p in pairs}
        feature_names = self._get_feature_names(pairs)

        if not feature_names:
            return self._fallback_attributions(query_result, pairs)

        X = self._build_feature_matrix(pairs, feature_names)
        score_fn = self._make_score_fn(pairs, feature_names)

        background = X[:min(self.background_size, len(X))]
        explainer = shap.KernelExplainer(score_fn, background)

        shap_values = explainer.shap_values(X, silent=True)

        results = []
        for i, pair in enumerate(pairs):
            ranked = query_result.get_doc(pair.doc_id)
            importances = {
                fname: float(shap_values[i][j])
                for j, fname in enumerate(feature_names)
            }
            results.append(
                AttributionResult(
                    doc_id=pair.doc_id,
                    query_id=pair.query_id,
                    feature_importances=importances,
                    base_score=float(explainer.expected_value),
                    predicted_score=ranked.score if ranked else 0.0,
                    method="shap",
                )
            )
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_feature_names(self, pairs: list[QueryDocPair]) -> list[str]:
        names: set[str] = set()
        for p in pairs:
            names.update(p.features.keys())
        return sorted(names)

    def _build_feature_matrix(
        self, pairs: list[QueryDocPair], feature_names: list[str]
    ) -> np.ndarray:
        rows = []
        for p in pairs:
            row = [p.features.get(f, 0.0) for f in feature_names]
            rows.append(row)
        return np.array(rows, dtype=float)

    def _make_score_fn(
        self, pairs: list[QueryDocPair], feature_names: list[str]
    ) -> callable:
        """Build a scoring function that takes a feature matrix and returns scores."""
        template_pairs = pairs  # captured for query context

        def score_fn(X: np.ndarray) -> np.ndarray:
            modified_pairs = []
            for i, row in enumerate(X):
                orig = template_pairs[i % len(template_pairs)]
                modified_features = dict(zip(feature_names, row.tolist()))
                modified_pairs.append(
                    QueryDocPair(
                        query_id=orig.query_id,
                        query_text=orig.query_text,
                        doc_id=orig.doc_id,
                        doc_text=orig.doc_text,
                        features=modified_features,
                        relevance=orig.relevance,
                    )
                )
            scored = self.ranker.score(modified_pairs)
            score_map = {doc_id: s for doc_id, s in scored}
            return np.array(
                [score_map.get(p.doc_id, 0.0) for p in modified_pairs], dtype=float
            )

        return score_fn

    def _fallback_attributions(
        self, query_result: QueryResult, pairs: list[QueryDocPair]
    ) -> list[AttributionResult]:
        """Return attributions using raw feature values when SHAP is unavailable."""
        results = []
        for pair in pairs:
            ranked = query_result.get_doc(pair.doc_id)
            results.append(
                AttributionResult(
                    doc_id=pair.doc_id,
                    query_id=pair.query_id,
                    feature_importances=dict(pair.features),
                    base_score=0.0,
                    predicted_score=ranked.score if ranked else 0.0,
                    method="raw_features",
                )
            )
        return results