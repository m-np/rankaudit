"""Core audit pipeline — orchestrates scoring, metrics, attribution, and bias."""

from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable

from .report import (
    AuditReport,
    AttributionResult,
    BiasResult,
    MetricResult,
)
from .types import QueryDocPair, QueryResult, RankedResult


@runtime_checkable
class Ranker(Protocol):
    """
    Minimal interface a ranker must satisfy.

    The ranker receives a list of QueryDocPair objects for a single query
    and returns a list of (doc_id, score) tuples in descending score order.
    """

    def score(self, pairs: list[QueryDocPair]) -> list[tuple[str, float]]:
        ...


class AuditPipeline:
    """
    Orchestrates a full ranking audit.

    Parameters
    ----------
    ranker:
        Any object implementing the Ranker protocol.
    metrics:
        List of metric names to compute. Supported: "ndcg", "precision",
        "map", "mrr", "fairness", "stability".
    attribution_method:
        "shap" | "lime" | None.  When None, attribution is skipped.
    run_counterfactuals:
        Whether to run the counterfactual flip engine.
    run_bias:
        Whether to run the bias detector.
    top_k:
        Cutoff for metric computation.
    """

    def __init__(
        self,
        ranker: Ranker,
        metrics: list[str] | None = None,
        attribution_method: str | None = "shap",
        run_counterfactuals: bool = True,
        run_bias: bool = True,
        top_k: int = 10,
    ) -> None:
        self.ranker = ranker
        self.metrics = metrics or ["ndcg", "fairness"]
        self.attribution_method = attribution_method
        self.run_counterfactuals = run_counterfactuals
        self.run_bias = run_bias
        self.top_k = top_k

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, pairs: list[QueryDocPair]) -> AuditReport:
        """
        Run the full audit pipeline on a list of QueryDocPair objects.

        Returns an AuditReport.
        """
        report = AuditReport()
        queries = self._group_by_query(pairs)

        for query_id, qpairs in queries.items():
            # 1. Score & rank
            query_result = self._score_and_rank(query_id, qpairs)
            report.query_results.append(query_result)

            # 2. IR metrics
            if self.metrics:
                metric_result = self._compute_metrics(query_result, qpairs)
                report.metrics.append(metric_result)

            # 3. Feature attribution
            if self.attribution_method:
                attributions = self._compute_attributions(query_result, qpairs)
                report.attributions.extend(attributions)

            # 4. Counterfactuals
            if self.run_counterfactuals:
                cfs = self._compute_counterfactuals(query_result, qpairs)
                report.counterfactuals.extend(cfs)

            # 5. Bias
            if self.run_bias or "fairness" in self.metrics:
                bias = self._compute_bias(query_result, qpairs)
                report.bias.append(bias)

        return report

    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------

    def _group_by_query(
        self, pairs: list[QueryDocPair]
    ) -> dict[str, list[QueryDocPair]]:
        groups: dict[str, list[QueryDocPair]] = {}
        for p in pairs:
            groups.setdefault(p.query_id, []).append(p)
        return groups

    def _score_and_rank(
        self, query_id: str, pairs: list[QueryDocPair]
    ) -> QueryResult:
        scored = self.ranker.score(pairs)
        # sort descending by score
        scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)

        pair_map = {p.doc_id: p for p in pairs}
        results = []
        for rank_idx, (doc_id, score) in enumerate(scored_sorted, start=1):
            pair = pair_map.get(doc_id)
            results.append(
                RankedResult(
                    doc_id=doc_id,
                    doc_text=pair.doc_text if pair else "",
                    rank=rank_idx,
                    score=score,
                    features=pair.features if pair else {},
                    metadata=pair.metadata if pair else {},
                )
            )

        query_text = pairs[0].query_text if pairs else ""
        return QueryResult(query_id=query_id, query_text=query_text, results=results)

    def _compute_metrics(
        self, qr: QueryResult, pairs: list[QueryDocPair]
    ) -> MetricResult:
        from ..metrics.ndcg import compute_ndcg, compute_precision, compute_map, compute_mrr

        rel_map = {p.doc_id: (p.relevance or 0.0) for p in pairs}
        ranked_ids = [r.doc_id for r in qr.results]
        ks = [k for k in [1, 3, 5, 10, 20] if k <= len(ranked_ids)]

        return MetricResult(
            query_id=qr.query_id,
            ndcg_at_k={k: compute_ndcg(ranked_ids, rel_map, k) for k in ks},
            precision_at_k={k: compute_precision(ranked_ids, rel_map, k) for k in ks},
            map_score=compute_map(ranked_ids, rel_map),
            mrr_score=compute_mrr(ranked_ids, rel_map),
        )

    def _compute_attributions(
        self, qr: QueryResult, pairs: list[QueryDocPair]
    ) -> list[AttributionResult]:
        if self.attribution_method == "shap":
            from ..attribution.shap_adapter import SHAPRankingAdapter
            adapter = SHAPRankingAdapter(self.ranker)
        elif self.attribution_method == "lime":
            from ..attribution.lime_adapter import LIMERankingAdapter
            adapter = LIMERankingAdapter(self.ranker)
        else:
            return []

        return adapter.explain_query(qr, pairs)

    def _compute_counterfactuals(
        self, qr: QueryResult, pairs: list[QueryDocPair]
    ) -> list:
        from ..counterfactual.engine import CounterfactualEngine
        engine = CounterfactualEngine(self.ranker)
        return engine.generate(qr, pairs)

    def _compute_bias(self, qr: QueryResult, pairs: list[QueryDocPair]) -> BiasResult:
        from ..bias.detector import BiasDetector
        detector = BiasDetector()
        return detector.analyse(qr, pairs)