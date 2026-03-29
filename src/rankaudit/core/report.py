"""AuditReport — structured output of a ranking audit."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .types import QueryResult


@dataclass
class AttributionResult:
    """Feature attribution for a single document."""

    doc_id: str
    query_id: str
    feature_importances: dict[str, float]
    base_score: float
    predicted_score: float
    method: str  # "shap" | "lime"


@dataclass
class CounterfactualResult:
    """Minimal feature change that would flip the ranking order of two documents."""

    query_id: str
    doc_a_id: str  # currently ranked higher
    doc_b_id: str  # currently ranked lower
    original_rank_a: int
    original_rank_b: int
    flipping_changes: dict[str, tuple[float, float]]  # feature -> (current, needed)
    delta_score: float  # score gap to close


@dataclass
class BiasResult:
    """Fairness and bias metrics for a query's ranked list."""

    query_id: str
    demographic_parity: dict[str, float] | None = None
    exposure_bias: dict[str, float] | None = None
    position_bias: float | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class MetricResult:
    """Standard IR metrics for a query."""

    query_id: str
    ndcg_at_k: dict[int, float] = field(default_factory=dict)
    precision_at_k: dict[int, float] = field(default_factory=dict)
    map_score: float | None = None
    mrr_score: float | None = None


@dataclass
class AuditReport:
    """
    Full audit report produced by `rankaudit.audit(...)`.

    Holds ranked results, metrics, attributions, counterfactuals, and bias results
    for every query in the audit run.
    """

    query_results: list[QueryResult] = field(default_factory=list)
    metrics: list[MetricResult] = field(default_factory=list)
    attributions: list[AttributionResult] = field(default_factory=list)
    counterfactuals: list[CounterfactualResult] = field(default_factory=list)
    bias: list[BiasResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def explain(self, doc_id: str, query_id: str | None = None) -> str:
        """
        Return a human-readable explanation of why `doc_id` ranked where it did.

        If `query_id` is omitted, uses the first query that contains `doc_id`.
        """
        matches = [
            a for a in self.attributions
            if a.doc_id == doc_id and (query_id is None or a.query_id == query_id)
        ]
        if not matches:
            return f"No attribution data found for doc_id={doc_id!r}."

        att = matches[0]
        sorted_features = sorted(
            att.feature_importances.items(), key=lambda x: abs(x[1]), reverse=True
        )
        lines = [
            f"Explanation for doc '{doc_id}' (query '{att.query_id}') — method: {att.method}",
            f"  Base score   : {att.base_score:.4f}",
            f"  Final score  : {att.predicted_score:.4f}",
            "  Top feature contributions:",
        ]
        for feat, imp in sorted_features[:10]:
            sign = "+" if imp >= 0 else ""
            lines.append(f"    {feat:<30} {sign}{imp:.4f}")
        return "\n".join(lines)

    def summary(self) -> str:
        """Return a brief text summary of the audit report."""
        n_queries = len(self.query_results)
        n_docs = sum(len(qr.results) for qr in self.query_results)
        ndcg_vals = [
            v
            for m in self.metrics
            for k, v in m.ndcg_at_k.items()
            if k == 10
        ]
        avg_ndcg = sum(ndcg_vals) / len(ndcg_vals) if ndcg_vals else None
        lines = [
            "=== RankAudit Report ===",
            f"Queries audited : {n_queries}",
            f"Documents ranked: {n_docs}",
        ]
        if avg_ndcg is not None:
            lines.append(f"Avg NDCG@10     : {avg_ndcg:.4f}")
        if self.attributions:
            lines.append(f"Attributions    : {len(self.attributions)} documents explained")
        if self.counterfactuals:
            lines.append(f"Counterfactuals : {len(self.counterfactuals)} flip examples")
        if self.bias:
            lines.append(f"Bias checks     : {len(self.bias)} queries analysed")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the report to a plain dictionary."""
        def _asdict(obj: Any) -> Any:
            if hasattr(obj, "__dataclass_fields__"):
                return {k: _asdict(v) for k, v in obj.__dict__.items()}
            if isinstance(obj, list):
                return [_asdict(i) for i in obj]
            if isinstance(obj, dict):
                return {k: _asdict(v) for k, v in obj.items()}
            if isinstance(obj, tuple):
                return [_asdict(i) for i in obj]
            return obj

        return _asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialise the report to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)