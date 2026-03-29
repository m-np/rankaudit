"""
RankAudit — auditing and explainability library for ranking systems.

Quick start
-----------
>>> import rankaudit as ra
>>>
>>> report = ra.audit(
...     ranker=my_ranker,
...     queries=["best laptop under $1000"],
...     docs=doc_corpus,
...     metrics=["ndcg", "fairness"],
... )
>>> print(report.summary())
>>> print(report.explain(doc_id="doc_42"))
"""

from __future__ import annotations

from .core.pipeline import AuditPipeline, Ranker
from .core.report import (
    AuditReport,
    AttributionResult,
    BiasResult,
    CounterfactualResult,
    MetricResult,
)
from .core.types import QueryDocPair, QueryResult, RankedResult
from .explain.rank_explain import RankExplain
from .loaders.dataframe import from_dataframe
from .loaders.trec import load_letor, load_trec


def audit(
    ranker: Ranker,
    pairs: list[QueryDocPair] | None = None,
    *,
    queries: list[str] | None = None,
    docs: list[dict] | None = None,
    metrics: list[str] | None = None,
    attribution: str | None = "shap",
    counterfactuals: bool = True,
    bias: bool = True,
    top_k: int = 10,
) -> AuditReport:
    """
    Run a full ranking audit and return an AuditReport.

    You can pass data in two ways:

    **Option 1 — pre-built pairs (recommended):**

    >>> pairs = [QueryDocPair(query_id="q1", query_text="...", doc_id="d1", ...)]
    >>> report = ra.audit(ranker=my_ranker, pairs=pairs)

    **Option 2 — raw queries + doc dicts:**

    >>> report = ra.audit(
    ...     ranker=my_ranker,
    ...     queries=["best laptop under $1000"],
    ...     docs=[{"id": "doc1", "text": "...", "relevance": 2}],
    ... )

    Parameters
    ----------
    ranker:
        Any object implementing `score(pairs) -> list[(doc_id, score)]`.
    pairs:
        Pre-built list of QueryDocPair objects.
    queries:
        Raw query strings (used with `docs`).
    docs:
        List of dicts with keys: ``id``, ``text``, and optionally
        ``relevance``, ``features`` (dict), ``group`` (str).
    metrics:
        Metric names to compute.  Supported: "ndcg", "precision",
        "map", "mrr", "fairness", "stability".
    attribution:
        Feature attribution method: "shap", "lime", or None.
    counterfactuals:
        Whether to run the counterfactual flip engine.
    bias:
        Whether to run bias analysis.
    top_k:
        Ranking cut-off for metric computation.

    Returns
    -------
    AuditReport
    """
    if pairs is None:
        if queries is None or docs is None:
            raise ValueError(
                "Provide either `pairs` or both `queries` and `docs`."
            )
        pairs = _build_pairs(queries, docs)

    pipeline = AuditPipeline(
        ranker=ranker,
        metrics=metrics or ["ndcg", "fairness"],
        attribution_method=attribution,
        run_counterfactuals=counterfactuals,
        run_bias=bias,
        top_k=top_k,
    )
    return pipeline.run(pairs)


def _build_pairs(
    queries: list[str], docs: list[dict]
) -> list[QueryDocPair]:
    """Build QueryDocPair objects from raw query strings and doc dicts."""
    pairs = []
    for q_idx, query_text in enumerate(queries):
        query_id = f"q{q_idx}"
        for doc in docs:
            metadata = {
                k: v for k, v in doc.items()
                if k not in {"id", "text", "relevance", "features", "group"}
            }
            if "group" in doc:
                metadata["group"] = doc["group"]
            pairs.append(
                QueryDocPair(
                    query_id=query_id,
                    query_text=query_text,
                    doc_id=str(doc.get("id", f"doc_{len(pairs)}")),
                    doc_text=str(doc.get("text", "")),
                    features=dict(doc.get("features") or {}),
                    relevance=doc.get("relevance"),
                    metadata=metadata,
                )
            )
    return pairs


__all__ = [
    "audit",
    "AuditPipeline",
    "AuditReport",
    "AttributionResult",
    "BiasResult",
    "CounterfactualResult",
    "MetricResult",
    "QueryDocPair",
    "QueryResult",
    "RankedResult",
    "Ranker",
    "RankExplain",
    "from_dataframe",
    "load_trec",
    "load_letor",
]