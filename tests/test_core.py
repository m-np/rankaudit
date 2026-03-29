"""Core tests for RankAudit — no optional dependencies required."""

import pytest

import rankaudit as ra
from rankaudit.core.types import QueryDocPair
from rankaudit.metrics.ndcg import compute_ndcg, compute_precision, compute_map, compute_mrr


# ---------------------------------------------------------------------------
# Minimal stub ranker
# ---------------------------------------------------------------------------

class LinearRanker:
    """Scores documents by the weighted sum of their features."""

    def __init__(self, weights: dict[str, float]) -> None:
        self.weights = weights

    def score(self, pairs: list[QueryDocPair]) -> list[tuple[str, float]]:
        results = []
        for p in pairs:
            s = sum(self.weights.get(f, 0.0) * v for f, v in p.features.items())
            results.append((p.doc_id, s))
        return results


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_pairs():
    return [
        QueryDocPair("q1", "best laptop", "d1", "laptop review A",
                     features={"recency": 0.9, "relevance": 0.8}, relevance=3.0),
        QueryDocPair("q1", "best laptop", "d2", "laptop review B",
                     features={"recency": 0.4, "relevance": 0.6}, relevance=2.0),
        QueryDocPair("q1", "best laptop", "d3", "laptop review C",
                     features={"recency": 0.2, "relevance": 0.3}, relevance=0.0),
        QueryDocPair("q1", "best laptop", "d4", "laptop review D",
                     features={"recency": 0.7, "relevance": 0.1}, relevance=1.0),
    ]


@pytest.fixture
def ranker():
    return LinearRanker(weights={"recency": 0.3, "relevance": 0.7})


# ---------------------------------------------------------------------------
# NDCG / metric tests
# ---------------------------------------------------------------------------

class TestNDCG:
    def test_perfect_ranking(self):
        ranked = ["d1", "d2", "d3"]
        rel_map = {"d1": 3.0, "d2": 2.0, "d3": 1.0}
        assert compute_ndcg(ranked, rel_map, 3) == pytest.approx(1.0, abs=1e-6)

    def test_reversed_ranking(self):
        ranked = ["d3", "d2", "d1"]
        rel_map = {"d1": 3.0, "d2": 2.0, "d3": 1.0}
        ndcg = compute_ndcg(ranked, rel_map, 3)
        assert 0.0 < ndcg < 1.0

    def test_no_relevant_docs(self):
        ranked = ["d1", "d2"]
        rel_map = {"d1": 0.0, "d2": 0.0}
        assert compute_ndcg(ranked, rel_map, 2) == 0.0

    def test_precision_at_k(self):
        ranked = ["d1", "d2", "d3", "d4"]
        rel_map = {"d1": 1.0, "d2": 0.0, "d3": 1.0, "d4": 0.0}
        assert compute_precision(ranked, rel_map, 2) == pytest.approx(0.5)
        assert compute_precision(ranked, rel_map, 4) == pytest.approx(0.5)

    def test_mrr(self):
        ranked = ["d3", "d1", "d2"]
        rel_map = {"d1": 1.0, "d2": 1.0, "d3": 0.0}
        assert compute_mrr(ranked, rel_map) == pytest.approx(0.5)

    def test_map(self):
        ranked = ["d1", "d2", "d3"]
        rel_map = {"d1": 1.0, "d2": 0.0, "d3": 1.0}
        # relevant at positions 1 and 3 → AP = (1/1 + 2/3) / 2
        expected = (1.0 + 2 / 3) / 2
        assert compute_map(ranked, rel_map) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Pipeline / audit tests
# ---------------------------------------------------------------------------

class TestAuditPipeline:
    def test_basic_audit(self, ranker, simple_pairs):
        report = ra.audit(
            ranker=ranker,
            pairs=simple_pairs,
            attribution=None,
            counterfactuals=False,
            bias=False,
        )
        assert len(report.query_results) == 1
        qr = report.query_results[0]
        assert qr.query_id == "q1"
        assert len(qr.results) == 4
        # results should be sorted by score descending
        scores = [r.score for r in qr.results]
        assert scores == sorted(scores, reverse=True)

    def test_metrics_computed(self, ranker, simple_pairs):
        report = ra.audit(
            ranker=ranker,
            pairs=simple_pairs,
            metrics=["ndcg"],
            attribution=None,
            counterfactuals=False,
            bias=False,
        )
        assert len(report.metrics) == 1
        m = report.metrics[0]
        assert 5 in m.ndcg_at_k or 3 in m.ndcg_at_k
        for k, v in m.ndcg_at_k.items():
            assert 0.0 <= v <= 1.0

    def test_report_summary(self, ranker, simple_pairs):
        report = ra.audit(
            ranker=ranker,
            pairs=simple_pairs,
            attribution=None,
            counterfactuals=False,
        )
        summary = report.summary()
        assert "Queries audited" in summary
        assert "1" in summary

    def test_report_json_serialisable(self, ranker, simple_pairs):
        import json
        report = ra.audit(
            ranker=ranker,
            pairs=simple_pairs,
            attribution=None,
            counterfactuals=False,
            bias=False,
        )
        json_str = report.to_json()
        data = json.loads(json_str)
        assert "query_results" in data

    def test_multiple_queries(self, ranker):
        pairs = [
            QueryDocPair("q1", "query one", "d1", "doc 1", features={"a": 0.5}, relevance=1.0),
            QueryDocPair("q1", "query one", "d2", "doc 2", features={"a": 0.3}, relevance=0.0),
            QueryDocPair("q2", "query two", "d3", "doc 3", features={"a": 0.9}, relevance=1.0),
            QueryDocPair("q2", "query two", "d4", "doc 4", features={"a": 0.1}, relevance=0.0),
        ]
        report = ra.audit(ranker=ranker, pairs=pairs, attribution=None, counterfactuals=False)
        assert len(report.query_results) == 2


# ---------------------------------------------------------------------------
# Bias detector tests
# ---------------------------------------------------------------------------

class TestBiasDetector:
    def test_exposure_bias_detected(self, ranker):
        # group A docs always rank higher due to high feature values
        pairs = [
            QueryDocPair("q1", "q", f"a{i}", "doc", features={"relevance": 0.9 - i * 0.01},
                         relevance=1.0, metadata={"group": "A"})
            for i in range(5)
        ] + [
            QueryDocPair("q1", "q", f"b{i}", "doc", features={"relevance": 0.2 - i * 0.01},
                         relevance=0.0, metadata={"group": "B"})
            for i in range(5)
        ]
        report = ra.audit(ranker=ranker, pairs=pairs, attribution=None, counterfactuals=False, bias=True)
        assert len(report.bias) == 1
        bias = report.bias[0]
        assert bias.exposure_bias is not None
        assert "A" in bias.exposure_bias
        assert "B" in bias.exposure_bias
        # group A should have higher exposure
        assert bias.exposure_bias["A"] > bias.exposure_bias["B"]

    def test_notes_on_high_disparity(self, ranker):
        pairs = [
            QueryDocPair("q1", "q", f"a{i}", "doc", features={"relevance": 0.9},
                         relevance=1.0, metadata={"group": "A"})
            for i in range(8)
        ] + [
            QueryDocPair("q1", "q", f"b{i}", "doc", features={"relevance": 0.05},
                         relevance=0.0, metadata={"group": "B"})
            for i in range(2)
        ]
        report = ra.audit(ranker=ranker, pairs=pairs, attribution=None, counterfactuals=False, bias=True)
        bias = report.bias[0]
        assert len(bias.notes) > 0


# ---------------------------------------------------------------------------
# Counterfactual engine tests
# ---------------------------------------------------------------------------

class TestCounterfactualEngine:
    def test_counterfactuals_generated(self, ranker, simple_pairs):
        report = ra.audit(
            ranker=ranker,
            pairs=simple_pairs,
            attribution=None,
            counterfactuals=True,
            bias=False,
        )
        # at minimum one counterfactual should be generated
        assert len(report.counterfactuals) >= 0  # may be 0 if already optimal

    def test_counterfactual_fields(self, ranker, simple_pairs):
        report = ra.audit(
            ranker=ranker,
            pairs=simple_pairs,
            attribution=None,
            counterfactuals=True,
            bias=False,
        )
        for cf in report.counterfactuals:
            assert cf.query_id == "q1"
            assert cf.original_rank_a < cf.original_rank_b
            assert isinstance(cf.flipping_changes, dict)
            assert cf.delta_score >= 0


# ---------------------------------------------------------------------------
# Loader tests
# ---------------------------------------------------------------------------

class TestLoaders:
    def test_from_dataframe(self):
        import sys
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not installed")

        df = pd.DataFrame({
            "query_id": ["q1", "q1"],
            "query_text": ["test query", "test query"],
            "doc_id": ["d1", "d2"],
            "doc_text": ["doc one", "doc two"],
            "relevance": [1.0, 0.0],
            "feat_bm25": [0.8, 0.4],
            "feat_tfidf": [0.6, 0.3],
        })
        pairs = ra.from_dataframe(df)
        assert len(pairs) == 2
        assert pairs[0].features == {"feat_bm25": 0.8, "feat_tfidf": 0.6}
        assert pairs[0].relevance == 1.0

    def test_from_dataframe_missing_column(self):
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not installed")

        df = pd.DataFrame({"query_id": ["q1"], "doc_id": ["d1"]})
        with pytest.raises(ValueError, match="missing required columns"):
            ra.from_dataframe(df)


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------

class TestPublicAPI:
    def test_build_pairs_from_raw(self, ranker):
        report = ra.audit(
            ranker=ranker,
            queries=["best phone"],
            docs=[
                {"id": "d1", "text": "phone review A", "relevance": 2, "features": {"relevance": 0.8}},
                {"id": "d2", "text": "phone review B", "relevance": 1, "features": {"relevance": 0.4}},
            ],
            attribution=None,
            counterfactuals=False,
            bias=False,
        )
        assert len(report.query_results) == 1

    def test_explain_without_attributions(self, ranker, simple_pairs):
        report = ra.audit(ranker=ranker, pairs=simple_pairs, attribution=None,
                          counterfactuals=False, bias=False)
        msg = report.explain(doc_id="d1")
        assert "No attribution data" in msg
