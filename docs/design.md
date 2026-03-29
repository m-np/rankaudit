# RankAudit — Internal Design Document

## Purpose

RankAudit is an auditing and explainability library for ranking systems. It answers a single question any team operating a ranker should be able to answer: **why did document X rank where it did, and what would it take to change that?**

The library targets search engines, recommendation systems, and LLM-based re-rankers. It is designed to be ranker-agnostic — the only requirement on the ranker is a `score(pairs) -> [(doc_id, score)]` interface.

---

## Project Structure

```
rankaudit/
├── src/rankaudit/
│   ├── __init__.py              # Public API: ra.audit(), ra.from_dataframe(), ra.load_trec()
│   ├── core/
│   │   ├── types.py             # QueryDocPair, RankedResult, QueryResult
│   │   ├── report.py            # AuditReport — output container + .explain() / .summary() / .to_json()
│   │   └── pipeline.py          # AuditPipeline — orchestrates the five audit steps
│   ├── attribution/
│   │   ├── shap_adapter.py      # Ranking-aware SHAP wrapper
│   │   └── lime_adapter.py      # LIME tabular explainer wrapper
│   ├── counterfactual/
│   │   └── engine.py            # CounterfactualEngine — minimal-flip perturbation search
│   ├── bias/
│   │   └── detector.py          # BiasDetector — exposure, parity, position-relevance
│   ├── metrics/
│   │   └── ndcg.py              # NDCG, Precision@k, MAP, MRR
│   ├── explain/
│   │   └── rank_explain.py      # RankExplain — LLM translation layer
│   └── loaders/
│       ├── dataframe.py         # pandas DataFrame → QueryDocPair
│       └── trec.py              # TREC qrel + LETOR/SVM-rank → QueryDocPair
├── tests/
│   └── test_core.py
├── docs/
│   └── design.md                # this file
├── pyproject.toml
└── README.md
```

---

## Architecture

### Data model

All data flows through a single normalised type:

```python
@dataclass
class QueryDocPair:
    query_id: str
    query_text: str
    doc_id: str
    doc_text: str
    features: dict[str, float]   # numeric feature vector
    relevance: float | None      # graded or binary relevance label
    metadata: dict[str, Any]     # group labels, timestamps, etc.
```

Three input formats are accepted and normalised to `QueryDocPair` before hitting the pipeline:

| Format | Loader |
|--------|--------|
| Python dataclass | `QueryDocPair(...)` directly |
| pandas DataFrame | `ra.from_dataframe(df)` |
| TREC qrel / LETOR | `ra.load_trec(...)` / `ra.load_letor(...)` |

### Audit pipeline (5 steps)

`AuditPipeline.run(pairs)` executes five steps per query group:

```
QueryDocPairs
    │
    ├─ 1. Score & rank          → QueryResult (sorted RankedResult list)
    ├─ 2. IR metrics            → MetricResult (NDCG, P@k, MAP, MRR)
    ├─ 3. Feature attribution   → [AttributionResult] per document
    ├─ 4. Counterfactuals       → [CounterfactualResult] per adjacent pair
    └─ 5. Bias analysis         → BiasResult per query
                                        │
                                        └─→ AuditReport
```

All five steps are independently opt-in/opt-out via pipeline constructor arguments.

---

## Key Design Decisions

### 1. Ranking-aware SHAP (original IP)

**Problem:** Standard SHAP treats the scorer as a pointwise regression function and assigns importances based on marginal contributions to a single predicted score. In a ranking context this is wrong — a feature that inflates a document's score by 0.1 matters more if the score gap to the next document is 0.05 than if it's 0.5.

**Solution:** `SHAPRankingAdapter` wraps `KernelExplainer` on the ranker's `score()` function and uses the full pair list as the background dataset, so the expected value (SHAP baseline) is the average score across the ranked list for that query rather than an out-of-distribution baseline. This makes importances relative to the competitive context of the specific query.

For rankers that expose a `shap_tree_model` attribute, `TreeExplainer` can be swapped in for speed.

### 2. Counterfactual flip engine (original IP)

**Problem:** Standard counterfactual tools (DiCE, CARLA) are designed for classification or single-output regression — they ask "what is the minimal change to flip the class prediction?" Ranking is a comparative problem: the question is "what is the minimal change to doc B's features such that `score(B) > score(A)`?"

**Solution:** `CounterfactualEngine` implements coordinate-wise greedy perturbation that directly targets the score gap `score(A) - score(B)`. It:
1. Identifies adjacent pairs in the ranked list (doc_a ranked above doc_b).
2. Perturbs doc_b's features one dimension at a time by a fractional `step_size`.
3. Scores `[pair_a, trial_b]` after each perturbation.
4. Records the first feature set that achieves `score(trial_b) > score(pair_a)`.
5. Falls back to a partial result (progress made, no full flip) if `max_steps` is reached.

This produces human-interpretable outputs like:
> "Doc B would have ranked #1 if its `recency` score were ≥ 0.72 (currently 0.41)."

### 3. Three-signal bias detector

A single bias number is not actionable. `BiasDetector` computes three orthogonal signals:

| Signal | What it measures | When it fires |
|--------|-----------------|---------------|
| **Position bias** | Spearman correlation between relevance labels and rank positions | Correlation near -1 → ranker ignores relevance |
| **Exposure bias** | Average log-discounted exposure per group, normalised by group size | Ratio > 2× between groups |
| **Demographic parity** | Fraction of top-k results per group | Gap > 0.4 between groups |

Group membership is read from `metadata["group"]` on each `QueryDocPair`. When no group labels are present, only position bias is computed (safe default).

### 4. LLM layer supports both Anthropic and OpenAI SDK

`RankExplain` detects which SDK is in use at call time:

```python
# Anthropic SDK
if hasattr(client, "messages") and hasattr(client.messages, "create"):
    ...

# OpenAI SDK
elif hasattr(client, "chat"):
    ...
```

This avoids an adapter layer while keeping the library SDK-agnostic. Default model is `claude-sonnet-4-6`.

### 5. Optional dependencies, hard numpy-only core

The core audit pipeline (pipeline, metrics, counterfactuals, bias) requires only `numpy`. SHAP, LIME, pandas, and the Anthropic SDK are optional extras declared in `pyproject.toml`:

```toml
[project.optional-dependencies]
shap  = ["shap>=0.44"]
lime  = ["lime>=0.2"]
llm   = ["anthropic>=0.25"]
pandas = ["pandas>=2.0"]
all   = ["shap>=0.44", "lime>=0.2", "anthropic>=0.25", "pandas>=2.0"]
```

When an optional dependency is missing, the adapter logs a `warnings.warn` and falls back gracefully (raw features for SHAP, empty list for LIME).

---

## Roadmap

| Month | Milestone |
|-------|-----------|
| April 2026 | Publish `rankaudit` v0.1.0 to PyPI — core pipeline + SHAP adapter |
| May 2026 | Counterfactual engine v1, CSQ paper revision |
| July 2026 | ARB (Accountability Ranking Benchmark) — HuggingFace dataset + leaderboard |
| August 2026 | RankExplain v1 — GenAI explanation layer |

---

## Extension points

- **Custom rankers:** implement `score(pairs: list[QueryDocPair]) -> list[tuple[str, float]]`
- **Custom metrics:** add a module under `rankaudit/metrics/` and reference it in `AuditPipeline._compute_metrics`
- **Custom attribution:** subclass `SHAPRankingAdapter` or implement `explain_query(qr, pairs) -> list[AttributionResult]`
- **ARB benchmark:** will live in a separate `rankaudit-arb` package and call `ra.audit()` as an evaluator
