# RankAudit

**Auditing and explainability library for ranking systems.**

[![PyPI](https://img.shields.io/pypi/v/rankaudit)](https://pypi.org/project/rankaudit/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-19%20passed-brightgreen)]()

---

## Why does this exist?

Ranking systems — search engines, recommendation feeds, LLM re-rankers — decide what people see and in what order. They influence hiring decisions, loan approvals, news consumption, and product discovery. Yet most rankers are **black boxes**: they produce a ranked list with no explanation of why document A ranked above document B, whether the ranking is fair across demographic groups, or what it would take for a lower-ranked item to move up.

Existing tools don't close this gap:

- **Standard explainability libraries** (SHAP, LIME) treat ranking as pointwise regression — they explain a single document's score in isolation, ignoring that ranking is inherently comparative. A feature that boosts a score by 0.1 matters very differently depending on the gap to the next document.
- **IR evaluation libraries** (pytrec_eval, ranx) compute quality metrics (NDCG, MAP) but give no insight into *why* a ranker performs the way it does or where it is unfair.
- **Fairness toolkits** (Fairlearn, AI Fairness 360) are designed for classifiers, not ranked lists.

**RankAudit fills this gap.** It is a single toolkit that lets engineers and researchers:

1. **Measure** ranking quality with standard IR metrics (NDCG, MAP, MRR, Precision@k).
2. **Explain** why each document ranked where it did using ranking-aware feature attribution.
3. **Interrogate** ranking decisions with counterfactual questions: *"What would it take for this document to rank #1?"*
4. **Detect** bias across demographic groups using exposure, parity, and position-relevance signals.
5. **Communicate** findings in plain English via an LLM explanation layer.

---

## Installation

```bash
pip install rankaudit                  # core only (numpy)
pip install "rankaudit[shap]"          # + SHAP attribution
pip install "rankaudit[lime]"          # + LIME attribution
pip install "rankaudit[llm]"           # + RankExplain (Anthropic/OpenAI)
pip install "rankaudit[pandas]"        # + DataFrame loader
pip install "rankaudit[all]"           # everything
```

---

## Quick start

```python
import rankaudit as ra

report = ra.audit(
    ranker=my_ranker,
    queries=["best laptop under $1000"],
    docs=doc_corpus,
    metrics=["ndcg", "fairness"],
)

print(report.summary())
print(report.explain(doc_id="doc_42"))
```

---

## Inputs

### Option 1 — Raw queries + doc list (simplest)

```python
queries = ["best laptop under $1000", "noise cancelling headphones"]

docs = [
    {
        "id": "doc_1",
        "text": "The MacBook Air M3 offers excellent value...",
        "relevance": 3,          # graded relevance label (0–3), optional
        "group": "apple",        # group label for bias analysis, optional
        "features": {            # numeric features your ranker uses
            "bm25": 0.82,
            "semantic_sim": 0.91,
            "recency": 0.74,
            "click_rate": 0.55,
        },
    },
    {
        "id": "doc_2",
        "text": "Dell XPS 15 review: powerful but heavy...",
        "relevance": 2,
        "group": "dell",
        "features": {
            "bm25": 0.61,
            "semantic_sim": 0.78,
            "recency": 0.45,
            "click_rate": 0.38,
        },
    },
    # ... more docs
]

report = ra.audit(ranker=my_ranker, queries=queries, docs=docs)
```

### Option 2 — QueryDocPair objects (full control)

```python
from rankaudit import QueryDocPair

pairs = [
    QueryDocPair(
        query_id="q1",
        query_text="best laptop under $1000",
        doc_id="doc_1",
        doc_text="The MacBook Air M3 offers excellent value...",
        features={"bm25": 0.82, "semantic_sim": 0.91, "recency": 0.74},
        relevance=3.0,
        metadata={"group": "apple"},
    ),
    # ...
]

report = ra.audit(ranker=my_ranker, pairs=pairs)
```

### Option 3 — pandas DataFrame

```python
import pandas as pd
import rankaudit as ra

df = pd.DataFrame({
    "query_id":   ["q1",    "q1",    "q1"],
    "query_text": ["best laptop"] * 3,
    "doc_id":     ["doc_1", "doc_2", "doc_3"],
    "doc_text":   ["MacBook Air...", "Dell XPS...", "Lenovo ThinkPad..."],
    "relevance":  [3.0,     2.0,     1.0],
    "feat_bm25":  [0.82,    0.61,    0.55],      # columns prefixed feat_ become features
    "feat_semantic_sim": [0.91, 0.78, 0.70],
    "group":      ["apple", "dell", "lenovo"],   # any other column goes to metadata
})

pairs = ra.from_dataframe(df)
report = ra.audit(ranker=my_ranker, pairs=pairs)
```

### Option 4 — TREC / LETOR benchmark files

```python
pairs = ra.load_trec(qrel_path="robust04.qrel", run_path="bm25.run")
pairs = ra.load_letor(path="MQ2007/Fold1/train.txt")
report = ra.audit(ranker=my_ranker, pairs=pairs)
```

### Implementing a ranker

Any object with a `score` method works:

```python
class MyRanker:
    def score(self, pairs: list[QueryDocPair]) -> list[tuple[str, float]]:
        # return (doc_id, score) for every pair — order doesn't matter
        return [
            (p.doc_id, sum(p.features.values()))
            for p in pairs
        ]
```

---

## Outputs

`ra.audit(...)` returns an `AuditReport` object.

### report.summary()

```
=== RankAudit Report ===
Queries audited : 2
Documents ranked: 20
Avg NDCG@10     : 0.7841
Attributions    : 20 documents explained
Counterfactuals : 4 flip examples
Bias checks     : 2 queries analysed
```

### report.metrics

NDCG, Precision@k, MAP, and MRR for every query:

```python
for m in report.metrics:
    print(m.query_id, m.ndcg_at_k, m.map_score, m.mrr_score)

# q1  {1: 1.0, 3: 0.92, 5: 0.87, 10: 0.78}  0.81  1.0
# q2  {1: 0.5, 3: 0.71, 5: 0.74, 10: 0.69}  0.67  0.5
```

### report.explain(doc_id)

Feature-level explanation for why a document ranked where it did:

```python
print(report.explain(doc_id="doc_2"))
```

```
Explanation for doc 'doc_2' (query 'q1') — method: shap
  Base score   : 0.5312
  Final score  : 0.7841
  Top feature contributions:
    semantic_sim                   +0.1820
    bm25                           +0.0934
    recency                        -0.0381
    click_rate                     -0.0144
```

### report.counterfactuals

Minimal feature change that would flip the order of two documents:

```python
for cf in report.counterfactuals:
    print(f"Query {cf.query_id}: '{cf.doc_b_id}' (rank {cf.original_rank_b}) "
          f"would overtake '{cf.doc_a_id}' (rank {cf.original_rank_a}) if:")
    for feat, (current, needed) in cf.flipping_changes.items():
        print(f"  {feat}: {current:.3f} → {needed:.3f}")
```

```
Query q1: 'doc_2' (rank 2) would overtake 'doc_1' (rank 1) if:
  recency: 0.450 → 0.720
```

### report.bias

Fairness metrics across group labels:

```python
for b in report.bias:
    print(b.query_id, b.exposure_bias, b.demographic_parity)
    for note in b.notes:
        print(" [!]", note)
```

```
q1  {'apple': 0.4821, 'dell': 0.2103, 'lenovo': 0.1874}  {'apple': 0.6, 'dell': 0.3, 'lenovo': 0.1}
 [!] Exposure ratio between groups is 2.57x — significant disparity detected.
 [!] Demographic parity gap is 0.50 — one group dominates top-k results.
```

### Plain-English explanations via RankExplain

```python
from anthropic import Anthropic
from rankaudit import RankExplain

explainer = RankExplain(client=Anthropic(), model="claude-sonnet-4-6")

print(explainer.explain_report(report))
print(explainer.explain_doc(report, doc_id="doc_2"))
print(explainer.explain_counterfactuals(report))
print(explainer.explain_bias(report))
```

```
The ranking system performs well overall (NDCG@10: 0.78) but shows a notable
recency penalty — newer documents are discounted by ~30% relative to older ones
with comparable semantic scores. The 'apple' group receives 2.5x more exposure
than other groups, which may warrant investigation if group identity should not
influence ranking position...
```

### Serialise to JSON

```python
report.to_json()   # → full report as a JSON string
report.to_dict()   # → plain Python dict
```

---

## Roadmap

| Month | Milestone |
|-------|-----------|
| April 2026 | v0.1.0 on PyPI — core pipeline + SHAP adapter |
| May 2026 | Counterfactual engine v1 |
| July 2026 | ARB — Accountability Ranking Benchmark on HuggingFace |
| August 2026 | RankExplain v1 — GenAI explanation layer |

---

## Contributing

Contributions are welcome. Please open an issue before submitting a large PR.

## License

Copyright 2026 Mandar Narendra Parab. Licensed under the [Apache License 2.0](LICENSE).