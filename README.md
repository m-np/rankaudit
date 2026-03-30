# RankAudit

**Auditing and explainability toolkit for ranking systems.**

[![PyPI](https://img.shields.io/pypi/v/rankaudit)](https://pypi.org/project/rankaudit/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/m-np/rankaudit/blob/main/LICENSE)
[![Tests](https://img.shields.io/badge/tests-19%20passed-brightgreen)](https://github.com/m-np/rankaudit/blob/main)
[![Python](https://img.shields.io/pypi/pyversions/rankaudit)](https://pypi.org/project/rankaudit/)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-ARB%20Benchmark-yellow)](https://huggingface.co/datasets/m-np/arb)

> Ranking systems decide what people see and in what order — jobs, loans, news, products. RankAudit makes them inspectable.

---

## What is RankAudit?

Most ranking systems — search engines, recommendation feeds, LLM re-rankers are black boxes. They produce an ordered list with no explanation of *why* document A ranked above B, whether the ranking is fair across demographic groups, or what a lower-ranked item would need to change to move up.

RankAudit is a single Python library that closes this gap. It gives engineers and researchers four capabilities:

| Capability | What it answers |
|---|---|
| **Measure** | How good is the ranking? (NDCG, MAP, MRR, Precision@k) |
| **Explain** | Why did this document rank here? (ranking-aware SHAP/LIME) |
| **Interrogate** | What would it take for this document to rank higher? (counterfactuals) |
| **Detect** | Is the ranking fair across groups? (exposure, parity, position-relevance) |

---

## Why not existing tools?

| Tool | Gap |
|---|---|
| SHAP / LIME | Explain a single score in isolation — ignore that ranking is comparative |
| pytrec_eval / ranx | Compute quality metrics but give no insight into *why* or *where* unfairness lives |
| Fairlearn / AI Fairness 360 | Designed for classifiers, not ranked lists |
| **RankAudit** | Ranking-native attribution + counterfactuals + fairness, unified API |

---

## Installation

```bash
pip install rankaudit                  # core only (numpy)
pip install "rankaudit[shap]"          # + SHAP attribution
pip install "rankaudit[lime]"          # + LIME attribution
pip install "rankaudit[llm]"           # + RankExplain (Anthropic / OpenAI)
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

RankAudit accepts four input formats. All normalise to the same internal `QueryDocPair` representation.

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
    ...
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
    ...
]

report = ra.audit(ranker=my_ranker, pairs=pairs)
```

### Option 3 — pandas DataFrame

```python
import pandas as pd
import rankaudit as ra

df = pd.DataFrame({
    "query_id":          ["q1",    "q1",    "q1"],
    "query_text":        ["best laptop"] * 3,
    "doc_id":            ["doc_1", "doc_2", "doc_3"],
    "doc_text":          ["MacBook Air...", "Dell XPS...", "Lenovo ThinkPad..."],
    "relevance":         [3.0,     2.0,     1.0],
    "feat_bm25":         [0.82,    0.61,    0.55],   # feat_ prefix → features dict
    "feat_semantic_sim": [0.91,    0.78,    0.70],
    "group":             ["apple", "dell",  "lenovo"],
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

Any object with a `score` method works — scikit-learn estimators, PyTorch modules, ONNX sessions, and plain callables all qualify.

```python
class MyRanker:
    def score(self, pairs: list[QueryDocPair]) -> list[tuple[str, float]]:
        # Return (doc_id, score) for every pair — order doesn't matter
        return [(p.doc_id, sum(p.features.values())) for p in pairs]
```

---

## Outputs

`ra.audit(...)` returns an `AuditReport` object with four output surfaces.

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

### report.metrics — ranking quality

NDCG, Precision@k, MAP, and MRR for every query:

```python
for m in report.metrics:
    print(m.query_id, m.ndcg_at_k, m.map_score, m.mrr_score)

# q1  {1: 1.0, 3: 0.92, 5: 0.87, 10: 0.78}  0.81  1.0
# q2  {1: 0.5, 3: 0.71, 5: 0.74, 10: 0.69}  0.67  0.5
```

### report.explain() — feature attribution

Ranking-aware attribution showing which features drove a document's position:

```python
print(report.explain(doc_id="doc_2"))
```

```
Explanation for doc 'doc_2' (query 'q1') — method: shap
  Base score   : 0.5312
  Final score  : 0.7841
  Top feature contributions:
    semantic_sim    +0.1820
    bm25            +0.0934
    recency         -0.0381
    click_rate      -0.0144
```

### report.counterfactuals — rank flip analysis

Minimal feature change that would flip the order of two documents:

```python
for cf in report.counterfactuals:
    print(f"Query {cf.query_id}: '{cf.doc_b_id}' (rank {cf.original_rank_b}) "
          f"would overtake '{cf.doc_a_id}' (rank {cf.original_rank_a}) if:")
    for feat, (current, needed) in cf.flipping_changes.items():
        print(f"  {feat}: {current:.3f} → {needed:.3f}")

# Query q1: 'doc_2' (rank 2) would overtake 'doc_1' (rank 1) if:
#   recency: 0.450 → 0.720
```

### report.bias — fairness metrics

Exposure and demographic parity across group labels:

```python
for b in report.bias:
    print(b.query_id, b.exposure_bias, b.demographic_parity)
    for note in b.notes:
        print(" [!]", note)

# q1  {'apple': 0.4821, 'dell': 0.2103, 'lenovo': 0.1874}  ...
# [!] Exposure ratio between groups is 2.57x — significant disparity detected.
# [!] Demographic parity gap is 0.50 — one group dominates top-k results.
```

### RankExplain — plain-English explanations via LLM

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

### Serialisation

```python
report.to_json()   # full report as a JSON string
report.to_dict()   # plain Python dict
```

---

## ARB — Accountability Ranking Benchmark

RankAudit ships with **ARB**, a standardised evaluation suite for auditing rankers across three axes: fairness, stability under perturbation, and explainability consistency. ARB is available on HuggingFace and provides:

- Curated query-document sets across news, e-commerce, and government document domains
- Ground-truth relevance labels and protected-attribute annotations
- A public leaderboard for comparing ranker accountability across systems

> **Coming July 2026** — ARB and the leaderboard are under active development.

[View the ARB dataset on HuggingFace →](https://huggingface.co/datasets/m-np/arb)

---

## Roadmap

| Milestone | Target | Status |
|---|---|---|
| v0.1.0 on PyPI — core pipeline + SHAP adapter | April 2026 | 🔄 In progress |
| Counterfactual engine v1 | May 2026 | 📋 Planned |
| ARB benchmark on HuggingFace | July 2026 | 📋 Planned |
| RankExplain v1 — LLM explanation layer | August 2026 | 📋 Planned |
| ONNX ranker adapter | Q3 2026 | 📋 Planned |
| Streamlit / Gradio web demo | Q3 2026 | 📋 Planned |

---

## Citation

If you use RankAudit in your research, please cite:

```bibtex
@software{parab2026rankaudit,
  author  = {Parab, Mandar Narendra},
  title   = {{RankAudit}: Auditing and Explainability for Ranking Systems},
  year    = {2026},
  url     = {https://github.com/m-np/rankaudit},
  license = {Apache-2.0}
}
```

---

## Contributing

Contributions are welcome. Please open an issue before submitting a large PR so we can align on direction. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

Copyright 2026 Mandar Narendra Parab. Licensed under the [Apache License 2.0](https://github.com/m-np/rankaudit/blob/main/LICENSE).