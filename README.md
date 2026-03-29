# RankAudit

Auditing and explainability library for ranking systems.

**RankAudit** lets engineers and researchers inspect, debug, and explain why a ranker (search engine, recommendation system, or LLM-based re-ranker) produces the output it does.

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

## Features

- **Audit pipeline** — structured audit object covering metrics, attribution, counterfactuals, and bias
- **Feature attribution** — SHAP and LIME adapters tuned for ranking (pointwise, pairwise, listwise)
- **Counterfactual engine** — minimal feature change that flips the ranking order of two documents
- **Bias detector** — demographic parity, exposure bias, position bias across ranked lists
- **RankExplain** — GenAI layer that translates audit reports into plain English (Anthropic/OpenAI)
- **Data loaders** — QueryDocPair, pandas DataFrame, TREC qrel, LETOR/SVM-rank

## Installation

```bash
pip install rankaudit                  # core only (numpy)
pip install "rankaudit[shap]"          # + SHAP attribution
pip install "rankaudit[lime]"          # + LIME attribution
pip install "rankaudit[llm]"           # + RankExplain (Anthropic)
pip install "rankaudit[all]"           # everything
```

## License

Apache 2.0
