"""
Load TREC / LETOR format files into QueryDocPair lists.

TREC qrel format (relevance judgements):
    <query_id> 0 <doc_id> <relevance>

TREC run format (ranked results):
    <query_id> Q0 <doc_id> <rank> <score> <run_name>

LETOR SVM-rank format:
    <relevance> qid:<query_id> <feat_id>:<value> ... # <doc_id>
"""

from __future__ import annotations

import re
from pathlib import Path

from ..core.types import QueryDocPair


def load_trec(
    qrel_path: str | Path,
    run_path: str | Path | None = None,
    query_texts: dict[str, str] | None = None,
    doc_texts: dict[str, str] | None = None,
) -> list[QueryDocPair]:
    """
    Load TREC qrel (and optionally run) files into QueryDocPair objects.

    Parameters
    ----------
    qrel_path:
        Path to a TREC qrel file.
    run_path:
        Optional path to a TREC run file.  When provided, only documents
        that appear in the run file are included.
    query_texts:
        Mapping from query_id to query text.  If omitted, query_id is used
        as a placeholder text.
    doc_texts:
        Mapping from doc_id to document text.  If omitted, doc_id is used
        as a placeholder text.

    Returns
    -------
    list[QueryDocPair]
    """
    qrels = _parse_qrel(Path(qrel_path))
    run_scores: dict[tuple[str, str], float] = {}

    if run_path is not None:
        run_scores = _parse_run(Path(run_path))
        # filter to only docs in the run
        doc_ids_in_run = {doc_id for _, doc_id in run_scores}
        qrels = {
            qid: {did: rel for did, rel in docs.items() if did in doc_ids_in_run}
            for qid, docs in qrels.items()
        }

    pairs = []
    for query_id, doc_rels in qrels.items():
        qt = (query_texts or {}).get(query_id, query_id)
        for doc_id, relevance in doc_rels.items():
            dt = (doc_texts or {}).get(doc_id, doc_id)
            pairs.append(
                QueryDocPair(
                    query_id=query_id,
                    query_text=qt,
                    doc_id=doc_id,
                    doc_text=dt,
                    relevance=float(relevance),
                )
            )
    return pairs


def load_letor(path: str | Path) -> list[QueryDocPair]:
    """
    Load a LETOR / SVM-rank format file.

    Each line: ``<relevance> qid:<id> <feat>:<val> ... # <doc_id>``

    Returns
    -------
    list[QueryDocPair]
    """
    pairs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # strip inline comment to get doc_id
            doc_id = ""
            if "#" in line:
                line, comment = line.split("#", 1)
                doc_id = comment.strip()

            tokens = line.split()
            relevance = float(tokens[0])
            query_id = tokens[1].split(":")[1]

            features: dict[str, float] = {}
            for tok in tokens[2:]:
                if ":" in tok:
                    fid, fval = tok.split(":", 1)
                    features[f"feat_{fid}"] = float(fval)

            pairs.append(
                QueryDocPair(
                    query_id=query_id,
                    query_text=query_id,
                    doc_id=doc_id or f"doc_{len(pairs)}",
                    doc_text=doc_id or "",
                    features=features,
                    relevance=relevance,
                )
            )
    return pairs


# ------------------------------------------------------------------
# Parsers
# ------------------------------------------------------------------

def _parse_qrel(path: Path) -> dict[str, dict[str, float]]:
    """Parse a TREC qrel file → {query_id: {doc_id: relevance}}."""
    qrels: dict[str, dict[str, float]] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            qid, _, doc_id, rel = parts[0], parts[1], parts[2], parts[3]
            qrels.setdefault(qid, {})[doc_id] = float(rel)
    return qrels


def _parse_run(path: Path) -> dict[tuple[str, str], float]:
    """Parse a TREC run file → {(query_id, doc_id): score}."""
    scores: dict[tuple[str, str], float] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            qid, _, doc_id, _, score, _ = parts
            scores[(qid, doc_id)] = float(score)
    return scores