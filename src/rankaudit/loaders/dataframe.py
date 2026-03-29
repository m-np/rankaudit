"""Load QueryDocPair lists from pandas DataFrames."""

from __future__ import annotations

from ..core.types import QueryDocPair


def from_dataframe(df) -> list[QueryDocPair]:
    """
    Convert a pandas DataFrame to a list of QueryDocPair objects.

    Expected columns
    ----------------
    Required:
        query_id      — str identifier for the query
        query_text    — raw query text
        doc_id        — str identifier for the document
        doc_text      — raw document text

    Optional:
        relevance     — float relevance label (e.g. 0–3 graded or 0/1 binary)
        features      — dict of float features (already parsed), OR individual
                        columns prefixed with ``feat_`` are collected automatically
        <any other>   — stored in metadata

    Parameters
    ----------
    df:
        A pandas DataFrame.

    Returns
    -------
    list[QueryDocPair]
    """
    required = {"query_id", "query_text", "doc_id", "doc_text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    reserved = required | {"relevance", "features"}

    pairs = []
    for _, row in df.iterrows():
        # Build features dict
        if "features" in df.columns and isinstance(row["features"], dict):
            features = dict(row["features"])
        else:
            features = {col: float(row[col]) for col in feat_cols}

        # Build metadata from remaining columns
        metadata = {
            col: row[col]
            for col in df.columns
            if col not in reserved and not col.startswith("feat_")
        }

        pairs.append(
            QueryDocPair(
                query_id=str(row["query_id"]),
                query_text=str(row["query_text"]),
                doc_id=str(row["doc_id"]),
                doc_text=str(row["doc_text"]),
                features=features,
                relevance=float(row["relevance"]) if "relevance" in df.columns else None,
                metadata=metadata,
            )
        )
    return pairs