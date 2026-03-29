"""Core data types for RankAudit."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class QueryDocPair:
    """A single query-document pair with optional features and relevance label."""

    query_id: str
    query_text: str
    doc_id: str
    doc_text: str
    features: dict[str, float] = field(default_factory=dict)
    relevance: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"QueryDocPair(query_id={self.query_id!r}, doc_id={self.doc_id!r}, "
            f"relevance={self.relevance})"
        )


@dataclass
class RankedResult:
    """A document with its rank position and score."""

    doc_id: str
    doc_text: str
    rank: int
    score: float
    features: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryResult:
    """Ranked results for a single query."""

    query_id: str
    query_text: str
    results: list[RankedResult]

    def __len__(self) -> int:
        return len(self.results)

    def get_doc(self, doc_id: str) -> RankedResult | None:
        for r in self.results:
            if r.doc_id == doc_id:
                return r
        return None