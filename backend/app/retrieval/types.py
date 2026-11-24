"""Shared types and metadata filter helpers for MindVault retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class SearchFilters:
    """Optional metadata constraints applied across semantic and BM25 retrieval."""

    document_ids: Optional[list[str]] = None
    file_types: Optional[list[str]] = None
    epoch_min: Optional[int] = None
    epoch_max: Optional[int] = None

    def is_empty(self) -> bool:
        return (
            self.document_ids is None
            and self.file_types is None
            and self.epoch_min is None
            and self.epoch_max is None
        )

    def matches(self, chunk: dict[str, Any]) -> bool:
        """Return True if a chunk record satisfies all configured filters."""
        if self.document_ids is not None:
            if chunk.get("document_id") not in self.document_ids:
                return False
        if self.file_types is not None:
            if chunk.get("file_type") not in self.file_types:
                return False
        epoch = int(chunk.get("epoch", 0))
        if self.epoch_min is not None and epoch < self.epoch_min:
            return False
        if self.epoch_max is not None and epoch > self.epoch_max:
            return False
        return True


@dataclass(frozen=True)
class SearchHit:
    """Unified retrieval hit hydrated with SQLite source metadata."""

    chunk_id: str
    document_id: str
    filename: str
    file_type: str
    source_path: str
    chunk_index: int
    epoch: int
    content: str
    token_count: int
    score: float
    rrf_score: float
    semantic_rank: Optional[int] = None
    keyword_rank: Optional[int] = None
    semantic_score: Optional[float] = None
    keyword_score: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "filename": self.filename,
            "file_type": self.file_type,
            "source_path": self.source_path,
            "chunk_index": self.chunk_index,
            "epoch": self.epoch,
            "content": self.content,
            "token_count": self.token_count,
            "score": self.score,
            "rrf_score": self.rrf_score,
            "semantic_rank": self.semantic_rank,
            "keyword_rank": self.keyword_rank,
            "semantic_score": self.semantic_score,
            "keyword_score": self.keyword_score,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class SearchResult:
    """Full response returned by RetrievalService.search()."""

    query: str
    mode: str
    total: int
    degraded_to_keyword: bool
    warnings: list[str]
    hits: list[SearchHit]
