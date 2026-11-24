"""Thread-safe local BM25 keyword index backed by rank_bm25."""

from __future__ import annotations

import re
import threading
from typing import Any, Optional

from rank_bm25 import BM25L

from app.retrieval.types import SearchFilters

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def tokenize_for_bm25(text: str) -> list[str]:
    """Tokenize text deterministically for offline BM25 indexing and querying."""
    return _TOKEN_RE.findall((text or "").lower())


class BM25Index:
    """In-memory BM25L index synchronized with SQLite indexed chunks."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._entries: dict[str, dict[str, Any]] = {}
        self._ordered_ids: list[str] = []
        self._bm25: Optional[BM25L] = None

    def _rebuild_model(self) -> None:
        ordered_ids: list[str] = []
        corpus: list[list[str]] = []
        for chunk_id, entry in self._entries.items():
            tokens = entry["tokens"]
            if tokens:
                ordered_ids.append(chunk_id)
                corpus.append(tokens)
        self._ordered_ids = ordered_ids
        self._bm25 = BM25L(corpus) if corpus else None

    @staticmethod
    def _make_entry(chunk: dict[str, Any]) -> dict[str, Any]:
        tokens = tokenize_for_bm25(str(chunk.get("content", "")))
        return {
            "id": str(chunk["id"]),
            "document_id": str(chunk["document_id"]),
            "file_type": str(chunk.get("file_type", "")),
            "epoch": int(chunk.get("epoch", 0)),
            "tokens": tokens,
            "token_set": frozenset(tokens),
        }

    def build(self, chunks: list[dict[str, Any]]) -> None:
        """Replace index contents from a full list of indexed chunk records."""
        with self._lock:
            self._entries = {str(c["id"]): self._make_entry(c) for c in chunks}
            self._rebuild_model()

    def upsert_chunks(self, chunks: list[dict[str, Any]]) -> None:
        """Insert or update chunks in the BM25 index."""
        if not chunks:
            return
        with self._lock:
            for chunk in chunks:
                cid = str(chunk["id"])
                self._entries[cid] = self._make_entry(chunk)
            self._rebuild_model()

    def remove_chunks(self, chunk_ids: list[str]) -> None:
        """Remove specific chunk IDs from the BM25 index."""
        if not chunk_ids:
            return
        with self._lock:
            removed = False
            for cid in chunk_ids:
                if cid in self._entries:
                    del self._entries[cid]
                    removed = True
            if removed:
                self._rebuild_model()

    def remove_document(self, doc_id: str) -> None:
        """Remove all chunks belonging to a document ID."""
        with self._lock:
            to_delete = [
                cid for cid, entry in self._entries.items() if entry["document_id"] == doc_id
            ]
            if to_delete:
                for cid in to_delete:
                    del self._entries[cid]
                self._rebuild_model()

    def count(self) -> int:
        """Return total number of tracked chunks in the BM25 index."""
        with self._lock:
            return len(self._entries)

    def search(
        self,
        query: str,
        top_k: int,
        filters: Optional[SearchFilters] = None,
    ) -> list[tuple[str, float]]:
        """Return top_k (chunk_id, bm25_score) tuples matching query and filters."""
        if top_k <= 0:
            return []
        query_tokens = tokenize_for_bm25(query)
        if not query_tokens:
            return []
        query_token_set = frozenset(query_tokens)

        with self._lock:
            if self._bm25 is None or not self._ordered_ids:
                return []

            raw_scores = self._bm25.get_scores(query_tokens)
            scored: list[tuple[float, str]] = []
            for idx, cid in enumerate(self._ordered_ids):
                score = float(raw_scores[idx])
                if score <= 0.0:
                    continue
                entry = self._entries[cid]
                if not (entry["token_set"] & query_token_set):
                    continue
                if filters is not None and not filters.matches(entry):
                    continue
                scored.append((score, cid))

            scored.sort(key=lambda item: (-item[0], item[1]))
            return [(cid, round(score, 6)) for score, cid in scored[:top_k]]
