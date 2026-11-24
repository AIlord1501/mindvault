"""Persistent local ChromaDB vector store synchronized with SQLite chunk IDs."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.retrieval.types import SearchFilters

# Silence noisy telemetry bug in chromadb 0.4.22 even when telemetry is disabled.
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)


def build_chroma_where(filters: Optional[SearchFilters]) -> Optional[dict[str, Any]]:
    """Build a ChromaDB 0.4.22-compatible metadata `where` clause."""
    if filters is None or filters.is_empty():
        return None

    clauses: list[dict[str, Any]] = []
    if filters.document_ids is not None:
        if len(filters.document_ids) == 1:
            clauses.append({"document_id": {"$eq": filters.document_ids[0]}})
        else:
            clauses.append({"document_id": {"$in": list(filters.document_ids)}})

    if filters.file_types is not None:
        if len(filters.file_types) == 1:
            clauses.append({"file_type": {"$eq": filters.file_types[0]}})
        else:
            clauses.append({"file_type": {"$in": list(filters.file_types)}})

    if filters.epoch_min is not None:
        clauses.append({"epoch": {"$gte": int(filters.epoch_min)}})

    if filters.epoch_max is not None:
        clauses.append({"epoch": {"$lte": int(filters.epoch_max)}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


class ChromaVectorStore:
    """Manages a persistent local ChromaDB collection keyed by SQLite chunk IDs."""

    def __init__(self, persist_dir: Path, collection_name: str = "mindvault_chunks") -> None:
        self._persist_dir = persist_dir
        self._collection_name = collection_name
        self._lock = threading.RLock()
        self._client: Any = None
        self._collection: Any = None

    def _ensure_collection(self) -> Any:
        with self._lock:
            if self._collection is not None:
                return self._collection

            self._persist_dir.mkdir(parents=True, exist_ok=True)
            if self._client is None:
                self._client = chromadb.PersistentClient(
                    path=str(self._persist_dir),
                    settings=ChromaSettings(
                        anonymized_telemetry=False,
                        allow_reset=True,
                    ),
                )
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            return self._collection

    def upsert_chunks(
        self,
        chunks: list[dict[str, Any]],
        embeddings: list[list[float]],
        batch_size: int = 32,
    ) -> None:
        """Idempotently upsert chunk vectors keyed by SQLite chunk IDs."""
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have identical lengths.")
        if not chunks:
            return

        safe_batch = max(1, int(batch_size))
        with self._lock:
            collection = self._ensure_collection()
            for start in range(0, len(chunks), safe_batch):
                batch_chunks = chunks[start : start + safe_batch]
                batch_vecs = embeddings[start : start + safe_batch]
                collection.upsert(
                    ids=[str(item["id"]) for item in batch_chunks],
                    embeddings=batch_vecs,
                    documents=[str(item["content"]) for item in batch_chunks],
                    metadatas=[
                        {
                            "document_id": str(item["document_id"]),
                            "file_type": str(item.get("file_type", "")),
                            "filename": str(item.get("filename", "")),
                            "chunk_index": int(item.get("chunk_index", 0)),
                            "epoch": int(item.get("epoch", 0)),
                        }
                        for item in batch_chunks
                    ],
                )

    def delete_by_document_id(self, doc_id: str) -> None:
        """Delete all vectors belonging to a document ID."""
        with self._lock:
            collection = self._ensure_collection()
            if collection.count() == 0:
                return
            collection.delete(where={"document_id": {"$eq": doc_id}})

    def delete_by_chunk_ids(self, chunk_ids: list[str]) -> None:
        """Delete vectors matching the given chunk IDs."""
        if not chunk_ids:
            return
        with self._lock:
            collection = self._ensure_collection()
            if collection.count() == 0:
                return
            collection.delete(ids=list(chunk_ids))

    def reset_collection(self) -> None:
        """Drop and recreate the Chroma collection cleanly."""
        with self._lock:
            self._ensure_collection()
            try:
                self._client.delete_collection(name=self._collection_name)
            except Exception:
                pass
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )

    def count(self) -> int:
        """Return number of persisted vector records."""
        with self._lock:
            collection = self._ensure_collection()
            return int(collection.count())

    def query(
        self,
        query_embedding: list[float],
        top_k: int,
        filters: Optional[SearchFilters] = None,
    ) -> list[tuple[str, float]]:
        """Query top_k vector matches with optional metadata filtering."""
        if top_k <= 0 or not query_embedding:
            return []
        if filters is not None:
            if filters.document_ids is not None and len(filters.document_ids) == 0:
                return []
            if filters.file_types is not None and len(filters.file_types) == 0:
                return []

        with self._lock:
            collection = self._ensure_collection()
            total = int(collection.count())
            if total == 0:
                return []

            where = build_chroma_where(filters)
            if where is not None:
                matched = collection.get(where=where, include=[])
                matched_count = len(matched.get("ids") or [])
                if matched_count == 0:
                    return []
                n_results = min(top_k, matched_count)
            else:
                n_results = min(top_k, total)

            raw = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
                include=["distances"],
            )

            ids = (raw.get("ids") or [[]])[0]
            distances = (raw.get("distances") or [[]])[0]
            results: list[tuple[str, float]] = []
            for idx, cid in enumerate(ids):
                dist = float(distances[idx]) if idx < len(distances) else 1.0
                similarity = round(1.0 - dist, 6)
                if similarity <= 0.0:
                    continue
                results.append((str(cid), similarity))
            return results

    def get_embeddings(
        self,
        chunk_ids: list[str],
        batch_size: int = 200,
    ) -> dict[str, list[float]]:
        """Retrieve persisted embeddings for given chunk IDs without re-encoding."""
        if not chunk_ids:
            return {}
        unique_ids = list(dict.fromkeys(chunk_ids))
        safe_batch = max(1, int(batch_size))
        vectors: dict[str, list[float]] = {}

        with self._lock:
            collection = self._ensure_collection()
            if int(collection.count()) == 0:
                return {}

            for start in range(0, len(unique_ids), safe_batch):
                batch = unique_ids[start : start + safe_batch]
                raw = collection.get(ids=batch, include=["embeddings"])
                ids = raw.get("ids") or []
                embs = raw.get("embeddings")
                if embs is None:
                    continue
                for idx, cid in enumerate(ids):
                    if idx >= len(embs) or embs[idx] is None:
                        continue
                    emb_item = embs[idx]
                    row_vec = emb_item.tolist() if hasattr(emb_item, "tolist") else list(emb_item)
                    vectors[str(cid)] = [float(val) for val in row_vec]
        return vectors

    def query_neighbors(
        self,
        query_embeddings: list[list[float]],
        top_k: int,
        batch_size: int = 64,
    ) -> list[list[tuple[str, float]]]:
        """Batch HNSW neighbor lookup returning top_k (chunk_id, similarity) per query."""
        if not query_embeddings or top_k <= 0:
            return [[] for _ in query_embeddings]

        safe_batch = max(1, int(batch_size))
        results: list[list[tuple[str, float]]] = []

        with self._lock:
            collection = self._ensure_collection()
            total = int(collection.count())
            if total == 0:
                return [[] for _ in query_embeddings]

            n_results = min(int(top_k) + 1, total)
            for start in range(0, len(query_embeddings), safe_batch):
                batch_vecs = query_embeddings[start : start + safe_batch]
                raw = collection.query(
                    query_embeddings=batch_vecs,
                    n_results=n_results,
                    include=["distances"],
                )
                ids_batch = raw.get("ids") or []
                dist_batch = raw.get("distances") or []
                for q_idx in range(len(batch_vecs)):
                    q_ids = ids_batch[q_idx] if q_idx < len(ids_batch) else []
                    q_dists = dist_batch[q_idx] if q_idx < len(dist_batch) else []
                    neighbors: list[tuple[str, float]] = []
                    for n_idx, cid in enumerate(q_ids):
                        dist = float(q_dists[n_idx]) if n_idx < len(q_dists) else 1.0
                        sim = round(1.0 - dist, 6)
                        if sim <= 0.0:
                            continue
                        neighbors.append((str(cid), sim))
                    results.append(neighbors)
        return results

    def close(self) -> None:
        """Stop ChromaDB local system handles cleanly on shutdown."""
        with self._lock:
            client = self._client
            self._collection = None
            self._client = None
            if client is not None:
                system = getattr(client, "_system", None)
                stop_fn = getattr(system, "stop", None)
                if callable(stop_fn):
                    try:
                        stop_fn()
                    except Exception:
                        pass
