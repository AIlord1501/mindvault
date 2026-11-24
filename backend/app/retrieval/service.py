"""Hybrid local retrieval service combining ChromaDB semantic search and BM25L."""

from __future__ import annotations

from typing import Any, Optional

from app.core.config import Settings
from app.db.repositories import DocumentRepository
from app.models.embeddings import BaseEmbeddingAdapter, EmbeddingUnavailableError
from app.retrieval.bm25_index import BM25Index, tokenize_for_bm25
from app.retrieval.types import SearchFilters, SearchHit, SearchResult
from app.retrieval.vector_store import ChromaVectorStore

VALID_SEARCH_MODES = frozenset({"hybrid", "semantic", "keyword"})


class RetrievalService:
    """Coordinates SQLite chunks, BM25L keyword index, ChromaDB vectors, and RRF fusion."""

    def __init__(
        self,
        *,
        settings: Settings,
        repo: DocumentRepository,
        vector_store: ChromaVectorStore,
        bm25_index: BM25Index,
        embedder: BaseEmbeddingAdapter,
    ) -> None:
        self._settings = settings
        self._repo = repo
        self._vector_store = vector_store
        self._bm25 = bm25_index
        self._embedder = embedder

    def sync_bm25_from_sqlite(self) -> int:
        """Load all indexed chunks from SQLite into the in-memory BM25 index."""
        chunks = self._repo.list_indexed_chunks()
        self._bm25.build(chunks)
        return len(chunks)

    def index_document_chunks(
        self,
        *,
        doc_id: str,
        filename: str,
        file_type: str,
        chunks: list[dict[str, Any]],
    ) -> list[str]:
        """Synchronize newly indexed document chunks into BM25 and ChromaDB."""
        enriched: list[dict[str, Any]] = []
        for item in chunks:
            copy = dict(item)
            copy["document_id"] = doc_id
            copy["filename"] = filename
            copy["file_type"] = file_type
            enriched.append(copy)

        self._bm25.remove_document(doc_id)
        self._bm25.upsert_chunks(enriched)
        self._vector_store.delete_by_document_id(doc_id)

        if not enriched:
            return []

        available, reason = self._embedder.is_available()
        if not available:
            return [f"Vector indexing skipped: {reason or 'embedding model unavailable.'}"]

        texts = [str(item["content"]) for item in enriched]
        embeddings = self._embedder.embed_documents(
            texts,
            batch_size=self._settings.embedding_batch_size,
        )
        self._vector_store.upsert_chunks(
            enriched,
            embeddings,
            batch_size=self._settings.embedding_batch_size,
        )
        return []

    def remove_document(self, doc_id: str, chunk_ids: Optional[list[str]] = None) -> None:
        """Remove a document's chunks from BM25 and ChromaDB."""
        self._bm25.remove_document(doc_id)
        if chunk_ids:
            self._bm25.remove_chunks(chunk_ids)
            self._vector_store.delete_by_chunk_ids(chunk_ids)
        self._vector_store.delete_by_document_id(doc_id)

    def rebuild_index(self, batch_size: Optional[int] = None) -> dict[str, Any]:
        """Rebuild both BM25 and ChromaDB indexes from SQLite source of truth."""
        effective_batch = (
            max(1, int(batch_size))
            if batch_size is not None
            else self._settings.embedding_batch_size
        )
        chunks = self._repo.list_indexed_chunks()
        self._bm25.build(chunks)
        self._vector_store.reset_collection()

        warnings: list[str] = []
        available, reason = self._embedder.is_available()
        vector_indexed = 0

        if not available:
            if chunks:
                warnings.append(
                    f"Vector rebuild skipped: {reason or 'embedding model unavailable.'}"
                )
        elif chunks:
            texts = [str(item["content"]) for item in chunks]
            embeddings = self._embedder.embed_documents(texts, batch_size=effective_batch)
            self._vector_store.upsert_chunks(
                chunks,
                embeddings,
                batch_size=effective_batch,
            )
            vector_indexed = len(chunks)

        return {
            "status": "rebuilt" if available or not chunks else "partial",
            "sqlite_indexed_chunks": len(chunks),
            "bm25_indexed_chunks": self._bm25.count(),
            "vector_indexed_chunks": self._vector_store.count(),
            "vectors_written": vector_indexed,
            "embedding_available": available,
            "warnings": warnings,
        }

    def get_index_status(self) -> dict[str, Any]:
        """Return synchronization and embedding availability status."""
        doc_count, chunk_count = self._repo.count_indexed_documents_and_chunks()
        bm25_count = self._bm25.count()
        vector_count = self._vector_store.count()
        available, reason = self._embedder.is_available()

        synchronized = bm25_count == chunk_count and (
            vector_count == chunk_count if available else True
        )

        return {
            "sqlite_indexed_documents": doc_count,
            "sqlite_indexed_chunks": chunk_count,
            "bm25_indexed_chunks": bm25_count,
            "vector_indexed_chunks": vector_count,
            "synchronized": synchronized,
            "embedding_available": available,
            "embedding_model": self._embedder.model_name,
            "embedding_error": reason,
            "chroma_dir": str(self._settings.chroma_dir),
            "chroma_collection": self._settings.chroma_collection,
            "default_top_k": self._settings.retrieval_top_k,
            "semantic_weight": self._settings.semantic_weight,
            "bm25_weight": self._settings.bm25_weight,
            "rrf_k": self._settings.rrf_k,
            "rerank_enabled": self._settings.rerank_enabled,
        }

    def _normalize_weights(
        self,
        semantic_weight: Optional[float],
        bm25_weight: Optional[float],
    ) -> tuple[float, float]:
        w_sem = (
            self._settings.semantic_weight
            if semantic_weight is None
            else float(semantic_weight)
        )
        w_kw = self._settings.bm25_weight if bm25_weight is None else float(bm25_weight)
        if w_sem < 0.0 or w_kw < 0.0:
            raise ValueError("semantic_weight and bm25_weight must be non-negative.")
        total = w_sem + w_kw
        if total <= 0.0:
            raise ValueError("At least one of semantic_weight or bm25_weight must be positive.")
        return w_sem / total, w_kw / total

    @staticmethod
    def _rerank_boost(query_tokens: list[str], content: str, rrf_k: int) -> float:
        if not query_tokens:
            return 0.0
        content_tokens = tokenize_for_bm25(content)
        if not content_tokens:
            return 0.0
        unique_query = frozenset(query_tokens)
        unique_content = frozenset(content_tokens)
        coverage = len(unique_query & unique_content) / len(unique_query)

        phrase_bonus = 0.0
        if len(query_tokens) > 1:
            query_phrase = " ".join(query_tokens)
            content_phrase = " ".join(content_tokens)
            if query_phrase in content_phrase:
                phrase_bonus = 0.25

        scale = 1.0 / (rrf_k + 1)
        return (0.15 * coverage + phrase_bonus) * scale

    def search(
        self,
        *,
        query: str,
        mode: str = "hybrid",
        top_k: Optional[int] = None,
        semantic_weight: Optional[float] = None,
        bm25_weight: Optional[float] = None,
        rerank: Optional[bool] = None,
        filters: Optional[SearchFilters] = None,
    ) -> SearchResult:
        """Execute semantic, keyword, or weighted RRF hybrid retrieval."""
        cleaned_query = (query or "").strip()
        if not cleaned_query:
            raise ValueError("Search query must not be empty.")
        if mode not in VALID_SEARCH_MODES:
            raise ValueError(f"Unsupported search mode '{mode}'.")

        resolved_top_k = self._settings.retrieval_top_k if top_k is None else int(top_k)
        if resolved_top_k <= 0:
            raise ValueError("top_k must be positive.")
        candidate_k = max(resolved_top_k, self._settings.retrieval_candidate_k)
        rrf_k = self._settings.rrf_k
        rerank_active = self._settings.rerank_enabled if rerank is None else bool(rerank)
        w_sem, w_kw = self._normalize_weights(semantic_weight, bm25_weight)

        warnings: list[str] = []
        degraded_to_keyword = False

        semantic_hits: list[tuple[str, float]] = []
        keyword_hits: list[tuple[str, float]] = []

        if mode in {"semantic", "hybrid"}:
            available, reason = self._embedder.is_available()
            if not available:
                msg = reason or "Local embedding model is unavailable."
                if mode == "semantic":
                    raise EmbeddingUnavailableError(msg)
                degraded_to_keyword = True
                warnings.append(f"Semantic retrieval unavailable; degraded to BM25 keyword: {msg}")
            else:
                query_vec = self._embedder.embed_query(cleaned_query)
                semantic_hits = self._vector_store.query(
                    query_vec,
                    top_k=candidate_k,
                    filters=filters,
                )

        if mode in {"keyword", "hybrid"}:
            keyword_hits = self._bm25.search(
                cleaned_query,
                top_k=candidate_k,
                filters=filters,
            )

        semantic_map: dict[str, tuple[int, float]] = {
            cid: (rank + 1, score) for rank, (cid, score) in enumerate(semantic_hits)
        }
        keyword_map: dict[str, tuple[int, float]] = {
            cid: (rank + 1, score) for rank, (cid, score) in enumerate(keyword_hits)
        }

        candidate_ids = list(dict.fromkeys(list(semantic_map.keys()) + list(keyword_map.keys())))
        if not candidate_ids:
            return SearchResult(
                query=cleaned_query,
                mode=mode,
                total=0,
                degraded_to_keyword=degraded_to_keyword,
                warnings=warnings,
                hits=[],
            )

        hydrated = self._repo.get_chunks_by_ids(candidate_ids)
        query_tokens = tokenize_for_bm25(cleaned_query)

        if mode == "semantic":
            eff_sem, eff_kw = 1.0, 0.0
        elif mode == "keyword" or degraded_to_keyword:
            eff_sem, eff_kw = 0.0, 1.0
        else:
            eff_sem, eff_kw = w_sem, w_kw

        ranked_hits: list[SearchHit] = []
        for cid in candidate_ids:
            chunk = hydrated.get(cid)
            if chunk is None:
                continue
            if filters is not None and not filters.matches(chunk):
                continue

            sem_rank, sem_score = semantic_map.get(cid, (None, None))
            kw_rank, kw_score = keyword_map.get(cid, (None, None))

            rrf_score = 0.0
            if sem_rank is not None:
                rrf_score += eff_sem / (rrf_k + sem_rank)
            if kw_rank is not None:
                rrf_score += eff_kw / (rrf_k + kw_rank)
            rrf_score = round(rrf_score, 6)

            boost = (
                self._rerank_boost(query_tokens, str(chunk["content"]), rrf_k)
                if rerank_active
                else 0.0
            )
            final_score = round(rrf_score + boost, 6)

            ranked_hits.append(
                SearchHit(
                    chunk_id=cid,
                    document_id=str(chunk["document_id"]),
                    filename=str(chunk["filename"]),
                    file_type=str(chunk["file_type"]),
                    source_path=str(chunk["source_path"]),
                    chunk_index=int(chunk["chunk_index"]),
                    epoch=int(chunk["epoch"]),
                    content=str(chunk["content"]),
                    token_count=int(chunk["token_count"]),
                    score=final_score,
                    rrf_score=rrf_score,
                    semantic_rank=sem_rank,
                    keyword_rank=kw_rank,
                    semantic_score=sem_score,
                    keyword_score=kw_score,
                    metadata=dict(chunk["metadata"]),
                )
            )

        ranked_hits.sort(key=lambda hit: (-hit.score, -hit.rrf_score, hit.chunk_id))
        top_hits = ranked_hits[:resolved_top_k]

        return SearchResult(
            query=cleaned_query,
            mode=mode,
            total=len(top_hits),
            degraded_to_keyword=degraded_to_keyword,
            warnings=warnings,
            hits=top_hits,
        )
