"""REST API endpoints for hybrid search and local retrieval index management."""

from __future__ import annotations

from typing import Any, Literal, Optional

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.models.embeddings import EmbeddingUnavailableError
from app.retrieval.service import RetrievalService
from app.retrieval.types import SearchFilters

router = APIRouter(tags=["retrieval"])


class SearchRequest(BaseModel):
    """Validated query payload for semantic, keyword, or hybrid search."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, max_length=4000)
    mode: Literal["hybrid", "semantic", "keyword"] = "hybrid"
    top_k: Optional[int] = Field(default=None, ge=1, le=100)
    semantic_weight: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    bm25_weight: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    rerank: Optional[bool] = None
    document_ids: Optional[list[str]] = None
    file_types: Optional[list[str]] = None
    epoch_min: Optional[int] = Field(default=None, ge=0)
    epoch_max: Optional[int] = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_constraints(self) -> SearchRequest:
        if not self.query.strip():
            raise ValueError("query must not be blank.")
        if (
            self.epoch_min is not None
            and self.epoch_max is not None
            and self.epoch_min > self.epoch_max
        ):
            raise ValueError("epoch_min must be less than or equal to epoch_max.")
        return self


class SearchHitResponse(BaseModel):
    """Unified retrieval result with source metadata and scoring breakdown."""

    model_config = ConfigDict(extra="forbid")

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
    metadata: dict[str, Any]


class SearchResponse(BaseModel):
    """Structured search response payload."""

    model_config = ConfigDict(extra="forbid")

    query: str
    mode: str
    total: int
    degraded_to_keyword: bool
    warnings: list[str]
    results: list[SearchHitResponse]


class RebuildIndexRequest(BaseModel):
    """Optional configuration for index rebuild requests."""

    model_config = ConfigDict(extra="forbid")

    batch_size: Optional[int] = Field(default=None, ge=1, le=512)


class RebuildIndexResponse(BaseModel):
    """Summary returned after rebuilding local BM25 and ChromaDB indexes."""

    model_config = ConfigDict(extra="forbid")

    status: str
    sqlite_indexed_chunks: int
    bm25_indexed_chunks: int
    vector_indexed_chunks: int
    vectors_written: int
    embedding_available: bool
    warnings: list[str]


class IndexStatusResponse(BaseModel):
    """Synchronization and local model status for retrieval indexes."""

    model_config = ConfigDict(extra="forbid")

    sqlite_indexed_documents: int
    sqlite_indexed_chunks: int
    bm25_indexed_chunks: int
    vector_indexed_chunks: int
    synchronized: bool
    embedding_available: bool
    embedding_model: str
    embedding_error: Optional[str] = None
    chroma_dir: str
    chroma_collection: str
    default_top_k: int
    semantic_weight: float
    bm25_weight: float
    rrf_k: int
    rerank_enabled: bool


def _get_retrieval_service(request: Request) -> RetrievalService:
    service = getattr(request.app.state, "retrieval_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Retrieval service is not initialized.",
        )
    return service


@router.post("/search", response_model=SearchResponse)
def search_vault(request: Request, payload: SearchRequest) -> SearchResponse:
    """Search indexed local chunks using semantic, keyword (BM25), or RRF hybrid retrieval."""
    service = _get_retrieval_service(request)
    filters = SearchFilters(
        document_ids=payload.document_ids,
        file_types=payload.file_types,
        epoch_min=payload.epoch_min,
        epoch_max=payload.epoch_max,
    )

    try:
        result = service.search(
            query=payload.query,
            mode=payload.mode,
            top_k=payload.top_k,
            semantic_weight=payload.semantic_weight,
            bm25_weight=payload.bm25_weight,
            rerank=payload.rerank,
            filters=filters if not filters.is_empty() else None,
        )
    except EmbeddingUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    return SearchResponse(
        query=result.query,
        mode=result.mode,
        total=result.total,
        degraded_to_keyword=result.degraded_to_keyword,
        warnings=result.warnings,
        results=[SearchHitResponse(**hit.to_dict()) for hit in result.hits],
    )


@router.post("/index/rebuild", response_model=RebuildIndexResponse)
def rebuild_retrieval_index(
    request: Request,
    payload: Optional[RebuildIndexRequest] = None,
) -> RebuildIndexResponse:
    """Rebuild local BM25 and ChromaDB indexes from SQLite indexed chunks."""
    service = _get_retrieval_service(request)
    summary = service.rebuild_index(batch_size=payload.batch_size if payload else None)
    return RebuildIndexResponse(**summary)


@router.get("/index/status", response_model=IndexStatusResponse)
def get_retrieval_index_status(request: Request) -> IndexStatusResponse:
    """Return index synchronization status and embedding availability."""
    service = _get_retrieval_service(request)
    return IndexStatusResponse(**service.get_index_status())
