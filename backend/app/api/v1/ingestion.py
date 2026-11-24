"""REST API endpoints for local document ingestion and document management."""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from app.ingestion.pipeline import IngestionPipeline, IngestionResult

router = APIRouter(tags=["ingestion"])


class IngestionSummaryResponse(BaseModel):
    """Structured summary for a single file ingestion attempt."""

    model_config = ConfigDict(extra="forbid")

    document_id: Optional[str] = None
    filename: str
    file_type: str
    status: str
    duplicate: bool = False
    chunk_count: int = 0
    content_hash: Optional[str] = None
    epoch: Optional[int] = None
    source_path: Optional[str] = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    warnings: list[str] = []
    progress: list[str] = []


class BatchIngestionResponse(BaseModel):
    """Aggregated summary for a multi-file ingestion batch."""

    model_config = ConfigDict(extra="forbid")

    total: int
    indexed: int
    duplicates: int
    failed: int
    results: list[IngestionSummaryResponse]


class ChunkResponse(BaseModel):
    """Persisted chunk representation."""

    model_config = ConfigDict(extra="forbid")

    id: str
    document_id: str
    chunk_index: int
    content: str
    token_count: int
    epoch: int
    created_at: str
    metadata: dict[str, Any]


class DocumentSummaryResponse(BaseModel):
    """Document metadata record with chunk count."""

    model_config = ConfigDict(extra="forbid")

    id: str
    filename: str
    file_type: str
    source_path: str
    content_hash: str
    created_at: str
    updated_at: str
    status: str
    chunk_count: int
    metadata: dict[str, Any]


class DocumentListResponse(BaseModel):
    """Paginated list of documents."""

    model_config = ConfigDict(extra="forbid")

    total: int
    limit: int
    offset: int
    documents: list[DocumentSummaryResponse]


class DocumentDetailResponse(DocumentSummaryResponse):
    """Full document record including its ordered chunks."""

    chunks: list[ChunkResponse]


class DeleteDocumentResponse(BaseModel):
    """Confirmation payload after deleting a document."""

    model_config = ConfigDict(extra="forbid")

    deleted: bool
    document_id: str


def _get_pipeline(request: Request) -> IngestionPipeline:
    pipeline = getattr(request.app.state, "ingestion_pipeline", None)
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Ingestion pipeline is not initialized.",
        )
    return pipeline


def _http_status_for_result(result: IngestionResult) -> int:
    if result.status == "indexed":
        return status.HTTP_201_CREATED
    if result.status == "duplicate":
        return status.HTTP_200_OK
    code = result.error_code
    if code == "unsupported_file_type":
        return status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
    if code == "file_size_limit":
        return status.HTTP_400_BAD_REQUEST
    if code == "adapter_unavailable":
        return status.HTTP_503_SERVICE_UNAVAILABLE
    return status.HTTP_422_UNPROCESSABLE_ENTITY


@router.post("/ingest/file", response_model=IngestionSummaryResponse)
async def ingest_file(
    request: Request,
    file: UploadFile = File(...),
    epoch: Optional[int] = Form(default=None, ge=0),
    chunk_size: Optional[int] = Form(default=None, gt=0, le=50000),
    chunk_overlap: Optional[int] = Form(default=None, ge=0, le=25000),
) -> JSONResponse:
    """Ingest a single local document into MindVault."""
    pipeline = _get_pipeline(request)
    raw_bytes = await file.read()
    result = pipeline.ingest_bytes(
        filename=file.filename or "untitled",
        content=raw_bytes,
        epoch=epoch,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    payload = IngestionSummaryResponse(**result.to_dict())
    return JSONResponse(
        status_code=_http_status_for_result(result),
        content=payload.model_dump(),
    )


@router.post("/ingest/batch", response_model=BatchIngestionResponse)
async def ingest_batch(
    request: Request,
    files: list[UploadFile] = File(...),
    epoch: Optional[int] = Form(default=None, ge=0),
    chunk_size: Optional[int] = Form(default=None, gt=0, le=50000),
    chunk_overlap: Optional[int] = Form(default=None, ge=0, le=25000),
) -> BatchIngestionResponse:
    """Ingest multiple local documents and report per-file progress and errors."""
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one file must be provided for batch ingestion.",
        )

    pipeline = _get_pipeline(request)
    results: list[IngestionSummaryResponse] = []
    indexed = 0
    duplicates = 0
    failed = 0

    for upload in files:
        raw_bytes = await upload.read()
        res = pipeline.ingest_bytes(
            filename=upload.filename or "untitled",
            content=raw_bytes,
            epoch=epoch,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        if res.status == "indexed":
            indexed += 1
        elif res.status == "duplicate":
            duplicates += 1
        else:
            failed += 1
        results.append(IngestionSummaryResponse(**res.to_dict()))

    return BatchIngestionResponse(
        total=len(results),
        indexed=indexed,
        duplicates=duplicates,
        failed=failed,
        results=results,
    )


@router.get("/documents", response_model=DocumentListResponse)
def list_documents(
    request: Request,
    status_filter: Optional[str] = Query(
        default=None,
        alias="status",
        pattern="^(pending|processing|indexed|failed|deleted)$",
    ),
    file_type: Optional[str] = Query(default=None, min_length=1, max_length=32),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> DocumentListResponse:
    """List ingested documents with optional filtering and pagination."""
    repo = request.app.state.document_repo
    docs, total = repo.list_documents(
        status=status_filter,
        file_type=file_type,
        limit=limit,
        offset=offset,
    )
    return DocumentListResponse(
        total=total,
        limit=limit,
        offset=offset,
        documents=[DocumentSummaryResponse(**doc) for doc in docs],
    )


@router.get("/documents/{doc_id}", response_model=DocumentDetailResponse)
def get_document(request: Request, doc_id: str) -> DocumentDetailResponse:
    """Return a single document and its ordered chunks."""
    repo = request.app.state.document_repo
    doc = repo.get_document_with_chunks(doc_id)
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_id}' not found.",
        )
    return DocumentDetailResponse(**doc)


@router.delete("/documents/{doc_id}", response_model=DeleteDocumentResponse)
def delete_document(request: Request, doc_id: str) -> DeleteDocumentResponse:
    """Delete a document, cascade-remove its chunks, and clean up its managed vault copy."""
    pipeline = _get_pipeline(request)
    deleted = pipeline.delete_document_and_cleanup(doc_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_id}' not found.",
        )
    return DeleteDocumentResponse(deleted=True, document_id=doc_id)
