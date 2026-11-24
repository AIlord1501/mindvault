"""REST API endpoints for memory retention tracking and spaced repetition reviews."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request, status
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.scheduler.models import (
    ReviewOutcome,
    parse_iso_datetime,
    resolve_outcome_and_rating,
)
from app.scheduler.service import ChunkNotFoundForReviewError, RetentionService

router = APIRouter(tags=["retention"])


class FlashcardResponse(BaseModel):
    """Self-recall flashcard derived from a chunk."""

    model_config = ConfigDict(extra="forbid")

    prompt: str
    hint: str
    answer_revealed: bool
    answer: Optional[str] = None
    key_beliefs: Optional[list[str]] = None


class ChunkRetentionItemResponse(BaseModel):
    """Retention telemetry and scheduling metadata for one chunk."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    document_id: str
    filename: str
    file_type: str
    chunk_index: int
    epoch: int
    stability: float
    retention: float
    elapsed_days: float
    last_reviewed_at: Optional[str] = None
    next_review_at: Optional[str] = None
    review_count: int
    eligible_for_review: bool


class RetentionSummaryResponse(BaseModel):
    """Aggregate retention metrics over matched chunks."""

    model_config = ConfigDict(extra="forbid")

    total_chunks: int
    due_count: int
    average_retention: float
    average_stability: float


class RetentionListResponse(BaseModel):
    """Response returned by GET /api/v1/retention."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_name: str
    retention_threshold: float
    evaluated_at: str
    summary: RetentionSummaryResponse
    total: int
    limit: int
    offset: int
    items: list[ChunkRetentionItemResponse]


class ReviewQueueItemResponse(ChunkRetentionItemResponse):
    """Queued chunk paired with its self-recall flashcard."""

    flashcard: FlashcardResponse


class ReviewQueueResponse(BaseModel):
    """Response returned by GET /api/v1/review/queue."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_name: str
    retention_threshold: float
    evaluated_at: str
    due_total: int
    returned_count: int
    include_answer: bool
    include_upcoming: bool
    items: list[ReviewQueueItemResponse]


class SubmitReviewRequest(BaseModel):
    """Request body for POST /api/v1/review/{chunk_id}."""

    model_config = ConfigDict(extra="forbid")

    outcome: Optional[ReviewOutcome] = None
    rating: Optional[int] = Field(default=None, ge=1, le=4)
    reveal_answer: bool = True
    reviewed_at: Optional[str] = None
    notes: Optional[str] = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def validate_outcome_and_timestamp(self) -> SubmitReviewRequest:
        resolve_outcome_and_rating(outcome=self.outcome, rating=self.rating)
        if self.reviewed_at is not None:
            parse_iso_datetime(self.reviewed_at)
        return self


class SubmitReviewResponse(BaseModel):
    """Response returned after recording a flashcard recall review."""

    model_config = ConfigDict(extra="forbid")

    review_id: str
    chunk_id: str
    document_id: str
    filename: str
    chunk_index: int
    epoch: int
    outcome: ReviewOutcome
    rating: int
    previous_stability: float
    new_stability: float
    retention_before: float
    retention_after: float
    elapsed_days: float
    next_interval_days: float
    review_count: int
    last_reviewed_at: str
    next_review_at: str
    notes: Optional[str] = None
    flashcard: FlashcardResponse


class ReviewHistoryItemResponse(BaseModel):
    """Single logged spaced repetition review event."""

    model_config = ConfigDict(extra="forbid")

    id: str
    chunk_id: str
    document_id: str
    filename: str
    chunk_index: int
    epoch: int
    outcome: ReviewOutcome
    rating: int
    previous_stability: float
    new_stability: float
    retention_before: float
    elapsed_days: float
    next_review_at: str
    reviewed_at: str
    notes: Optional[str] = None


class ReviewHistoryResponse(BaseModel):
    """Paginated response returned by GET /api/v1/review/history."""

    model_config = ConfigDict(extra="forbid")

    total: int
    limit: int
    offset: int
    items: list[ReviewHistoryItemResponse]


def _get_retention_service(request: Request) -> RetentionService:
    service = getattr(request.app.state, "retention_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Retention service is not initialized.",
        )
    return service


def _validate_epoch_and_as_of(
    epoch_min: Optional[int],
    epoch_max: Optional[int],
    as_of: Optional[str],
) -> None:
    if epoch_min is not None and epoch_max is not None and epoch_min > epoch_max:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="epoch_min must be less than or equal to epoch_max.",
        )
    if as_of is not None:
        try:
            parse_iso_datetime(as_of)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid ISO-8601 timestamp for 'as_of': {exc}",
            ) from exc


@router.get("/review/queue", response_model=ReviewQueueResponse)
def get_review_queue(
    request: Request,
    document_id: Optional[str] = Query(default=None),
    epoch_min: Optional[int] = Query(default=None, ge=0),
    epoch_max: Optional[int] = Query(default=None, ge=0),
    include_answer: bool = Query(
        default=False,
        description="Reveal full chunk answer and beliefs (hidden by default for self-recall).",
    ),
    include_upcoming: bool = Query(
        default=False,
        description="Include not-yet-due chunks after due items.",
    ),
    as_of: Optional[str] = Query(
        default=None,
        description="Optional ISO-8601 UTC timestamp for deterministic decay evaluation.",
    ),
    limit: int = Query(default=20, ge=1, le=200),
) -> ReviewQueueResponse:
    """Return prioritized flashcards eligible for spaced repetition review."""
    _validate_epoch_and_as_of(epoch_min, epoch_max, as_of)
    service = _get_retention_service(request)
    payload = service.get_review_queue(
        document_id=document_id,
        epoch_min=epoch_min,
        epoch_max=epoch_max,
        include_answer=include_answer,
        include_upcoming=include_upcoming,
        as_of=as_of,
        limit=limit,
    )
    return ReviewQueueResponse(**payload)


@router.get("/retention", response_model=RetentionListResponse)
def get_retention_scores(
    request: Request,
    document_id: Optional[str] = Query(default=None),
    epoch_min: Optional[int] = Query(default=None, ge=0),
    epoch_max: Optional[int] = Query(default=None, ge=0),
    due_only: bool = Query(default=False),
    as_of: Optional[str] = Query(
        default=None,
        description="Optional ISO-8601 UTC timestamp for deterministic decay evaluation.",
    ),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> RetentionListResponse:
    """Evaluate R = exp(-t / S) and return current memory retention stats across chunks."""
    _validate_epoch_and_as_of(epoch_min, epoch_max, as_of)
    service = _get_retention_service(request)
    payload = service.get_retention_list(
        document_id=document_id,
        epoch_min=epoch_min,
        epoch_max=epoch_max,
        due_only=due_only,
        as_of=as_of,
        limit=limit,
        offset=offset,
    )
    return RetentionListResponse(**payload)


@router.get("/review/history", response_model=ReviewHistoryResponse)
def get_review_history(
    request: Request,
    chunk_id: Optional[str] = Query(default=None),
    document_id: Optional[str] = Query(default=None),
    outcome: Optional[ReviewOutcome] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> ReviewHistoryResponse:
    """Return historical spaced repetition review events."""
    service = _get_retention_service(request)
    payload = service.get_review_history(
        chunk_id=chunk_id,
        document_id=document_id,
        outcome=outcome,
        limit=limit,
        offset=offset,
    )
    return ReviewHistoryResponse(**payload)


@router.post("/review/{chunk_id}", response_model=SubmitReviewResponse)
def submit_chunk_review(
    request: Request,
    chunk_id: str,
    payload: SubmitReviewRequest,
) -> SubmitReviewResponse:
    """Record a spaced repetition recall outcome, update stability, and reschedule."""
    service = _get_retention_service(request)
    try:
        result = service.record_chunk_review(
            chunk_id,
            outcome=payload.outcome,
            rating=payload.rating,
            reveal_answer=payload.reveal_answer,
            reviewed_at=payload.reviewed_at,
            notes=payload.notes,
        )
    except ChunkNotFoundForReviewError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    return SubmitReviewResponse(**result)
