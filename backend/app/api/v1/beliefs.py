"""REST API endpoints for belief extraction, evolution tracking, and contradiction review."""

from __future__ import annotations

from typing import Literal, Optional

from fastapi import APIRouter, Body, HTTPException, Query, Request, status
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.beliefs.service import BeliefService, ChunkNotFoundError
from app.models.llm import LLMGenerationError, LLMUnavailableError

router = APIRouter(tags=["beliefs"])


class ExtractBeliefsRequest(BaseModel):
    """Optional configuration overrides when extracting beliefs from a chunk."""

    model_config = ConfigDict(extra="forbid")

    max_beliefs: int = Field(default=5, ge=1, le=5)
    dedup_threshold: Optional[float] = Field(default=None, gt=0.0, le=1.0)
    min_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    detect_contradictions: bool = True
    candidate_top_k: Optional[int] = Field(default=None, ge=1, le=100)
    candidate_min_similarity: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    contradiction_threshold: Optional[float] = Field(default=None, gt=0.0, le=1.0)
    entailment_threshold: Optional[float] = Field(default=None, gt=0.0, le=1.0)


class BeliefSummaryResponse(BaseModel):
    """Belief record with source chunk, confidence, epoch, and lineage references."""

    model_config = ConfigDict(extra="forbid")

    id: str
    chunk_id: str
    document_id: Optional[str] = None
    filename: Optional[str] = None
    belief_text: str
    normalized_text: str
    confidence: float
    epoch: int
    status: str
    superseded_by_id: Optional[str] = None
    context_note: Optional[str] = None
    supersedes_ids: list[str] = []
    created_at: str
    updated_at: Optional[str] = None


class BeliefRelationResponse(BaseModel):
    """Stored NLI relationship scores between two beliefs."""

    model_config = ConfigDict(extra="forbid")

    id: str
    source_belief_id: str
    target_belief_id: str
    label: str
    contradiction_score: float
    entailment_score: float
    neutral_score: float
    similarity_score: float
    created_at: str


class ContradictionBaseResponse(BaseModel):
    """Core contradiction record."""

    model_config = ConfigDict(extra="forbid")

    id: str
    belief_a_id: str
    belief_b_id: str
    score: float
    entailment_score: float
    neutral_score: float
    similarity_score: float
    status: str
    resolution: Optional[str] = None
    resolution_note: Optional[str] = None
    created_at: str
    resolved_at: Optional[str] = None


class ContradictionResponse(ContradictionBaseResponse):
    """Contradiction record enriched with both belief summaries."""

    belief_a: Optional[BeliefSummaryResponse] = None
    belief_b: Optional[BeliefSummaryResponse] = None


class ExtractBeliefsResponse(BaseModel):
    """Summary returned after extracting beliefs and checking contradictions."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    document_id: str
    epoch: int
    extracted_count: int
    duplicates_skipped: int
    low_confidence_skipped: int
    candidates_evaluated: int
    nli_mode: str
    nli_model: str
    beliefs: list[BeliefSummaryResponse]
    relations: list[BeliefRelationResponse]
    contradictions: list[ContradictionResponse]
    warnings: list[str]


class BeliefListResponse(BaseModel):
    """Paginated list of beliefs."""

    model_config = ConfigDict(extra="forbid")

    total: int
    limit: int
    offset: int
    beliefs: list[BeliefSummaryResponse]


class BeliefDetailResponse(BeliefSummaryResponse):
    """Detailed view of a single belief including lineage history and NLI relations."""

    source_path: str
    chunk_content: str
    lineage_history: list[BeliefSummaryResponse]
    relations: list[BeliefRelationResponse]
    contradictions: list[ContradictionBaseResponse]


class ContradictionListResponse(BaseModel):
    """Paginated list of contradictions."""

    model_config = ConfigDict(extra="forbid")

    total: int
    limit: int
    offset: int
    contradictions: list[ContradictionResponse]


class ResolveContradictionRequest(BaseModel):
    """Manual review action for resolving a detected contradiction."""

    model_config = ConfigDict(extra="forbid")

    resolution: Literal[
        "accept_newer",
        "accept_older",
        "keep_both",
        "context_dependent",
        "false_positive",
    ]
    note: Optional[str] = Field(default=None, max_length=1000)


class DismissContradictionRequest(BaseModel):
    """Optional note when dismissing a contradiction as a false positive."""

    model_config = ConfigDict(extra="forbid")

    note: Optional[str] = Field(default=None, max_length=1000)


class EpochValidationQuery(BaseModel):
    """Helper to validate epoch_min <= epoch_max on GET query params."""

    epoch_min: Optional[int]
    epoch_max: Optional[int]

    @model_validator(mode="after")
    def validate_bounds(self) -> EpochValidationQuery:
        if (
            self.epoch_min is not None
            and self.epoch_max is not None
            and self.epoch_min > self.epoch_max
        ):
            raise ValueError("epoch_min must be less than or equal to epoch_max.")
        return self


def _get_belief_service(request: Request) -> BeliefService:
    service = getattr(request.app.state, "belief_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Belief service is not initialized.",
        )
    return service


@router.post(
    "/beliefs/extract/{chunk_id}",
    response_model=ExtractBeliefsResponse,
    status_code=status.HTTP_201_CREATED,
)
def extract_beliefs(
    request: Request,
    chunk_id: str,
    payload: Optional[ExtractBeliefsRequest] = Body(default=None),
) -> ExtractBeliefsResponse:
    """Extract 1-5 beliefs from a chunk, deduplicate, and detect contradictions."""
    service = _get_belief_service(request)
    opts = payload or ExtractBeliefsRequest()

    try:
        summary = service.extract_from_chunk(
            chunk_id,
            max_beliefs=opts.max_beliefs,
            dedup_threshold=opts.dedup_threshold,
            min_confidence=opts.min_confidence,
            detect_contradictions=opts.detect_contradictions,
            candidate_top_k=opts.candidate_top_k,
            candidate_min_similarity=opts.candidate_min_similarity,
            contradiction_threshold=opts.contradiction_threshold,
            entailment_threshold=opts.entailment_threshold,
        )
    except ChunkNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except LLMUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except LLMGenerationError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc

    return ExtractBeliefsResponse(**summary)


@router.get("/beliefs", response_model=BeliefListResponse)
def list_beliefs(
    request: Request,
    status_filter: Optional[str] = Query(
        default=None,
        alias="status",
        pattern="^(active|superseded|contradicted|archived)$",
    ),
    document_id: Optional[str] = Query(default=None, min_length=1),
    chunk_id: Optional[str] = Query(default=None, min_length=1),
    epoch_min: Optional[int] = Query(default=None, ge=0),
    epoch_max: Optional[int] = Query(default=None, ge=0),
    min_confidence: Optional[float] = Query(default=None, ge=0.0, le=1.0),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> BeliefListResponse:
    """List extracted beliefs with status, epoch, and confidence filters."""
    try:
        EpochValidationQuery(epoch_min=epoch_min, epoch_max=epoch_max)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    service = _get_belief_service(request)
    items, total = service.list_beliefs(
        status=status_filter,
        document_id=document_id,
        chunk_id=chunk_id,
        epoch_min=epoch_min,
        epoch_max=epoch_max,
        min_confidence=min_confidence,
        limit=limit,
        offset=offset,
    )
    return BeliefListResponse(
        total=total,
        limit=limit,
        offset=offset,
        beliefs=[BeliefSummaryResponse(**item) for item in items],
    )


@router.get("/beliefs/{belief_id}", response_model=BeliefDetailResponse)
def get_belief(request: Request, belief_id: str) -> BeliefDetailResponse:
    """Fetch a single belief with epoch lineage history, relations, and contradictions."""
    service = _get_belief_service(request)
    detail = service.get_belief(belief_id)
    if detail is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Belief '{belief_id}' not found.",
        )
    return BeliefDetailResponse(**detail)


@router.get("/contradictions", response_model=ContradictionListResponse)
def list_contradictions(
    request: Request,
    status_filter: Optional[str] = Query(
        default=None,
        alias="status",
        pattern="^(open|resolved|dismissed)$",
    ),
    min_score: Optional[float] = Query(default=None, ge=0.0, le=1.0),
    epoch_min: Optional[int] = Query(default=None, ge=0),
    epoch_max: Optional[int] = Query(default=None, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> ContradictionListResponse:
    """List detected contradictions and their manual review statuses."""
    try:
        EpochValidationQuery(epoch_min=epoch_min, epoch_max=epoch_max)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    service = _get_belief_service(request)
    items, total = service.list_contradictions(
        status=status_filter,
        min_score=min_score,
        epoch_min=epoch_min,
        epoch_max=epoch_max,
        limit=limit,
        offset=offset,
    )
    return ContradictionListResponse(
        total=total,
        limit=limit,
        offset=offset,
        contradictions=[ContradictionResponse(**item) for item in items],
    )


@router.post(
    "/contradictions/{contradiction_id}/resolve",
    response_model=ContradictionResponse,
)
def resolve_contradiction(
    request: Request,
    contradiction_id: str,
    payload: ResolveContradictionRequest,
) -> ContradictionResponse:
    """Resolve a contradiction while retaining full epoch history and traceability."""
    service = _get_belief_service(request)
    updated = service.resolve_contradiction(
        contradiction_id,
        resolution=payload.resolution,
        note=payload.note,
    )
    if updated is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Contradiction '{contradiction_id}' not found.",
        )
    return ContradictionResponse(**updated)


@router.post(
    "/contradictions/{contradiction_id}/dismiss",
    response_model=ContradictionResponse,
)
def dismiss_contradiction(
    request: Request,
    contradiction_id: str,
    payload: Optional[DismissContradictionRequest] = Body(default=None),
) -> ContradictionResponse:
    """Dismiss a contradiction as a false positive without deleting any beliefs."""
    service = _get_belief_service(request)
    note = payload.note if payload is not None else None
    updated = service.dismiss_contradiction(contradiction_id, note=note)
    if updated is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Contradiction '{contradiction_id}' not found.",
        )
    return ContradictionResponse(**updated)
