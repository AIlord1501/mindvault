"""REST API endpoints for MindVault dashboard analytics and Morning Briefs."""

from __future__ import annotations

from datetime import date
from typing import Any, Literal, Optional

from fastapi import APIRouter, Body, HTTPException, Query, Request, status
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.scheduler.insights import BRIEF_CATEGORIES, VaultInsightsService
from app.scheduler.models import parse_iso_datetime

router = APIRouter(tags=["dashboard"])

BriefCategory = Literal["decaying_notes", "contradictions", "dormant_epochs", "major_clusters"]


class BeliefDriftSummaryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_beliefs: int
    total_epochs: int
    active_total: int
    superseded_total: int
    uncertain_total: int
    open_contradictions: int
    transition_count: int
    overall_drift_rate: float


class BeliefDriftEpochResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    epoch: int
    total_count: int
    active_count: int
    superseded_count: int
    uncertain_count: int
    archived_count: int
    contradiction_count: int
    open_contradiction_count: int
    average_confidence: float
    drift_score: float
    beliefs: list[dict[str, Any]]


class BeliefTransitionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    from_belief_id: str
    to_belief_id: str
    from_epoch: int
    to_epoch: int
    from_text: str
    to_text: str


class BeliefDriftResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: BeliefDriftSummaryResponse
    epochs: list[BeliefDriftEpochResponse]
    transitions: list[BeliefTransitionResponse]


class ConceptVelocitySummaryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_distinct_concepts: int
    total_epochs: int
    peak_epoch: Optional[int] = None
    average_velocity: float


class ConceptVelocityResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: ConceptVelocitySummaryResponse
    epochs: list[dict[str, Any]]
    concepts: list[dict[str, Any]]


class DecayHeatmapSummaryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_chunks: int
    due_count: int
    average_retention: float
    average_stability: float
    critical_count: int
    warning_count: int
    healthy_count: int
    retention_threshold: float


class DecayHeatmapResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_name: str
    evaluated_at: str
    retention_threshold: float
    summary: DecayHeatmapSummaryResponse
    cells: list[dict[str, Any]]
    epoch_buckets: list[dict[str, Any]]
    document_buckets: list[dict[str, Any]]


class HealthAxisResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    label: str
    score: float
    value_pct: float
    detail: str
    target_view: str


class KnowledgeHealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    overall_score: float
    status_label: str
    total_chunks: int
    axes: list[HealthAxisResponse]


class BriefCitationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    document_id: str
    filename: str
    chunk_index: int
    epoch: int
    snippet: str


class BriefInsightResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    category: BriefCategory
    title: str
    summary: str
    priority_score: float
    recommended_action: str
    citations: list[BriefCitationResponse]


class MorningBriefResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    brief_date: str
    content: str
    categories: list[BriefCategory]
    max_insights: int
    insights: list[BriefInsightResponse]
    citations: list[BriefCitationResponse]
    regenerated: bool
    created_at: str


class MorningBriefListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total: int
    limit: int
    offset: int
    items: list[MorningBriefResponse]


class GenerateBriefRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    brief_date: Optional[str] = None
    categories: Optional[list[BriefCategory]] = Field(default=None, min_length=1, max_length=4)
    max_insights: int = Field(default=3, ge=1, le=10)
    regenerate: bool = False
    epoch_min: Optional[int] = Field(default=None, ge=0)
    epoch_max: Optional[int] = Field(default=None, ge=0)
    as_of: Optional[str] = None

    @model_validator(mode="after")
    def validate_fields(self) -> GenerateBriefRequest:
        if (
            self.epoch_min is not None
            and self.epoch_max is not None
            and self.epoch_min > self.epoch_max
        ):
            raise ValueError("epoch_min must be less than or equal to epoch_max.")
        if self.brief_date is not None:
            date.fromisoformat(self.brief_date.strip())
        if self.as_of is not None:
            parse_iso_datetime(self.as_of)
        if self.categories is not None:
            for cat in self.categories:
                if cat not in BRIEF_CATEGORIES:
                    raise ValueError(f"Unsupported category: {cat}")
        return self


class DashboardKPIsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_documents: int
    total_chunks: int
    due_flashcards: int
    open_contradictions: int
    active_beliefs: int
    average_retention: float
    health_score: float
    epoch_min: Optional[int] = None
    epoch_max: Optional[int] = None


class DashboardOverviewResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    kpis: DashboardKPIsResponse
    belief_drift: BeliefDriftResponse
    concept_velocity: ConceptVelocityResponse
    decay: DecayHeatmapResponse
    health: KnowledgeHealthResponse
    review_queue: dict[str, Any]
    contradictions: dict[str, Any]
    latest_brief: Optional[MorningBriefResponse] = None


def _get_insights_service(request: Request) -> VaultInsightsService:
    service = getattr(request.app.state, "insights_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vault insights service is not initialized.",
        )
    return service


def _validate_bounds(
    epoch_min: Optional[int],
    epoch_max: Optional[int],
    as_of: Optional[str] = None,
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


@router.get("/dashboard/overview", response_model=DashboardOverviewResponse)
def get_dashboard_overview(
    request: Request,
    epoch_min: Optional[int] = Query(default=None, ge=0),
    epoch_max: Optional[int] = Query(default=None, ge=0),
    as_of: Optional[str] = Query(default=None),
) -> DashboardOverviewResponse:
    """Return aggregated metrics and payloads across all 6 dashboard views."""
    _validate_bounds(epoch_min, epoch_max, as_of)
    service = _get_insights_service(request)
    payload = service.get_dashboard_overview(
        epoch_min=epoch_min,
        epoch_max=epoch_max,
        as_of=as_of,
    )
    return DashboardOverviewResponse(**payload)


@router.get("/dashboard/belief-drift", response_model=BeliefDriftResponse)
def get_belief_drift(
    request: Request,
    epoch_min: Optional[int] = Query(default=None, ge=0),
    epoch_max: Optional[int] = Query(default=None, ge=0),
    status_filter: Optional[str] = Query(default=None, alias="status"),
    document_id: Optional[str] = Query(default=None),
) -> BeliefDriftResponse:
    """Return Belief Drift Timeline series and supersession transitions across epochs."""
    _validate_bounds(epoch_min, epoch_max)
    service = _get_insights_service(request)
    payload = service.get_belief_drift(
        epoch_min=epoch_min,
        epoch_max=epoch_max,
        status=status_filter,
        document_id=document_id,
    )
    return BeliefDriftResponse(**payload)


@router.get("/dashboard/concept-velocity", response_model=ConceptVelocityResponse)
def get_concept_velocity(
    request: Request,
    epoch_min: Optional[int] = Query(default=None, ge=0),
    epoch_max: Optional[int] = Query(default=None, ge=0),
    top_concepts: int = Query(default=8, ge=1, le=30),
) -> ConceptVelocityResponse:
    """Return Concept Velocity metrics across epochs."""
    _validate_bounds(epoch_min, epoch_max)
    service = _get_insights_service(request)
    payload = service.get_concept_velocity(
        epoch_min=epoch_min,
        epoch_max=epoch_max,
        top_concepts=top_concepts,
    )
    return ConceptVelocityResponse(**payload)


@router.get("/dashboard/decay", response_model=DecayHeatmapResponse)
def get_decay_heatmap(
    request: Request,
    epoch_min: Optional[int] = Query(default=None, ge=0),
    epoch_max: Optional[int] = Query(default=None, ge=0),
    document_id: Optional[str] = Query(default=None),
    as_of: Optional[str] = Query(default=None),
) -> DecayHeatmapResponse:
    """Return Knowledge Decay Heatmap cells and bucket summaries."""
    _validate_bounds(epoch_min, epoch_max, as_of)
    service = _get_insights_service(request)
    payload = service.get_decay_heatmap(
        epoch_min=epoch_min,
        epoch_max=epoch_max,
        document_id=document_id,
        as_of=as_of,
    )
    return DecayHeatmapResponse(**payload)


@router.get("/dashboard/health", response_model=KnowledgeHealthResponse)
def get_knowledge_health(
    request: Request,
    epoch_min: Optional[int] = Query(default=None, ge=0),
    epoch_max: Optional[int] = Query(default=None, ge=0),
    as_of: Optional[str] = Query(default=None),
) -> KnowledgeHealthResponse:
    """Return Knowledge Health Radar scores across 5 normalized axes."""
    _validate_bounds(epoch_min, epoch_max, as_of)
    service = _get_insights_service(request)
    payload = service.get_knowledge_health(
        epoch_min=epoch_min,
        epoch_max=epoch_max,
        as_of=as_of,
    )
    return KnowledgeHealthResponse(**payload)


@router.get("/briefs", response_model=MorningBriefListResponse)
def list_morning_briefs(
    request: Request,
    limit: int = Query(default=30, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> MorningBriefListResponse:
    """List persisted Morning Briefs ordered by brief_date descending."""
    service = _get_insights_service(request)
    payload = service.list_briefs(limit=limit, offset=offset)
    return MorningBriefListResponse(**payload)


@router.post(
    "/briefs/generate",
    response_model=MorningBriefResponse,
    status_code=status.HTTP_201_CREATED,
)
def generate_morning_brief(
    request: Request,
    payload: Optional[GenerateBriefRequest] = Body(default=None),
) -> MorningBriefResponse:
    """Generate (or regenerate) a persisted local Morning Brief with citations."""
    service = _get_insights_service(request)
    opts = payload or GenerateBriefRequest()
    try:
        brief = service.generate_morning_brief(
            brief_date=opts.brief_date,
            categories=list(opts.categories) if opts.categories else None,
            max_insights=opts.max_insights,
            regenerate=opts.regenerate,
            epoch_min=opts.epoch_min,
            epoch_max=opts.epoch_max,
            as_of=opts.as_of,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    return MorningBriefResponse(**brief)
