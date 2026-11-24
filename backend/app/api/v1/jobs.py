"""REST API endpoints for MindVault background job scheduling and telemetry."""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict

from app.scheduler.background import BackgroundJobScheduler, JobNotFoundError

router = APIRouter(tags=["jobs"])


class JobStatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    name: str
    description: str
    interval_seconds: int
    paused: bool
    state: str
    run_count: int
    success_count: int
    failure_count: int
    skipped_count: int
    last_run_at: Optional[str] = None
    last_duration_ms: Optional[float] = None
    next_run_at: Optional[str] = None
    last_result: dict[str, Any]
    last_error: Optional[str] = None


class JobDetailResponse(JobStatusResponse):
    history: list[dict[str, Any]]


class SchedulerStatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    running: bool
    shutdown_completed: bool
    backend: str
    jobs: list[JobStatusResponse]
    recent_executions: list[dict[str, Any]]


class JobExecutionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    name: str
    status: str
    ran_at: str
    duration_ms: float
    result: dict[str, Any]
    error: Optional[str] = None


def _get_scheduler(request: Request) -> BackgroundJobScheduler:
    scheduler = getattr(request.app.state, "job_scheduler", None)
    if scheduler is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Background job scheduler is not initialized.",
        )
    return scheduler


@router.get("/jobs", response_model=SchedulerStatusResponse)
def list_jobs(request: Request) -> SchedulerStatusResponse:
    """Return scheduler health, configured job intervals, and recent runs."""
    scheduler = _get_scheduler(request)
    return SchedulerStatusResponse(**scheduler.get_status())


@router.get("/jobs/{job_id}", response_model=JobDetailResponse)
def get_job_status(request: Request, job_id: str) -> JobDetailResponse:
    """Return telemetry and execution history for one background job."""
    scheduler = _get_scheduler(request)
    try:
        payload = scheduler.get_job(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return JobDetailResponse(**payload)


@router.post("/jobs/{job_id}/run", response_model=JobExecutionResponse)
def run_job_now(request: Request, job_id: str) -> JobExecutionResponse:
    """Trigger one background job synchronously and return its execution record."""
    scheduler = _get_scheduler(request)
    try:
        result = scheduler.run_job_now(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return JobExecutionResponse(**result)


@router.post("/jobs/{job_id}/pause", response_model=JobStatusResponse)
def pause_job(request: Request, job_id: str) -> JobStatusResponse:
    """Pause periodic execution for a background job."""
    scheduler = _get_scheduler(request)
    try:
        payload = scheduler.pause_job(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return JobStatusResponse(**payload)


@router.post("/jobs/{job_id}/resume", response_model=JobStatusResponse)
def resume_job(request: Request, job_id: str) -> JobStatusResponse:
    """Resume periodic execution for a paused background job."""
    scheduler = _get_scheduler(request)
    try:
        payload = scheduler.resume_job(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return JobStatusResponse(**payload)
