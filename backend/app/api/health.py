"""Health check endpoint for the local MindVault backend."""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter(tags=["system"])


class HealthResponse(BaseModel):
    """Structured response returned by the backend health check."""

    status: str
    service: str
    version: str
    offline_only: bool


@router.get("/health", response_model=HealthResponse)
def get_health(request: Request) -> HealthResponse:
    """Return local service readiness metadata."""
    settings = request.app.state.settings
    return HealthResponse(
        status="ok",
        service="mindvault-backend",
        version=settings.version,
        offline_only=settings.offline_only,
    )
