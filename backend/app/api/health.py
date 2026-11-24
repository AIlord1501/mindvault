"""Health check endpoint for the local MindVault backend."""

from __future__ import annotations

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

router = APIRouter(tags=["system"])


class HealthResponse(BaseModel):
    """Structured response returned by the backend health check."""

    status: str
    service: str
    version: str
    offline_only: bool
    database: str
    schema_version: int


def build_health_response(request: Request) -> tuple[HealthResponse, int]:
    """Compute live service and SQLite database health."""
    settings = request.app.state.settings
    db = getattr(request.app.state, "db", None)
    migrations = getattr(request.app.state, "migrations", None)

    db_status = "uninitialized"
    schema_version = 0
    is_healthy = False

    if db is not None:
        ok, _err = db.health_check()
        if ok:
            db_status = "ok"
            is_healthy = True
            if migrations is not None:
                schema_version = migrations.current_version()
        else:
            db_status = "error"

    payload = HealthResponse(
        status="ok" if is_healthy else "degraded",
        service="mindvault-backend",
        version=settings.version,
        offline_only=settings.offline_only,
        database=db_status,
        schema_version=schema_version,
    )
    http_status = status.HTTP_200_OK if is_healthy else status.HTTP_503_SERVICE_UNAVAILABLE
    return payload, http_status


@router.get("/health", response_model=HealthResponse)
def get_health(request: Request) -> JSONResponse:
    """Return local service and SQLite readiness metadata."""
    payload, http_status = build_health_response(request)
    return JSONResponse(status_code=http_status, content=payload.model_dump())
