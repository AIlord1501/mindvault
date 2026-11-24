"""Version 1 (/api/v1) FastAPI router."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, ConfigDict

from app.api.health import router as health_router
from app.api.v1.beliefs import router as beliefs_router
from app.api.v1.chat import router as chat_router
from app.api.v1.dashboard import router as dashboard_router
from app.api.v1.graph import router as graph_router
from app.api.v1.ingestion import router as ingestion_router
from app.api.v1.retention import router as retention_router
from app.api.v1.retrieval import router as retrieval_router

router = APIRouter()
router.include_router(health_router)
router.include_router(beliefs_router)
router.include_router(chat_router)
router.include_router(dashboard_router)
router.include_router(graph_router)
router.include_router(ingestion_router)
router.include_router(retention_router)
router.include_router(retrieval_router)


class LocalPathsInfo(BaseModel):
    """Configured local storage paths."""

    model_config = ConfigDict(extra="forbid")

    data_dir: str
    sqlite_path: str
    chroma_dir: str
    models_dir: str
    logs_dir: str


class SystemStatusResponse(BaseModel):
    """Structured backend status returned under /api/v1/system/status."""

    model_config = ConfigDict(extra="forbid")

    service: str
    version: str
    api_version: str
    offline_only: bool
    database_ok: bool
    applied_migrations: list[int]
    paths: Optional[LocalPathsInfo] = None


@router.get("/system/status", response_model=SystemStatusResponse, tags=["system"])
def get_system_status(
    request: Request,
    include_paths: bool = Query(
        default=False,
        description="Include configured local storage paths in the response.",
    ),
) -> SystemStatusResponse:
    """Return backend runtime status and applied SQLite schema migrations."""
    settings = request.app.state.settings
    db = getattr(request.app.state, "db", None)
    migrations = getattr(request.app.state, "migrations", None)

    db_ok = False
    applied: list[int] = []
    if db is not None:
        db_ok, _ = db.health_check()
        if db_ok and migrations is not None:
            applied = migrations.applied_versions()

    paths = None
    if include_paths:
        paths = LocalPathsInfo(
            data_dir=str(settings.data_dir),
            sqlite_path=str(settings.sqlite_path),
            chroma_dir=str(settings.chroma_dir),
            models_dir=str(settings.models_dir),
            logs_dir=str(settings.logs_dir),
        )

    return SystemStatusResponse(
        service="mindvault-backend",
        version=settings.version,
        api_version="v1",
        offline_only=settings.offline_only,
        database_ok=db_ok,
        applied_migrations=applied,
        paths=paths,
    )
