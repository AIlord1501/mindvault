"""Version 1 (/api/v1) FastAPI router."""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, ConfigDict

from app.api.health import router as health_router
from app.api.v1.beliefs import router as beliefs_router
from app.api.v1.chat import router as chat_router
from app.api.v1.dashboard import router as dashboard_router
from app.api.v1.graph import router as graph_router
from app.api.v1.ingestion import router as ingestion_router
from app.api.v1.jobs import router as jobs_router
from app.api.v1.retention import router as retention_router
from app.api.v1.retrieval import router as retrieval_router

router = APIRouter()
router.include_router(health_router)
router.include_router(beliefs_router)
router.include_router(chat_router)
router.include_router(dashboard_router)
router.include_router(graph_router)
router.include_router(ingestion_router)
router.include_router(jobs_router)
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
    models: dict[str, Any] = {}
    index: dict[str, Any] = {}
    scheduler: dict[str, Any] = {}
    recovery: dict[str, Any] = {}


@router.get("/system/status", response_model=SystemStatusResponse, tags=["system"])
def get_system_status(
    request: Request,
    include_paths: bool = Query(
        default=False,
        description="Include configured local storage paths in the response.",
    ),
) -> SystemStatusResponse:
    """Return backend runtime status, model diagnostics, index status, and jobs."""
    settings = request.app.state.settings
    db = getattr(request.app.state, "db", None)
    migrations = getattr(request.app.state, "migrations", None)
    retrieval_service = getattr(request.app.state, "retrieval_service", None)
    llm_provider = getattr(request.app.state, "llm_provider", None)
    nli_adapter = getattr(request.app.state, "nli_adapter", None)
    job_scheduler = getattr(request.app.state, "job_scheduler", None)
    recovery_info = getattr(request.app.state, "recovery_info", {}) or {}

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

    index_status: dict[str, Any] = {}
    embedding_info: dict[str, Any] = {"available": False, "model": settings.embedding_model}
    if db_ok and retrieval_service is not None:
        try:
            index_status = retrieval_service.get_index_status()
            embedding_info = {
                "available": bool(index_status.get("embedding_available")),
                "model": str(index_status.get("embedding_model") or ""),
                "reason": index_status.get("embedding_error"),
            }
        except Exception as exc:
            index_status = {"error": str(exc)}

    llm_info: dict[str, Any] = {"available": False, "provider": settings.llm_provider}
    if llm_provider is not None:
        llm_ok, llm_reason = llm_provider.is_available()
        llm_info = {
            "available": bool(llm_ok),
            "provider": llm_provider.provider_name,
            "model": llm_provider.model_name,
            "reason": llm_reason,
        }

    nli_info: dict[str, Any] = {"available": False, "model": settings.nli_model_path}
    if nli_adapter is not None:
        nli_ok, nli_reason = nli_adapter.is_available()
        nli_info = {
            "available": bool(nli_ok),
            "model": nli_adapter.model_name,
            "reason": nli_reason,
        }

    scheduler_info = job_scheduler.get_status() if job_scheduler is not None else {}

    return SystemStatusResponse(
        service="mindvault-backend",
        version=settings.version,
        api_version="v1",
        offline_only=settings.offline_only,
        database_ok=db_ok,
        applied_migrations=applied,
        paths=paths,
        models={
            "embedding": embedding_info,
            "llm": llm_info,
            "nli": nli_info,
        },
        index=index_status,
        scheduler=scheduler_info,
        recovery=recovery_info,
    )
