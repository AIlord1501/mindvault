"""FastAPI application factory, lifespan management, and middleware for MindVault."""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator, Awaitable, Callable, Optional

from fastapi import FastAPI, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.health import router as health_router
from app.api.v1.router import router as v1_router
from app.core.config import Settings
from app.core.logging import configure_logging, shutdown_logging
from app.db.database import DatabaseManager
from app.db.migrations import MigrationRunner
from app.db.repositories import DocumentRepository
from app.ingestion.pipeline import IngestionPipeline
from app.models.embeddings import BaseEmbeddingAdapter, SentenceTransformerEmbeddingAdapter
from app.retrieval.bm25_index import BM25Index
from app.retrieval.service import RetrievalService
from app.retrieval.vector_store import ChromaVectorStore


def create_app(
    settings: Optional[Settings] = None,
    embedding_adapter: Optional[BaseEmbeddingAdapter] = None,
) -> FastAPI:
    """Create and configure the local MindVault FastAPI application."""
    resolved_settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        resolved_settings.ensure_local_dirs()
        log_file = (
            resolved_settings.logs_dir / "mindvault.log"
            if resolved_settings.log_to_file
            else None
        )
        logger = configure_logging(level=resolved_settings.log_level, log_file=log_file)
        db = DatabaseManager(
            db_path=resolved_settings.sqlite_path,
            busy_timeout_ms=resolved_settings.sqlite_busy_timeout_ms,
        )
        migrations = MigrationRunner(db)
        newly_applied = migrations.apply_pending()
        document_repo = DocumentRepository(db)

        embedder = embedding_adapter or SentenceTransformerEmbeddingAdapter(
            model_name_or_path=resolved_settings.embedding_model,
            models_dir=resolved_settings.models_dir,
            offline_only=resolved_settings.offline_only,
        )
        vector_store = ChromaVectorStore(
            persist_dir=resolved_settings.chroma_dir,
            collection_name=resolved_settings.chroma_collection,
        )
        bm25_index = BM25Index()
        retrieval_service = RetrievalService(
            settings=resolved_settings,
            repo=document_repo,
            vector_store=vector_store,
            bm25_index=bm25_index,
            embedder=embedder,
        )
        synced_chunks = retrieval_service.sync_bm25_from_sqlite()

        ingestion_pipeline = IngestionPipeline(
            settings=resolved_settings,
            repo=document_repo,
            retrieval_service=retrieval_service,
        )

        application.state.logger = logger
        application.state.db = db
        application.state.migrations = migrations
        application.state.document_repo = document_repo
        application.state.vector_store = vector_store
        application.state.retrieval_service = retrieval_service
        application.state.ingestion_pipeline = ingestion_pipeline

        logger.info(
            "MindVault backend started",
            extra={
                "event": "startup",
                "host": resolved_settings.host,
                "port": resolved_settings.port,
                "sqlite_path": str(resolved_settings.sqlite_path),
                "chroma_dir": str(resolved_settings.chroma_dir),
                "applied_migrations": newly_applied,
                "schema_version": migrations.current_version(),
                "bm25_synced_chunks": synced_chunks,
                "offline_only": resolved_settings.offline_only,
            },
        )
        try:
            yield
        finally:
            logger.info("MindVault backend shutting down", extra={"event": "shutdown"})
            vector_store.close()
            db.close()
            shutdown_logging()

    app = FastAPI(
        title=resolved_settings.app_name,
        version=resolved_settings.version,
        docs_url="/docs",
        redoc_url=None,
        lifespan=lifespan,
    )
    app.state.settings = resolved_settings
    app.state.db = None
    app.state.migrations = None

    # Restrict CORS to loopback origins and Electron file:// ('null') origin.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["null"],
        allow_origin_regex=r"^http://127\.0\.0\.1(:\d+)?$",
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Accept", "X-Request-ID"],
    )

    @app.middleware("http")
    async def structured_request_logging(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - start) * 1000.0, 2)
        response.headers["X-Request-ID"] = request_id

        logger = getattr(request.app.state, "logger", None)
        if logger is not None:
            logger.info(
                "HTTP request completed",
                extra={
                    "event": "http_request",
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": duration_ms,
                },
            )
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        _request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        details = [
            {
                "loc": list(err.get("loc", ())),
                "msg": err.get("msg", "Invalid value"),
                "type": err.get("type", "validation_error"),
            }
            for err in exc.errors()
        ]
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "validation_error",
                "message": "Request validation failed.",
                "details": details,
            },
        )

    app.include_router(health_router)
    app.include_router(v1_router, prefix=resolved_settings.api_prefix)
    return app


app = create_app()
