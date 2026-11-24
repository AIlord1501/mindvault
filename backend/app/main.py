"""FastAPI application factory for MindVault."""

from __future__ import annotations

from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.health import router as health_router
from app.core.config import Settings


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    """Create and configure the local MindVault FastAPI application."""
    resolved_settings = settings or Settings.from_env()

    app = FastAPI(
        title=resolved_settings.app_name,
        version=resolved_settings.version,
        docs_url="/docs",
        redoc_url=None,
    )
    app.state.settings = resolved_settings

    # Restrict CORS to loopback origins and Electron file:// ('null') origin.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["null"],
        allow_origin_regex=r"^http://127\.0\.0\.1(:\d+)?$",
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Accept"],
    )

    app.include_router(health_router)
    return app


app = create_app()
