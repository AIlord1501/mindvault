"""Integration tests for FastAPI lifespan, /health, /api/v1, validation, and shutdown."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app


def _make_settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path,
        sqlite_path=tmp_path / "mindvault.db",
        chroma_dir=tmp_path / "chroma",
        models_dir=tmp_path / "models",
        logs_dir=tmp_path / "logs",
    )


def test_lifespan_startup_endpoints_and_graceful_shutdown(tmp_path: Path) -> None:
    """App lifespan should initialize SQLite, serve /health and /api/v1, and close cleanly."""
    app = create_app(_make_settings(tmp_path))

    with TestClient(app) as client:
        # Root /health
        health = client.get("/health")
        assert health.status_code == 200
        assert "X-Request-ID" in health.headers
        assert health.json() == {
            "status": "ok",
            "service": "mindvault-backend",
            "version": "0.1.0",
            "offline_only": True,
            "database": "ok",
            "schema_version": 3,
        }

        # Versioned /api/v1/health
        v1_health = client.get("/api/v1/health", headers={"X-Request-ID": "custom-req-id"})
        assert v1_health.status_code == 200
        assert v1_health.headers["X-Request-ID"] == "custom-req-id"
        assert v1_health.json()["status"] == "ok"

        # Versioned /api/v1/system/status without paths
        status_resp = client.get("/api/v1/system/status")
        assert status_resp.status_code == 200
        body = status_resp.json()
        assert body["api_version"] == "v1"
        assert body["database_ok"] is True
        assert body["applied_migrations"] == [1, 2, 3]
        assert body["paths"] is None

        # Versioned /api/v1/system/status with include_paths=true
        status_with_paths = client.get("/api/v1/system/status?include_paths=true")
        assert status_with_paths.status_code == 200
        paths_body = status_with_paths.json()["paths"]
        assert paths_body["sqlite_path"] == str(tmp_path / "mindvault.db")
        assert paths_body["data_dir"] == str(tmp_path)

        assert app.state.db.is_closed is False

    # After TestClient exits, lifespan shutdown must have closed the SQLite manager.
    assert app.state.db.is_closed is True


def test_request_validation_returns_structured_422(tmp_path: Path) -> None:
    """Invalid query parameters should return a clean structured 422 payload."""
    app = create_app(_make_settings(tmp_path))
    with TestClient(app) as client:
        resp = client.get("/api/v1/system/status?include_paths=not-a-boolean")
        assert resp.status_code == 422
        payload = resp.json()
        assert payload["error"] == "validation_error"
        assert payload["message"] == "Request validation failed."
        assert isinstance(payload["details"], list)
        assert len(payload["details"]) >= 1
        assert payload["details"][0]["loc"] == ["query", "include_paths"]


def test_health_reports_degraded_when_db_closed(tmp_path: Path) -> None:
    """GET /health should return 503 degraded if the database becomes unavailable."""
    app = create_app(_make_settings(tmp_path))
    with TestClient(app) as client:
        app.state.db.close()
        resp = client.get("/health")
        assert resp.status_code == 503
        payload = resp.json()
        assert payload["status"] == "degraded"
        assert payload["database"] == "error"
