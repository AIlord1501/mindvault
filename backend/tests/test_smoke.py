"""Bootstrap smoke tests for the MindVault backend."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "module_name",
    [
        "app",
        "app.api",
        "app.api.health",
        "app.api.v1",
        "app.api.v1.chat",
        "app.api.v1.ingestion",
        "app.api.v1.retrieval",
        "app.api.v1.router",
        "app.beliefs",
        "app.chat",
        "app.chat.planner",
        "app.chat.prompts",
        "app.chat.service",
        "app.core",
        "app.core.config",
        "app.core.logging",
        "app.db",
        "app.db.database",
        "app.db.migrations",
        "app.db.repositories",
        "app.graph",
        "app.ingestion",
        "app.main",
        "app.models",
        "app.models.embeddings",
        "app.models.llm",
        "app.retrieval",
        "app.retrieval.bm25_index",
        "app.retrieval.service",
        "app.retrieval.types",
        "app.retrieval.vector_store",
        "app.scheduler",
    ],
)
def test_backend_packages_importable(module_name: str) -> None:
    """All scaffolded backend packages should import cleanly."""
    module = importlib.import_module(module_name)
    assert module is not None


def test_settings_defaults_enforce_loopback_and_offline() -> None:
    """Default settings must bind only to 127.0.0.1 and default to offline mode."""
    settings = Settings()
    assert settings.host == "127.0.0.1"
    assert settings.port == 8765
    assert settings.api_prefix == "/api/v1"
    assert settings.offline_only is True


@pytest.mark.parametrize("bad_host", ["0.0.0.0", "192.168.1.10", "example.com", "::"])
def test_settings_reject_non_loopback_host(bad_host: str) -> None:
    """Binding to any non-loopback address must be rejected."""
    with pytest.raises(ValueError, match="127.0.0.1"):
        Settings(host=bad_host)


def test_health_endpoint_returns_ok(tmp_path: Path) -> None:
    """GET /health should return structured readiness metadata after lifespan startup."""
    settings = Settings(
        data_dir=tmp_path,
        sqlite_path=tmp_path / "mindvault.db",
        chroma_dir=tmp_path / "chroma",
        models_dir=tmp_path / "models",
        logs_dir=tmp_path / "logs",
        vault_dir=tmp_path / "vault",
    )
    app = create_app(settings)
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {
            "status": "ok",
            "service": "mindvault-backend",
            "version": "0.1.0",
            "offline_only": True,
            "database": "ok",
            "schema_version": 1,
        }


def test_launcher_check_mode() -> None:
    """run.py --check should validate configuration and exit cleanly."""
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "run.py"), "--check"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert '"host": "127.0.0.1"' in result.stdout
