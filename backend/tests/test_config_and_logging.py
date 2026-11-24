"""Unit tests for configuration management and structured JSON logging."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from app.core.config import Settings
from app.core.logging import JsonLogFormatter, configure_logging, shutdown_logging


def test_settings_from_env_and_ensure_local_dirs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Settings.from_env should resolve local paths and create them on demand."""
    data_dir = tmp_path / "vault"
    monkeypatch.setenv("MINDVAULT_HOST", "127.0.0.1")
    monkeypatch.setenv("MINDVAULT_PORT", "8765")
    monkeypatch.setenv("MINDVAULT_DATA_DIR", str(data_dir))
    monkeypatch.setenv("MINDVAULT_LOG_LEVEL", "debug")
    monkeypatch.setenv("MINDVAULT_LOG_TO_FILE", "true")

    settings = Settings.from_env()
    assert settings.host == "127.0.0.1"
    assert settings.port == 8765
    assert settings.data_dir == data_dir
    assert settings.sqlite_path == data_dir / "mindvault.db"
    assert settings.chroma_dir == data_dir / "chroma"
    assert settings.models_dir == data_dir / "models"
    assert settings.logs_dir == data_dir / "logs"
    assert settings.log_level == "DEBUG"
    assert settings.log_to_file is True

    settings.ensure_local_dirs()
    assert settings.data_dir.is_dir()
    assert settings.chroma_dir.is_dir()
    assert settings.models_dir.is_dir()
    assert settings.logs_dir.is_dir()


def test_settings_validation_rejects_invalid_values() -> None:
    """Settings must validate ports, API prefix, log level, and timeouts."""
    with pytest.raises(ValueError, match="Port"):
        Settings(port=0)
    with pytest.raises(ValueError, match="api_prefix"):
        Settings(api_prefix="api/v1")
    with pytest.raises(ValueError, match="api_prefix"):
        Settings(api_prefix="/api/v1/")
    with pytest.raises(ValueError, match="log_level"):
        Settings(log_level="VERBOSE")
    with pytest.raises(ValueError, match="sqlite_busy_timeout_ms"):
        Settings(sqlite_busy_timeout_ms=-1)


def test_json_log_formatter_includes_extra_fields() -> None:
    """JsonLogFormatter should emit valid JSON with structured extra fields."""
    formatter = JsonLogFormatter()
    record = logging.LogRecord(
        name="mindvault.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="hello %s",
        args=("world",),
        exc_info=None,
    )
    record.request_id = "req-123"
    record.duration_ms = 4.2

    parsed = json.loads(formatter.format(record))
    assert parsed["level"] == "INFO"
    assert parsed["logger"] == "mindvault.test"
    assert parsed["message"] == "hello world"
    assert parsed["request_id"] == "req-123"
    assert parsed["duration_ms"] == 4.2
    assert "timestamp" in parsed


def test_configure_and_shutdown_logging_writes_file(tmp_path: Path) -> None:
    """configure_logging should write structured JSON to a local file and shut down cleanly."""
    log_file = tmp_path / "logs" / "mindvault.log"
    logger = configure_logging(level="INFO", log_file=log_file)
    logger.info("startup check", extra={"event": "test_event"})
    shutdown_logging()

    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["message"] == "startup check"
    assert entry["event"] == "test_event"
    assert len(logger.handlers) == 0
