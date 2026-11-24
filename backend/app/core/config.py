"""Application configuration with loopback, local paths, and offline-first safeguards."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ALLOWED_HOSTS = frozenset({"127.0.0.1"})
ALLOWED_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})


def _parse_bool(value: str, default: bool = True) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


@dataclass(frozen=True)
class Settings:
    """Runtime settings for the MindVault backend."""

    app_name: str = "MindVault"
    version: str = "0.1.0"
    api_prefix: str = "/api/v1"
    host: str = "127.0.0.1"
    port: int = 8765
    data_dir: Path = Path("data")
    sqlite_path: Path = Path("data/mindvault.db")
    chroma_dir: Path = Path("data/chroma")
    models_dir: Path = Path("data/models")
    logs_dir: Path = Path("data/logs")
    log_level: str = "INFO"
    log_to_file: bool = False
    sqlite_busy_timeout_ms: int = 5000
    embedding_model: str = ""
    llm_model_path: str = ""
    offline_only: bool = True

    def __post_init__(self) -> None:
        if self.host not in ALLOWED_HOSTS:
            raise ValueError(
                f"Invalid host '{self.host}'. MindVault backend must bind exclusively to 127.0.0.1."
            )
        if not (1 <= self.port <= 65535):
            raise ValueError(f"Invalid port '{self.port}'. Port must be between 1 and 65535.")
        if not self.api_prefix.startswith("/") or self.api_prefix.endswith("/"):
            raise ValueError(
                f"Invalid api_prefix '{self.api_prefix}'. Must start with '/' and not end with '/'."
            )
        normalized_level = self.log_level.upper()
        if normalized_level not in ALLOWED_LOG_LEVELS:
            allowed = sorted(ALLOWED_LOG_LEVELS)
            raise ValueError(f"Invalid log_level '{self.log_level}'. Must be one of {allowed}.")
        object.__setattr__(self, "log_level", normalized_level)
        if self.sqlite_busy_timeout_ms < 0:
            raise ValueError("sqlite_busy_timeout_ms must be non-negative.")

    def ensure_local_dirs(self) -> None:
        """Create required local storage directories if they do not exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> Settings:
        """Load settings from environment variables with safe local defaults."""
        host = os.getenv("MINDVAULT_HOST", "127.0.0.1").strip()
        port_raw = os.getenv("MINDVAULT_PORT", "8765").strip()
        try:
            port = int(port_raw)
        except ValueError as exc:
            raise ValueError(f"MINDVAULT_PORT must be an integer, got '{port_raw}'.") from exc

        timeout_raw = os.getenv("MINDVAULT_SQLITE_BUSY_TIMEOUT_MS", "5000").strip()
        try:
            sqlite_busy_timeout_ms = int(timeout_raw)
        except ValueError as exc:
            raise ValueError(
                f"MINDVAULT_SQLITE_BUSY_TIMEOUT_MS must be an integer, got '{timeout_raw}'."
            ) from exc

        data_dir = Path(os.getenv("MINDVAULT_DATA_DIR", "data")).expanduser()
        sqlite_path = Path(
            os.getenv("MINDVAULT_SQLITE_PATH", str(data_dir / "mindvault.db"))
        ).expanduser()
        chroma_dir = Path(
            os.getenv("MINDVAULT_CHROMA_DIR", str(data_dir / "chroma"))
        ).expanduser()
        models_dir = Path(
            os.getenv("MINDVAULT_MODELS_DIR", str(data_dir / "models"))
        ).expanduser()
        logs_dir = Path(
            os.getenv("MINDVAULT_LOGS_DIR", str(data_dir / "logs"))
        ).expanduser()
        log_level = os.getenv("MINDVAULT_LOG_LEVEL", "INFO").strip()
        log_to_file = _parse_bool(os.getenv("MINDVAULT_LOG_TO_FILE", "false"), default=False)
        embedding_model = os.getenv("MINDVAULT_EMBEDDING_MODEL", "").strip()
        llm_model_path = os.getenv("MINDVAULT_LLM_MODEL_PATH", "").strip()
        offline_only = _parse_bool(os.getenv("MINDVAULT_OFFLINE_ONLY", "true"), default=True)

        return cls(
            host=host,
            port=port,
            data_dir=data_dir,
            sqlite_path=sqlite_path,
            chroma_dir=chroma_dir,
            models_dir=models_dir,
            logs_dir=logs_dir,
            log_level=log_level,
            log_to_file=log_to_file,
            sqlite_busy_timeout_ms=sqlite_busy_timeout_ms,
            embedding_model=embedding_model,
            llm_model_path=llm_model_path,
            offline_only=offline_only,
        )
