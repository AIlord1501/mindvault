"""Application configuration with loopback and offline-first safeguards."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ALLOWED_HOSTS = frozenset({"127.0.0.1"})


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
    host: str = "127.0.0.1"
    port: int = 8765
    data_dir: Path = Path("data")
    sqlite_path: Path = Path("data/mindvault.db")
    chroma_dir: Path = Path("data/chroma")
    models_dir: Path = Path("data/models")
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

    @classmethod
    def from_env(cls) -> Settings:
        """Load settings from environment variables with safe local defaults."""
        host = os.getenv("MINDVAULT_HOST", "127.0.0.1").strip()
        port_raw = os.getenv("MINDVAULT_PORT", "8765").strip()
        try:
            port = int(port_raw)
        except ValueError as exc:
            raise ValueError(f"MINDVAULT_PORT must be an integer, got '{port_raw}'.") from exc

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
            embedding_model=embedding_model,
            llm_model_path=llm_model_path,
            offline_only=offline_only,
        )
