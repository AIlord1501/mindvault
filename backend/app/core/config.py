"""Application configuration with loopback, local paths, and offline-first safeguards."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

ALLOWED_HOSTS = frozenset({"127.0.0.1"})
ALLOWED_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost"})
ALLOWED_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})
ALLOWED_LLM_PROVIDERS = frozenset({"llamacpp", "ollama"})


def validate_loopback_http_url(url: str, field_name: str = "url") -> str:
    """Ensure a local service URL targets http://127.0.0.1 or http://localhost only."""
    cleaned = url.strip().rstrip("/")
    parsed = urlparse(cleaned)
    if parsed.scheme != "http" or parsed.hostname not in ALLOWED_LOOPBACK_HOSTS:
        raise ValueError(
            f"Invalid {field_name} '{url}'. Only local loopback http://127.0.0.1 or "
            "http://localhost URLs are permitted."
        )
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"Invalid port in {field_name} '{url}'.") from exc
    if port is not None and not (1 <= port <= 65535):
        raise ValueError(f"Invalid port '{port}' in {field_name}.")
    return cleaned


def _parse_bool(value: str, default: bool = True) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _parse_int(env_name: str, default: int) -> int:
    raw = os.getenv(env_name, str(default)).strip()
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{env_name} must be an integer, got '{raw}'.") from exc


def _parse_float(env_name: str, default: float) -> float:
    raw = os.getenv(env_name, str(default)).strip()
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{env_name} must be a float, got '{raw}'.") from exc


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
    chroma_collection: str = "mindvault_chunks"
    models_dir: Path = Path("data/models")
    logs_dir: Path = Path("data/logs")
    vault_dir: Path = Path("data/vault")
    log_level: str = "INFO"
    log_to_file: bool = False
    sqlite_busy_timeout_ms: int = 5000
    max_upload_size_bytes: int = 50 * 1024 * 1024
    chunk_size: int = 1000
    chunk_overlap: int = 150
    embedding_model: str = ""
    embedding_batch_size: int = 32
    retrieval_top_k: int = 10
    retrieval_candidate_k: int = 30
    semantic_weight: float = 0.6
    bm25_weight: float = 0.4
    rrf_k: int = 60
    rerank_enabled: bool = True
    llm_provider: str = "llamacpp"
    llm_model_path: str = ""
    llm_context_size: int = 4096
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.2
    llm_n_gpu_layers: int = 0
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = ""
    ollama_timeout_seconds: float = 30.0
    chat_max_subquestions: int = 3
    chat_max_context_chunks: int = 12
    belief_dedup_threshold: float = 0.90
    belief_min_confidence: float = 0.30
    belief_candidate_top_k: int = 10
    belief_candidate_min_similarity: float = 0.35
    contradiction_threshold: float = 0.70
    entailment_threshold: float = 0.70
    nli_model_path: str = ""
    whisper_model_path: str = ""
    tesseract_cmd: str = ""
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
        if not self.chroma_collection.strip():
            raise ValueError("chroma_collection must not be empty.")
        if self.sqlite_busy_timeout_ms < 0:
            raise ValueError("sqlite_busy_timeout_ms must be non-negative.")
        if self.max_upload_size_bytes <= 0:
            raise ValueError("max_upload_size_bytes must be positive.")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and strictly less than chunk_size.")
        if self.embedding_batch_size <= 0:
            raise ValueError("embedding_batch_size must be positive.")
        if self.retrieval_top_k <= 0:
            raise ValueError("retrieval_top_k must be positive.")
        if self.retrieval_candidate_k < 1:
            raise ValueError("retrieval_candidate_k must be at least 1.")
        if self.semantic_weight < 0.0:
            raise ValueError("semantic_weight must be non-negative.")
        if self.bm25_weight < 0.0:
            raise ValueError("bm25_weight must be non-negative.")
        if self.semantic_weight + self.bm25_weight <= 0.0:
            raise ValueError("semantic_weight + bm25_weight must be greater than 0.")
        if self.rrf_k < 1:
            raise ValueError("rrf_k must be at least 1.")

        normalized_provider = self.llm_provider.strip().lower()
        if normalized_provider not in ALLOWED_LLM_PROVIDERS:
            allowed_p = sorted(ALLOWED_LLM_PROVIDERS)
            raise ValueError(
                f"Invalid llm_provider '{self.llm_provider}'. Must be one of {allowed_p}."
            )
        object.__setattr__(self, "llm_provider", normalized_provider)
        if self.llm_context_size < 256:
            raise ValueError("llm_context_size must be at least 256.")
        if self.llm_max_tokens < 1:
            raise ValueError("llm_max_tokens must be at least 1.")
        if not (0.0 <= self.llm_temperature <= 2.0):
            raise ValueError("llm_temperature must be between 0.0 and 2.0.")
        normalized_ollama_url = validate_loopback_http_url(
            self.ollama_base_url,
            field_name="ollama_base_url",
        )
        object.__setattr__(self, "ollama_base_url", normalized_ollama_url)
        if self.ollama_timeout_seconds <= 0.0:
            raise ValueError("ollama_timeout_seconds must be positive.")
        if not (1 <= self.chat_max_subquestions <= 5):
            raise ValueError("chat_max_subquestions must be between 1 and 5.")
        if not (1 <= self.chat_max_context_chunks <= 50):
            raise ValueError("chat_max_context_chunks must be between 1 and 50.")
        if not (0.0 < self.belief_dedup_threshold <= 1.0):
            raise ValueError("belief_dedup_threshold must be in (0.0, 1.0].")
        if not (0.0 <= self.belief_min_confidence <= 1.0):
            raise ValueError("belief_min_confidence must be between 0.0 and 1.0.")
        if not (1 <= self.belief_candidate_top_k <= 100):
            raise ValueError("belief_candidate_top_k must be between 1 and 100.")
        if not (0.0 <= self.belief_candidate_min_similarity <= 1.0):
            raise ValueError("belief_candidate_min_similarity must be between 0.0 and 1.0.")
        if not (0.0 < self.contradiction_threshold <= 1.0):
            raise ValueError("contradiction_threshold must be in (0.0, 1.0].")
        if not (0.0 < self.entailment_threshold <= 1.0):
            raise ValueError("entailment_threshold must be in (0.0, 1.0].")

    def ensure_local_dirs(self) -> None:
        """Create required local storage directories if they do not exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.vault_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> Settings:
        """Load settings from environment variables with safe local defaults."""
        host = os.getenv("MINDVAULT_HOST", "127.0.0.1").strip()
        port = _parse_int("MINDVAULT_PORT", 8765)
        sqlite_busy_timeout_ms = _parse_int("MINDVAULT_SQLITE_BUSY_TIMEOUT_MS", 5000)
        max_upload_size_bytes = _parse_int("MINDVAULT_MAX_UPLOAD_SIZE_BYTES", 50 * 1024 * 1024)
        chunk_size = _parse_int("MINDVAULT_CHUNK_SIZE", 1000)
        chunk_overlap = _parse_int("MINDVAULT_CHUNK_OVERLAP", 150)
        embedding_batch_size = _parse_int("MINDVAULT_EMBEDDING_BATCH_SIZE", 32)
        retrieval_top_k = _parse_int("MINDVAULT_RETRIEVAL_TOP_K", 10)
        retrieval_candidate_k = _parse_int("MINDVAULT_RETRIEVAL_CANDIDATE_K", 30)
        semantic_weight = _parse_float("MINDVAULT_SEMANTIC_WEIGHT", 0.6)
        bm25_weight = _parse_float("MINDVAULT_BM25_WEIGHT", 0.4)
        rrf_k = _parse_int("MINDVAULT_RRF_K", 60)
        rerank_enabled = _parse_bool(os.getenv("MINDVAULT_RERANK_ENABLED", "true"), default=True)

        llm_provider = os.getenv("MINDVAULT_LLM_PROVIDER", "llamacpp").strip()
        llm_model_path = os.getenv("MINDVAULT_LLM_MODEL_PATH", "").strip()
        llm_context_size = _parse_int("MINDVAULT_LLM_CONTEXT_SIZE", 4096)
        llm_max_tokens = _parse_int("MINDVAULT_LLM_MAX_TOKENS", 1024)
        llm_temperature = _parse_float("MINDVAULT_LLM_TEMPERATURE", 0.2)
        llm_n_gpu_layers = _parse_int("MINDVAULT_LLM_N_GPU_LAYERS", 0)
        ollama_base_url = os.getenv("MINDVAULT_OLLAMA_BASE_URL", "http://127.0.0.1:11434").strip()
        ollama_model = os.getenv("MINDVAULT_OLLAMA_MODEL", "").strip()
        ollama_timeout_seconds = _parse_float("MINDVAULT_OLLAMA_TIMEOUT_SECONDS", 30.0)
        chat_max_subquestions = _parse_int("MINDVAULT_CHAT_MAX_SUBQUESTIONS", 3)
        chat_max_context_chunks = _parse_int("MINDVAULT_CHAT_MAX_CONTEXT_CHUNKS", 12)
        belief_dedup_threshold = _parse_float("MINDVAULT_BELIEF_DEDUP_THRESHOLD", 0.90)
        belief_min_confidence = _parse_float("MINDVAULT_BELIEF_MIN_CONFIDENCE", 0.30)
        belief_candidate_top_k = _parse_int("MINDVAULT_BELIEF_CANDIDATE_TOP_K", 10)
        belief_candidate_min_similarity = _parse_float(
            "MINDVAULT_BELIEF_CANDIDATE_MIN_SIMILARITY",
            0.35,
        )
        contradiction_threshold = _parse_float("MINDVAULT_CONTRADICTION_THRESHOLD", 0.70)
        entailment_threshold = _parse_float("MINDVAULT_ENTAILMENT_THRESHOLD", 0.70)
        nli_model_path = os.getenv("MINDVAULT_NLI_MODEL_PATH", "").strip()

        data_dir = Path(os.getenv("MINDVAULT_DATA_DIR", "data")).expanduser()
        sqlite_path = Path(
            os.getenv("MINDVAULT_SQLITE_PATH", str(data_dir / "mindvault.db"))
        ).expanduser()
        chroma_dir = Path(
            os.getenv("MINDVAULT_CHROMA_DIR", str(data_dir / "chroma"))
        ).expanduser()
        chroma_collection = os.getenv("MINDVAULT_CHROMA_COLLECTION", "mindvault_chunks").strip()
        models_dir = Path(
            os.getenv("MINDVAULT_MODELS_DIR", str(data_dir / "models"))
        ).expanduser()
        logs_dir = Path(
            os.getenv("MINDVAULT_LOGS_DIR", str(data_dir / "logs"))
        ).expanduser()
        vault_dir = Path(
            os.getenv("MINDVAULT_VAULT_DIR", str(data_dir / "vault"))
        ).expanduser()

        log_level = os.getenv("MINDVAULT_LOG_LEVEL", "INFO").strip()
        log_to_file = _parse_bool(os.getenv("MINDVAULT_LOG_TO_FILE", "false"), default=False)
        embedding_model = os.getenv("MINDVAULT_EMBEDDING_MODEL", "").strip()
        whisper_model_path = os.getenv("MINDVAULT_WHISPER_MODEL_PATH", "").strip()
        tesseract_cmd = os.getenv("MINDVAULT_TESSERACT_CMD", "").strip()
        offline_only = _parse_bool(os.getenv("MINDVAULT_OFFLINE_ONLY", "true"), default=True)

        return cls(
            host=host,
            port=port,
            data_dir=data_dir,
            sqlite_path=sqlite_path,
            chroma_dir=chroma_dir,
            chroma_collection=chroma_collection,
            models_dir=models_dir,
            logs_dir=logs_dir,
            vault_dir=vault_dir,
            log_level=log_level,
            log_to_file=log_to_file,
            sqlite_busy_timeout_ms=sqlite_busy_timeout_ms,
            max_upload_size_bytes=max_upload_size_bytes,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_model=embedding_model,
            embedding_batch_size=embedding_batch_size,
            retrieval_top_k=retrieval_top_k,
            retrieval_candidate_k=retrieval_candidate_k,
            semantic_weight=semantic_weight,
            bm25_weight=bm25_weight,
            rrf_k=rrf_k,
            rerank_enabled=rerank_enabled,
            llm_provider=llm_provider,
            llm_model_path=llm_model_path,
            llm_context_size=llm_context_size,
            llm_max_tokens=llm_max_tokens,
            llm_temperature=llm_temperature,
            llm_n_gpu_layers=llm_n_gpu_layers,
            ollama_base_url=ollama_base_url,
            ollama_model=ollama_model,
            ollama_timeout_seconds=ollama_timeout_seconds,
            chat_max_subquestions=chat_max_subquestions,
            chat_max_context_chunks=chat_max_context_chunks,
            belief_dedup_threshold=belief_dedup_threshold,
            belief_min_confidence=belief_min_confidence,
            belief_candidate_top_k=belief_candidate_top_k,
            belief_candidate_min_similarity=belief_candidate_min_similarity,
            contradiction_threshold=contradiction_threshold,
            entailment_threshold=entailment_threshold,
            nli_model_path=nli_model_path,
            whisper_model_path=whisper_model_path,
            tesseract_cmd=tesseract_cmd,
            offline_only=offline_only,
        )
