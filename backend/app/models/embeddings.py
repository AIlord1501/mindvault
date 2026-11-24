"""Local sentence-transformer-compatible embedding adapter with strict offline guarantees."""

from __future__ import annotations

import importlib
import os
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Optional


class EmbeddingUnavailableError(RuntimeError):
    """Raised when no local embedding model is configured or available offline."""


class BaseEmbeddingAdapter(ABC):
    """Contract implemented by all local embedding adapters."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return configured model identifier or path."""

    @abstractmethod
    def is_available(self) -> tuple[bool, Optional[str]]:
        """Return (available, reason_if_unavailable) without downloading external assets."""

    @abstractmethod
    def embed_documents(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Embed a batch of document texts into dense float vectors."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single search query into a dense float vector."""


class SentenceTransformerEmbeddingAdapter(BaseEmbeddingAdapter):
    """Offline-first local embedding adapter backed by sentence-transformers."""

    def __init__(
        self,
        model_name_or_path: str = "",
        models_dir: Path = Path("data/models"),
        offline_only: bool = True,
        loader: Optional[Callable[..., Any]] = None,
    ) -> None:
        self._raw_model = model_name_or_path.strip()
        self._models_dir = models_dir
        self._offline_only = offline_only
        self._loader = loader
        self._lock = threading.Lock()
        self._model: Any = None
        self._resolved_target: Optional[str] = None
        self._load_error: Optional[str] = None

    @property
    def model_name(self) -> str:
        return self._resolved_target or self._raw_model

    def _resolve_model_target(self) -> str:
        if not self._raw_model:
            raise EmbeddingUnavailableError(
                "No local embedding model configured. Set MINDVAULT_EMBEDDING_MODEL."
            )

        direct = Path(self._raw_model).expanduser()
        if direct.exists():
            return str(direct)

        in_models_dir = (self._models_dir / self._raw_model).expanduser()
        if in_models_dir.exists():
            return str(in_models_dir)

        looks_like_path = (
            self._raw_model.startswith((".", "/", "\\"))
            or "\\" in self._raw_model
            or ":" in self._raw_model
        )
        if looks_like_path or (self._offline_only and self._loader is None):
            raise EmbeddingUnavailableError(
                f"Configured local embedding model path does not exist offline: {self._raw_model}"
            )

        return self._raw_model

    def _load_sentence_transformer_cls(self) -> Any:
        if self._loader is not None:
            return self._loader
        try:
            mod = importlib.import_module("sentence_transformers")
            return mod.SentenceTransformer
        except ImportError as exc:
            raise EmbeddingUnavailableError(
                "sentence-transformers is not installed in the current environment."
            ) from exc

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model

        with self._lock:
            if self._model is not None:
                return self._model
            if self._load_error is not None:
                raise EmbeddingUnavailableError(self._load_error)

            try:
                target = self._resolve_model_target()
                if self._offline_only:
                    os.environ["HF_HUB_OFFLINE"] = "1"
                    os.environ["TRANSFORMERS_OFFLINE"] = "1"

                st_cls = self._load_sentence_transformer_cls()
                try:
                    self._model = st_cls(target, local_files_only=True)
                except TypeError:
                    # Compatibility fallback for custom or older constructors while keeping
                    # HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE active.
                    self._model = st_cls(target)
                self._resolved_target = target
                return self._model
            except EmbeddingUnavailableError as exc:
                self._load_error = str(exc)
                raise
            except Exception as exc:
                msg = (
                    f"Failed to load local embedding model '{self._raw_model}' offline: {exc}"
                )
                self._load_error = msg
                raise EmbeddingUnavailableError(msg) from exc

    def is_available(self) -> tuple[bool, Optional[str]]:
        if not self._raw_model:
            return False, "No local embedding model configured. Set MINDVAULT_EMBEDDING_MODEL."
        if self._model is not None:
            return True, None
        if self._load_error is not None:
            return False, self._load_error
        try:
            self._ensure_model()
            return True, None
        except EmbeddingUnavailableError as exc:
            return False, str(exc)

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        if not texts:
            return []
        model = self._ensure_model()
        safe_batch = max(1, int(batch_size))

        try:
            encoded = model.encode(
                texts,
                batch_size=safe_batch,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except TypeError:
            encoded = model.encode(texts, batch_size=safe_batch)

        vectors: list[list[float]] = []
        for row in encoded:
            raw_row = row.tolist() if hasattr(row, "tolist") else list(row)
            vectors.append([float(val) for val in raw_row])
        return vectors

    def embed_query(self, text: str) -> list[float]:
        vectors = self.embed_documents([text], batch_size=1)
        if not vectors:
            raise EmbeddingUnavailableError("Failed to compute embedding for query.")
        return vectors[0]
