"""Local-only LLM provider abstraction for llama-cpp-python (GGUF) and loopback Ollama."""

from __future__ import annotations

import importlib
import importlib.util
import json
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

import httpx

from app.core.config import Settings, validate_loopback_http_url


class LLMError(RuntimeError):
    """Base exception for local LLM provider errors."""


class LLMUnavailableError(LLMError):
    """Raised when no local LLM model or runtime is configured or reachable."""


class LLMGenerationError(LLMError):
    """Raised when local LLM inference fails during generation."""


class LLMCancelledError(LLMError):
    """Raised when streaming generation is cancelled by caller or client disconnect."""


CancelCheck = Callable[[], bool]


def _check_cancelled(cancel_check: Optional[CancelCheck]) -> None:
    if cancel_check is not None and cancel_check():
        raise LLMCancelledError("Local LLM generation cancelled.")


class BaseLLMProvider(ABC):
    """Contract implemented by all local LLM providers."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return short provider identifier."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return configured model identifier or path."""

    @abstractmethod
    def is_available(self) -> tuple[bool, Optional[str]]:
        """Check whether the local LLM model/runtime is ready without remote calls."""

    @abstractmethod
    def stream_generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.2,
        cancel_check: Optional[CancelCheck] = None,
    ) -> Iterator[str]:
        """Yield generated text tokens from the local LLM."""

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.2,
        cancel_check: Optional[CancelCheck] = None,
    ) -> str:
        """Generate a complete response by collecting streamed tokens."""
        parts = list(
            self.stream_generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                cancel_check=cancel_check,
            )
        )
        return "".join(parts)


class LlamaCppProvider(BaseLLMProvider):
    """Local GGUF inference provider powered by llama-cpp-python."""

    def __init__(
        self,
        *,
        model_path: str = "",
        models_dir: Path = Path("data/models"),
        context_size: int = 4096,
        n_gpu_layers: int = 0,
        llama_loader: Optional[Callable[..., Any]] = None,
    ) -> None:
        self._raw_model_path = model_path.strip()
        self._models_dir = models_dir
        self._context_size = max(256, int(context_size))
        self._n_gpu_layers = int(n_gpu_layers)
        self._llama_loader = llama_loader
        self._lock = threading.Lock()
        self._model: Any = None
        self._resolved_path: Optional[Path] = None
        self._load_error: Optional[str] = None

    @property
    def provider_name(self) -> str:
        return "llamacpp"

    @property
    def model_name(self) -> str:
        if self._resolved_path is not None:
            return self._resolved_path.name
        return Path(self._raw_model_path).name if self._raw_model_path else ""

    def _resolve_gguf_path(self) -> Path:
        if not self._raw_model_path:
            raise LLMUnavailableError(
                "No local GGUF model configured. Set MINDVAULT_LLM_MODEL_PATH."
            )

        direct = Path(self._raw_model_path).expanduser()
        in_models_dir = (self._models_dir / self._raw_model_path).expanduser()
        candidate = direct if direct.exists() else in_models_dir

        if not candidate.exists() or not candidate.is_file():
            raise LLMUnavailableError(
                f"Configured GGUF model file does not exist: {self._raw_model_path}"
            )
        if candidate.suffix.lower() != ".gguf":
            raise LLMUnavailableError(
                f"Model file '{candidate.name}' must have a .gguf extension."
            )
        if candidate.stat().st_size <= 0:
            raise LLMUnavailableError(f"GGUF model file '{candidate.name}' is empty.")
        return candidate

    def _runtime_installed(self) -> bool:
        if self._llama_loader is not None:
            return True
        return importlib.util.find_spec("llama_cpp") is not None

    def is_available(self) -> tuple[bool, Optional[str]]:
        if self._load_error is not None:
            return False, self._load_error
        try:
            self._resolved_path = self._resolve_gguf_path()
        except LLMUnavailableError as exc:
            return False, str(exc)

        if not self._runtime_installed():
            return False, "llama-cpp-python is not installed in the current environment."
        return True, None

    def _load_llama_cls(self) -> Any:
        if self._llama_loader is not None:
            return self._llama_loader
        try:
            mod = importlib.import_module("llama_cpp")
            return mod.Llama
        except ImportError as exc:
            raise LLMUnavailableError(
                "llama-cpp-python is not installed in the current environment."
            ) from exc

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model

        with self._lock:
            if self._model is not None:
                return self._model
            if self._load_error is not None:
                raise LLMUnavailableError(self._load_error)

            try:
                resolved = self._resolve_gguf_path()
                llama_cls = self._load_llama_cls()
                self._model = llama_cls(
                    model_path=str(resolved),
                    n_ctx=self._context_size,
                    n_gpu_layers=self._n_gpu_layers,
                    verbose=False,
                )
                self._resolved_path = resolved
                return self._model
            except LLMUnavailableError as exc:
                self._load_error = str(exc)
                raise
            except Exception as exc:
                msg = f"Failed to load GGUF model '{self._raw_model_path}': {exc}"
                self._load_error = msg
                raise LLMUnavailableError(msg) from exc

    def stream_generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.2,
        cancel_check: Optional[CancelCheck] = None,
    ) -> Iterator[str]:
        _check_cancelled(cancel_check)
        model = self._ensure_model()
        _check_cancelled(cancel_check)

        try:
            stream = model.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max(1, int(max_tokens)),
                temperature=float(temperature),
                stream=True,
            )
            for chunk in stream:
                _check_cancelled(cancel_check)
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                piece = delta.get("content")
                if isinstance(piece, str) and piece:
                    yield piece
        except LLMCancelledError:
            raise
        except LLMError:
            raise
        except Exception as exc:
            raise LLMGenerationError(f"llama-cpp generation failed: {exc}") from exc


class OllamaProvider(BaseLLMProvider):
    """Optional loopback-only Ollama adapter (never calls remote hosts or /api/pull)."""

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:11434",
        model: str = "",
        timeout_seconds: float = 30.0,
        http_transport: Optional[httpx.BaseTransport] = None,
    ) -> None:
        self._base_url = validate_loopback_http_url(base_url, field_name="ollama_base_url")
        self._model = model.strip()
        self._timeout = float(timeout_seconds)
        if self._timeout <= 0.0:
            raise ValueError("timeout_seconds must be positive.")
        self._transport = http_transport

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model

    def _make_client(self, timeout: Optional[float] = None) -> httpx.Client:
        # Disable proxy environment lookup so local Ollama traffic never leaves loopback.
        return httpx.Client(
            base_url=self._base_url,
            timeout=timeout if timeout is not None else self._timeout,
            trust_env=False,
            transport=self._transport,
        )

    def _matches_local_tag(self, available_names: list[str]) -> bool:
        if self._model in available_names:
            return True
        target_base = self._model.split(":", 1)[0]
        return any(name.split(":", 1)[0] == target_base for name in available_names)

    def is_available(self) -> tuple[bool, Optional[str]]:
        if not self._model:
            return False, "No Ollama model configured. Set MINDVAULT_OLLAMA_MODEL."
        try:
            with self._make_client(timeout=min(2.0, self._timeout)) as client:
                resp = client.get("/api/tags")
            if resp.status_code != 200:
                return False, f"Local Ollama returned HTTP {resp.status_code} on /api/tags."
            payload = resp.json()
            models = payload.get("models") or []
            names = [str(m.get("name", "")) for m in models if isinstance(m, dict)]
            if not self._matches_local_tag(names):
                return (
                    False,
                    f"Ollama model '{self._model}' is not pulled locally (offline mode).",
                )
            return True, None
        except Exception as exc:
            return False, f"Local Ollama is unreachable at {self._base_url}: {exc}"

    def stream_generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.2,
        cancel_check: Optional[CancelCheck] = None,
    ) -> Iterator[str]:
        _check_cancelled(cancel_check)
        if not self._model:
            raise LLMUnavailableError("No Ollama model configured. Set MINDVAULT_OLLAMA_MODEL.")

        req_body = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": True,
            "options": {
                "temperature": float(temperature),
                "num_predict": max(1, int(max_tokens)),
            },
        }

        try:
            with self._make_client() as client:
                with client.stream("POST", "/api/chat", json=req_body) as resp:
                    if resp.status_code != 200:
                        detail = resp.read().decode("utf-8", errors="replace")
                        raise LLMGenerationError(
                            f"Local Ollama returned HTTP {resp.status_code}: {detail[:300]}"
                        )
                    for line in resp.iter_lines():
                        _check_cancelled(cancel_check)
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError as exc:
                            raise LLMGenerationError(
                                f"Invalid JSON frame from local Ollama: {exc}"
                            ) from exc
                        if "error" in data and data["error"]:
                            raise LLMGenerationError(str(data["error"]))
                        msg = data.get("message") or {}
                        piece = msg.get("content")
                        if isinstance(piece, str) and piece:
                            yield piece
                        if data.get("done") is True:
                            break
        except LLMCancelledError:
            raise
        except LLMError:
            raise
        except httpx.HTTPError as exc:
            raise LLMUnavailableError(
                f"Failed communicating with local Ollama at {self._base_url}: {exc}"
            ) from exc
        except Exception as exc:
            raise LLMGenerationError(f"Ollama generation failed: {exc}") from exc


def create_llm_provider(settings: Settings) -> BaseLLMProvider:
    """Instantiate the configured local LLM provider from Settings."""
    if settings.llm_provider == "ollama":
        return OllamaProvider(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            timeout_seconds=settings.ollama_timeout_seconds,
        )
    return LlamaCppProvider(
        model_path=settings.llm_model_path,
        models_dir=settings.models_dir,
        context_size=settings.llm_context_size,
        n_gpu_layers=settings.llm_n_gpu_layers,
    )
