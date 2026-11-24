"""Modular local extraction adapters for PDF, Markdown/TXT, Image OCR, and offline Audio."""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from app.ingestion.errors import (
    AdapterUnavailableError,
    CorruptedFileError,
    UnsupportedFileTypeError,
)


@dataclass(frozen=True)
class ExtractedDocument:
    """Normalized adapter extraction result before chunking."""

    text: str
    file_type: str
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


class BaseExtractionAdapter(ABC):
    """Contract implemented by all local file extraction adapters."""

    supported_extensions: frozenset[str] = frozenset()
    file_type: str = "unknown"

    @abstractmethod
    def extract(self, file_path: Path) -> ExtractedDocument:
        """Extract text and metadata from a local file treated as untrusted input."""


class TextMarkdownAdapter(BaseExtractionAdapter):
    """Extracts plain text and Markdown files locally."""

    supported_extensions = frozenset({".txt", ".md", ".markdown"})
    file_type = "text"

    def extract(self, file_path: Path) -> ExtractedDocument:
        suffix = file_path.suffix.lower()
        resolved_type = "markdown" if suffix in {".md", ".markdown"} else "txt"
        try:
            raw = file_path.read_bytes()
        except OSError as exc:
            raise CorruptedFileError(f"Failed to read file '{file_path.name}': {exc}") from exc

        warnings: list[str] = []
        try:
            text = raw.decode("utf-8")
            encoding = "utf-8"
        except UnicodeDecodeError:
            text = raw.decode("utf-8", errors="replace")
            encoding = "utf-8-replace"
            warnings.append("File contained invalid UTF-8 byte sequences; replaced invalid bytes.")

        return ExtractedDocument(
            text=text,
            file_type=resolved_type,
            metadata={"encoding": encoding, "byte_size": len(raw)},
            warnings=warnings,
        )


class PdfAdapter(BaseExtractionAdapter):
    """Extracts text from PDF documents using PyMuPDF (fitz)."""

    supported_extensions = frozenset({".pdf"})
    file_type = "pdf"

    def __init__(self, loader: Optional[Callable[[], Any]] = None) -> None:
        self._loader = loader

    def _load_fitz(self) -> Any:
        if self._loader is not None:
            return self._loader()
        try:
            return importlib.import_module("fitz")
        except ImportError as exc:
            raise AdapterUnavailableError(
                "PDF extraction requires PyMuPDF (`pip install PyMuPDF`)."
            ) from exc

    def extract(self, file_path: Path) -> ExtractedDocument:
        fitz = self._load_fitz()
        pages_text: list[str] = []
        warnings: list[str] = []

        try:
            doc = fitz.open(str(file_path))
        except Exception as exc:
            raise CorruptedFileError(f"Unable to open PDF '{file_path.name}': {exc}") from exc

        try:
            if getattr(doc, "is_encrypted", False):
                raise CorruptedFileError(
                    f"PDF '{file_path.name}' is password-protected and cannot be extracted."
                )
            page_count = len(doc)
            for idx, page in enumerate(doc):
                page_text = (page.get_text("text") or "").strip()
                if page_text:
                    pages_text.append(page_text)
                else:
                    warnings.append(f"PDF page {idx + 1} contained no extractable text.")
        except CorruptedFileError:
            raise
        except Exception as exc:
            raise CorruptedFileError(
                f"Failed extracting text from PDF '{file_path.name}': {exc}"
            ) from exc
        finally:
            close_fn = getattr(doc, "close", None)
            if callable(close_fn):
                close_fn()

        return ExtractedDocument(
            text="\n\n".join(pages_text),
            file_type="pdf",
            metadata={"page_count": page_count, "extractor": "pymupdf"},
            warnings=warnings,
        )


class ImageOcrAdapter(BaseExtractionAdapter):
    """Extracts text from local images using Pillow and optional pytesseract."""

    supported_extensions = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"})
    file_type = "image"

    def __init__(
        self,
        tesseract_cmd: str = "",
        pil_loader: Optional[Callable[[], Any]] = None,
        pytesseract_loader: Optional[Callable[[], Any]] = None,
    ) -> None:
        self._tesseract_cmd = tesseract_cmd.strip()
        self._pil_loader = pil_loader
        self._pytesseract_loader = pytesseract_loader

    def _load_pil_image(self) -> Any:
        if self._pil_loader is not None:
            return self._pil_loader()
        try:
            return importlib.import_module("PIL.Image")
        except ImportError as exc:
            raise AdapterUnavailableError(
                "Image OCR requires Pillow (`pip install Pillow`)."
            ) from exc

    def _load_pytesseract(self) -> Any:
        if self._pytesseract_loader is not None:
            return self._pytesseract_loader()
        try:
            return importlib.import_module("pytesseract")
        except ImportError as exc:
            raise AdapterUnavailableError(
                "Image OCR is unavailable because optional dependency `pytesseract` "
                "is not installed."
            ) from exc

    def extract(self, file_path: Path) -> ExtractedDocument:
        image_mod = self._load_pil_image()
        pytesseract = self._load_pytesseract()

        if self._tesseract_cmd and hasattr(pytesseract, "pytesseract"):
            pytesseract.pytesseract.tesseract_cmd = self._tesseract_cmd

        try:
            with image_mod.open(str(file_path)) as img:
                width, height = img.size
                img_format = img.format or file_path.suffix.lstrip(".").upper()
                try:
                    text = pytesseract.image_to_string(img) or ""
                except Exception as exc:
                    exc_name = type(exc).__name__
                    if "TesseractNotFound" in exc_name or isinstance(exc, FileNotFoundError):
                        raise AdapterUnavailableError(
                            "Local Tesseract OCR binary was not found. "
                            "Install Tesseract or set MINDVAULT_TESSERACT_CMD."
                        ) from exc
                    raise CorruptedFileError(
                        f"OCR extraction failed for '{file_path.name}': {exc}"
                    ) from exc
        except (AdapterUnavailableError, CorruptedFileError):
            raise
        except Exception as exc:
            raise CorruptedFileError(
                f"Unable to decode image '{file_path.name}': {exc}"
            ) from exc

        warnings: list[str] = []
        if not text.strip():
            warnings.append("OCR completed but detected no readable text in image.")

        return ExtractedDocument(
            text=text,
            file_type="image",
            metadata={
                "width": width,
                "height": height,
                "image_format": img_format,
                "extractor": "pytesseract",
            },
            warnings=warnings,
        )


class AudioWhisperAdapter(BaseExtractionAdapter):
    """Transcribes local audio files using offline Whisper without network downloads."""

    supported_extensions = frozenset({".wav", ".mp3", ".m4a", ".ogg", ".flac"})
    file_type = "audio"

    def __init__(
        self,
        model_path: str = "",
        backend_loader: Optional[Callable[[], tuple[str, Any]]] = None,
    ) -> None:
        self._model_path = model_path.strip()
        self._backend_loader = backend_loader

    def _resolve_backend(self) -> tuple[str, Any]:
        if self._backend_loader is not None:
            return self._backend_loader()
        try:
            return "faster-whisper", importlib.import_module("faster_whisper")
        except ImportError:
            pass
        try:
            return "openai-whisper", importlib.import_module("whisper")
        except ImportError as exc:
            raise AdapterUnavailableError(
                "Offline audio transcription is unavailable because neither `faster-whisper` "
                "nor `openai-whisper` is installed."
            ) from exc

    def _require_local_model_path(self) -> Path:
        if not self._model_path:
            raise AdapterUnavailableError(
                "Offline audio transcription requires a local Whisper model path configured via "
                "MINDVAULT_WHISPER_MODEL_PATH (network downloads are disabled)."
            )
        resolved = Path(self._model_path).expanduser()
        if not resolved.exists():
            raise AdapterUnavailableError(
                f"Configured local Whisper model path does not exist: {resolved}"
            )
        return resolved

    def extract(self, file_path: Path) -> ExtractedDocument:
        backend_name, module = self._resolve_backend()
        model_path = self._require_local_model_path()

        try:
            if backend_name == "faster-whisper":
                model = module.WhisperModel(str(model_path), local_files_only=True)
                segments, info = model.transcribe(str(file_path))
                transcript = "\n".join(
                    seg.text.strip() for seg in segments if getattr(seg, "text", "").strip()
                )
                language = getattr(info, "language", None)
                duration = getattr(info, "duration", None)
            else:
                model = module.load_model(str(model_path), download_root=str(model_path.parent))
                result = model.transcribe(str(file_path), fp16=False)
                transcript = str(result.get("text", "")).strip()
                language = result.get("language")
                duration = None
        except (AdapterUnavailableError, CorruptedFileError):
            raise
        except Exception as exc:
            raise CorruptedFileError(
                f"Offline audio transcription failed for '{file_path.name}': {exc}"
            ) from exc

        warnings: list[str] = []
        if not transcript.strip():
            warnings.append("Audio transcription completed with empty transcript.")

        metadata: dict[str, Any] = {
            "extractor": backend_name,
            "model_path": str(model_path),
        }
        if language is not None:
            metadata["language"] = language
        if duration is not None:
            metadata["duration_seconds"] = duration

        return ExtractedDocument(
            text=transcript,
            file_type="audio",
            metadata=metadata,
            warnings=warnings,
        )


class AdapterRegistry:
    """Dispatches local files to the appropriate extraction adapter by file extension."""

    def __init__(self, adapters: list[BaseExtractionAdapter]) -> None:
        self._by_ext: dict[str, BaseExtractionAdapter] = {}
        for adapter in adapters:
            for ext in adapter.supported_extensions:
                self._by_ext[ext.lower()] = adapter

    @classmethod
    def default(
        cls,
        *,
        whisper_model_path: str = "",
        tesseract_cmd: str = "",
    ) -> AdapterRegistry:
        return cls(
            [
                TextMarkdownAdapter(),
                PdfAdapter(),
                ImageOcrAdapter(tesseract_cmd=tesseract_cmd),
                AudioWhisperAdapter(model_path=whisper_model_path),
            ]
        )

    @property
    def supported_extensions(self) -> frozenset[str]:
        return frozenset(self._by_ext.keys())

    def get_adapter(self, filename: str) -> BaseExtractionAdapter:
        ext = Path(filename).suffix.lower()
        adapter = self._by_ext.get(ext)
        if adapter is None:
            supported = ", ".join(sorted(self._by_ext.keys()))
            raise UnsupportedFileTypeError(
                f"Unsupported file extension '{ext or '(none)'}'. Supported: {supported}"
            )
        return adapter
