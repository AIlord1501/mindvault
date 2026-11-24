"""Structured error types for the local document ingestion pipeline."""

from __future__ import annotations


class IngestionError(Exception):
    """Base error for document ingestion failures."""

    code: str = "ingestion_error"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code


class UnsupportedFileTypeError(IngestionError):
    """Raised when an uploaded file extension is not supported."""

    code = "unsupported_file_type"


class FileSizeLimitError(IngestionError):
    """Raised when a file is empty or exceeds the configured size limit."""

    code = "file_size_limit"


class CorruptedFileError(IngestionError):
    """Raised when a file cannot be decoded or parsed safely."""

    code = "corrupted_file"


class AdapterUnavailableError(IngestionError):
    """Raised when an optional local extraction dependency or model is unavailable."""

    code = "adapter_unavailable"
