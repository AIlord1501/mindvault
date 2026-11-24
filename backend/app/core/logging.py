"""Structured JSON logging utilities for the MindVault backend."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Optional

_RESERVED_RECORD_FIELDS = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
    }
)


class JsonLogFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for key, value in record.__dict__.items():
            if key not in _RESERVED_RECORD_FIELDS and not key.startswith("_"):
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str, ensure_ascii=True)


def configure_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """Configure and return the root MindVault structured logger."""
    logger = logging.getLogger("mindvault")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    # Clear existing handlers to keep configuration idempotent across app instances.
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    formatter = JsonLogFormatter()

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            filename=str(log_file),
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def shutdown_logging() -> None:
    """Flush and close MindVault logger handlers cleanly on shutdown."""
    logger = logging.getLogger("mindvault")
    for handler in list(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)
