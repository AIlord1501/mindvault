"""Minimal local launcher for the MindVault FastAPI backend."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the local MindVault launcher."""
    parser = argparse.ArgumentParser(description="Run the local MindVault backend server.")
    parser.add_argument(
        "--host",
        default=None,
        help="Loopback bind host (must be 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Loopback bind port (default: MINDVAULT_PORT or 8765).",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable Uvicorn auto-reload for local development.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate configuration and exit without starting the server.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Validate settings and optionally start Uvicorn bound to 127.0.0.1."""
    if str(BACKEND_DIR) not in sys.path:
        sys.path.insert(0, str(BACKEND_DIR))
    from app.core.config import Settings

    args = parse_args(argv)
    base = Settings.from_env()
    settings = Settings(
        app_name=base.app_name,
        version=base.version,
        host=args.host if args.host is not None else base.host,
        port=args.port if args.port is not None else base.port,
        data_dir=base.data_dir,
        sqlite_path=base.sqlite_path,
        chroma_dir=base.chroma_dir,
        models_dir=base.models_dir,
        embedding_model=base.embedding_model,
        llm_model_path=base.llm_model_path,
        offline_only=base.offline_only,
    )

    if args.check:
        summary = {
            "status": "ok",
            "host": settings.host,
            "port": settings.port,
            "offline_only": settings.offline_only,
            "data_dir": str(settings.data_dir),
        }
        print(json.dumps(summary))
        return 0

    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=args.reload,
        app_dir=str(BACKEND_DIR),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
