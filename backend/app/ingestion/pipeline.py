"""Local 12-step document ingestion pipeline."""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from app.core.config import Settings
from app.db.repositories import DocumentRepository
from app.ingestion.adapters import AdapterRegistry
from app.ingestion.chunker import chunk_text, normalize_text
from app.ingestion.errors import (
    FileSizeLimitError,
    IngestionError,
)

if TYPE_CHECKING:
    from app.graph.service import GraphService
    from app.retrieval.service import RetrievalService

_SAFE_FILENAME_RE = re.compile(r"[^a-zA-Z0-9._-]+")


def sanitize_filename(raw_name: str) -> str:
    """Strip directory traversal components and sanitize unsafe characters."""
    leaf = Path(raw_name or "").name.strip()
    if not leaf or leaf in {".", ".."}:
        return "untitled"
    cleaned = _SAFE_FILENAME_RE.sub("_", leaf)
    return cleaned[:180] or "untitled"


def compute_sha256(data: bytes) -> str:
    """Return hex SHA-256 digest of raw file bytes."""
    return hashlib.sha256(data).hexdigest()


@dataclass
class IngestionResult:
    """Structured outcome returned after ingesting a single file."""

    document_id: Optional[str]
    filename: str
    file_type: str
    status: str
    duplicate: bool = False
    chunk_count: int = 0
    content_hash: Optional[str] = None
    epoch: Optional[int] = None
    source_path: Optional[str] = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    warnings: list[str] = field(default_factory=list)
    progress: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "file_type": self.file_type,
            "status": self.status,
            "duplicate": self.duplicate,
            "chunk_count": self.chunk_count,
            "content_hash": self.content_hash,
            "epoch": self.epoch,
            "source_path": self.source_path,
            "error": self.error,
            "error_code": self.error_code,
            "warnings": self.warnings,
            "progress": self.progress,
        }


class IngestionPipeline:
    """Coordinates validation, vault import, hashing, deduplication, extraction, and persistence."""

    def __init__(
        self,
        *,
        settings: Settings,
        repo: DocumentRepository,
        registry: Optional[AdapterRegistry] = None,
        retrieval_service: Optional[RetrievalService] = None,
        graph_service: Optional[GraphService] = None,
    ) -> None:
        self._settings = settings
        self._repo = repo
        self._registry = registry or AdapterRegistry.default(
            whisper_model_path=settings.whisper_model_path,
            tesseract_cmd=settings.tesseract_cmd,
        )
        self._retrieval_service = retrieval_service
        self._graph_service = graph_service

    def _store_in_vault(self, content: bytes, content_hash: str, safe_name: str) -> Path:
        """Persist raw bytes inside the managed vault directory."""
        self._settings.vault_dir.mkdir(parents=True, exist_ok=True)
        target = self._settings.vault_dir / f"{content_hash[:16]}_{safe_name}"
        if not target.exists():
            tmp_path = target.with_suffix(target.suffix + f".{uuid.uuid4().hex[:8]}.tmp")
            tmp_path.write_bytes(content)
            tmp_path.replace(target)
        return target

    def ingest_bytes(
        self,
        *,
        filename: str,
        content: bytes,
        epoch: Optional[int] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        extra_metadata: Optional[dict[str, Any]] = None,
    ) -> IngestionResult:
        """Execute the full 12-step local ingestion pipeline for one file."""
        progress: list[str] = []
        safe_name = sanitize_filename(filename)
        resolved_chunk_size = chunk_size if chunk_size is not None else self._settings.chunk_size
        resolved_overlap = (
            chunk_overlap if chunk_overlap is not None else self._settings.chunk_overlap
        )
        resolved_epoch = (
            int(epoch)
            if epoch is not None
            else int(datetime.now(timezone.utc).timestamp())
        )

        # 1. Validate file type and size
        try:
            adapter = self._registry.get_adapter(safe_name)
        except IngestionError as exc:
            return IngestionResult(
                document_id=None,
                filename=safe_name,
                file_type="unknown",
                status="failed",
                error=str(exc),
                error_code=exc.code,
                progress=progress,
            )

        if len(content) == 0:
            err = FileSizeLimitError(f"File '{safe_name}' is empty (0 bytes).")
            return IngestionResult(
                document_id=None,
                filename=safe_name,
                file_type=adapter.file_type,
                status="failed",
                error=str(err),
                error_code=err.code,
                progress=progress,
            )

        if len(content) > self._settings.max_upload_size_bytes:
            err = FileSizeLimitError(
                f"File '{safe_name}' ({len(content)} bytes) exceeds max allowed size "
                f"({self._settings.max_upload_size_bytes} bytes)."
            )
            return IngestionResult(
                document_id=None,
                filename=safe_name,
                file_type=adapter.file_type,
                status="failed",
                error=str(err),
                error_code=err.code,
                progress=progress,
            )

        if (
            resolved_chunk_size <= 0
            or resolved_overlap < 0
            or resolved_overlap >= resolved_chunk_size
        ):
            return IngestionResult(
                document_id=None,
                filename=safe_name,
                file_type=adapter.file_type,
                status="failed",
                error="Invalid chunking parameters: require 0 <= chunk_overlap < chunk_size.",
                error_code="invalid_chunk_config",
                progress=progress,
            )

        progress.append("validated")

        # 2 & 3. Calculate content hash & detect duplicate files
        content_hash = compute_sha256(content)
        progress.append("hashed")

        existing = self._repo.find_indexed_by_hash(content_hash)
        if existing is not None:
            progress.append("duplicate_detected")
            return IngestionResult(
                document_id=existing["id"],
                filename=existing["filename"],
                file_type=existing["file_type"],
                status="duplicate",
                duplicate=True,
                chunk_count=int(existing.get("chunk_count", 0)),
                content_hash=content_hash,
                epoch=resolved_epoch,
                source_path=existing["source_path"],
                warnings=["Identical content hash already indexed; skipped duplicate ingestion."],
                progress=progress,
            )
        progress.append("duplicate_checked")

        # 4. Import into managed vault storage
        managed_path = self._store_in_vault(content, content_hash, safe_name)
        progress.append("stored_in_vault")

        # Create initial document record in processing state
        doc_id = uuid.uuid4().hex
        now_iso = datetime.now(timezone.utc).isoformat()
        doc_metadata: dict[str, Any] = {
            "byte_size": len(content),
            "original_filename": filename,
            "epoch": resolved_epoch,
            "chunk_size": resolved_chunk_size,
            "chunk_overlap": resolved_overlap,
        }
        if extra_metadata:
            doc_metadata.update(extra_metadata)

        # Ensure unique (source_path, content_hash) if a prior failed attempt existed for same path
        source_path_str = str(managed_path)
        unique_source_path = source_path_str
        try:
            self._repo.create_document(
                doc_id=doc_id,
                filename=safe_name,
                file_type=adapter.file_type,
                source_path=unique_source_path,
                content_hash=content_hash,
                created_at=now_iso,
                updated_at=now_iso,
                status="processing",
                metadata=doc_metadata,
            )
        except Exception:
            # A failed document with the same (source_path, content_hash) may already exist;
            # disambiguate source_path with attempt id while keeping vault file clean.
            unique_source_path = f"{source_path_str}#{doc_id[:8]}"
            self._repo.create_document(
                doc_id=doc_id,
                filename=safe_name,
                file_type=adapter.file_type,
                source_path=unique_source_path,
                content_hash=content_hash,
                created_at=now_iso,
                updated_at=now_iso,
                status="processing",
                metadata=doc_metadata,
            )

        # 5. Extract text via adapter
        try:
            extracted = adapter.extract(managed_path)
        except IngestionError as exc:
            failed_at = datetime.now(timezone.utc).isoformat()
            doc_metadata["error"] = str(exc)
            doc_metadata["error_code"] = exc.code
            self._repo.update_document_status(
                doc_id=doc_id,
                status="failed",
                updated_at=failed_at,
                metadata=doc_metadata,
            )
            return IngestionResult(
                document_id=doc_id,
                filename=safe_name,
                file_type=adapter.file_type,
                status="failed",
                content_hash=content_hash,
                epoch=resolved_epoch,
                source_path=unique_source_path,
                error=str(exc),
                error_code=exc.code,
                progress=progress,
            )

        progress.append("extracted")
        doc_metadata.update(extracted.metadata)
        if extracted.warnings:
            doc_metadata["warnings"] = extracted.warnings

        # 6. Normalize text
        normalized = normalize_text(extracted.text)
        progress.append("normalized")

        # 7, 8, 9. Chunk text with indices, epoch, and source metadata
        chunk_source_meta = {
            "filename": safe_name,
            "file_type": extracted.file_type,
            "source_path": unique_source_path,
            "content_hash": content_hash,
        }
        chunks = chunk_text(
            normalized,
            chunk_size=resolved_chunk_size,
            chunk_overlap=resolved_overlap,
            source_metadata=chunk_source_meta,
        )
        progress.append("chunked")

        # 10 & 11. Persist chunks and mark document indexed atomically
        indexed_at = datetime.now(timezone.utc).isoformat()
        chunk_rows = [
            {
                "id": uuid.uuid4().hex,
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
                "token_count": chunk.token_count,
                "epoch": resolved_epoch,
                "created_at": indexed_at,
                "metadata": chunk.metadata,
            }
            for chunk in chunks
        ]
        doc_metadata["file_type"] = extracted.file_type
        doc_metadata["chunk_count"] = len(chunk_rows)

        self._repo.persist_chunks_and_index(
            doc_id=doc_id,
            file_type=extracted.file_type,
            chunks=chunk_rows,
            updated_at=indexed_at,
            metadata=doc_metadata,
        )
        progress.append("persisted")

        all_warnings = list(extracted.warnings)
        if self._retrieval_service is not None:
            index_warnings = self._retrieval_service.index_document_chunks(
                doc_id=doc_id,
                filename=safe_name,
                file_type=extracted.file_type,
                chunks=chunk_rows,
            )
            all_warnings.extend(index_warnings)
            progress.append("indexed_retrieval")

        if self._graph_service is not None:
            self._graph_service.on_document_indexed(
                doc_id=doc_id,
                filename=safe_name,
                chunks=chunk_rows,
            )
            progress.append("indexed_graph")

        # 12. Return ingestion summary
        return IngestionResult(
            document_id=doc_id,
            filename=safe_name,
            file_type=extracted.file_type,
            status="indexed",
            duplicate=False,
            chunk_count=len(chunk_rows),
            content_hash=content_hash,
            epoch=resolved_epoch,
            source_path=unique_source_path,
            warnings=all_warnings,
            progress=progress,
        )

    def delete_document_and_cleanup(self, doc_id: str) -> bool:
        """Delete document/chunks from SQLite and remove managed vault file if unreferenced."""
        chunk_ids = self._repo.get_chunk_ids_for_document(doc_id)
        belief_ids = self._repo.get_belief_ids_for_document(doc_id)
        deleted, source_path, remaining_refs = self._repo.delete_document(doc_id)
        if not deleted:
            return False

        if self._retrieval_service is not None:
            self._retrieval_service.remove_document(doc_id, chunk_ids)
        if self._graph_service is not None:
            self._graph_service.remove_entities([doc_id] + chunk_ids + belief_ids)

        if source_path and remaining_refs == 0:
            raw_path = source_path.split("#", 1)[0]
            candidate = Path(raw_path).resolve()
            vault_root = self._settings.vault_dir.resolve()
            # Only delete files strictly inside the managed vault directory.
            if vault_root in candidate.parents and candidate.is_file():
                try:
                    candidate.unlink()
                except OSError:
                    pass
        return True
