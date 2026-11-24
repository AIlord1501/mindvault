"""Parameterized SQLite repositories for documents and chunks."""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Optional

from app.db.database import DatabaseManager


def _decode_metadata(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, ValueError):
        return {}


def _row_to_document(row: sqlite3.Row) -> dict[str, Any]:
    keys = row.keys()
    doc: dict[str, Any] = {
        "id": row["id"],
        "filename": row["filename"],
        "file_type": row["file_type"],
        "source_path": row["source_path"],
        "content_hash": row["content_hash"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "status": row["status"],
        "metadata": _decode_metadata(row["metadata_json"]),
    }
    if "chunk_count" in keys:
        doc["chunk_count"] = int(row["chunk_count"])
    return doc


def _row_to_chunk(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "document_id": row["document_id"],
        "chunk_index": int(row["chunk_index"]),
        "content": row["content"],
        "token_count": int(row["token_count"]),
        "epoch": int(row["epoch"]),
        "created_at": row["created_at"],
        "metadata": _decode_metadata(row["metadata_json"]),
    }


def _row_to_indexed_chunk(row: sqlite3.Row) -> dict[str, Any]:
    item = _row_to_chunk(row)
    item["filename"] = row["filename"]
    item["file_type"] = row["file_type"]
    item["source_path"] = row["source_path"]
    item["content_hash"] = row["content_hash"]
    return item


class DocumentRepository:
    """Data-access layer for `documents` and `chunks` tables."""

    def __init__(self, db: DatabaseManager) -> None:
        self._db = db

    def find_indexed_by_hash(self, content_hash: str) -> Optional[dict[str, Any]]:
        """Return an existing indexed document matching content_hash, if any."""
        with self._db.connection() as conn:
            row = conn.execute(
                """
                SELECT d.*, COUNT(c.id) AS chunk_count
                FROM documents d
                LEFT JOIN chunks c ON c.document_id = d.id
                WHERE d.content_hash = ? AND d.status = 'indexed'
                GROUP BY d.id
                ORDER BY d.created_at DESC
                LIMIT 1
                """,
                (content_hash,),
            ).fetchone()
            return _row_to_document(row) if row is not None else None

    def create_document(
        self,
        *,
        doc_id: str,
        filename: str,
        file_type: str,
        source_path: str,
        content_hash: str,
        created_at: str,
        updated_at: str,
        status: str,
        metadata: dict[str, Any],
    ) -> None:
        """Insert a new document record."""
        metadata_json = json.dumps(metadata, ensure_ascii=True)
        with self._db.transaction(immediate=True) as conn:
            conn.execute(
                """
                INSERT INTO documents (
                    id, filename, file_type, source_path, content_hash,
                    created_at, updated_at, status, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc_id,
                    filename,
                    file_type,
                    source_path,
                    content_hash,
                    created_at,
                    updated_at,
                    status,
                    metadata_json,
                ),
            )

    def update_document_status(
        self,
        *,
        doc_id: str,
        status: str,
        updated_at: str,
        metadata: dict[str, Any],
    ) -> None:
        """Update document status and metadata."""
        metadata_json = json.dumps(metadata, ensure_ascii=True)
        with self._db.transaction(immediate=True) as conn:
            conn.execute(
                """
                UPDATE documents
                SET status = ?, updated_at = ?, metadata_json = ?
                WHERE id = ?
                """,
                (status, updated_at, metadata_json, doc_id),
            )

    def persist_chunks_and_index(
        self,
        *,
        doc_id: str,
        file_type: str,
        chunks: list[dict[str, Any]],
        updated_at: str,
        metadata: dict[str, Any],
    ) -> None:
        """Atomically insert all chunks and mark the parent document indexed."""
        metadata_json = json.dumps(metadata, ensure_ascii=True)
        with self._db.transaction(immediate=True) as conn:
            conn.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
            if chunks:
                conn.executemany(
                    """
                    INSERT INTO chunks (
                        id, document_id, chunk_index, content, token_count,
                        epoch, created_at, metadata_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            item["id"],
                            doc_id,
                            int(item["chunk_index"]),
                            item["content"],
                            int(item["token_count"]),
                            int(item["epoch"]),
                            item["created_at"],
                            json.dumps(item["metadata"], ensure_ascii=True),
                        )
                        for item in chunks
                    ],
                )
            conn.execute(
                """
                UPDATE documents
                SET file_type = ?, status = 'indexed', updated_at = ?, metadata_json = ?
                WHERE id = ?
                """,
                (file_type, updated_at, metadata_json, doc_id),
            )

    def list_documents(
        self,
        *,
        status: Optional[str] = None,
        file_type: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """List documents with optional status/file_type filters and chunk counts."""
        where_clauses: list[str] = []
        params: list[Any] = []

        if status is not None:
            where_clauses.append("d.status = ?")
            params.append(status)
        if file_type is not None:
            where_clauses.append("d.file_type = ?")
            params.append(file_type)

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        with self._db.connection() as conn:
            total_row = conn.execute(
                f"SELECT COUNT(*) AS total FROM documents d {where_sql}",
                tuple(params),
            ).fetchone()
            total = int(total_row["total"]) if total_row is not None else 0

            rows = conn.execute(
                f"""
                SELECT d.*, COUNT(c.id) AS chunk_count
                FROM documents d
                LEFT JOIN chunks c ON c.document_id = d.id
                {where_sql}
                GROUP BY d.id
                ORDER BY d.created_at DESC
                LIMIT ? OFFSET ?
                """,
                tuple(params + [limit, offset]),
            ).fetchall()
            return [_row_to_document(row) for row in rows], total

    def get_document_with_chunks(self, doc_id: str) -> Optional[dict[str, Any]]:
        """Fetch a single document along with its ordered chunks."""
        with self._db.connection() as conn:
            doc_row = conn.execute(
                "SELECT * FROM documents WHERE id = ?",
                (doc_id,),
            ).fetchone()
            if doc_row is None:
                return None

            chunk_rows = conn.execute(
                """
                SELECT * FROM chunks
                WHERE document_id = ?
                ORDER BY chunk_index ASC
                """,
                (doc_id,),
            ).fetchall()

            doc = _row_to_document(doc_row)
            doc["chunks"] = [_row_to_chunk(row) for row in chunk_rows]
            doc["chunk_count"] = len(doc["chunks"])
            return doc

    def delete_document(self, doc_id: str) -> tuple[bool, Optional[str], int]:
        """Delete a document (cascading to chunks) and return (deleted, source_path, refs)."""
        with self._db.transaction(immediate=True) as conn:
            row = conn.execute(
                "SELECT source_path FROM documents WHERE id = ?",
                (doc_id,),
            ).fetchone()
            if row is None:
                return False, None, 0

            source_path = str(row["source_path"])
            conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))

            ref_row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM documents WHERE source_path = ?",
                (source_path,),
            ).fetchone()
            remaining_refs = int(ref_row["cnt"]) if ref_row is not None else 0
            return True, source_path, remaining_refs

    def list_indexed_chunks(
        self,
        *,
        document_ids: Optional[list[str]] = None,
        file_types: Optional[list[str]] = None,
        epoch_min: Optional[int] = None,
        epoch_max: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Return indexed chunks joined with parent document metadata."""
        if document_ids is not None and len(document_ids) == 0:
            return []
        if file_types is not None and len(file_types) == 0:
            return []

        where_clauses = ["d.status = 'indexed'"]
        params: list[Any] = []

        if document_ids:
            placeholders = ", ".join("?" for _ in document_ids)
            where_clauses.append(f"c.document_id IN ({placeholders})")
            params.extend(document_ids)
        if file_types:
            placeholders = ", ".join("?" for _ in file_types)
            where_clauses.append(f"d.file_type IN ({placeholders})")
            params.extend(file_types)
        if epoch_min is not None:
            where_clauses.append("c.epoch >= ?")
            params.append(int(epoch_min))
        if epoch_max is not None:
            where_clauses.append("c.epoch <= ?")
            params.append(int(epoch_max))

        where_sql = " AND ".join(where_clauses)
        with self._db.connection() as conn:
            rows = conn.execute(
                f"""
                SELECT
                    c.*,
                    d.filename AS filename,
                    d.file_type AS file_type,
                    d.source_path AS source_path,
                    d.content_hash AS content_hash
                FROM chunks c
                INNER JOIN documents d ON d.id = c.document_id
                WHERE {where_sql}
                ORDER BY c.created_at ASC, c.chunk_index ASC
                """,
                tuple(params),
            ).fetchall()
            return [_row_to_indexed_chunk(row) for row in rows]

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Hydrate indexed chunks by chunk ID with parent document metadata."""
        if not chunk_ids:
            return {}

        unique_ids = list(dict.fromkeys(chunk_ids))
        result: dict[str, dict[str, Any]] = {}

        # Batch IN clauses safely below SQLite's parameter limit.
        with self._db.connection() as conn:
            for start in range(0, len(unique_ids), 500):
                batch = unique_ids[start : start + 500]
                placeholders = ", ".join("?" for _ in batch)
                rows = conn.execute(
                    f"""
                    SELECT
                        c.*,
                        d.filename AS filename,
                        d.file_type AS file_type,
                        d.source_path AS source_path,
                        d.content_hash AS content_hash
                    FROM chunks c
                    INNER JOIN documents d ON d.id = c.document_id
                    WHERE d.status = 'indexed' AND c.id IN ({placeholders})
                    """,
                    tuple(batch),
                ).fetchall()
                for row in rows:
                    item = _row_to_indexed_chunk(row)
                    result[item["id"]] = item
        return result

    def get_chunk_ids_for_document(self, doc_id: str) -> list[str]:
        """Return all chunk IDs associated with a document ID."""
        with self._db.connection() as conn:
            rows = conn.execute(
                "SELECT id FROM chunks WHERE document_id = ? ORDER BY chunk_index ASC",
                (doc_id,),
            ).fetchall()
            return [str(row["id"]) for row in rows]

    def count_indexed_documents_and_chunks(self) -> tuple[int, int]:
        """Return counts of indexed documents and chunks in SQLite."""
        with self._db.connection() as conn:
            doc_row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM documents WHERE status = 'indexed'"
            ).fetchone()
            chunk_row = conn.execute(
                """
                SELECT COUNT(*) AS cnt
                FROM chunks c
                INNER JOIN documents d ON d.id = c.document_id
                WHERE d.status = 'indexed'
                """
            ).fetchone()
            docs = int(doc_row["cnt"]) if doc_row is not None else 0
            chunks = int(chunk_row["cnt"]) if chunk_row is not None else 0
            return docs, chunks
