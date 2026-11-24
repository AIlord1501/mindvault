"""Parameterized SQLite repository for retention tracking and spaced repetition history."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import timedelta
from typing import Any, Optional

from app.db.database import DatabaseManager
from app.scheduler.models import format_iso_datetime, parse_iso_datetime


def _decode_metadata(raw: Optional[str]) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, ValueError):
        return {}


class RetentionRepository:
    """Data-access layer for `retention_scores` and `review_history` tables."""

    def __init__(self, db: DatabaseManager) -> None:
        self._db = db

    def ensure_chunk_retention_rows(
        self,
        chunks: list[dict[str, Any]],
        *,
        initial_stability: float,
        interval_days: float,
        default_retention: float = 1.0,
    ) -> int:
        """Create initial retention_scores rows for newly indexed chunks if absent."""
        rows: list[tuple[str, str, float, float, Optional[str], str, int]] = []
        for chunk in chunks:
            chunk_id = str(chunk.get("id") or "").strip()
            created_at = str(chunk.get("created_at") or "").strip()
            if not chunk_id or not created_at:
                continue
            created_dt = parse_iso_datetime(created_at)
            next_review_iso = format_iso_datetime(created_dt + timedelta(days=interval_days))
            rows.append(
                (
                    uuid.uuid4().hex,
                    chunk_id,
                    float(initial_stability),
                    float(default_retention),
                    None,
                    next_review_iso,
                    0,
                )
            )

        if not rows:
            return 0

        with self._db.transaction(immediate=True) as conn:
            conn.executemany(
                """
                INSERT INTO retention_scores (
                    id, chunk_id, strength, retention,
                    last_reviewed_at, next_review_at, review_count
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (chunk_id) DO NOTHING
                """,
                rows,
            )
        return len(rows)

    def sync_missing_retention_rows(
        self,
        *,
        initial_stability: float,
        interval_days: float,
    ) -> int:
        """Backfill retention_scores rows for any indexed chunks lacking one."""
        with self._db.connection() as conn:
            missing = conn.execute(
                """
                SELECT c.id, c.created_at
                FROM chunks c
                INNER JOIN documents d ON d.id = c.document_id
                LEFT JOIN retention_scores rs ON rs.chunk_id = c.id
                WHERE d.status = 'indexed' AND rs.id IS NULL
                ORDER BY c.created_at ASC, c.chunk_index ASC
                """
            ).fetchall()

        if not missing:
            return 0

        chunks = [{"id": row["id"], "created_at": row["created_at"]} for row in missing]
        return self.ensure_chunk_retention_rows(
            chunks,
            initial_stability=initial_stability,
            interval_days=interval_days,
        )

    def list_chunk_retention_records(
        self,
        *,
        document_id: Optional[str] = None,
        chunk_id: Optional[str] = None,
        epoch_min: Optional[int] = None,
        epoch_max: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Load indexed chunks paired with retention state and active beliefs."""
        clauses: list[str] = ["d.status = 'indexed'"]
        params: list[Any] = []

        if document_id is not None:
            clauses.append("c.document_id = ?")
            params.append(document_id)
        if chunk_id is not None:
            clauses.append("c.id = ?")
            params.append(chunk_id)
        if epoch_min is not None:
            clauses.append("c.epoch >= ?")
            params.append(int(epoch_min))
        if epoch_max is not None:
            clauses.append("c.epoch <= ?")
            params.append(int(epoch_max))

        where_sql = " AND ".join(clauses)
        with self._db.connection() as conn:
            rows = conn.execute(
                f"""
                SELECT
                    rs.id AS score_id,
                    c.id AS chunk_id,
                    c.document_id,
                    d.filename,
                    d.file_type,
                    c.chunk_index,
                    c.content,
                    c.token_count,
                    c.epoch,
                    c.created_at AS chunk_created_at,
                    c.metadata_json,
                    rs.strength AS stability,
                    rs.retention AS stored_retention,
                    rs.last_reviewed_at,
                    rs.next_review_at,
                    rs.review_count
                FROM chunks c
                INNER JOIN documents d ON d.id = c.document_id
                LEFT JOIN retention_scores rs ON rs.chunk_id = c.id
                WHERE {where_sql}
                ORDER BY c.epoch ASC, c.chunk_index ASC, c.id ASC
                """,
                tuple(params),
            ).fetchall()

            chunk_ids = [row["chunk_id"] for row in rows]
            beliefs_by_chunk = self._load_beliefs_by_chunk(conn, chunk_ids)

        records: list[dict[str, Any]] = []
        for row in rows:
            cid = row["chunk_id"]
            records.append(
                {
                    "score_id": row["score_id"],
                    "chunk_id": cid,
                    "document_id": row["document_id"],
                    "filename": row["filename"],
                    "file_type": row["file_type"],
                    "chunk_index": int(row["chunk_index"]),
                    "content": row["content"],
                    "token_count": int(row["token_count"]),
                    "epoch": int(row["epoch"]),
                    "chunk_created_at": row["chunk_created_at"],
                    "metadata": _decode_metadata(row["metadata_json"]),
                    "stability": float(row["stability"]) if row["stability"] is not None else None,
                    "stored_retention": (
                        float(row["stored_retention"])
                        if row["stored_retention"] is not None
                        else None
                    ),
                    "last_reviewed_at": row["last_reviewed_at"],
                    "next_review_at": row["next_review_at"],
                    "review_count": (
                        int(row["review_count"]) if row["review_count"] is not None else 0
                    ),
                    "beliefs": beliefs_by_chunk.get(cid, []),
                }
            )
        return records

    @staticmethod
    def _load_beliefs_by_chunk(
        conn: sqlite3.Connection,
        chunk_ids: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        beliefs_map: dict[str, list[dict[str, Any]]] = {}
        if not chunk_ids:
            return beliefs_map

        for start in range(0, len(chunk_ids), 400):
            batch = chunk_ids[start : start + 400]
            placeholders = ", ".join("?" for _ in batch)
            belief_rows = conn.execute(
                f"""
                SELECT id, chunk_id, belief_text, confidence, status
                FROM beliefs
                WHERE chunk_id IN ({placeholders}) AND status = 'active'
                ORDER BY confidence DESC, created_at ASC
                """,
                tuple(batch),
            ).fetchall()
            for b_row in belief_rows:
                beliefs_map.setdefault(b_row["chunk_id"], []).append(
                    {
                        "id": b_row["id"],
                        "belief_text": b_row["belief_text"],
                        "confidence": float(b_row["confidence"]),
                        "status": b_row["status"],
                    }
                )
        return beliefs_map

    def batch_update_live_retention(
        self,
        updates: list[tuple[float, Optional[str], str]],
    ) -> None:
        """Persist recalculated live retention scores (and missing next_review_at) by score ID."""
        if not updates:
            return
        with self._db.transaction(immediate=True) as conn:
            conn.executemany(
                """
                UPDATE retention_scores
                SET retention = ?,
                    next_review_at = COALESCE(next_review_at, ?)
                WHERE id = ?
                """,
                updates,
            )

    def record_review(
        self,
        *,
        chunk_id: str,
        outcome: str,
        rating: int,
        previous_stability: float,
        new_stability: float,
        retention_before: float,
        elapsed_days: float,
        reviewed_at: str,
        next_review_at: str,
        review_count: int,
        notes: Optional[str] = None,
    ) -> dict[str, Any]:
        """Atomically update retention_scores and append a review_history entry."""
        review_id = uuid.uuid4().hex
        with self._db.transaction(immediate=True) as conn:
            existing = conn.execute(
                "SELECT id FROM retention_scores WHERE chunk_id = ?",
                (chunk_id,),
            ).fetchone()
            if existing is None:
                conn.execute(
                    """
                    INSERT INTO retention_scores (
                        id, chunk_id, strength, retention,
                        last_reviewed_at, next_review_at, review_count
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        uuid.uuid4().hex,
                        chunk_id,
                        float(new_stability),
                        1.0,
                        reviewed_at,
                        next_review_at,
                        int(review_count),
                    ),
                )
            else:
                conn.execute(
                    """
                    UPDATE retention_scores
                    SET strength = ?,
                        retention = 1.0,
                        last_reviewed_at = ?,
                        next_review_at = ?,
                        review_count = ?
                    WHERE chunk_id = ?
                    """,
                    (
                        float(new_stability),
                        reviewed_at,
                        next_review_at,
                        int(review_count),
                        chunk_id,
                    ),
                )

            conn.execute(
                """
                INSERT INTO review_history (
                    id, chunk_id, outcome, rating, previous_stability, new_stability,
                    retention_before, elapsed_days, next_review_at, reviewed_at, notes
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    review_id,
                    chunk_id,
                    outcome,
                    int(rating),
                    float(previous_stability),
                    float(new_stability),
                    float(retention_before),
                    float(elapsed_days),
                    next_review_at,
                    reviewed_at,
                    notes,
                ),
            )

        return {
            "review_id": review_id,
            "chunk_id": chunk_id,
            "outcome": outcome,
            "rating": int(rating),
            "previous_stability": float(previous_stability),
            "new_stability": float(new_stability),
            "retention_before": float(retention_before),
            "elapsed_days": float(elapsed_days),
            "next_review_at": next_review_at,
            "reviewed_at": reviewed_at,
            "notes": notes,
        }

    def list_review_history(
        self,
        *,
        chunk_id: Optional[str] = None,
        document_id: Optional[str] = None,
        outcome: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """List spaced repetition review events joined with chunk and document metadata."""
        clauses: list[str] = ["1=1"]
        params: list[Any] = []

        if chunk_id is not None:
            clauses.append("rh.chunk_id = ?")
            params.append(chunk_id)
        if document_id is not None:
            clauses.append("c.document_id = ?")
            params.append(document_id)
        if outcome is not None:
            clauses.append("rh.outcome = ?")
            params.append(outcome)

        where_sql = " AND ".join(clauses)
        with self._db.connection() as conn:
            total = int(
                conn.execute(
                    f"""
                    SELECT COUNT(*) AS cnt
                    FROM review_history rh
                    INNER JOIN chunks c ON c.id = rh.chunk_id
                    INNER JOIN documents d ON d.id = c.document_id
                    WHERE {where_sql}
                    """,
                    tuple(params),
                ).fetchone()["cnt"]
            )
            rows = conn.execute(
                f"""
                SELECT
                    rh.id,
                    rh.chunk_id,
                    c.document_id,
                    d.filename,
                    c.chunk_index,
                    c.epoch,
                    rh.outcome,
                    rh.rating,
                    rh.previous_stability,
                    rh.new_stability,
                    rh.retention_before,
                    rh.elapsed_days,
                    rh.next_review_at,
                    rh.reviewed_at,
                    rh.notes
                FROM review_history rh
                INNER JOIN chunks c ON c.id = rh.chunk_id
                INNER JOIN documents d ON d.id = c.document_id
                WHERE {where_sql}
                ORDER BY rh.reviewed_at DESC, rh.rowid DESC
                LIMIT ? OFFSET ?
                """,
                tuple(params + [int(limit), int(offset)]),
            ).fetchall()

        items = [
            {
                "id": row["id"],
                "chunk_id": row["chunk_id"],
                "document_id": row["document_id"],
                "filename": row["filename"],
                "chunk_index": int(row["chunk_index"]),
                "epoch": int(row["epoch"]),
                "outcome": row["outcome"],
                "rating": int(row["rating"]),
                "previous_stability": float(row["previous_stability"]),
                "new_stability": float(row["new_stability"]),
                "retention_before": float(row["retention_before"]),
                "elapsed_days": float(row["elapsed_days"]),
                "next_review_at": row["next_review_at"],
                "reviewed_at": row["reviewed_at"],
                "notes": row["notes"],
            }
            for row in rows
        ]
        return items, total
