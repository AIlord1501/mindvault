"""Unit tests for SQLite connection management, migrations, schema, and constraints."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from app.db.database import DatabaseManager
from app.db.migrations import MigrationRunner

EXPECTED_SCHEMA: dict[str, set[str]] = {
    "documents": {
        "id",
        "filename",
        "file_type",
        "source_path",
        "content_hash",
        "created_at",
        "updated_at",
        "status",
        "metadata_json",
    },
    "chunks": {
        "id",
        "document_id",
        "chunk_index",
        "content",
        "token_count",
        "epoch",
        "created_at",
        "metadata_json",
    },
    "beliefs": {
        "id",
        "chunk_id",
        "belief_text",
        "normalized_text",
        "confidence",
        "epoch",
        "status",
        "created_at",
        "superseded_by_id",
        "context_note",
        "embedding_json",
        "updated_at",
    },
    "contradictions": {
        "id",
        "belief_a_id",
        "belief_b_id",
        "score",
        "status",
        "resolution",
        "created_at",
        "resolved_at",
        "entailment_score",
        "neutral_score",
        "similarity_score",
        "resolution_note",
    },
    "belief_relations": {
        "id",
        "source_belief_id",
        "target_belief_id",
        "label",
        "contradiction_score",
        "entailment_score",
        "neutral_score",
        "similarity_score",
        "created_at",
    },
    "graph_edges": {
        "id",
        "source_id",
        "target_id",
        "edge_type",
        "weight",
        "created_at",
    },
    "retention_scores": {
        "id",
        "chunk_id",
        "strength",
        "retention",
        "last_reviewed_at",
        "next_review_at",
        "review_count",
    },
    "review_history": {
        "id",
        "chunk_id",
        "outcome",
        "rating",
        "previous_stability",
        "new_stability",
        "retention_before",
        "elapsed_days",
        "next_review_at",
        "reviewed_at",
        "notes",
    },
    "citations": {
        "id",
        "query_id",
        "chunk_id",
        "citation_type",
        "created_at",
    },
    "morning_briefs": {
        "id",
        "brief_date",
        "content",
        "metadata_json",
        "created_at",
    },
}


def test_migrations_create_expected_tables_columns_and_indexes(tmp_path: Path) -> None:
    """Initial migration should create all 8 domain tables with required columns and indexes."""
    db = DatabaseManager(tmp_path / "mindvault.db")
    try:
        runner = MigrationRunner(db)
        applied_first = runner.apply_pending()
        assert applied_first == [1, 2, 3]
        assert runner.current_version() == 3

        # Running migrations again must be idempotent.
        assert runner.apply_pending() == []

        with db.connection() as conn:
            tables = {
                row["name"]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
            for table_name, expected_cols in EXPECTED_SCHEMA.items():
                assert table_name in tables, f"Missing table: {table_name}"
                actual_cols = {
                    row["name"]
                    for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
                }
                assert actual_cols == expected_cols, (
                    f"Column mismatch in {table_name}: {actual_cols} != {expected_cols}"
                )

            indexes = {
                row["name"]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'index'"
                ).fetchall()
            }
            assert "idx_documents_content_hash" in indexes
            assert "idx_chunks_document_id" in indexes
            assert "idx_beliefs_chunk_id" in indexes
            assert "idx_beliefs_superseded_by_id" in indexes
            assert "idx_contradictions_status" in indexes
            assert "idx_belief_relations_label" in indexes
            assert "idx_graph_edges_source_id" in indexes
            assert "idx_retention_scores_next_review_at" in indexes
            assert "idx_review_history_chunk_id" in indexes
            assert "idx_review_history_reviewed_at" in indexes
            assert "idx_citations_query_id" in indexes
            assert "idx_morning_briefs_brief_date" in indexes
    finally:
        db.close()


def test_foreign_keys_constraints_and_cascades(tmp_path: Path) -> None:
    """Foreign keys, CHECK/UNIQUE constraints, and cascade deletes must be enforced."""
    db = DatabaseManager(tmp_path / "mindvault.db")
    try:
        MigrationRunner(db).apply_pending()
        now = "2026-11-24T10:00:00+00:00"

        # Orphan chunk insert must fail foreign key check.
        with pytest.raises(sqlite3.IntegrityError):
            with db.transaction() as conn:
                conn.execute(
                    """
                    INSERT INTO chunks (
                        id, document_id, chunk_index, content, token_count,
                        epoch, created_at, metadata_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    ("chk-orphan", "doc-missing", 0, "text", 1, 1, now, "{}"),
                )

        # Insert valid document, chunk, beliefs, contradiction, retention, citation, edge, brief.
        with db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO documents (
                    id, filename, file_type, source_path, content_hash,
                    created_at, updated_at, status, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "doc-1",
                    "notes.md",
                    "markdown",
                    "/vault/notes.md",
                    "abc123",
                    now,
                    now,
                    "indexed",
                    "{}",
                ),
            )
            conn.execute(
                """
                INSERT INTO chunks (
                    id, document_id, chunk_index, content, token_count,
                    epoch, created_at, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                ("chk-1", "doc-1", 0, "Local-first software is resilient.", 5, 1, now, "{}"),
            )
            conn.execute(
                """
                INSERT INTO beliefs (
                    id, chunk_id, belief_text, normalized_text,
                    confidence, epoch, status, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "bel-1",
                    "chk-1",
                    "Local-first is resilient",
                    "local-first is resilient",
                    0.9,
                    1,
                    "active",
                    now,
                ),
            )
            conn.execute(
                """
                INSERT INTO beliefs (
                    id, chunk_id, belief_text, normalized_text,
                    confidence, epoch, status, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "bel-2",
                    "chk-1",
                    "Cloud-only is resilient",
                    "cloud-only is resilient",
                    0.4,
                    2,
                    "active",
                    now,
                ),
            )
            conn.execute(
                """
                INSERT INTO contradictions (
                    id, belief_a_id, belief_b_id, score, status,
                    resolution, created_at, resolved_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                ("con-1", "bel-1", "bel-2", 0.85, "open", None, now, None),
            )
            conn.execute(
                """
                INSERT INTO retention_scores (
                    id, chunk_id, strength, retention,
                    last_reviewed_at, next_review_at, review_count
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                ("ret-1", "chk-1", 1.5, 0.8, now, now, 1),
            )
            conn.execute(
                """
                INSERT INTO citations (id, query_id, chunk_id, citation_type, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                ("cit-1", "qry-1", "chk-1", "retrieval", now),
            )
            conn.execute(
                """
                INSERT INTO graph_edges (id, source_id, target_id, edge_type, weight, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                ("edge-1", "doc-1", "bel-1", "supports", 0.9, now),
            )
            conn.execute(
                """
                INSERT INTO morning_briefs (id, brief_date, content, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                ("brief-1", "2026-11-24", "Daily summary", "{}", now),
            )

        # CHECK constraint: confidence outside [0.0, 1.0] must fail.
        with pytest.raises(sqlite3.IntegrityError):
            with db.transaction() as conn:
                conn.execute(
                    """
                    INSERT INTO beliefs (
                        id, chunk_id, belief_text, normalized_text,
                        confidence, epoch, status, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    ("bel-bad", "chk-1", "Invalid", "invalid", 1.5, 1, "active", now),
                )

        # CHECK constraint: self-contradiction must fail.
        with pytest.raises(sqlite3.IntegrityError):
            with db.transaction() as conn:
                conn.execute(
                    """
                    INSERT INTO contradictions (
                        id, belief_a_id, belief_b_id, score, status,
                        resolution, created_at, resolved_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    ("con-self", "bel-1", "bel-1", 0.5, "open", None, now, None),
                )

        # CHECK constraint: self-loop graph edge must fail.
        with pytest.raises(sqlite3.IntegrityError):
            with db.transaction() as conn:
                conn.execute(
                    """
                    INSERT INTO graph_edges (
                        id, source_id, target_id, edge_type, weight, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    ("edge-loop", "doc-1", "doc-1", "related", 1.0, now),
                )

        # Deleting the document cascades to chunks, beliefs, contradictions, retention, citations.
        with db.transaction() as conn:
            conn.execute("DELETE FROM documents WHERE id = ?", ("doc-1",))

        with db.connection() as conn:
            for table in ("chunks", "beliefs", "contradictions", "retention_scores", "citations"):
                count = conn.execute(f"SELECT COUNT(*) AS cnt FROM {table}").fetchone()["cnt"]
                assert count == 0, f"Expected cascade delete in {table}, found {count} rows"
    finally:
        db.close()


def test_migration_checksum_mismatch_raises(tmp_path: Path) -> None:
    """Modifying an already-applied migration script must raise a RuntimeError."""
    migrations_dir = tmp_path / "migrations"
    migrations_dir.mkdir()
    mig_file = migrations_dir / "001_test.sql"
    mig_file.write_text("CREATE TABLE sample (id TEXT PRIMARY KEY);", encoding="utf-8")

    db = DatabaseManager(tmp_path / "test.db")
    try:
        runner = MigrationRunner(db, migrations_dir=migrations_dir)
        assert runner.apply_pending() == [1]

        mig_file.write_text(
            "CREATE TABLE sample (id TEXT PRIMARY KEY, extra TEXT);",
            encoding="utf-8",
        )
        with pytest.raises(RuntimeError, match="Checksum mismatch"):
            runner.apply_pending()
    finally:
        db.close()


def test_database_manager_close_blocks_new_connections(tmp_path: Path) -> None:
    """Closed DatabaseManager must report unhealthy and reject new connections."""
    db = DatabaseManager(tmp_path / "close_test.db")
    ok, err = db.health_check()
    assert ok is True
    assert err is None

    db.close()
    assert db.is_closed is True
    ok_after, err_after = db.health_check()
    assert ok_after is False
    assert err_after is not None
    with pytest.raises(RuntimeError, match="closed"):
        with db.connection():
            pass
