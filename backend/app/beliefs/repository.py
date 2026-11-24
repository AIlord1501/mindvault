"""Parameterized SQLite repository for beliefs, NLI relations, and contradictions."""

from __future__ import annotations

import json
import sqlite3
import uuid
from typing import Any, Optional

from app.db.database import DatabaseManager

VALID_RESOLUTIONS = frozenset(
    {
        "accept_newer",
        "accept_older",
        "keep_both",
        "context_dependent",
        "false_positive",
    }
)


def _decode_embedding(raw: Optional[str]) -> Optional[list[float]]:
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [float(x) for x in parsed]
    except (TypeError, ValueError):
        return None
    return None


def _row_to_belief(row: sqlite3.Row, *, include_embedding: bool = False) -> dict[str, Any]:
    keys = row.keys()
    item: dict[str, Any] = {
        "id": str(row["id"]),
        "chunk_id": str(row["chunk_id"]),
        "belief_text": str(row["belief_text"]),
        "normalized_text": str(row["normalized_text"]),
        "confidence": float(row["confidence"]),
        "epoch": int(row["epoch"]),
        "status": str(row["status"]),
        "created_at": str(row["created_at"]),
        "superseded_by_id": (
            str(row["superseded_by_id"])
            if "superseded_by_id" in keys and row["superseded_by_id"] is not None
            else None
        ),
        "context_note": (
            str(row["context_note"])
            if "context_note" in keys and row["context_note"] is not None
            else None
        ),
        "updated_at": (
            str(row["updated_at"])
            if "updated_at" in keys and row["updated_at"] is not None
            else None
        ),
    }
    if "document_id" in keys:
        item["document_id"] = str(row["document_id"]) if row["document_id"] is not None else None
    if "filename" in keys:
        item["filename"] = str(row["filename"]) if row["filename"] is not None else None
    if include_embedding and "embedding_json" in keys:
        item["embedding"] = _decode_embedding(row["embedding_json"])
    return item


def _row_to_contradiction(row: sqlite3.Row) -> dict[str, Any]:
    keys = row.keys()
    return {
        "id": str(row["id"]),
        "belief_a_id": str(row["belief_a_id"]),
        "belief_b_id": str(row["belief_b_id"]),
        "score": float(row["score"]),
        "entailment_score": (
            float(row["entailment_score"])
            if "entailment_score" in keys and row["entailment_score"] is not None
            else 0.0
        ),
        "neutral_score": (
            float(row["neutral_score"])
            if "neutral_score" in keys and row["neutral_score"] is not None
            else 0.0
        ),
        "similarity_score": (
            float(row["similarity_score"])
            if "similarity_score" in keys and row["similarity_score"] is not None
            else 0.0
        ),
        "status": str(row["status"]),
        "resolution": str(row["resolution"]) if row["resolution"] is not None else None,
        "resolution_note": (
            str(row["resolution_note"])
            if "resolution_note" in keys and row["resolution_note"] is not None
            else None
        ),
        "created_at": str(row["created_at"]),
        "resolved_at": str(row["resolved_at"]) if row["resolved_at"] is not None else None,
    }


def _row_to_relation(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": str(row["id"]),
        "source_belief_id": str(row["source_belief_id"]),
        "target_belief_id": str(row["target_belief_id"]),
        "label": str(row["label"]),
        "contradiction_score": float(row["contradiction_score"]),
        "entailment_score": float(row["entailment_score"]),
        "neutral_score": float(row["neutral_score"]),
        "similarity_score": float(row["similarity_score"]),
        "created_at": str(row["created_at"]),
    }


class BeliefRepository:
    """Data-access layer for beliefs, NLI relations, and contradictions."""

    def __init__(self, db: DatabaseManager) -> None:
        self._db = db

    @staticmethod
    def _older_newer_ids(
        belief_x: sqlite3.Row,
        belief_y: sqlite3.Row,
    ) -> tuple[str, str]:
        key_x = (int(belief_x["epoch"]), str(belief_x["created_at"]), str(belief_x["id"]))
        key_y = (int(belief_y["epoch"]), str(belief_y["created_at"]), str(belief_y["id"]))
        if key_x <= key_y:
            return str(belief_x["id"]), str(belief_y["id"])
        return str(belief_y["id"]), str(belief_x["id"])

    @staticmethod
    def _upsert_graph_edge(
        conn: sqlite3.Connection,
        *,
        source_id: str,
        target_id: str,
        edge_type: str,
        weight: float,
        created_at: str,
    ) -> None:
        if source_id == target_id:
            return
        conn.execute(
            """
            INSERT INTO graph_edges (id, source_id, target_id, edge_type, weight, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (source_id, target_id, edge_type) DO UPDATE SET
                weight = excluded.weight,
                created_at = excluded.created_at
            """,
            (uuid.uuid4().hex, source_id, target_id, edge_type, float(weight), created_at),
        )

    @staticmethod
    def _sync_belief_active_if_resolved(
        conn: sqlite3.Connection,
        belief_id: str,
        updated_at: str,
    ) -> None:
        row = conn.execute(
            "SELECT status FROM beliefs WHERE id = ?",
            (belief_id,),
        ).fetchone()
        if row is None or str(row["status"]) != "contradicted":
            return

        open_row = conn.execute(
            """
            SELECT COUNT(*) AS cnt
            FROM contradictions
            WHERE status = 'open' AND (belief_a_id = ? OR belief_b_id = ?)
            """,
            (belief_id, belief_id),
        ).fetchone()
        open_cnt = int(open_row["cnt"]) if open_row is not None else 0
        if open_cnt == 0:
            conn.execute(
                "UPDATE beliefs SET status = 'active', updated_at = ? WHERE id = ?",
                (updated_at, belief_id),
            )

    def _get_supersedes_map(
        self,
        conn: sqlite3.Connection,
        belief_ids: list[str],
    ) -> dict[str, list[str]]:
        if not belief_ids:
            return {}
        unique_ids = list(dict.fromkeys(belief_ids))
        mapping: dict[str, list[str]] = {bid: [] for bid in unique_ids}
        for start in range(0, len(unique_ids), 500):
            batch = unique_ids[start : start + 500]
            placeholders = ", ".join("?" for _ in batch)
            rows = conn.execute(
                f"""
                SELECT id, superseded_by_id
                FROM beliefs
                WHERE superseded_by_id IN ({placeholders})
                ORDER BY epoch ASC, created_at ASC
                """,
                tuple(batch),
            ).fetchall()
            for row in rows:
                sup_by = str(row["superseded_by_id"])
                if sup_by in mapping:
                    mapping[sup_by].append(str(row["id"]))
        return mapping

    def list_active_beliefs_for_comparison(
        self,
        *,
        exclude_chunk_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Fetch active or contradicted beliefs (never deleting historical records)."""
        where_sql = "status IN ('active', 'contradicted')"
        params: list[Any] = []
        if exclude_chunk_id is not None:
            where_sql += " AND chunk_id <> ?"
            params.append(exclude_chunk_id)

        with self._db.connection() as conn:
            rows = conn.execute(
                f"""
                SELECT *
                FROM beliefs
                WHERE {where_sql}
                ORDER BY epoch DESC, created_at DESC
                """,
                tuple(params),
            ).fetchall()
            return [_row_to_belief(row, include_embedding=True) for row in rows]

    def create_belief(
        self,
        *,
        belief_id: str,
        chunk_id: str,
        belief_text: str,
        normalized_text: str,
        confidence: float,
        epoch: int,
        status: str = "active",
        embedding: Optional[list[float]] = None,
        created_at: str,
    ) -> dict[str, Any]:
        """Insert a new belief record and return its representation."""
        emb_json = json.dumps(embedding, ensure_ascii=True) if embedding is not None else None
        with self._db.transaction(immediate=True) as conn:
            conn.execute(
                """
                INSERT INTO beliefs (
                    id, chunk_id, belief_text, normalized_text, confidence,
                    epoch, status, created_at, embedding_json, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    belief_id,
                    chunk_id,
                    belief_text,
                    normalized_text,
                    float(confidence),
                    int(epoch),
                    status,
                    created_at,
                    emb_json,
                    created_at,
                ),
            )
            row = conn.execute(
                """
                SELECT b.*, c.document_id AS document_id, d.filename AS filename
                FROM beliefs b
                INNER JOIN chunks c ON c.id = b.chunk_id
                INNER JOIN documents d ON d.id = c.document_id
                WHERE b.id = ?
                """,
                (belief_id,),
            ).fetchone()
            item = _row_to_belief(row, include_embedding=False)
            item["supersedes_ids"] = []
            return item

    def save_relation(
        self,
        *,
        source_belief_id: str,
        target_belief_id: str,
        label: str,
        contradiction_score: float,
        entailment_score: float,
        neutral_score: float,
        similarity_score: float,
        created_at: str,
    ) -> dict[str, Any]:
        """Persist NLI scores for a belief pair in `belief_relations`."""
        rel_id = uuid.uuid4().hex
        with self._db.transaction(immediate=True) as conn:
            conn.execute(
                """
                INSERT INTO belief_relations (
                    id, source_belief_id, target_belief_id, label,
                    contradiction_score, entailment_score, neutral_score,
                    similarity_score, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (source_belief_id, target_belief_id) DO UPDATE SET
                    label = excluded.label,
                    contradiction_score = excluded.contradiction_score,
                    entailment_score = excluded.entailment_score,
                    neutral_score = excluded.neutral_score,
                    similarity_score = excluded.similarity_score,
                    created_at = excluded.created_at
                """,
                (
                    rel_id,
                    source_belief_id,
                    target_belief_id,
                    label,
                    float(contradiction_score),
                    float(entailment_score),
                    float(neutral_score),
                    float(similarity_score),
                    created_at,
                ),
            )
            if label == "entailment":
                self._upsert_graph_edge(
                    conn,
                    source_id=source_belief_id,
                    target_id=target_belief_id,
                    edge_type="entails",
                    weight=float(entailment_score),
                    created_at=created_at,
                )
            row = conn.execute(
                """
                SELECT * FROM belief_relations
                WHERE source_belief_id = ? AND target_belief_id = ?
                """,
                (source_belief_id, target_belief_id),
            ).fetchone()
            return _row_to_relation(row)

    def create_contradiction(
        self,
        *,
        belief_x_id: str,
        belief_y_id: str,
        score: float,
        entailment_score: float,
        neutral_score: float,
        similarity_score: float,
        created_at: str,
    ) -> dict[str, Any]:
        """Create or update an open contradiction between two beliefs (older -> newer)."""
        with self._db.transaction(immediate=True) as conn:
            row_x = conn.execute("SELECT * FROM beliefs WHERE id = ?", (belief_x_id,)).fetchone()
            row_y = conn.execute("SELECT * FROM beliefs WHERE id = ?", (belief_y_id,)).fetchone()
            if row_x is None or row_y is None:
                raise ValueError("Both beliefs must exist before creating a contradiction.")

            older_id, newer_id = self._older_newer_ids(row_x, row_y)

            existing = conn.execute(
                """
                SELECT * FROM contradictions
                WHERE (belief_a_id = ? AND belief_b_id = ?)
                   OR (belief_a_id = ? AND belief_b_id = ?)
                LIMIT 1
                """,
                (older_id, newer_id, newer_id, older_id),
            ).fetchone()

            if existing is not None:
                con_id = str(existing["id"])
                conn.execute(
                    """
                    UPDATE contradictions
                    SET score = ?, entailment_score = ?, neutral_score = ?,
                        similarity_score = ?
                    WHERE id = ?
                    """,
                    (
                        float(score),
                        float(entailment_score),
                        float(neutral_score),
                        float(similarity_score),
                        con_id,
                    ),
                )
                target_status = str(existing["status"])
            else:
                con_id = uuid.uuid4().hex
                conn.execute(
                    """
                    INSERT INTO contradictions (
                        id, belief_a_id, belief_b_id, score,
                        entailment_score, neutral_score, similarity_score,
                        status, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, 'open', ?)
                    """,
                    (
                        con_id,
                        older_id,
                        newer_id,
                        float(score),
                        float(entailment_score),
                        float(neutral_score),
                        float(similarity_score),
                        created_at,
                    ),
                )
                target_status = "open"

            if target_status == "open":
                conn.execute(
                    """
                    UPDATE beliefs
                    SET status = 'contradicted', updated_at = ?
                    WHERE id IN (?, ?) AND status = 'active'
                    """,
                    (created_at, older_id, newer_id),
                )
                self._upsert_graph_edge(
                    conn,
                    source_id=older_id,
                    target_id=newer_id,
                    edge_type="contradicts",
                    weight=float(score),
                    created_at=created_at,
                )

            return self._fetch_enriched_contradiction(conn, con_id)

    def _fetch_enriched_contradiction(
        self,
        conn: sqlite3.Connection,
        contradiction_id: str,
    ) -> Optional[dict[str, Any]]:
        row = conn.execute(
            "SELECT * FROM contradictions WHERE id = ?",
            (contradiction_id,),
        ).fetchone()
        if row is None:
            return None
        con = _row_to_contradiction(row)
        bel_a = conn.execute(
            """
            SELECT b.*, c.document_id AS document_id, d.filename AS filename
            FROM beliefs b
            INNER JOIN chunks c ON c.id = b.chunk_id
            INNER JOIN documents d ON d.id = c.document_id
            WHERE b.id = ?
            """,
            (con["belief_a_id"],),
        ).fetchone()
        bel_b = conn.execute(
            """
            SELECT b.*, c.document_id AS document_id, d.filename AS filename
            FROM beliefs b
            INNER JOIN chunks c ON c.id = b.chunk_id
            INNER JOIN documents d ON d.id = c.document_id
            WHERE b.id = ?
            """,
            (con["belief_b_id"],),
        ).fetchone()
        con["belief_a"] = _row_to_belief(bel_a) if bel_a is not None else None
        con["belief_b"] = _row_to_belief(bel_b) if bel_b is not None else None
        return con

    def list_beliefs(
        self,
        *,
        status: Optional[str] = None,
        document_id: Optional[str] = None,
        chunk_id: Optional[str] = None,
        epoch_min: Optional[int] = None,
        epoch_max: Optional[int] = None,
        min_confidence: Optional[float] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """List beliefs with optional filtering and pagination."""
        where_clauses: list[str] = []
        params: list[Any] = []

        if status is not None:
            where_clauses.append("b.status = ?")
            params.append(status)
        if document_id is not None:
            where_clauses.append("c.document_id = ?")
            params.append(document_id)
        if chunk_id is not None:
            where_clauses.append("b.chunk_id = ?")
            params.append(chunk_id)
        if epoch_min is not None:
            where_clauses.append("b.epoch >= ?")
            params.append(int(epoch_min))
        if epoch_max is not None:
            where_clauses.append("b.epoch <= ?")
            params.append(int(epoch_max))
        if min_confidence is not None:
            where_clauses.append("b.confidence >= ?")
            params.append(float(min_confidence))

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        with self._db.connection() as conn:
            total_row = conn.execute(
                f"""
                SELECT COUNT(*) AS total
                FROM beliefs b
                INNER JOIN chunks c ON c.id = b.chunk_id
                {where_sql}
                """,
                tuple(params),
            ).fetchone()
            total = int(total_row["total"]) if total_row is not None else 0

            rows = conn.execute(
                f"""
                SELECT b.*, c.document_id AS document_id, d.filename AS filename
                FROM beliefs b
                INNER JOIN chunks c ON c.id = b.chunk_id
                INNER JOIN documents d ON d.id = c.document_id
                {where_sql}
                ORDER BY b.epoch DESC, b.created_at DESC
                LIMIT ? OFFSET ?
                """,
                tuple(params + [limit, offset]),
            ).fetchall()

            beliefs = [_row_to_belief(row) for row in rows]
            sup_map = self._get_supersedes_map(conn, [b["id"] for b in beliefs])
            for belief in beliefs:
                belief["supersedes_ids"] = sup_map.get(belief["id"], [])
            return beliefs, total

    def get_belief_detail(self, belief_id: str) -> Optional[dict[str, Any]]:
        """Fetch one belief with lineage history, relations, and contradictions."""
        with self._db.connection() as conn:
            row = conn.execute(
                """
                SELECT b.*, c.document_id AS document_id, d.filename AS filename,
                       c.content AS chunk_content, d.source_path AS source_path
                FROM beliefs b
                INNER JOIN chunks c ON c.id = b.chunk_id
                INNER JOIN documents d ON d.id = c.document_id
                WHERE b.id = ?
                """,
                (belief_id,),
            ).fetchone()
            if row is None:
                return None

            belief = _row_to_belief(row)
            belief["source_path"] = str(row["source_path"])
            belief["chunk_content"] = str(row["chunk_content"])

            sup_map = self._get_supersedes_map(conn, [belief_id])
            belief["supersedes_ids"] = sup_map.get(belief_id, [])

            # Walk supersession chain with cycle guard to preserve complete epoch lineage.
            lineage_ids: set[str] = {belief_id}
            frontier = [belief_id]
            while frontier:
                cur_id = frontier.pop(0)
                cur_row = conn.execute(
                    "SELECT superseded_by_id FROM beliefs WHERE id = ?",
                    (cur_id,),
                ).fetchone()
                if cur_row and cur_row["superseded_by_id"] is not None:
                    nxt = str(cur_row["superseded_by_id"])
                    if nxt not in lineage_ids:
                        lineage_ids.add(nxt)
                        frontier.append(nxt)

                anc_rows = conn.execute(
                    "SELECT id FROM beliefs WHERE superseded_by_id = ?",
                    (cur_id,),
                ).fetchall()
                for anc in anc_rows:
                    aid = str(anc["id"])
                    if aid not in lineage_ids:
                        lineage_ids.add(aid)
                        frontier.append(aid)

            placeholders = ", ".join("?" for _ in lineage_ids)
            lineage_rows = conn.execute(
                f"""
                SELECT b.*, c.document_id AS document_id, d.filename AS filename
                FROM beliefs b
                INNER JOIN chunks c ON c.id = b.chunk_id
                INNER JOIN documents d ON d.id = c.document_id
                WHERE b.id IN ({placeholders})
                ORDER BY b.epoch ASC, b.created_at ASC
                """,
                tuple(lineage_ids),
            ).fetchall()
            belief["lineage_history"] = [_row_to_belief(r) for r in lineage_rows]

            rel_rows = conn.execute(
                """
                SELECT * FROM belief_relations
                WHERE source_belief_id = ? OR target_belief_id = ?
                ORDER BY created_at DESC
                """,
                (belief_id, belief_id),
            ).fetchall()
            belief["relations"] = [_row_to_relation(r) for r in rel_rows]

            con_rows = conn.execute(
                """
                SELECT * FROM contradictions
                WHERE belief_a_id = ? OR belief_b_id = ?
                ORDER BY created_at DESC
                """,
                (belief_id, belief_id),
            ).fetchall()
            belief["contradictions"] = [_row_to_contradiction(r) for r in con_rows]
            return belief

    def list_contradictions(
        self,
        *,
        status: Optional[str] = None,
        min_score: Optional[float] = None,
        epoch_min: Optional[int] = None,
        epoch_max: Optional[int] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """List contradictions joined with both belief records."""
        where_clauses: list[str] = []
        params: list[Any] = []

        if status is not None:
            where_clauses.append("con.status = ?")
            params.append(status)
        if min_score is not None:
            where_clauses.append("con.score >= ?")
            params.append(float(min_score))
        if epoch_min is not None:
            where_clauses.append("(ba.epoch >= ? OR bb.epoch >= ?)")
            params.extend([int(epoch_min), int(epoch_min)])
        if epoch_max is not None:
            where_clauses.append("(ba.epoch <= ? AND bb.epoch <= ?)")
            params.extend([int(epoch_max), int(epoch_max)])

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        with self._db.connection() as conn:
            total_row = conn.execute(
                f"""
                SELECT COUNT(*) AS total
                FROM contradictions con
                INNER JOIN beliefs ba ON ba.id = con.belief_a_id
                INNER JOIN beliefs bb ON bb.id = con.belief_b_id
                {where_sql}
                """,
                tuple(params),
            ).fetchone()
            total = int(total_row["total"]) if total_row is not None else 0

            rows = conn.execute(
                f"""
                SELECT con.id AS con_id
                FROM contradictions con
                INNER JOIN beliefs ba ON ba.id = con.belief_a_id
                INNER JOIN beliefs bb ON bb.id = con.belief_b_id
                {where_sql}
                ORDER BY con.score DESC, con.created_at DESC
                LIMIT ? OFFSET ?
                """,
                tuple(params + [limit, offset]),
            ).fetchall()

            items: list[dict[str, Any]] = []
            for row in rows:
                enriched = self._fetch_enriched_contradiction(conn, str(row["con_id"]))
                if enriched is not None:
                    items.append(enriched)
            return items, total

    def get_contradiction(self, contradiction_id: str) -> Optional[dict[str, Any]]:
        """Fetch a single contradiction with embedded belief summaries."""
        with self._db.connection() as conn:
            return self._fetch_enriched_contradiction(conn, contradiction_id)

    def resolve_contradiction(
        self,
        *,
        contradiction_id: str,
        resolution: str,
        note: Optional[str],
        resolved_at: str,
    ) -> Optional[dict[str, Any]]:
        """Apply one of the 5 resolution/dismissal actions atomically."""
        if resolution not in VALID_RESOLUTIONS:
            raise ValueError(f"Unsupported resolution '{resolution}'.")

        with self._db.transaction(immediate=True) as conn:
            con_row = conn.execute(
                "SELECT * FROM contradictions WHERE id = ?",
                (contradiction_id,),
            ).fetchone()
            if con_row is None:
                return None

            belief_a_id = str(con_row["belief_a_id"])
            belief_b_id = str(con_row["belief_b_id"])
            row_a = conn.execute("SELECT * FROM beliefs WHERE id = ?", (belief_a_id,)).fetchone()
            row_b = conn.execute("SELECT * FROM beliefs WHERE id = ?", (belief_b_id,)).fetchone()
            if row_a is None or row_b is None:
                raise ValueError("Contradiction references missing belief records.")

            older_id, newer_id = self._older_newer_ids(row_a, row_b)
            next_status = "dismissed" if resolution == "false_positive" else "resolved"

            conn.execute(
                """
                UPDATE contradictions
                SET status = ?, resolution = ?, resolution_note = ?, resolved_at = ?
                WHERE id = ?
                """,
                (next_status, resolution, note, resolved_at, contradiction_id),
            )

            if resolution == "accept_newer":
                conn.execute(
                    """
                    UPDATE beliefs
                    SET status = 'superseded', superseded_by_id = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (newer_id, resolved_at, older_id),
                )
                self._upsert_graph_edge(
                    conn,
                    source_id=newer_id,
                    target_id=older_id,
                    edge_type="supersedes",
                    weight=1.0,
                    created_at=resolved_at,
                )
                self._sync_belief_active_if_resolved(conn, newer_id, resolved_at)

            elif resolution == "accept_older":
                conn.execute(
                    """
                    UPDATE beliefs
                    SET status = 'superseded', superseded_by_id = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (older_id, resolved_at, newer_id),
                )
                self._upsert_graph_edge(
                    conn,
                    source_id=older_id,
                    target_id=newer_id,
                    edge_type="supersedes",
                    weight=1.0,
                    created_at=resolved_at,
                )
                self._sync_belief_active_if_resolved(conn, older_id, resolved_at)

            elif resolution == "context_dependent":
                context_label = note.strip() if note and note.strip() else "Context-dependent"
                conn.execute(
                    "UPDATE beliefs SET context_note = ?, updated_at = ? WHERE id IN (?, ?)",
                    (context_label, resolved_at, older_id, newer_id),
                )
                self._sync_belief_active_if_resolved(conn, older_id, resolved_at)
                self._sync_belief_active_if_resolved(conn, newer_id, resolved_at)

            elif resolution == "keep_both":
                self._sync_belief_active_if_resolved(conn, older_id, resolved_at)
                self._sync_belief_active_if_resolved(conn, newer_id, resolved_at)

            elif resolution == "false_positive":
                conn.execute(
                    """
                    DELETE FROM graph_edges
                    WHERE edge_type = 'contradicts'
                      AND ((source_id = ? AND target_id = ?)
                        OR (source_id = ? AND target_id = ?))
                    """,
                    (older_id, newer_id, newer_id, older_id),
                )
                self._sync_belief_active_if_resolved(conn, older_id, resolved_at)
                self._sync_belief_active_if_resolved(conn, newer_id, resolved_at)

            return self._fetch_enriched_contradiction(conn, contradiction_id)
