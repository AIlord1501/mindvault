"""Parameterized SQLite repository for MindVault Concept and Citation graphs."""

from __future__ import annotations

import json
import uuid
from typing import Any, Optional

from app.db.database import DatabaseManager


def _decode_json(raw: Optional[str]) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, ValueError):
        return {}


def _short_label(text: str, max_len: int = 56) -> str:
    collapsed = " ".join((text or "").split())
    if len(collapsed) <= max_len:
        return collapsed
    return collapsed[: max_len - 3].rstrip() + "..."


class GraphRepository:
    """Data-access layer for polymorphic `graph_edges` and graph node hydration."""

    def __init__(self, db: DatabaseManager) -> None:
        self._db = db

    def upsert_edges(self, edges: list[tuple[str, str, str, float, str]]) -> int:
        """Upsert (source_id, target_id, edge_type, weight, created_at) edges."""
        rows: list[tuple[str, str, str, str, float, str]] = []
        for source_id, target_id, edge_type, weight, created_at in edges:
            if not source_id or not target_id or source_id == target_id:
                continue
            rows.append(
                (
                    uuid.uuid4().hex,
                    source_id,
                    target_id,
                    edge_type,
                    float(weight),
                    created_at,
                )
            )
        if not rows:
            return 0

        with self._db.transaction(immediate=True) as conn:
            conn.executemany(
                """
                INSERT INTO graph_edges (id, source_id, target_id, edge_type, weight, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (source_id, target_id, edge_type) DO UPDATE SET
                    weight = excluded.weight,
                    created_at = excluded.created_at
                """,
                rows,
            )
        return len(rows)

    def delete_edges_for_entities(self, entity_ids: list[str]) -> int:
        """Delete all edges incident to any of the given entity IDs."""
        unique_ids = list(dict.fromkeys(eid for eid in entity_ids if eid))
        if not unique_ids:
            return 0

        deleted = 0
        with self._db.transaction(immediate=True) as conn:
            for start in range(0, len(unique_ids), 400):
                batch = unique_ids[start : start + 400]
                placeholders = ", ".join("?" for _ in batch)
                cur = conn.execute(
                    f"""
                    DELETE FROM graph_edges
                    WHERE source_id IN ({placeholders})
                       OR target_id IN ({placeholders})
                    """,
                    tuple(batch + batch),
                )
                deleted += cur.rowcount if cur.rowcount is not None else 0
        return deleted

    def delete_edges_by_types(self, edge_types: list[str]) -> int:
        """Delete all graph edges of the specified types."""
        unique_types = list(dict.fromkeys(t for t in edge_types if t))
        if not unique_types:
            return 0

        placeholders = ", ".join("?" for _ in unique_types)
        with self._db.transaction(immediate=True) as conn:
            cur = conn.execute(
                f"DELETE FROM graph_edges WHERE edge_type IN ({placeholders})",
                tuple(unique_types),
            )
            return cur.rowcount if cur.rowcount is not None else 0

    def prune_dangling_edges(self) -> int:
        """Remove any edges pointing to non-existent documents, chunks, beliefs, or queries."""
        with self._db.transaction(immediate=True) as conn:
            valid_sql = """
                SELECT id FROM documents WHERE status = 'indexed'
                UNION
                SELECT c.id FROM chunks c
                INNER JOIN documents d ON d.id = c.document_id
                WHERE d.status = 'indexed'
                UNION
                SELECT id FROM beliefs
                UNION
                SELECT DISTINCT ('query:' || query_id) AS id FROM citations
            """
            cur = conn.execute(
                f"""
                DELETE FROM graph_edges
                WHERE source_id NOT IN ({valid_sql})
                   OR target_id NOT IN ({valid_sql})
                """
            )
            return cur.rowcount if cur.rowcount is not None else 0

    def fetch_document_nodes(self) -> list[dict[str, Any]]:
        """Return D3-ready document nodes for all indexed documents."""
        with self._db.connection() as conn:
            rows = conn.execute(
                """
                SELECT
                    d.*,
                    COUNT(c.id) AS chunk_count,
                    COALESCE(MAX(c.epoch), 0) AS max_epoch
                FROM documents d
                LEFT JOIN chunks c ON c.document_id = d.id
                WHERE d.status = 'indexed'
                GROUP BY d.id
                ORDER BY d.created_at DESC
                """
            ).fetchall()

        nodes: list[dict[str, Any]] = []
        for row in rows:
            meta = _decode_json(row["metadata_json"])
            epoch = int(row["max_epoch"]) if int(row["chunk_count"]) > 0 else int(
                meta.get("epoch", 0)
            )
            doc_id = str(row["id"])
            filename = str(row["filename"])
            file_type = str(row["file_type"])
            nodes.append(
                {
                    "id": doc_id,
                    "node_type": "document",
                    "label": filename,
                    "document_id": doc_id,
                    "filename": filename,
                    "file_type": file_type,
                    "epoch": epoch,
                    "search_blob": f"{filename} {file_type} {row['source_path']}".lower(),
                    "metadata": {
                        "source_path": str(row["source_path"]),
                        "chunk_count": int(row["chunk_count"]),
                        "content_hash": str(row["content_hash"]),
                    },
                }
            )
        return nodes

    def fetch_chunk_nodes(self) -> list[dict[str, Any]]:
        """Return D3-ready chunk nodes belonging to indexed documents."""
        with self._db.connection() as conn:
            rows = conn.execute(
                """
                SELECT
                    c.*,
                    d.filename AS filename,
                    d.file_type AS file_type,
                    d.source_path AS source_path
                FROM chunks c
                INNER JOIN documents d ON d.id = c.document_id
                WHERE d.status = 'indexed'
                ORDER BY c.epoch DESC, c.created_at DESC, c.chunk_index ASC
                """
            ).fetchall()

        nodes: list[dict[str, Any]] = []
        for row in rows:
            meta = _decode_json(row["metadata_json"])
            chunk_idx = int(row["chunk_index"])
            filename = str(row["filename"])
            content = str(row["content"])
            section = str(meta.get("section_heading", "")).strip()
            label_base = f"{filename} #{chunk_idx}"
            tail = section if section else _short_label(content, 40)
            label = f"{label_base}: {tail}"
            nodes.append(
                {
                    "id": str(row["id"]),
                    "node_type": "chunk",
                    "label": label,
                    "document_id": str(row["document_id"]),
                    "filename": filename,
                    "file_type": str(row["file_type"]),
                    "epoch": int(row["epoch"]),
                    "search_blob": f"{filename} {section} {content}".lower(),
                    "metadata": {
                        "chunk_index": chunk_idx,
                        "token_count": int(row["token_count"]),
                        "section_heading": section or None,
                        "excerpt": _short_label(content, 160),
                    },
                }
            )
        return nodes

    def fetch_belief_nodes(self) -> list[dict[str, Any]]:
        """Return D3-ready belief nodes linked to indexed chunks."""
        with self._db.connection() as conn:
            rows = conn.execute(
                """
                SELECT
                    b.*,
                    c.document_id AS document_id,
                    d.filename AS filename,
                    d.file_type AS file_type
                FROM beliefs b
                INNER JOIN chunks c ON c.id = b.chunk_id
                INNER JOIN documents d ON d.id = c.document_id
                WHERE d.status = 'indexed'
                ORDER BY b.epoch DESC, b.created_at DESC
                """
            ).fetchall()

        nodes: list[dict[str, Any]] = []
        for row in rows:
            text = str(row["belief_text"])
            filename = str(row["filename"])
            nodes.append(
                {
                    "id": str(row["id"]),
                    "node_type": "belief",
                    "label": _short_label(text, 64),
                    "document_id": str(row["document_id"]),
                    "filename": filename,
                    "file_type": str(row["file_type"]),
                    "epoch": int(row["epoch"]),
                    "search_blob": f"{text} {row['status']} {filename}".lower(),
                    "metadata": {
                        "chunk_id": str(row["chunk_id"]),
                        "belief_text": text,
                        "confidence": float(row["confidence"]),
                        "status": str(row["status"]),
                        "superseded_by_id": (
                            str(row["superseded_by_id"])
                            if row["superseded_by_id"] is not None
                            else None
                        ),
                    },
                }
            )
        return nodes

    def fetch_query_nodes(self) -> list[dict[str, Any]]:
        """Return D3-ready query nodes derived from persisted citations."""
        with self._db.connection() as conn:
            rows = conn.execute(
                """
                SELECT
                    cit.query_id AS query_id,
                    MIN(cit.created_at) AS created_at,
                    MAX(c.epoch) AS max_epoch,
                    COUNT(DISTINCT cit.chunk_id) AS cited_chunk_count,
                    GROUP_CONCAT(DISTINCT c.document_id) AS doc_ids_csv,
                    GROUP_CONCAT(DISTINCT d.filename) AS filenames_csv
                FROM citations cit
                INNER JOIN chunks c ON c.id = cit.chunk_id
                INNER JOIN documents d ON d.id = c.document_id
                WHERE d.status = 'indexed'
                GROUP BY cit.query_id
                ORDER BY created_at DESC
                """
            ).fetchall()

        nodes: list[dict[str, Any]] = []
        for row in rows:
            qid = str(row["query_id"])
            doc_ids = {
                part.strip()
                for part in str(row["doc_ids_csv"] or "").split(",")
                if part.strip()
            }
            filenames = [
                part.strip()
                for part in str(row["filenames_csv"] or "").split(",")
                if part.strip()
            ]
            nodes.append(
                {
                    "id": f"query:{qid}",
                    "node_type": "query",
                    "label": f"Query {qid[:8]}",
                    "document_id": next(iter(doc_ids)) if len(doc_ids) == 1 else None,
                    "document_ids": doc_ids,
                    "filename": filenames[0] if len(filenames) == 1 else None,
                    "file_type": None,
                    "epoch": int(row["max_epoch"] or 0),
                    "search_blob": f"query {qid} {' '.join(filenames)}".lower(),
                    "metadata": {
                        "query_id": qid,
                        "cited_chunk_count": int(row["cited_chunk_count"]),
                        "document_ids": sorted(doc_ids),
                        "created_at": str(row["created_at"]),
                    },
                }
            )
        return nodes

    def fetch_all_nodes_map(self) -> dict[str, dict[str, Any]]:
        """Return all indexed nodes indexed by node ID."""
        node_map: dict[str, dict[str, Any]] = {}
        for getter in (
            self.fetch_document_nodes,
            self.fetch_chunk_nodes,
            self.fetch_belief_nodes,
            self.fetch_query_nodes,
        ):
            for node in getter():
                node_map[node["id"]] = node
        return node_map

    def resolve_node_id(self, raw_id: str) -> Optional[str]:
        """Resolve an arbitrary node ID (or bare query ID) to a canonical node ID."""
        node_map = self.fetch_all_nodes_map()
        if raw_id in node_map:
            return raw_id
        prefixed = f"query:{raw_id}"
        if prefixed in node_map:
            return prefixed
        return None

    def fetch_edges_between(
        self,
        node_ids: set[str],
        *,
        edge_types: Optional[list[str]] = None,
        limit: int = 2000,
    ) -> list[dict[str, Any]]:
        """Fetch edges whose endpoints both belong to `node_ids`."""
        if not node_ids or (edge_types is not None and len(edge_types) == 0):
            return []

        where_clauses: list[str] = []
        params: list[Any] = []
        if edge_types:
            placeholders = ", ".join("?" for _ in edge_types)
            where_clauses.append(f"edge_type IN ({placeholders})")
            params.extend(edge_types)

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        with self._db.connection() as conn:
            rows = conn.execute(
                f"""
                SELECT id, source_id, target_id, edge_type, weight, created_at
                FROM graph_edges
                {where_sql}
                ORDER BY weight DESC, id ASC
                """,
                tuple(params),
            ).fetchall()

        edges: list[dict[str, Any]] = []
        for row in rows:
            src = str(row["source_id"])
            tgt = str(row["target_id"])
            if src in node_ids and tgt in node_ids:
                edges.append(
                    {
                        "id": str(row["id"]),
                        "source": src,
                        "target": tgt,
                        "edge_type": str(row["edge_type"]),
                        "weight": round(float(row["weight"]), 6),
                        "created_at": str(row["created_at"]),
                    }
                )
                if len(edges) >= limit:
                    break
        return edges

    def fetch_incident_edges(
        self,
        frontier_ids: set[str],
        *,
        edge_types: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """Fetch all edges incident to any node ID in `frontier_ids`."""
        if not frontier_ids or (edge_types is not None and len(edge_types) == 0):
            return []

        id_list = sorted(frontier_ids)
        edges_by_id: dict[str, dict[str, Any]] = {}
        type_sql = ""
        type_params: list[str] = []
        if edge_types:
            type_placeholders = ", ".join("?" for _ in edge_types)
            type_sql = f" AND edge_type IN ({type_placeholders})"
            type_params = list(edge_types)

        with self._db.connection() as conn:
            for start in range(0, len(id_list), 400):
                batch = id_list[start : start + 400]
                placeholders = ", ".join("?" for _ in batch)
                rows = conn.execute(
                    f"""
                    SELECT id, source_id, target_id, edge_type, weight, created_at
                    FROM graph_edges
                    WHERE (source_id IN ({placeholders}) OR target_id IN ({placeholders}))
                    {type_sql}
                    ORDER BY weight DESC, id ASC
                    """,
                    tuple(batch + batch + type_params),
                ).fetchall()
                for row in rows:
                    eid = str(row["id"])
                    edges_by_id[eid] = {
                        "id": eid,
                        "source": str(row["source_id"]),
                        "target": str(row["target_id"]),
                        "edge_type": str(row["edge_type"]),
                        "weight": round(float(row["weight"]), 6),
                        "created_at": str(row["created_at"]),
                    }
        return list(edges_by_id.values())

    def list_indexed_documents_brief(self) -> list[dict[str, Any]]:
        """Return (id, filename) for all indexed documents."""
        with self._db.connection() as conn:
            rows = conn.execute(
                "SELECT id, filename, created_at FROM documents WHERE status = 'indexed'"
            ).fetchall()
            return [
                {
                    "id": str(row["id"]),
                    "filename": str(row["filename"]),
                    "created_at": str(row["created_at"]),
                }
                for row in rows
            ]

    def list_indexed_chunks_for_rebuild(self) -> list[dict[str, Any]]:
        """Return all indexed chunks ordered by document and chunk_index."""
        with self._db.connection() as conn:
            rows = conn.execute(
                """
                SELECT c.id, c.document_id, c.chunk_index, c.content, c.created_at,
                       d.filename AS filename
                FROM chunks c
                INNER JOIN documents d ON d.id = c.document_id
                WHERE d.status = 'indexed'
                ORDER BY c.document_id ASC, c.chunk_index ASC
                """
            ).fetchall()
            return [
                {
                    "id": str(row["id"]),
                    "document_id": str(row["document_id"]),
                    "chunk_index": int(row["chunk_index"]),
                    "content": str(row["content"]),
                    "created_at": str(row["created_at"]),
                    "filename": str(row["filename"]),
                }
                for row in rows
            ]

    def list_beliefs_and_relations_for_rebuild(self) -> dict[str, list[dict[str, Any]]]:
        """Load beliefs, relations, and contradictions to rebuild belief edges."""
        with self._db.connection() as conn:
            beliefs = conn.execute(
                """
                SELECT id, chunk_id, confidence, superseded_by_id, created_at, updated_at
                FROM beliefs
                """
            ).fetchall()
            relations = conn.execute(
                """
                SELECT source_belief_id, target_belief_id, entailment_score, created_at
                FROM belief_relations
                WHERE label = 'entailment'
                """
            ).fetchall()
            contradictions = conn.execute(
                """
                SELECT belief_a_id, belief_b_id, score, created_at
                FROM contradictions
                WHERE status = 'open'
                """
            ).fetchall()
            citations = conn.execute(
                """
                SELECT cit.query_id, cit.chunk_id, cit.created_at
                FROM citations cit
                INNER JOIN chunks c ON c.id = cit.chunk_id
                INNER JOIN documents d ON d.id = c.document_id
                WHERE d.status = 'indexed'
                ORDER BY cit.query_id ASC, cit.rowid ASC
                """
            ).fetchall()

        return {
            "beliefs": [dict(r) for r in beliefs],
            "entailments": [dict(r) for r in relations],
            "contradictions": [dict(r) for r in contradictions],
            "citations": [dict(r) for r in citations],
        }
