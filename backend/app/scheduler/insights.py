"""Dashboard analytics and offline Morning Brief synthesis for MindVault."""

from __future__ import annotations

import json
import re
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Optional, Union

from app.beliefs.repository import BeliefRepository
from app.core.config import Settings
from app.db.database import DatabaseManager
from app.scheduler.models import (
    _STOPWORDS,
    format_iso_datetime,
    parse_iso_datetime,
    utc_now,
)
from app.scheduler.service import RetentionService

BRIEF_CATEGORIES: tuple[str, ...] = (
    "decaying_notes",
    "contradictions",
    "dormant_epochs",
    "major_clusters",
)

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{3,}")
_EXTRA_STOPWORDS = frozenset(
    {
        "this",
        "that",
        "with",
        "from",
        "have",
        "will",
        "into",
        "more",
        "most",
        "when",
        "where",
        "what",
        "which",
        "notes",
        "note",
        "section",
        "document",
        "chunk",
        "data",
        "system",
        "local",
    }
)


def _snippet(text: str, max_len: int = 160) -> str:
    collapsed = " ".join((text or "").split())
    if len(collapsed) <= max_len:
        return collapsed
    return collapsed[: max_len - 3].rstrip() + "..."


def _decode_json(raw: Optional[str]) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, ValueError):
        return {}


def _extract_concepts(text: str) -> list[str]:
    counts: Counter[str] = Counter()
    for match in _TOKEN_RE.findall(text or ""):
        token = match.lower().strip("_-")
        if len(token) < 4 or token in _STOPWORDS or token in _EXTRA_STOPWORDS:
            continue
        counts[token] += 1
    return [tok for tok, _cnt in counts.most_common(12)]


class VaultInsightsService:
    """Computes local dashboard visualizations and synthesizes persistent Morning Briefs."""

    def __init__(
        self,
        *,
        db: DatabaseManager,
        settings: Settings,
        retention_service: RetentionService,
        belief_repo: Optional[BeliefRepository] = None,
    ) -> None:
        self._db = db
        self._settings = settings
        self._retention_service = retention_service
        self._belief_repo = belief_repo or BeliefRepository(db)

    def get_belief_drift(
        self,
        *,
        epoch_min: Optional[int] = None,
        epoch_max: Optional[int] = None,
        status: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Compute belief evolution metrics and supersession transitions across epochs."""
        clauses: list[str] = ["d.status = 'indexed'"]
        params: list[Any] = []
        if epoch_min is not None:
            clauses.append("b.epoch >= ?")
            params.append(int(epoch_min))
        if epoch_max is not None:
            clauses.append("b.epoch <= ?")
            params.append(int(epoch_max))
        if status is not None:
            clauses.append("b.status = ?")
            params.append(status)
        if document_id is not None:
            clauses.append("c.document_id = ?")
            params.append(document_id)

        where_sql = " AND ".join(clauses)
        with self._db.connection() as conn:
            belief_rows = conn.execute(
                f"""
                SELECT
                    b.id,
                    b.chunk_id,
                    c.document_id,
                    d.filename,
                    c.chunk_index,
                    b.belief_text,
                    b.confidence,
                    b.epoch,
                    b.status,
                    b.superseded_by_id,
                    b.created_at
                FROM beliefs b
                INNER JOIN chunks c ON c.id = b.chunk_id
                INNER JOIN documents d ON d.id = c.document_id
                WHERE {where_sql}
                ORDER BY b.epoch ASC, b.created_at ASC
                """,
                tuple(params),
            ).fetchall()

            contradiction_rows = conn.execute(
                """
                SELECT
                    con.id,
                    con.belief_a_id,
                    con.belief_b_id,
                    con.score,
                    con.status,
                    ba.epoch AS epoch_a,
                    bb.epoch AS epoch_b
                FROM contradictions con
                INNER JOIN beliefs ba ON ba.id = con.belief_a_id
                INNER JOIN beliefs bb ON bb.id = con.belief_b_id
                """
            ).fetchall()

        contradiction_ids_in_scope: set[str] = set()
        open_contradictions_by_epoch: Counter[int] = Counter()
        total_contradictions_by_epoch: Counter[int] = Counter()
        belief_ids_set = {row["id"] for row in belief_rows}

        for crow in contradiction_rows:
            in_a = crow["belief_a_id"] in belief_ids_set
            in_b = crow["belief_b_id"] in belief_ids_set
            if not (in_a or in_b):
                continue
            contradiction_ids_in_scope.add(crow["id"])
            epochs_hit = set()
            if in_a:
                epochs_hit.add(int(crow["epoch_a"]))
            if in_b:
                epochs_hit.add(int(crow["epoch_b"]))
            for ep in epochs_hit:
                total_contradictions_by_epoch[ep] += 1
                if crow["status"] == "open":
                    open_contradictions_by_epoch[ep] += 1

        by_id: dict[str, dict[str, Any]] = {}
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in belief_rows:
            item = {
                "id": row["id"],
                "chunk_id": row["chunk_id"],
                "document_id": row["document_id"],
                "filename": row["filename"],
                "chunk_index": int(row["chunk_index"]),
                "belief_text": row["belief_text"],
                "confidence": float(row["confidence"]),
                "epoch": int(row["epoch"]),
                "status": row["status"],
                "superseded_by_id": row["superseded_by_id"],
                "created_at": row["created_at"],
            }
            by_id[item["id"]] = item
            grouped[item["epoch"]].append(item)

        epochs_list: list[dict[str, Any]] = []
        active_total = 0
        superseded_total = 0
        uncertain_total = 0

        for epoch_val in sorted(grouped.keys()):
            items = grouped[epoch_val]
            active_c = sum(1 for x in items if x["status"] == "active")
            superseded_c = sum(1 for x in items if x["status"] == "superseded")
            uncertain_c = sum(1 for x in items if x["status"] == "uncertain")
            archived_c = sum(1 for x in items if x["status"] == "archived")
            active_total += active_c
            superseded_total += superseded_c
            uncertain_total += uncertain_c
            avg_conf = sum(x["confidence"] for x in items) / len(items)
            con_c = int(total_contradictions_by_epoch.get(epoch_val, 0))
            drift_ratio = min(1.0, (superseded_c + uncertain_c + con_c) / max(1, len(items)))

            epochs_list.append(
                {
                    "epoch": epoch_val,
                    "total_count": len(items),
                    "active_count": active_c,
                    "superseded_count": superseded_c,
                    "uncertain_count": uncertain_c,
                    "archived_count": archived_c,
                    "contradiction_count": con_c,
                    "open_contradiction_count": int(
                        open_contradictions_by_epoch.get(epoch_val, 0)
                    ),
                    "average_confidence": round(avg_conf, 4),
                    "drift_score": round(drift_ratio, 4),
                    "beliefs": items,
                }
            )

        transitions: list[dict[str, Any]] = []
        for item in by_id.values():
            target_id = item.get("superseded_by_id")
            if target_id and target_id in by_id:
                target = by_id[target_id]
                transitions.append(
                    {
                        "from_belief_id": item["id"],
                        "to_belief_id": target["id"],
                        "from_epoch": item["epoch"],
                        "to_epoch": target["epoch"],
                        "from_text": item["belief_text"],
                        "to_text": target["belief_text"],
                    }
                )

        total_beliefs = len(belief_rows)
        open_con_total = sum(
            1
            for crow in contradiction_rows
            if crow["id"] in contradiction_ids_in_scope and crow["status"] == "open"
        )
        overall_drift = (
            min(1.0, (superseded_total + uncertain_total + open_con_total) / total_beliefs)
            if total_beliefs
            else 0.0
        )

        return {
            "summary": {
                "total_beliefs": total_beliefs,
                "total_epochs": len(epochs_list),
                "active_total": active_total,
                "superseded_total": superseded_total,
                "uncertain_total": uncertain_total,
                "open_contradictions": open_con_total,
                "transition_count": len(transitions),
                "overall_drift_rate": round(overall_drift, 4),
            },
            "epochs": epochs_list,
            "transitions": transitions,
        }

    def get_concept_velocity(
        self,
        *,
        epoch_min: Optional[int] = None,
        epoch_max: Optional[int] = None,
        top_concepts: int = 8,
    ) -> dict[str, Any]:
        """Measure concept emergence, frequency growth, and velocity across epochs."""
        clauses: list[str] = ["d.status = 'indexed'"]
        params: list[Any] = []
        if epoch_min is not None:
            clauses.append("c.epoch >= ?")
            params.append(int(epoch_min))
        if epoch_max is not None:
            clauses.append("c.epoch <= ?")
            params.append(int(epoch_max))

        where_sql = " AND ".join(clauses)
        with self._db.connection() as conn:
            chunk_rows = conn.execute(
                f"""
                SELECT
                    c.id AS chunk_id,
                    c.document_id,
                    d.filename,
                    c.chunk_index,
                    c.epoch,
                    c.content,
                    c.metadata_json
                FROM chunks c
                INNER JOIN documents d ON d.id = c.document_id
                WHERE {where_sql}
                ORDER BY c.epoch ASC, c.chunk_index ASC
                """,
                tuple(params),
            ).fetchall()

            belief_rows = conn.execute(
                f"""
                SELECT b.chunk_id, b.belief_text, b.epoch
                FROM beliefs b
                INNER JOIN chunks c ON c.id = b.chunk_id
                INNER JOIN documents d ON d.id = c.document_id
                WHERE {where_sql}
                """,
                tuple(params),
            ).fetchall()

        belief_texts_by_chunk: dict[str, list[str]] = defaultdict(list)
        belief_counts_by_epoch: Counter[int] = Counter()
        for brow in belief_rows:
            belief_texts_by_chunk[brow["chunk_id"]].append(brow["belief_text"])
            belief_counts_by_epoch[int(brow["epoch"])] += 1

        epoch_chunk_counts: Counter[int] = Counter()
        epoch_concepts: dict[int, Counter[str]] = defaultdict(Counter)
        concept_totals: Counter[str] = Counter()
        concept_epochs: dict[str, Counter[int]] = defaultdict(Counter)
        concept_samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
        seen_samples: dict[str, set[str]] = defaultdict(set)

        for row in chunk_rows:
            ep = int(row["epoch"])
            cid = row["chunk_id"]
            epoch_chunk_counts[ep] += 1
            meta = _decode_json(row["metadata_json"])
            heading = str(meta.get("section_heading") or "")
            combined_text = " ".join(
                [heading, str(row["content"] or ""), *belief_texts_by_chunk.get(cid, [])]
            )
            concepts = _extract_concepts(combined_text)
            for concept in concepts:
                epoch_concepts[ep][concept] += 1
                concept_totals[concept] += 1
                concept_epochs[concept][ep] += 1
                if cid not in seen_samples[concept] and len(concept_samples[concept]) < 4:
                    seen_samples[concept].add(cid)
                    concept_samples[concept].append(
                        {
                            "chunk_id": cid,
                            "document_id": row["document_id"],
                            "filename": row["filename"],
                            "chunk_index": int(row["chunk_index"]),
                            "epoch": ep,
                            "snippet": _snippet(str(row["content"])),
                        }
                    )

        sorted_epochs = sorted(epoch_chunk_counts.keys())
        seen_concepts: set[str] = set()
        epoch_series: list[dict[str, Any]] = []
        peak_epoch: Optional[int] = None
        peak_velocity = -1.0

        for ep in sorted_epochs:
            ep_concept_map = epoch_concepts[ep]
            emerging = sorted(c for c in ep_concept_map if c not in seen_concepts)
            seen_concepts.update(ep_concept_map.keys())
            new_chunks = int(epoch_chunk_counts[ep])
            new_beliefs = int(belief_counts_by_epoch.get(ep, 0))
            distinct_c = len(ep_concept_map)
            velocity_score = round(
                float(new_chunks) + 0.5 * float(new_beliefs) + 0.75 * float(len(emerging)),
                3,
            )
            if velocity_score > peak_velocity:
                peak_velocity = velocity_score
                peak_epoch = ep

            epoch_series.append(
                {
                    "epoch": ep,
                    "new_chunks": new_chunks,
                    "new_beliefs": new_beliefs,
                    "distinct_concepts": distinct_c,
                    "emerging_concepts_count": len(emerging),
                    "emerging_concepts": emerging[:6],
                    "cumulative_concepts": len(seen_concepts),
                    "velocity_score": velocity_score,
                }
            )

        top_items: list[dict[str, Any]] = []
        for concept, total_mentions in concept_totals.most_common(max(1, top_concepts)):
            ep_counts = concept_epochs[concept]
            ep_keys = sorted(ep_counts.keys())
            first_ep = ep_keys[0]
            last_ep = ep_keys[-1]
            span = max(1, len(ep_keys))
            series = [{"epoch": ep, "count": int(ep_counts.get(ep, 0))} for ep in sorted_epochs]
            top_items.append(
                {
                    "concept": concept,
                    "total_mentions": int(total_mentions),
                    "first_epoch": first_ep,
                    "last_epoch": last_ep,
                    "active_epochs": len(ep_keys),
                    "growth_rate": round(float(total_mentions) / float(span), 3),
                    "epoch_series": series,
                    "sample_chunks": concept_samples[concept],
                }
            )

        avg_velocity = (
            round(sum(x["velocity_score"] for x in epoch_series) / len(epoch_series), 3)
            if epoch_series
            else 0.0
        )
        return {
            "summary": {
                "total_distinct_concepts": len(concept_totals),
                "total_epochs": len(epoch_series),
                "peak_epoch": peak_epoch,
                "average_velocity": avg_velocity,
            },
            "epochs": epoch_series,
            "concepts": top_items,
        }

    def get_decay_heatmap(
        self,
        *,
        epoch_min: Optional[int] = None,
        epoch_max: Optional[int] = None,
        document_id: Optional[str] = None,
        as_of: Optional[Union[datetime, str]] = None,
    ) -> dict[str, Any]:
        """Build knowledge decay heatmap cells, epoch buckets, and document summaries."""
        retention_payload = self._retention_service.get_retention_list(
            document_id=document_id,
            epoch_min=epoch_min,
            epoch_max=epoch_max,
            due_only=False,
            as_of=as_of,
            limit=500,
            offset=0,
        )
        items = retention_payload["items"]
        threshold = float(retention_payload["retention_threshold"])
        chunk_ids = [item["chunk_id"] for item in items]

        chunk_meta_map: dict[str, tuple[str, str]] = {}
        if chunk_ids:
            with self._db.connection() as conn:
                for start in range(0, len(chunk_ids), 400):
                    batch = chunk_ids[start : start + 400]
                    placeholders = ", ".join("?" for _ in batch)
                    rows = conn.execute(
                        f"""
                        SELECT id, content, metadata_json
                        FROM chunks
                        WHERE id IN ({placeholders})
                        """,
                        tuple(batch),
                    ).fetchall()
                    for row in rows:
                        meta = _decode_json(row["metadata_json"])
                        heading = str(meta.get("section_heading") or "")
                        chunk_meta_map[row["id"]] = (heading, _snippet(str(row["content"])))

        cells: list[dict[str, Any]] = []
        critical_count = 0
        warning_count = 0
        healthy_count = 0
        by_epoch: dict[int, list[dict[str, Any]]] = defaultdict(list)
        by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for item in items:
            ret = float(item["retention"])
            if ret < 0.50:
                band = "critical"
                critical_count += 1
            elif ret <= threshold + 1e-9:
                band = "warning"
                warning_count += 1
            else:
                band = "healthy"
                healthy_count += 1

            heading, snippet = chunk_meta_map.get(item["chunk_id"], ("", ""))
            cell = {
                **item,
                "section_heading": heading,
                "snippet": snippet,
                "decay_band": band,
            }
            cells.append(cell)
            by_epoch[int(item["epoch"])].append(cell)
            by_doc[str(item["document_id"])].append(cell)

        epoch_buckets = [
            {
                "epoch": ep,
                "chunk_count": len(ep_cells),
                "due_count": sum(1 for c in ep_cells if c["eligible_for_review"]),
                "average_retention": round(
                    sum(c["retention"] for c in ep_cells) / len(ep_cells), 4
                ),
                "min_retention": round(min(c["retention"] for c in ep_cells), 4),
            }
            for ep, ep_cells in sorted(by_epoch.items())
        ]

        doc_buckets = [
            {
                "document_id": did,
                "filename": d_cells[0]["filename"],
                "chunk_count": len(d_cells),
                "due_count": sum(1 for c in d_cells if c["eligible_for_review"]),
                "average_retention": round(
                    sum(c["retention"] for c in d_cells) / len(d_cells), 4
                ),
            }
            for did, d_cells in sorted(by_doc.items(), key=lambda pair: pair[1][0]["filename"])
        ]

        summary = dict(retention_payload["summary"])
        summary["critical_count"] = critical_count
        summary["warning_count"] = warning_count
        summary["healthy_count"] = healthy_count
        summary["retention_threshold"] = threshold

        return {
            "model_name": retention_payload["model_name"],
            "evaluated_at": retention_payload["evaluated_at"],
            "retention_threshold": threshold,
            "summary": summary,
            "cells": cells,
            "epoch_buckets": epoch_buckets,
            "document_buckets": doc_buckets,
        }

    def get_knowledge_health(
        self,
        *,
        epoch_min: Optional[int] = None,
        epoch_max: Optional[int] = None,
        as_of: Optional[Union[datetime, str]] = None,
    ) -> dict[str, Any]:
        """Compute 5-axis Knowledge Health Radar metrics in [0.0, 1.0]."""
        decay = self.get_decay_heatmap(
            epoch_min=epoch_min,
            epoch_max=epoch_max,
            as_of=as_of,
        )
        drift = self.get_belief_drift(epoch_min=epoch_min, epoch_max=epoch_max)

        chunk_ids = [cell["chunk_id"] for cell in decay["cells"]]
        total_chunks = len(chunk_ids)
        due_count = int(decay["summary"]["due_count"])
        avg_retention = float(decay["summary"]["average_retention"]) if total_chunks else 1.0

        active_beliefs = int(drift["summary"]["active_total"])
        open_contradictions = int(drift["summary"]["open_contradictions"])

        connected_chunks = 0
        covered_chunks = 0
        if chunk_ids:
            belief_chunk_ids: set[str] = set()
            reviewed_chunk_ids: set[str] = set()
            graph_chunk_ids: set[str] = set()
            with self._db.connection() as conn:
                for start in range(0, len(chunk_ids), 400):
                    batch = chunk_ids[start : start + 400]
                    placeholders = ", ".join("?" for _ in batch)
                    for row in conn.execute(
                        f"SELECT DISTINCT chunk_id FROM beliefs WHERE chunk_id IN ({placeholders})",
                        tuple(batch),
                    ).fetchall():
                        belief_chunk_ids.add(row["chunk_id"])
                    for row in conn.execute(
                        f"""
                        SELECT DISTINCT chunk_id FROM review_history
                        WHERE chunk_id IN ({placeholders})
                        """,
                        tuple(batch),
                    ).fetchall():
                        reviewed_chunk_ids.add(row["chunk_id"])
                    for row in conn.execute(
                        f"""
                        SELECT source_id AS cid FROM graph_edges WHERE source_id IN ({placeholders})
                        UNION
                        SELECT target_id AS cid FROM graph_edges WHERE target_id IN ({placeholders})
                        """,
                        tuple(batch + batch),
                    ).fetchall():
                        graph_chunk_ids.add(row["cid"])

            connected_chunks = len(graph_chunk_ids)
            covered_chunks = len(belief_chunk_ids | reviewed_chunk_ids | graph_chunk_ids)

        if total_chunks == 0:
            axes = [
                {
                    "key": "retention_health",
                    "label": "Memory Retention",
                    "score": 0.0,
                    "value_pct": 0.0,
                    "detail": "No indexed chunks in selected epoch scope.",
                    "target_view": "decay",
                },
                {
                    "key": "belief_consistency",
                    "label": "Belief Consistency",
                    "score": 0.0,
                    "value_pct": 0.0,
                    "detail": "No active beliefs extracted yet.",
                    "target_view": "contradictions",
                },
                {
                    "key": "graph_connectivity",
                    "label": "Graph Connectivity",
                    "score": 0.0,
                    "value_pct": 0.0,
                    "detail": "No graph links present.",
                    "target_view": "concept-velocity",
                },
                {
                    "key": "review_freshness",
                    "label": "Review Freshness",
                    "score": 0.0,
                    "value_pct": 0.0,
                    "detail": "Ingest notes to begin spaced recall tracking.",
                    "target_view": "flashcards",
                },
                {
                    "key": "knowledge_coverage",
                    "label": "Knowledge Coverage",
                    "score": 0.0,
                    "value_pct": 0.0,
                    "detail": "No chunks indexed yet.",
                    "target_view": "belief-drift",
                },
            ]
            return {
                "overall_score": 0.0,
                "status_label": "empty",
                "total_chunks": 0,
                "axes": axes,
            }

        retention_score = max(0.0, min(1.0, avg_retention))
        consistency_score = max(
            0.0,
            1.0 - min(1.0, float(open_contradictions) / float(max(1, active_beliefs))),
        )
        connectivity_score = max(0.0, min(1.0, float(connected_chunks) / float(total_chunks)))
        freshness_score = max(0.0, 1.0 - (float(due_count) / float(total_chunks)))
        coverage_score = max(0.0, min(1.0, float(covered_chunks) / float(total_chunks)))

        axes = [
            {
                "key": "retention_health",
                "label": "Memory Retention",
                "score": round(retention_score, 4),
                "value_pct": round(retention_score * 100.0, 1),
                "detail": (
                    f"Mean live retention R={retention_score:.2f} across "
                    f"{total_chunks} chunks."
                ),
                "target_view": "decay",
            },
            {
                "key": "belief_consistency",
                "label": "Belief Consistency",
                "score": round(consistency_score, 4),
                "value_pct": round(consistency_score * 100.0, 1),
                "detail": (
                    f"{open_contradictions} open contradiction(s) vs "
                    f"{active_beliefs} active belief(s)."
                ),
                "target_view": "contradictions",
            },
            {
                "key": "graph_connectivity",
                "label": "Graph Connectivity",
                "score": round(connectivity_score, 4),
                "value_pct": round(connectivity_score * 100.0, 1),
                "detail": f"{connected_chunks}/{total_chunks} chunk(s) connected in graph edges.",
                "target_view": "concept-velocity",
            },
            {
                "key": "review_freshness",
                "label": "Review Freshness",
                "score": round(freshness_score, 4),
                "value_pct": round(freshness_score * 100.0, 1),
                "detail": f"{total_chunks - due_count}/{total_chunks} chunk(s) currently fresh.",
                "target_view": "flashcards",
            },
            {
                "key": "knowledge_coverage",
                "label": "Knowledge Coverage",
                "score": round(coverage_score, 4),
                "value_pct": round(coverage_score * 100.0, 1),
                "detail": (
                    f"{covered_chunks}/{total_chunks} chunk(s) enriched with "
                    "beliefs/reviews."
                ),
                "target_view": "belief-drift",
            },
        ]

        overall = round(sum(a["score"] for a in axes) / len(axes), 4)
        if overall >= 0.80:
            label = "excellent"
        elif overall >= 0.55:
            label = "good"
        else:
            label = "needs_attention"

        return {
            "overall_score": overall,
            "status_label": label,
            "total_chunks": total_chunks,
            "axes": axes,
        }

    def get_dashboard_overview(
        self,
        *,
        epoch_min: Optional[int] = None,
        epoch_max: Optional[int] = None,
        as_of: Optional[Union[datetime, str]] = None,
    ) -> dict[str, Any]:
        """Aggregate all 6 dashboard views and latest morning brief in one response."""
        belief_drift = self.get_belief_drift(epoch_min=epoch_min, epoch_max=epoch_max)
        concept_velocity = self.get_concept_velocity(epoch_min=epoch_min, epoch_max=epoch_max)
        decay = self.get_decay_heatmap(epoch_min=epoch_min, epoch_max=epoch_max, as_of=as_of)
        health = self.get_knowledge_health(epoch_min=epoch_min, epoch_max=epoch_max, as_of=as_of)
        review_queue = self._retention_service.get_review_queue(
            epoch_min=epoch_min,
            epoch_max=epoch_max,
            include_answer=False,
            include_upcoming=False,
            as_of=as_of,
            limit=20,
        )
        contradictions, total_open_con = self._belief_repo.list_contradictions(
            status="open",
            epoch_min=epoch_min,
            epoch_max=epoch_max,
            limit=20,
            offset=0,
        )
        briefs_page = self.list_briefs(limit=1, offset=0)
        latest_brief = briefs_page["items"][0] if briefs_page["items"] else None

        doc_ids = {cell["document_id"] for cell in decay["cells"]}
        epoch_values = sorted(
            {cell["epoch"] for cell in decay["cells"]}
            | {ep["epoch"] for ep in belief_drift["epochs"]}
        )

        return {
            "kpis": {
                "total_documents": len(doc_ids),
                "total_chunks": decay["summary"]["total_chunks"],
                "due_flashcards": review_queue["due_total"],
                "open_contradictions": total_open_con,
                "active_beliefs": belief_drift["summary"]["active_total"],
                "average_retention": round(decay["summary"]["average_retention"], 4),
                "health_score": health["overall_score"],
                "epoch_min": epoch_values[0] if epoch_values else None,
                "epoch_max": epoch_values[-1] if epoch_values else None,
            },
            "belief_drift": belief_drift,
            "concept_velocity": concept_velocity,
            "decay": decay,
            "health": health,
            "review_queue": review_queue,
            "contradictions": {
                "total": total_open_con,
                "items": contradictions,
            },
            "latest_brief": latest_brief,
        }

    def list_briefs(self, *, limit: int = 30, offset: int = 0) -> dict[str, Any]:
        """List persisted Morning Briefs ordered newest brief_date first."""
        with self._db.connection() as conn:
            total = int(
                conn.execute("SELECT COUNT(*) AS cnt FROM morning_briefs").fetchone()["cnt"]
            )
            rows = conn.execute(
                """
                SELECT id, brief_date, content, metadata_json, created_at
                FROM morning_briefs
                ORDER BY brief_date DESC, created_at DESC
                LIMIT ? OFFSET ?
                """,
                (int(limit), int(offset)),
            ).fetchall()

        items = [self._row_to_brief(row) for row in rows]
        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "items": items,
        }

    @staticmethod
    def _row_to_brief(row: Any) -> dict[str, Any]:
        meta = _decode_json(row["metadata_json"])
        return {
            "id": row["id"],
            "brief_date": row["brief_date"],
            "content": row["content"],
            "categories": list(meta.get("categories") or BRIEF_CATEGORIES),
            "max_insights": int(meta.get("max_insights", 3)),
            "insights": list(meta.get("insights") or []),
            "citations": list(meta.get("citations") or []),
            "regenerated": bool(meta.get("regenerated", False)),
            "created_at": row["created_at"],
        }

    def generate_morning_brief(
        self,
        *,
        brief_date: Optional[str] = None,
        categories: Optional[list[str]] = None,
        max_insights: int = 3,
        regenerate: bool = False,
        epoch_min: Optional[int] = None,
        epoch_max: Optional[int] = None,
        as_of: Optional[Union[datetime, str]] = None,
    ) -> dict[str, Any]:
        """Synthesize a prioritized daily local Morning Brief with citations and persistence."""
        eval_dt = (
            parse_iso_datetime(as_of)
            if isinstance(as_of, str)
            else (as_of.astimezone(timezone.utc) if as_of is not None else utc_now())
        )
        resolved_date = (brief_date or eval_dt.date().isoformat()).strip()
        if not resolved_date:
            raise ValueError("brief_date must not be empty.")

        requested_categories = (
            list(dict.fromkeys(categories)) if categories else list(BRIEF_CATEGORIES)
        )
        for cat in requested_categories:
            if cat not in BRIEF_CATEGORIES:
                raise ValueError(
                    f"Invalid brief category '{cat}'. Expected one of {list(BRIEF_CATEGORIES)}."
                )

        if not regenerate:
            with self._db.connection() as conn:
                existing = conn.execute(
                    """
                    SELECT id, brief_date, content, metadata_json, created_at
                    FROM morning_briefs
                    WHERE brief_date = ?
                    """,
                    (resolved_date,),
                ).fetchone()
            if existing is not None:
                parsed = self._row_to_brief(existing)
                parsed["regenerated"] = False
                return parsed

        candidates: list[dict[str, Any]] = []
        if "contradictions" in requested_categories:
            candidates.extend(
                self._build_contradiction_insights(epoch_min=epoch_min, epoch_max=epoch_max)
            )
        if "decaying_notes" in requested_categories:
            candidates.extend(
                self._build_decaying_note_insights(
                    epoch_min=epoch_min,
                    epoch_max=epoch_max,
                    as_of=eval_dt,
                )
            )
        if "dormant_epochs" in requested_categories:
            candidates.extend(
                self._build_dormant_epoch_insights(
                    epoch_min=epoch_min,
                    epoch_max=epoch_max,
                    as_of=eval_dt,
                )
            )
        if "major_clusters" in requested_categories:
            candidates.extend(
                self._build_cluster_insights(epoch_min=epoch_min, epoch_max=epoch_max)
            )

        candidates.sort(key=lambda c: (-float(c["priority_score"]), c["category"], c["title"]))
        selected = candidates[: max(1, int(max_insights))]

        all_citations: list[dict[str, Any]] = []
        seen_cids: set[str] = set()
        for insight in selected:
            for cit in insight["citations"]:
                if cit["chunk_id"] not in seen_cids:
                    seen_cids.add(cit["chunk_id"])
                    all_citations.append(cit)

        content_md = self._format_brief_markdown(
            brief_date=resolved_date,
            insights=selected,
            categories=requested_categories,
        )
        now_iso = format_iso_datetime(eval_dt)
        brief_id = uuid.uuid4().hex
        query_key = f"brief:{resolved_date}"

        metadata_payload = {
            "categories": requested_categories,
            "max_insights": int(max_insights),
            "insights": selected,
            "citations": all_citations,
            "regenerated": bool(regenerate),
            "epoch_min": epoch_min,
            "epoch_max": epoch_max,
        }
        metadata_json = json.dumps(metadata_payload, ensure_ascii=True)

        with self._db.transaction(immediate=True) as conn:
            conn.execute(
                """
                INSERT INTO morning_briefs (id, brief_date, content, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (brief_date) DO UPDATE SET
                    id = excluded.id,
                    content = excluded.content,
                    metadata_json = excluded.metadata_json,
                    created_at = excluded.created_at
                """,
                (brief_id, resolved_date, content_md, metadata_json, now_iso),
            )
            conn.execute(
                "DELETE FROM citations WHERE query_id = ? AND citation_type = 'brief'",
                (query_key,),
            )
            for cit in all_citations:
                conn.execute(
                    """
                    INSERT INTO citations (id, query_id, chunk_id, citation_type, created_at)
                    VALUES (?, ?, ?, 'brief', ?)
                    """,
                    (uuid.uuid4().hex, query_key, cit["chunk_id"], now_iso),
                )

        return {
            "id": brief_id,
            "brief_date": resolved_date,
            "content": content_md,
            "categories": requested_categories,
            "max_insights": int(max_insights),
            "insights": selected,
            "citations": all_citations,
            "regenerated": bool(regenerate),
            "created_at": now_iso,
        }

    def _build_decaying_note_insights(
        self,
        *,
        epoch_min: Optional[int],
        epoch_max: Optional[int],
        as_of: datetime,
    ) -> list[dict[str, Any]]:
        decay = self.get_decay_heatmap(
            epoch_min=epoch_min,
            epoch_max=epoch_max,
            as_of=as_of,
        )
        due_cells = [c for c in decay["cells"] if c["eligible_for_review"]]
        target_cells = due_cells or decay["cells"]
        if not target_cells:
            return []

        lowest = sorted(target_cells, key=lambda c: (c["retention"], c["epoch"]))[:2]
        min_ret = float(lowest[0]["retention"])
        citations = [
            {
                "chunk_id": c["chunk_id"],
                "document_id": c["document_id"],
                "filename": c["filename"],
                "chunk_index": c["chunk_index"],
                "epoch": c["epoch"],
                "snippet": c["snippet"],
            }
            for c in lowest
        ]
        urgent = len(due_cells) > 0
        priority = round(0.92 - 0.35 * min_ret if urgent else 0.45 - 0.20 * min_ret, 4)
        return [
            {
                "id": f"insight-decay-{lowest[0]['chunk_id'][:8]}",
                "category": "decaying_notes",
                "title": (
                    f"{len(due_cells)} memory chunk(s) due for spaced review"
                    if urgent
                    else "Retention check: lowest-scoring note monitored"
                ),
                "summary": (
                    f"Lowest live retention is R={min_ret:.2f} in '{lowest[0]['filename']}' "
                    f"(epoch {lowest[0]['epoch']}, stability S={lowest[0]['stability']:.2f}d)."
                ),
                "priority_score": priority,
                "recommended_action": "Review due flashcards in the Review Flashcards view.",
                "citations": citations,
            }
        ]

    def _build_contradiction_insights(
        self,
        *,
        epoch_min: Optional[int],
        epoch_max: Optional[int],
    ) -> list[dict[str, Any]]:
        open_cons, total = self._belief_repo.list_contradictions(
            status="open",
            epoch_min=epoch_min,
            epoch_max=epoch_max,
            limit=3,
            offset=0,
        )
        if not open_cons:
            return []

        top = open_cons[0]
        ba = top.get("belief_a") or {}
        bb = top.get("belief_b") or {}
        chunk_ids = [cid for cid in (ba.get("chunk_id"), bb.get("chunk_id")) if cid]
        citations = self._lookup_citations(chunk_ids)
        score = float(top["score"])
        priority = round(0.75 + 0.25 * score, 4)
        return [
            {
                "id": f"insight-con-{top['id'][:8]}",
                "category": "contradictions",
                "title": f"{total} unresolved belief contradiction(s) detected",
                "summary": (
                    f"Highest conflict score ({score:.2f}) between "
                    f"'{ba.get('belief_text', 'Belief A')}' and "
                    f"'{bb.get('belief_text', 'Belief B')}'."
                ),
                "priority_score": priority,
                "recommended_action": "Resolve or supersede claims in the Contradiction Resolver.",
                "citations": citations,
            }
        ]

    def _build_dormant_epoch_insights(
        self,
        *,
        epoch_min: Optional[int],
        epoch_max: Optional[int],
        as_of: datetime,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = ["d.status = 'indexed'"]
        params: list[Any] = []
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
                    c.id AS chunk_id,
                    c.document_id,
                    d.filename,
                    c.chunk_index,
                    c.epoch,
                    c.content,
                    c.created_at,
                    rs.last_reviewed_at,
                    rs.review_count
                FROM chunks c
                INNER JOIN documents d ON d.id = c.document_id
                LEFT JOIN retention_scores rs ON rs.chunk_id = c.id
                WHERE {where_sql}
                ORDER BY c.epoch ASC, c.chunk_index ASC
                """,
                tuple(params),
            ).fetchall()

        if not rows:
            return []

        max_epoch = max(int(r["epoch"]) for r in rows)
        by_epoch: dict[int, list[Any]] = defaultdict(list)
        for row in rows:
            by_epoch[int(row["epoch"])].append(row)

        dormant_candidates: list[tuple[float, int, list[Any]]] = []
        for ep, ep_rows in by_epoch.items():
            ages_days: list[float] = []
            unreviewed = 0
            for r in ep_rows:
                ref_iso = r["last_reviewed_at"] or r["created_at"]
                ref_dt = parse_iso_datetime(ref_iso)
                ages_days.append(max(0.0, (as_of - ref_dt).total_seconds() / 86400.0))
                if not r["review_count"]:
                    unreviewed += 1
            avg_age = sum(ages_days) / len(ages_days)
            epoch_gap = float(max_epoch - ep)
            dormancy_score = avg_age + (0.25 if epoch_gap > 0 else 0.0) + (0.15 * unreviewed)
            dormant_candidates.append((dormancy_score, ep, ep_rows))

        dormant_candidates.sort(key=lambda t: (-t[0], t[1]))
        top_score, top_ep, top_rows = dormant_candidates[0]
        citations = [
            {
                "chunk_id": r["chunk_id"],
                "document_id": r["document_id"],
                "filename": r["filename"],
                "chunk_index": int(r["chunk_index"]),
                "epoch": int(r["epoch"]),
                "snippet": _snippet(str(r["content"])),
            }
            for r in top_rows[:2]
        ]
        priority = round(min(0.84, 0.42 + 0.08 * top_score), 4)
        return [
            {
                "id": f"insight-dormant-ep-{top_ep}",
                "category": "dormant_epochs",
                "title": f"Dormant Epoch {top_ep} needs refresher attention",
                "summary": (
                    f"Epoch {top_ep} contains {len(top_rows)} chunk(s) with minimal recent review "
                    f"activity compared to active knowledge."
                ),
                "priority_score": priority,
                "recommended_action": f"Filter dashboard to Epoch {top_ep} and revisit key notes.",
                "citations": citations,
            }
        ]

    def _build_cluster_insights(
        self,
        *,
        epoch_min: Optional[int],
        epoch_max: Optional[int],
    ) -> list[dict[str, Any]]:
        clauses: list[str] = ["d.status = 'indexed'"]
        params: list[Any] = []
        if epoch_min is not None:
            clauses.append("c.epoch >= ?")
            params.append(int(epoch_min))
        if epoch_max is not None:
            clauses.append("c.epoch <= ?")
            params.append(int(epoch_max))
        where_sql = " AND ".join(clauses)

        with self._db.connection() as conn:
            chunk_rows = conn.execute(
                f"""
                SELECT
                    c.id AS chunk_id,
                    c.document_id,
                    d.filename,
                    c.chunk_index,
                    c.epoch,
                    c.content
                FROM chunks c
                INNER JOIN documents d ON d.id = c.document_id
                WHERE {where_sql}
                """,
                tuple(params),
            ).fetchall()

            if not chunk_rows:
                return []

            edges = conn.execute(
                "SELECT source_id, target_id FROM graph_edges"
            ).fetchall()

        chunk_map = {row["chunk_id"]: row for row in chunk_rows}
        degree_counts: Counter[str] = Counter()
        for edge in edges:
            if edge["source_id"] in chunk_map:
                degree_counts[edge["source_id"]] += 1
            if edge["target_id"] in chunk_map:
                degree_counts[edge["target_id"]] += 1

        if degree_counts:
            top_ids = [cid for cid, _deg in degree_counts.most_common(2)]
            lead_deg = degree_counts[top_ids[0]]
            lead_row = chunk_map[top_ids[0]]
            title = f"Major concept cluster anchored in '{lead_row['filename']}'"
            summary = (
                f"Chunk #{int(lead_row['chunk_index'])} (epoch {int(lead_row['epoch'])}) "
                f"forms a high-connectivity hub with {lead_deg} graph relationship(s)."
            )
            priority = round(min(0.78, 0.45 + 0.05 * float(lead_deg)), 4)
            citations = [
                {
                    "chunk_id": chunk_map[cid]["chunk_id"],
                    "document_id": chunk_map[cid]["document_id"],
                    "filename": chunk_map[cid]["filename"],
                    "chunk_index": int(chunk_map[cid]["chunk_index"]),
                    "epoch": int(chunk_map[cid]["epoch"]),
                    "snippet": _snippet(str(chunk_map[cid]["content"])),
                }
                for cid in top_ids
            ]
        else:
            doc_counts: Counter[str] = Counter(row["document_id"] for row in chunk_rows)
            lead_doc_id, count = doc_counts.most_common(1)[0]
            doc_rows = [r for r in chunk_rows if r["document_id"] == lead_doc_id]
            title = f"Primary document cluster in '{doc_rows[0]['filename']}'"
            summary = (
                f"'{doc_rows[0]['filename']}' holds {count} indexed chunk(s), "
                "representing your largest concept concentration."
            )
            priority = 0.40
            citations = [
                {
                    "chunk_id": r["chunk_id"],
                    "document_id": r["document_id"],
                    "filename": r["filename"],
                    "chunk_index": int(r["chunk_index"]),
                    "epoch": int(r["epoch"]),
                    "snippet": _snippet(str(r["content"])),
                }
                for r in doc_rows[:2]
            ]

        return [
            {
                "id": f"insight-cluster-{citations[0]['chunk_id'][:8]}",
                "category": "major_clusters",
                "title": title,
                "summary": summary,
                "priority_score": priority,
                "recommended_action": "Explore Concept Velocity or Knowledge Graph neighbors.",
                "citations": citations,
            }
        ]

    def _lookup_citations(self, chunk_ids: list[str]) -> list[dict[str, Any]]:
        unique = list(dict.fromkeys(cid for cid in chunk_ids if cid))
        if not unique:
            return []
        placeholders = ", ".join("?" for _ in unique)
        with self._db.connection() as conn:
            rows = conn.execute(
                f"""
                SELECT
                    c.id AS chunk_id,
                    c.document_id,
                    d.filename,
                    c.chunk_index,
                    c.epoch,
                    c.content
                FROM chunks c
                INNER JOIN documents d ON d.id = c.document_id
                WHERE c.id IN ({placeholders})
                ORDER BY c.epoch ASC, c.chunk_index ASC
                """,
                tuple(unique),
            ).fetchall()
        return [
            {
                "chunk_id": row["chunk_id"],
                "document_id": row["document_id"],
                "filename": row["filename"],
                "chunk_index": int(row["chunk_index"]),
                "epoch": int(row["epoch"]),
                "snippet": _snippet(str(row["content"])),
            }
            for row in rows
        ]

    @staticmethod
    def _format_brief_markdown(
        *,
        brief_date: str,
        insights: list[dict[str, Any]],
        categories: list[str],
    ) -> str:
        lines: list[str] = [
            f"# MindVault Morning Brief ({brief_date})",
            f"Categories: {', '.join(categories)}",
            "",
        ]
        if not insights:
            lines.append("No vault insights available yet. Ingest local documents to begin.")
            return "\n".join(lines)

        for idx, ins in enumerate(insights, start=1):
            lines.append(f"## {idx}. [{ins['category']}] {ins['title']}")
            lines.append(ins["summary"])
            lines.append(f"Action: {ins['recommended_action']}")
            if ins["citations"]:
                refs = ", ".join(
                    f"{c['filename']}#chunk-{c['chunk_index']}" for c in ins["citations"]
                )
                lines.append(f"Citations: {refs}")
            lines.append("")
        return "\n".join(lines).strip()
