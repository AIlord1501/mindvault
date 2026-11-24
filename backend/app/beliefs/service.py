"""High-level orchestration service for belief extraction and contradiction workflows."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from app.beliefs.contradictions import ContradictionDetector
from app.beliefs.extractor import MAX_BELIEFS_PER_CHUNK, BeliefExtractor
from app.beliefs.repository import BeliefRepository
from app.core.config import Settings
from app.db.repositories import DocumentRepository


class ChunkNotFoundError(LookupError):
    """Raised when belief extraction targets an unknown chunk ID."""


class BeliefService:
    """Coordinates BeliefExtractor, ContradictionDetector, and SQLite persistence."""

    def __init__(
        self,
        *,
        settings: Settings,
        doc_repo: DocumentRepository,
        belief_repo: BeliefRepository,
        extractor: BeliefExtractor,
        detector: ContradictionDetector,
    ) -> None:
        self._settings = settings
        self._doc_repo = doc_repo
        self._belief_repo = belief_repo
        self._extractor = extractor
        self._detector = detector

    def extract_from_chunk(
        self,
        chunk_id: str,
        *,
        max_beliefs: int = MAX_BELIEFS_PER_CHUNK,
        dedup_threshold: Optional[float] = None,
        min_confidence: Optional[float] = None,
        detect_contradictions: bool = True,
        candidate_top_k: Optional[int] = None,
        candidate_min_similarity: Optional[float] = None,
        contradiction_threshold: Optional[float] = None,
        entailment_threshold: Optional[float] = None,
    ) -> dict[str, Any]:
        """Extract beliefs from a persisted chunk and evaluate contradictions."""
        chunk = self._doc_repo.get_chunk_by_id(chunk_id)
        if chunk is None:
            raise ChunkNotFoundError(f"Chunk '{chunk_id}' not found.")

        existing_pool = self._belief_repo.list_active_beliefs_for_comparison()
        extraction = self._extractor.extract(
            chunk_id=chunk_id,
            chunk_text=str(chunk["content"]),
            epoch=int(chunk["epoch"]),
            existing_beliefs=existing_pool,
            max_beliefs=max_beliefs,
            dedup_threshold=dedup_threshold,
            min_confidence=min_confidence,
        )

        now_iso = datetime.now(timezone.utc).isoformat()
        created_beliefs: list[tuple[dict[str, Any], Optional[list[float]]]] = []
        for cand in extraction.candidates:
            created = self._belief_repo.create_belief(
                belief_id=uuid.uuid4().hex,
                chunk_id=cand.chunk_id,
                belief_text=cand.belief_text,
                normalized_text=cand.normalized_text,
                confidence=cand.confidence,
                epoch=cand.epoch,
                status="active",
                embedding=cand.embedding,
                created_at=now_iso,
            )
            created_beliefs.append((created, cand.embedding))

        warnings = list(extraction.warnings)
        relations_recorded: list[dict[str, Any]] = []
        contradictions_created: list[dict[str, Any]] = []
        candidates_evaluated = 0
        nli_mode = "skipped"
        nli_model = ""

        if detect_contradictions and created_beliefs:
            detection = self._detector.evaluate_new_beliefs(
                new_beliefs=created_beliefs,
                active_pool=existing_pool,
                candidate_top_k=candidate_top_k,
                candidate_min_similarity=candidate_min_similarity,
                contradiction_threshold=contradiction_threshold,
                entailment_threshold=entailment_threshold,
            )
            nli_mode = detection.nli_mode
            nli_model = detection.nli_model
            candidates_evaluated = detection.candidates_evaluated
            relations_recorded = detection.relations_recorded
            contradictions_created = detection.contradictions_created
            warnings.extend(detection.warnings)

        # Re-read created beliefs so status reflects any newly detected contradictions.
        belief_ids = [b["id"] for b, _vec in created_beliefs]
        refreshed_beliefs: list[dict[str, Any]] = []
        for bid in belief_ids:
            detail = self._belief_repo.get_belief_detail(bid)
            if detail is not None:
                refreshed_beliefs.append(
                    {
                        "id": detail["id"],
                        "chunk_id": detail["chunk_id"],
                        "document_id": detail.get("document_id"),
                        "filename": detail.get("filename"),
                        "belief_text": detail["belief_text"],
                        "normalized_text": detail["normalized_text"],
                        "confidence": detail["confidence"],
                        "epoch": detail["epoch"],
                        "status": detail["status"],
                        "superseded_by_id": detail.get("superseded_by_id"),
                        "context_note": detail.get("context_note"),
                        "supersedes_ids": detail.get("supersedes_ids", []),
                        "created_at": detail["created_at"],
                        "updated_at": detail.get("updated_at"),
                    }
                )

        return {
            "chunk_id": chunk_id,
            "document_id": chunk["document_id"],
            "epoch": int(chunk["epoch"]),
            "extracted_count": len(refreshed_beliefs),
            "duplicates_skipped": extraction.duplicates_skipped,
            "low_confidence_skipped": extraction.low_confidence_skipped,
            "candidates_evaluated": candidates_evaluated,
            "nli_mode": nli_mode,
            "nli_model": nli_model,
            "beliefs": refreshed_beliefs,
            "relations": relations_recorded,
            "contradictions": contradictions_created,
            "warnings": warnings,
        }

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
        return self._belief_repo.list_beliefs(
            status=status,
            document_id=document_id,
            chunk_id=chunk_id,
            epoch_min=epoch_min,
            epoch_max=epoch_max,
            min_confidence=min_confidence,
            limit=limit,
            offset=offset,
        )

    def get_belief(self, belief_id: str) -> Optional[dict[str, Any]]:
        return self._belief_repo.get_belief_detail(belief_id)

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
        return self._belief_repo.list_contradictions(
            status=status,
            min_score=min_score,
            epoch_min=epoch_min,
            epoch_max=epoch_max,
            limit=limit,
            offset=offset,
        )

    def resolve_contradiction(
        self,
        contradiction_id: str,
        *,
        resolution: str,
        note: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        resolved_at = datetime.now(timezone.utc).isoformat()
        return self._belief_repo.resolve_contradiction(
            contradiction_id=contradiction_id,
            resolution=resolution,
            note=note,
            resolved_at=resolved_at,
        )

    def dismiss_contradiction(
        self,
        contradiction_id: str,
        *,
        note: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        return self.resolve_contradiction(
            contradiction_id,
            resolution="false_positive",
            note=note,
        )
