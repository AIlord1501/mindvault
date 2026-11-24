"""Service layer for MindVault memory retention tracking, flashcards, and spaced repetition."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional, Union

from app.core.config import Settings
from app.db.database import DatabaseManager
from app.scheduler.models import (
    BaseRetentionModel,
    ExponentialDecayRetentionModel,
    build_flashcard_for_chunk,
    format_iso_datetime,
    parse_iso_datetime,
    resolve_outcome_and_rating,
    utc_now,
)
from app.scheduler.repository import RetentionRepository


class ChunkNotFoundForReviewError(LookupError):
    """Raised when attempting to review or query a missing chunk."""


class RetentionService:
    """Coordinates memory decay calculation, flashcard queue, review updates, and history."""

    def __init__(
        self,
        *,
        db: DatabaseManager,
        settings: Settings,
        repository: Optional[RetentionRepository] = None,
        model: Optional[BaseRetentionModel] = None,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self._db = db
        self._settings = settings
        self._repo = repository or RetentionRepository(db)
        self._model = model or ExponentialDecayRetentionModel.from_settings(settings)
        self._clock = clock or utc_now

    @property
    def model(self) -> BaseRetentionModel:
        """Expose the active retention model instance."""
        return self._model

    def _resolve_time(self, timestamp: Optional[Union[datetime, str]]) -> datetime:
        if timestamp is None:
            current = self._clock()
            if current.tzinfo is None:
                return current.replace(tzinfo=timezone.utc)
            return current.astimezone(timezone.utc)
        if isinstance(timestamp, str):
            return parse_iso_datetime(timestamp)
        if timestamp.tzinfo is None:
            return timestamp.replace(tzinfo=timezone.utc)
        return timestamp.astimezone(timezone.utc)

    def _sync_missing_rows(self) -> int:
        initial_stability = self._settings.retention_initial_stability_days
        interval_days = self._model.compute_next_interval_days(initial_stability)
        return self._repo.sync_missing_retention_rows(
            initial_stability=initial_stability,
            interval_days=interval_days,
        )

    def initialize_chunks(self, chunks: list[dict[str, Any]]) -> int:
        """Initialize retention_scores rows for newly indexed chunks."""
        initial_stability = self._settings.retention_initial_stability_days
        interval_days = self._model.compute_next_interval_days(initial_stability)
        return self._repo.ensure_chunk_retention_rows(
            chunks,
            initial_stability=initial_stability,
            interval_days=interval_days,
            default_retention=1.0,
        )

    def _evaluate_records(
        self,
        records: list[dict[str, Any]],
        eval_dt: datetime,
        *,
        persist: bool = True,
    ) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        """Compute live retention R = exp(-t / S) and review eligibility for chunk records."""
        updates: list[tuple[float, Optional[str], str]] = []
        evaluated: list[tuple[dict[str, Any], dict[str, Any]]] = []

        for rec in records:
            stability = (
                float(rec["stability"])
                if rec["stability"] is not None and rec["stability"] > 0.0
                else float(self._settings.retention_initial_stability_days)
            )
            reference_iso = rec["last_reviewed_at"] or rec["chunk_created_at"]
            elapsed_days = self._model.compute_elapsed_days(reference_iso, eval_dt)
            retention = self._model.compute_retention(elapsed_days, stability)

            next_review_at = rec["next_review_at"]
            if next_review_at is None:
                ref_dt = parse_iso_datetime(reference_iso)
                interval_days = self._model.compute_next_interval_days(stability)
                next_review_at = format_iso_datetime(ref_dt + timedelta(days=interval_days))

            eligible = self._model.is_eligible_for_review(retention, next_review_at, eval_dt)
            if rec.get("score_id"):
                updates.append((retention, next_review_at, str(rec["score_id"])))

            item = {
                "chunk_id": rec["chunk_id"],
                "document_id": rec["document_id"],
                "filename": rec["filename"],
                "file_type": rec["file_type"],
                "chunk_index": rec["chunk_index"],
                "epoch": rec["epoch"],
                "stability": stability,
                "retention": retention,
                "elapsed_days": elapsed_days,
                "last_reviewed_at": rec["last_reviewed_at"],
                "next_review_at": next_review_at,
                "review_count": rec["review_count"],
                "eligible_for_review": eligible,
            }
            evaluated.append((item, rec))

        if persist and updates:
            self._repo.batch_update_live_retention(updates)

        return evaluated

    def get_retention_list(
        self,
        *,
        document_id: Optional[str] = None,
        epoch_min: Optional[int] = None,
        epoch_max: Optional[int] = None,
        due_only: bool = False,
        as_of: Optional[Union[datetime, str]] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Recalculate and return current memory retention metrics across chunks."""
        eval_dt = self._resolve_time(as_of)
        self._sync_missing_rows()
        records = self._repo.list_chunk_retention_records(
            document_id=document_id,
            epoch_min=epoch_min,
            epoch_max=epoch_max,
        )
        pairs = self._evaluate_records(records, eval_dt, persist=True)
        all_items = [item for item, _ in pairs]

        total_chunks = len(all_items)
        due_count = sum(1 for item in all_items if item["eligible_for_review"])
        avg_retention = (
            sum(item["retention"] for item in all_items) / total_chunks if total_chunks else 0.0
        )
        avg_stability = (
            sum(item["stability"] for item in all_items) / total_chunks if total_chunks else 0.0
        )

        filtered = (
            [item for item in all_items if item["eligible_for_review"]]
            if due_only
            else list(all_items)
        )
        filtered.sort(
            key=lambda x: (
                x["retention"],
                x["epoch"],
                x["chunk_index"],
                x["chunk_id"],
            )
        )
        page = filtered[offset : offset + limit]

        return {
            "model_name": self._model.model_name,
            "retention_threshold": self._settings.retention_threshold,
            "evaluated_at": format_iso_datetime(eval_dt),
            "summary": {
                "total_chunks": total_chunks,
                "due_count": due_count,
                "average_retention": avg_retention,
                "average_stability": avg_stability,
            },
            "total": len(filtered),
            "limit": limit,
            "offset": offset,
            "items": page,
        }

    def get_review_queue(
        self,
        *,
        document_id: Optional[str] = None,
        epoch_min: Optional[int] = None,
        epoch_max: Optional[int] = None,
        include_answer: bool = False,
        include_upcoming: bool = False,
        as_of: Optional[Union[datetime, str]] = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Return prioritized spaced repetition flashcards without leaking answers by default."""
        eval_dt = self._resolve_time(as_of)
        self._sync_missing_rows()
        records = self._repo.list_chunk_retention_records(
            document_id=document_id,
            epoch_min=epoch_min,
            epoch_max=epoch_max,
        )
        pairs = self._evaluate_records(records, eval_dt, persist=True)
        due_total = sum(1 for item, _ in pairs if item["eligible_for_review"])

        selected_pairs = (
            list(pairs)
            if include_upcoming
            else [(item, rec) for item, rec in pairs if item["eligible_for_review"]]
        )
        selected_pairs.sort(
            key=lambda pair: (
                0 if pair[0]["eligible_for_review"] else 1,
                pair[0]["retention"],
                pair[0]["next_review_at"] or "",
                pair[0]["epoch"],
                pair[0]["chunk_index"],
                pair[0]["chunk_id"],
            )
        )

        queued_items: list[dict[str, Any]] = []
        for item, rec in selected_pairs[:limit]:
            flashcard = build_flashcard_for_chunk(
                rec,
                rec.get("beliefs", []),
                include_answer=include_answer,
            )
            queued_items.append(
                {
                    **item,
                    "flashcard": flashcard,
                }
            )

        return {
            "model_name": self._model.model_name,
            "retention_threshold": self._settings.retention_threshold,
            "evaluated_at": format_iso_datetime(eval_dt),
            "due_total": due_total,
            "returned_count": len(queued_items),
            "include_answer": include_answer,
            "include_upcoming": include_upcoming,
            "items": queued_items,
        }

    def record_chunk_review(
        self,
        chunk_id: str,
        *,
        outcome: Optional[str] = None,
        rating: Optional[int] = None,
        reveal_answer: bool = True,
        reviewed_at: Optional[Union[datetime, str]] = None,
        notes: Optional[str] = None,
    ) -> dict[str, Any]:
        """Record a self-recall review outcome, update stability S, and schedule next review."""
        canonical_outcome, canonical_rating = resolve_outcome_and_rating(
            outcome=outcome,
            rating=rating,
        )
        review_dt = self._resolve_time(reviewed_at)
        self._sync_missing_rows()

        records = self._repo.list_chunk_retention_records(chunk_id=chunk_id)
        if not records:
            raise ChunkNotFoundForReviewError(f"Indexed chunk '{chunk_id}' was not found.")
        rec = records[0]

        prev_stability = (
            float(rec["stability"])
            if rec["stability"] is not None and rec["stability"] > 0.0
            else float(self._settings.retention_initial_stability_days)
        )
        reference_iso = rec["last_reviewed_at"] or rec["chunk_created_at"]
        elapsed_days = self._model.compute_elapsed_days(reference_iso, review_dt)
        retention_before = self._model.compute_retention(elapsed_days, prev_stability)
        new_stability = self._model.update_stability(
            prev_stability,
            canonical_outcome,
            retention_before,
        )
        next_interval_days = self._model.compute_next_interval_days(new_stability)

        reviewed_at_iso = format_iso_datetime(review_dt)
        next_review_dt = review_dt + timedelta(days=next_interval_days)
        next_review_iso = format_iso_datetime(next_review_dt)
        new_review_count = int(rec["review_count"]) + 1

        history_entry = self._repo.record_review(
            chunk_id=chunk_id,
            outcome=canonical_outcome,
            rating=canonical_rating,
            previous_stability=prev_stability,
            new_stability=new_stability,
            retention_before=retention_before,
            elapsed_days=elapsed_days,
            reviewed_at=reviewed_at_iso,
            next_review_at=next_review_iso,
            review_count=new_review_count,
            notes=notes.strip() if isinstance(notes, str) and notes.strip() else None,
        )
        flashcard = build_flashcard_for_chunk(
            rec,
            rec.get("beliefs", []),
            include_answer=reveal_answer,
        )

        return {
            "review_id": history_entry["review_id"],
            "chunk_id": chunk_id,
            "document_id": rec["document_id"],
            "filename": rec["filename"],
            "chunk_index": rec["chunk_index"],
            "epoch": rec["epoch"],
            "outcome": canonical_outcome,
            "rating": canonical_rating,
            "previous_stability": prev_stability,
            "new_stability": new_stability,
            "retention_before": retention_before,
            "retention_after": 1.0,
            "elapsed_days": elapsed_days,
            "next_interval_days": next_interval_days,
            "review_count": new_review_count,
            "last_reviewed_at": reviewed_at_iso,
            "next_review_at": next_review_iso,
            "notes": history_entry["notes"],
            "flashcard": flashcard,
        }

    def get_review_history(
        self,
        *,
        chunk_id: Optional[str] = None,
        document_id: Optional[str] = None,
        outcome: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Return historical review log filtered optionally by chunk, document, or outcome."""
        normalized_outcome: Optional[str] = None
        if outcome is not None:
            normalized_outcome, _ = resolve_outcome_and_rating(outcome=outcome)

        items, total = self._repo.list_review_history(
            chunk_id=chunk_id,
            document_id=document_id,
            outcome=normalized_outcome,
            limit=limit,
            offset=offset,
        )
        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "items": items,
        }
