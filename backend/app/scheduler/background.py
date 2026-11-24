"""APScheduler-backed background job orchestration and status tracking for MindVault."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional

from app.core.config import Settings
from app.db.database import DatabaseManager
from app.scheduler.insights import VaultInsightsService
from app.scheduler.models import format_iso_datetime, utc_now
from app.scheduler.service import RetentionService

try:
    from apscheduler.schedulers.background import BackgroundScheduler as _APSBgScheduler
    from apscheduler.triggers.interval import IntervalTrigger as _APSIntervalTrigger

    _HAS_APSCHEDULER = True
except ImportError:
    _APSBgScheduler = None  # type: ignore[assignment]
    _APSIntervalTrigger = None  # type: ignore[assignment]
    _HAS_APSCHEDULER = False


class JobNotFoundError(KeyError):
    """Raised when querying or controlling an unknown background job ID."""


@dataclass
class _TrackedJob:
    job_id: str
    name: str
    description: str
    interval_seconds: int
    fn: Callable[[], dict[str, Any]]
    paused: bool = False
    state: str = "idle"
    run_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    skipped_count: int = 0
    last_run_at: Optional[str] = None
    last_duration_ms: Optional[float] = None
    next_run_at: Optional[str] = None
    last_result: dict[str, Any] = field(default_factory=dict)
    last_error: Optional[str] = None
    run_lock: threading.Lock = field(default_factory=threading.Lock)


class _EmbeddedCompatScheduler:
    """Daemon-timer fallback used when APScheduler is absent in offline test environments."""

    def __init__(self, jobs: dict[str, _TrackedJob], runner: Callable[[str], Any]) -> None:
        self._jobs = jobs
        self._runner = runner
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.running = False

    def start(self) -> None:
        if self.running:
            return
        self._stop_event.clear()
        now = utc_now()
        for job in self._jobs.values():
            if job.next_run_at is None:
                job.next_run_at = format_iso_datetime(
                    now + timedelta(seconds=job.interval_seconds)
                )
        self._thread = threading.Thread(
            target=self._loop,
            name="mindvault-scheduler-compat",
            daemon=True,
        )
        self.running = True
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop_event.wait(timeout=1.0):
            now = utc_now()
            for job_id, job in list(self._jobs.items()):
                if self._stop_event.is_set():
                    return
                if job.paused or job.next_run_at is None:
                    continue
                try:
                    due_dt = datetime.fromisoformat(job.next_run_at)
                except ValueError:
                    due_dt = now
                if now >= due_dt:
                    self._runner(job_id)

    def shutdown(self, wait: bool = True) -> None:
        self._stop_event.set()
        self.running = False
        if wait and self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)


class BackgroundJobScheduler:
    """Manages local scheduled jobs for belief extraction, memory decay, and morning briefs."""

    def __init__(
        self,
        *,
        db: DatabaseManager,
        settings: Settings,
        retention_service: RetentionService,
        insights_service: VaultInsightsService,
        belief_service: Optional[Any] = None,
        graph_service: Optional[Any] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._db = db
        self._settings = settings
        self._retention_service = retention_service
        self._insights_service = insights_service
        self._belief_service = belief_service
        self._graph_service = graph_service
        self._logger = logger or logging.getLogger("mindvault.scheduler")

        self._lock = threading.RLock()
        self._history: deque[dict[str, Any]] = deque(maxlen=50)
        self._scheduler: Any = None
        self._started = False
        self._shutdown_completed = False
        self._backend_name = "apscheduler" if _HAS_APSCHEDULER else "embedded_compat"

        self._jobs: dict[str, _TrackedJob] = {
            "belief_extraction": _TrackedJob(
                job_id="belief_extraction",
                name="Belief Extraction",
                description=(
                    "Extracts factual claims from newly indexed chunks via local LLM only."
                ),
                interval_seconds=settings.job_belief_interval_seconds,
                fn=self._job_extract_beliefs,
            ),
            "retention_recalculation": _TrackedJob(
                job_id="retention_recalculation",
                name="Retention Recalculation",
                description=(
                    "Re-evaluates exponential decay R = exp(-t / S) across indexed chunks."
                ),
                interval_seconds=settings.job_retention_interval_seconds,
                fn=self._job_recalculate_retention,
            ),
            "morning_brief_generation": _TrackedJob(
                job_id="morning_brief_generation",
                name="Morning Brief Synthesis",
                description="Generates and persists today's prioritized local Morning Brief.",
                interval_seconds=settings.job_brief_interval_seconds,
                fn=self._job_generate_morning_brief,
            ),
        }

    @property
    def backend_name(self) -> str:
        return self._backend_name

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._started and not self._shutdown_completed

    @property
    def shutdown_completed(self) -> bool:
        with self._lock:
            return self._shutdown_completed

    def start(self) -> None:
        """Start background scheduling if enabled by configuration."""
        with self._lock:
            if self._started or not self._settings.scheduler_enabled:
                return
            self._shutdown_completed = False
            now = utc_now()
            for job in self._jobs.values():
                job.next_run_at = format_iso_datetime(
                    now + timedelta(seconds=job.interval_seconds)
                )

            if _HAS_APSCHEDULER and _APSBgScheduler is not None and _APSIntervalTrigger is not None:
                scheduler = _APSBgScheduler(
                    timezone=timezone.utc,
                    job_defaults={
                        "coalesce": True,
                        "max_instances": 1,
                        "misfire_grace_time": 60,
                    },
                )
                for job in self._jobs.values():
                    scheduler.add_job(
                        func=self.run_job_now,
                        args=[job.job_id],
                        trigger=_APSIntervalTrigger(seconds=job.interval_seconds),
                        id=job.job_id,
                        name=job.name,
                        replace_existing=True,
                    )
                scheduler.start()
                self._scheduler = scheduler
            else:
                compat = _EmbeddedCompatScheduler(self._jobs, self.run_job_now)
                compat.start()
                self._scheduler = compat

            self._started = True
            self._logger.info(
                "Background job scheduler started",
                extra={
                    "event": "scheduler_start",
                    "backend": self._backend_name,
                    "job_count": len(self._jobs),
                },
            )

    def shutdown(self, wait: bool = True) -> None:
        """Gracefully stop APScheduler and prevent new jobs from starting."""
        with self._lock:
            if self._shutdown_completed:
                return
            scheduler = self._scheduler
            self._scheduler = None
            self._started = False
            self._shutdown_completed = True

        if scheduler is not None:
            try:
                scheduler.shutdown(wait=wait)
            except Exception as exc:
                self._logger.warning(
                    "Scheduler shutdown encountered non-fatal error",
                    extra={"event": "scheduler_shutdown_error", "error": str(exc)},
                )
        self._logger.info("Background job scheduler stopped", extra={"event": "scheduler_stop"})

    def _require_job(self, job_id: str) -> _TrackedJob:
        job = self._jobs.get(job_id)
        if job is None:
            raise JobNotFoundError(f"Unknown background job '{job_id}'.")
        return job

    def pause_job(self, job_id: str) -> dict[str, Any]:
        """Pause scheduled runs for the specified job ID."""
        with self._lock:
            job = self._require_job(job_id)
            job.paused = True
            job.state = "paused"
            if self._scheduler is not None and _HAS_APSCHEDULER and hasattr(
                self._scheduler, "pause_job"
            ):
                try:
                    self._scheduler.pause_job(job_id)
                except Exception:
                    pass
            return self._serialize_job(job)

    def resume_job(self, job_id: str) -> dict[str, Any]:
        """Resume scheduled runs for the specified job ID."""
        with self._lock:
            job = self._require_job(job_id)
            job.paused = False
            if job.state == "paused":
                job.state = "idle"
            if self._scheduler is not None and _HAS_APSCHEDULER and hasattr(
                self._scheduler, "resume_job"
            ):
                try:
                    self._scheduler.resume_job(job_id)
                except Exception:
                    pass
            return self._serialize_job(job)

    def run_job_now(self, job_id: str) -> dict[str, Any]:
        """Execute a registered job immediately and record telemetry."""
        job = self._require_job(job_id)
        if self._db.is_closed:
            return {
                "job_id": job_id,
                "status": "skipped",
                "reason": "database_closed",
                "ran_at": format_iso_datetime(utc_now()),
            }

        with job.run_lock:
            started_dt = utc_now()
            t0 = time.perf_counter()
            job.state = "running"
            outcome_status = "ok"
            result: dict[str, Any] = {}
            err_msg: Optional[str] = None

            try:
                result = job.fn()
                if result.get("skipped"):
                    outcome_status = "skipped"
                    job.skipped_count += 1
                    job.state = "paused" if job.paused else "skipped"
                else:
                    job.success_count += 1
                    job.state = "paused" if job.paused else "idle"
                job.last_error = None
            except Exception as exc:
                outcome_status = "error"
                err_msg = str(exc)
                job.failure_count += 1
                job.last_error = err_msg
                job.state = "error"
                self._logger.error(
                    "Scheduled job failed",
                    extra={"event": "job_error", "job_id": job_id, "error": err_msg},
                )

            duration_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            finished_iso = format_iso_datetime(utc_now())
            job.run_count += 1
            job.last_run_at = finished_iso
            job.last_duration_ms = duration_ms
            job.last_result = result
            if not job.paused:
                job.next_run_at = format_iso_datetime(
                    started_dt + timedelta(seconds=job.interval_seconds)
                )

            entry = {
                "job_id": job.job_id,
                "name": job.name,
                "status": outcome_status,
                "ran_at": finished_iso,
                "duration_ms": duration_ms,
                "result": result,
                "error": err_msg,
            }
            with self._lock:
                self._history.appendleft(entry)
            return entry

    def get_job(self, job_id: str) -> dict[str, Any]:
        """Return status and recent execution log for a single job."""
        job = self._require_job(job_id)
        with self._lock:
            history = [h for h in self._history if h["job_id"] == job_id][:15]
            return {
                **self._serialize_job(job),
                "history": history,
            }

    def get_status(self) -> dict[str, Any]:
        """Return overall scheduler health, job statuses, and recent executions."""
        with self._lock:
            return {
                "enabled": bool(self._settings.scheduler_enabled),
                "running": self.is_running,
                "shutdown_completed": self._shutdown_completed,
                "backend": self._backend_name,
                "jobs": [self._serialize_job(j) for j in self._jobs.values()],
                "recent_executions": list(self._history)[:20],
            }

    @staticmethod
    def _serialize_job(job: _TrackedJob) -> dict[str, Any]:
        return {
            "job_id": job.job_id,
            "name": job.name,
            "description": job.description,
            "interval_seconds": job.interval_seconds,
            "paused": job.paused,
            "state": job.state,
            "run_count": job.run_count,
            "success_count": job.success_count,
            "failure_count": job.failure_count,
            "skipped_count": job.skipped_count,
            "last_run_at": job.last_run_at,
            "last_duration_ms": job.last_duration_ms,
            "next_run_at": job.next_run_at,
            "last_result": job.last_result,
            "last_error": job.last_error,
        }

    def _job_extract_beliefs(self) -> dict[str, Any]:
        if self._belief_service is None:
            return {"skipped": True, "reason": "BeliefService unavailable."}

        llm_provider = getattr(self._belief_service._extractor, "_llm", None)
        if llm_provider is not None:
            avail, reason = llm_provider.is_available()
            if not avail:
                return {
                    "skipped": True,
                    "reason": reason or "Local LLM unavailable.",
                    "processed_chunks": 0,
                }

        batch_size = self._settings.job_belief_batch_size
        with self._db.connection() as conn:
            rows = conn.execute(
                """
                SELECT c.id
                FROM chunks c
                INNER JOIN documents d ON d.id = c.document_id
                LEFT JOIN beliefs b ON b.chunk_id = c.id
                WHERE d.status = 'indexed' AND b.id IS NULL
                ORDER BY c.created_at ASC, c.chunk_index ASC
                LIMIT ?
                """,
                (int(batch_size),),
            ).fetchall()

        if not rows:
            return {"skipped": False, "processed_chunks": 0, "beliefs_created": 0}

        processed = 0
        created = 0
        contradictions = 0
        for row in rows:
            summary = self._belief_service.extract_from_chunk(row["id"])
            processed += 1
            created += int(summary.get("extracted_count", 0))
            contradictions += len(summary.get("contradictions", []))

        return {
            "skipped": False,
            "processed_chunks": processed,
            "beliefs_created": created,
            "contradictions_detected": contradictions,
        }

    def _job_recalculate_retention(self) -> dict[str, Any]:
        payload = self._retention_service.get_retention_list(limit=500, offset=0)
        summary = payload["summary"]
        return {
            "skipped": False,
            "total_chunks": summary["total_chunks"],
            "due_count": summary["due_count"],
            "average_retention": round(float(summary["average_retention"]), 4),
        }

    def _job_generate_morning_brief(self) -> dict[str, Any]:
        with self._db.connection() as conn:
            cnt = int(
                conn.execute(
                    """
                    SELECT COUNT(*) AS cnt
                    FROM chunks c
                    INNER JOIN documents d ON d.id = c.document_id
                    WHERE d.status = 'indexed'
                    """
                ).fetchone()["cnt"]
            )
        if cnt == 0:
            return {"skipped": True, "reason": "No indexed chunks in vault yet."}

        brief = self._insights_service.generate_morning_brief(regenerate=True)
        return {
            "skipped": False,
            "brief_id": brief["id"],
            "brief_date": brief["brief_date"],
            "insight_count": len(brief["insights"]),
        }
