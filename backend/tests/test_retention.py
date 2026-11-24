"""Unit and integration tests for MindVault memory decay, flashcards, and spaced repetition."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.scheduler.models import (
    BaseRetentionModel,
    ExponentialDecayRetentionModel,
    format_iso_datetime,
    parse_iso_datetime,
    resolve_outcome_and_rating,
)


def _make_settings(
    tmp_path: Path,
    *,
    initial_stability: float = 2.0,
    min_stability: float = 0.25,
    retention_threshold: float = 0.75,
) -> Settings:
    return Settings(
        data_dir=tmp_path,
        sqlite_path=tmp_path / "mindvault.db",
        chroma_dir=tmp_path / "chroma",
        models_dir=tmp_path / "models",
        logs_dir=tmp_path / "logs",
        vault_dir=tmp_path / "vault",
        retention_initial_stability_days=initial_stability,
        retention_min_stability_days=min_stability,
        retention_threshold=retention_threshold,
    )


def _ingest_markdown(
    client: TestClient,
    *,
    filename: str,
    text: str,
    epoch: int = 100,
) -> tuple[str, list[dict]]:
    resp = client.post(
        "/api/v1/ingest/file",
        files={"file": (filename, text.encode("utf-8"), "text/markdown")},
        data={"epoch": str(epoch)},
    )
    assert resp.status_code == 201, resp.text
    doc_id = resp.json()["document_id"]
    doc_resp = client.get(f"/api/v1/documents/{doc_id}")
    assert doc_resp.status_code == 200
    return doc_id, doc_resp.json()["chunks"]


def test_exponential_decay_model_formula_and_parameters() -> None:
    """Verify exact R = exp(-t / S), interval solver, and outcome stability monotonicity."""
    model = ExponentialDecayRetentionModel(
        initial_stability_days=2.0,
        min_stability_days=0.5,
        max_stability_days=100.0,
        retention_threshold=0.75,
        factor_forgot=0.5,
        factor_hard=1.2,
        factor_good=2.2,
        factor_easy=3.2,
    )

    for elapsed in (0.0, 0.5, 1.0, 2.0, 5.0):
        expected = math.exp(-elapsed / 2.0)
        assert model.compute_retention(elapsed, 2.0) == pytest.approx(expected, rel=1e-12)

    assert model.compute_retention(-3.0, 2.0) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="stability_days must be positive"):
        model.compute_retention(1.0, 0.0)

    interval = model.compute_next_interval_days(2.0)
    assert interval == pytest.approx(-2.0 * math.log(0.75), rel=1e-12)
    assert model.compute_retention(interval, 2.0) == pytest.approx(0.75, rel=1e-12)

    s_forgot = model.update_stability(2.0, "forgot", 0.50)
    s_hard = model.update_stability(2.0, "hard", 0.50)
    s_good = model.update_stability(2.0, "good", 0.50)
    s_easy = model.update_stability(2.0, "easy", 0.50)
    assert s_forgot == pytest.approx(1.0)
    assert s_forgot < 2.0
    assert s_forgot < s_hard < s_good < s_easy

    # Minimum stability floor enforced on repeated lapses.
    assert model.update_stability(0.5, "forgot", 0.2) == pytest.approx(0.5)

    # Outcome / rating resolver validation.
    assert resolve_outcome_and_rating("good", None) == ("good", 3)
    assert resolve_outcome_and_rating(None, 4) == ("easy", 4)
    with pytest.raises(ValueError, match="Conflicting outcome"):
        resolve_outcome_and_rating("easy", 1)
    with pytest.raises(ValueError, match="required"):
        resolve_outcome_and_rating(None, None)


def test_retention_decay_over_elapsed_time_and_eligibility(tmp_path: Path) -> None:
    """GET /api/v1/retention computes live R = exp(-t / S) and marks due items correctly."""
    app = create_app(_make_settings(tmp_path, initial_stability=2.0, retention_threshold=0.75))
    with TestClient(app) as client:
        doc_id, chunks = _ingest_markdown(
            client,
            filename="memory_architecture.md",
            text=(
                "# Storage Notes\n\n"
                "SQLite WAL mode improves concurrent readers while keeping writes durable.\n"
            ),
            epoch=10,
        )
        assert len(chunks) == 1
        created_dt = parse_iso_datetime(chunks[0]["created_at"])

        # At t = 0 days: R = exp(0) = 1.0 (not due).
        at_t0 = format_iso_datetime(created_dt)
        resp_t0 = client.get("/api/v1/retention", params={"as_of": at_t0})
        assert resp_t0.status_code == 200
        body_t0 = resp_t0.json()
        assert body_t0["model_name"] == "exponential_decay"
        assert body_t0["summary"]["total_chunks"] == 1
        assert body_t0["summary"]["due_count"] == 0
        assert body_t0["items"][0]["retention"] == pytest.approx(1.0, abs=1e-9)
        assert body_t0["items"][0]["eligible_for_review"] is False

        # At t = 0.5 days: R = exp(-0.5 / 2.0) = exp(-0.25) ~= 0.7788 (still > 0.75).
        at_half_day = format_iso_datetime(created_dt + timedelta(hours=12))
        resp_half = client.get("/api/v1/retention", params={"as_of": at_half_day})
        assert resp_half.status_code == 200
        item_half = resp_half.json()["items"][0]
        assert item_half["elapsed_days"] == pytest.approx(0.5, abs=1e-9)
        assert item_half["retention"] == pytest.approx(math.exp(-0.25), rel=1e-9)
        assert item_half["eligible_for_review"] is False

        # At t = 2.0 days: R = exp(-2.0 / 2.0) = exp(-1) ~= 0.3679 (<= 0.75 -> due!).
        at_two_days = format_iso_datetime(created_dt + timedelta(days=2))
        resp_due = client.get(
            "/api/v1/retention",
            params={"as_of": at_two_days, "due_only": True, "document_id": doc_id},
        )
        assert resp_due.status_code == 200
        body_due = resp_due.json()
        assert body_due["summary"]["due_count"] == 1
        assert len(body_due["items"]) == 1
        item_due = body_due["items"][0]
        assert item_due["retention"] == pytest.approx(math.exp(-1.0), rel=1e-9)
        assert item_due["eligible_for_review"] is True


def test_flashcard_self_recall_workflow_and_hidden_answer(tmp_path: Path) -> None:
    """Queue presents recall prompt without answer, supports reveal, and updates schedule."""
    app = create_app(_make_settings(tmp_path, initial_stability=1.0, retention_threshold=0.75))
    secret_fact = "ChromaDB stores vector embeddings keyed identically to SQLite chunk IDs."
    with TestClient(app) as client:
        _, chunks = _ingest_markdown(
            client,
            filename="vector_sync.md",
            text=f"# Vector Synchronization\n\n{secret_fact}\n",
            epoch=20,
        )
        chunk_id = chunks[0]["id"]
        created_dt = parse_iso_datetime(chunks[0]["created_at"])
        due_time = format_iso_datetime(created_dt + timedelta(days=1))

        # 1. Present prompt without answer (include_answer defaults to False).
        q_hidden = client.get("/api/v1/review/queue", params={"as_of": due_time})
        assert q_hidden.status_code == 200
        q_body = q_hidden.json()
        assert q_body["due_total"] == 1
        assert q_body["include_answer"] is False
        assert len(q_body["items"]) == 1

        card_hidden = q_body["items"][0]["flashcard"]
        assert card_hidden["answer_revealed"] is False
        assert card_hidden["answer"] is None
        assert card_hidden["key_beliefs"] is None
        assert "Vector Synchronization" in card_hidden["prompt"]
        assert secret_fact not in card_hidden["prompt"]

        # 2. Reveal answer explicitly on demand.
        q_revealed = client.get(
            "/api/v1/review/queue",
            params={"as_of": due_time, "include_answer": True},
        )
        assert q_revealed.status_code == 200
        card_revealed = q_revealed.json()["items"][0]["flashcard"]
        assert card_revealed["answer_revealed"] is True
        assert secret_fact in (card_revealed["answer"] or "")

        # 3. Record recall outcome -> updates stability, review_count, next_review_at.
        review_resp = client.post(
            f"/api/v1/review/{chunk_id}",
            json={
                "outcome": "good",
                "reviewed_at": due_time,
                "notes": "Recalled sync invariant cleanly.",
            },
        )
        assert review_resp.status_code == 200, review_resp.text
        rev = review_resp.json()
        assert rev["outcome"] == "good"
        assert rev["rating"] == 3
        assert rev["previous_stability"] == pytest.approx(1.0)
        assert rev["new_stability"] > 2.2
        assert rev["retention_before"] == pytest.approx(math.exp(-1.0), rel=1e-9)
        assert rev["retention_after"] == pytest.approx(1.0)
        assert rev["review_count"] == 1
        assert rev["last_reviewed_at"] == due_time
        assert parse_iso_datetime(rev["next_review_at"]) > parse_iso_datetime(due_time)
        assert secret_fact in (rev["flashcard"]["answer"] or "")

        # Queue is now empty at due_time because chunk was just reviewed.
        q_after = client.get("/api/v1/review/queue", params={"as_of": due_time})
        assert q_after.status_code == 200
        assert q_after.json()["due_total"] == 0
        assert q_after.json()["items"] == []

        # Upcoming queue can still preview scheduled items.
        q_upcoming = client.get(
            "/api/v1/review/queue",
            params={"as_of": due_time, "include_upcoming": True},
        )
        assert len(q_upcoming.json()["items"]) == 1


def test_review_outcomes_progression_and_history_logging(tmp_path: Path) -> None:
    """Test multi-review spacing progression across forgot, hard, good, easy outcomes."""
    app = create_app(_make_settings(tmp_path, initial_stability=2.0, retention_threshold=0.75))
    with TestClient(app) as client:
        doc_a, chunks_a = _ingest_markdown(
            client,
            filename="outcomes_a.md",
            text="Local LLMs execute GGUF weights strictly on loopback.",
            epoch=1,
        )
        doc_b, chunks_b = _ingest_markdown(
            client,
            filename="outcomes_b.md",
            text="Contradiction detection filters candidates with cosine similarity before NLI.",
            epoch=2,
        )
        doc_c, chunks_c = _ingest_markdown(
            client,
            filename="outcomes_c.md",
            text="Orphan nodes remain visible in D3 knowledge graph responses.",
            epoch=3,
        )
        doc_d, chunks_d = _ingest_markdown(
            client,
            filename="outcomes_d.md",
            text="Migration scripts validate SHA-256 checksums before starting FastAPI.",
            epoch=4,
        )

        # Evaluate all 4 chunks after 1 day with identical initial stability S=2.0.
        base_dt = parse_iso_datetime(chunks_a[0]["created_at"])
        day1_iso = format_iso_datetime(base_dt + timedelta(days=1))

        res_forgot = client.post(
            f"/api/v1/review/{chunks_a[0]['id']}",
            json={"rating": 1, "reviewed_at": day1_iso},
        ).json()
        res_hard = client.post(
            f"/api/v1/review/{chunks_b[0]['id']}",
            json={"rating": 2, "reviewed_at": day1_iso},
        ).json()
        res_good = client.post(
            f"/api/v1/review/{chunks_c[0]['id']}",
            json={"outcome": "good", "reviewed_at": day1_iso},
        ).json()
        res_easy = client.post(
            f"/api/v1/review/{chunks_d[0]['id']}",
            json={"outcome": "easy", "reviewed_at": day1_iso},
        ).json()

        assert res_forgot["outcome"] == "forgot"
        assert res_hard["outcome"] == "hard"
        assert (
            res_forgot["new_stability"]
            < 2.0
            < res_hard["new_stability"]
            < res_good["new_stability"]
            < res_easy["new_stability"]
        )
        assert (
            res_forgot["next_interval_days"]
            < res_hard["next_interval_days"]
            < res_good["next_interval_days"]
            < res_easy["next_interval_days"]
        )

        # Multi-step review progression on chunk_c across days 1 -> 3 -> 9.
        day3_iso = format_iso_datetime(base_dt + timedelta(days=3))
        day9_iso = format_iso_datetime(base_dt + timedelta(days=9))
        step2 = client.post(
            f"/api/v1/review/{chunks_c[0]['id']}",
            json={"outcome": "good", "reviewed_at": day3_iso},
        ).json()
        step3 = client.post(
            f"/api/v1/review/{chunks_c[0]['id']}",
            json={"outcome": "easy", "reviewed_at": day9_iso},
        ).json()
        assert step2["review_count"] == 2
        assert step3["review_count"] == 3
        assert step3["new_stability"] > step2["new_stability"] > res_good["new_stability"]

        # Review history endpoint filters cleanly.
        hist_all = client.get("/api/v1/review/history").json()
        assert hist_all["total"] == 6

        hist_c = client.get(
            "/api/v1/review/history",
            params={"chunk_id": chunks_c[0]["id"]},
        ).json()
        assert hist_c["total"] == 3
        assert [item["rating"] for item in hist_c["items"]] == [4, 3, 3]
        assert hist_c["items"][0]["reviewed_at"] == day9_iso
        assert hist_c["items"][0]["outcome"] == "easy"

        hist_doc_a = client.get(
            "/api/v1/review/history",
            params={"document_id": doc_a, "outcome": "forgot"},
        ).json()
        assert hist_doc_a["total"] == 1
        assert hist_doc_a["items"][0]["filename"] == "outcomes_a.md"

        # Cascade deletion removes document retention_scores & review_history rows.
        del_resp = client.delete(f"/api/v1/documents/{doc_b}")
        assert del_resp.status_code == 200
        hist_after_del = client.get("/api/v1/review/history").json()
        assert hist_after_del["total"] == 5

        # Epoch filter works on retention overview.
        epoch_filtered = client.get(
            "/api/v1/retention",
            params={"epoch_min": 3, "epoch_max": 3, "as_of": day9_iso},
        ).json()
        assert epoch_filtered["summary"]["total_chunks"] == 1
        assert epoch_filtered["items"][0]["document_id"] == doc_c


def test_retention_validation_errors_and_extensible_model(tmp_path: Path) -> None:
    """Verify 404/422 validation errors and custom BaseRetentionModel pluggability."""

    class _ConstantRetentionModel(BaseRetentionModel):
        model_name = "custom_linear_test"

        def compute_elapsed_days(self, reference_iso: str, now: datetime) -> float:
            ref = parse_iso_datetime(reference_iso)
            return max(0.0, (now.astimezone(timezone.utc) - ref).total_seconds() / 86400.0)

        def compute_retention(self, elapsed_days: float, stability_days: float) -> float:
            return max(0.0, min(1.0, 1.0 - (elapsed_days / max(1.0, stability_days))))

        def compute_next_interval_days(self, stability_days: float) -> float:
            return 0.5 * stability_days

        def update_stability(
            self,
            current_stability: float,
            outcome: str,
            retention_now: float,
        ) -> float:
            return current_stability + 1.0

        def is_eligible_for_review(
            self,
            retention: float,
            next_review_at: Optional[str],
            now: datetime,
        ) -> bool:
            return retention <= 0.6

    app = create_app(
        _make_settings(tmp_path),
        retention_model=_ConstantRetentionModel(),
    )
    with TestClient(app) as client:
        _, chunks = _ingest_markdown(
            client,
            filename="custom_model.md",
            text="Custom retention models plug cleanly into RetentionService.",
            epoch=5,
        )
        created_dt = parse_iso_datetime(chunks[0]["created_at"])
        after_one_day = format_iso_datetime(created_dt + timedelta(days=1))

        ret = client.get("/api/v1/retention", params={"as_of": after_one_day})
        assert ret.status_code == 200
        assert ret.json()["model_name"] == "custom_linear_test"
        assert ret.json()["items"][0]["retention"] == pytest.approx(0.5, rel=1e-9)

        # Missing chunk -> 404
        missing_resp = client.post("/api/v1/review/nonexistent", json={"outcome": "good"})
        assert missing_resp.status_code == 404

        # Missing outcome & rating -> 422
        empty_body = client.post(f"/api/v1/review/{chunks[0]['id']}", json={})
        assert empty_body.status_code == 422

        # Conflicting outcome & rating -> 422
        conflict = client.post(
            f"/api/v1/review/{chunks[0]['id']}",
            json={"outcome": "forgot", "rating": 4},
        )
        assert conflict.status_code == 422

        # Invalid as_of / epoch_min > epoch_max -> 422
        assert (
            client.get("/api/v1/retention", params={"as_of": "not-a-timestamp"}).status_code == 422
        )
        assert (
            client.get(
                "/api/v1/review/queue",
                params={"epoch_min": 10, "epoch_max": 2},
            ).status_code
            == 422
        )
