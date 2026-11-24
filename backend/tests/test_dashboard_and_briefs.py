"""Integration tests for MindVault dashboard views, analytics, and Morning Briefs."""

from __future__ import annotations

import json
import math
from datetime import timedelta
from pathlib import Path
from typing import Iterator, Optional

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.models.embeddings import BaseEmbeddingAdapter
from app.models.llm import BaseLLMProvider, CancelCheck
from app.models.nli import BaseNLIAdapter, NLIScores
from app.scheduler.models import format_iso_datetime, parse_iso_datetime


class _TopicEmbedder(BaseEmbeddingAdapter):
    """Deterministic embedder separating storage/sync concepts from astronomy."""

    @property
    def model_name(self) -> str:
        return "insights-test-embedder"

    def is_available(self) -> tuple[bool, Optional[str]]:
        return True, None

    @staticmethod
    def _vec(text: str) -> list[float]:
        lower = text.lower()
        wal = 1.0 if ("wal" in lower or "sqlite" in lower or "concurrency" in lower) else 0.0
        vector = 1.0 if ("vector" in lower or "chromadb" in lower or "embedding" in lower) else 0.0
        neg = 1.1 if ("not" in lower or "never" in lower or "degrades" in lower) else 0.0
        raw = [wal + 0.2, vector + 0.1, neg, 0.25]
        norm = math.sqrt(sum(v * v for v in raw))
        return [v / norm for v in raw]

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class _DeterministicBeliefLLM(BaseLLMProvider):
    """Extracts crisp claims matching chunk text for offline deterministic tests."""

    @property
    def provider_name(self) -> str:
        return "deterministic-local"

    @property
    def model_name(self) -> str:
        return "deterministic-belief-llm"

    def is_available(self) -> tuple[bool, Optional[str]]:
        return True, None

    def stream_generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.2,
        cancel_check: Optional[CancelCheck] = None,
    ) -> Iterator[str]:
        lower = user_prompt.lower()
        if "improves read concurrency" in lower:
            payload = {
                "beliefs": [
                    {
                        "text": "SQLite WAL mode improves concurrent read throughput",
                        "confidence": 0.92,
                    }
                ]
            }
        elif "degrades write latency" in lower:
            payload = {
                "beliefs": [
                    {
                        "text": "SQLite WAL mode never improves concurrency cleanly",
                        "confidence": 0.88,
                    }
                ]
            }
        else:
            payload = {
                "beliefs": [
                    {
                        "text": "ChromaDB indexes vector embeddings keyed by chunk identifiers",
                        "confidence": 0.85,
                    }
                ]
            }
        yield json.dumps(payload)


class _DeterministicNLI(BaseNLIAdapter):
    """Flags conflicting SQLite WAL claims as high-score contradictions."""

    @property
    def model_name(self) -> str:
        return "deterministic-nli"

    def is_available(self) -> tuple[bool, Optional[str]]:
        return True, None

    @staticmethod
    def _score_one(premise: str, hypothesis: str) -> NLIScores:
        p = premise.lower()
        h = hypothesis.lower()
        if ("improves" in p and "never" in h) or ("never" in p and "improves" in h):
            return NLIScores(contradiction=0.91, entailment=0.04, neutral=0.05)
        return NLIScores(contradiction=0.05, entailment=0.20, neutral=0.75)

    def predict_pairs(self, pairs: list[tuple[str, str]]) -> list[NLIScores]:
        return [self._score_one(p, h) for p, h in pairs]


def _make_settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path,
        sqlite_path=tmp_path / "mindvault.db",
        chroma_dir=tmp_path / "chroma",
        models_dir=tmp_path / "models",
        logs_dir=tmp_path / "logs",
        vault_dir=tmp_path / "vault",
        belief_dedup_threshold=0.95,
        belief_candidate_min_similarity=0.20,
        contradiction_threshold=0.70,
        graph_similarity_threshold=0.50,
        retention_initial_stability_days=1.0,
        retention_threshold=0.75,
    )


def _build_client(tmp_path: Path) -> TestClient:
    app = create_app(
        _make_settings(tmp_path),
        embedding_adapter=_TopicEmbedder(),
        llm_provider=_DeterministicBeliefLLM(),
        nli_adapter=_DeterministicNLI(),
    )
    return TestClient(app)


def test_dashboard_empty_vault_states(tmp_path: Path) -> None:
    """Dashboard and brief endpoints should return structured empty payloads safely."""
    with _build_client(tmp_path) as client:
        overview = client.get("/api/v1/dashboard/overview")
        assert overview.status_code == 200
        body = overview.json()
        assert body["kpis"]["total_documents"] == 0
        assert body["kpis"]["total_chunks"] == 0
        assert body["belief_drift"]["epochs"] == []
        assert body["concept_velocity"]["epochs"] == []
        assert body["decay"]["cells"] == []
        assert body["health"]["status_label"] == "empty"
        assert body["latest_brief"] is None

        brief_resp = client.post("/api/v1/briefs/generate", json={"brief_date": "2026-11-24"})
        assert brief_resp.status_code == 201
        assert brief_resp.json()["insights"] == []
        assert "No vault insights available yet" in brief_resp.json()["content"]


def test_dashboard_views_and_epoch_filtering(tmp_path: Path) -> None:
    """Populate multi-epoch vault and verify all 5 dashboard data endpoints + overview."""
    with _build_client(tmp_path) as client:
        d1 = client.post(
            "/api/v1/ingest/file",
            files={
                "file": (
                    "epoch1_storage.md",
                    b"# Storage Engine\n\nSQLite WAL mode improves read concurrency reliably.\n",
                    "text/markdown",
                )
            },
            data={"epoch": "1"},
        ).json()["document_id"]
        d2 = client.post(
            "/api/v1/ingest/file",
            files={
                "file": (
                    "epoch2_vectors.md",
                    (
                        b"# Vector Sync\n\n"
                        b"ChromaDB indexes dense vector embeddings for hybrid retrieval.\n"
                    ),
                    "text/markdown",
                )
            },
            data={"epoch": "2"},
        ).json()["document_id"]
        d3 = client.post(
            "/api/v1/ingest/file",
            files={
                "file": (
                    "epoch3_wal_revisit.md",
                    (
                        b"# Latency Tradeoff\n\n"
                        b"SQLite WAL mode degrades write latency under checkpoint stalls.\n"
                    ),
                    "text/markdown",
                )
            },
            data={"epoch": "3"},
        ).json()["document_id"]

        c1 = client.get(f"/api/v1/documents/{d1}").json()["chunks"][0]
        c2 = client.get(f"/api/v1/documents/{d2}").json()["chunks"][0]
        c3 = client.get(f"/api/v1/documents/{d3}").json()["chunks"][0]

        assert client.post(f"/api/v1/beliefs/extract/{c1['id']}", json={}).status_code == 201
        assert client.post(f"/api/v1/beliefs/extract/{c2['id']}", json={}).status_code == 201
        assert client.post(f"/api/v1/beliefs/extract/{c3['id']}", json={}).status_code == 201
        assert client.post("/api/v1/graph/rebuild", json={}).status_code == 200

        eval_dt = parse_iso_datetime(c3["created_at"]) + timedelta(days=2)
        as_of_iso = format_iso_datetime(eval_dt)

        # 1. Belief Drift Timeline
        drift_resp = client.get("/api/v1/dashboard/belief-drift")
        assert drift_resp.status_code == 200
        drift = drift_resp.json()
        assert drift["summary"]["total_beliefs"] == 3
        assert drift["summary"]["open_contradictions"] == 1
        assert [ep["epoch"] for ep in drift["epochs"]] == [1, 2, 3]
        assert drift["epochs"][0]["open_contradiction_count"] == 1

        # Epoch filter on Belief Drift
        drift_ep2 = client.get(
            "/api/v1/dashboard/belief-drift",
            params={"epoch_min": 2, "epoch_max": 2},
        ).json()
        assert [ep["epoch"] for ep in drift_ep2["epochs"]] == [2]

        # 2. Concept Velocity
        vel_resp = client.get("/api/v1/dashboard/concept-velocity")
        assert vel_resp.status_code == 200
        vel = vel_resp.json()
        assert vel["summary"]["total_epochs"] == 3
        assert vel["summary"]["total_distinct_concepts"] >= 5
        assert len(vel["concepts"]) >= 3
        assert len(vel["concepts"][0]["sample_chunks"]) >= 1

        # 3. Knowledge Decay Heatmap
        decay_resp = client.get("/api/v1/dashboard/decay", params={"as_of": as_of_iso})
        assert decay_resp.status_code == 200
        decay = decay_resp.json()
        assert decay["summary"]["total_chunks"] == 3
        assert decay["summary"]["due_count"] == 3
        assert decay["summary"]["critical_count"] == 3
        assert len(decay["epoch_buckets"]) == 3
        assert len(decay["document_buckets"]) == 3

        # 4. Knowledge Health Radar
        health_resp = client.get("/api/v1/dashboard/health", params={"as_of": as_of_iso})
        assert health_resp.status_code == 200
        health = health_resp.json()
        assert health["total_chunks"] == 3
        assert len(health["axes"]) == 5
        for axis in health["axes"]:
            assert 0.0 <= axis["score"] <= 1.0
            assert 0.0 <= axis["value_pct"] <= 100.0

        # 5. Aggregated Dashboard Overview (keeps flashcard answers hidden!)
        ov_resp = client.get("/api/v1/dashboard/overview", params={"as_of": as_of_iso})
        assert ov_resp.status_code == 200
        ov = ov_resp.json()
        assert ov["kpis"]["total_documents"] == 3
        assert ov["kpis"]["due_flashcards"] == 3
        assert ov["kpis"]["open_contradictions"] == 1
        assert ov["contradictions"]["total"] == 1
        assert ov["review_queue"]["include_answer"] is False
        assert ov["review_queue"]["items"][0]["flashcard"]["answer"] is None


def test_morning_brief_synthesis_categories_citations_and_regeneration(tmp_path: Path) -> None:
    """Morning Brief enforces max 3 insights, prioritizes risk signals, cites, and persists."""
    with _build_client(tmp_path) as client:
        d1 = client.post(
            "/api/v1/ingest/file",
            files={
                "file": (
                    "ep1_notes.md",
                    b"# Architecture\n\nSQLite WAL mode improves read concurrency.\n",
                    "text/markdown",
                )
            },
            data={"epoch": "1"},
        ).json()["document_id"]
        d2 = client.post(
            "/api/v1/ingest/file",
            files={
                "file": (
                    "ep4_update.md",
                    b"# Regression\n\nSQLite WAL mode degrades write latency severely.\n",
                    "text/markdown",
                )
            },
            data={"epoch": "4"},
        ).json()["document_id"]

        c1 = client.get(f"/api/v1/documents/{d1}").json()["chunks"][0]
        c2 = client.get(f"/api/v1/documents/{d2}").json()["chunks"][0]
        client.post(f"/api/v1/beliefs/extract/{c1['id']}", json={})
        client.post(f"/api/v1/beliefs/extract/{c2['id']}", json={})
        client.post("/api/v1/graph/rebuild", json={})

        as_of_iso = format_iso_datetime(parse_iso_datetime(c2["created_at"]) + timedelta(days=3))

        # Default generation: limited to 3 major insights across 4 candidate categories.
        gen1 = client.post(
            "/api/v1/briefs/generate",
            json={"brief_date": "2026-11-24", "as_of": as_of_iso},
        )
        assert gen1.status_code == 201, gen1.text
        brief1 = gen1.json()
        assert brief1["brief_date"] == "2026-11-24"
        assert brief1["max_insights"] == 3
        assert len(brief1["insights"]) == 3
        assert len(brief1["citations"]) >= 1
        assert brief1["regenerated"] is False

        # Contradictions & decaying notes should rank at the top.
        categories_returned = {ins["category"] for ins in brief1["insights"]}
        assert "contradictions" in categories_returned
        assert "decaying_notes" in categories_returned
        for ins in brief1["insights"]:
            assert len(ins["citations"]) >= 1

        # Verify brief citations persisted to SQLite citations table.
        with client.app.state.db.connection() as conn:
            cnt = conn.execute(
                "SELECT COUNT(*) AS cnt FROM citations WHERE citation_type = 'brief'"
            ).fetchone()["cnt"]
            assert cnt >= 1

        # Without regenerate=True, same brief_date returns persisted row.
        gen_cached = client.post(
            "/api/v1/briefs/generate",
            json={
                "brief_date": "2026-11-24",
                "categories": ["major_clusters"],
                "regenerate": False,
            },
        )
        assert gen_cached.status_code == 201
        assert gen_cached.json()["id"] == brief1["id"]
        assert gen_cached.json()["regenerated"] is False

        # With regenerate=True and filtered categories, persisted brief is updated.
        gen_regen = client.post(
            "/api/v1/briefs/generate",
            json={
                "brief_date": "2026-11-24",
                "categories": ["major_clusters", "dormant_epochs"],
                "max_insights": 2,
                "regenerate": True,
                "as_of": as_of_iso,
            },
        )
        assert gen_regen.status_code == 201
        brief_regen = gen_regen.json()
        assert brief_regen["regenerated"] is True
        assert brief_regen["id"] != brief1["id"]
        assert brief_regen["categories"] == ["major_clusters", "dormant_epochs"]
        assert len(brief_regen["insights"]) == 2
        assert {ins["category"] for ins in brief_regen["insights"]}.issubset(
            {"major_clusters", "dormant_epochs"}
        )

        listed = client.get("/api/v1/briefs")
        assert listed.status_code == 200
        assert listed.json()["total"] == 1
        assert listed.json()["items"][0]["id"] == brief_regen["id"]

        # Validation error checks (422).
        assert (
            client.get(
                "/api/v1/dashboard/overview",
                params={"epoch_min": 5, "epoch_max": 2},
            ).status_code
            == 422
        )
        assert (
            client.post(
                "/api/v1/briefs/generate",
                json={"categories": ["invalid_cat"]},
            ).status_code
            == 422
        )
