"""End-to-end workflow, APScheduler jobs, corruption recovery, and security hardening tests."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterator, Optional

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.models.embeddings import (
    BaseEmbeddingAdapter,
    SentenceTransformerEmbeddingAdapter,
)
from app.models.llm import BaseLLMProvider, CancelCheck
from app.models.nli import (
    BaseNLIAdapter,
    HeuristicLexicalNLIAdapter,
    LocalSequenceClassificationNLIAdapter,
    NLIScores,
)


class _HardeningEmbedder(BaseEmbeddingAdapter):
    @property
    def model_name(self) -> str:
        return "hardening-local-embedder"

    def is_available(self) -> tuple[bool, Optional[str]]:
        return True, None

    @staticmethod
    def _vec(text: str) -> list[float]:
        lower = text.lower()
        wal = 1.0 if ("wal" in lower or "sqlite" in lower or "durability" in lower) else 0.0
        graph = 1.0 if ("graph" in lower or "edge" in lower or "citation" in lower) else 0.0
        decay = 1.0 if ("retention" in lower or "decay" in lower or "spaced" in lower) else 0.0
        raw = [wal + 0.2, graph + 0.1, decay + 0.1, 0.25]
        norm = math.sqrt(sum(v * v for v in raw))
        return [v / norm for v in raw]

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class _HardeningLLM(BaseLLMProvider):
    @property
    def provider_name(self) -> str:
        return "hardening-local-llm"

    @property
    def model_name(self) -> str:
        return "local-gguf-mock"

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
        if "extract" in system_prompt.lower() or "json" in system_prompt.lower():
            yield json.dumps(
                {
                    "beliefs": [
                        {
                            "text": "SQLite WAL guarantees local transactional durability",
                            "confidence": 0.93,
                        }
                    ]
                }
            )
            return
        if "socratic" in system_prompt.lower():
            yield "What tradeoffs arise between WAL durability and write amplification [1]?"
            return
        yield "MindVault stores local chunks in SQLite WAL [1] for offline durability."


class _HardeningNLI(BaseNLIAdapter):
    @property
    def model_name(self) -> str:
        return "hardening-local-nli"

    def is_available(self) -> tuple[bool, Optional[str]]:
        return True, None

    def predict_pairs(self, pairs: list[tuple[str, str]]) -> list[NLIScores]:
        return [NLIScores(contradiction=0.05, entailment=0.25, neutral=0.70) for _ in pairs]


def _make_settings(tmp_path: Path, **overrides: object) -> Settings:
    kwargs: dict[str, object] = {
        "data_dir": tmp_path,
        "sqlite_path": tmp_path / "mindvault.db",
        "chroma_dir": tmp_path / "chroma",
        "models_dir": tmp_path / "models",
        "logs_dir": tmp_path / "logs",
        "vault_dir": tmp_path / "vault",
        "scheduler_enabled": True,
        "job_belief_interval_seconds": 60,
        "job_retention_interval_seconds": 60,
        "job_brief_interval_seconds": 60,
    }
    kwargs.update(overrides)
    return Settings(**kwargs)  # type: ignore[arg-type]


def test_end_to_end_ingestion_to_chat_and_scheduler_lifecycle(tmp_path: Path) -> None:
    """Exercise full ingestion -> jobs -> search -> graph -> review -> brief -> chat pipeline."""
    app = create_app(
        _make_settings(tmp_path),
        embedding_adapter=_HardeningEmbedder(),
        llm_provider=_HardeningLLM(),
        nli_adapter=_HardeningNLI(),
    )
    with TestClient(app) as client:
        # 1. Ingest Markdown note locally
        ingest = client.post(
            "/api/v1/ingest/file",
            files={
                "file": (
                    "durability_notes.md",
                    (
                        b"# Storage Engine\n\n"
                        b"SQLite WAL ensures local transactional durability and concurrency.\n"
                    ),
                    "text/markdown",
                )
            },
            data={"epoch": "5"},
        )
        assert ingest.status_code == 201, ingest.text
        doc_id = ingest.json()["document_id"]
        chunks = client.get(f"/api/v1/documents/{doc_id}").json()["chunks"]
        chunk_id = chunks[0]["id"]

        # 2. Hybrid search returns citation hit
        search = client.post("/api/v1/search", json={"query": "SQLite WAL durability"})
        assert search.status_code == 200
        assert search.json()["total"] == 1
        assert search.json()["results"][0]["chunk_id"] == chunk_id

        # 3. Background jobs run cleanly and track execution telemetry
        jobs = client.get("/api/v1/jobs")
        assert jobs.status_code == 200
        assert jobs.json()["running"] is True
        assert len(jobs.json()["jobs"]) == 3

        belief_job = client.post("/api/v1/jobs/belief_extraction/run")
        assert belief_job.status_code == 200
        assert belief_job.json()["status"] == "ok"
        assert belief_job.json()["result"]["beliefs_created"] >= 1

        ret_job = client.post("/api/v1/jobs/retention_recalculation/run")
        assert ret_job.status_code == 200
        assert ret_job.json()["result"]["total_chunks"] == 1

        brief_job = client.post("/api/v1/jobs/morning_brief_generation/run")
        assert brief_job.status_code == 200
        assert brief_job.json()["result"]["insight_count"] >= 1

        # Pause & resume controls
        assert client.post("/api/v1/jobs/belief_extraction/pause").json()["paused"] is True
        assert client.post("/api/v1/jobs/belief_extraction/resume").json()["paused"] is False
        assert client.get("/api/v1/jobs/unknown").status_code == 404

        # 4. Graph rebuild & query
        assert client.post("/api/v1/graph/rebuild", json={}).status_code == 200
        graph = client.get("/api/v1/graph/concepts").json()
        assert graph["stats"]["node_count"] >= 2

        # 5. Review submission updates stability
        rev = client.post(f"/api/v1/review/{chunk_id}", json={"outcome": "good"})
        assert rev.status_code == 200
        assert rev.json()["review_count"] == 1

        # 6. Chat streaming SSE & Socratic mode
        stream_resp = client.post(
            "/api/v1/chat/stream",
            json={
                "query": "How does SQLite WAL work?",
                "answer_mode": "summary",
                "socratic": True,
                "epoch_min": 5,
            },
        )
        assert stream_resp.status_code == 200
        body_text = stream_resp.text
        assert "event: status" in body_text
        assert "event: citation" in body_text
        assert "event: token" in body_text
        assert "event: done" in body_text
        assert "tradeoffs" in body_text

        # 7. Enriched system status diagnostics
        sys_status = client.get("/api/v1/system/status?include_paths=true").json()
        assert sys_status["offline_only"] is True
        assert sys_status["models"]["llm"]["available"] is True
        assert sys_status["models"]["embedding"]["available"] is True
        assert sys_status["index"]["synchronized"] is True
        assert sys_status["scheduler"]["running"] is True
        assert sys_status["recovery"]["database"]["recovered"] is False

    # Graceful shutdown stops scheduler and closes SQLite cleanly.
    assert app.state.job_scheduler.shutdown_completed is True
    assert app.state.db.is_closed is True


def test_corrupted_sqlite_and_vector_index_recovery(tmp_path: Path) -> None:
    """Startup must quarantine corrupted SQLite and Chroma stores and recover cleanly."""
    db_path = tmp_path / "mindvault.db"
    chroma_dir = tmp_path / "chroma"
    chroma_dir.mkdir(parents=True, exist_ok=True)
    db_path.write_bytes(b"THIS IS NOT A VALID SQLITE FILE HEADER" * 16)
    (chroma_dir / "chroma.sqlite3").write_bytes(b"CORRUPTED CHROMA METADATA" * 16)

    app = create_app(_make_settings(tmp_path))
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        status_payload = client.get("/api/v1/system/status").json()
        assert status_payload["database_ok"] is True
        assert status_payload["applied_migrations"] == [1, 2, 3]
        assert status_payload["recovery"]["database"]["recovered"] is True
        assert status_payload["recovery"]["vector_store"]["recovered"] is True

        quarantined_db = Path(status_payload["recovery"]["database"]["quarantined_path"])
        quarantined_vec = Path(status_payload["recovery"]["vector_store"]["quarantined_dir"])
        assert quarantined_db.exists()
        assert quarantined_vec.exists()

        # Vault works normally after recovery.
        resp = client.post(
            "/api/v1/ingest/file",
            files={"file": ("recovered.md", b"Clean note after recovery.", "text/markdown")},
        )
        assert resp.status_code == 201


def test_hardening_offline_guards_large_files_and_missing_models(tmp_path: Path) -> None:
    """Enforce loopback guards, upload size caps, and clean missing-model handling."""
    with pytest.raises(ValueError, match="127.0.0.1"):
        Settings(host="0.0.0.0")
    with pytest.raises(ValueError, match="loopback"):
        Settings(ollama_base_url="https://api.remote.example.com")

    app = create_app(_make_settings(tmp_path, max_upload_size_bytes=64))
    with TestClient(app) as client:
        # Large upload rejected cleanly without crashing.
        big_resp = client.post(
            "/api/v1/ingest/file",
            files={"file": ("huge.txt", b"X" * 200, "text/plain")},
        )
        assert big_resp.status_code in (200, 400, 413)
        if big_resp.status_code == 200:
            assert big_resp.json()["status"] == "failed"

        # Missing local LLM returns 503 with zero remote fallback.
        chat_resp = client.post("/api/v1/chat", json={"query": "Test missing LLM"})
        assert chat_resp.status_code == 503

        # Scheduled belief extraction skips cleanly when local LLM is absent.
        run_belief = client.post("/api/v1/jobs/belief_extraction/run")
        assert run_belief.status_code == 200
        assert run_belief.json()["status"] == "skipped"

        # Whitespace-only note must fail with 422 empty_extracted_text instead of 0-chunk index.
        blank_resp = client.post(
            "/api/v1/ingest/file",
            files={"file": ("blank.txt", b"   \n\t\r\n   ", "text/plain")},
        )
        assert blank_resp.status_code == 422
        assert blank_resp.json()["status"] == "failed"
        assert blank_resp.json()["error_code"] == "empty_extracted_text"
        docs_after_blank = client.get("/api/v1/documents?status=indexed").json()
        assert docs_after_blank["total"] == 0


def test_audit_regressions_nli_offline_models_and_graph_prune(tmp_path: Path) -> None:
    """Verify heuristic NLI precision, strict offline model paths, and dangling edge cleanup."""
    nli = HeuristicLexicalNLIAdapter()
    # Unrelated sentences sharing only one noun + negation must NOT trigger false contradiction.
    unrelated = nli.predict_pairs(
        [
            ("SQLite supports concurrent readers.", "SQLite does not require external servers."),
        ]
    )[0]
    assert unrelated.contradiction < 0.70

    # True multi-token contradiction on the same claim triggers contradiction.
    conflict = nli.predict_pairs(
        [
            (
                "SQLite WAL improves read concurrency.",
                "SQLite WAL never improves read concurrency.",
            ),
        ]
    )[0]
    assert conflict.contradiction >= 0.70

    # Offline adapters must refuse uncached remote HuggingFace identifiers.
    emb = SentenceTransformerEmbeddingAdapter(
        model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
        models_dir=tmp_path / "models",
        offline_only=True,
    )
    ok_emb, msg_emb = emb.is_available()
    assert ok_emb is False
    assert "does not exist offline" in (msg_emb or "")

    nli_cls = LocalSequenceClassificationNLIAdapter(
        model_path="facebook/bart-large-mnli",
        models_dir=tmp_path / "models",
        offline_only=True,
    )
    ok_nli, msg_nli = nli_cls.is_available()
    assert ok_nli is False
    assert "does not exist offline" in (msg_nli or "")

    # Deleting a document cited by a chat query must prune dangling query:* graph edges.
    app = create_app(
        _make_settings(tmp_path),
        embedding_adapter=_HardeningEmbedder(),
        llm_provider=_HardeningLLM(),
        nli_adapter=_HardeningNLI(),
    )
    with TestClient(app) as client:
        ingest = client.post(
            "/api/v1/ingest/file",
            files={
                "file": (
                    "prune.md",
                    b"SQLite WAL ensures durable local storage.",
                    "text/markdown",
                )
            },
        )
        doc_id = ingest.json()["document_id"]
        assert client.post("/api/v1/chat", json={"query": "SQLite WAL durable"}).status_code == 200
        assert client.post("/api/v1/graph/rebuild", json={}).status_code == 200
        before_graph = client.get("/api/v1/graph/citations").json()
        assert before_graph["stats"]["edge_count"] >= 1

        assert client.delete(f"/api/v1/documents/{doc_id}").status_code == 200
        with client.app.state.db.connection() as conn:
            edge_cnt = conn.execute("SELECT COUNT(*) AS cnt FROM graph_edges").fetchone()["cnt"]
            assert edge_cnt == 0
