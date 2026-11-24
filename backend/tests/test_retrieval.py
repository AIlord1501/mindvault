"""Unit and integration tests for offline ChromaDB + BM25 hybrid retrieval."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Optional

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.models.embeddings import (
    BaseEmbeddingAdapter,
    EmbeddingUnavailableError,
    SentenceTransformerEmbeddingAdapter,
)
from app.retrieval.bm25_index import BM25Index


class ConceptEmbeddingAdapter(BaseEmbeddingAdapter):
    """Deterministic offline concept-axis embedder for testing semantic & hybrid retrieval."""

    def __init__(self, model_name: str = "concept-test-v1") -> None:
        self._model_name = model_name
        self.recorded_batches: list[list[str]] = []

    @property
    def model_name(self) -> str:
        return self._model_name

    def is_available(self) -> tuple[bool, Optional[str]]:
        return True, None

    @staticmethod
    def _vectorize(text: str) -> list[float]:
        lower = text.lower()
        bio = sum(
            1.0
            for kw in ("mitochondria", "atp", "cellular", "respiration", "biology", "enzyme")
            if kw in lower
        )
        crypto = sum(
            1.0
            for kw in ("cipher", "cryptography", "encryption", "aes", "key", "zero-knowledge")
            if kw in lower
        )
        cosmos = sum(
            1.0
            for kw in ("nebula", "astronomy", "cosmos", "galaxy", "telescope", "orbit")
            if kw in lower
        )
        fallback = 0.25 if (bio == 0.0 and crypto == 0.0 and cosmos == 0.0) else 0.0
        vec = [bio, crypto, cosmos, fallback]
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [round(v / norm, 6) for v in vec]

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        self.recorded_batches.append(list(texts))
        return [self._vectorize(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vectorize(text)


def _make_settings(tmp_path: Path, **overrides: object) -> Settings:
    kwargs: dict[str, object] = {
        "data_dir": tmp_path,
        "sqlite_path": tmp_path / "mindvault.db",
        "chroma_dir": tmp_path / "chroma",
        "models_dir": tmp_path / "models",
        "logs_dir": tmp_path / "logs",
        "vault_dir": tmp_path / "vault",
    }
    kwargs.update(overrides)
    return Settings(**kwargs)  # type: ignore[arg-type]


def test_sentence_transformer_adapter_enforces_offline_and_local_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Embedding adapter must refuse unconfigured/missing local models and enforce offline env."""
    unconfigured = SentenceTransformerEmbeddingAdapter(
        model_name_or_path="",
        models_dir=tmp_path / "models",
        offline_only=True,
    )
    ok, reason = unconfigured.is_available()
    assert ok is False
    assert reason is not None
    with pytest.raises(EmbeddingUnavailableError):
        unconfigured.embed_query("test")

    missing_path = SentenceTransformerEmbeddingAdapter(
        model_name_or_path=str(tmp_path / "does-not-exist"),
        models_dir=tmp_path / "models",
        offline_only=True,
    )
    ok_missing, reason_missing = missing_path.is_available()
    assert ok_missing is False
    assert "does not exist" in (reason_missing or "")

    # Local model folder inside models_dir resolves cleanly and loads with local_files_only=True.
    local_model_dir = tmp_path / "models" / "bge-small-local"
    local_model_dir.mkdir(parents=True)

    captured: dict[str, object] = {}

    class _FakeSentenceTransformer:
        def __init__(self, target: str, **kwargs: object) -> None:
            captured["target"] = target
            captured["kwargs"] = kwargs

        def encode(
            self,
            texts: list[str],
            batch_size: int = 32,
            **_kwargs: object,
        ) -> list[list[float]]:
            captured["batch_size"] = batch_size
            return [[float(idx + 1), 0.0] for idx, _ in enumerate(texts)]

    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    adapter = SentenceTransformerEmbeddingAdapter(
        model_name_or_path="bge-small-local",
        models_dir=tmp_path / "models",
        offline_only=True,
        loader=_FakeSentenceTransformer,
    )
    ok_local, err_local = adapter.is_available()
    assert ok_local is True
    assert err_local is None
    assert captured["target"] == str(local_model_dir)
    assert captured["kwargs"] == {"local_files_only": True}
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"

    vecs = adapter.embed_documents(["alpha", "beta", "gamma"], batch_size=2)
    assert len(vecs) == 3
    assert captured["batch_size"] == 2


def test_bm25_index_works_on_single_and_multi_doc_corpora() -> None:
    """BM25L index should return strictly positive scores even for a 1-document vault."""
    index = BM25Index()
    index.build(
        [
            {
                "id": "chk-only",
                "document_id": "doc-only",
                "file_type": "txt",
                "epoch": 1,
                "content": "Rare diagnostic code ERR_QUORUM_419 detected.",
            }
        ]
    )
    hits = index.search("ERR_QUORUM_419", top_k=5)
    assert len(hits) == 1
    assert hits[0][0] == "chk-only"
    assert hits[0][1] > 0.0

    # Zero matching tokens must return empty list.
    assert index.search("unrelated_token_xyz", top_k=5) == []


def test_empty_vault_searches_and_status(tmp_path: Path) -> None:
    """Searches against an empty vault must succeed cleanly with empty results."""
    embedder = ConceptEmbeddingAdapter()
    app = create_app(_make_settings(tmp_path), embedding_adapter=embedder)

    with TestClient(app) as client:
        status_resp = client.get("/api/v1/index/status")
        assert status_resp.status_code == 200
        status_body = status_resp.json()
        assert status_body["sqlite_indexed_documents"] == 0
        assert status_body["sqlite_indexed_chunks"] == 0
        assert status_body["bm25_indexed_chunks"] == 0
        assert status_body["vector_indexed_chunks"] == 0
        assert status_body["synchronized"] is True
        assert status_body["embedding_available"] is True

        for mode in ("hybrid", "semantic", "keyword"):
            resp = client.post(
                "/api/v1/search",
                json={"query": "cellular respiration", "mode": mode, "top_k": 5},
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["mode"] == mode
            assert body["total"] == 0
            assert body["results"] == []


def test_semantic_keyword_hybrid_and_rerank_search(tmp_path: Path) -> None:
    """Verify semantic, BM25 keyword, RRF hybrid weights, and optional reranking."""
    embedder = ConceptEmbeddingAdapter()
    app = create_app(_make_settings(tmp_path), embedding_adapter=embedder)

    with TestClient(app) as client:
        doc_bio = client.post(
            "/api/v1/ingest/file",
            files={
                "file": (
                    "bio.md",
                    b"Mitochondria drive ATP synthesis across biological membranes.",
                    "text/markdown",
                )
            },
            data={"epoch": "10"},
        )
        assert doc_bio.status_code == 201
        bio_id = doc_bio.json()["document_id"]

        doc_code = client.post(
            "/api/v1/ingest/file",
            files={
                "file": (
                    "ops.txt",
                    b"Error code ERR_QUORUM_419 occurred during local cluster sync.",
                    "text/plain",
                )
            },
            data={"epoch": "20"},
        )
        assert doc_code.status_code == 201
        code_id = doc_code.json()["document_id"]

        doc_phrase = client.post(
            "/api/v1/ingest/file",
            files={
                "file": (
                    "crypto.md",
                    b"AES encryption protects local vault keys. Zero-knowledge storage matters.",
                    "text/markdown",
                )
            },
            data={"epoch": "30"},
        )
        assert doc_phrase.status_code == 201

        # 1. Semantic search finds synonym match with zero shared words ("cellular respiration")
        sem_resp = client.post(
            "/api/v1/search",
            json={"query": "cellular respiration", "mode": "semantic", "top_k": 3},
        )
        assert sem_resp.status_code == 200
        sem_hits = sem_resp.json()["results"]
        assert len(sem_hits) >= 1
        assert sem_hits[0]["document_id"] == bio_id
        assert sem_hits[0]["semantic_rank"] == 1
        assert sem_hits[0]["keyword_rank"] is None
        assert sem_hits[0]["semantic_score"] > 0.9

        # 2. Keyword search finds exact rare identifier
        kw_resp = client.post(
            "/api/v1/search",
            json={"query": "ERR_QUORUM_419", "mode": "keyword", "top_k": 3},
        )
        assert kw_resp.status_code == 200
        kw_hits = kw_resp.json()["results"]
        assert len(kw_hits) == 1
        assert kw_hits[0]["document_id"] == code_id
        assert kw_hits[0]["keyword_rank"] == 1
        assert kw_hits[0]["semantic_rank"] is None

        # 3. Hybrid search combines both signals via configurable RRF weights
        # Query hits bio.md semantically ("cellular") and ops.txt lexically ("ERR_QUORUM_419")
        sem_heavy = client.post(
            "/api/v1/search",
            json={
                "query": "cellular ERR_QUORUM_419",
                "mode": "hybrid",
                "semantic_weight": 0.9,
                "bm25_weight": 0.1,
                "rerank": False,
                "top_k": 3,
            },
        )
        assert sem_heavy.status_code == 200
        assert sem_heavy.json()["results"][0]["document_id"] == bio_id

        kw_heavy = client.post(
            "/api/v1/search",
            json={
                "query": "cellular ERR_QUORUM_419",
                "mode": "hybrid",
                "semantic_weight": 0.1,
                "bm25_weight": 0.9,
                "rerank": False,
                "top_k": 3,
            },
        )
        assert kw_heavy.status_code == 200
        assert kw_heavy.json()["results"][0]["document_id"] == code_id

        # 4. Reranking promotes exact multi-term phrase match above partial term match
        client.post(
            "/api/v1/ingest/file",
            files={
                "file": (
                    "vault_only.txt",
                    b"Keys are stored in a vault without AES encryption here.",
                    "text/plain",
                )
            },
            data={"epoch": "31"},
        )
        reranked = client.post(
            "/api/v1/search",
            json={
                "query": "AES encryption",
                "mode": "hybrid",
                "rerank": True,
                "top_k": 4,
            },
        )
        assert reranked.status_code == 200
        top = reranked.json()["results"][0]
        assert top["filename"] == "crypto.md"
        assert top["score"] > top["rrf_score"]


def test_metadata_filters_by_epoch_document_and_file_type(tmp_path: Path) -> None:
    """Verify epoch range, document_ids, and file_types filtering across search modes."""
    embedder = ConceptEmbeddingAdapter()
    app = create_app(_make_settings(tmp_path), embedding_adapter=embedder)

    with TestClient(app) as client:
        r1 = client.post(
            "/api/v1/ingest/file",
            files={
                "file": (
                    "early.md",
                    b"Mitochondria cellular respiration pathway note.",
                    "text/markdown",
                )
            },
            data={"epoch": "5"},
        )
        r2 = client.post(
            "/api/v1/ingest/file",
            files={
                "file": (
                    "mid.txt",
                    b"Mitochondria ATP respiration pathway summary.",
                    "text/plain",
                )
            },
            data={"epoch": "15"},
        )
        r3 = client.post(
            "/api/v1/ingest/file",
            files={
                "file": (
                    "late.md",
                    b"Mitochondria cellular respiration benchmark update.",
                    "text/markdown",
                )
            },
            data={"epoch": "25"},
        )
        id1 = r1.json()["document_id"]
        id2 = r2.json()["document_id"]
        id3 = r3.json()["document_id"]

        # Epoch range filter across all 3 modes
        for mode in ("hybrid", "semantic", "keyword"):
            epoch_resp = client.post(
                "/api/v1/search",
                json={
                    "query": "Mitochondria respiration",
                    "mode": mode,
                    "epoch_min": 10,
                    "epoch_max": 20,
                },
            )
            assert epoch_resp.status_code == 200
            hits = epoch_resp.json()["results"]
            assert [h["document_id"] for h in hits] == [id2]

        # Compound filter: file_types=['markdown'] AND epoch_min=10 -> matches only late.md
        compound_resp = client.post(
            "/api/v1/search",
            json={
                "query": "Mitochondria respiration",
                "mode": "hybrid",
                "file_types": ["markdown"],
                "epoch_min": 10,
            },
        )
        assert compound_resp.status_code == 200
        assert [h["document_id"] for h in compound_resp.json()["results"]] == [id3]

        # Document ID filter
        doc_resp = client.post(
            "/api/v1/search",
            json={
                "query": "Mitochondria",
                "mode": "hybrid",
                "document_ids": [id1],
            },
        )
        assert doc_resp.status_code == 200
        assert [h["document_id"] for h in doc_resp.json()["results"]] == [id1]


def test_index_rebuild_deduplication_and_deletion_sync(tmp_path: Path) -> None:
    """Verify ChromaDB + BM25 stay synchronized with SQLite on rebuild and delete."""
    embedder = ConceptEmbeddingAdapter()
    app = create_app(_make_settings(tmp_path, embedding_batch_size=2), embedding_adapter=embedder)

    with TestClient(app) as client:
        r1 = client.post(
            "/api/v1/ingest/file",
            files={"file": ("doc1.txt", b"Astronomy nebula telescope observation.", "text/plain")},
            data={"epoch": "1"},
        )
        r2 = client.post(
            "/api/v1/ingest/file",
            files={"file": ("doc2.txt", b"Galaxy orbit telemetry report.", "text/plain")},
            data={"epoch": "2"},
        )
        id1 = r1.json()["document_id"]
        id2 = r2.json()["document_id"]

        status_before = client.get("/api/v1/index/status").json()
        assert status_before["sqlite_indexed_documents"] == 2
        assert status_before["sqlite_indexed_chunks"] == 2
        assert status_before["bm25_indexed_chunks"] == 2
        assert status_before["vector_indexed_chunks"] == 2
        assert status_before["synchronized"] is True

        # Rebuild must not create duplicate vector records
        rebuild_resp = client.post("/api/v1/index/rebuild", json={"batch_size": 1})
        assert rebuild_resp.status_code == 200
        rebuild_body = rebuild_resp.json()
        assert rebuild_body["status"] == "rebuilt"
        assert rebuild_body["sqlite_indexed_chunks"] == 2
        assert rebuild_body["bm25_indexed_chunks"] == 2
        assert rebuild_body["vector_indexed_chunks"] == 2
        assert rebuild_body["vectors_written"] == 2

        # Deleting doc1 must remove it from SQLite, BM25, and ChromaDB
        del_resp = client.delete(f"/api/v1/documents/{id1}")
        assert del_resp.status_code == 200

        status_after = client.get("/api/v1/index/status").json()
        assert status_after["sqlite_indexed_documents"] == 1
        assert status_after["sqlite_indexed_chunks"] == 1
        assert status_after["bm25_indexed_chunks"] == 1
        assert status_after["vector_indexed_chunks"] == 1
        assert status_after["synchronized"] is True

        search_after = client.post(
            "/api/v1/search",
            json={"query": "Astronomy nebula", "mode": "hybrid"},
        )
        assert search_after.status_code == 200
        returned_doc_ids = [h["document_id"] for h in search_after.json()["results"]]
        assert id1 not in returned_doc_ids
        assert id2 in returned_doc_ids


def test_graceful_degradation_when_embedding_model_unavailable(tmp_path: Path) -> None:
    """When no embedding model is available, ingestion and keyword/hybrid search still work."""
    app = create_app(_make_settings(tmp_path, embedding_model=""))

    with TestClient(app) as client:
        ingest_resp = client.post(
            "/api/v1/ingest/file",
            files={
                "file": (
                    "offline.md",
                    b"Local-first software keeps vector indexing optional.",
                    "text/markdown",
                )
            },
        )
        assert ingest_resp.status_code == 201
        assert any("Vector indexing skipped" in w for w in ingest_resp.json()["warnings"])

        status_body = client.get("/api/v1/index/status").json()
        assert status_body["embedding_available"] is False
        assert status_body["sqlite_indexed_chunks"] == 1
        assert status_body["bm25_indexed_chunks"] == 1
        assert status_body["vector_indexed_chunks"] == 0

        # Semantic-only search cleanly reports 503 unavailable
        sem_resp = client.post(
            "/api/v1/search",
            json={"query": "vector indexing", "mode": "semantic"},
        )
        assert sem_resp.status_code == 503

        # Hybrid search degrades gracefully to BM25 keyword results
        hyb_resp = client.post(
            "/api/v1/search",
            json={"query": "vector indexing", "mode": "hybrid"},
        )
        assert hyb_resp.status_code == 200
        hyb_body = hyb_resp.json()
        assert hyb_body["degraded_to_keyword"] is True
        assert len(hyb_body["warnings"]) >= 1
        assert hyb_body["total"] == 1
        assert hyb_body["results"][0]["filename"] == "offline.md"
        assert hyb_body["results"][0]["keyword_rank"] == 1


def test_search_validation_rejects_blank_query_and_inverted_epochs(tmp_path: Path) -> None:
    """Request validation must reject blank queries and epoch_min > epoch_max with 422."""
    app = create_app(_make_settings(tmp_path), embedding_adapter=ConceptEmbeddingAdapter())
    with TestClient(app) as client:
        blank_resp = client.post("/api/v1/search", json={"query": "   "})
        assert blank_resp.status_code == 422

        bad_epoch = client.post(
            "/api/v1/search",
            json={"query": "test", "epoch_min": 50, "epoch_max": 10},
        )
        assert bad_epoch.status_code == 422
