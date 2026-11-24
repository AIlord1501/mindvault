"""Unit and integration tests for MindVault Concept and Citation knowledge graphs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterator, Optional

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.models.embeddings import BaseEmbeddingAdapter
from app.models.llm import BaseLLMProvider, CancelCheck
from app.models.nli import BaseNLIAdapter, NLIScores


class _GraphConceptEmbedder(BaseEmbeddingAdapter):
    """Deterministic topic embedder for chunk similarity & orphan isolation tests."""

    @property
    def model_name(self) -> str:
        return "graph-topic-embedder"

    def is_available(self) -> tuple[bool, Optional[str]]:
        return True, None

    @staticmethod
    def _vec(text: str) -> list[float]:
        lower = text.lower()
        vault_db = sum(
            1.0
            for kw in ("sqlite", "database", "storage", "wal", "chromadb", "vector", "local")
            if kw in lower
        )
        astro = sum(
            1.0 for kw in ("galaxy", "nebula", "telescope", "orbit", "exoplanet") if kw in lower
        )
        music = sum(
            1.0 for kw in ("violin", "symphony", "sonata", "harmony", "tempo") if kw in lower
        )
        neg_axis = 1.5 if ("not" in lower and "only" in lower) else 0.0
        fallback = (
            0.25
            if (vault_db == 0.0 and astro == 0.0 and music == 0.0 and neg_axis == 0.0)
            else 0.0
        )
        vec = [vault_db, astro, music, neg_axis, fallback]
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [round(v / norm, 6) for v in vec]

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class _GraphMockLLM(BaseLLMProvider):
    """Mock local LLM supporting both chat streaming and JSON belief extraction."""

    @property
    def provider_name(self) -> str:
        return "mock-local"

    @property
    def model_name(self) -> str:
        return "mock-graph-gguf"

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
        if "extract" in system_prompt.lower() and "vault_chunk" in user_prompt.lower():
            if "sqlite" in user_prompt.lower():
                yield json.dumps(
                    {
                        "beliefs": [
                            {
                                "text": "MindVault uses SQLite WAL locally.",
                                "confidence": 0.92,
                            }
                        ]
                    }
                )
            elif "chromadb" in user_prompt.lower():
                yield json.dumps(
                    {
                        "beliefs": [
                            {
                                "text": "MindVault does not use SQLite and uses ChromaDB only.",
                                "confidence": 0.88,
                            }
                        ]
                    }
                )
            else:
                yield json.dumps(
                    {"beliefs": [{"text": "Offline data stays local.", "confidence": 0.80}]}
                )
            return

        yield "SQLite WAL and ChromaDB store vault data locally [C1]."


class _GraphMockNLI(BaseNLIAdapter):
    """Mock NLI adapter creating a contradiction edge between SQLite and Chroma-only beliefs."""

    @property
    def model_name(self) -> str:
        return "mock-graph-nli"

    def is_available(self) -> tuple[bool, Optional[str]]:
        return True, None

    def predict_pairs(self, pairs: list[tuple[str, str]]) -> list[NLIScores]:
        out: list[NLIScores] = []
        for premise, hypothesis in pairs:
            if "not use sqlite" in hypothesis.lower() and "uses sqlite" in premise.lower():
                out.append(NLIScores(contradiction=0.91, entailment=0.03, neutral=0.06))
            else:
                out.append(NLIScores(contradiction=0.05, entailment=0.10, neutral=0.85))
        return out


def _make_settings(tmp_path: Path, **overrides: object) -> Settings:
    kwargs: dict[str, object] = {
        "data_dir": tmp_path,
        "sqlite_path": tmp_path / "mindvault.db",
        "chroma_dir": tmp_path / "chroma",
        "models_dir": tmp_path / "models",
        "logs_dir": tmp_path / "logs",
        "vault_dir": tmp_path / "vault",
        "graph_similarity_threshold": 0.65,
        "graph_max_neighbors_per_node": 6,
    }
    kwargs.update(overrides)
    return Settings(**kwargs)  # type: ignore[arg-type]


def test_concept_graph_ann_similarity_beliefs_and_orphan_visibility(tmp_path: Path) -> None:
    """Concept graph builds similarity/belief edges in D3 format and keeps orphan nodes."""
    app = create_app(
        _make_settings(tmp_path),
        embedding_adapter=_GraphConceptEmbedder(),
        llm_provider=_GraphMockLLM(),
        nli_adapter=_GraphMockNLI(),
    )

    with TestClient(app) as client:
        r1 = client.post(
            "/api/v1/ingest/file",
            files={
                "file": (
                    "db_note.md",
                    b"SQLite database uses local WAL storage.",
                    "text/markdown",
                )
            },
            data={"epoch": "10"},
        )
        r2 = client.post(
            "/api/v1/ingest/file",
            files={
                "file": (
                    "vec_note.md",
                    b"ChromaDB vector storage persists local chunks.",
                    "text/markdown",
                )
            },
            data={"epoch": "20"},
        )
        r_iso = client.post(
            "/api/v1/ingest/file",
            files={
                "file": (
                    "music_orphan.md",
                    b"Violin symphony harmony uses gentle tempo.",
                    "text/markdown",
                )
            },
            data={"epoch": "30"},
        )
        assert r1.status_code == 201 and r2.status_code == 201 and r_iso.status_code == 201
        doc1 = r1.json()["document_id"]
        doc2 = r2.json()["document_id"]
        doc_iso = r_iso.json()["document_id"]

        c1_id = client.get(f"/api/v1/documents/{doc1}").json()["chunks"][0]["id"]
        c2_id = client.get(f"/api/v1/documents/{doc2}").json()["chunks"][0]["id"]
        c_iso_id = client.get(f"/api/v1/documents/{doc_iso}").json()["chunks"][0]["id"]

        # Extract beliefs on both storage chunks -> creates derived_from + contradicts edges
        assert client.post(f"/api/v1/beliefs/extract/{c1_id}").status_code == 201
        assert client.post(f"/api/v1/beliefs/extract/{c2_id}").status_code == 201

        graph_resp = client.get("/api/v1/graph/concepts")
        assert graph_resp.status_code == 200
        body = graph_resp.json()
        assert body["graph_type"] == "concepts"

        node_by_id = {n["id"]: n for n in body["nodes"]}
        assert c1_id in node_by_id
        assert c2_id in node_by_id
        assert c_iso_id in node_by_id

        # Orphan chunk must remain visible with degree=0 and orphan=True
        assert node_by_id[c_iso_id]["orphan"] is True
        assert node_by_id[c_iso_id]["degree"] == 0
        assert body["stats"]["orphan_count"] >= 1

        # All returned D3 edges must reference existing node IDs in the response
        edge_types = {e["edge_type"] for e in body["edges"]}
        assert "similar_chunk" in edge_types
        assert "derived_from" in edge_types
        assert "contradicts" in edge_types
        for edge in body["edges"]:
            assert edge["source"] in node_by_id
            assert edge["target"] in node_by_id


def test_citation_graph_cross_refs_queries_and_co_citations(tmp_path: Path) -> None:
    """Citation graph models source_of, adjacent_chunk, cross_reference, cites, co_cited."""
    app = create_app(
        _make_settings(tmp_path),
        embedding_adapter=_GraphConceptEmbedder(),
        llm_provider=_GraphMockLLM(),
        nli_adapter=_GraphMockNLI(),
    )

    with TestClient(app) as client:
        # Doc 1 has two chunks (adjacent_chunk) and links to Doc 2 (cross_reference)
        overview_text = (
            "# Intro\n\n"
            "Local SQLite storage details are in [Storage Spec](storage_spec.md).\n\n"
            "# Vector Layer\n\n"
            "ChromaDB persists local vector indexes."
        ).encode("utf-8")
        r_overview = client.post(
            "/api/v1/ingest/file",
            files={"file": ("overview.md", overview_text, "text/markdown")},
            data={"epoch": "10", "chunk_size": "95", "chunk_overlap": "0"},
        )
        r_spec = client.post(
            "/api/v1/ingest/file",
            files={
                "file": (
                    "storage_spec.md",
                    b"SQLite WAL database storage specification.",
                    "text/markdown",
                )
            },
            data={"epoch": "15"},
        )
        assert r_overview.status_code == 201 and r_spec.status_code == 201
        assert r_overview.json()["chunk_count"] >= 2

        # Run a grounded chat query referencing >=2 chunks -> creates query node, cites, co_cited
        chat_resp = client.post(
            "/api/v1/chat",
            json={"query": "How do SQLite and ChromaDB work locally?", "top_k": 5},
        )
        assert chat_resp.status_code == 200
        query_id = chat_resp.json()["query_id"]
        assert len(chat_resp.json()["citations"]) >= 2

        cit_resp = client.get("/api/v1/graph/citations")
        assert cit_resp.status_code == 200
        cit_body = cit_resp.json()
        assert cit_body["graph_type"] == "citations"

        node_types = {n["node_type"] for n in cit_body["nodes"]}
        edge_types = {e["edge_type"] for e in cit_body["edges"]}
        assert {"document", "chunk", "query"} <= node_types
        assert {"source_of", "adjacent_chunk", "cross_reference", "cites", "co_cited"} <= edge_types

        # Query ego-network exploration via bare query_id works cleanly
        ego_resp = client.get(f"/api/v1/graph/node/{query_id}?hops=1")
        assert ego_resp.status_code == 200
        assert ego_resp.json()["center_id"] == f"query:{query_id}"
        assert ego_resp.json()["stats"]["edge_count"] >= 2


def test_graph_filters_epoch_document_node_edge_and_search(tmp_path: Path) -> None:
    """Verify epoch, document, node_type, edge_type, and search filters + 422 validation."""
    app = create_app(
        _make_settings(tmp_path),
        embedding_adapter=_GraphConceptEmbedder(),
        llm_provider=_GraphMockLLM(),
        nli_adapter=_GraphMockNLI(),
    )

    with TestClient(app) as client:
        d1 = client.post(
            "/api/v1/ingest/file",
            files={"file": ("alpha.md", b"SQLite database note epoch 10.", "text/markdown")},
            data={"epoch": "10"},
        ).json()["document_id"]
        d2 = client.post(
            "/api/v1/ingest/file",
            files={"file": ("beta.md", b"Galaxy telescope orbit note epoch 40.", "text/markdown")},
            data={"epoch": "40"},
        ).json()["document_id"]

        c1_id = client.get(f"/api/v1/documents/{d1}").json()["chunks"][0]["id"]
        client.post(f"/api/v1/beliefs/extract/{c1_id}")

        # Epoch filter
        epoch_res = client.get("/api/v1/graph/concepts?epoch_min=30&epoch_max=50").json()
        assert all(30 <= n["epoch"] <= 50 for n in epoch_res["nodes"])
        assert len(epoch_res["nodes"]) == 1
        assert epoch_res["nodes"][0]["document_id"] == d2

        # Document filter
        doc_res = client.get(f"/api/v1/graph/concepts?document_id={d1}").json()
        assert len(doc_res["nodes"]) == 2  # 1 chunk + 1 belief
        assert all(n["document_id"] == d1 for n in doc_res["nodes"])

        # Node type filter
        belief_only = client.get("/api/v1/graph/concepts?node_type=belief").json()
        assert len(belief_only["nodes"]) == 1
        assert belief_only["nodes"][0]["node_type"] == "belief"

        # Edge type filter
        derived_only = client.get("/api/v1/graph/concepts?edge_type=derived_from").json()
        assert len(derived_only["edges"]) == 1
        assert derived_only["edges"][0]["edge_type"] == "derived_from"

        # Search filter (content match finds chunk; filename match finds document + chunk)
        search_chunk = client.get("/api/v1/graph/citations?search=telescope").json()
        assert len(search_chunk["nodes"]) == 1
        assert search_chunk["nodes"][0]["node_type"] == "chunk"
        assert search_chunk["nodes"][0]["document_id"] == d2

        search_doc = client.get("/api/v1/graph/citations?search=beta").json()
        assert all(n["document_id"] == d2 for n in search_doc["nodes"])
        assert len(search_doc["nodes"]) == 2  # document + chunk

        # Inverted epochs return 422
        assert client.get("/api/v1/graph/concepts?epoch_min=50&epoch_max=10").status_code == 422
        assert client.get("/api/v1/graph/node/unknown-id").status_code == 404


def test_incremental_updates_rebuild_and_deletion_consistency(tmp_path: Path) -> None:
    """Incremental indexing, rebuild deduplication, and deletion keep graph consistent."""
    app = create_app(
        _make_settings(tmp_path),
        embedding_adapter=_GraphConceptEmbedder(),
        llm_provider=_GraphMockLLM(),
        nli_adapter=_GraphMockNLI(),
    )

    with TestClient(app) as client:
        d1 = client.post(
            "/api/v1/ingest/file",
            files={"file": ("s1.md", b"SQLite local storage note one.", "text/markdown")},
            data={"epoch": "1"},
        ).json()["document_id"]
        d2 = client.post(
            "/api/v1/ingest/file",
            files={"file": ("s2.md", b"ChromaDB vector storage note two.", "text/markdown")},
            data={"epoch": "2"},
        ).json()["document_id"]

        # Incremental update after 2nd doc already creates similar_chunk edge without /rebuild
        concepts_2 = client.get("/api/v1/graph/concepts?edge_type=similar_chunk").json()
        assert concepts_2["stats"]["edge_count"] == 1

        # Adding a 3rd similar doc incrementally adds edges to prior chunks
        d3 = client.post(
            "/api/v1/ingest/file",
            files={"file": ("s3.md", b"Local database storage note three.", "text/markdown")},
            data={"epoch": "3"},
        ).json()["document_id"]
        concepts_3 = client.get("/api/v1/graph/concepts?edge_type=similar_chunk").json()
        assert concepts_3["stats"]["edge_count"] == 3

        # Rebuild cleanly preserves exact edge counts without duplicates
        rebuild = client.post("/api/v1/graph/rebuild", json={"similarity_threshold": 0.65})
        assert rebuild.status_code == 200
        rebuild_body = rebuild.json()
        assert rebuild_body["status"] == "rebuilt"
        assert rebuild_body["edge_counts"]["similar_chunk"] == 3
        assert rebuild_body["edge_counts"]["source_of"] == 3

        # Deleting doc 1 removes its document/chunk/belief nodes AND all incident edges
        c1_id = client.get(f"/api/v1/documents/{d1}").json()["chunks"][0]["id"]
        assert client.delete(f"/api/v1/documents/{d1}").status_code == 200

        after_del = client.get("/api/v1/graph/concepts").json()
        remaining_ids = {n["id"] for n in after_del["nodes"]}
        assert c1_id not in remaining_ids
        for edge in after_del["edges"]:
            assert edge["source"] != c1_id and edge["target"] != c1_id
            assert edge["source"] in remaining_ids and edge["target"] in remaining_ids

        cit_after = client.get("/api/v1/graph/citations").json()
        cit_ids = {n["id"] for n in cit_after["nodes"]}
        assert d1 not in cit_ids and d2 in cit_ids and d3 in cit_ids
