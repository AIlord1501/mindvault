"""Unit and integration tests for MindVault belief extraction and contradiction engine."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Iterator, Optional

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.models.embeddings import BaseEmbeddingAdapter
from app.models.llm import BaseLLMProvider, CancelCheck
from app.models.nli import BaseNLIAdapter, LocalSequenceClassificationNLIAdapter, NLIScores


class _BeliefTopicEmbedder(BaseEmbeddingAdapter):
    """Deterministic multi-axis embedder modeling topic similarity vs semantic divergence."""

    @property
    def model_name(self) -> str:
        return "belief-topic-embedder"

    def is_available(self) -> tuple[bool, Optional[str]]:
        return True, None

    @staticmethod
    def _vec(text: str) -> list[float]:
        lower = text.lower()
        db_topic = sum(
            1.0
            for kw in ("sqlite", "postgres", "database", "storage", "metadata", "local")
            if kw in lower
        )
        neg_pg = sum(1.5 for kw in ("not", "postgres", "migrated") if kw in lower)
        offline_axis = sum(1.5 for kw in ("offline", "strictly") if kw in lower)
        entail_axis = 2.8 if "keeps" in lower else 0.0
        loopback_axis = sum(1.5 for kw in ("loopback", "enforces", "desktop") if kw in lower)
        cache_axis = sum(1.0 for kw in ("redis", "cache", "ephemeral", "session") if kw in lower)
        ui_axis = sum(
            1.0 for kw in ("electron", "sandbox", "theme", "frameless", "palette") if kw in lower
        )
        astro_axis = sum(
            1.0 for kw in ("jupiter", "orbit", "telescope", "galaxy", "nebula") if kw in lower
        )
        vec = [
            db_topic,
            neg_pg,
            offline_axis,
            entail_axis,
            loopback_axis,
            cache_axis,
            ui_axis,
            astro_axis,
        ]
        if sum(vec) == 0.0:
            vec.append(1.0)
        else:
            vec.append(0.0)
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [round(v / norm, 6) for v in vec]

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class _MockExtractionLLM(BaseLLMProvider):
    """Mock local LLM returning deterministic JSON belief payloads based on chunk content."""

    def __init__(self, *, available: bool = True) -> None:
        self._available = available

    @property
    def provider_name(self) -> str:
        return "mock-local"

    @property
    def model_name(self) -> str:
        return "mock-extractor-gguf"

    def is_available(self) -> tuple[bool, Optional[str]]:
        if not self._available:
            return False, "Mock GGUF extractor unavailable"
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
        if "overflow_many_and_long" in lower:
            payload = {
                "beliefs": [
                    {
                        "text": (
                            "MindVault enforces strict local loopback desktop architecture "
                            "across every workstation even when users configure custom "
                            "storage locations and extra metadata fields"
                        ),
                        "confidence": 0.92,
                    },
                    {"text": "Electron disables nodeIntegration in renderer.", "confidence": 0.88},
                    {"text": "Jupiter orbits the sun every twelve years.", "confidence": 0.85},
                    {"text": "Low confidence guess ignored.", "confidence": 0.15},
                    {"text": "Belief four stays under limit.", "confidence": 0.77},
                    {"text": "Belief five stays under limit.", "confidence": 0.76},
                    {"text": "Sixth belief must be dropped by cap.", "confidence": 0.95},
                ]
            }
        elif "duplicate_test" in lower:
            payload = {
                "beliefs": [
                    {"text": "Redis cache stores ephemeral session state.", "confidence": 0.90},
                    {
                        "text": "Ephemeral session state uses Redis cache.",
                        "confidence": 0.85,
                    },
                ]
            }
        elif "epoch_10_sqlite" in lower:
            payload = {
                "beliefs": [
                    {
                        "text": "MindVault uses SQLite for local database storage.",
                        "confidence": 0.91,
                    }
                ]
            }
        elif "epoch_25_postgres" in lower:
            payload = {
                "beliefs": [
                    {
                        "text": "MindVault does not use SQLite and uses Postgres storage.",
                        "confidence": 0.94,
                    },
                    {
                        "text": "Local database storage runs strictly offline.",
                        "confidence": 0.89,
                    },
                ]
            }
        elif "entailment_probe" in lower:
            payload = {
                "beliefs": [
                    {
                        "text": "Local database storage keeps SQLite metadata.",
                        "confidence": 0.86,
                    }
                ]
            }
        else:
            payload = {
                "beliefs": [
                    {"text": "Offline knowledge base runs locally.", "confidence": 0.80}
                ]
            }
        yield json.dumps(payload)


class _MockNLIAdapter(BaseNLIAdapter):
    """Mock local NLI model recording evaluated pairs and scoring relationships."""

    def __init__(self, *, available: bool = True) -> None:
        self._available = available
        self.calls: list[list[tuple[str, str]]] = []

    @property
    def model_name(self) -> str:
        return "mock-deberta-nli"

    def is_available(self) -> tuple[bool, Optional[str]]:
        if not self._available:
            return False, "Mock NLI unavailable"
        return True, None

    def predict_pairs(self, pairs: list[tuple[str, str]]) -> list[NLIScores]:
        self.calls.append(list(pairs))
        out: list[NLIScores] = []
        for premise, hypothesis in pairs:
            p = premise.lower()
            h = hypothesis.lower()
            if ("not use sqlite" in h and "uses sqlite" in p) or (
                "not use sqlite" in p and "uses sqlite" in h
            ):
                out.append(NLIScores(contradiction=0.93, entailment=0.02, neutral=0.05))
            elif "keeps sqlite metadata" in h and "uses sqlite" in p:
                out.append(NLIScores(contradiction=0.03, entailment=0.88, neutral=0.09))
            else:
                out.append(NLIScores(contradiction=0.04, entailment=0.11, neutral=0.85))
        return out


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


def _ingest_and_get_chunk_id(client: TestClient, filename: str, text: str, epoch: int) -> str:
    ingest_resp = client.post(
        "/api/v1/ingest/file",
        files={"file": (filename, text.encode("utf-8"), "text/markdown")},
        data={"epoch": str(epoch)},
    )
    assert ingest_resp.status_code == 201
    doc_id = ingest_resp.json()["document_id"]
    detail = client.get(f"/api/v1/documents/{doc_id}")
    assert detail.status_code == 200
    return detail.json()["chunks"][0]["id"]


def test_local_nli_adapter_enforces_offline_env_and_missing_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """LocalSequenceClassificationNLIAdapter enforces offline flags and local paths."""
    unconfigured = LocalSequenceClassificationNLIAdapter(model_path="", models_dir=tmp_path)
    ok, reason = unconfigured.is_available()
    assert ok is False
    assert "MINDVAULT_NLI_MODEL_PATH" in (reason or "")

    missing = LocalSequenceClassificationNLIAdapter(
        model_path=str(tmp_path / "missing-nli"),
        models_dir=tmp_path,
    )
    ok_missing, reason_missing = missing.is_available()
    assert ok_missing is False
    assert "does not exist" in (reason_missing or "")

    model_dir = tmp_path / "nli-small"
    model_dir.mkdir()
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    loaded_targets: list[str] = []

    def _fake_loader(target: str):
        loaded_targets.append(target)
        return object(), object()

    adapter = LocalSequenceClassificationNLIAdapter(
        model_path="nli-small",
        models_dir=tmp_path,
        offline_only=True,
        loader=_fake_loader,
    )
    assert adapter.is_available() == (True, None)
    assert loaded_targets == [str(model_dir)]
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"


def test_belief_extractor_enforces_caps_20_word_limit_and_dedup(tmp_path: Path) -> None:
    """Extractor caps beliefs at 1-5, <=20 words, filters low confidence, and deduplicates."""
    app = create_app(
        _make_settings(tmp_path, belief_dedup_threshold=0.90, belief_min_confidence=0.30),
        embedding_adapter=_BeliefTopicEmbedder(),
        llm_provider=_MockExtractionLLM(),
        nli_adapter=_MockNLIAdapter(),
    )

    with TestClient(app) as client:
        chunk_1 = _ingest_and_get_chunk_id(client, "c1.md", "overflow_many_and_long note", epoch=5)
        resp_1 = client.post(f"/api/v1/beliefs/extract/{chunk_1}")
        assert resp_1.status_code == 201
        body_1 = resp_1.json()
        assert 1 <= body_1["extracted_count"] <= 5
        assert body_1["low_confidence_skipped"] == 1
        assert body_1["epoch"] == 5

        for belief in body_1["beliefs"]:
            assert len(belief["belief_text"].split()) <= 20
            assert belief["chunk_id"] == chunk_1
            assert belief["epoch"] == 5
            assert 0.30 <= belief["confidence"] <= 1.0

        # Duplicate beliefs with cosine >= 0.90 are deduplicated automatically
        chunk_2 = _ingest_and_get_chunk_id(client, "c2.md", "duplicate_test content", epoch=6)
        resp_2 = client.post(f"/api/v1/beliefs/extract/{chunk_2}")
        assert resp_2.status_code == 201
        body_2 = resp_2.json()
        assert body_2["extracted_count"] == 1
        assert body_2["duplicates_skipped"] == 1


def test_extractor_handles_missing_chunk_and_unavailable_llm(tmp_path: Path) -> None:
    """Missing chunk returns 404 and unavailable local LLM returns 503."""
    app = create_app(
        _make_settings(tmp_path),
        embedding_adapter=_BeliefTopicEmbedder(),
        llm_provider=_MockExtractionLLM(available=False),
        nli_adapter=_MockNLIAdapter(),
    )

    with TestClient(app) as client:
        assert client.post("/api/v1/beliefs/extract/does-not-exist").status_code == 404

        chunk_id = _ingest_and_get_chunk_id(client, "a.md", "Offline text.", epoch=1)
        resp = client.post(f"/api/v1/beliefs/extract/{chunk_id}")
        assert resp.status_code == 503


def test_contradiction_detector_filters_candidates_and_stores_all_nli_scores(
    tmp_path: Path,
) -> None:
    """Semantic filtering skips unrelated beliefs before NLI and stores all 3 relation types."""
    mock_nli = _MockNLIAdapter()
    app = create_app(
        _make_settings(
            tmp_path,
            belief_candidate_top_k=5,
            belief_candidate_min_similarity=0.35,
        ),
        embedding_adapter=_BeliefTopicEmbedder(),
        llm_provider=_MockExtractionLLM(),
        nli_adapter=mock_nli,
    )

    with TestClient(app) as client:
        # 1. Seed UI, loopback, and unrelated astronomy beliefs
        c_seed = _ingest_and_get_chunk_id(client, "seed.md", "overflow_many_and_long", epoch=5)
        r_seed = client.post(f"/api/v1/beliefs/extract/{c_seed}")
        assert r_seed.status_code == 201

        # 2. Extract epoch 10 sqlite belief
        c_e10 = _ingest_and_get_chunk_id(client, "e10.md", "epoch_10_sqlite", epoch=10)
        r_e10 = client.post(f"/api/v1/beliefs/extract/{c_e10}")
        assert r_e10.status_code == 201
        sqlite_belief_id = r_e10.json()["beliefs"][0]["id"]

        # 3. Extract epoch 25 conflicting postgres belief + neutral offline belief
        mock_nli.calls.clear()
        c_e25 = _ingest_and_get_chunk_id(client, "e25.md", "epoch_25_postgres", epoch=25)
        r_e25 = client.post(f"/api/v1/beliefs/extract/{c_e25}")
        assert r_e25.status_code == 201
        e25_body = r_e25.json()
        assert e25_body["nli_mode"] == "local_nli_model"
        assert e25_body["nli_model"] == "mock-deberta-nli"
        assert len(e25_body["contradictions"]) == 1

        # Unrelated astronomy belief ("Jupiter orbits...") must NOT be sent to NLI
        evaluated_premises = [p for batch in mock_nli.calls for p, _h in batch]
        assert not any("Jupiter" in p for p in evaluated_premises)

        # Verify entailment, neutral, and contradiction relations stored in belief_relations
        c_ent = _ingest_and_get_chunk_id(client, "ent.md", "entailment_probe", epoch=26)
        r_ent = client.post(f"/api/v1/beliefs/extract/{c_ent}")
        assert r_ent.status_code == 201
        labels = {rel["label"] for rel in r_ent.json()["relations"] + e25_body["relations"]}
        assert "contradiction" in labels
        assert "entailment" in labels
        assert "neutral" in labels

        detail = client.get(f"/api/v1/beliefs/{sqlite_belief_id}").json()
        assert len(detail["relations"]) >= 1
        assert len(detail["contradictions"]) == 1


def test_all_five_contradiction_resolutions_and_epoch_history_retention(tmp_path: Path) -> None:
    """Verify accept_newer, accept_older, keep_both, context_dependent, and false_positive."""

    def _create_isolated_contradiction(subdir: str) -> tuple[TestClient, str, str, str]:
        case_dir = tmp_path / subdir
        app = create_app(
            _make_settings(case_dir),
            embedding_adapter=_BeliefTopicEmbedder(),
            llm_provider=_MockExtractionLLM(),
            nli_adapter=_MockNLIAdapter(),
        )
        client = TestClient(app)
        client.__enter__()

        c_old = _ingest_and_get_chunk_id(client, "old.md", "epoch_10_sqlite", epoch=10)
        r_old = client.post(f"/api/v1/beliefs/extract/{c_old}")
        old_id = r_old.json()["beliefs"][0]["id"]

        c_new = _ingest_and_get_chunk_id(client, "new.md", "epoch_25_postgres", epoch=25)
        r_new = client.post(f"/api/v1/beliefs/extract/{c_new}")
        cons = r_new.json()["contradictions"]
        assert len(cons) == 1
        return client, cons[0]["id"], old_id, cons[0]["belief_b_id"]

    # 1. accept_newer -> older belief superseded (retained, not deleted), newer active
    c1, con1, old1, new1 = _create_isolated_contradiction("case1")
    try:
        res1 = c1.post(
            f"/api/v1/contradictions/{con1}/resolve",
            json={"resolution": "accept_newer", "note": "Migrated in epoch 25"},
        )
        assert res1.status_code == 200
        b1 = res1.json()
        assert b1["status"] == "resolved"
        assert b1["resolution"] == "accept_newer"
        assert b1["belief_a"]["status"] == "superseded"
        assert b1["belief_a"]["superseded_by_id"] == new1
        assert b1["belief_b"]["status"] == "active"

        new_detail = c1.get(f"/api/v1/beliefs/{new1}").json()
        assert old1 in new_detail["supersedes_ids"]
        assert [item["epoch"] for item in new_detail["lineage_history"]] == [10, 25]
        sup_list = c1.get("/api/v1/beliefs?status=superseded").json()
        assert sup_list["total"] == 1
    finally:
        c1.__exit__(None, None, None)

    # 2. accept_older -> newer belief superseded by older belief
    c2, con2, old2, _new2 = _create_isolated_contradiction("case2")
    try:
        res2 = c2.post(
            f"/api/v1/contradictions/{con2}/resolve",
            json={"resolution": "accept_older"},
        )
        assert res2.status_code == 200
        assert res2.json()["belief_a"]["status"] == "active"
        assert res2.json()["belief_b"]["status"] == "superseded"
        assert res2.json()["belief_b"]["superseded_by_id"] == old2
    finally:
        c2.__exit__(None, None, None)

    # 3. keep_both -> both retained as active
    c3, con3, old3, new3 = _create_isolated_contradiction("case3")
    try:
        res3 = c3.post(
            f"/api/v1/contradictions/{con3}/resolve",
            json={"resolution": "keep_both"},
        )
        assert res3.status_code == 200
        assert res3.json()["resolution"] == "keep_both"
        assert c3.get(f"/api/v1/beliefs/{old3}").json()["status"] == "active"
        assert c3.get(f"/api/v1/beliefs/{new3}").json()["status"] == "active"
    finally:
        c3.__exit__(None, None, None)

    # 4. context_dependent -> both active with context_note
    c4, con4, _old4, _new4 = _create_isolated_contradiction("case4")
    try:
        note_text = "Desktop uses SQLite, server Postgres"
        res4 = c4.post(
            f"/api/v1/contradictions/{con4}/resolve",
            json={"resolution": "context_dependent", "note": note_text},
        )
        assert res4.status_code == 200
        assert res4.json()["resolution"] == "context_dependent"
        assert res4.json()["belief_a"]["context_note"] == note_text
        assert res4.json()["belief_b"]["context_note"] == note_text
    finally:
        c4.__exit__(None, None, None)

    # 5. dismiss (false_positive) -> status='dismissed', both beliefs restored active
    c5, con5, old5, new5 = _create_isolated_contradiction("case5")
    try:
        res5 = c5.post(
            f"/api/v1/contradictions/{con5}/dismiss",
            json={"note": "Different scopes"},
        )
        assert res5.status_code == 200
        assert res5.json()["status"] == "dismissed"
        assert res5.json()["resolution"] == "false_positive"
        assert c5.get(f"/api/v1/beliefs/{old5}").json()["status"] == "active"
        assert c5.get(f"/api/v1/beliefs/{new5}").json()["status"] == "active"

        dismissed = c5.get("/api/v1/contradictions?status=dismissed").json()
        assert dismissed["total"] == 1
    finally:
        c5.__exit__(None, None, None)


def test_heuristic_nli_fallback_when_no_model_configured(tmp_path: Path) -> None:
    """Without MINDVAULT_NLI_MODEL_PATH, detector degrades gracefully to heuristic NLI."""
    app = create_app(
        _make_settings(tmp_path, nli_model_path=""),
        embedding_adapter=_BeliefTopicEmbedder(),
        llm_provider=_MockExtractionLLM(),
    )

    with TestClient(app) as client:
        c1 = _ingest_and_get_chunk_id(client, "h1.md", "epoch_10_sqlite", epoch=10)
        client.post(f"/api/v1/beliefs/extract/{c1}")

        c2 = _ingest_and_get_chunk_id(client, "h2.md", "epoch_25_postgres", epoch=25)
        r2 = client.post(f"/api/v1/beliefs/extract/{c2}")
        assert r2.status_code == 201
        body = r2.json()
        assert body["nli_mode"] == "heuristic_fallback"
        assert body["nli_model"] == "heuristic-lexical-nli"
        assert len(body["warnings"]) >= 1
        assert len(body["contradictions"]) == 1
