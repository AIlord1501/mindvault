"""Integration tests for /api/v1/ingest and /api/v1/documents endpoints."""

from __future__ import annotations

from pathlib import Path

import fitz
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app


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


def _make_pdf_bytes(text: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def test_ingest_markdown_deduplicate_list_get_and_delete(tmp_path: Path) -> None:
    app = create_app(_make_settings(tmp_path))
    md_bytes = (
        b"# Principles\n\n"
        b"MindVault stores all knowledge locally on disk.\n\n"
        b"# Retrieval\n\n"
        b"Hybrid retrieval combines lexical and vector search."
    )

    with TestClient(app) as client:
        # 1. Ingest Markdown file
        resp = client.post(
            "/api/v1/ingest/file",
            files={"file": ("../unsafe/notes.md", md_bytes, "text/markdown")},
            data={"epoch": "42", "chunk_size": "80", "chunk_overlap": "10"},
        )
        assert resp.status_code == 201
        summary = resp.json()
        assert summary["status"] == "indexed"
        assert summary["duplicate"] is False
        assert summary["filename"] == "notes.md"
        assert summary["file_type"] == "markdown"
        assert summary["epoch"] == 42
        assert summary["chunk_count"] >= 2
        assert "persisted" in summary["progress"]
        doc_id = summary["document_id"]
        vault_file = Path(summary["source_path"])
        assert vault_file.is_file()

        # 2. Duplicate upload detected without duplicating chunks
        dup = client.post(
            "/api/v1/ingest/file",
            files={"file": ("notes_copy.md", md_bytes, "text/markdown")},
        )
        assert dup.status_code == 200
        dup_body = dup.json()
        assert dup_body["status"] == "duplicate"
        assert dup_body["duplicate"] is True
        assert dup_body["document_id"] == doc_id

        # 3. List documents
        listing = client.get("/api/v1/documents?status=indexed&file_type=markdown")
        assert listing.status_code == 200
        list_body = listing.json()
        assert list_body["total"] == 1
        assert list_body["documents"][0]["id"] == doc_id
        assert list_body["documents"][0]["chunk_count"] == summary["chunk_count"]

        # 4. Get document details with ordered chunks
        detail = client.get(f"/api/v1/documents/{doc_id}")
        assert detail.status_code == 200
        detail_body = detail.json()
        assert detail_body["id"] == doc_id
        assert len(detail_body["chunks"]) == summary["chunk_count"]
        assert detail_body["chunks"][0]["chunk_index"] == 0
        assert detail_body["chunks"][0]["epoch"] == 42
        assert detail_body["chunks"][0]["metadata"]["filename"] == "notes.md"

        # 5. Delete document removes DB rows and managed vault file
        deleted = client.delete(f"/api/v1/documents/{doc_id}")
        assert deleted.status_code == 200
        assert deleted.json() == {"deleted": True, "document_id": doc_id}
        assert not vault_file.exists()

        # 6. Subsequent GET/DELETE return 404
        assert client.get(f"/api/v1/documents/{doc_id}").status_code == 404
        assert client.delete(f"/api/v1/documents/{doc_id}").status_code == 404


def test_ingest_pdf_corrupt_unsupported_oversized_and_graceful_audio(tmp_path: Path) -> None:
    app = create_app(_make_settings(tmp_path, max_upload_size_bytes=50_000))
    pdf_bytes = _make_pdf_bytes("PDF ingestion works completely offline.")

    with TestClient(app) as client:
        # Real PDF succeeds
        ok_pdf = client.post(
            "/api/v1/ingest/file",
            files={"file": ("report.pdf", pdf_bytes, "application/pdf")},
        )
        assert ok_pdf.status_code == 201
        assert ok_pdf.json()["file_type"] == "pdf"
        assert ok_pdf.json()["chunk_count"] == 1

        # Corrupted PDF returns 422 and records failed status
        bad_pdf = client.post(
            "/api/v1/ingest/file",
            files={"file": ("corrupt.pdf", b"%PDF-broken", "application/pdf")},
        )
        assert bad_pdf.status_code == 422
        bad_body = bad_pdf.json()
        assert bad_body["status"] == "failed"
        assert bad_body["error_code"] == "corrupted_file"
        assert bad_body["document_id"] is not None

        # Unsupported extension returns 415
        bad_ext = client.post(
            "/api/v1/ingest/file",
            files={"file": ("script.exe", b"MZ...", "application/octet-stream")},
        )
        assert bad_ext.status_code == 415
        assert bad_ext.json()["error_code"] == "unsupported_file_type"

        # Empty and oversized files return 400
        empty_resp = client.post(
            "/api/v1/ingest/file",
            files={"file": ("empty.txt", b"", "text/plain")},
        )
        assert empty_resp.status_code == 400
        assert empty_resp.json()["error_code"] == "file_size_limit"

        big_resp = client.post(
            "/api/v1/ingest/file",
            files={"file": ("big.txt", b"a" * 60_000, "text/plain")},
        )
        assert big_resp.status_code == 400
        assert big_resp.json()["error_code"] == "file_size_limit"

        # Optional audio adapter degrades gracefully with 503 when Whisper is absent
        audio_resp = client.post(
            "/api/v1/ingest/file",
            files={"file": ("memo.wav", b"RIFF0000WAVE", "audio/wav")},
        )
        assert audio_resp.status_code == 503
        assert audio_resp.json()["status"] == "failed"
        assert audio_resp.json()["error_code"] == "adapter_unavailable"


def test_ingest_batch_handles_mixed_valid_duplicate_and_failed(tmp_path: Path) -> None:
    app = create_app(_make_settings(tmp_path))

    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/ingest/batch",
            files=[
                ("files", ("a.txt", b"First local note.", "text/plain")),
                ("files", ("a_dup.txt", b"First local note.", "text/plain")),
                ("files", ("bad.zip", b"PK\x03\x04", "application/zip")),
            ],
            data={"epoch": "7"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 3
        assert body["indexed"] == 1
        assert body["duplicates"] == 1
        assert body["failed"] == 1
        assert len(body["results"]) == 3
        assert body["results"][0]["status"] == "indexed"
        assert body["results"][1]["status"] == "duplicate"
        assert body["results"][2]["status"] == "failed"
