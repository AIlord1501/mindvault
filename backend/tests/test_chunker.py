"""Unit tests for text normalization and boundary-aware chunking."""

from __future__ import annotations

import pytest

from app.ingestion.chunker import chunk_text, normalize_text


def test_normalize_text_strips_control_chars_and_preserves_paragraphs() -> None:
    raw = "Line 1\x00\r\nLine 2\r\n\r\n\r\n\r\n# Section\r\nBody text   \n"
    normalized = normalize_text(raw)
    assert "\x00" not in normalized
    assert "\r" not in normalized
    assert normalized == "Line 1\nLine 2\n\n# Section\nBody text"


def test_chunk_text_preserves_headings_and_source_metadata() -> None:
    doc = (
        "# Architecture\n\n"
        "MindVault runs completely offline on loopback.\n\n"
        "# Security\n\n"
        "All inputs are treated as untrusted data and validated."
    )
    chunks = chunk_text(
        doc,
        chunk_size=120,
        chunk_overlap=20,
        source_metadata={"filename": "design.md", "file_type": "markdown"},
    )
    assert len(chunks) == 2
    assert chunks[0].chunk_index == 0
    assert chunks[0].metadata["section_heading"] == "Architecture"
    assert chunks[0].metadata["filename"] == "design.md"
    assert chunks[1].chunk_index == 1
    assert chunks[1].metadata["section_heading"] == "Security"
    assert "untrusted data" in chunks[1].content


def test_chunk_text_never_loses_extracted_text_on_oversized_blocks() -> None:
    words = [f"token_{i:03d}" for i in range(120)]
    long_block = " ".join(words)
    chunks = chunk_text(long_block, chunk_size=100, chunk_overlap=20)

    assert len(chunks) > 1
    combined = " ".join(c.content for c in chunks)
    for token in words:
        assert token in combined, f"Lost token '{token}' during chunking!"

    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert all(c.token_count > 0 for c in chunks)


def test_chunk_text_validates_configuration() -> None:
    with pytest.raises(ValueError, match="chunk_size"):
        chunk_text("hello", chunk_size=0, chunk_overlap=0)
    with pytest.raises(ValueError, match="chunk_overlap"):
        chunk_text("hello", chunk_size=100, chunk_overlap=100)
    with pytest.raises(ValueError, match="chunk_overlap"):
        chunk_text("hello", chunk_size=100, chunk_overlap=-5)
