"""Text normalization and boundary-aware chunking for local documents."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class TextChunk:
    """Single normalized chunk with sequential index, token estimate, and metadata."""

    chunk_index: int
    content: str
    token_count: int
    metadata: dict[str, Any] = field(default_factory=dict)


def normalize_text(raw: str) -> str:
    """Sanitize and normalize extracted untrusted text while preserving structure."""
    if not raw:
        return ""
    cleaned = raw.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = _CONTROL_CHAR_RE.sub("", cleaned)
    lines = [line.rstrip() for line in cleaned.split("\n")]
    normalized = "\n".join(lines)
    normalized = _MULTI_NEWLINE_RE.sub("\n\n", normalized)
    return normalized.strip()


def estimate_token_count(text: str) -> int:
    """Deterministic local token count estimate (no external tokenizer required)."""
    if not text or not text.strip():
        return 0
    return len(text.split())


def _split_oversized_block(block: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split a block larger than chunk_size without losing any text."""
    if len(block) <= chunk_size:
        return [block]

    sentences = _SENTENCE_SPLIT_RE.split(block)
    pieces: list[str] = []
    current = ""

    def _hard_slice(segment: str) -> list[str]:
        step = max(1, chunk_size - chunk_overlap)
        slices: list[str] = []
        start = 0
        while start < len(segment):
            end = min(len(segment), start + chunk_size)
            if end < len(segment):
                ws = segment.rfind(" ", start + 1, end)
                if ws > start + (chunk_size // 3):
                    end = ws
            piece = segment[start:end].strip()
            if piece:
                slices.append(piece)
            if end >= len(segment):
                break
            next_start = max(end - chunk_overlap, start + 1)
            if next_start <= start:
                next_start = start + step
            start = next_start
        return slices

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > chunk_size:
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(_hard_slice(sentence))
            continue

        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                pieces.append(current)
            if chunk_overlap > 0 and current:
                tail = current[-chunk_overlap:].lstrip()
                overlapped = f"{tail} {sentence}".strip()
                current = overlapped if len(overlapped) <= chunk_size else sentence
            else:
                current = sentence

    if current:
        pieces.append(current)
    return pieces


def _extract_blocks(text: str) -> list[tuple[Optional[str], str, bool]]:
    """Split normalized text into (heading, block_text, is_heading) tuples."""
    blocks: list[tuple[Optional[str], str, bool]] = []
    current_heading: Optional[str] = None

    for raw_para in text.split("\n\n"):
        para = raw_para.strip()
        if not para:
            continue

        # Handle heading lines directly followed by text without a blank line.
        lines = para.split("\n")
        first_match = _HEADING_RE.match(lines[0])
        if first_match:
            current_heading = first_match.group(2).strip()
            if len(lines) == 1:
                blocks.append((current_heading, para, True))
            else:
                blocks.append((current_heading, lines[0].strip(), True))
                rest = "\n".join(lines[1:]).strip()
                if rest:
                    blocks.append((current_heading, rest, False))
        else:
            blocks.append((current_heading, para, False))

    return blocks


def chunk_text(
    text: str,
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    source_metadata: Optional[dict[str, Any]] = None,
) -> list[TextChunk]:
    """Split normalized text into boundary-preserving chunks without losing content."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and strictly less than chunk_size.")

    normalized = normalize_text(text)
    if not normalized:
        return []

    base_meta = dict(source_metadata or {})
    blocks = _extract_blocks(normalized)

    raw_chunks: list[tuple[str, Optional[str]]] = []
    buffer_parts: list[str] = []
    buffer_len = 0
    buffer_heading: Optional[str] = None

    def _flush() -> str:
        nonlocal buffer_parts, buffer_len, buffer_heading
        if not buffer_parts:
            return ""
        content = "\n\n".join(buffer_parts).strip()
        if content:
            raw_chunks.append((content, buffer_heading))
        buffer_parts = []
        buffer_len = 0
        return content

    for heading, block, is_heading in blocks:
        # Start a new chunk cleanly at a new heading if we already have content.
        if is_heading and buffer_parts:
            _flush()
            buffer_heading = heading

        if len(block) > chunk_size:
            prev_content = _flush()
            sub_pieces = _split_oversized_block(block, chunk_size, chunk_overlap)
            for idx, piece in enumerate(sub_pieces):
                if idx == 0 and chunk_overlap > 0 and prev_content and not is_heading:
                    tail = prev_content[-chunk_overlap:].lstrip()
                    combined = f"{tail}\n\n{piece}".strip()
                    if len(combined) <= chunk_size:
                        piece = combined
                raw_chunks.append((piece, heading))
            buffer_heading = heading
            continue

        sep_len = 2 if buffer_parts else 0
        if buffer_len + sep_len + len(block) <= chunk_size:
            if not buffer_parts:
                buffer_heading = heading
            buffer_parts.append(block)
            buffer_len += sep_len + len(block)
        else:
            prev_content = _flush()
            buffer_heading = heading
            if chunk_overlap > 0 and prev_content and not is_heading:
                tail = prev_content[-chunk_overlap:].lstrip()
                if tail and len(tail) + 2 + len(block) <= chunk_size:
                    buffer_parts = [tail, block]
                    buffer_len = len(tail) + 2 + len(block)
                    continue
            buffer_parts = [block]
            buffer_len = len(block)

    _flush()

    results: list[TextChunk] = []
    for idx, (content, heading) in enumerate(raw_chunks):
        chunk_meta = dict(base_meta)
        chunk_meta["chunk_index"] = idx
        chunk_meta["char_length"] = len(content)
        if heading:
            chunk_meta["section_heading"] = heading
        results.append(
            TextChunk(
                chunk_index=idx,
                content=content,
                token_count=estimate_token_count(content),
                metadata=chunk_meta,
            )
        )
    return results
