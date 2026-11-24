"""Unit tests for all local document extraction adapters and graceful degradation."""

from __future__ import annotations

import types
from pathlib import Path

import fitz
import pytest
from PIL import Image

from app.ingestion.adapters import (
    AdapterRegistry,
    AudioWhisperAdapter,
    ImageOcrAdapter,
    PdfAdapter,
    TextMarkdownAdapter,
)
from app.ingestion.errors import (
    AdapterUnavailableError,
    CorruptedFileError,
    UnsupportedFileTypeError,
)


def _create_sample_pdf(path: Path, pages: list[str]) -> Path:
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()
    return path


def _create_sample_png(path: Path) -> Path:
    img = Image.new("RGB", (64, 32), color=(255, 255, 255))
    img.save(str(path), format="PNG")
    return path


def test_text_and_markdown_adapter_extracts_and_handles_bad_utf8(tmp_path: Path) -> None:
    adapter = TextMarkdownAdapter()

    md_file = tmp_path / "notes.md"
    md_file.write_text("# Heading\n\nOffline knowledge.", encoding="utf-8")
    md_res = adapter.extract(md_file)
    assert md_res.file_type == "markdown"
    assert "# Heading" in md_res.text
    assert md_res.warnings == []

    txt_file = tmp_path / "latin.txt"
    txt_file.write_bytes(b"Hello \xff world")
    txt_res = adapter.extract(txt_file)
    assert txt_res.file_type == "txt"
    assert "Hello" in txt_res.text
    assert len(txt_res.warnings) == 1


def test_pdf_adapter_extracts_real_pdf_and_rejects_corrupt(tmp_path: Path) -> None:
    adapter = PdfAdapter()
    pdf_path = _create_sample_pdf(
        tmp_path / "sample.pdf",
        ["First page of MindVault.", "Second page content."],
    )
    res = adapter.extract(pdf_path)
    assert res.file_type == "pdf"
    assert res.metadata["page_count"] == 2
    assert "First page of MindVault." in res.text
    assert "Second page content." in res.text

    corrupt_pdf = tmp_path / "broken.pdf"
    corrupt_pdf.write_bytes(b"%PDF-1.4 not really a valid pdf stream")
    with pytest.raises(CorruptedFileError):
        adapter.extract(corrupt_pdf)


def test_pdf_adapter_degrades_when_pymupdf_unavailable(tmp_path: Path) -> None:
    def _missing_loader():
        raise AdapterUnavailableError("PyMuPDF missing")

    adapter = PdfAdapter(loader=_missing_loader)
    dummy = tmp_path / "a.pdf"
    dummy.write_bytes(b"%PDF")
    with pytest.raises(AdapterUnavailableError):
        adapter.extract(dummy)


def test_image_ocr_adapter_with_mocked_pytesseract_and_degradation(tmp_path: Path) -> None:
    png_path = _create_sample_png(tmp_path / "scan.png")

    # Success path with mocked pytesseract
    fake_tesseract = types.SimpleNamespace(
        pytesseract=types.SimpleNamespace(tesseract_cmd=""),
        image_to_string=lambda img: "Extracted OCR text from image",
    )
    adapter = ImageOcrAdapter(
        tesseract_cmd="/local/bin/tesseract",
        pytesseract_loader=lambda: fake_tesseract,
    )
    res = adapter.extract(png_path)
    assert res.file_type == "image"
    assert res.text == "Extracted OCR text from image"
    assert res.metadata["width"] == 64
    assert res.metadata["height"] == 32
    assert fake_tesseract.pytesseract.tesseract_cmd == "/local/bin/tesseract"

    # Corrupt image raises CorruptedFileError
    bad_img = tmp_path / "corrupt.png"
    bad_img.write_bytes(b"not an image")
    with pytest.raises(CorruptedFileError):
        adapter.extract(bad_img)

    # Missing tesseract binary degrades gracefully to AdapterUnavailableError
    def _raise_not_found(_img):
        raise FileNotFoundError("tesseract not found")

    missing_bin_adapter = ImageOcrAdapter(
        pytesseract_loader=lambda: types.SimpleNamespace(image_to_string=_raise_not_found)
    )
    with pytest.raises(AdapterUnavailableError, match="Tesseract"):
        missing_bin_adapter.extract(png_path)


def test_audio_whisper_adapter_offline_only_and_degradation(tmp_path: Path) -> None:
    audio_path = tmp_path / "memo.wav"
    audio_path.write_bytes(b"RIFF....WAVEfmt ")
    model_dir = tmp_path / "whisper-tiny-local"
    model_dir.mkdir()

    # Without a configured local model path, must refuse network downloads.
    captured_kwargs: dict[str, object] = {}

    class _FakeFasterWhisperModel:
        def __init__(self, model_path: str, **kwargs: object) -> None:
            captured_kwargs["model_path"] = model_path
            captured_kwargs.update(kwargs)

        def transcribe(self, _audio_path: str):
            segments = [types.SimpleNamespace(text="Offline voice note.")]
            info = types.SimpleNamespace(language="en", duration=2.5)
            return segments, info

    fake_fw = types.SimpleNamespace(WhisperModel=_FakeFasterWhisperModel)

    no_model_adapter = AudioWhisperAdapter(
        model_path="",
        backend_loader=lambda: ("faster-whisper", fake_fw),
    )
    with pytest.raises(AdapterUnavailableError, match="MINDVAULT_WHISPER_MODEL_PATH"):
        no_model_adapter.extract(audio_path)

    # With local model path, transcribes offline with local_files_only=True.
    ok_adapter = AudioWhisperAdapter(
        model_path=str(model_dir),
        backend_loader=lambda: ("faster-whisper", fake_fw),
    )
    res = ok_adapter.extract(audio_path)
    assert res.file_type == "audio"
    assert res.text == "Offline voice note."
    assert res.metadata["language"] == "en"
    assert captured_kwargs["local_files_only"] is True

    # Default adapter with no installed whisper package degrades gracefully.
    real_missing = AudioWhisperAdapter(model_path=str(model_dir))
    with pytest.raises(AdapterUnavailableError):
        real_missing.extract(audio_path)


def test_adapter_registry_dispatches_and_rejects_unsupported() -> None:
    registry = AdapterRegistry.default()
    assert isinstance(registry.get_adapter("doc.pdf"), PdfAdapter)
    assert isinstance(registry.get_adapter("notes.md"), TextMarkdownAdapter)
    assert isinstance(registry.get_adapter("plain.txt"), TextMarkdownAdapter)
    assert isinstance(registry.get_adapter("photo.png"), ImageOcrAdapter)
    assert isinstance(registry.get_adapter("voice.wav"), AudioWhisperAdapter)

    with pytest.raises(UnsupportedFileTypeError):
        registry.get_adapter("malware.exe")
