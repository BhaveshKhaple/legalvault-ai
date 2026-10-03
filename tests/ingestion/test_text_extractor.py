"""Tests for the .txt extractor (eval-corpus branch)."""

from pathlib import Path

import pytest

from backend.app.ingestion.text_extractor import extract_text


def test_small_file_is_one_page(tmp_path: Path):
    p = tmp_path / "note.txt"
    p.write_text("Hello world.\n\nSecond paragraph.", encoding="utf-8")

    pages = extract_text(str(p))
    assert len(pages) == 1
    assert pages[0]["page"] == 1
    assert pages[0]["doc_id"] == "note"
    assert "Second paragraph" in pages[0]["text"]


def test_form_feed_splits_pages(tmp_path: Path):
    p = tmp_path / "formfeed.txt"
    p.write_text("Page one body.\x0cPage two body.\x0cPage three.", encoding="utf-8")

    pages = extract_text(str(p))
    assert [page["page"] for page in pages] == [1, 2, 3]
    assert "Page two" in pages[1]["text"]


def test_large_file_virtual_pagination(tmp_path: Path):
    p = tmp_path / "long.txt"
    para = ("The quick brown fox jumps over the lazy dog. " * 20).strip()
    p.write_text("\n\n".join([para] * 20), encoding="utf-8")

    pages = extract_text(str(p))
    # Should split into multiple virtual pages (>3000 chars triggers splitting)
    assert len(pages) >= 2
    # Every page has a paragraph boundary — no page starts mid-sentence
    for pg in pages:
        assert pg["text"].strip()


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        extract_text("/nonexistent/path.txt")


def test_latin1_fallback(tmp_path: Path):
    p = tmp_path / "latin.txt"
    # Byte 0xe9 = "é" in latin-1, invalid UTF-8 start byte
    p.write_bytes(b"Caf\xe9 au lait")

    pages = extract_text(str(p))
    assert len(pages) == 1
    assert "Caf" in pages[0]["text"]
