"""
Tests for Task 1.1 — pdf_extractor.extract_pdf().

We create test PDFs in-memory using pymupdf so the suite needs no
pre-existing files on disk. Three scenarios required by the tracker:

    (a) 2-page text contract
    (b) 20-page contract
    (c) Scanned/image page (no extractable text)
"""

import logging
import os
import tempfile

import pymupdf
import pytest

from backend.app.ingestion.pdf_extractor import extract_pdf


# ─── helpers ──────────────────────────────────────────────────────────────────


def _make_text_pdf(page_texts: list[str]) -> str:
    """Create a temporary PDF with given per-page text content. Returns path.

    We serialise to bytes first then write to a closed temp file so Windows
    doesn't raise PermissionError (open file handles block pymupdf saves).
    """
    doc = pymupdf.open()
    for text in page_texts:
        page = doc.new_page()
        page.insert_text((50, 72), text, fontsize=12)
    pdf_bytes = doc.tobytes()
    doc.close()

    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp.write(pdf_bytes)
    tmp.close()  # close before returning — Windows needs the handle free
    return tmp.name


def _make_image_pdf() -> str:
    """Create a PDF whose single page has no text — simulates a scanned doc."""
    doc = pymupdf.open()
    doc.new_page()  # blank page, no text inserted
    pdf_bytes = doc.tobytes()
    doc.close()

    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp.write(pdf_bytes)
    tmp.close()
    return tmp.name


# ─── (a) 2-page contract ──────────────────────────────────────────────────────


class TestShortContract:
    def setup_method(self):
        self.path = _make_text_pdf(
            [
                "Clause 1. This agreement is entered into by Acme Corp and Beta Ltd.",
                "Clause 2. Termination: Either party may terminate with 30 days notice.",
            ]
        )

    def teardown_method(self):
        os.unlink(self.path)

    def test_returns_two_pages(self):
        result = extract_pdf(self.path)
        assert len(result) == 2

    def test_page_numbers_are_one_indexed(self):
        result = extract_pdf(self.path)
        assert result[0]["page"] == 1
        assert result[1]["page"] == 2

    def test_text_preserved(self):
        result = extract_pdf(self.path)
        assert "Acme Corp" in result[0]["text"]
        assert "Termination" in result[1]["text"]

    def test_doc_id_is_filename_stem(self):
        from pathlib import Path
        result = extract_pdf(self.path)
        expected_stem = Path(self.path).stem
        assert all(r["doc_id"] == expected_stem for r in result)

    def test_dict_keys_present(self):
        result = extract_pdf(self.path)
        for row in result:
            assert set(row.keys()) == {"text", "page", "doc_id"}


# ─── (b) 20+ page contract ────────────────────────────────────────────────────


class TestLongContract:
    PAGE_COUNT = 22

    def setup_method(self):
        texts = [f"Section {i + 1}. Content of clause {i + 1}." for i in range(self.PAGE_COUNT)]
        self.path = _make_text_pdf(texts)

    def teardown_method(self):
        os.unlink(self.path)

    def test_correct_page_count(self):
        result = extract_pdf(self.path)
        assert len(result) == self.PAGE_COUNT

    def test_pages_are_sequential(self):
        result = extract_pdf(self.path)
        pages = [r["page"] for r in result]
        assert pages == list(range(1, self.PAGE_COUNT + 1))

    def test_last_page_text_correct(self):
        result = extract_pdf(self.path)
        assert f"Section {self.PAGE_COUNT}" in result[-1]["text"]

    def test_no_page_shares_doc_id(self):
        """All rows share the same doc_id derived from the filename."""
        result = extract_pdf(self.path)
        doc_ids = {r["doc_id"] for r in result}
        assert len(doc_ids) == 1


# ─── (c) Scanned / image PDF ──────────────────────────────────────────────────


class TestScannedPdf:
    def setup_method(self):
        self.path = _make_image_pdf()

    def teardown_method(self):
        os.unlink(self.path)

    def test_does_not_crash(self):
        """Must return normally — not raise an exception."""
        result = extract_pdf(self.path)
        assert isinstance(result, list)

    def test_returns_one_page_entry(self):
        result = extract_pdf(self.path)
        assert len(result) == 1

    def test_empty_text_returned(self):
        result = extract_pdf(self.path)
        assert result[0]["text"].strip() == ""

    def test_warning_logged(self, caplog):
        with caplog.at_level(logging.WARNING, logger="backend.app.ingestion.pdf_extractor"):
            extract_pdf(self.path)
        assert any("scanned" in record.message.lower() for record in caplog.records)


# ─── error handling ───────────────────────────────────────────────────────────


class TestErrors:
    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="PDF not found"):
            extract_pdf("/nonexistent/path/file.pdf")

    def test_invalid_pdf(self, tmp_path):
        bad = tmp_path / "bad.pdf"
        bad.write_bytes(b"this is not a pdf")
        with pytest.raises(ValueError, match="Cannot open PDF"):
            extract_pdf(str(bad))
