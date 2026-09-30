"""
Tests for Task 5.1 — claim_extractor.chunk_report().

PDF reading + spaCy are both mocked so the suite needs no real files.
"""

import os
import tempfile
from unittest.mock import patch, MagicMock

import pymupdf
import pytest

from backend.app.shield.claim_extractor import chunk_report, _split_sentences


# ─── helpers ──────────────────────────────────────────────────────────────────


def _make_pdf(text: str) -> str:
    """Create a real PDF with given text for end-to-end testing."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 72), text, fontsize=11)
    pdf_bytes = doc.tobytes()
    doc.close()
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp.write(pdf_bytes)
    tmp.close()
    return tmp.name


# ─── _split_sentences fallback ───────────────────────────────────────────────


class TestSplitSentencesFallback:
    def test_basic_split(self):
        text = "The penalty is 2%. The contract expires in 2027."
        sentences = _split_sentences(text)
        assert len(sentences) >= 2

    def test_single_sentence_returns_one(self):
        result = _split_sentences("Only one claim here.")
        assert len(result) == 1

    def test_empty_string_returns_empty(self):
        result = _split_sentences("")
        assert result == []


# ─── chunk_report — numbered bullets ─────────────────────────────────────────


class TestNumberedBullets:
    def setup_method(self):
        text = (
            "Compliance Report 2026\n\n"
            "1. The company has filed all required returns.\n"
            "2. The borrowing limit has not been exceeded.\n"
            "3. All directors have submitted their declarations."
        )
        self.path = _make_pdf(text)

    def teardown_method(self):
        os.unlink(self.path)

    def test_returns_list(self):
        result = chunk_report(self.path)
        assert isinstance(result, list)

    def test_multiple_claims_extracted(self):
        result = chunk_report(self.path)
        assert len(result) >= 2

    def test_claims_are_strings(self):
        result = chunk_report(self.path)
        for claim in result:
            assert isinstance(claim, str)

    def test_minimum_claim_length(self):
        """Noise (single words, page numbers) should be filtered."""
        result = chunk_report(self.path)
        for claim in result:
            assert len(claim) >= 10

    def test_no_duplicate_claims(self):
        result = chunk_report(self.path)
        assert len(result) == len(set(result))


# ─── chunk_report — prose paragraphs ─────────────────────────────────────────


class TestProseParagraphs:
    def setup_method(self):
        text = (
            "The company was incorporated in 2020 under the Companies Act. "
            "It has maintained proper books of account throughout the year. "
            "All statutory filings are up to date.\n\n"
            "The board met four times during the year. "
            "All resolutions were passed with requisite majority."
        )
        self.path = _make_pdf(text)

    def teardown_method(self):
        os.unlink(self.path)

    def test_sentences_extracted(self):
        result = chunk_report(self.path)
        assert len(result) >= 3

    def test_claims_are_complete_sentences(self):
        result = chunk_report(self.path)
        for claim in result:
            # Each claim should have some meaningful content
            assert len(claim.split()) >= 3


# ─── error handling ──────────────────────────────────────────────────────────


class TestErrors:
    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="Report PDF not found"):
            chunk_report("/nonexistent/report.pdf")

    def test_invalid_pdf(self, tmp_path):
        bad = tmp_path / "bad.pdf"
        bad.write_bytes(b"not a pdf")
        with pytest.raises(ValueError, match="Cannot read PDF"):
            chunk_report(str(bad))
