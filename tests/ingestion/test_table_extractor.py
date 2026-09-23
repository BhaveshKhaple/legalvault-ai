"""
Tests for Task 1.4 — table_extractor.extract_tables().

Creates test PDFs in-memory using reportlab (already a transitive dep of
pdfplumber via pypdfium2 stack). Uses pymupdf to draw a table-shaped PDF
with drawn lines, which pdfplumber then recognises as a table.
"""

import os
import tempfile

import pymupdf
import pytest

from backend.app.ingestion.table_extractor import extract_tables


# ─── helpers ──────────────────────────────────────────────────────────────────


def _make_pdf_with_table(rows: list[list[str]]) -> str:
    """Draw a bordered table on one PDF page so pdfplumber can detect it.

    We draw explicit rectangles + text so pdfplumber's geometric detection
    finds real cells (rather than guessing from whitespace alignment).
    """
    doc = pymupdf.open()
    page = doc.new_page()

    n_rows = len(rows)
    n_cols = max(len(r) for r in rows)
    cell_w = 100
    cell_h = 30
    x0, y0 = 50, 50

    for r_idx, row in enumerate(rows):
        for c_idx in range(n_cols):
            x = x0 + c_idx * cell_w
            y = y0 + r_idx * cell_h
            rect = pymupdf.Rect(x, y, x + cell_w, y + cell_h)
            page.draw_rect(rect, color=(0, 0, 0), width=1)
            if c_idx < len(row):
                page.insert_text((x + 5, y + 20), row[c_idx], fontsize=10)

    pdf_bytes = doc.tobytes()
    doc.close()

    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp.write(pdf_bytes)
    tmp.close()
    return tmp.name


def _make_pdf_no_tables() -> str:
    """PDF with only prose — no tables, no drawn rectangles."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 72), "This is plain contract prose with no tabular data.", fontsize=12)
    pdf_bytes = doc.tobytes()
    doc.close()

    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp.write(pdf_bytes)
    tmp.close()
    return tmp.name


# ─── happy path: fee schedule table ──────────────────────────────────────────


class TestFeeScheduleTable:
    def setup_method(self):
        self.rows = [
            ["Service", "Fee", "Currency"],
            ["Setup", "5000", "INR"],
            ["Monthly", "1500", "INR"],
            ["Overage", "100", "INR"],
        ]
        self.path = _make_pdf_with_table(self.rows)

    def teardown_method(self):
        os.unlink(self.path)

    def test_at_least_one_table_extracted(self):
        result = extract_tables(self.path)
        assert len(result) >= 1

    def test_table_is_2d_grid(self):
        result = extract_tables(self.path)
        table = result[0]
        assert isinstance(table, list)
        for row in table:
            assert isinstance(row, list)
            for cell in row:
                assert isinstance(cell, str)

    def test_header_row_captured(self):
        result = extract_tables(self.path)
        table = result[0]
        header = table[0]
        assert "Service" in header
        assert "Fee" in header
        assert "Currency" in header

    def test_data_rows_captured(self):
        result = extract_tables(self.path)
        table = result[0]
        # flatten and verify our data cells show up somewhere
        flat = [cell for row in table for cell in row]
        assert "Setup" in flat
        assert "5000" in flat
        assert "Monthly" in flat

    def test_no_none_cells(self):
        """Empty cells should be '' not None so downstream code is None-safe."""
        result = extract_tables(self.path)
        for table in result:
            for row in table:
                for cell in row:
                    assert cell is not None


# ─── negative: no tables in PDF ──────────────────────────────────────────────


class TestNoTables:
    def setup_method(self):
        self.path = _make_pdf_no_tables()

    def teardown_method(self):
        os.unlink(self.path)

    def test_returns_empty_list(self):
        result = extract_tables(self.path)
        assert result == []

    def test_does_not_crash(self):
        # If this returns anything at all (even []), it hasn't raised
        extract_tables(self.path)


# ─── error handling ──────────────────────────────────────────────────────────


class TestErrors:
    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="PDF not found"):
            extract_tables("/nonexistent/table.pdf")

    def test_invalid_pdf(self, tmp_path):
        bad = tmp_path / "bad.pdf"
        bad.write_bytes(b"not a pdf")
        with pytest.raises(ValueError, match="Cannot open PDF"):
            extract_tables(str(bad))
