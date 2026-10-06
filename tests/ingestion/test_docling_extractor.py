import os
import tempfile

import pymupdf
import pytest

from backend.app.ingestion.docling_extractor import extract_docling


def _make_text_pdf(page_texts: list[str]) -> str:
    doc = pymupdf.open()
    for text in page_texts:
        page = doc.new_page()
        page.insert_text((50, 72), text, fontsize=12)
    pdf_bytes = doc.tobytes()
    doc.close()

    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp.write(pdf_bytes)
    tmp.close()
    return tmp.name


def _make_image_pdf() -> str:
    doc = pymupdf.open()
    doc.new_page()
    pdf_bytes = doc.tobytes()
    doc.close()

    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp.write(pdf_bytes)
    tmp.close()
    return tmp.name


class TestDoclingExtractor:
    def test_extract_docling_standard(self):
        path = _make_text_pdf(
            [
                "Clause 1. This agreement is entered into by Acme Corp and Beta Ltd.",
                "Clause 2. Termination: Either party may terminate with 30 days notice.",
            ]
        )
        try:
            pages = extract_docling(path)
            assert len(pages) == 2
            assert pages[0]["page"] == 1
            assert pages[1]["page"] == 2
            assert "Acme Corp" in pages[0]["text"]
            assert "items" in pages[0]
            assert isinstance(pages[0]["items"], list)

            # Should tag as body/text
            assert any(item["heading_level"] == "body" for item in pages[0]["items"])
        finally:
            os.unlink(path)

    def test_extract_docling_emits_bbox_and_page_size(self):
        """Phase: UI citation preview. Every item carries a bbox (or None)
        and every page carries width/height so the overlay layer can scale."""
        path = _make_text_pdf(
            ["Section 1. Scope. This clause defines the scope of the contract."]
        )
        try:
            pages = extract_docling(path)
            assert len(pages) == 1
            page = pages[0]

            # Page-level: width/height present and positive
            assert "width" in page and "height" in page
            assert page["width"] is not None and page["width"] > 0
            assert page["height"] is not None and page["height"] > 0

            # Item-level: every item has a bbox key; at least one is a 4-tuple
            # in TOPLEFT origin (y0 < y1, x0 < x1), with coords inside the page
            assert all("bbox" in item for item in page["items"])
            bboxes = [item["bbox"] for item in page["items"] if item["bbox"] is not None]
            assert bboxes, "expected at least one item with a bbox"
            for bb in bboxes:
                assert len(bb) == 4
                x0, y0, x1, y1 = bb
                assert x0 < x1, f"expected x0<x1, got {bb}"
                assert y0 < y1, f"expected y0<y1 (TOPLEFT origin), got {bb}"
                assert 0 <= x0 and x1 <= page["width"] + 1  # +1 for float rounding
                assert 0 <= y0 and y1 <= page["height"] + 1
        finally:
            os.unlink(path)

    def test_extract_docling_scanned_raises_runtime_error(self):
        path = _make_image_pdf()
        try:
            with pytest.raises(RuntimeError, match="fallback to PyMuPDF"):
                extract_docling(path)
        finally:
            os.unlink(path)

    def test_extract_docling_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            extract_docling("non_existent_file.pdf")
