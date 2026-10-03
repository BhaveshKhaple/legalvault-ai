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
