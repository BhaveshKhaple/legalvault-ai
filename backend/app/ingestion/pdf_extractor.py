"""
Task 1.1 — PDF extractor.

Extracts text page-by-page from a PDF, preserving page provenance so the
AI can later cite "see page 5 of Contract.pdf".

Scanned/image PDFs (where OCR has not been run) are detected by very low
text yield and warned — they are NOT crashed on. OCR is scoped to a later
milestone.
"""

import logging
from pathlib import Path

import pymupdf  # formerly fitz — new API since pymupdf 1.24+

logger = logging.getLogger(__name__)

# A page whose text length is below this fraction of its pixel area is
# considered image-only. Rough heuristic; tighten if needed.
_SCANNED_TEXT_THRESHOLD = 2  # characters — anything below this is "blank"


def extract_pdf(path: str) -> list[dict]:
    """Extract text page-by-page from a PDF file.

    Args:
        path: Absolute or relative path to the PDF.

    Returns:
        List of dicts, one per page::

            [
                {
                    "text":   str,   # raw page text (may be empty for image pages)
                    "page":   int,   # 1-indexed page number
                    "doc_id": str,   # derived from filename stem
                },
                ...
            ]

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be opened as a PDF.
    """
    pdf_path = Path(path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc_id = pdf_path.stem
    results: list[dict] = []

    try:
        doc = pymupdf.open(str(pdf_path))
    except Exception as exc:
        raise ValueError(f"Cannot open PDF '{pdf_path.name}': {exc}") from exc

    try:
        for page_index in range(len(doc)):
            page = doc[page_index]
            page_number = page_index + 1  # human-readable 1-indexed
            text = page.get_text()

            if len(text.strip()) < _SCANNED_TEXT_THRESHOLD:
                logger.warning(
                    "Page %d of '%s' appears to be a scanned/image page — "
                    "no extractable text. Returning empty text for this page. "
                    "OCR support is planned for a later milestone.",
                    page_number,
                    pdf_path.name,
                )

            results.append(
                {
                    "text": text,
                    "page": page_number,
                    "doc_id": doc_id,
                }
            )
    finally:
        doc.close()

    return results
