"""
eval-corpus — DOCX extractor.

Uses python-docx to pull body text + tables from a Word document.

DOCX has no native page concept at the text level (pagination happens at
render time), so we use an approximation: Heading 1 / Heading 2 paragraphs
become page breaks, otherwise ~3000-char groups form a virtual page. The
clause_chunker still runs on each page's text, so retrieval quality isn't
affected — only citation granularity.

Tables are flattened into pipe-separated rows and appended to the page
they appear on. Images and embedded objects are ignored (OCR scoped to a
later milestone).
"""

import logging
from pathlib import Path

from docx import Document

logger = logging.getLogger(__name__)

_HEADING_STYLES = {"Heading 1", "Heading 2", "Title"}
_VIRTUAL_PAGE_CHARS = 3000


def extract_docx(path: str) -> list[dict]:
    """Extract text + tables from a .docx file.

    Returns the same shape as `extract_pdf`:

        [
            {"text": str, "page": int, "doc_id": str},
            ...
        ]
    """
    docx_path = Path(path)
    if not docx_path.exists():
        raise FileNotFoundError(f"DOCX file not found: {docx_path}")

    doc_id = docx_path.stem

    try:
        doc = Document(str(docx_path))
    except Exception as exc:
        raise ValueError(f"Cannot open DOCX '{docx_path.name}': {exc}") from exc

    pages: list[list[str]] = [[]]
    current_len = 0

    # Paragraphs: break on top-level headings or at ~3000-char accumulation
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue

        is_heading = para.style and para.style.name in _HEADING_STYLES
        if is_heading and pages[-1]:
            pages.append([])
            current_len = 0

        pages[-1].append(text)
        current_len += len(text) + 1

        if current_len > _VIRTUAL_PAGE_CHARS:
            pages.append([])
            current_len = 0

    # Append tables as pipe-separated rows at the end of the last populated page
    for table in doc.tables:
        rows = []
        for row in table.rows:
            cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
            rows.append(" | ".join(cells))
        if rows:
            if not pages[-1]:
                pages[-1] = rows
            else:
                pages[-1].extend([""] + rows)

    results = []
    for idx, page_lines in enumerate(pages, start=1):
        text = "\n".join(page_lines).strip()
        if text:
            results.append({"text": text, "page": idx, "doc_id": doc_id})

    if not results:
        logger.warning("DOCX '%s' produced no extractable text", docx_path.name)
        results = [{"text": "", "page": 1, "doc_id": doc_id}]

    return results
