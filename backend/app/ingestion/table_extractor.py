"""
Task 1.4 — Table extractor via pdfplumber.

Pulls tables out of PDFs as clean 2D grids of cells. Body prose extraction
is Task 1.1's job; this module ONLY handles tables so they don't get mangled
into paragraph text.

pdfplumber uses geometric analysis (line positions, cell edges) to detect
tables. It's not perfect on tables drawn without borders — those return
partial results, not exceptions.
"""

import logging
from pathlib import Path

import pdfplumber

logger = logging.getLogger(__name__)


def extract_tables(pdf_path: str) -> list[list[list[str]]]:
    """Extract all tables from a PDF.

    Args:
        pdf_path: Absolute or relative path to the PDF.

    Returns:
        List of tables. Each table is a 2D grid — a list of rows, where
        each row is a list of cell strings. Empty cells are empty strings
        (not None) so downstream code doesn't need None-handling.

        Empty list if no tables are detected.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be opened as a PDF.
    """
    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    tables: list[list[list[str]]] = []

    try:
        pdf = pdfplumber.open(str(path))
    except Exception as exc:
        raise ValueError(f"Cannot open PDF '{path.name}': {exc}") from exc

    try:
        for page_index, page in enumerate(pdf.pages):
            raw_tables = page.extract_tables() or []
            for raw in raw_tables:
                cleaned = [
                    [(cell if cell is not None else "").strip() for cell in row]
                    for row in raw
                ]
                if any(any(c for c in row) for row in cleaned):
                    tables.append(cleaned)

            logger.debug(
                "Page %d of '%s': extracted %d table(s).",
                page_index + 1,
                path.name,
                len(raw_tables),
            )
    finally:
        pdf.close()

    logger.info(
        "Extracted %d table(s) total from '%s'.",
        len(tables),
        path.name,
    )
    return tables
