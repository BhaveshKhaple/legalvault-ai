"""
eval-corpus — Plain-text extractor.

Mirrors the pdf_extractor interface so the ingestion dispatcher can swap
extractors by extension without routing-specific logic in the upload path.

A .txt file has no concept of pages. We split on form-feed (`\x0c`) when
present (some tools emit it between logical pages); otherwise the whole
file is a single "page" with page=1. The clause_chunker runs on the text
regardless, so retrieval still works — only the citation granularity is
cruder (file-level, not page-level).
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# When no form-feed is present, treat the whole file as one page. If the file
# is enormous (> this many chars), split into virtual pages so chunking has a
# reasonable page tag to attach.
_VIRTUAL_PAGE_CHARS = 3000


def extract_text(path: str) -> list[dict]:
    """Extract text from a .txt file.

    Returns the same shape as `extract_pdf` so downstream code is identical:

        [
            {"text": str, "page": int, "doc_id": str},
            ...
        ]
    """
    txt_path = Path(path)
    if not txt_path.exists():
        raise FileNotFoundError(f"Text file not found: {txt_path}")

    doc_id = txt_path.stem

    try:
        raw = txt_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Fall back to latin-1 — never crashes, but may mangle non-ASCII
        logger.warning("Falling back to latin-1 decode for '%s'", txt_path.name)
        raw = txt_path.read_text(encoding="latin-1")

    # Prefer form-feed splits if the source provided them
    if "\x0c" in raw:
        pages_text = raw.split("\x0c")
    elif len(raw) > _VIRTUAL_PAGE_CHARS:
        # Virtual-pagination: split on paragraph boundaries near 3000 chars
        pages_text = _virtual_paginate(raw, _VIRTUAL_PAGE_CHARS)
    else:
        pages_text = [raw]

    return [
        {"text": chunk, "page": idx + 1, "doc_id": doc_id}
        for idx, chunk in enumerate(pages_text)
        if chunk.strip()
    ] or [{"text": "", "page": 1, "doc_id": doc_id}]


def _virtual_paginate(text: str, target: int) -> list[str]:
    """Break text into virtual pages near paragraph boundaries, ~target chars each."""
    paragraphs = text.split("\n\n")
    pages: list[str] = []
    current: list[str] = []
    current_len = 0
    for para in paragraphs:
        if current_len + len(para) > target and current:
            pages.append("\n\n".join(current))
            current = [para]
            current_len = len(para)
        else:
            current.append(para)
            current_len += len(para) + 2  # +2 for the "\n\n" separator
    if current:
        pages.append("\n\n".join(current))
    return pages
