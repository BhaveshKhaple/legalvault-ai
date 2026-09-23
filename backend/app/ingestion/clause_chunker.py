"""
Task 1.2 — Clause-aware chunker for legal documents.

Splits per-page text (output of pdf_extractor.extract_pdf) into legally
meaningful CHUNKS. A chunk = one clause / section, regardless of how the
underlying PDF wraps text across pages.

Why not fixed-size chunking:
    "Clause 3.1 Termination: This contract may be..." would get cut in
    the middle by naive 500-char splits, destroying the semantic unit
    that RAG needs to retrieve.

Boundary rules (in priority order):
    1. Numbered clauses: '3', '3.1', '3.1.2', '3.1.2.4' followed by space.
    2. Section/Article/Chapter keywords: 'Section 4', 'Article IX'.
    3. All-caps single-line headings (max 8 words): 'DEFINITIONS'.

If a numbered clause spans multiple PDF pages, it is kept as ONE chunk.
Page number of the chunk = first page it starts on.
"""

import re
from typing import Iterable

# ─── boundary patterns ────────────────────────────────────────────────────────

# Numbered clause: 1.  1.1  1.1.2.4  (nested form OR mandatory trailing dot)
# Anchored at start-of-line so we don't match "see clause 3.1" mid-sentence.
# A bare digit like "30 days" is NOT a clause — prose numbers must be followed
# by lowercase and lack the trailing dot / nested form.
_NUMBERED_CLAUSE = re.compile(
    r"^(?P<num>\d+\.\d+(?:\.\d+)*\.?|\d+\.)\s+(?P<rest>.+)$"
)

# 'Section 4', 'Article IX', 'Chapter 2', 'Part III' — case-insensitive keyword,
# followed by a number or Roman numeral.
_NAMED_SECTION = re.compile(
    r"^(?P<kw>Section|Article|Chapter|Part)\s+"
    r"(?P<num>[IVXLCDM]+|\d+(?:\.\d+)*)"
    r"\b(?P<rest>.*)$",
    re.IGNORECASE,
)

# All-caps heading: line is entirely uppercase letters/spaces/hyphens,
# 1–8 words long. Filters section-heading style like "DEFINITIONS" or
# "TERMINATION AND SURVIVAL".
_ALLCAPS_HEADING = re.compile(r"^[A-Z][A-Z\s\-&]{2,}$")


def _is_boundary(line: str) -> tuple[str | None, str | None]:
    """Return (heading_number, heading_title) if line starts a new chunk, else (None, None)."""
    stripped = line.strip()
    if not stripped:
        return None, None

    m = _NUMBERED_CLAUSE.match(stripped)
    if m:
        return m.group("num"), stripped

    m = _NAMED_SECTION.match(stripped)
    if m:
        return f"{m.group('kw')} {m.group('num')}", stripped

    if _ALLCAPS_HEADING.match(stripped) and 1 <= len(stripped.split()) <= 8:
        return stripped, stripped

    return None, None


# ─── main API ─────────────────────────────────────────────────────────────────


def chunk_legal_doc(pages: Iterable[dict]) -> list[dict]:
    """Split pages into clause-aligned chunks.

    Args:
        pages: Iterable of page dicts from pdf_extractor.extract_pdf().
               Each must have keys: text (str), page (int), doc_id (str).

    Returns:
        List of chunk dicts::

            [
                {
                    "content":        str,   # full clause text (may span pages)
                    "page":           int,   # page number where the chunk STARTS
                    "section_title":  str,   # boundary line that started the chunk
                    "doc_id":         str,   # inherited from page
                },
                ...
            ]

        If no boundary is found on the first page, an implicit "Preamble"
        chunk collects text up to the first boundary.
    """
    chunks: list[dict] = []
    current_content: list[str] = []
    current_page: int | None = None
    current_title: str = "Preamble"
    doc_id: str | None = None

    def _flush():
        """Emit accumulated content as a chunk if non-empty."""
        if current_content and current_page is not None and doc_id is not None:
            joined = "\n".join(current_content).strip()
            if joined:
                chunks.append(
                    {
                        "content": joined,
                        "page": current_page,
                        "section_title": current_title,
                        "doc_id": doc_id,
                    }
                )

    for page in pages:
        doc_id = page["doc_id"]
        page_num = page["page"]
        text = page.get("text", "")

        for line in text.splitlines():
            _, boundary_title = _is_boundary(line)
            if boundary_title is not None:
                # New clause boundary → flush what we have, start fresh
                _flush()
                current_content = [line]
                current_page = page_num
                current_title = boundary_title
            else:
                # First real content of the doc has no boundary yet
                if current_page is None and line.strip():
                    current_page = page_num
                current_content.append(line)

    _flush()
    return chunks
