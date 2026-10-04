"""
Phase 2 — Hierarchical (parent-child) chunker.

Takes the output of extract_docling() (preferred) or extract_pdf() (fallback)
and produces TWO kinds of chunks:

    CHILD  (~200 chars)   — one per clause / paragraph / table row.
                            These are embedded, stored in Qdrant, and used
                            for retrieval (precision).
    PARENT (~1500 chars)  — the full section a child belongs to.
                            Stored only in SQLite (never embedded).
                            After retrieval, the RAG service swaps each
                            top-K child for its parent and sends the parent
                            to the LLM (recall / context).

Why this helps:
    phi3:mini can echo a verbatim phrase from a long chunk but struggles
    to extract a specific number from a 1700-char monolith. By making
    children small (precise retrieval) and parents larger (full context
    for the LLM), we fix the "d05 0% LLM accuracy" failure mode from the
    eval_corpus run while not hurting Recall@5.

Return shape:
    {
        "parents":  [parent_dict, ...],
        "children": [child_dict_with_parent_index, ...],
    }

    Each child has an integer `parent_index` into the parents list. The
    ingestion layer persists parents first, grabs their UUIDs, then
    persists children with parent_chunk_id = parents[parent_index].id.

Tables are atomic: one child == one parent for the whole table, with the
caption and header row prepended so the retrieved snippet is self-contained.
"""

from __future__ import annotations

import re
from typing import Any

# ─── target sizes (characters) ───────────────────────────────────────────────
CHILD_TARGET_CHARS = 200
CHILD_MAX_CHARS = 450          # split a paragraph into multiple children above this
# Below this, consecutive body items are merged into one child. Fixes two
# real retrieval failures seen on the test corpus:
#   - Q&A docs where "Q: foo?" and "A: bar." land as separate 80-char items;
#     BM25 ranks the Q over the A for user queries.
#   - Numbered-clause PDFs where "3.1" and the clause body get split.
CHILD_MERGE_BELOW_CHARS = 180
PARENT_TARGET_CHARS = 1500
PARENT_MAX_CHARS = 3000        # split a section into multiple parents above this
MIN_CHUNK_CHARS = 40           # smaller than this gets merged with neighbour


# ─── sentence splitter (regex-based; no spaCy dependency) ────────────────────
# Breaks on `. ! ?` followed by whitespace + capital letter, OR on `;` + space.
# Not perfect for legalese (e.g. "Mr.", "Section 1."), but good enough — the
# small over-splits don't hurt retrieval and the clause-boundary regex below
# catches most important cases.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Zऀ-ॿ])|(?<=;)\s+")

# Numbered clause boundary — same priority rules as clause_chunker.py
_NUMBERED_CLAUSE = re.compile(r"^(?P<num>\d+\.\d+(?:\.\d+)*\.?|\d+\.)\s+")


def _split_long_paragraph(text: str, max_chars: int) -> list[str]:
    """Split a paragraph that exceeds max_chars into sub-chunks on sentence boundaries."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []

    sentences = _SENTENCE_SPLIT.split(text)
    chunks: list[str] = []
    buf = ""
    for s in sentences:
        if not s.strip():
            continue
        if len(buf) + len(s) + 1 > max_chars and buf:
            chunks.append(buf.strip())
            buf = s
        else:
            buf = f"{buf} {s}" if buf else s
    if buf.strip():
        chunks.append(buf.strip())
    return chunks or [text]


def _is_heading_level(label: str) -> int:
    """Return heading depth (1=H1, 2=H2, 3=H3) or 0 if not a heading."""
    mapping = {"H1": 1, "title": 1, "H2": 2, "H3": 3}
    return mapping.get(label, 0)


def _is_table(label: str) -> bool:
    return label == "table"


def _flush_parent(
    parents: list[dict],
    buf: list[dict],
    section_title: str | None,
    first_page: int | None,
) -> int:
    """Create a parent chunk from buffered items and return its index."""
    text = "\n".join(it["text"] for it in buf if it.get("text", "").strip())
    if not text.strip():
        return -1
    parents.append({
        "chunk_role": "parent",
        "content": text,
        "page": first_page,
        "section_title": section_title,
        "is_table": False,
    })
    return len(parents) - 1


def _make_children_for_item(
    item: dict,
    parent_index: int,
    section_title: str | None,
) -> list[dict]:
    """Produce one or more children for a single Docling item."""
    text = item.get("text", "").strip()
    if not text:
        return []
    pieces = _split_long_paragraph(text, CHILD_MAX_CHARS)
    return [
        {
            "chunk_role": "child",
            "content": piece,
            "page": item.get("page"),
            "section_title": section_title,
            "parent_index": parent_index,
            "is_table": False,
        }
        for piece in pieces
    ]


def _chunk_docling_pages(pages: list[dict]) -> dict:
    """Chunk pages that came from extract_docling() (have `items` with labels)."""
    parents: list[dict] = []
    children: list[dict] = []

    buf: list[dict] = []            # items accumulated for current parent
    buf_chars = 0
    section_title: str | None = None
    section_first_page: int | None = None

    def flush():
        nonlocal buf, buf_chars
        if not buf:
            return -1
        idx = _flush_parent(parents, buf, section_title, section_first_page)
        buf = []
        buf_chars = 0
        return idx

    for page in pages:
        page_no = page.get("page", 1)
        items = page.get("items") or []

        # Fallback: no items (plain-text extractor) — treat full page text
        # as one item so we at least get page-level parent/child grouping.
        if not items and page.get("text", "").strip():
            items = [{"text": page["text"], "heading_level": "body", "page": page_no}]

        for raw in items:
            text = (raw.get("text") or "").strip()
            if not text:
                continue

            label = raw.get("heading_level") or raw.get("label") or "body"
            item = {"text": text, "page": page_no}

            # Table — atomic parent + atomic child, flush any pending buffer first
            if _is_table(label):
                flush()
                parents.append({
                    "chunk_role": "parent",
                    "content": text,
                    "page": page_no,
                    "section_title": section_title or "Table",
                    "is_table": True,
                })
                parent_idx = len(parents) - 1
                children.append({
                    "chunk_role": "child",
                    "content": text,
                    "page": page_no,
                    "section_title": section_title or "Table",
                    "parent_index": parent_idx,
                    "is_table": True,
                })
                continue

            # Heading — start a new section (flush current buffer first)
            depth = _is_heading_level(label)
            if depth > 0:
                flush()
                section_title = text[:200]
                section_first_page = page_no
                # The heading itself is NOT a child chunk; it's just section metadata.
                continue

            # Numbered clause boundary — treat like a mini-heading if we have
            # enough buffered text to make a parent and we're not inside a table.
            m = _NUMBERED_CLAUSE.match(text)
            if m and buf_chars > PARENT_TARGET_CHARS:
                flush()
                section_title = text[:80]
                section_first_page = page_no

            # Add to current parent buffer
            if section_first_page is None:
                section_first_page = page_no
            buf.append(item)
            buf_chars += len(text) + 1

            # Parent size cap — split before it gets unwieldy
            if buf_chars >= PARENT_MAX_CHARS:
                flush()

    # Final flush
    flush()

    # Second pass: emit children, each tagged with its parent_index
    # We rebuild by walking the parents' source items — but we don't have
    # per-parent item lists any more, so walk pages a second time matching
    # by text equality. Simpler: redo the pass alongside parent creation.
    # (Rebuild below so we don't duplicate logic.)
    return _emit_children_for_docling(pages, parents)


def _emit_children_for_docling(pages: list[dict], parents: list[dict]) -> dict:
    """Second pass: assign every item to the parent whose text contains it,
    merging consecutive short body items into one child so Q/A pairs and
    numbered-clause prefix+body don't end up as separate retrieval units.
    """
    children: list[dict] = []
    parent_cursor = 0

    # Pending merge buffer — body items under CHILD_MERGE_BELOW_CHARS get
    # concatenated until buf length crosses the threshold or we hit a boundary
    # (heading, table, parent change).
    pending_text: str = ""
    pending_page: int | None = None
    pending_parent_idx: int = -1

    def flush_pending():
        nonlocal pending_text, pending_page, pending_parent_idx
        if pending_text.strip() and 0 <= pending_parent_idx < len(parents):
            children.append({
                "chunk_role": "child",
                "content": pending_text.strip(),
                "page": pending_page,
                "section_title": parents[pending_parent_idx].get("section_title"),
                "parent_index": pending_parent_idx,
                "is_table": False,
            })
        pending_text = ""
        pending_page = None
        pending_parent_idx = -1

    for page in pages:
        page_no = page.get("page", 1)
        items = page.get("items") or (
            [{"text": page["text"], "heading_level": "body", "page": page_no}]
            if page.get("text", "").strip() else []
        )

        for raw in items:
            text = (raw.get("text") or "").strip()
            if not text:
                continue
            label = raw.get("heading_level") or raw.get("label") or "body"

            # Table — atomic child, flush pending prose first
            if _is_table(label):
                flush_pending()
                for idx, p in enumerate(parents):
                    if p.get("is_table") and p["content"] == text:
                        children.append({
                            "chunk_role": "child",
                            "content": text,
                            "page": page_no,
                            "section_title": p.get("section_title"),
                            "parent_index": idx,
                            "is_table": True,
                        })
                        break
                continue

            # Headings — flush pending prose before the next section
            if _is_heading_level(label) > 0:
                flush_pending()
                continue

            # Advance parent cursor until this item's text is inside it
            while parent_cursor < len(parents):
                p = parents[parent_cursor]
                if p.get("is_table"):
                    parent_cursor += 1
                    continue
                if text in p["content"]:
                    break
                parent_cursor += 1
            if parent_cursor >= len(parents):
                parent_cursor = len(parents) - 1 if parents else 0

            # If moving to a new parent, flush the current merge buffer
            if pending_parent_idx != -1 and pending_parent_idx != parent_cursor:
                flush_pending()

            # Long item → split into normal-sized children, flush pending first
            if len(text) > CHILD_MERGE_BELOW_CHARS:
                flush_pending()
                for piece in _split_long_paragraph(text, CHILD_MAX_CHARS):
                    children.append({
                        "chunk_role": "child",
                        "content": piece,
                        "page": page_no,
                        "section_title": parents[parent_cursor].get("section_title") if parents else None,
                        "parent_index": parent_cursor,
                        "is_table": False,
                    })
                continue

            # Short item → add to merge buffer; flush when it crosses target
            if pending_text:
                pending_text = f"{pending_text}\n{text}"
            else:
                pending_text = text
                pending_page = page_no
                pending_parent_idx = parent_cursor

            if len(pending_text) >= CHILD_TARGET_CHARS:
                flush_pending()

    # Trailing flush
    flush_pending()

    return {"parents": parents, "children": children}


def _chunk_fallback_pages(pages: list[dict]) -> dict:
    """Chunk pages that came from a non-Docling extractor (no `items` field).

    Strategy: use the legacy clause_chunker to produce clause-level units.
    Each clause becomes a child. Clauses on the same page form one parent;
    if a page's clauses exceed PARENT_MAX_CHARS we split the parent.
    """
    from backend.app.ingestion.clause_chunker import chunk_legal_doc

    legacy_chunks = chunk_legal_doc(pages)
    if not legacy_chunks:
        return {"parents": [], "children": []}

    parents: list[dict] = []
    children: list[dict] = []
    buf: list[str] = []
    buf_chars = 0
    buf_page: int | None = None
    buf_section: str | None = None

    def flush_parent():
        nonlocal buf, buf_chars, buf_page, buf_section
        if not buf:
            return -1
        parents.append({
            "chunk_role": "parent",
            "content": "\n".join(buf),
            "page": buf_page,
            "section_title": buf_section,
            "is_table": False,
        })
        buf = []
        buf_chars = 0
        return len(parents) - 1

    for ch in legacy_chunks:
        content = (ch.get("content") or "").strip()
        if not content:
            continue
        page = ch.get("page")
        section = ch.get("section_title")

        if buf_chars + len(content) > PARENT_MAX_CHARS and buf:
            parent_idx = flush_parent()
        else:
            parent_idx = None

        if not buf:
            buf_page = page
            buf_section = section
        buf.append(content)
        buf_chars += len(content) + 1

    # Capture trailing buffer, then re-walk and attach each legacy chunk
    # as a child of whichever parent contains it.
    flush_parent()

    # Second pass: emit children
    parent_cursor = 0
    for ch in legacy_chunks:
        content = (ch.get("content") or "").strip()
        if not content:
            continue
        # Advance parent cursor to find the one containing this clause
        while parent_cursor < len(parents) and content not in parents[parent_cursor]["content"]:
            parent_cursor += 1
        if parent_cursor >= len(parents):
            parent_cursor = len(parents) - 1

        for piece in _split_long_paragraph(content, CHILD_MAX_CHARS):
            children.append({
                "chunk_role": "child",
                "content": piece,
                "page": ch.get("page"),
                "section_title": ch.get("section_title"),
                "parent_index": parent_cursor,
                "is_table": False,
            })

    return {"parents": parents, "children": children}


def chunk_hierarchical(pages: list[dict]) -> dict:
    """Entry point: produce parent + child chunks from extractor pages.

    Args:
        pages: Output of extract_docling() or extract_pdf()/extract_text()/extract_docx().
               Must be a list of per-page dicts with at least `text` and `page`.
               If the extractor was Docling, each page also has `items` with
               `heading_level` metadata — the hierarchical path is used.
               Otherwise falls back to clause_chunker + page-group parents.

    Returns:
        {
            "parents":  [ {chunk_role, content, page, section_title, is_table}, ... ],
            "children": [ {chunk_role, content, page, section_title, parent_index, is_table}, ... ],
        }
    """
    if not pages:
        return {"parents": [], "children": []}

    # Docling pages have an `items` list on at least one page
    has_items = any(p.get("items") for p in pages)
    if has_items:
        return _chunk_docling_pages(pages)
    return _chunk_fallback_pages(pages)
