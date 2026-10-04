"""
Task Phase 1 — Docling parser.

Extracts text page-by-page from a PDF or DOCX using Docling.
Emits pages with heading_level (H1/H2/H3/body/table) metadata on chunks.
"""

import logging
from pathlib import Path
from docling.document_converter import DocumentConverter

logger = logging.getLogger(__name__)

# Cache converter instance to avoid reloading models on every call
_converter = None


def get_converter():
    global _converter
    if _converter is None:
        _converter = DocumentConverter()
    return _converter


def extract_docling(path: str) -> list[dict]:
    """Extract text page-by-page from a PDF/DOCX using Docling.

    Args:
        path: Absolute or relative path to the PDF/DOCX.

    Returns:
        List of dicts, one per page::

            [
                {
                    "text":   str,
                    "page":   int,
                    "doc_id": str,
                    "items":  list[dict] # { "label": str, "text": str }
                },
                ...
            ]

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError/Exception: If the file cannot be opened.
        RuntimeError: If Docling fails on scanned-image PDFs.
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    doc_id = file_path.stem
    converter = get_converter()

    try:
        res = converter.convert(str(file_path))
    except Exception as exc:
        raise ValueError(f"Docling cannot convert '{file_path.name}': {exc}") from exc

    pages_map = {}
    total_text_len = 0

    for item, level in res.document.iterate_items():
        text = getattr(item, "text", "")
        # Phase 2 — table items usually have empty `.text`; export as markdown
        # so the cell data lands in the index. Without this, fee schedules /
        # pricing tables silently become empty children and retrieval misses
        # every numeric fact inside them.
        label_check = getattr(item, "label", None)
        label_check_val = getattr(label_check, "value", label_check)
        if (not text) and label_check_val == "table":
            try:
                text = item.export_to_markdown(doc=res.document)
            except Exception:
                try:
                    text = item.export_to_markdown()
                except Exception:
                    text = ""
        if text:
            total_text_len += len(text.strip())

        page_no = None
        prov = getattr(item, "prov", None)
        if prov and len(prov) > 0:
            page_no = prov[0].page_no

        if page_no is None:
            page_no = 1

        if page_no not in pages_map:
            pages_map[page_no] = {
                "text": [],
                "page": page_no,
                "doc_id": doc_id,
                "items": [],
            }

        # Determine heading level logic (H1/H2/H3/body/table)
        label_val = getattr(item, "label", "unknown")
        if hasattr(label_val, "value"):
            label_val = (
                label_val.value
            )  # e.g. DocItemLabel.SECTION_HEADER -> 'section_header'

        meta_label = "body"
        if label_val == "section_header":
            # Assign heading level based on level or some logic.
            # level is given by iterate_items
            if level == 1:
                meta_label = "H1"
            elif level == 2:
                meta_label = "H2"
            else:
                meta_label = "H3"
        elif label_val == "table":
            meta_label = "table"
        elif label_val == "text":
            meta_label = "body"
        else:
            meta_label = label_val

        pages_map[page_no]["items"].append(
            {
                "label": label_val,
                "heading_level": meta_label,
                "text": text,
                "level": level,
            }
        )
        if text:
            pages_map[page_no]["text"].append(text)

    # 1.1 fallback rule - if less than 2 characters returned, it is probably a scanned page
    _SCANNED_TEXT_THRESHOLD = 2
    if total_text_len < _SCANNED_TEXT_THRESHOLD:
        raise RuntimeError(
            "DocumentConversionError: Docling returned virtually no text, fallback to PyMuPDF."
        )

    results = []
    if not pages_map:
        return []

    for page_num in sorted(pages_map.keys()):
        page_dict = pages_map[page_num]
        page_dict["text"] = "\n".join(page_dict["text"])
        results.append(page_dict)

    return results
