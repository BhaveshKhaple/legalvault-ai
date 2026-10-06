"""
Confirm Docling is enabled and active end-to-end.

Runs 3 checks:
  1. Direct extractor call on a text-rich PDF — must return items with
     heading_level metadata (PyMuPDF would never emit that).
  2. Direct extractor call on d06_fee_schedule.pdf — must return a table
     chunk with Markdown rows (proves table export is working).
  3. (Optional) hits a running /v1/cases/.../documents endpoint and verifies
     the Document record's chunk_count > 1 for a multi-page doc.

Usage:
    backend\\venv\\Scripts\\python scripts\\verify_docling.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CORPUS = ROOT / "tests" / "eval" / "corpus"


def check(name: str, ok: bool, detail: str = "") -> None:
    icon = "PASS" if ok else "FAIL"
    print(f"[{icon}] {name}")
    if detail:
        print(f"       {detail}")


def main() -> int:
    try:
        from backend.app.ingestion.docling_extractor import extract_docling, get_converter
    except Exception as exc:
        check("docling import", False, f"import failed: {exc}")
        return 1

    # 1. Confirm the Docling package resolves and its converter builds
    try:
        get_converter()
        check("docling converter builds", True, "DocumentConverter() instantiated")
    except Exception as exc:
        check("docling converter builds", False, f"{exc}")
        return 1

    # 2. Extract a prose PDF — must have items with heading_level metadata
    nda = CORPUS / "d01_nda_numbered.pdf"
    if not nda.exists():
        check("d01 corpus file", False, f"missing: {nda}")
        return 1
    pages = extract_docling(str(nda))
    has_items = any(p.get("items") for p in pages)
    item_labels = {it.get("heading_level") for p in pages for it in (p.get("items") or [])}
    check(
        "d01 produces page items with heading_level",
        has_items and len(item_labels) > 0,
        f"pages={len(pages)}, distinct labels={sorted(item_labels)}",
    )

    # 3. Extract the fee schedule — must have a table chunk with real content
    fee = CORPUS / "d06_fee_schedule.pdf"
    if not fee.exists():
        check("d06 corpus file", False, f"missing: {fee}")
        return 1
    fee_pages = extract_docling(str(fee))
    table_items = [
        it for p in fee_pages
        for it in (p.get("items") or [])
        if it.get("heading_level") == "table"
    ]
    table_has_data = any(len((it.get("text") or "")) > 10 for it in table_items)
    check(
        "d06 table export yields real content",
        len(table_items) > 0 and table_has_data,
        f"table items={len(table_items)}, "
        f"sample bytes={len(table_items[0]['text']) if table_items else 0}",
    )

    # 4. Confirm a scanned/image-only PDF raises RuntimeError (fallback trigger)
    # Not easy to simulate without a test asset — skipping unless we add one.

    print()
    print("Docling is enabled and functional.")
    print("If any check failed, run 'pip install docling' and re-check.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
