# Task 1: Phase 1 — Docling parser

## What I built
- Installed `docling` package and added it to `backend/requirements.txt`.
- Created `docling_extractor.py` as a wrapper over `Docling's` `DocumentConverter` with chunk metadata `heading_level` extraction.
- Replaced `extract_pdf` usage with `extract_docling` in `documents.py` and `shield.py`.
- Added a fallback to `pymupdf` using `extract_pdf` when `Docling` fails because a document appears to be scanned image-only.
- Added DOCX file support as `.docx` suffix.

## Files touched
- backend/requirements.txt
- backend/app/ingestion/docling_extractor.py
- backend/app/routers/documents.py
- backend/app/routers/shield.py

## How to test
- Run `pytest tests/eval/eval_retrieval.py`
- Test normal extraction works properly and fails back correctly.

## Known gaps
- Docling handles standard hierarchical metadata in headers, but some edge case labels mapping to H1/H2/H3 might need tweaking.
