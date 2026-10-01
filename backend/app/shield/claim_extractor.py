"""
Task 5.1 — Report claim extractor.

Takes a FINISHED report (compliance filing, audit report, legal opinion)
and splits it into individual verifiable CLAIMS — each a single statement
that can be independently checked against the case corpus in Task 5.2.

Distinction from Module 1 chunking:
    Module 1 chunks are for SEARCH (keep clauses together for retrieval).
    Claims here are for VERIFICATION (smallest independently checkable unit —
    usually one sentence, sometimes one numbered bullet).

Splitting strategy (in priority order):
    1. Numbered bullets/clauses: '1. ', '(a) ', '• ' at line start.
       These are usually atomic claims in Indian legal/compliance docs.
    2. Sentence-level fallback for prose paragraphs: spaCy en_core_web_sm.
       Paragraphs with no numbered structure get split into sentences.
    3. If spaCy model is unavailable (CI without the download), falls back
       to a simple regex sentence splitter.

spaCy model download (one-time, not in this file):
    python -m spacy download en_core_web_sm
"""

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# ─── spaCy lazy-load ──────────────────────────────────────────────────────────

_nlp = None


def _get_nlp():
    """Load spaCy model once; return None if not installed (triggers fallback)."""
    global _nlp
    if _nlp is not None:
        return _nlp
    try:
        import spacy  # noqa: PLC0415
        _nlp = spacy.load("en_core_web_sm")
        logger.debug("spaCy en_core_web_sm loaded for claim extraction.")
    except (ImportError, OSError) as exc:
        logger.warning(
            "spaCy or en_core_web_sm not available (%s). "
            "Falling back to regex sentence splitter.",
            exc,
        )
        _nlp = False  # sentinel — don't retry
    return _nlp


# ─── boundary patterns (numbered bullets / clause markers) ────────────────────

# Matches: '1. ', '1) ', '(1) ', '(a) ', '(i) ', '• ', '- '
_BULLET_RE = re.compile(
    r"^(?:\d+[\.\)]\s|\([a-z0-9]+\)\s|[•\-–]\s)",
    re.MULTILINE,
)

# Simple sentence splitter fallback (period/exclamation/question + space + capital)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def _split_sentences(text: str) -> list[str]:
    """Split prose into sentences using spaCy if available, else regex."""
    nlp = _get_nlp()
    if nlp:
        doc = nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    # Regex fallback
    parts = _SENTENCE_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def _extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract plain text from a PDF using PyMuPDF (Task 1.1 dependency)."""
    import pymupdf  # noqa: PLC0415

    doc = pymupdf.open(str(pdf_path))
    pages = [doc[i].get_text() for i in range(len(doc))]
    doc.close()
    return "\n".join(pages)


# ─── public API ───────────────────────────────────────────────────────────────


def extract_claims(raw_text: str) -> list[str]:
    """Split raw report text into verifiable claims (one claim = one sentence/bullet).

    Pure-text version of chunk_report() — used by shield.py router and by
    the 10.2 eval harness. chunk_report() calls this after PDF extraction.
    """
    claims: list[str] = []
    for paragraph in _split_paragraphs(raw_text):
        para = paragraph.strip()
        if not para:
            continue
        if _BULLET_RE.search(para):
            items = _BULLET_RE.split(para)
            for item in items:
                item = item.strip()
                if item:
                    claims.append(item)
        else:
            claims.extend(_split_sentences(para))

    seen: set[str] = set()
    clean: list[str] = []
    for claim in claims:
        normalised = claim.strip()
        if normalised and len(normalised) >= 10 and normalised not in seen:
            seen.add(normalised)
            clean.append(normalised)
    return clean


def chunk_report(pdf_path: str) -> list[str]:
    """Extract individual verifiable claims from a report PDF.

    Args:
        pdf_path: Path to the uploaded report PDF (already ingested as a doc).

    Returns:
        List of claim strings. Each claim is a single verifiable statement —
        either one numbered bullet or one sentence. Empty claims are omitted.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be opened as a PDF.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"Report PDF not found: {path}")

    try:
        raw_text = _extract_text_from_pdf(path)
    except Exception as exc:
        raise ValueError(f"Cannot read PDF '{path.name}': {exc}") from exc

    clean = extract_claims(raw_text)
    logger.info(
        "chunk_report: extracted %d claims from '%s'.",
        len(clean),
        path.name,
    )
    return clean


def _split_paragraphs(text: str) -> list[str]:
    """Split on blank lines (paragraph boundaries)."""
    return re.split(r"\n{2,}", text)
