"""
embedding-dispatcher — Route a document to e5-small-v2 or BGE-M3 at ingestion time.

Three signals, ANY ONE of which triggers the upgrade to BGE-M3:

  1. File size  > 1 MB  (large PDF → needs BGE-M3's 8192-token context)
  2. Page count > 10    (long document — more context per chunk matters)
  3. Non-ASCII script   (Devanagari, Arabic, CJK, etc. — e5-small-v2 is
                         English-only; BGE-M3 handles 100+ languages)

e5-small-v2 is the default when all three signals are absent. It is faster,
uses less VRAM, and is perfectly accurate on short English documents.

Output: the tier key ("2" for e5-small or "bhavesh"/"1" for BGE-M3,
whichever non-e5 tier is configured) plus a human-readable reason list.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# ─── thresholds ───────────────────────────────────────────────────────────────
SIZE_THRESHOLD_MB: float = float(os.environ.get("DISPATCHER_SIZE_MB", "1.0"))
PAGE_THRESHOLD: int = int(os.environ.get("DISPATCHER_PAGE_COUNT", "10"))

# Unicode script ranges that e5-small-v2 handles poorly
_MULTILINGUAL_PATTERN = re.compile(
    r"[ऀ-ॿ"   # Devanagari (Hindi, Marathi, Sanskrit)
    r"؀-ۿ"   # Arabic
    r"一-鿿"   # CJK Unified Ideographs (Chinese, Japanese, Korean)
    r"Ѐ-ӿ"   # Cyrillic
    r"֐-׿"   # Hebrew
    r"஀-௿"   # Tamil
    r"ಀ-೿"   # Kannada
    r"ഀ-ൿ"   # Malayalam
    r"઀-૿"   # Gujarati
    r"਀-੿"   # Gurmukhi (Punjabi)
    r"]"
)


@dataclass
class DispatchDecision:
    tier: str              # "e5_small" or "bge_m3"
    reasons: list[str] = field(default_factory=list)
    size_mb: float = 0.0
    page_count: int = 0
    multilingual: bool = False

    @property
    def collection_name(self) -> str:
        return "legalvault_bge" if self.tier == "bge_m3" else "legalvault_e5"

    @property
    def model_name(self) -> str:
        """Return the sentence-transformers model string for this tier."""
        if self.tier == "bge_m3":
            return "BAAI/bge-m3"
        return "intfloat/e5-small-v2"

    @property
    def embedding_dim(self) -> int:
        return 1024 if self.tier == "bge_m3" else 384


def dispatch(
    file_path: str,
    pages: list[dict],
) -> DispatchDecision:
    """Decide which embedding tier to use for a document.

    Args:
        file_path: Path to the stored document file (used for size check).
        pages:     Extracted pages from pdf/txt/docx extractor (list of
                   {'text': str, 'page': int, 'doc_id': str}).

    Returns:
        DispatchDecision with tier, collection_name, and model_name set.
    """
    size_mb = Path(file_path).stat().st_size / (1024 * 1024) if Path(file_path).exists() else 0.0
    page_count = len(pages)
    full_text = " ".join(p.get("text", "") for p in pages)
    multilingual = bool(_MULTILINGUAL_PATTERN.search(full_text))

    reasons: list[str] = []
    use_bge = False

    if size_mb > SIZE_THRESHOLD_MB:
        use_bge = True
        reasons.append(f"file size {size_mb:.1f} MB > {SIZE_THRESHOLD_MB} MB threshold")

    if page_count > PAGE_THRESHOLD:
        use_bge = True
        reasons.append(f"page count {page_count} > {PAGE_THRESHOLD} threshold")

    if multilingual:
        use_bge = True
        reasons.append("non-ASCII script detected (multilingual content)")

    tier = "bge_m3" if use_bge else "e5_small"

    decision = DispatchDecision(
        tier=tier,
        reasons=reasons,
        size_mb=size_mb,
        page_count=page_count,
        multilingual=multilingual,
    )

    if use_bge:
        logger.info(
            "dispatcher: routed to BGE-M3 — %s",
            "; ".join(reasons),
        )
    else:
        logger.debug("dispatcher: routed to e5-small-v2 (no upgrade triggers)")

    return decision
