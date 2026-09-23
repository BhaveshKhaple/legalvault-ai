"""
Task 3.1 — BM25 sparse keyword index.

Semantic search (Module 2) understands meaning but misses exact tokens:
clause section numbers like '8.3', exact dates, monetary amounts. BM25
fixes this — it scores documents by literal word overlap with the query.

We combine BM25 and dense search later in Task 3.2 (Hybrid Fusion). This
module only builds and queries the BM25 side.

Performance target (from tracker):
    bm25.query(text, k=10) returns top-10 in < 100ms on a 1000-chunk corpus.
    rank_bm25 is pure Python and easily hits this on modern hardware.

Usage:
    from backend.app.retrieval.bm25_index import BM25Index, build_bm25

    index = build_bm25(chunks)          # chunks = output of clause_chunker
    results = index.query("8.3 penalty", k=10)
    # results → [{"chunk": {...}, "score": float}, ...]
"""

import re
from dataclasses import dataclass, field

from rank_bm25 import BM25Okapi


# ─── tokeniser ────────────────────────────────────────────────────────────────

def _tokenise(text: str) -> list[str]:
    """Lowercase + split on non-alphanumeric boundaries.

    Preserves dotted numbers like '8.3' and '3.1.2' as single tokens so
    BM25 can match exact clause references.
    """
    text = text.lower()
    # Keep alphanumeric + dots-between-digits intact; split on everything else.
    tokens = re.findall(r"\d+(?:\.\d+)+|[a-z0-9]+", text)
    return tokens if tokens else [""]


# ─── index class ──────────────────────────────────────────────────────────────

@dataclass
class BM25Index:
    """Wrapper around BM25Okapi that keeps chunks alongside their BM25 tokens."""

    _bm25: BM25Okapi = field(repr=False)
    _chunks: list[dict] = field(repr=False)

    def query(self, text: str, k: int = 10) -> list[dict]:
        """Return top-k chunks matching the query, sorted by BM25 score descending.

        Args:
            text: Natural-language or keyword query string.
            k: Maximum results to return.

        Returns:
            List of result dicts::

                [
                    {
                        "chunk": { ...original chunk dict... },
                        "score": float,   # BM25 Okapi score
                    },
                    ...
                ]

            Chunks with score == 0.0 are excluded (no term overlap at all).
            Result list may be shorter than k if fewer matches exist.
        """
        if not text.strip():
            return []

        tokens = _tokenise(text)
        scores = self._bm25.get_scores(tokens)

        ranked = sorted(
            zip(scores, self._chunks),
            key=lambda pair: pair[0],
            reverse=True,
        )

        return [
            {"chunk": chunk, "score": float(score)}
            for score, chunk in ranked[:k]
            if score > 0.0
        ]


# ─── factory function ─────────────────────────────────────────────────────────

def build_bm25(chunks: list[dict]) -> BM25Index:
    """Build a BM25 index from a list of chunk dicts.

    Args:
        chunks: Output of clause_chunker.chunk_legal_doc() — each dict must
                have at least a 'content' key (str).

    Returns:
        A BM25Index ready for querying.

    Raises:
        ValueError: If chunks is empty (BM25Okapi cannot index zero documents).
    """
    if not chunks:
        raise ValueError("Cannot build BM25 index from an empty chunk list.")

    tokenised = [_tokenise(chunk["content"]) for chunk in chunks]
    bm25 = BM25Okapi(tokenised)
    return BM25Index(_bm25=bm25, _chunks=list(chunks))
