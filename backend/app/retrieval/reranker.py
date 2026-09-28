"""
Task 3.3 — Cross-encoder reranker.

Fusion (Task 3.2) hands us ~20 candidate chunks. The LLM can't read all
of them — too much context, too slow, too expensive. Just picking the
top-5 by fusion score is mediocre quality.

A CROSS-ENCODER takes (query, chunk) as a PAIR and gives a precise
relevance score. It's slower per-pair than a bi-encoder (which is what
BGE-M3 is — encodes query and chunk separately then compares vectors),
but far more accurate because the model sees both texts at once and can
learn true relevance signals.

Standard flow:
    fusion → ~20 candidates → rerank → top 5-10 for the LLM

Model choice: cross-encoder/ms-marco-MiniLM-L-6-v2 — the go-to reranker
for RAG. Small (~90MB), CPU-viable, ~20-30% precision lift over dense-only.

The model is loaded once per process (module-level singleton) to avoid
reload cost on every query.
"""

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Module-level singleton, keyed by model name.
_models: dict[str, Any] = {}


def _get_model(model_name: str) -> Any:
    """Load cross-encoder once; return cached instance thereafter."""
    if model_name not in _models:
        from sentence_transformers import CrossEncoder  # noqa: PLC0415

        logger.info(
            "Loading cross-encoder '%s' — first load takes a few seconds.",
            model_name,
        )
        t0 = time.perf_counter()
        _models[model_name] = CrossEncoder(model_name)
        logger.info(
            "Cross-encoder '%s' loaded in %.1fs.",
            model_name,
            time.perf_counter() - t0,
        )
    return _models[model_name]


def _get_text(candidate: dict) -> str:
    """Extract the chunk text for scoring. Looks in payload.content, then content."""
    payload = candidate.get("payload", {})
    text = payload.get("content") or candidate.get("content")
    if not text:
        raise ValueError(
            f"Candidate is missing chunk text — expected payload['content'] or 'content'. "
            f"Got keys: {list(candidate.keys())}"
        )
    return text


def rerank(
    query: str,
    candidates: list[dict],
    k: int = 5,
    model_name: str = _DEFAULT_MODEL,
) -> list[dict]:
    """Rerank candidates by cross-encoder relevance to the query.

    Args:
        query:      The user's question.
        candidates: Output of hybrid.fuse() — list of dicts each with
                    'payload' (which must contain 'content') OR a top-level
                    'content' key. Every dict is returned in the output
                    with its original keys plus a new 'rerank_score'.
        k:          Max candidates to return (usually 5-10 before LLM synthesis).
        model_name: Cross-encoder model ID. Defaults to ms-marco-MiniLM-L-6-v2.

    Returns:
        Top-k candidates sorted descending by rerank_score. Each dict is
        the original candidate dict with an added 'rerank_score' key (float).

    Raises:
        ValueError: If any candidate is missing its chunk text.
    """
    if not candidates:
        return []

    texts = [_get_text(c) for c in candidates]
    pairs = [(query, text) for text in texts]

    t0 = time.perf_counter()
    model = _get_model(model_name)
    scores = model.predict(pairs)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    # Attach scores to candidates, sort descending, take top k
    scored = []
    for candidate, score in zip(candidates, scores):
        enriched = dict(candidate)
        enriched["rerank_score"] = float(score)
        scored.append(enriched)

    scored.sort(key=lambda c: c["rerank_score"], reverse=True)

    logger.info(
        "Reranker: scored %d pairs in %.1fms (%.1fms/pair), returning top %d.",
        len(pairs),
        elapsed_ms,
        elapsed_ms / max(len(pairs), 1),
        min(k, len(scored)),
    )
    return scored[:k]
