"""
Task 3.2 — Hybrid retrieval: fuse dense (Qdrant) + sparse (BM25) results.

Uses Reciprocal Rank Fusion (RRF): a widely used, hyperparameter-light
technique that combines multiple ranked lists into one. Each candidate's
final score is the sum of 1/(k + rank_i) across the lists it appears in,
where k is a smoothing constant (60 is the standard from the original
Cormack et al. paper).

Why RRF beats naive score averaging:
    Dense scores (cosine 0-1) and BM25 scores (unbounded log-odds) live
    on different scales. Averaging them is meaningless. RRF only uses
    ranks, so scale doesn't matter.

Dedup: results from both paths often overlap. We keep only ONE entry per
unique chunk_id and sum the RRF contributions from each list.

Chunk identity:
    The 'chunk_id' key is required on every result so we can dedup and
    later cite the chunk back to the LLM. Both callers must supply it:
      - Dense results: 'id' from Qdrant (server-assigned UUID) OR you can
        pass 'chunk_id' explicitly in the payload.
      - Sparse results: BM25 doesn't know about chunk_ids, so the caller
        must inject one before fusion — usually the same UUID stored in
        Qdrant's payload.

    fuse() looks first at result['chunk_id'], falling back to result['id']
    (Qdrant convention). It raises ValueError if neither is present.
"""

import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

_RRF_K = 60  # smoothing constant from Cormack et al. 2009


def _extract_chunk_id(result: dict, source: str, position: int) -> str:
    """Get the identifier we dedup on. Raises if the caller forgot to set one."""
    chunk_id = result.get("chunk_id") or result.get("id")
    if chunk_id is None:
        raise ValueError(
            f"{source} result at position {position} has neither 'chunk_id' nor 'id'. "
            "Fusion cannot dedup without a stable identifier."
        )
    return str(chunk_id)


def fuse(
    dense_results: list[dict],
    sparse_results: list[dict],
    k: int = 20,
    rrf_k: int = _RRF_K,
) -> list[dict]:
    """Merge dense and sparse ranked lists via Reciprocal Rank Fusion.

    Args:
        dense_results:  Output of vector_store.query() — list of dicts, each
                        with at least 'id' (or 'chunk_id') + 'payload'.
                        Order matters: index 0 = top-1 result.
        sparse_results: Output of BM25Index.query() — list of dicts, each
                        with 'chunk' (dict, must contain 'chunk_id' or 'id')
                        + 'score'. Order matters.
        k:              Max fused results to return.
        rrf_k:          RRF smoothing constant. 60 is standard; smaller
                        values reward top-ranked items more.

    Returns:
        List of fused result dicts::

            [
                {
                    "chunk_id": str,      # unique identifier
                    "score":    float,    # combined RRF score
                    "payload":  dict,     # chunk metadata (page, content, etc.)
                    "sources":  list[str] # which list(s) surfaced it: 'dense'|'sparse'
                },
                ...
            ]

        Sorted descending by fused score. Length ≤ k.
    """
    if not dense_results and not sparse_results:
        return []

    # ── accumulate RRF contributions keyed by chunk_id ────────────────────
    scores: dict[str, float] = defaultdict(float)
    payloads: dict[str, dict] = {}
    sources: dict[str, list[str]] = defaultdict(list)

    for rank, result in enumerate(dense_results):
        chunk_id = _extract_chunk_id(result, "dense", rank)
        scores[chunk_id] += 1.0 / (rrf_k + rank + 1)
        # Prefer whichever list gave us a payload first
        if chunk_id not in payloads:
            payloads[chunk_id] = result.get("payload", {})
        sources[chunk_id].append("dense")

    for rank, result in enumerate(sparse_results):
        # BM25 returns {'chunk': {...}, 'score': ...} — chunk dict is our payload
        chunk = result.get("chunk", {})
        chunk_id = _extract_chunk_id(chunk, "sparse", rank)
        scores[chunk_id] += 1.0 / (rrf_k + rank + 1)
        if chunk_id not in payloads:
            payloads[chunk_id] = chunk
        sources[chunk_id].append("sparse")

    # ── sort by fused score, take top k ────────────────────────────────────
    ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)[:k]

    fused = [
        {
            "chunk_id": chunk_id,
            "score": score,
            "payload": payloads[chunk_id],
            "sources": sources[chunk_id],
        }
        for chunk_id, score in ranked
    ]

    logger.info(
        "Fusion: %d dense + %d sparse → %d unique → top %d returned.",
        len(dense_results),
        len(sparse_results),
        len(scores),
        len(fused),
    )
    return fused
