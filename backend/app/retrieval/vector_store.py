"""
Task 2.2 — Qdrant vector store (embedded mode) + Task 2.3 case isolation.

Qdrant runs INSIDE the Python process (no separate Docker container) and
persists vectors to disk at QDRANT_PATH. The same QdrantClient instance is
reused across calls via a module-level singleton.

Security (Task 2.3 requirement):
    Every vector is tagged with case_id at insert time. Every query MUST
    include a case_id filter — server-side, never trust-the-client.
    Lawyer A's documents must never appear in Lawyer B's results, even if
    the query text perfectly matches.

Vector dimension: 1024 (matches BGE-M3 embed() output from Task 2.1).
Distance metric: Cosine (equivalent to dot-product on L2-normalised vecs).
"""

import logging
import os
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_COLLECTION = "legalvault_chunks"
_VECTOR_SIZE = 1024

# Module-level singleton — one client for the process lifetime.
_client: Any = None


def _get_client() -> Any:
    """Return the cached QdrantClient, creating it on first call."""
    global _client
    if _client is None:
        from qdrant_client import QdrantClient  # noqa: PLC0415

        qdrant_path = os.environ.get("QDRANT_PATH", "./data/qdrant")
        Path(qdrant_path).mkdir(parents=True, exist_ok=True)
        logger.info("Opening embedded Qdrant at '%s'.", qdrant_path)
        _client = QdrantClient(path=qdrant_path)
        _ensure_collection(_client)
    return _client


def _ensure_collection(client: Any) -> None:
    """Create the collection if it doesn't exist yet."""
    from qdrant_client.models import Distance, VectorParams  # noqa: PLC0415

    existing = {c.name for c in client.get_collections().collections}
    if _COLLECTION not in existing:
        client.create_collection(
            collection_name=_COLLECTION,
            vectors_config=VectorParams(size=_VECTOR_SIZE, distance=Distance.COSINE),
        )
        logger.info("Created Qdrant collection '%s' (dim=%d).", _COLLECTION, _VECTOR_SIZE)
    else:
        logger.debug("Qdrant collection '%s' already exists.", _COLLECTION)


# ─── public API ───────────────────────────────────────────────────────────────


def insert(
    vectors: list[list[float]],
    payloads: list[dict],
) -> list[str]:
    """Insert vectors into Qdrant with associated metadata payloads.

    Each payload MUST contain 'case_id' (str) and 'doc_id' (str).
    The payload is stored verbatim and returned at query time.

    Args:
        vectors:  List of 1024-float embeddings (output of embed_batch()).
        payloads: List of dicts, one per vector. Required keys:
                  - case_id: str — security boundary (e.g. UUID of the case)
                  - doc_id:  str — which document the chunk came from
                  Recommended keys (for citation later):
                  - page: int | None
                  - ts_start: float | None
                  - section_title: str | None
                  - content: str  (chunk text)

    Returns:
        List of Qdrant point IDs (UUIDs as strings), same order as input.

    Raises:
        ValueError: If vectors and payloads lengths differ, or if any
                    payload is missing 'case_id' or 'doc_id'.
    """
    if len(vectors) != len(payloads):
        raise ValueError(
            f"vectors ({len(vectors)}) and payloads ({len(payloads)}) must have the same length."
        )
    for i, p in enumerate(payloads):
        if "case_id" not in p or "doc_id" not in p:
            raise ValueError(f"Payload at index {i} must contain 'case_id' and 'doc_id'.")

    from qdrant_client.models import PointStruct  # noqa: PLC0415

    client = _get_client()
    ids = [str(uuid.uuid4()) for _ in vectors]
    points = [
        PointStruct(id=point_id, vector=vec, payload=payload)
        for point_id, vec, payload in zip(ids, vectors, payloads)
    ]
    client.upsert(collection_name=_COLLECTION, points=points)
    logger.info("Inserted %d vectors into Qdrant.", len(points))
    return ids


def query(
    vector: list[float],
    case_id: str,
    top_k: int = 30,
) -> list[dict]:
    """Search for the top-k nearest vectors, filtered by case_id.

    The case_id filter is MANDATORY and applied server-side. This is the
    security boundary that isolates tenants.

    Args:
        vector:  Query embedding (1024 floats from embed()).
        case_id: The case whose corpus to search. Cross-case results are
                 impossible by design.
        top_k:   Maximum neighbours to return.

    Returns:
        List of result dicts::

            [
                {
                    "id":      str,    # Qdrant point ID
                    "score":   float,  # cosine similarity (0–1)
                    "payload": dict,   # original metadata dict
                },
                ...
            ]

        Sorted descending by score. May be fewer than top_k if the case
        has fewer indexed chunks.
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue  # noqa: PLC0415

    client = _get_client()

    case_filter = Filter(
        must=[FieldCondition(key="case_id", match=MatchValue(value=case_id))]
    )

    results = client.search(
        collection_name=_COLLECTION,
        query_vector=vector,
        query_filter=case_filter,
        limit=top_k,
        with_payload=True,
    )

    return [
        {
            "id": str(r.id),
            "score": float(r.score),
            "payload": r.payload or {},
        }
        for r in results
    ]


def delete_by_doc(doc_id: str) -> int:
    """Remove all vectors belonging to a document (used when a doc is deleted).

    Args:
        doc_id: The document whose vectors to purge.

    Returns:
        Number of points deleted.
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue  # noqa: PLC0415

    client = _get_client()
    doc_filter = Filter(
        must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
    )
    result = client.delete(
        collection_name=_COLLECTION,
        points_selector=doc_filter,
    )
    deleted = getattr(result, "deleted", 0) or 0
    logger.info("Purged %d vectors for doc_id='%s'.", deleted, doc_id)
    return deleted
