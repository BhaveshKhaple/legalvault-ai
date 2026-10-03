"""
Task 2.2 — Qdrant vector store (embedded mode) + Task 2.3 case isolation.
embedding-dispatcher — Extended to support two named collections:

    legalvault_e5   (384-dim)  — intfloat/e5-small-v2
    legalvault_bge  (1024-dim) — BAAI/bge-m3

Each collection is created on first use. The public API takes an optional
`collection_name` parameter. Legacy callers that omit it get the e5 collection
(default tier = e5-small-v2, same as before the dispatcher).

Security (Task 2.3):
    Every vector is tagged with case_id at insert time. Every query MUST
    include a case_id filter — server-side, never trust-the-client.
    Lawyer A's documents must never appear in Lawyer B's results, even if
    the query text perfectly matches.

Distance metric: Cosine on all collections.
"""

import logging
import os
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Collection names — one per embedding tier
COLLECTION_E5 = "legalvault_e5"
COLLECTION_BGE = "legalvault_bge"

_COLLECTION_DIMS = {
    COLLECTION_E5: 384,
    COLLECTION_BGE: 1024,
}

# Default collection (backwards-compatible with code that doesn't pass collection_name)
_DEFAULT_COLLECTION = COLLECTION_E5

# Module-level singleton — one client for the process lifetime.
_client: Any = None


def _get_client() -> Any:
    """Return the cached QdrantClient, creating it on first call."""
    global _client
    if _client is None:
        from qdrant_client import QdrantClient

        qdrant_path = os.environ.get("QDRANT_PATH", "./data/qdrant")
        Path(qdrant_path).mkdir(parents=True, exist_ok=True)
        logger.info("Opening embedded Qdrant at '%s'.", qdrant_path)
        _client = QdrantClient(path=qdrant_path)
    return _client


def _ensure_collection(client: Any, collection_name: str) -> None:
    """Create a collection if it doesn't exist yet."""
    from qdrant_client.models import Distance, VectorParams

    dim = _COLLECTION_DIMS.get(collection_name)
    if dim is None:
        raise ValueError(
            f"Unknown collection '{collection_name}'. "
            f"Valid: {list(_COLLECTION_DIMS.keys())}"
        )
    existing = {c.name for c in client.get_collections().collections}
    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        logger.info("Created Qdrant collection '%s' (dim=%d).", collection_name, dim)
    else:
        logger.debug("Qdrant collection '%s' already exists.", collection_name)


def _get_or_create(collection_name: str) -> Any:
    client = _get_client()
    _ensure_collection(client, collection_name)
    return client


# ─── public API ───────────────────────────────────────────────────────────────


def insert(
    vectors: list[list[float]],
    payloads: list[dict],
    collection_name: str = _DEFAULT_COLLECTION,
) -> list[str]:
    """Insert vectors into a Qdrant collection.

    Each payload MUST contain 'case_id' (str) and 'doc_id' (str).
    The collection name is stamped into every payload under
    'embedding_collection' so the query path knows the origin.

    Args:
        vectors:         List of float embeddings.
        payloads:        One dict per vector. Required: case_id, doc_id.
        collection_name: Which collection to write to (default: legalvault_e5).

    Returns:
        List of point UUIDs (strings), same order as input.
    """
    if len(vectors) != len(payloads):
        raise ValueError(
            f"vectors ({len(vectors)}) and payloads ({len(payloads)}) must have the same length."
        )
    for i, p in enumerate(payloads):
        if "case_id" not in p or "doc_id" not in p:
            raise ValueError(f"Payload[{i}] missing 'case_id' or 'doc_id'.")

    from qdrant_client.models import PointStruct

    client = _get_or_create(collection_name)
    ids = [str(uuid.uuid4()) for _ in vectors]

    for p in payloads:
        p.setdefault("embedding_collection", collection_name)

    points = [
        PointStruct(id=pid, vector=vec, payload=payload)
        for pid, vec, payload in zip(ids, vectors, payloads)
    ]
    client.upsert(collection_name=collection_name, points=points)
    logger.info("Inserted %d vectors into '%s'.", len(points), collection_name)
    return ids


def query(
    vector: list[float],
    case_id: str,
    top_k: int = 30,
    collection_name: str = _DEFAULT_COLLECTION,
) -> list[dict]:
    """Search a collection for the top-k vectors matching case_id.

    Returns empty list if the collection doesn't exist yet — callers that
    fan-out across both collections should handle this gracefully.

    Returns:
        List of {'id', 'score', 'payload'} sorted descending by score.
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    client = _get_client()

    existing = {c.name for c in client.get_collections().collections}
    if collection_name not in existing:
        logger.debug("Collection '%s' not found — returning empty results.", collection_name)
        return []

    case_filter = Filter(
        must=[FieldCondition(key="case_id", match=MatchValue(value=case_id))]
    )

    response = client.query_points(
        collection_name=collection_name,
        query=vector,
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
        for r in response.points
    ]


def query_all_collections(
    vectors_by_collection: dict,
    case_id: str,
    top_k: int = 30,
) -> list[dict]:
    """Fan-out query across multiple collections and merge by score.

    Used when a case has docs in both legalvault_e5 and legalvault_bge.
    Each collection gets its own correctly-dimensioned query vector.

    Args:
        vectors_by_collection: {collection_name: query_vector}
        case_id:               Security filter applied to all collections.
        top_k:                 Max results per collection before merge.

    Returns:
        Merged list sorted descending by score. Caller passes this to
        the hybrid RRF fusion step alongside BM25 results.
    """
    all_results = []
    for coll, vec in vectors_by_collection.items():
        results = query(vec, case_id, top_k=top_k, collection_name=coll)
        all_results.extend(results)
    all_results.sort(key=lambda r: r["score"], reverse=True)
    return all_results


def delete_by_doc(
    doc_id: str,
    collection_name: str = None,
) -> int:
    """Remove all vectors for a document.

    If collection_name is None, deletes from ALL known collections
    (safe for legacy docs that don't track which collection was used).

    Returns total points deleted.
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    client = _get_client()
    existing = {c.name for c in client.get_collections().collections}
    doc_filter = Filter(
        must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
    )

    targets = [collection_name] if collection_name else list(_COLLECTION_DIMS.keys())
    total_deleted = 0

    for coll in targets:
        if coll not in existing:
            continue
        count = client.count(
            collection_name=coll,
            count_filter=doc_filter,
            exact=True,
        ).count
        if count:
            client.delete(collection_name=coll, points_selector=doc_filter)
            logger.info("Purged %d vectors for doc='%s' from '%s'.", count, doc_id, coll)
            total_deleted += count

    return total_deleted
