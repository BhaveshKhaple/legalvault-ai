"""
Task 2.1 — BGE-M3 embedding wrapper.

Turns text into 1024-dimensional dense vectors so semantically similar
clauses end up geometrically close in vector space. BGE-M3 is chosen for:
  - Top MTEB score (multilingual — Hindi + English).
  - 8192-token context window (handles long clauses without truncation).
  - Apache licence (redistribution-safe).
  - 1024-dim output — strong nuance, reasonable storage cost.

The model is loaded ONCE per process (module-level singleton keyed by
model name) and reused across all calls. First call takes ~5–15s; all
subsequent calls are fast.

Latency is logged to stdout on every call so the dev can see if something
is unexpectedly slow.

Usage:
    from backend.app.retrieval.embeddings import embed, embed_batch

    vec  = embed("Clause 3.1 termination clause...")        # → list[float] len 1024
    vecs = embed_batch(["clause a", "clause b", ...])       # → list[list[float]]
"""

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# Tier config (Task 4.2 will add model_selector.py that drives this).
# Hardcoded default here; override via MODEL_NAME env var if needed.
_DEFAULT_MODEL = "BAAI/bge-m3"
_EXPECTED_DIM = 1024

# Module-level singleton — keyed by model name so switching models
# in tests doesn't pollute the cache.
_models: dict[str, Any] = {}


def _get_model(model_name: str) -> Any:
    """Load model once; return cached instance on subsequent calls."""
    if model_name not in _models:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        logger.info(
            "Loading embedding model '%s' — first load takes ~5–15s.", model_name
        )
        t0 = time.perf_counter()
        _models[model_name] = SentenceTransformer(model_name)
        logger.info(
            "Model '%s' loaded in %.1fs.", model_name, time.perf_counter() - t0
        )
    return _models[model_name]


def embed(text: str, model_name: str = _DEFAULT_MODEL) -> list[float]:
    """Embed a single text string.

    Args:
        text: The text to embed. Can be up to 8192 tokens for BGE-M3.
        model_name: HuggingFace model ID. Defaults to BAAI/bge-m3.

    Returns:
        List of 1024 floats (dense vector).
    """
    t0 = time.perf_counter()
    model = _get_model(model_name)
    vec = model.encode(text, normalize_embeddings=True).tolist()
    logger.debug("embed() latency: %.1fms", (time.perf_counter() - t0) * 1000)
    return vec


def embed_batch(
    texts: list[str],
    model_name: str = _DEFAULT_MODEL,
    batch_size: int = 32,
) -> list[list[float]]:
    """Embed a list of texts in one batched forward pass.

    ~10× faster than calling embed() in a loop for large lists.

    Args:
        texts: List of texts to embed.
        model_name: HuggingFace model ID. Defaults to BAAI/bge-m3.
        batch_size: Sentences per forward pass. Tune to VRAM/RAM.

    Returns:
        List of vectors, same order as input. Each vector is 1024 floats.
    """
    if not texts:
        return []

    t0 = time.perf_counter()
    model = _get_model(model_name)
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "embed_batch() — %d texts in %.1fms (%.1fms/text).",
        len(texts),
        elapsed_ms,
        elapsed_ms / len(texts),
    )
    return vecs
