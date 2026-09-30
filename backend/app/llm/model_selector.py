"""
Task 4.2 — Tiered model configuration.

One env variable MODEL_TIER switches the whole stack: embedding model,
embedding dimension, LLM model name, and vector store backend.

Accepted values: "1", "2", "3" (numeric tiers) or named tiers such as
"bhavesh". All tier_* keys in configs/tiers.yaml are loaded automatically.

The tier config is loaded ONCE at module import time from configs/tiers.yaml.
No hot-swap during a running instance — change MODEL_TIER and restart.

Design: a dead-simple factory (get_model()) + dataclass (TierConfig). No
abstract base classes, no plugins, no dependency injection.

See also: docs/TRD.md §11 for hardware requirements per tier.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# ─── tier config dataclass ────────────────────────────────────────────────────


@dataclass(frozen=True)
class TierConfig:
    tier: int | str
    description: str
    embedding: str       # HuggingFace model ID for sentence-transformers
    embedding_dim: int   # output dimension of the embedding model
    llm: str             # Ollama model name (passed to ollama_client.generate)
    vector_store: str    # 'qdrant' | 'pgvector'


# ─── load tiers from YAML ────────────────────────────────────────────────────

def _load_tiers(yaml_path: Path) -> dict[int | str, TierConfig]:
    """Parse all tier_* keys from tiers.yaml into a dict keyed by tier id.

    Numeric suffixes (tier_1, tier_2, tier_3) produce integer keys.
    Named suffixes (tier_bhavesh) produce string keys.
    """
    with yaml_path.open() as f:
        raw = yaml.safe_load(f)

    tiers: dict[int | str, TierConfig] = {}
    for yaml_key, entry in raw.items():
        if not yaml_key.startswith("tier_"):
            continue
        suffix = yaml_key[len("tier_"):]
        tier_key: int | str = int(suffix) if suffix.isdigit() else suffix
        tiers[tier_key] = TierConfig(
            tier=tier_key,
            description=entry["description"],
            embedding=entry["embedding"],
            embedding_dim=entry["embedding_dim"],
            llm=entry["llm"],
            vector_store=entry["vector_store"],
        )
    return tiers


# Resolve configs/tiers.yaml relative to this file (backend/app/llm/ → repo root)
_YAML_PATH = Path(__file__).parents[3] / "configs" / "tiers.yaml"
_TIERS: dict[int | str, TierConfig] = _load_tiers(_YAML_PATH)


# ─── public API ───────────────────────────────────────────────────────────────


def get_model() -> TierConfig:
    """Return the TierConfig for the current MODEL_TIER env variable.

    Reads MODEL_TIER on every call so tests can change it via monkeypatch
    without reloading the module.

    Accepted values: numeric strings "1"/"2"/"3" or named tier keys such
    as "bhavesh". All tier_* entries in configs/tiers.yaml are valid.

    Returns:
        TierConfig for the active tier.

    Raises:
        ValueError: If MODEL_TIER is not a recognised tier key.
    """
    raw = os.environ.get("MODEL_TIER", "2")

    # Numeric string → int key for backward compat; otherwise string key.
    try:
        key: int | str = int(raw)
    except ValueError:
        key = raw

    if key not in _TIERS:
        valid = ", ".join(str(k) for k in _TIERS)
        raise ValueError(
            f"MODEL_TIER='{raw}' is not valid. Valid values: {valid}."
        )

    config = _TIERS[key]
    logger.debug(
        "Active tier: %s (%s) | llm=%s | embedding=%s (dim=%d) | vector_store=%s",
        config.tier,
        config.description,
        config.llm,
        config.embedding,
        config.embedding_dim,
        config.vector_store,
    )
    return config
