"""
Task 4.2 — Tiered model configuration.

One env variable MODEL_TIER ∈ {1, 2, 3} switches the whole stack:
embedding model, LLM model name, and vector store backend.

The tier config is loaded ONCE at module import time from configs/tiers.yaml.
No hot-swap during a running instance — change MODEL_TIER and restart.

Design: a dead-simple factory (get_model()) + dataclass (TierConfig). No
abstract base classes, no plugins, no dependency injection. Three tiers is
not enough complexity to justify any of those patterns.

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
    tier: int
    description: str
    embedding: str      # HuggingFace model ID for sentence-transformers
    llm: str            # Ollama model name (passed to ollama_client.generate)
    vector_store: str   # 'qdrant' | 'pgvector'


# ─── load tiers from YAML ────────────────────────────────────────────────────

def _load_tiers(yaml_path: Path) -> dict[int, TierConfig]:
    """Parse tiers.yaml and return {1: TierConfig, 2: TierConfig, 3: TierConfig}."""
    with yaml_path.open() as f:
        raw = yaml.safe_load(f)

    tiers: dict[int, TierConfig] = {}
    for tier_num in (1, 2, 3):
        key = f"tier_{tier_num}"
        if key not in raw:
            raise ValueError(
                f"Missing '{key}' in {yaml_path}. "
                "tiers.yaml must define tier_1, tier_2, and tier_3."
            )
        entry = raw[key]
        tiers[tier_num] = TierConfig(
            tier=tier_num,
            description=entry["description"],
            embedding=entry["embedding"],
            llm=entry["llm"],
            vector_store=entry["vector_store"],
        )
    return tiers


# Resolve configs/tiers.yaml relative to this file (backend/app/llm/ → repo root)
_YAML_PATH = Path(__file__).parents[3] / "configs" / "tiers.yaml"
_TIERS: dict[int, TierConfig] = _load_tiers(_YAML_PATH)


# ─── public API ───────────────────────────────────────────────────────────────


def get_model() -> TierConfig:
    """Return the TierConfig for the current MODEL_TIER env variable.

    Reads MODEL_TIER on every call so tests can change it via monkeypatch
    without reloading the module.

    Returns:
        TierConfig for the active tier.

    Raises:
        ValueError: If MODEL_TIER is set to something outside {1, 2, 3}.
    """
    raw = os.environ.get("MODEL_TIER", "2")
    try:
        tier_num = int(raw)
    except ValueError:
        raise ValueError(
            f"MODEL_TIER must be an integer (1, 2, or 3), got '{raw}'."
        )

    if tier_num not in _TIERS:
        raise ValueError(
            f"MODEL_TIER={tier_num} is not valid. Choose 1, 2, or 3."
        )

    config = _TIERS[tier_num]
    logger.debug(
        "Active tier: %d (%s) | llm=%s | embedding=%s | vector_store=%s",
        config.tier,
        config.description,
        config.llm,
        config.embedding,
        config.vector_store,
    )
    return config
