"""
Tests for Task 2.1 — embeddings.embed() and embed_batch().

SentenceTransformer model loading (~5–15s, ~500MB download) is mocked so
the suite runs in CI without network access or GPU.

The mock returns vectors of the correct dimension (1024) using numpy zeros/ones
so downstream shape checks are realistic.
"""

import logging
from unittest.mock import MagicMock, patch

import pytest

from backend.app.retrieval.embeddings import embed, embed_batch, _EXPECTED_DIM

# ─── shared mock setup ───────────────────────────────────────────────────────


def _make_fake_model(n_dims: int = _EXPECTED_DIM):
    """Return a mock SentenceTransformer whose encode() returns plausible arrays."""
    import numpy as np

    fake = MagicMock()

    def _encode(texts, **kwargs):
        # Single string → 1D array; list → 2D array
        if isinstance(texts, str):
            return np.zeros(n_dims, dtype="float32")
        return np.zeros((len(texts), n_dims), dtype="float32")

    fake.encode.side_effect = _encode
    return fake


@pytest.fixture(autouse=True)
def _patch_model(monkeypatch):
    """Replace _get_model globally for every test in this file."""
    fake = _make_fake_model()
    monkeypatch.setattr(
        "backend.app.retrieval.embeddings._get_model", lambda name: fake
    )
    return fake


# ─── embed() ─────────────────────────────────────────────────────────────────


class TestEmbed:
    def test_returns_list(self):
        result = embed("Clause 3.1 termination")
        assert isinstance(result, list)

    def test_correct_dimension(self):
        result = embed("any text")
        assert len(result) == _EXPECTED_DIM

    def test_elements_are_floats(self):
        result = embed("test")
        assert all(isinstance(v, float) for v in result)

    def test_empty_string_accepted(self):
        # Edge case: model should still return a vector
        result = embed("")
        assert len(result) == _EXPECTED_DIM

    def test_long_text_accepted(self):
        long_text = "contract term " * 500
        result = embed(long_text)
        assert len(result) == _EXPECTED_DIM

    def test_latency_logged(self, caplog):
        with caplog.at_level(logging.DEBUG, logger="backend.app.retrieval.embeddings"):
            embed("test latency")
        assert any("latency" in r.message.lower() for r in caplog.records)


# ─── embed_batch() ───────────────────────────────────────────────────────────


class TestEmbedBatch:
    def test_returns_list_of_lists(self):
        result = embed_batch(["clause a", "clause b", "clause c"])
        assert isinstance(result, list)
        for vec in result:
            assert isinstance(vec, list)

    def test_correct_output_count(self):
        texts = ["a", "b", "c", "d", "e"]
        result = embed_batch(texts)
        assert len(result) == len(texts)

    def test_each_vector_correct_dim(self):
        result = embed_batch(["x", "y"])
        for vec in result:
            assert len(vec) == _EXPECTED_DIM

    def test_empty_list_returns_empty(self):
        result = embed_batch([])
        assert result == []

    def test_single_item_batch(self):
        result = embed_batch(["only one"])
        assert len(result) == 1
        assert len(result[0]) == _EXPECTED_DIM

    def test_large_batch_accepted(self):
        texts = [f"chunk {i}" for i in range(200)]
        result = embed_batch(texts)
        assert len(result) == 200

    def test_latency_logged_at_info(self, caplog):
        with caplog.at_level(logging.INFO, logger="backend.app.retrieval.embeddings"):
            embed_batch(["a", "b"])
        assert any("embed_batch" in r.message for r in caplog.records)

    def test_order_preserved(self):
        """Each output vector must correspond to its input index (mocked as zeros
        so the real check is count + per-position type, not values)."""
        texts = ["alpha", "beta", "gamma"]
        result = embed_batch(texts)
        # We can't check semantic order with mocks, but length and dim must align
        assert [len(v) for v in result] == [_EXPECTED_DIM] * len(texts)
