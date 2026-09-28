"""Tests for Task 3.2 — hybrid.fuse()."""

import pytest

from backend.app.retrieval.hybrid import fuse


# ─── helpers ──────────────────────────────────────────────────────────────────


def _dense(chunk_id: str, payload_extra: dict | None = None) -> dict:
    """Build a dense-result dict matching vector_store.query() shape."""
    return {
        "id": chunk_id,
        "score": 0.9,
        "payload": {"content": f"content of {chunk_id}", **(payload_extra or {})},
    }


def _sparse(chunk_id: str, score: float = 5.0) -> dict:
    """Build a sparse-result dict matching BM25Index.query() shape."""
    return {
        "chunk": {"chunk_id": chunk_id, "content": f"content of {chunk_id}"},
        "score": score,
    }


# ─── happy path ──────────────────────────────────────────────────────────────


class TestFuseHappyPath:
    def test_empty_inputs_return_empty(self):
        assert fuse([], []) == []

    def test_dense_only(self):
        result = fuse([_dense("a"), _dense("b")], [])
        assert len(result) == 2
        assert {r["chunk_id"] for r in result} == {"a", "b"}

    def test_sparse_only(self):
        result = fuse([], [_sparse("x"), _sparse("y")])
        assert len(result) == 2
        assert {r["chunk_id"] for r in result} == {"x", "y"}

    def test_return_shape(self):
        result = fuse([_dense("a")], [_sparse("b")])
        assert set(result[0].keys()) == {"chunk_id", "score", "payload", "sources"}

    def test_scores_are_floats_and_positive(self):
        result = fuse([_dense("a")], [_sparse("b")])
        for r in result:
            assert isinstance(r["score"], float)
            assert r["score"] > 0

    def test_sorted_descending(self):
        # Item 'top' appears at rank 1 in both lists, others only once
        dense = [_dense("top"), _dense("d2"), _dense("d3")]
        sparse = [_sparse("top"), _sparse("s2"), _sparse("s3")]
        result = fuse(dense, sparse)
        scores = [r["score"] for r in result]
        assert scores == sorted(scores, reverse=True)
        assert result[0]["chunk_id"] == "top"


# ─── dedup ───────────────────────────────────────────────────────────────────


class TestDedup:
    def test_overlapping_chunk_appears_once(self):
        result = fuse([_dense("shared"), _dense("d_only")], [_sparse("shared"), _sparse("s_only")])
        ids = [r["chunk_id"] for r in result]
        assert ids.count("shared") == 1

    def test_overlap_scored_higher_than_singletons(self):
        result = fuse(
            [_dense("shared"), _dense("dense_only")],
            [_sparse("shared"), _sparse("sparse_only")],
        )
        by_id = {r["chunk_id"]: r["score"] for r in result}
        # 'shared' gets RRF contributions from BOTH lists at rank 1
        assert by_id["shared"] > by_id["dense_only"]
        assert by_id["shared"] > by_id["sparse_only"]

    def test_sources_field_reflects_lists(self):
        result = fuse(
            [_dense("shared"), _dense("dense_only")],
            [_sparse("shared"), _sparse("sparse_only")],
        )
        by_id = {r["chunk_id"]: r["sources"] for r in result}
        assert set(by_id["shared"]) == {"dense", "sparse"}
        assert by_id["dense_only"] == ["dense"]
        assert by_id["sparse_only"] == ["sparse"]


# ─── RRF math ────────────────────────────────────────────────────────────────


class TestRRFMath:
    def test_rrf_formula_dense_only(self):
        """Position 0 → 1/(60+1); position 1 → 1/(60+2)."""
        result = fuse([_dense("first"), _dense("second")], [], rrf_k=60)
        by_id = {r["chunk_id"]: r["score"] for r in result}
        assert by_id["first"] == pytest.approx(1 / 61)
        assert by_id["second"] == pytest.approx(1 / 62)

    def test_rrf_formula_combined(self):
        """A chunk at rank 0 in both lists gets 2/(60+1)."""
        result = fuse([_dense("x")], [_sparse("x")], rrf_k=60)
        assert result[0]["score"] == pytest.approx(2 / 61)

    def test_smaller_rrf_k_rewards_top_more(self):
        """Lower rrf_k → bigger gap between rank 1 and rank 2."""
        low = fuse([_dense("a"), _dense("b")], [], rrf_k=1)
        high = fuse([_dense("a"), _dense("b")], [], rrf_k=1000)
        low_gap = low[0]["score"] - low[1]["score"]
        high_gap = high[0]["score"] - high[1]["score"]
        assert low_gap > high_gap


# ─── k limit ────────────────────────────────────────────────────────────────


class TestKLimit:
    def test_k_caps_result_length(self):
        dense = [_dense(f"d{i}") for i in range(15)]
        result = fuse(dense, [], k=5)
        assert len(result) == 5

    def test_k_larger_than_available_returns_all(self):
        result = fuse([_dense("only")], [], k=100)
        assert len(result) == 1


# ─── payload / identity ─────────────────────────────────────────────────────


class TestPayloadIdentity:
    def test_payload_preserved(self):
        result = fuse(
            [_dense("a", payload_extra={"page": 5, "doc_id": "contract"})],
            [],
        )
        assert result[0]["payload"]["page"] == 5
        assert result[0]["payload"]["doc_id"] == "contract"

    def test_chunk_id_key_takes_priority_over_id(self):
        weird = {"id": "wrong", "chunk_id": "right", "payload": {}}
        result = fuse([weird], [])
        assert result[0]["chunk_id"] == "right"


# ─── validation ─────────────────────────────────────────────────────────────


class TestValidation:
    def test_dense_missing_id_raises(self):
        with pytest.raises(ValueError, match="chunk_id.*id"):
            fuse([{"payload": {}}], [])

    def test_sparse_chunk_missing_id_raises(self):
        with pytest.raises(ValueError, match="chunk_id.*id"):
            fuse([], [{"chunk": {"content": "no id here"}, "score": 1.0}])
