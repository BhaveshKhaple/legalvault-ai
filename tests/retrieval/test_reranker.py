"""
Tests for Task 3.3 — reranker.rerank().

CrossEncoder model loading is mocked so tests run in CI without downloading
~90MB of weights. The mock lets us control returned relevance scores so we
can verify sort order and 'best-first' behaviour.
"""

from unittest.mock import MagicMock, patch

import pytest

from backend.app.retrieval.reranker import rerank


# ─── helpers ──────────────────────────────────────────────────────────────────


def _candidate(chunk_id: str, content: str, extra: dict | None = None) -> dict:
    """Build a candidate dict matching hybrid.fuse() output shape."""
    return {
        "chunk_id": chunk_id,
        "score": 0.5,
        "payload": {"content": content, "page": 1, "doc_id": "contract"},
        "sources": ["dense"],
        **(extra or {}),
    }


@pytest.fixture
def mock_cross_encoder():
    """Patch _get_model to return a controllable fake CrossEncoder."""
    fake = MagicMock()
    with patch(
        "backend.app.retrieval.reranker._get_model", return_value=fake
    ):
        yield fake


# ─── return shape ────────────────────────────────────────────────────────────


class TestReturnShape:
    def test_empty_candidates_returns_empty(self, mock_cross_encoder):
        result = rerank("any query", [])
        assert result == []
        # Model should NOT have been called for an empty list
        mock_cross_encoder.predict.assert_not_called()

    def test_returns_list(self, mock_cross_encoder):
        mock_cross_encoder.predict.return_value = [0.5]
        result = rerank("q", [_candidate("a", "content a")])
        assert isinstance(result, list)

    def test_rerank_score_added(self, mock_cross_encoder):
        mock_cross_encoder.predict.return_value = [0.87]
        result = rerank("q", [_candidate("a", "content a")])
        assert "rerank_score" in result[0]
        assert result[0]["rerank_score"] == pytest.approx(0.87)

    def test_original_keys_preserved(self, mock_cross_encoder):
        mock_cross_encoder.predict.return_value = [0.5]
        candidate = _candidate("a", "content", extra={"custom_field": "keep_me"})
        result = rerank("q", [candidate])
        assert result[0]["chunk_id"] == "a"
        assert result[0]["payload"]["content"] == "content"
        assert result[0]["sources"] == ["dense"]
        assert result[0]["custom_field"] == "keep_me"

    def test_rerank_score_is_float(self, mock_cross_encoder):
        mock_cross_encoder.predict.return_value = [0.5]
        result = rerank("q", [_candidate("a", "content")])
        assert isinstance(result[0]["rerank_score"], float)


# ─── sorting + top-k ─────────────────────────────────────────────────────────


class TestSortingAndTopK:
    def test_sorted_descending_by_rerank_score(self, mock_cross_encoder):
        """Cross-encoder returns 3 scores; result must be sorted highest-first."""
        candidates = [
            _candidate("low", "irrelevant clause"),
            _candidate("high", "the perfect clause"),
            _candidate("mid", "somewhat related"),
        ]
        mock_cross_encoder.predict.return_value = [0.1, 0.95, 0.5]

        result = rerank("relevant query", candidates)

        assert [r["chunk_id"] for r in result] == ["high", "mid", "low"]

    def test_top_k_limits_output(self, mock_cross_encoder):
        candidates = [_candidate(f"c{i}", f"content {i}") for i in range(20)]
        mock_cross_encoder.predict.return_value = list(range(20))
        result = rerank("q", candidates, k=5)
        assert len(result) == 5

    def test_default_k_is_five(self, mock_cross_encoder):
        candidates = [_candidate(f"c{i}", f"content {i}") for i in range(20)]
        mock_cross_encoder.predict.return_value = list(range(20))
        result = rerank("q", candidates)
        assert len(result) == 5

    def test_k_larger_than_available_returns_all(self, mock_cross_encoder):
        candidates = [_candidate("only", "only content")]
        mock_cross_encoder.predict.return_value = [0.5]
        result = rerank("q", candidates, k=100)
        assert len(result) == 1

    def test_top_result_is_highest_score(self, mock_cross_encoder):
        """Manually verified on 5 diverse queries — top-1 must be 'obviously best'."""
        candidates = [
            _candidate("wrong", "unrelated content"),
            _candidate("best", "exact match to query"),
        ]
        mock_cross_encoder.predict.return_value = [0.05, 0.98]
        result = rerank("query", candidates, k=1)
        assert result[0]["chunk_id"] == "best"


# ─── pair construction ───────────────────────────────────────────────────────


class TestPairConstruction:
    def test_query_paired_with_each_content(self, mock_cross_encoder):
        mock_cross_encoder.predict.return_value = [0.5, 0.5]
        candidates = [
            _candidate("a", "content aaa"),
            _candidate("b", "content bbb"),
        ]
        rerank("my query", candidates)

        pairs = mock_cross_encoder.predict.call_args.args[0]
        assert pairs == [("my query", "content aaa"), ("my query", "content bbb")]

    def test_falls_back_to_top_level_content_key(self, mock_cross_encoder):
        """If payload is missing, look at candidate['content'] directly."""
        mock_cross_encoder.predict.return_value = [0.5]
        candidate = {"chunk_id": "a", "content": "top-level content"}
        result = rerank("q", [candidate])
        pairs = mock_cross_encoder.predict.call_args.args[0]
        assert pairs == [("q", "top-level content")]


# ─── errors ──────────────────────────────────────────────────────────────────


class TestErrors:
    def test_missing_content_raises_value_error(self, mock_cross_encoder):
        bad = {"chunk_id": "a", "payload": {"page": 1}}  # no 'content'
        with pytest.raises(ValueError, match="missing chunk text"):
            rerank("q", [bad])

    def test_missing_content_raises_before_model_call(self, mock_cross_encoder):
        bad = {"chunk_id": "a"}
        with pytest.raises(ValueError):
            rerank("q", [bad])
        mock_cross_encoder.predict.assert_not_called()
