"""
Tests for Task 3.1 — bm25_index.build_bm25() + BM25Index.query().

rank_bm25 is pure Python — no mocks needed.
Performance test: query must return in < 100ms on a 1000-chunk corpus.
"""

import time

import pytest

from backend.app.retrieval.bm25_index import BM25Index, build_bm25


# ─── helpers ──────────────────────────────────────────────────────────────────


def _make_chunks(texts: list[str], doc_id: str = "contract") -> list[dict]:
    return [
        {
            "content": t,
            "page": i + 1,
            "section_title": f"Clause {i + 1}",
            "doc_id": doc_id,
        }
        for i, t in enumerate(texts)
    ]


# ─── build_bm25 ──────────────────────────────────────────────────────────────


class TestBuildBm25:
    def test_returns_bm25index(self):
        index = build_bm25(_make_chunks(["some clause text"]))
        assert isinstance(index, BM25Index)

    def test_empty_raises_value_error(self):
        with pytest.raises(ValueError, match="empty"):
            build_bm25([])

    def test_preserves_chunk_count(self):
        chunks = _make_chunks(["a", "b", "c"])
        index = build_bm25(chunks)
        assert len(index._chunks) == 3


# ─── query — basic ──────────────────────────────────────────────────────────


class TestQueryBasic:
    def setup_method(self):
        chunks = _make_chunks(
            [
                "Clause 8.3 penalty interest charged at 2% per month.",
                "Indemnity clause holds party harmless from third-party claims.",
                "Governing law shall be the laws of Maharashtra India.",
                "Force majeure events include natural disasters and pandemics.",
                "Termination notice period is thirty 30 days written notice.",
            ]
        )
        self.index = build_bm25(chunks)

    def test_returns_list(self):
        result = self.index.query("penalty")
        assert isinstance(result, list)

    def test_result_dict_keys(self):
        result = self.index.query("penalty")
        assert len(result) >= 1
        assert set(result[0].keys()) == {"chunk", "score"}

    def test_chunk_is_original_dict(self):
        result = self.index.query("penalty")
        chunk = result[0]["chunk"]
        assert "content" in chunk and "page" in chunk and "doc_id" in chunk

    def test_score_is_float(self):
        result = self.index.query("penalty")
        assert isinstance(result[0]["score"], float)

    def test_relevant_chunk_top_ranked(self):
        """Chunk mentioning 'penalty' must appear before unrelated chunks."""
        result = self.index.query("penalty interest 2%")
        assert "penalty" in result[0]["chunk"]["content"].lower()

    def test_dotted_number_matched(self):
        """'8.3' must be preserved as a single token — exact clause reference."""
        result = self.index.query("8.3")
        assert len(result) >= 1
        assert "8.3" in result[0]["chunk"]["content"]

    def test_zero_score_chunks_excluded(self):
        """Chunks with BM25 score 0 (no term overlap) must not appear in results."""
        result = self.index.query("penalty")
        for r in result:
            assert r["score"] > 0.0

    def test_k_limits_results(self):
        chunks = _make_chunks([f"clause {i} penalty term" for i in range(20)])
        index = build_bm25(chunks)
        result = index.query("penalty", k=3)
        assert len(result) <= 3

    def test_empty_query_returns_empty(self):
        result = self.index.query("")
        assert result == []

    def test_whitespace_only_query_returns_empty(self):
        result = self.index.query("   ")
        assert result == []

    def test_sorted_descending_by_score(self):
        result = self.index.query("penalty interest clause")
        scores = [r["score"] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_no_match_returns_empty(self):
        result = self.index.query("xyzzy frobozz nonexistent")
        assert result == []


# ─── performance ─────────────────────────────────────────────────────────────


class TestPerformance:
    def test_query_under_100ms_on_1000_chunks(self):
        """Tracker requirement: top-K query < 100ms on 1000-chunk corpus."""
        chunks = _make_chunks(
            [
                f"This is clause {i}. It covers penalty interest rate {i}% per month "
                f"under section {i}.1 of the governing law."
                for i in range(1000)
            ]
        )
        index = build_bm25(chunks)
        t0 = time.perf_counter()
        index.query("penalty interest rate section", k=10)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert elapsed_ms < 100, f"BM25 query took {elapsed_ms:.1f}ms — target is <100ms"


# ─── multi-doc isolation ────────────────────────────────────────────────────


class TestMultiDoc:
    def test_doc_id_preserved_in_results(self):
        # BM25 IDF = log((N - df + 0.5) / (df + 0.5)). With 2 docs and df=1
        # this equals log(1) = 0. Need ≥3 docs for "penalty" to score > 0.
        chunks = [
            {"content": "penalty interest charged at 2% per month", "page": 1, "section_title": "S1", "doc_id": "doc_A"},
            {"content": "indemnity clause holds party harmless", "page": 2, "section_title": "S2", "doc_id": "doc_B"},
            {"content": "governing law shall be Maharashtra", "page": 3, "section_title": "S3", "doc_id": "doc_C"},
        ]
        index = build_bm25(chunks)
        result = index.query("penalty", k=5)
        assert len(result) >= 1
        assert result[0]["chunk"]["doc_id"] == "doc_A"
