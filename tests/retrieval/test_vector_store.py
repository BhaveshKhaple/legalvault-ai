"""
Tests for Tasks 2.2 + 2.3 — vector_store.insert(), query(), delete_by_doc().

QdrantClient is mocked so tests run without Qdrant installed or on disk.
Case isolation (Task 2.3) is explicitly tested: queries for case A must
never return results tagged case B.
"""

from unittest.mock import MagicMock, patch, call
import pytest

import backend.app.retrieval.vector_store as vs_module
from backend.app.retrieval.vector_store import insert, query, delete_by_doc


# ─── fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def reset_client():
    """Reset the module-level singleton before every test."""
    original = vs_module._client
    vs_module._client = None
    yield
    vs_module._client = original


def _fake_client():
    """Build a mock QdrantClient that passes all basic sanity checks."""
    client = MagicMock()
    client.get_collections.return_value.collections = []
    return client


def _make_vec(val: float = 0.1, size: int = 1024) -> list[float]:
    return [val] * size


def _hit(point_id: str, score: float, payload: dict) -> MagicMock:
    h = MagicMock()
    h.id = point_id
    h.score = score
    h.payload = payload
    return h


# ─── insert ──────────────────────────────────────────────────────────────────


class TestInsert:
    def test_returns_ids(self):
        with patch("backend.app.retrieval.vector_store._get_client", return_value=_fake_client()):
            ids = insert(
                vectors=[_make_vec()],
                payloads=[{"case_id": "cA", "doc_id": "d1", "content": "hello"}],
            )
        assert len(ids) == 1
        assert isinstance(ids[0], str)

    def test_multiple_vectors(self):
        with patch("backend.app.retrieval.vector_store._get_client", return_value=_fake_client()):
            ids = insert(
                vectors=[_make_vec(), _make_vec(0.2), _make_vec(0.3)],
                payloads=[
                    {"case_id": "cA", "doc_id": "d1"},
                    {"case_id": "cA", "doc_id": "d1"},
                    {"case_id": "cA", "doc_id": "d1"},
                ],
            )
        assert len(ids) == 3
        assert len(set(ids)) == 3  # all unique UUIDs

    def test_ids_are_uuid_strings(self):
        import uuid
        with patch("backend.app.retrieval.vector_store._get_client", return_value=_fake_client()):
            ids = insert([_make_vec()], [{"case_id": "c1", "doc_id": "d1"}])
        uuid.UUID(ids[0])  # raises if not valid UUID

    def test_length_mismatch_raises(self):
        with patch("backend.app.retrieval.vector_store._get_client", return_value=_fake_client()):
            with pytest.raises(ValueError, match="same length"):
                insert(
                    vectors=[_make_vec(), _make_vec()],
                    payloads=[{"case_id": "c1", "doc_id": "d1"}],
                )

    def test_missing_case_id_raises(self):
        with patch("backend.app.retrieval.vector_store._get_client", return_value=_fake_client()):
            with pytest.raises(ValueError, match="case_id"):
                insert([_make_vec()], [{"doc_id": "d1"}])

    def test_missing_doc_id_raises(self):
        with patch("backend.app.retrieval.vector_store._get_client", return_value=_fake_client()):
            with pytest.raises(ValueError, match="doc_id"):
                insert([_make_vec()], [{"case_id": "c1"}])

    def test_upsert_called_with_points(self):
        client = _fake_client()
        with patch("backend.app.retrieval.vector_store._get_client", return_value=client):
            insert([_make_vec()], [{"case_id": "cA", "doc_id": "d1"}])
        assert client.upsert.call_count == 1
        call_kwargs = client.upsert.call_args
        assert call_kwargs.kwargs["collection_name"] == "legalvault_chunks"


# ─── query ────────────────────────────────────────────────────────────────────


class TestQuery:
    def test_returns_list(self):
        client = _fake_client()
        client.search.return_value = []
        with patch("backend.app.retrieval.vector_store._get_client", return_value=client):
            result = query(_make_vec(), case_id="cA")
        assert result == []

    def test_result_dict_shape(self):
        client = _fake_client()
        client.search.return_value = [
            _hit("abc-123", 0.92, {"case_id": "cA", "content": "penalty clause"}),
        ]
        with patch("backend.app.retrieval.vector_store._get_client", return_value=client):
            result = query(_make_vec(), case_id="cA")
        assert len(result) == 1
        r = result[0]
        assert set(r.keys()) == {"id", "score", "payload"}
        assert r["id"] == "abc-123"
        assert r["score"] == pytest.approx(0.92)
        assert r["payload"]["content"] == "penalty clause"

    def test_case_filter_passed_to_client(self):
        """SECURITY: case_id filter must always be sent to the server."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        client = _fake_client()
        client.search.return_value = []
        with patch("backend.app.retrieval.vector_store._get_client", return_value=client):
            query(_make_vec(), case_id="secret_case_B")

        call_kwargs = client.search.call_args.kwargs
        f = call_kwargs["query_filter"]
        assert isinstance(f, Filter)
        condition = f.must[0]
        assert condition.key == "case_id"
        assert condition.match.value == "secret_case_B"

    def test_case_a_results_never_from_case_b(self):
        """Task 2.3 isolation: a query for case A must only show case A results."""
        client = _fake_client()
        # Simulate server correctly filtering — only case_A payload returned
        client.search.return_value = [
            _hit("id-1", 0.9, {"case_id": "case_A", "content": "A doc"}),
        ]
        with patch("backend.app.retrieval.vector_store._get_client", return_value=client):
            result = query(_make_vec(), case_id="case_A")
        for r in result:
            assert r["payload"]["case_id"] == "case_A"

    def test_top_k_passed(self):
        client = _fake_client()
        client.search.return_value = []
        with patch("backend.app.retrieval.vector_store._get_client", return_value=client):
            query(_make_vec(), case_id="cA", top_k=5)
        assert client.search.call_args.kwargs["limit"] == 5

    def test_empty_result_returns_empty_list(self):
        client = _fake_client()
        client.search.return_value = []
        with patch("backend.app.retrieval.vector_store._get_client", return_value=client):
            assert query(_make_vec(), case_id="cA") == []


# ─── delete_by_doc ───────────────────────────────────────────────────────────


class TestDeleteByDoc:
    def test_delete_called(self):
        client = _fake_client()
        client.delete.return_value = MagicMock(deleted=3)
        with patch("backend.app.retrieval.vector_store._get_client", return_value=client):
            n = delete_by_doc("doc_xyz")
        assert client.delete.call_count == 1
        call_kwargs = client.delete.call_args.kwargs
        assert call_kwargs["collection_name"] == "legalvault_chunks"

    def test_returns_deleted_count(self):
        client = _fake_client()
        client.delete.return_value = MagicMock(deleted=7)
        with patch("backend.app.retrieval.vector_store._get_client", return_value=client):
            n = delete_by_doc("doc_to_remove")
        assert n == 7
