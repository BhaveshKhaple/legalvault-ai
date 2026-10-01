"""Tests for Task 5.2 — Reverse-RAG claim verifier."""

import uuid
from unittest.mock import patch, MagicMock

import pytest

from backend.app.shield.verifier import verify_claim, verify_claims


def _mock_embed(text, model_name=None):
    return [0.1] * 1024


def _make_candidates(score=0.8, content="Penalty interest at 2% per month."):
    return [
        {
            "id": str(uuid.uuid4()),
            "score": score,
            "payload": {
                "content": content,
                "page": 8,
                "filename": "contract.pdf",
                "section_title": "8.3 Penalties",
                "case_id": "test-case",
                "doc_id": "test-doc",
            },
        }
    ]


@pytest.fixture(autouse=True)
def mock_pipeline(monkeypatch):
    """Mock all heavy dependencies so tests run without models."""
    monkeypatch.setattr(
        "backend.app.shield.verifier._embed_claim", _mock_embed
    )
    monkeypatch.setattr(
        "backend.app.shield.verifier._search_evidence",
        lambda vec, case_id, top_k=10: _make_candidates(),
    )

    def _fake_rerank(query, candidates, k=3):
        return [
            {**c, "rerank_score": 0.85}
            for c in candidates[:k]
        ]

    monkeypatch.setattr(
        "backend.app.shield.verifier._rerank_evidence", _fake_rerank
    )
    monkeypatch.setattr(
        "backend.app.shield.verifier._llm_support_or_contradict",
        lambda claim, texts, model: "Verified",
    )


class TestVerifyClaim:
    def test_returns_dict_with_required_keys(self):
        result = verify_claim("Penalty is 2% per month.", "case-123")
        assert "claim" in result
        assert "status" in result
        assert "evidence" in result

    def test_verified_when_evidence_above_threshold(self):
        result = verify_claim("Penalty is 2% per month.", "case-123")
        assert result["status"] == "Verified"

    def test_unverified_when_no_evidence(self, monkeypatch):
        monkeypatch.setattr(
            "backend.app.shield.verifier._search_evidence",
            lambda vec, case_id, top_k=10: [],
        )
        result = verify_claim("Penalty is 2% per month.", "case-123")
        assert result["status"] == "Unverified"
        assert result["evidence"] == []

    def test_unverified_when_score_below_threshold(self, monkeypatch):
        # Threshold is 0.3 on sigmoid-normalized score. sigmoid(-5) ≈ 0.007 — below 0.3.
        monkeypatch.setattr(
            "backend.app.shield.verifier._rerank_evidence",
            lambda q, candidates, k=3: [{**c, "rerank_score": -5.0} for c in candidates[:k]],
        )
        result = verify_claim("Penalty is 2% per month.", "case-123")
        assert result["status"] == "Unverified"

    def test_contradicted_when_llm_says_contradict(self, monkeypatch):
        monkeypatch.setattr(
            "backend.app.shield.verifier._llm_support_or_contradict",
            lambda claim, texts, model: "Contradicted",
        )
        result = verify_claim("There are no penalties.", "case-123")
        assert result["status"] == "Contradicted"

    def test_empty_claim_returns_unverified(self):
        result = verify_claim("", "case-123")
        assert result["status"] == "Unverified"
        assert result["evidence"] == []

    def test_evidence_has_correct_structure(self):
        result = verify_claim("Penalty is 2% per month.", "case-123")
        assert len(result["evidence"]) > 0
        ev = result["evidence"][0]
        assert "content" in ev
        assert "score" in ev
        assert "page" in ev
        assert "filename" in ev

    def test_claim_preserved_in_output(self):
        claim = "The contract terminates after 90 days."
        result = verify_claim(claim, "case-123")
        assert result["claim"] == claim


class TestVerifyClaims:
    def test_batch_returns_one_result_per_claim(self):
        claims = ["Claim A.", "Claim B.", "Claim C."]
        results = verify_claims(claims, "case-123")
        assert len(results) == len(claims)

    def test_batch_empty_list(self):
        assert verify_claims([], "case-123") == []
