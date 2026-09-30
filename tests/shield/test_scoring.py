"""Tests for Task 5.4 — Trust-score aggregator."""

import pytest
from backend.app.shield.scoring import aggregate, TrustScoreResult


def _make_results(verified=3, unverified=1, contradicted=0):
    results = []
    for _ in range(verified):
        results.append({"status": "Verified", "claim": "Claim.", "evidence": []})
    for _ in range(unverified):
        results.append({"status": "Unverified", "claim": "Claim.", "evidence": []})
    for _ in range(contradicted):
        results.append({"status": "Contradicted", "claim": "Claim.", "evidence": []})
    return results


def _make_gap(missing=0):
    return {
        "missing": [{"name": f"Section {i}", "description": "desc"} for i in range(missing)],
        "present": [],
    }


class TestAggregate:
    def test_returns_trust_score_result(self):
        result = aggregate(_make_results(), _make_gap())
        assert isinstance(result, TrustScoreResult)

    def test_perfect_score(self):
        result = aggregate(_make_results(verified=5, unverified=0, contradicted=0), _make_gap(0))
        assert result.trust_score == 100
        assert result.band == "strong"

    def test_score_penalises_contradictions(self):
        no_contra = aggregate(_make_results(contradicted=0), _make_gap())
        with_contra = aggregate(_make_results(contradicted=1), _make_gap())
        assert with_contra.trust_score < no_contra.trust_score

    def test_score_penalises_missing_sections(self):
        no_missing = aggregate(_make_results(), _make_gap(0))
        with_missing = aggregate(_make_results(), _make_gap(3))
        assert with_missing.trust_score < no_missing.trust_score

    def test_score_never_below_zero(self):
        result = aggregate(_make_results(contradicted=10, unverified=20), _make_gap(20))
        assert result.trust_score == 0

    def test_score_never_above_100(self):
        result = aggregate([], _make_gap(0))
        assert result.trust_score == 100

    def test_strong_band_above_80(self):
        result = aggregate(_make_results(verified=10), _make_gap(0))
        assert result.band == "strong"
        assert "ready to file" in result.band_label.lower()

    def test_at_risk_band_with_contradictions(self):
        result = aggregate(_make_results(contradicted=3), _make_gap(2))
        # 100 - 60 - 16 = 24
        assert result.band == "at_risk"
        assert "do not file" in result.band_label.lower()

    def test_review_band_mid_range(self):
        # 100 - 20 = 80 … need to get to 60-79
        result = aggregate(_make_results(contradicted=1, unverified=3), _make_gap(1))
        # 100 - 20 - 9 - 8 = 63
        assert result.band in ("review", "at_risk")

    def test_counts_match(self):
        result = aggregate(_make_results(verified=2, unverified=1, contradicted=1), _make_gap(3))
        assert result.verified_count == 2
        assert result.unverified_count == 1
        assert result.contradiction_count == 1
        assert result.missing_count == 3

    def test_contradictions_in_output(self):
        results = [{"status": "Contradicted", "claim": "X is Y.", "evidence": [{"content": "Z"}]}]
        result = aggregate(results, _make_gap())
        assert len(result.contradictions) == 1
        assert result.contradictions[0]["claim"] == "X is Y."

    def test_to_dict_has_all_keys(self):
        result = aggregate(_make_results(), _make_gap())
        d = result.to_dict()
        for key in ["trust_score", "band", "band_label", "verified_count",
                    "unverified_count", "contradiction_count", "missing_count",
                    "contradictions", "missing_sections"]:
            assert key in d
