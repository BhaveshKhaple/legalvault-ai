"""
Task 5.4 — Trust-score aggregator for Report Shield.

Rolls up the outputs from:
  - verifier.verify_claims()  → verified / unverified / contradicted counts
  - gap_detector.detect_gaps() → missing section count

Into a single trust score 0-100 and a human-readable verdict band.

Scoring formula (weights tuned with the team — see docs/TRD.md §5.4):
  Base score = 100
  - contradiction_penalty  = 20 × contradiction_count   (most severe)
  - missing_penalty        = 8  × missing_count          (legally required sections)
  - unverified_penalty     = 3  × unverified_count       (could not find support)

Score is clamped to [0, 100]. The weight choices are intentionally
conservative: a single contradiction in a legal document is a serious flag.

Trust bands (per DESIGN.md §4):
  ≥80  — "Strong — ready to file"   (green)
  60–79 — "Review before filing"    (amber)
  <60  — "At risk — do not file"    (red)
"""

from dataclasses import dataclass
from typing import Literal

TrustBand = Literal["strong", "review", "at_risk"]

_CONTRADICTION_PENALTY = 20
_MISSING_PENALTY = 8
_UNVERIFIED_PENALTY = 3


@dataclass
class TrustScoreResult:
    trust_score: int  # 0-100
    band: TrustBand
    band_label: str  # Human-readable label
    verified_count: int
    unverified_count: int
    contradiction_count: int
    missing_count: int
    contradictions: list[dict]  # full contradiction evidence for UI
    missing_sections: list[str]  # names of missing required sections

    def to_dict(self) -> dict:
        return {
            "trust_score": self.trust_score,
            "band": self.band,
            "band_label": self.band_label,
            "verified_count": self.verified_count,
            "unverified_count": self.unverified_count,
            "contradiction_count": self.contradiction_count,
            "missing_count": self.missing_count,
            "contradictions": self.contradictions,
            "missing_sections": self.missing_sections,
        }


def _band(score: int) -> tuple[TrustBand, str]:
    if score >= 80:
        return "strong", "Strong — ready to file"
    if score >= 60:
        return "review", "Review before filing"
    return "at_risk", "At risk — do not file"


def aggregate(
    verification_results: list[dict],
    gap_report_dict: dict,
) -> TrustScoreResult:
    """Compute a trust score from verifier + gap detector outputs.

    Args:
        verification_results: Output of verifier.verify_claims() — list of
            dicts each with 'status' and 'claim'.
        gap_report_dict: Output of GapReport.to_dict() — has 'missing' list.

    Returns:
        TrustScoreResult with trust_score, band, and detailed counts.
    """
    verified = [r for r in verification_results if r.get("status") == "Verified"]
    unverified = [r for r in verification_results if r.get("status") == "Unverified"]
    contradicted = [
        r for r in verification_results if r.get("status") == "Contradicted"
    ]

    missing = gap_report_dict.get("missing", [])

    score = 100
    score -= _CONTRADICTION_PENALTY * len(contradicted)
    score -= _MISSING_PENALTY * len(missing)
    score -= _UNVERIFIED_PENALTY * len(unverified)
    score = max(0, min(100, score))

    band, band_label = _band(score)

    return TrustScoreResult(
        trust_score=score,
        band=band,
        band_label=band_label,
        verified_count=len(verified),
        unverified_count=len(unverified),
        contradiction_count=len(contradicted),
        missing_count=len(missing),
        contradictions=[
            {
                "claim": r["claim"],
                "reason": r.get("reason", ""),
                "evidence": r.get("evidence", []),
            }
            for r in contradicted
        ],
        missing_sections=[s["name"] for s in missing],
    )
