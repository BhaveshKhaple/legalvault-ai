"""Tests for Task 5.3 — Gap detector."""

import pytest
from backend.app.shield.gap_detector import detect_gaps, GapReport


class TestDetectGaps:
    def test_returns_gap_report(self):
        result = detect_gaps("This is a contract.", "contract")
        assert isinstance(result, GapReport)

    def test_finds_present_section(self):
        text = "The parties hereby agree... The governing law shall be India."
        result = detect_gaps(text, "contract")
        present_names = [s.name for s in result.present]
        assert "Governing Law" in present_names

    def test_detects_missing_section(self):
        # Text with no indemnity keywords at all
        text = "Payment is due on the 1st. Term is 12 months. Parties agree."
        result = detect_gaps(text, "contract")
        missing_names = [s.name for s in result.missing]
        assert "Indemnity" in missing_names

    def test_unknown_doc_type_returns_empty_report(self):
        result = detect_gaps("Some text.", "unknown_type")
        assert result.present_count == 0
        assert result.missing_count == 0

    def test_case_insensitive_matching(self):
        text = "GOVERNING LAW: This agreement is subject to the laws of India."
        result = detect_gaps(text, "contract")
        present_names = [s.name for s in result.present]
        assert "Governing Law" in present_names

    def test_full_contract_with_all_sections(self):
        text = (
            "PARTIES: Alpha Corp and Beta Ltd. "
            "DEFINITIONS: 'Agreement' means this contract. "
            "SCOPE OF WORK: Services include software delivery. "
            "PAYMENT TERMS: Invoice payable within 30 days. "
            "TERM AND DURATION: Effective from 1 Jan 2026, expiry 31 Dec 2026. "
            "TERMINATION: Either party may terminate with 30 days notice. "
            "CONFIDENTIALITY: All information is confidential and non-disclosure applies. "
            "INDEMNITY: Each party shall indemnify the other. "
            "GOVERNING LAW: Courts of Delhi, laws of India. "
            "DISPUTE RESOLUTION: Arbitration under Indian Arbitration Act. "
            "LIMITATION OF LIABILITY: Maximum liability is Rs 10 Lakhs. "
            "FORCE MAJEURE: Act of God or natural disaster exempts obligations."
        )
        result = detect_gaps(text, "contract")
        assert result.missing_count == 0
        assert result.present_count == 12

    def test_to_dict_has_required_keys(self):
        result = detect_gaps("parties agree.", "contract")
        d = result.to_dict()
        assert "doc_type" in d
        assert "present_count" in d
        assert "missing_count" in d
        assert "missing" in d
        assert "present" in d


class TestDetectGapsAuditReport:
    def test_audit_report_template_loads(self):
        result = detect_gaps("In our opinion the accounts are true and fair.", "audit_report")
        assert isinstance(result, GapReport)
        present_names = [s.name for s in result.present]
        assert "Auditor Opinion" in present_names


class TestDetectGapsAgreement:
    def test_agreement_template_loads(self):
        result = detect_gaps("The parties hereby agree to the following obligations.", "agreement")
        assert isinstance(result, GapReport)
