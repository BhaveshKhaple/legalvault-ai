"""
Task 10.2 side-fix: extract_claims(text) was added so shield.py router
(which imports extract_claims from a module that previously only exported
chunk_report) stops crashing when hit by a real PDF.
"""

from backend.app.shield.claim_extractor import extract_claims


def test_extract_claims_basic_sentences():
    text = "The penalty is 2% per month. Payment is due in 30 days."
    out = extract_claims(text)
    assert len(out) == 2
    assert any("penalty" in c.lower() for c in out)


def test_extract_claims_numbered_bullets():
    text = (
        "1. Parties. Alpha and Beta.\n"
        "2. Term. Twenty-four months from effective date.\n"
        "3. Termination. Ninety days notice."
    )
    out = extract_claims(text)
    assert len(out) >= 3


def test_extract_claims_filters_short_noise():
    text = "A B. OK. The confidentiality period is five years from the effective date."
    out = extract_claims(text)
    # "A B.", "OK." are below 10 chars and should be filtered
    assert all(len(c) >= 10 for c in out)


def test_extract_claims_dedupes():
    text = "Payment is due in 30 days.\n\nPayment is due in 30 days.\n\nSeparate statement here now."
    out = extract_claims(text)
    assert len(out) == 2  # one dupe removed


def test_extract_claims_empty_text():
    assert extract_claims("") == []
    assert extract_claims("   \n\n   ") == []


def test_chunk_report_delegates_to_extract_claims(tmp_path):
    """chunk_report() still works and uses the same extract_claims() internals."""
    import pymupdf

    pdf_path = tmp_path / "demo.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "The termination notice period is ninety days from written notice.")
    doc.save(str(pdf_path))
    doc.close()

    from backend.app.shield.claim_extractor import chunk_report
    claims = chunk_report(str(pdf_path))
    assert any("ninety days" in c.lower() for c in claims)
