"""
eval-corpus — Build the 10-document mixed-format test corpus.

Writes synthetic documents into tests/eval/corpus/ in PDF, TXT, and DOCX
formats, covering different structural patterns the chunker + retriever
must handle:

  d01_nda_numbered.pdf           — 2 pages, numbered clauses (1.1, 1.2)
  d02_service_agreement.pdf      — ~6 pages, long numbered sections + lists
  d03_employment_contract.pdf    — 3 pages, prose + bullet lists
  d04_lease_deed.pdf             — 4 pages, defined-term heavy, "Lessor" / "Lessee"
  d05_privacy_policy.pdf         — 3 pages, heading-driven, no numbering
  d06_fee_schedule.pdf           — 1 page, table-heavy quotation
  d07_meeting_minutes.txt        — plain prose with action items
  d08_product_spec.docx          — H1/H2/H3 headings, numbered sections
  d09_faq.docx                   — Q&A format
  d10_bilingual_notice.pdf       — English + Hindi legal notice (tier dispatcher
                                    sanity data; still ingested with e5-small for now)

Idempotent — rerun safely; existing files overwritten.
"""

from pathlib import Path

from docx import Document
from docx.shared import Pt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

CORPUS_DIR = Path(__file__).parent / "corpus"


def _pdf_doc(path: Path):
    return SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=20 * mm, rightMargin=20 * mm,
        topMargin=20 * mm, bottomMargin=20 * mm,
    )


def _styles():
    s = getSampleStyleSheet()
    s.add(ParagraphStyle("Clause", parent=s["BodyText"], spaceAfter=6, leading=14))
    s.add(ParagraphStyle("Section", parent=s["Heading2"], spaceAfter=8, spaceBefore=12))
    return s


# ─── d01 — numbered NDA ─────────────────────────────────────────────────────


def build_d01_nda():
    path = CORPUS_DIR / "d01_nda_numbered.pdf"
    doc = _pdf_doc(path)
    S = _styles()
    story = [
        Paragraph("NON-DISCLOSURE AGREEMENT", S["Title"]),
        Paragraph("Between Acme Technologies Pvt. Ltd. and the Receiving Party.", S["BodyText"]),
        Spacer(1, 10),
        Paragraph("1. Definitions", S["Section"]),
        Paragraph("1.1 \"Confidential Information\" means any non-public business, technical, or financial information disclosed by Acme to the Receiving Party in connection with the Permitted Purpose.", S["Clause"]),
        Paragraph("1.2 \"Permitted Purpose\" means evaluation of a potential commercial relationship between the parties.", S["Clause"]),
        Paragraph("1.3 \"Representatives\" means directors, employees, and professional advisors of the Receiving Party.", S["Clause"]),
        Paragraph("2. Obligations", S["Section"]),
        Paragraph("2.1 The Receiving Party shall hold all Confidential Information in strict confidence for a period of five (5) years from the date of disclosure.", S["Clause"]),
        Paragraph("2.2 The Receiving Party shall not disclose any Confidential Information to any third party without prior written consent from Acme.", S["Clause"]),
        PageBreak(),
        Paragraph("3. Exclusions", S["Section"]),
        Paragraph("3.1 This Agreement does not apply to information that is or becomes publicly available through no fault of the Receiving Party.", S["Clause"]),
        Paragraph("3.2 Information independently developed by the Receiving Party without reference to Confidential Information is also excluded.", S["Clause"]),
        Paragraph("4. Governing Law", S["Section"]),
        Paragraph("4.1 This Agreement is governed by the laws of India. Any dispute arising under this Agreement shall be subject to the exclusive jurisdiction of the courts at Pune, Maharashtra.", S["Clause"]),
        Paragraph("5. Term and Termination", S["Section"]),
        Paragraph("5.1 This Agreement commences on the Effective Date and continues for three (3) years unless terminated earlier by mutual written agreement.", S["Clause"]),
    ]
    doc.build(story)


# ─── d02 — long service agreement ───────────────────────────────────────────


def build_d02_service_agreement():
    path = CORPUS_DIR / "d02_service_agreement.pdf"
    doc = _pdf_doc(path)
    S = _styles()
    story = [
        Paragraph("MASTER SERVICE AGREEMENT", S["Title"]),
        Paragraph("Between Northstar Legal LLP (\"Firm\") and Vendor (\"Service Provider\").", S["BodyText"]),
        Spacer(1, 10),

        Paragraph("1. Scope of Services", S["Section"]),
        Paragraph("1.1 The Service Provider shall provide document management services as described in Schedule A, which forms part of this Agreement.", S["Clause"]),
        Paragraph("1.2 All services shall be rendered in accordance with industry best practices and professional standards.", S["Clause"]),
        Paragraph("1.3 Any change to the scope must be documented in a written change order signed by authorised representatives of both parties.", S["Clause"]),

        Paragraph("2. Fees and Payment", S["Section"]),
        Paragraph("2.1 The Firm shall pay the Service Provider a monthly retainer of Rs. 2,50,000 (Rupees Two Lakh Fifty Thousand only), payable within thirty (30) days of invoice receipt.", S["Clause"]),
        Paragraph("2.2 Expenses pre-approved by the Firm shall be reimbursed at actuals against supporting documentation.", S["Clause"]),
        Paragraph("2.3 Late payment shall attract interest at the rate of 1.5% per month calculated on a simple-interest basis.", S["Clause"]),

        Paragraph("3. Term", S["Section"]),
        Paragraph("3.1 This Agreement shall commence on 1 April 2026 and shall remain in force for a period of twenty-four (24) months, renewable by mutual written consent.", S["Clause"]),

        PageBreak(),

        Paragraph("4. Confidentiality", S["Section"]),
        Paragraph("4.1 The Service Provider shall treat all client information as strictly confidential and shall implement reasonable technical and organisational measures to protect it.", S["Clause"]),
        Paragraph("4.2 The confidentiality obligations shall survive termination of this Agreement for a period of seven (7) years.", S["Clause"]),

        Paragraph("5. Service Levels", S["Section"]),
        Paragraph("5.1 The Service Provider shall meet the following service levels:", S["Clause"]),
        Paragraph("(a) Document retrieval requests shall be fulfilled within four (4) business hours.", S["Clause"]),
        Paragraph("(b) Scheduled maintenance shall be notified at least seventy-two (72) hours in advance.", S["Clause"]),
        Paragraph("(c) System uptime shall not be less than 99.5% measured monthly.", S["Clause"]),

        Paragraph("6. Termination", S["Section"]),
        Paragraph("6.1 Either party may terminate this Agreement with ninety (90) days written notice to the other party.", S["Clause"]),
        Paragraph("6.2 The Firm may terminate immediately in the event of a material breach that remains uncured for thirty (30) days after written notice.", S["Clause"]),

        PageBreak(),

        Paragraph("7. Limitation of Liability", S["Section"]),
        Paragraph("7.1 The aggregate liability of the Service Provider under this Agreement shall not exceed the total fees paid by the Firm in the twelve (12) months preceding the claim.", S["Clause"]),
        Paragraph("7.2 Neither party shall be liable for indirect, incidental, or consequential damages.", S["Clause"]),

        Paragraph("8. Indemnity", S["Section"]),
        Paragraph("8.1 The Service Provider shall indemnify the Firm against any third-party claims arising from the Service Provider's gross negligence or wilful misconduct.", S["Clause"]),

        Paragraph("9. Governing Law and Jurisdiction", S["Section"]),
        Paragraph("9.1 This Agreement shall be governed by and construed in accordance with the laws of India. The courts at Mumbai shall have exclusive jurisdiction to settle any dispute arising out of this Agreement.", S["Clause"]),

        Paragraph("10. Entire Agreement", S["Section"]),
        Paragraph("10.1 This Agreement constitutes the entire understanding between the parties with respect to the subject matter hereof and supersedes all prior discussions.", S["Clause"]),
    ]
    doc.build(story)


# ─── d03 — employment contract ──────────────────────────────────────────────


def build_d03_employment():
    path = CORPUS_DIR / "d03_employment_contract.pdf"
    doc = _pdf_doc(path)
    S = _styles()
    story = [
        Paragraph("EMPLOYMENT CONTRACT", S["Title"]),
        Paragraph("Between Lumen Software Pvt. Ltd. (\"Company\") and Ms. Priya Nair (\"Employee\").", S["BodyText"]),
        Spacer(1, 10),
        Paragraph("Position and Reporting", S["Section"]),
        Paragraph("The Employee is appointed as Senior Backend Engineer, reporting to the Vice President of Engineering. The role is based at the Company's Pune office with occasional travel.", S["Clause"]),
        Paragraph("Compensation", S["Section"]),
        Paragraph("The Employee's annual cost-to-company is Rs. 24,00,000 (Rupees Twenty-Four Lakh only), paid in twelve equal monthly instalments on or before the last working day of each month. The compensation includes:", S["Clause"]),
        Paragraph("• Base salary: Rs. 18,00,000", S["Clause"]),
        Paragraph("• Performance bonus (variable, up to): Rs. 4,00,000", S["Clause"]),
        Paragraph("• Health insurance and retirement contributions: Rs. 2,00,000", S["Clause"]),
        Paragraph("Probation", S["Section"]),
        Paragraph("The Employee shall serve a probation period of six (6) months from the date of joining. During probation, the Company may terminate the engagement with fifteen (15) days written notice.", S["Clause"]),
        PageBreak(),
        Paragraph("Notice Period", S["Section"]),
        Paragraph("After successful completion of probation, either party may terminate the engagement with sixty (60) days written notice or payment in lieu of notice.", S["Clause"]),
        Paragraph("Non-Compete", S["Section"]),
        Paragraph("For a period of twelve (12) months after termination, the Employee shall not accept employment with any direct competitor of the Company operating in India.", S["Clause"]),
        Paragraph("Leaves", S["Section"]),
        Paragraph("The Employee is entitled to:", S["Clause"]),
        Paragraph("• 24 days of paid earned leave per calendar year", S["Clause"]),
        Paragraph("• 12 days of sick leave per calendar year", S["Clause"]),
        Paragraph("• Public holidays as per the Company's holiday calendar", S["Clause"]),
    ]
    doc.build(story)


# ─── d04 — lease deed ───────────────────────────────────────────────────────


def build_d04_lease():
    path = CORPUS_DIR / "d04_lease_deed.pdf"
    doc = _pdf_doc(path)
    S = _styles()
    story = [
        Paragraph("LEASE DEED", S["Title"]),
        Paragraph("Between Mr. Rohan Deshpande (\"Lessor\") and Horizon Design Studio LLP (\"Lessee\").", S["BodyText"]),
        Spacer(1, 10),
        Paragraph("1. Premises", S["Section"]),
        Paragraph("The Lessor hereby grants to the Lessee the premises bearing Shop No. 4, Ground Floor, Deshpande Chambers, FC Road, Pune 411005, admeasuring approximately 850 square feet carpet area.", S["Clause"]),
        Paragraph("2. Term", S["Section"]),
        Paragraph("The lease term shall be for a period of thirty-three (33) months commencing from 1 May 2026 and ending on 31 January 2029.", S["Clause"]),
        Paragraph("3. Rent", S["Section"]),
        Paragraph("The Lessee shall pay a monthly rent of Rs. 85,000 (Rupees Eighty-Five Thousand only), payable in advance on or before the 5th of each calendar month.", S["Clause"]),
        PageBreak(),
        Paragraph("4. Security Deposit", S["Section"]),
        Paragraph("The Lessee has deposited with the Lessor an interest-free security deposit of Rs. 5,10,000, refundable upon termination subject to deduction of outstanding dues and cost of damage, if any.", S["Clause"]),
        Paragraph("5. Rent Escalation", S["Section"]),
        Paragraph("The rent shall escalate by 7% at the end of every twelve (12) months during the term of this Lease.", S["Clause"]),
        Paragraph("6. Lock-in Period", S["Section"]),
        Paragraph("The Lessee shall not terminate the Lease before completion of twenty-four (24) months (the Lock-in Period). Early termination within the Lock-in Period shall render the Lessee liable to pay the rent for the balance Lock-in Period.", S["Clause"]),
        PageBreak(),
        Paragraph("7. Permitted Use", S["Section"]),
        Paragraph("The Lessee shall use the Premises solely for the purpose of operating a design studio and ancillary office functions, and shall not use it for any other purpose without written consent of the Lessor.", S["Clause"]),
        Paragraph("8. Maintenance", S["Section"]),
        Paragraph("Routine maintenance of the Premises shall be the responsibility of the Lessee. Structural repairs shall be undertaken by the Lessor at the Lessor's cost.", S["Clause"]),
        Paragraph("9. Termination and Notice", S["Section"]),
        Paragraph("After expiry of the Lock-in Period, either party may terminate this Lease by giving the other party three (3) months prior written notice.", S["Clause"]),
    ]
    doc.build(story)


# ─── d05 — privacy policy (heading-driven, no numbering) ────────────────────


def build_d05_privacy():
    path = CORPUS_DIR / "d05_privacy_policy.pdf"
    doc = _pdf_doc(path)
    S = _styles()
    story = [
        Paragraph("PRIVACY POLICY", S["Title"]),
        Paragraph("Finvio Fintech Solutions Pvt. Ltd. — Effective 1 June 2026.", S["BodyText"]),
        Spacer(1, 10),
        Paragraph("What We Collect", S["Section"]),
        Paragraph("We collect your name, email address, mobile number, date of birth, PAN, and Aadhaar reference number for the purpose of KYC verification as mandated by the Reserve Bank of India.", S["Clause"]),
        Paragraph("We also collect device-level data such as IP address, device model, operating system, and app usage patterns to detect fraudulent activity and improve our service.", S["Clause"]),
        Paragraph("How We Use It", S["Section"]),
        Paragraph("Your personal information is used to open and operate your account, verify your identity under applicable laws, process transactions, and communicate account-related notifications.", S["Clause"]),
        Paragraph("We may also use anonymised aggregate data for internal analytics and product improvement.", S["Clause"]),
        PageBreak(),
        Paragraph("Who We Share It With", S["Section"]),
        Paragraph("Your personal information may be shared with regulatory authorities when legally compelled, with KYC bureaus for identity verification, and with payment partners who process transactions on our behalf.", S["Clause"]),
        Paragraph("We do not sell your personal information to third parties for marketing purposes.", S["Clause"]),
        Paragraph("Data Retention", S["Section"]),
        Paragraph("We retain your personal information for the duration of your account and for a further period of eight (8) years after account closure, as required under RBI's record-retention guidelines.", S["Clause"]),
        Paragraph("Your Rights", S["Section"]),
        Paragraph("You have the right to access, correct, and request deletion of your personal information. To exercise these rights, write to privacy@finvio.in from your registered email address.", S["Clause"]),
        PageBreak(),
        Paragraph("Security Measures", S["Section"]),
        Paragraph("We implement bank-grade encryption, two-factor authentication, and continuous monitoring to protect your data. All personal information is stored on servers located within India.", S["Clause"]),
        Paragraph("Contact", S["Section"]),
        Paragraph("For any privacy-related queries, you may reach our Data Protection Officer at dpo@finvio.in or on +91 20 1234 5678 during business hours.", S["Clause"]),
    ]
    doc.build(story)


# ─── d06 — table-heavy fee schedule ─────────────────────────────────────────


def build_d06_fee_schedule():
    path = CORPUS_DIR / "d06_fee_schedule.pdf"
    doc = _pdf_doc(path)
    S = _styles()
    story = [
        Paragraph("PROFESSIONAL FEE SCHEDULE", S["Title"]),
        Paragraph("Mehta & Associates — Chartered Accountants — Fiscal Year 2026-27", S["BodyText"]),
        Spacer(1, 12),
    ]

    data = [
        ["Service", "Scope", "Fee (INR)"],
        ["Statutory Audit", "Private Limited Company, turnover under Rs. 50 Cr", "1,25,000"],
        ["Tax Audit (44AB)", "Business entity, turnover under Rs. 10 Cr", "60,000"],
        ["GST Compliance", "Monthly filings, up to 500 invoices/month", "15,000 / month"],
        ["Income Tax Return — Individual", "Salaried, with capital-gains component", "7,500"],
        ["Income Tax Return — Company", "Private Limited Company", "35,000"],
        ["Transfer Pricing Study", "Domestic + international, per entity", "2,25,000"],
        ["Advisory — Hourly", "Partner-led consultation", "6,500 / hour"],
        ["Advisory — Hourly", "Associate-led consultation", "3,500 / hour"],
        ["ROC Annual Filings", "AOC-4, MGT-7, DIR-3 KYC", "22,000"],
    ]
    tbl = Table(data, colWidths=[55 * mm, 85 * mm, 30 * mm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1b2a4e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (2, 1), (2, -1), "RIGHT"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#777777")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 10))
    story.append(Paragraph("All fees are exclusive of GST. Travel, lodging, and out-of-pocket expenses billed at actuals. Rates valid until 31 March 2027.", S["BodyText"]))

    doc.build(story)


# ─── d07 — meeting minutes (plain text) ─────────────────────────────────────


def build_d07_meeting_minutes():
    path = CORPUS_DIR / "d07_meeting_minutes.txt"
    text = """\
MEETING MINUTES
Project: LegalVault AI
Date: 25 September 2026
Time: 11:00 AM – 12:30 PM IST
Venue: Conference Room, MIT Pune / Google Meet hybrid

Attendees
---------
Bhavesh Khaple (Lead), Shubham (Backend), Vijay (Frontend), Tejas (QA),
Prof. S. D. Sanap (Faculty Advisor)

Agenda
------
1. Review Module 6 (Backend API) completion status
2. Decide on authentication approach for Module 6.4
3. Plan Module 7 (Frontend) kickoff
4. Discuss evaluation methodology for Module 10

Discussion
----------
Bhavesh walked the team through the current demo. The RAG pipeline is
functioning end-to-end on his GTX 1650 setup using Phi-3 Mini and the
e5-small-v2 embedding model. Response latency is approximately 23 seconds
warm, which the team agreed is acceptable for the demo but needs
optimisation before deployment.

Shubham raised concerns about the lack of multi-tenancy. Prof. Sanap
emphasised that any legal tool must enforce strict case isolation. The
team agreed to prioritise Task 6.4 (Auth + JWT) in the next sprint.

Vijay confirmed that the HTML prototype covers the demo requirements.
The team deferred the SvelteKit migration (Tasks 7.1 to 7.4) until after
the first end-user evaluation.

Tejas raised the question of how to evaluate retrieval accuracy
objectively. Bhavesh proposed building a synthetic corpus with
hand-labelled queries, which the team supported.

Action Items
------------
AI-1  Bhavesh      Complete Task 6.4 (Auth + JWT) by 1 October 2026
AI-2  Shubham      Review Task 6.4 PR within 48 hours of submission
AI-3  Vijay        Submit a PR to retarget AI-Vijay:adding-my-code to v1
AI-4  Tejas        Draft 20 hand-labelled queries for retrieval eval by 5 October
AI-5  Prof. Sanap  Share real redacted contract samples for pilot test (optional)

Decisions
---------
D-1  JWT-based authentication selected over session cookies. Rationale:
     stateless, works across the mobile client if we ever ship one.

D-2  First user to register becomes admin automatically. Subsequent users
     are analysts. Role promotion only via DB write for now.

D-3  Cross-tenant case access returns 404, not 403, to avoid leaking case
     existence to unauthorised users.

Next Meeting
------------
2 October 2026 at 11:00 AM IST.
"""
    path.write_text(text, encoding="utf-8")


# ─── d08 — product spec (DOCX with headings) ────────────────────────────────


def build_d08_product_spec():
    path = CORPUS_DIR / "d08_product_spec.docx"
    doc = Document()
    doc.add_heading("Product Specification: Autocomplete v2", level=0)

    doc.add_heading("1. Overview", level=1)
    doc.add_paragraph(
        "The Autocomplete v2 subsystem suggests contextually-relevant completions "
        "to users as they type a query into the search box. It replaces the "
        "legacy Autocomplete v1 which relied on prefix-matched static terms."
    )

    doc.add_heading("2. Goals", level=1)
    doc.add_paragraph("The subsystem shall meet the following goals:")
    doc.add_paragraph("• Return the first suggestion within 150 milliseconds p99 latency.", style="List Bullet")
    doc.add_paragraph("• Produce semantically-relevant suggestions, not just prefix matches.", style="List Bullet")
    doc.add_paragraph("• Degrade gracefully to v1 behaviour if the semantic index is unavailable.", style="List Bullet")

    doc.add_heading("3. Non-Goals", level=1)
    doc.add_paragraph(
        "This specification does not address spelling correction, personalisation "
        "based on user history, or multilingual query handling. Those remain "
        "outside the scope of Autocomplete v2."
    )

    doc.add_heading("4. Technical Design", level=1)
    doc.add_heading("4.1 Indexing Pipeline", level=2)
    doc.add_paragraph(
        "Query logs from the last 30 days are mined nightly. Queries with at "
        "least five occurrences and a click-through rate above 2% are embedded "
        "using the e5-small-v2 model and stored in a Qdrant collection called "
        "autocomplete_phrases."
    )

    doc.add_heading("4.2 Serving", level=2)
    doc.add_paragraph(
        "On each keystroke after the user has typed three characters, the "
        "typed-so-far string is embedded and queried against the Qdrant "
        "collection with top-k set to eight. The eight candidates are "
        "re-ranked by a lightweight bigram-overlap heuristic to prefer "
        "candidates that start with the typed prefix."
    )

    doc.add_heading("4.3 Fallback", level=2)
    doc.add_paragraph(
        "If Qdrant returns an error or latency exceeds 100 milliseconds, the "
        "subsystem falls back to the legacy static-term index."
    )

    doc.add_heading("5. Metrics", level=1)
    doc.add_paragraph("Success will be measured by:")
    doc.add_paragraph("• Suggestion click-through rate (target: 18% vs 12% baseline)", style="List Bullet")
    doc.add_paragraph("• Session-level search completion rate (target: 72% vs 65% baseline)", style="List Bullet")
    doc.add_paragraph("• p99 latency under 150 ms", style="List Bullet")

    doc.add_heading("6. Rollout Plan", level=1)
    doc.add_paragraph(
        "Autocomplete v2 will be rolled out via a feature flag with a 1% / 10% "
        "/ 50% / 100% ramp over four weeks. Each ramp stage requires the above "
        "metrics to meet or exceed baseline before progressing."
    )

    doc.save(str(path))


# ─── d09 — FAQ (DOCX, Q&A format) ───────────────────────────────────────────


def build_d09_faq():
    path = CORPUS_DIR / "d09_faq.docx"
    doc = Document()
    doc.add_heading("Frequently Asked Questions — LegalVault AI", level=0)

    qa_pairs = [
        ("What is LegalVault AI?",
         "LegalVault AI is an on-premise legal document retrieval and verification "
         "system that lets lawyers and compliance teams ask questions about their "
         "case files and get cited, evidence-backed answers."),
        ("Does LegalVault AI send data to the cloud?",
         "No. All processing runs on your own hardware. The language model, "
         "embedding model, and vector database all run locally. No document, "
         "query, or response ever leaves your machine."),
        ("What hardware do I need to run it?",
         "Minimum: a laptop with 8 GB RAM, an Intel i5 or equivalent, and at "
         "least 20 GB of free disk space. For acceptable performance, a GPU "
         "with 4 GB VRAM (such as GTX 1650 or newer) is strongly recommended."),
        ("Which document formats are supported?",
         "LegalVault AI currently supports PDF files (text-based, not scanned), "
         "plain-text files, Microsoft Word documents in DOCX format, and audio "
         "files in MP3, WAV, M4A, or OGG format."),
        ("How are my documents kept private between cases?",
         "Every document is tagged with its case identifier at ingestion. "
         "Queries filter by case identifier at the vector-database layer, which "
         "makes it impossible for a query against case A to return results from "
         "case B. Users can only query cases they have access to."),
        ("Can I export my data?",
         "Yes. You can export any case as a ZIP archive containing the original "
         "documents, the generated answers with their citations, and the full "
         "list of extracted chunks. The export feature is available from the "
         "case detail page."),
        ("What languages does LegalVault AI support?",
         "English is fully supported in all tiers. For documents containing "
         "Hindi or other Indian languages, use the BGE-M3 embedding tier which "
         "supports more than 100 languages. The language model remains "
         "English-first in the current release."),
    ]

    for question, answer in qa_pairs:
        p_q = doc.add_paragraph()
        run_q = p_q.add_run("Q: " + question)
        run_q.bold = True
        run_q.font.size = Pt(11)

        p_a = doc.add_paragraph("A: " + answer)
        p_a.paragraph_format.space_after = Pt(10)

    doc.save(str(path))


# ─── d10 — bilingual notice (English + Hindi) ───────────────────────────────


def build_d10_bilingual_notice():
    path = CORPUS_DIR / "d10_bilingual_notice.pdf"
    doc = _pdf_doc(path)
    S = _styles()
    story = [
        Paragraph("LEGAL NOTICE / कानूनी नोटिस", S["Title"]),
        Paragraph("Issued under Section 138 of the Negotiable Instruments Act, 1881.", S["BodyText"]),
        Spacer(1, 10),
        Paragraph("To / सेवा में,", S["Section"]),
        Paragraph("Mr. Anand Verma, 23 Model Colony, Pune 411016.", S["BodyText"]),
        Paragraph("श्री आनंद वर्मा, 23 मॉडल कॉलोनी, पुणे 411016.", S["BodyText"]),
        Spacer(1, 10),
        Paragraph("Subject / विषय", S["Section"]),
        Paragraph("Dishonour of cheque bearing number 008245 dated 12 August 2026 drawn on HDFC Bank for Rs. 2,50,000.", S["BodyText"]),
        Paragraph("चेक संख्या 008245 दिनांक 12 अगस्त 2026, एचडीएफसी बैंक पर आहरित, राशि रुपये 2,50,000 के अनादरण के संबंध में।", S["BodyText"]),
        Spacer(1, 10),
        Paragraph("Statement of Facts", S["Section"]),
        Paragraph("1. The addressee issued the above-mentioned cheque in discharge of a legally enforceable debt.", S["Clause"]),
        Paragraph("2. The cheque was presented for payment at ICICI Bank, Deccan Branch, Pune on 18 August 2026.", S["Clause"]),
        Paragraph("3. The cheque was returned unpaid on 20 August 2026 with the remark 'Insufficient Funds'.", S["Clause"]),
        Paragraph("Demand", S["Section"]),
        Paragraph("You are hereby called upon to pay the said sum of Rs. 2,50,000 together with the dishonour charges of Rs. 350 within fifteen (15) days from receipt of this notice, failing which appropriate legal proceedings shall be initiated against you under Section 138 of the Negotiable Instruments Act, 1881.", S["BodyText"]),
    ]
    doc.build(story)


def main() -> None:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    builders = [
        build_d01_nda,
        build_d02_service_agreement,
        build_d03_employment,
        build_d04_lease,
        build_d05_privacy,
        build_d06_fee_schedule,
        build_d07_meeting_minutes,
        build_d08_product_spec,
        build_d09_faq,
        build_d10_bilingual_notice,
    ]
    for b in builders:
        b()
        print(f"built {b.__name__}")

    print(f"\nCorpus written to {CORPUS_DIR}")


if __name__ == "__main__":
    main()
