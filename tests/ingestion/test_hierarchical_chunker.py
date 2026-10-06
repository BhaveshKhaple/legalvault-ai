"""Tests for Phase 2 hierarchical (parent-child) chunker."""

import pytest

from backend.app.ingestion.hierarchical_chunker import (
    CHILD_MAX_CHARS,
    PARENT_MAX_CHARS,
    chunk_hierarchical,
)


def _docling_page(items, page=1):
    """Build a Docling-shaped page dict from a list of {label, text} tuples."""
    return {
        "text": "\n".join(t for _, t in items),
        "page": page,
        "doc_id": "d",
        "items": [{"heading_level": lbl, "text": txt, "page": page} for lbl, txt in items],
    }


def _docling_page_with_bboxes(items, page=1, width=612.0, height=792.0):
    """Like _docling_page but each item carries its own bbox tuple."""
    return {
        "text": "\n".join(t for _, t, _ in items),
        "page": page,
        "doc_id": "d",
        "width": width,
        "height": height,
        "items": [
            {"heading_level": lbl, "text": txt, "page": page, "bbox": bb}
            for lbl, txt, bb in items
        ],
    }


class TestDoclingHierarchy:
    def test_single_section_produces_one_parent_and_children(self):
        # Each body > CHILD_MERGE_BELOW_CHARS (180) so they stay separate.
        body_1 = "The parties agree that this agreement shall commence on the first day of April two thousand twenty-six and remain in force for twenty-four months unless formally terminated earlier by either party on sixty days written notice."
        body_2 = "During the term of this agreement, each party shall maintain insurance coverage sufficient to meet its obligations and indemnify the other party against third-party claims arising from gross negligence or wilful misconduct."
        assert len(body_1) > 180 and len(body_2) > 180
        pages = [_docling_page([
            ("H1", "Section 1 - Terms"),
            ("body", body_1),
            ("body", body_2),
        ])]
        r = chunk_hierarchical(pages)
        assert len(r["parents"]) == 1
        assert r["parents"][0]["section_title"] == "Section 1 - Terms"
        assert len(r["children"]) == 2
        assert all(c["parent_index"] == 0 for c in r["children"])

    def test_two_sections_produce_two_parents(self):
        # Use UNIQUE text per paragraph so the substring-match parent cursor
        # cannot false-match across sections.
        t_terms = "The agreement commences on the first of April two thousand twenty-six and continues for a period of twenty-four months unless terminated earlier under the termination clause described in section five below."
        t_pay_a = "Payment of the monthly retainer in the sum of two hundred fifty thousand rupees shall be made on or before the fifth day of each calendar month without any deduction or set-off against outstanding invoices."
        t_pay_b = "Late payment attracts interest at one and a half percent per month on the outstanding sum calculated on a simple-interest basis without compounding until the full balance has been cleared by the paying party."
        for s in (t_terms, t_pay_a, t_pay_b):
            assert len(s) > 180
        pages = [_docling_page([
            ("H1", "Section 1 - Terms"),
            ("body", t_terms),
            ("H1", "Section 2 - Payment"),
            ("body", t_pay_a),
            ("body", t_pay_b),
        ])]
        r = chunk_hierarchical(pages)
        assert len(r["parents"]) == 2
        titles = [p["section_title"] for p in r["parents"]]
        assert titles == ["Section 1 - Terms", "Section 2 - Payment"]
        child_parents = [c["parent_index"] for c in r["children"]]
        assert child_parents == [0, 1, 1]

    def test_short_consecutive_items_are_merged(self):
        # Q&A pattern — Q and A are both short, merge into one child so BM25
        # can't rank Q above A for a user-phrased query.
        pages = [_docling_page([
            ("H1", "FAQ"),
            ("body", "Q: Is data sent to the cloud?"),
            ("body", "A: No, everything runs locally."),
            ("body", "Q: What hardware is needed?"),
            ("body", "A: 8 GB RAM minimum."),
        ])]
        r = chunk_hierarchical(pages)
        # 4 short items -> some merging; expect <= 2 children (Q+A pairs)
        assert len(r["children"]) <= 2
        joined = " ".join(c["content"] for c in r["children"])
        assert "cloud" in joined and "No, everything runs locally" in joined
        assert "hardware" in joined and "8 GB RAM" in joined

    def test_table_without_md_separator_falls_back_to_atomic(self):
        # No `|---|---|` row → not a parseable markdown table → single atomic child
        pages = [_docling_page([
            ("H1", "Fees"),
            ("body", "The fees are as follows."),
            ("table", "Service | Amount\nSetup | 10000\nMonthly | 5000"),
        ])]
        r = chunk_hierarchical(pages)
        table_parents = [p for p in r["parents"] if p["is_table"]]
        table_children = [c for c in r["children"] if c["is_table"]]
        assert len(table_parents) == 1
        assert len(table_children) == 1   # fallback atomic path
        assert table_children[0]["content"] == table_parents[0]["content"]
        assert "Setup | 10000" in table_children[0]["content"]

    def test_markdown_table_splits_into_per_row_children(self):
        """Citation fix 3 — the fee table that failed on real data.

        When Docling exports a table as proper markdown, the chunker emits
        one child per data row with the header prepended. This makes the
        reranker see each row as a complete semantic unit instead of hunting
        for a term in a flat 10-row concatenation.
        """
        md_table = (
            "| Charge          | Amount                 | Notes                  |\n"
            "|-----------------|------------------------|------------------------|\n"
            "| Processing fee  | 1.00% of amount        | Deducted up-front      |\n"
            "| Prepayment fee  | 2.00% of principal     | Part or full prepay    |\n"
            "| Penal charges   | 2.00% per month        | On overdue amount      |"
        )
        pages = [_docling_page([
            ("H1", "6. FEES AND CHARGES"),
            ("table", md_table),
        ])]
        r = chunk_hierarchical(pages)

        table_parents = [p for p in r["parents"] if p["is_table"]]
        table_children = [c for c in r["children"] if c["is_table"]]

        # Still ONE parent (the whole table — stays available for LLM context)
        assert len(table_parents) == 1
        # Three data rows → three children
        assert len(table_children) == 3
        # Each child has the header prepended + its own row
        for child in table_children:
            assert "| Charge" in child["content"]
            assert "| Amount" in child["content"]
            # And exactly one data row's "Notes" column marker
            lines = [ln for ln in child["content"].splitlines() if ln.strip()]
            assert len(lines) == 2  # header + one data row

        # The specific row that fixed the real "prepayment charge" bug
        prepayment_row = next(
            (c for c in table_children if "Prepayment fee" in c["content"]),
            None,
        )
        assert prepayment_row is not None
        assert "2.00% of principal" in prepayment_row["content"]
        # Rank-5 chunk no longer at risk — now the reranker sees "Prepayment fee |
        # 2.00% of principal" as its own atomic unit, not buried in a 10-row dump.

    def test_long_paragraph_splits_into_multiple_children_same_parent(self):
        # One paragraph bigger than CHILD_MAX_CHARS so it must be sentence-split.
        long_body = (
            "This agreement is governed by the laws of India. "
            "Any dispute arising under this agreement shall be subject to the "
            "exclusive jurisdiction of the courts at Pune, Maharashtra. "
            "The parties waive any claim to a different venue. "
            "The parties further agree that service of process may be effected "
            "by registered mail at the address of record maintained with the "
            "registrar of companies. "
            "No waiver of any provision shall be effective unless in writing "
            "and signed by both parties. "
            "The failure of either party to enforce any provision shall not be "
            "construed as a waiver of that provision. "
            "This agreement supersedes all prior understandings between the "
            "parties on the subject matter contained herein."
        )
        assert len(long_body) > CHILD_MAX_CHARS
        pages = [_docling_page([("H1", "Governing Law"), ("body", long_body)])]
        r = chunk_hierarchical(pages)
        assert len(r["parents"]) == 1
        assert len(r["children"]) >= 2
        assert all(c["parent_index"] == 0 for c in r["children"])
        for c in r["children"]:
            assert len(c["content"]) <= CHILD_MAX_CHARS + 50  # +50 sentence slack


class TestBboxThreading:
    def test_long_item_child_inherits_item_bbox(self):
        long_body = (
            "This agreement is governed by the laws of India. "
            "Any dispute arising under this agreement shall be subject to the "
            "exclusive jurisdiction of the courts at Pune, Maharashtra. "
            "The parties waive any claim to a different venue. "
            "Service of process may be effected by registered mail at the "
            "address of record maintained with the registrar of companies."
        )
        assert len(long_body) > 180  # above CHILD_MERGE_BELOW_CHARS
        pages = [_docling_page_with_bboxes([
            ("H1", "Governing Law", [50.0, 60.0, 500.0, 80.0]),
            ("body", long_body, [50.0, 100.0, 500.0, 300.0]),
        ])]
        r = chunk_hierarchical(pages)
        body_children = [c for c in r["children"] if not c.get("is_table")]
        assert body_children
        for c in body_children:
            assert c["bbox"] == [50.0, 100.0, 500.0, 300.0]

    def test_short_merged_items_union_their_bboxes(self):
        # Two short Q/A items on the same page get merged into one child;
        # the child's bbox is the min-x/min-y/max-x/max-y rectangle.
        pages = [_docling_page_with_bboxes([
            ("H1", "FAQ", [50.0, 40.0, 500.0, 60.0]),
            ("body", "Q: Does it run offline?", [60.0, 100.0, 400.0, 120.0]),
            ("body", "A: Yes, fully local.",     [60.0, 125.0, 420.0, 145.0]),
        ])]
        r = chunk_hierarchical(pages)
        body_children = [c for c in r["children"] if not c.get("is_table")]
        assert len(body_children) == 1
        bb = body_children[0]["bbox"]
        assert bb == [60.0, 100.0, 420.0, 145.0]

    def test_table_child_carries_item_bbox(self):
        pages = [_docling_page_with_bboxes([
            ("H1", "Fees", [50.0, 40.0, 500.0, 60.0]),
            ("table", "Service | Amount\nSetup | 10000", [50.0, 200.0, 560.0, 340.0]),
        ])]
        r = chunk_hierarchical(pages)
        table_children = [c for c in r["children"] if c.get("is_table")]
        assert len(table_children) == 1
        assert table_children[0]["bbox"] == [50.0, 200.0, 560.0, 340.0]

    def test_items_without_bbox_yield_none(self):
        # Pages without bbox info (fallback extractors) still produce children,
        # just with bbox=None so downstream code can detect and skip overlay.
        pages = [_docling_page([
            ("H1", "FAQ"),
            ("body", "Q: Does it run offline?"),
            ("body", "A: Yes, fully local."),
        ])]
        r = chunk_hierarchical(pages)
        assert all("bbox" in c for c in r["children"])
        assert all(c["bbox"] is None for c in r["children"])

    def test_child_role_and_table_flag_set(self):
        pages = [_docling_page([("H1", "X"), ("body", "y"), ("table", "a|b\nc|d")])]
        r = chunk_hierarchical(pages)
        assert all(p["chunk_role"] == "parent" for p in r["parents"])
        assert all(c["chunk_role"] == "child" for c in r["children"])


class TestFallbackPath:
    def test_no_items_falls_back_to_clause_chunker(self):
        # Pages without `items` field — e.g. from pdf_extractor fallback
        pages = [
            {"text": "1. First clause intro.\n2. Second clause here.", "page": 1, "doc_id": "d"},
            {"text": "3. Third clause on page two.", "page": 2, "doc_id": "d"},
        ]
        r = chunk_hierarchical(pages)
        assert len(r["parents"]) >= 1
        assert len(r["children"]) >= 2
        # Every child must point to a valid parent index
        for c in r["children"]:
            assert 0 <= c["parent_index"] < len(r["parents"])


class TestEdgeCases:
    def test_empty_pages_returns_empty(self):
        assert chunk_hierarchical([]) == {"parents": [], "children": []}

    def test_only_whitespace_items_produces_nothing(self):
        pages = [_docling_page([("body", "   "), ("body", "\n\t ")])]
        r = chunk_hierarchical(pages)
        assert r["children"] == []

    def test_parent_sizes_within_cap(self):
        # Many short paragraphs under one heading — should trigger parent split
        items = [("H1", "Big Section")]
        items += [("body", "Short paragraph %d. " % i * 20) for i in range(30)]
        pages = [_docling_page(items)]
        r = chunk_hierarchical(pages)
        for p in r["parents"]:
            # Allow some slack above PARENT_MAX_CHARS because the splitter is
            # paragraph-boundary-based, but it should not blow up unboundedly.
            assert len(p["content"]) <= PARENT_MAX_CHARS * 1.3
