"""Tests for Task 1.2 — clause_chunker.chunk_legal_doc()."""

import pytest

from backend.app.ingestion.clause_chunker import chunk_legal_doc


# ─── helpers ──────────────────────────────────────────────────────────────────


def _pages(text_by_page: list[str], doc_id: str = "contract") -> list[dict]:
    """Build pages input matching pdf_extractor's output shape."""
    return [{"text": t, "page": i + 1, "doc_id": doc_id} for i, t in enumerate(text_by_page)]


# ─── return shape ─────────────────────────────────────────────────────────────


class TestReturnShape:
    def test_returns_list(self):
        result = chunk_legal_doc(_pages(["1. Hello"]))
        assert isinstance(result, list)

    def test_dict_keys(self):
        result = chunk_legal_doc(_pages(["1. Hello world"]))
        assert set(result[0].keys()) == {"content", "page", "section_title", "doc_id"}

    def test_doc_id_propagated(self):
        result = chunk_legal_doc(_pages(["1. Foo"], doc_id="my_contract"))
        assert result[0]["doc_id"] == "my_contract"

    def test_empty_input(self):
        assert chunk_legal_doc([]) == []


# ─── boundary detection ──────────────────────────────────────────────────────


class TestNumberedClauses:
    def test_top_level_clause(self):
        pages = _pages(["1. First clause text.\n2. Second clause text."])
        result = chunk_legal_doc(pages)
        assert len(result) == 2
        assert "First clause" in result[0]["content"]
        assert "Second clause" in result[1]["content"]

    def test_nested_clause(self):
        text = "3. Termination\n3.1 Either party may terminate.\n3.2 Notice period is 30 days."
        result = chunk_legal_doc(_pages([text]))
        assert len(result) == 3
        assert result[0]["section_title"].startswith("3.")
        assert result[1]["section_title"].startswith("3.1")
        assert result[2]["section_title"].startswith("3.2")

    def test_deep_nesting(self):
        text = "1.1.2.4 Very deep clause here."
        result = chunk_legal_doc(_pages([text]))
        assert len(result) == 1
        assert result[0]["section_title"].startswith("1.1.2.4")

    def test_inline_mention_not_a_boundary(self):
        """'see clause 3.1' inside prose must NOT start a new chunk."""
        text = "1. Some clause with a reference to see clause 3.1 inside it."
        result = chunk_legal_doc(_pages([text]))
        assert len(result) == 1
        assert "see clause 3.1" in result[0]["content"]


class TestNamedSections:
    def test_section_keyword(self):
        text = "Section 1 Definitions\nSome text.\nSection 2 Obligations\nMore text."
        result = chunk_legal_doc(_pages([text]))
        assert len(result) == 2
        assert "Section 1" in result[0]["section_title"]
        assert "Section 2" in result[1]["section_title"]

    def test_article_roman(self):
        text = "Article IX Governing Law\nThis agreement is governed by..."
        result = chunk_legal_doc(_pages([text]))
        assert len(result) == 1
        assert "Article IX" in result[0]["section_title"]

    def test_chapter(self):
        text = "Chapter 2 Payment Terms\nThe fee is 10%."
        result = chunk_legal_doc(_pages([text]))
        assert len(result) == 1


class TestAllCapsHeadings:
    def test_short_allcaps(self):
        text = "DEFINITIONS\nContract means this agreement.\nTERMINATION\n30 days notice."
        result = chunk_legal_doc(_pages([text]))
        assert len(result) == 2
        assert result[0]["section_title"] == "DEFINITIONS"
        assert result[1]["section_title"] == "TERMINATION"

    def test_multiword_allcaps(self):
        text = "TERMINATION AND SURVIVAL\nContent here."
        result = chunk_legal_doc(_pages([text]))
        assert len(result) == 1

    def test_long_allcaps_line_not_treated_as_heading(self):
        """A 10-word ALL-CAPS line is prose (e.g. emphasised paragraph), not a heading."""
        text = "THIS IS A VERY LONG ALL CAPS SENTENCE THAT IS ACTUALLY PROSE AND NOT A HEADING"
        result = chunk_legal_doc(_pages([text]))
        # Falls into Preamble because no other boundary was hit
        assert len(result) == 1
        assert result[0]["section_title"] == "Preamble"


# ─── page tracking + cross-page clauses ──────────────────────────────────────


class TestPageTracking:
    def test_page_number_is_start_page(self):
        pages = _pages(["1. First clause.", "2. Second clause."])
        result = chunk_legal_doc(pages)
        assert result[0]["page"] == 1
        assert result[1]["page"] == 2

    def test_clause_spans_pages_stays_one_chunk(self):
        """CRITICAL: a clause that starts on page 1 and continues on page 2 = ONE chunk."""
        pages = _pages(
            [
                "3.1 Termination: This contract may be terminated by",
                "either party upon thirty (30) days written notice.\n4. Governing Law.",
            ]
        )
        result = chunk_legal_doc(pages)
        assert len(result) == 2
        # First chunk is 3.1, starts page 1
        assert result[0]["section_title"].startswith("3.1")
        assert result[0]["page"] == 1
        assert "Termination" in result[0]["content"]
        assert "thirty (30) days" in result[0]["content"]  # continues from page 2
        # Second chunk is 4, starts page 2
        assert result[1]["section_title"].startswith("4.")
        assert result[1]["page"] == 2


# ─── preamble handling ──────────────────────────────────────────────────────


class TestPreamble:
    def test_text_before_first_boundary_becomes_preamble(self):
        text = "This is a contract between parties.\nSigned on 2026-09-23.\n1. Definitions.\nContract means..."
        result = chunk_legal_doc(_pages([text]))
        assert len(result) == 2
        assert result[0]["section_title"] == "Preamble"
        assert "contract between parties" in result[0]["content"]
        assert result[1]["section_title"].startswith("1.")

    def test_no_preamble_when_starts_with_boundary(self):
        text = "1. First clause.\n2. Second clause."
        result = chunk_legal_doc(_pages([text]))
        assert len(result) == 2
        assert all(r["section_title"] != "Preamble" for r in result)
