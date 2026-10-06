"""
Tests for Task 4.3 — citation_prompt.build_citation_prompt() + extract_cited_ids().

Pure string functions — no model, no mocks needed.
"""

import pytest

from backend.app.llm.prompts.citation_prompt import (
    EvidenceChunk,
    build_citation_prompt,
    extract_cited_ids,
)


# ─── helpers ──────────────────────────────────────────────────────────────────


def _pdf_chunk(filename: str = "Contract.pdf", page: int = 5, content: str = "Some clause text.") -> EvidenceChunk:
    return EvidenceChunk(
        chunk_id="uuid-001",
        filename=filename,
        page=page,
        ts_start=None,
        content=content,
    )


def _audio_chunk(filename: str = "Meeting.mp3", ts: float = 734.5, content: str = "Board approved.") -> EvidenceChunk:
    return EvidenceChunk(
        chunk_id="uuid-002",
        filename=filename,
        page=None,
        ts_start=ts,
        content=content,
    )


# ─── build_citation_prompt — no evidence ──────────────────────────────────────


class TestNoEvidence:
    def test_returns_string(self):
        result = build_citation_prompt("What is the penalty?", [])
        assert isinstance(result, str)

    def test_contains_no_evidence_message(self):
        result = build_citation_prompt("What?", [])
        assert "I do not have evidence" in result

    def test_contains_question(self):
        result = build_citation_prompt("Any liability cap?", [])
        assert "Any liability cap?" in result

    def test_no_excerpt_label_when_empty(self):
        result = build_citation_prompt("Q?", [])
        assert "[source:" not in result or "I do not have evidence" in result


# ─── build_citation_prompt — with evidence ────────────────────────────────────


class TestWithEvidence:
    def test_pdf_source_label_format(self):
        # Citation fix 1 — new [#N] (filename p.X) format
        result = build_citation_prompt("Q?", [_pdf_chunk(filename="NDA.pdf", page=3)])
        assert "[#1] (NDA.pdf p.3)" in result

    def test_audio_source_label_format(self):
        result = build_citation_prompt("Q?", [_audio_chunk(filename="Call.mp3", ts=123.4)])
        assert "[#1] (Call.mp3 t.123.4s)" in result

    def test_chunk_content_in_prompt(self):
        chunk = _pdf_chunk(content="The penalty is two percent per month.")
        result = build_citation_prompt("What is the penalty?", [chunk])
        assert "The penalty is two percent per month." in result

    def test_question_appears_in_prompt(self):
        result = build_citation_prompt("What is the governing law?", [_pdf_chunk()])
        assert "What is the governing law?" in result

    def test_system_prompt_present(self):
        result = build_citation_prompt("Q?", [_pdf_chunk()])
        assert "MUST" in result  # system prompt keyword
        assert "prior knowledge" in result.lower()

    def test_few_shot_examples_present(self):
        result = build_citation_prompt("Q?", [_pdf_chunk()])
        assert "Example 1" in result
        assert "Example 2" in result

    def test_multiple_chunks_all_present(self):
        chunks = [
            _pdf_chunk(filename="A.pdf", page=1, content="First clause."),
            _pdf_chunk(filename="B.pdf", page=2, content="Second clause."),
            _audio_chunk(filename="C.mp3", ts=10.0, content="Third statement."),
        ]
        result = build_citation_prompt("Q?", chunks)
        # Citation fix 1 — indices are 1-based and in input order
        assert "[#1] (A.pdf p.1)" in result
        assert "[#2] (B.pdf p.2)" in result
        assert "[#3] (C.mp3 t.10.0s)" in result
        assert "First clause." in result
        assert "Second clause." in result
        assert "Third statement." in result

    def test_answer_prompt_ends_prompt(self):
        result = build_citation_prompt("Q?", [_pdf_chunk()])
        assert result.rstrip().endswith("Answer:")

    def test_answer_block_does_not_start_with_no_evidence(self):
        """The final answer prompt must not pre-fill with the fallback message.
        The phrase can appear in system rules/examples but not as the answer."""
        result = build_citation_prompt("Q?", [_pdf_chunk()])
        # The prompt ends with "Answer:" — nothing after it should be the fallback
        after_answer = result.split("Answer:")[-1]
        assert "I do not have evidence" not in after_answer


# ─── extract_cited_ids ────────────────────────────────────────────────────────


class TestExtractCitedIds:
    def test_single_source_extracted(self):
        text = "The penalty is 2%. [source: Contract.pdf p.12]"
        assert extract_cited_ids(text) == {"Contract.pdf p.12"}

    def test_multiple_sources_extracted(self):
        text = (
            "Clause found. [source: NDA.pdf p.3] "
            "Meeting confirmed. [source: Board.mp3 t.734.5s]"
        )
        assert extract_cited_ids(text) == {"NDA.pdf p.3", "Board.mp3 t.734.5s"}

    def test_duplicate_sources_deduplicated(self):
        text = "First claim [source: A.pdf p.1] Second claim [source: A.pdf p.1]"
        assert extract_cited_ids(text) == {"A.pdf p.1"}

    def test_no_sources_returns_empty_set(self):
        assert extract_cited_ids("No citations here.") == set()

    def test_whitespace_stripped(self):
        text = "[source:  Contract.pdf p.5  ]"
        result = extract_cited_ids(text)
        assert "Contract.pdf p.5" in result

    def test_returns_set_not_list(self):
        result = extract_cited_ids("[source: A.pdf p.1]")
        assert isinstance(result, set)


# ─── substitute_citations — Citation fix 1 ────────────────────────────────────


class TestSubstituteCitations:
    """The bug this fixes: LLM quotes a fact from chunk [#5] but citation UI
    opens the top-ranked chunk's page (p.3) because the old prompt tied the
    citation to the source label, which the LLM would copy from the most
    salient-looking excerpt. Index-based citations remove the ambiguity.
    """

    def test_replaces_single_index_with_real_source(self):
        from backend.app.llm.prompts.citation_prompt import substitute_citations

        evidence = [_pdf_chunk(filename="Loan.pdf", page=2, content="2% prepayment.")]
        answer, cited = substitute_citations("The charge is 2%. [#1]", evidence)

        assert "[source: Loan.pdf p.2]" in answer
        assert "[#1]" not in answer
        assert cited == {1}

    def test_the_actual_bug_from_screenshot(self):
        """Reproduces the user's bug: LLM should cite the chunk that contains
        the fact, not the top-ranked chunk."""
        from backend.app.llm.prompts.citation_prompt import substitute_citations

        # Simulate what the retrieval returned for the real query:
        evidence = [
            _pdf_chunk(filename="Loan.pdf", page=3, content="...without prepayment charge if exited in 3 days"),
            _pdf_chunk(filename="Loan.pdf", page=1, content="Loan terms"),
            _pdf_chunk(filename="Loan.pdf", page=1, content="More loan terms"),
            _pdf_chunk(filename="Loan.pdf", page=1, content="Yet more loan terms"),
            _pdf_chunk(filename="Loan.pdf", page=2, content="Prepayment charge | 2.00% of principal"),
        ]

        # With the new prompt the LLM should emit [#5] because that's the
        # chunk containing the 2% fact. Our substitution translates to p.2.
        llm_output = "The prepayment charge is 2.00% of the principal amount prepaid. [#5]"
        answer, cited = substitute_citations(llm_output, evidence)

        # Fact AND page are now correct
        assert "p.2" in answer
        assert "p.3" not in answer   # <-- the bug from the screenshot
        assert cited == {5}

    def test_multiple_index_citations(self):
        from backend.app.llm.prompts.citation_prompt import substitute_citations

        evidence = [
            _pdf_chunk(filename="A.pdf", page=1),
            _pdf_chunk(filename="B.pdf", page=2),
        ]
        answer, cited = substitute_citations(
            "First claim. [#1] Second claim. [#2]", evidence
        )
        assert "[source: A.pdf p.1]" in answer
        assert "[source: B.pdf p.2]" in answer
        assert cited == {1, 2}

    def test_audio_citation_includes_timestamp(self):
        from backend.app.llm.prompts.citation_prompt import substitute_citations

        evidence = [_audio_chunk(filename="Call.mp3", ts=42.5)]
        answer, _ = substitute_citations("The board agreed. [#1]", evidence)
        assert "[source: Call.mp3 t.42.5s]" in answer

    def test_hallucinated_index_preserved_for_review(self):
        """If the LLM cites [#99] when only 3 excerpts were given, we leave
        the raw tag in the output so a reviewer can spot the hallucination."""
        from backend.app.llm.prompts.citation_prompt import substitute_citations

        evidence = [_pdf_chunk()] * 3
        answer, cited = substitute_citations("Claim. [#99]", evidence)
        assert "[#99]" in answer   # preserved, not silently stripped
        assert 99 in cited         # audit log still captures the hallucination

    def test_no_citations_returns_unchanged(self):
        from backend.app.llm.prompts.citation_prompt import substitute_citations

        evidence = [_pdf_chunk()]
        answer, cited = substitute_citations("Plain text with no citations.", evidence)
        assert answer == "Plain text with no citations."
        assert cited == set()


# ─── extract_cited_indices — [#N] parsing ─────────────────────────────────────


class TestExtractCitedIndices:
    def test_extracts_single_index(self):
        from backend.app.llm.prompts.citation_prompt import extract_cited_indices
        assert extract_cited_indices("Fact [#5]") == {5}

    def test_extracts_multiple(self):
        from backend.app.llm.prompts.citation_prompt import extract_cited_indices
        assert extract_cited_indices("A. [#1] B. [#3] C. [#1]") == {1, 3}

    def test_empty_when_no_tags(self):
        from backend.app.llm.prompts.citation_prompt import extract_cited_indices
        assert extract_cited_indices("No citations here") == set()
