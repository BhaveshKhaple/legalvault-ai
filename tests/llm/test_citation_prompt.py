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
        result = build_citation_prompt("Q?", [_pdf_chunk(filename="NDA.pdf", page=3)])
        assert "[source: NDA.pdf p.3]" in result

    def test_audio_source_label_format(self):
        result = build_citation_prompt("Q?", [_audio_chunk(filename="Call.mp3", ts=123.4)])
        assert "[source: Call.mp3 t.123.4s]" in result

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
        assert "[source: A.pdf p.1]" in result
        assert "[source: B.pdf p.2]" in result
        assert "[source: C.mp3 t.10.0s]" in result
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
