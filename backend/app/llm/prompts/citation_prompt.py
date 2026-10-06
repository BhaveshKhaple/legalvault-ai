"""
Task 4.3 — Citation-enforcing prompt template.
Citation fix 1 — switched from [source: ...] inline tags to [#N] index tags.

Why the switch:
    Legacy prompt gave the LLM `[source: filename p.X] <content>` on each
    excerpt and asked it to echo the tag back. When two excerpts mentioned
    the same term (e.g. "prepayment charge"), the LLM would pull the fact
    from one chunk but copy the citation tag of a different, higher-ranked
    chunk — making the UI open the wrong page.

    New format: each excerpt is labelled `[#N]` with the human-readable
    source shown in parens for context. The LLM echoes `[#N]` — a single
    token that unambiguously identifies which chunk it quoted. The server
    then substitutes `[#N]` with the real `[source: filename p.X]` using
    the evidence array by index. No ambiguity, no drift.

Anti-hallucination guard: the model is told to cite only indices that
appear in the excerpts. rag_service parses the response and verifies.

No-evidence fallback: if context_chunks is empty, the prompt instructs the
model to say 'I do not have evidence...' rather than making something up.
"""

import re
from dataclasses import dataclass


@dataclass
class EvidenceChunk:
    """One piece of retrieved evidence passed to the prompt."""
    chunk_id: str       # Qdrant point UUID (not shown to the LLM)
    filename: str       # e.g. 'NDA_v2.pdf' or 'Board_Meeting.mp3'
    page: int | None    # PDF page number (1-indexed), None for audio
    ts_start: float | None  # audio timestamp in seconds, None for PDF
    content: str        # the actual chunk text shown to the model


def _format_source_label(chunk: EvidenceChunk) -> str:
    """Human-readable source label for a chunk (used in the context block).

    Shown to the LLM only as "context" — the LLM cites by [#N] index, not
    by this label.
    """
    if chunk.page is not None:
        return f"{chunk.filename} p.{chunk.page}"
    if chunk.ts_start is not None:
        return f"{chunk.filename} t.{chunk.ts_start:.1f}s"
    return chunk.filename


_SYSTEM_PROMPT = """\
You are a legal document assistant. You MUST follow these rules exactly:

1. Answer using ONLY the provided document excerpts. Never use prior knowledge.
2. After every factual claim, add the index tag of the excerpt you quoted.
   Example: "The penalty is 2% per month. [#3]"
3. Each claim needs its own citation. Do NOT group multiple claims under one citation.
4. The index MUST match the actual excerpt that contains the fact.
   If you extracted "2% penalty" from excerpt [#3], write [#3] — not [#1] or [#2].
   This is critical: the system looks up the real source using your index.
5. Only cite indices [#1] through [#N] where N is the number of excerpts shown.
   Never invent a higher index.
6. Excerpts often use different wording than the question. If ANY excerpt is
   even partially relevant, extract what you can and cite it — do not refuse.
7. ONLY refuse if the excerpts are truly off-topic. In that rare case, respond with:
   "I do not have evidence in the provided documents to answer this question."
   Do NOT refuse just because the excerpts are terse or use unfamiliar phrasing.
"""

_FEW_SHOT_EXAMPLES = """\
### Example 1
Excerpts:
  [#1] (NDA_v2.pdf p.3) The confidentiality period is five (5) years from the Effective Date.
  [#2] (NDA_v2.pdf p.7) Breach of confidentiality entitles the non-breaching party to seek injunctive relief.

Question: What is the confidentiality period and what happens if it is breached?

Answer: The confidentiality period is five years from the Effective Date. [#1]
If the confidentiality obligation is breached, the non-breaching party may seek injunctive relief. [#2]

### Example 2
Excerpts:
  [#1] (Board_Meeting.mp3 t.734.5s) The board approved a maximum borrowing limit of fifty crore rupees.
  [#2] (Board_Meeting.mp3 t.812.0s) Any borrowing above this limit requires a special resolution.

Question: What is the approved borrowing limit?

Answer: The board approved a maximum borrowing limit of fifty crore rupees. [#1]
Any amount above this limit requires a special resolution. [#2]

### Example 3 — critical: cite the chunk that actually has the fact
Excerpts:
  [#1] (Loan.pdf p.3) The borrower may exit within three (3) days of disbursement without any prepayment charge.
  [#2] (Loan.pdf p.2) Prepayment charge | 2.00% of principal amount prepaid | Applicable to part or full prepayment.

Question: What is the prepayment charge on the loan?

Answer: The prepayment charge is 2.00% of the principal amount prepaid. [#2]

NOTE: Even though [#1] uses the exact phrase "prepayment charge", the actual
value (2.00% of principal) comes from [#2]. ALWAYS cite the excerpt that
contains the fact you quoted, not the one that only mentions the topic.

"""


def build_citation_prompt(
    question: str,
    evidence: list[EvidenceChunk],
) -> str:
    """Build the full prompt string sent to Ollama.

    Args:
        question: The user's natural-language question.
        evidence: Top-k reranked chunks from reranker.rerank().
                  If empty, the prompt instructs the model to say so.

    Returns:
        A complete prompt string ready to pass to ollama_client.generate()
        as the ``prompt`` argument (context already embedded).
    """
    if not evidence:
        return (
            f"{_SYSTEM_PROMPT}\n\n"
            "No document excerpts are available for this question.\n\n"
            f"Question: {question}\n\n"
            "Answer: I do not have evidence in the provided documents to answer this question."
        )

    # Each excerpt gets an [#N] tag + human source label in parens for the LLM's context.
    excerpts_lines = []
    for i, chunk in enumerate(evidence, start=1):
        label = _format_source_label(chunk)
        excerpts_lines.append(f"  [#{i}] ({label}) {chunk.content}")

    excerpts_block = "\n".join(excerpts_lines)

    return (
        f"{_SYSTEM_PROMPT}\n\n"
        f"{_FEW_SHOT_EXAMPLES}"
        f"### Your turn\n"
        f"Excerpts:\n{excerpts_block}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )


# Index-citation parsing. Matches [#1], [#12], etc.
_INDEX_CITE = re.compile(r"\[#(\d+)\]")

# Legacy source-citation parsing (for backward compat tests + verification).
_LEGACY_SOURCE = re.compile(r"\[source:\s*([^\]]+?)\s*\]")


def substitute_citations(
    answer: str,
    evidence: list[EvidenceChunk],
) -> tuple[str, set[int]]:
    """Replace [#N] in the LLM output with the real [source: filename p.N] tag.

    Called by rag_service after the LLM returns. The LLM commits to a specific
    chunk by index; we substitute to the human-readable source so the UI shows
    the right page and the citation audit log can verify.

    Args:
        answer: Raw LLM response text, possibly containing [#1], [#2], ... tags.
        evidence: The same evidence list that was passed to build_citation_prompt.

    Returns:
        (substituted_text, cited_indices_1based)
        Out-of-range indices are left as-is in the text (easy to spot in review)
        and still returned in the cited set so the audit log catches them.
    """
    cited: set[int] = set()

    def _repl(m: re.Match) -> str:
        idx = int(m.group(1))
        cited.add(idx)
        if 1 <= idx <= len(evidence):
            ch = evidence[idx - 1]
            if ch.page is not None:
                return f"[source: {ch.filename} p.{ch.page}]"
            if ch.ts_start is not None:
                return f"[source: {ch.filename} t.{ch.ts_start:.1f}s]"
            return f"[source: {ch.filename}]"
        # Preserve the raw [#N] so a human reviewer can see the model hallucinated
        return m.group(0)

    return _INDEX_CITE.sub(_repl, answer), cited


def extract_cited_ids(llm_response: str) -> set[str]:
    """Parse [source: ...] tags from a response (post-substitution).

    Still returns filenames because the audit log stores them for traceability.
    """
    return {m.group(1).strip() for m in _LEGACY_SOURCE.finditer(llm_response)}


def extract_cited_indices(llm_response: str) -> set[int]:
    """Parse [#N] tags from a raw LLM response (pre-substitution).

    Useful for the citation-verification unit test.
    """
    return {int(m.group(1)) for m in _INDEX_CITE.finditer(llm_response)}
