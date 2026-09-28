"""
Task 4.3 — Citation-enforcing prompt template.

Left alone, LLMs answer legal questions without proof. For legal work that
is useless — every claim must cite its source so a lawyer can verify it.

This module builds the final prompt that gets sent to Ollama. It uses a
few-shot approach: two worked examples show the model exactly what format
we want, making 'see how it's done' training far more effective than
instruction alone.

Citation format: [source: <filename> p.<page>] for PDFs
                 [source: <filename> t.<ts_start>s] for audio

Anti-hallucination guard: the system prompt explicitly tells the model that
it MUST only cite IDs that appeared in the context block. The post-check
in Task 6.3's RAG service will verify this server-side.

No-evidence fallback: if context_chunks is empty, the prompt instructs the
model to say 'I do not have evidence...' rather than making something up.
"""

from dataclasses import dataclass


@dataclass
class EvidenceChunk:
    """One piece of retrieved evidence passed to the prompt."""
    chunk_id: str       # Qdrant point UUID — must appear in [source: ] tag
    filename: str       # e.g. 'NDA_v2.pdf' or 'Board_Meeting.mp3'
    page: int | None    # PDF page number (1-indexed), None for audio
    ts_start: float | None  # audio timestamp in seconds, None for PDF
    content: str        # the actual chunk text shown to the model


def _format_source_label(chunk: EvidenceChunk) -> str:
    """Produce the human-readable [source: ...] tag for this chunk."""
    if chunk.page is not None:
        return f"[source: {chunk.filename} p.{chunk.page}]"
    if chunk.ts_start is not None:
        return f"[source: {chunk.filename} t.{chunk.ts_start:.1f}s]"
    return f"[source: {chunk.filename}]"


_SYSTEM_PROMPT = """\
You are a legal document assistant. You MUST follow these rules exactly:

1. Answer using ONLY the provided document excerpts. Never use prior knowledge.
2. After every factual claim, add the citation tag exactly as shown in the excerpts header.
   Example: "The penalty is 2% per month. [source: Contract.pdf p.12]"
3. Each claim needs its own citation. Do NOT group multiple claims under one citation.
4. Only cite source IDs that appear in the excerpts below. Never invent a filename or page number.
5. If the excerpts do not contain enough information to answer, respond with:
   "I do not have evidence in the provided documents to answer this question."
   Do NOT guess or paraphrase from memory.
"""

_FEW_SHOT_EXAMPLES = """\
### Example 1
Excerpts:
  [source: NDA_v2.pdf p.3] The confidentiality period is five (5) years from the Effective Date.
  [source: NDA_v2.pdf p.7] Breach of confidentiality entitles the non-breaching party to seek injunctive relief.

Question: What is the confidentiality period and what happens if it is breached?

Answer: The confidentiality period is five years from the Effective Date. [source: NDA_v2.pdf p.3]
If the confidentiality obligation is breached, the non-breaching party may seek injunctive relief. [source: NDA_v2.pdf p.7]

### Example 2
Excerpts:
  [source: Board_Meeting.mp3 t.734.5s] The board approved a maximum borrowing limit of fifty crore rupees.
  [source: Board_Meeting.mp3 t.812.0s] Any borrowing above this limit requires a special resolution.

Question: What is the approved borrowing limit?

Answer: The board approved a maximum borrowing limit of fifty crore rupees. [source: Board_Meeting.mp3 t.734.5s]
Any amount above this limit requires a special resolution. [source: Board_Meeting.mp3 t.812.0s]

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

    # Build the excerpts block — each chunk gets its citation label as a prefix
    excerpts_lines = []
    for chunk in evidence:
        label = _format_source_label(chunk)
        excerpts_lines.append(f"  {label} {chunk.content}")

    excerpts_block = "\n".join(excerpts_lines)

    return (
        f"{_SYSTEM_PROMPT}\n\n"
        f"{_FEW_SHOT_EXAMPLES}"
        f"### Your turn\n"
        f"Excerpts:\n{excerpts_block}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )


def extract_cited_ids(llm_response: str) -> set[str]:
    """Parse [source: ...] tags from an LLM response.

    Used by the post-check in Task 6.3 to verify every citation references
    a chunk that was actually in the context.

    Args:
        llm_response: Raw text from ollama_client.generate().

    Returns:
        Set of filename strings referenced in [source: ...] tags.
        (We match on filename, not chunk_id, since the model never sees UUIDs.)
    """
    import re
    pattern = re.compile(r"\[source:\s*([^\]]+?)\s*\]")
    return {m.group(1).strip() for m in pattern.finditer(llm_response)}
