"""
Task 5.2 — Reverse-RAG claim verifier (the flagship Report Shield feature).

Normal RAG:    user asks a question → find evidence → LLM answers.
Reverse RAG:   we already have a CLAIM (from claim_extractor) → search the
               case documents for evidence → label the claim as:
                 VERIFIED      — supporting evidence found (score ≥ threshold)
                 UNVERIFIED    — no relevant evidence at all (score < threshold)
                 CONTRADICTED  — evidence found but the LLM judges it opposes the claim

Pipeline per claim:
  1. Embed the claim text
  2. Qdrant semantic search restricted to case_id
  3. Cross-encoder rerank — highest score determines "is evidence relevant?"
  4. If relevant evidence exists → ask LLM whether it SUPPORTS or CONTRADICTS
  5. Return {status, evidence[]} dict

Thresholds (tunable):
  RELEVANCE_THRESHOLD = 0.3   — reranker score below which we call "unverified"
  LLM is only called when relevant evidence exists, saving latency on empty cases.
"""

import logging
from typing import Literal

logger = logging.getLogger(__name__)

# Reranker score below this → UNVERIFIED (not enough overlap to judge)
_RELEVANCE_THRESHOLD = 0.3
# Top-k evidence chunks fed to the LLM for the support/contradict judgment
_EVIDENCE_K = 3


Verdict = Literal["Verified", "Unverified", "Contradicted"]


def _get_embedding_model() -> str:
    from backend.app.llm.model_selector import get_model
    return get_model().embedding


def _get_llm_model() -> str:
    from backend.app.llm.model_selector import get_model
    return get_model().llm


def _embed_claim(claim: str) -> list[float]:
    from backend.app.retrieval.embeddings import embed
    return embed(claim, model_name=_get_embedding_model())


def _search_evidence(claim_vec: list[float], case_id: str, top_k: int = 10) -> list[dict]:
    from backend.app.retrieval.vector_store import query as qdrant_query
    return qdrant_query(claim_vec, case_id, top_k=top_k)


def _rerank_evidence(claim: str, candidates: list[dict], k: int) -> list[dict]:
    from backend.app.retrieval.reranker import rerank
    return rerank(claim, candidates, k=k)


def _llm_support_or_contradict(claim: str, evidence_texts: list[str], llm_model: str) -> Verdict:
    """Ask the LLM whether the evidence SUPPORTS or CONTRADICTS the claim."""
    from backend.app.llm.ollama_client import generate

    excerpts = "\n".join(f"  [{i+1}] {t}" for i, t in enumerate(evidence_texts))
    prompt = (
        "You are a legal document analyst. Decide whether the excerpts below "
        "SUPPORT or CONTRADICT the given claim.\n\n"
        f"Claim: {claim}\n\n"
        f"Excerpts:\n{excerpts}\n\n"
        "Reply with exactly one word: SUPPORT or CONTRADICT. Nothing else."
    )
    try:
        raw = generate(prompt, model=llm_model).strip().upper()
        if "CONTRADICT" in raw:
            return "Contradicted"
        return "Verified"
    except Exception as exc:
        logger.warning("LLM call failed in verifier: %s — defaulting to Verified", exc)
        return "Verified"


def verify_claim(claim: str, case_id: str) -> dict:
    """Verify a single claim against a case's indexed documents.

    Args:
        claim:   A single factual statement extracted from a report.
        case_id: UUID string of the case whose documents to search.

    Returns:
        dict::
            {
                "claim":   str,
                "status":  "Verified" | "Unverified" | "Contradicted",
                "evidence": [
                    {"content": str, "score": float, "page": int|None,
                     "filename": str|None, "section_title": str|None},
                    ...
                ],
            }
    """
    claim = claim.strip()
    if not claim:
        return {"claim": claim, "status": "Unverified", "evidence": []}

    # 1. Embed + search
    claim_vec = _embed_claim(claim)
    candidates = _search_evidence(claim_vec, case_id, top_k=10)

    if not candidates:
        return {"claim": claim, "status": "Unverified", "evidence": []}

    # 2. Rerank
    top = _rerank_evidence(claim, candidates, k=_EVIDENCE_K)

    if not top:
        return {"claim": claim, "status": "Unverified", "evidence": []}

    best_score = float(top[0].get("rerank_score", top[0].get("score", 0)))

    if best_score < _RELEVANCE_THRESHOLD:
        return {"claim": claim, "status": "Unverified", "evidence": []}

    # 3. LLM judgment — support or contradict?
    evidence_texts = [
        (c.get("payload", {}) or {}).get("content") or c.get("content", "")
        for c in top
    ]
    verdict: Verdict = _llm_support_or_contradict(claim, evidence_texts, _get_llm_model())

    # 4. Format evidence for the caller
    evidence_out = [
        {
            "content": (c.get("payload", {}) or {}).get("content", ""),
            "score": float(c.get("rerank_score", c.get("score", 0))),
            "page": (c.get("payload", {}) or {}).get("page"),
            "filename": (c.get("payload", {}) or {}).get("filename"),
            "section_title": (c.get("payload", {}) or {}).get("section_title"),
        }
        for c in top
    ]

    return {"claim": claim, "status": verdict, "evidence": evidence_out}


def verify_claims(claims: list[str], case_id: str) -> list[dict]:
    """Verify multiple claims. Returns one result dict per claim."""
    return [verify_claim(c, case_id) for c in claims]
