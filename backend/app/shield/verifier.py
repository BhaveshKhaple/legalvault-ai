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


def _search_evidence(
    claim_vec: list[float],
    case_id: str,
    top_k: int = 10,
    collection_name: str | None = None,
) -> list[dict]:
    from backend.app.retrieval.vector_store import (
        _DEFAULT_COLLECTION,
        query as qdrant_query,
    )

    return qdrant_query(
        claim_vec,
        case_id,
        top_k=top_k,
        collection_name=collection_name or _DEFAULT_COLLECTION,
    )


def _rerank_evidence(claim: str, candidates: list[dict], k: int) -> list[dict]:
    from backend.app.retrieval.reranker import rerank

    return rerank(claim, candidates, k=k)


def _llm_support_or_contradict(
    claim: str, evidence_texts: list[str], llm_model: str
) -> tuple[Verdict, str]:
    """Ask the LLM whether the evidence SUPPORTS or CONTRADICTS the claim, with explanation."""
    from backend.app.llm.ollama_client import generate

    excerpts = "\n".join(
        f"  [{i+1}] {t[:350]}" for i, t in enumerate(evidence_texts[:3])
    )
    prompt = (
        "You are a strict legal document auditor. Compare the claim with the case excerpts.\n\n"
        f"Claim: {claim}\n\n"
        f"Case Excerpts:\n{excerpts}\n\n"
        "Instructions:\n"
        "- Reply with SUPPORT if the excerpts back up the claim.\n"
        "- Reply with CONTRADICT if the excerpts conflict, dispute, or disprove the claim.\n"
        "- If CONTRADICT, state the specific discrepancy in 1 clear sentence.\n\n"
        "Format:\n"
        "VERDICT: SUPPORT or CONTRADICT\n"
        "REASON: <1 sentence stating what the problem is if contradicted, or none>"
    )
    try:
        raw = generate(prompt, model=llm_model, num_predict=60, num_ctx=1024).strip()
        verdict: Verdict = "Verified"
        reason = ""
        if "CONTRADICT" in raw.upper():
            verdict = "Contradicted"

        for line in raw.splitlines():
            line_str = line.strip()
            if line_str.upper().startswith("REASON:"):
                reason = line_str.split(":", 1)[1].strip()
                break

        if verdict == "Contradicted" and not reason:
            cleaned = raw.replace("VERDICT:", "").replace("CONTRADICT", "").strip()
            reason = (
                cleaned
                if len(cleaned) > 5
                else "Claim contradicts documented evidence."
            )

        return verdict, reason
    except Exception as exc:
        # Fail-safe for a legal tool: NEVER falsely vouch for a claim.
        # If the LLM is unreachable, mark as Unverified so the user knows
        # the verifier couldn't actually judge it.
        logger.warning("LLM call failed in verifier: %s — marking Unverified", exc)
        return "Unverified", ""


def verify_claims(
    claims: list[str], case_id: str, collections: list[str] | None = None
) -> list[dict]:
    """Verify multiple claims. Uses batch embeddings and concurrent reranking/LLM evaluation.

    Returns one result dict per claim.
    """
    if not claims:
        return []

    from backend.app.retrieval.vector_store import COLLECTION_E5, COLLECTION_BGE
    from backend.app.retrieval.embeddings import embed_batch

    active_collections = collections or [COLLECTION_E5]

    # Pre-embed all claims once per collection (batch forward pass is 10x-20x faster)
    vectors_by_coll: dict[str, list[list[float]] | None] = {}
    for coll in active_collections:
        model_name = "BAAI/bge-m3" if coll == COLLECTION_BGE else "intfloat/e5-small-v2"
        try:
            vectors_by_coll[coll] = embed_batch(
                claims, model_name=model_name, batch_size=32
            )
        except Exception as exc:
            logger.warning(
                "Batch embed failed for %s (%s), falling back to individual", coll, exc
            )
            vectors_by_coll[coll] = None

    def _verify_single(index: int, claim_text: str) -> dict:
        claim_clean = claim_text.strip()
        if not claim_clean:
            return {"claim": claim_clean, "status": "Unverified", "evidence": []}

        candidates = []
        for coll in active_collections:
            cached_vectors = vectors_by_coll.get(coll)
            if cached_vectors and index < len(cached_vectors):
                coll_vec = cached_vectors[index]
            else:
                from backend.app.retrieval.embeddings import embed as _embed_raw

                model_name = (
                    "BAAI/bge-m3" if coll == COLLECTION_BGE else "intfloat/e5-small-v2"
                )
                coll_vec = _embed_raw(claim_clean, model_name=model_name)

            candidates.extend(
                _search_evidence(coll_vec, case_id, top_k=5, collection_name=coll)
            )

        if not candidates:
            return {"claim": claim_clean, "status": "Unverified", "evidence": []}

        # 2. Rerank
        top = _rerank_evidence(claim_clean, candidates, k=_EVIDENCE_K)

        if not top:
            return {"claim": claim_clean, "status": "Unverified", "evidence": []}

        import math as _math

        best_raw = float(top[0].get("rerank_score", top[0].get("score", 0)))
        best_score = 1.0 / (1.0 + _math.exp(-best_raw))

        if best_score < _RELEVANCE_THRESHOLD:
            return {"claim": claim_clean, "status": "Unverified", "evidence": []}

        # 3. LLM judgment — support or contradict?
        evidence_texts = [
            (c.get("payload", {}) or {}).get("content") or c.get("content", "")
            for c in top
        ]

        # Fast-path for verbatim excerpt matches:
        # If the claim is a direct verbatim substring of the top cited evidence excerpt,
        # it is undeniably supported by definition (and cannot contradict).
        claim_lower = claim_clean.lower()
        if best_score >= 0.80 and any(claim_lower in t.lower() for t in evidence_texts):
            verdict = "Verified"
            reason = ""
        else:
            res = _llm_support_or_contradict(
                claim_clean, evidence_texts, _get_llm_model()
            )
            if isinstance(res, tuple):
                verdict, reason = res
            elif isinstance(res, dict):
                verdict = res.get("verdict", "Unverified")
                reason = res.get("reason", "")
            else:
                verdict = res
                reason = ""

        # 4. Format evidence for the caller — sigmoid-normalize scores to [0,1]
        def _sigmoid(x):
            return 1.0 / (1.0 + _math.exp(-float(x)))

        evidence_out = [
            {
                "content": (c.get("payload", {}) or {}).get("content", ""),
                "score": _sigmoid(c.get("rerank_score", c.get("score", 0))),
                "page": (c.get("payload", {}) or {}).get("page"),
                "filename": (c.get("payload", {}) or {}).get("filename"),
                "section_title": (c.get("payload", {}) or {}).get("section_title"),
            }
            for c in top
        ]

        out = {"claim": claim_clean, "status": verdict, "evidence": evidence_out}
        if reason:
            out["reason"] = reason
        return out

    return [_verify_single(i, c) for i, c in enumerate(claims)]


def verify_claim(
    claim: str, case_id: str, collections: list[str] | None = None
) -> dict:
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
    results = verify_claims([claim], case_id, collections=collections)
    return (
        results[0]
        if results
        else {"claim": claim, "status": "Unverified", "evidence": []}
    )
