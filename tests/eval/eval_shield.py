"""
Task 10.2 — Report Shield evaluation.

Runs the Report Shield pipeline on 5 synthetic reports with deliberately
planted flaws. Pass criterion: >=4/5 reports have at least one of their
planted flaws correctly detected.

Flaw types tested:
    - missing_section — gap_detector should flag the named section as missing
    - contradiction   — verifier should mark at least one claim as Contradicted
                        (requires LLM; use --real-llm to actually call Ollama)

Pipeline exercised per report:
    text → claim_extractor → claims
    text → gap_detector    → missing sections
    claims + source corpus → verifier → Verified/Unverified/Contradicted
    all above → scoring.aggregate → trust_score

Usage:
    # Fast — mocks LLM, exercises claim extraction + gap detection in full,
    # and uses rerank scores (NOT LLM judgment) for contradiction detection:
    backend/venv/Scripts/python tests/eval/eval_shield.py

    # Real pipeline — calls Ollama for every claim (slow: ~20-30 min on CPU,
    # ~2-3 min on GPU). Use when you need true-to-prod validation:
    backend/venv/Scripts/python tests/eval/eval_shield.py --real-llm

Uses a scratch Qdrant directory so ./data/qdrant/ is untouched.
Exits 1 if detection rate < 4/5 (CI-friendly).
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

_SCRATCH_QDRANT = Path(tempfile.mkdtemp(prefix="lv_shield_eval_"))
os.environ["QDRANT_PATH"] = str(_SCRATCH_QDRANT)

from backend.app.llm.model_selector import get_model              # noqa: E402
from backend.app.retrieval.embeddings import embed_batch          # noqa: E402
from backend.app.retrieval.vector_store import insert             # noqa: E402
from backend.app.shield.claim_extractor import extract_claims     # noqa: E402
from backend.app.shield.gap_detector import detect_gaps           # noqa: E402
from backend.app.shield.scoring import aggregate                  # noqa: E402

TARGET_HIT_COUNT = 4   # out of 5


def _index_corpus(corpus: list[dict], case_id: str) -> None:
    """Push the source corpus into the scratch Qdrant collection."""
    tier = get_model()
    texts = [c["content"] for c in corpus]
    vectors = embed_batch(texts, model_name=tier.embedding)
    payloads = [
        {
            "case_id": case_id,
            "doc_id": c["doc_id"],
            "chunk_id": c["id"],
            "content": c["content"],
            "page": c.get("page"),
            "filename": c["doc_id"] + ".pdf",
        }
        for c in corpus
    ]
    insert(vectors, payloads)


def _detect_report_flaws(report_text: str, doc_type: str, case_id: str, use_real_llm: bool) -> dict:
    """Run the Shield pipeline on one report and return detected flags."""
    claims = extract_claims(report_text)
    gap_report = detect_gaps(report_text, doc_type)
    missing_section_names = [s.name for s in gap_report.missing]

    # Verifier is the slow part — call it in two modes
    if use_real_llm:
        from backend.app.shield.verifier import verify_claims
        verification_results = verify_claims(claims, case_id)
    else:
        # Mock mode: use rerank score as a proxy for "evidence quality",
        # but DON'T ask Ollama — just mark everything Verified if evidence
        # is strong enough to pass the threshold. Contradictions will be
        # 0 in this mode, so contradiction-type planted flaws need
        # --real-llm to be fairly tested.
        verification_results = _mock_verify(claims, case_id)

    score = aggregate(
        verification_results,
        gap_report.to_dict(),
    )
    return {
        "claims_count": len(claims),
        "missing_sections": missing_section_names,
        "verification_results": verification_results,
        "contradiction_count": score.contradiction_count,
        "unverified_count": score.unverified_count,
        "trust_score": score.trust_score,
        "band": score.band,
    }


def _mock_verify(claims: list[str], case_id: str) -> list[dict]:
    """Faster alternative to verifier — uses retrieval + rerank score only."""
    from backend.app.retrieval.embeddings import embed
    from backend.app.retrieval.reranker import rerank
    from backend.app.retrieval.vector_store import query as qdrant_query
    tier = get_model()

    out = []
    for claim in claims:
        if not claim.strip():
            continue
        try:
            vec = embed(claim, model_name=tier.embedding)
            candidates = qdrant_query(vec, case_id, top_k=10)
            if not candidates:
                out.append({"claim": claim, "status": "Unverified", "evidence": []})
                continue
            top = rerank(claim, candidates, k=3)
            best = top[0].get("rerank_score", 0) if top else 0
            # Rerank threshold: 0.3 = evidence; else Unverified
            status = "Verified" if best >= 0.3 else "Unverified"
            out.append({
                "claim": claim,
                "status": status,
                "evidence": [{"content": (c.get("payload") or {}).get("content", "")} for c in top],
            })
        except Exception:
            out.append({"claim": claim, "status": "Unverified", "evidence": []})
    return out


def _hit(planted: dict, detected: dict) -> bool:
    """Did the Shield detect a flag matching this planted flaw?"""
    if planted["type"] == "missing_section":
        return planted["section"] in detected["missing_sections"]
    if planted["type"] == "contradiction":
        return detected["contradiction_count"] >= 1
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--real-llm",
        action="store_true",
        help="Call Ollama for every claim (slow, true to production).",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("LegalVault AI - Task 10.2 Report Shield evaluation")
    print("=" * 70)

    data = json.loads(
        (Path(__file__).parent / "sample_reports.json").read_text(encoding="utf-8")
    )
    corpus = data["source_corpus"]
    reports = data["reports"]

    tier = get_model()
    print(f"\nSource corpus: {len(corpus)} chunks")
    print(f"Reports:       {len(reports)}  (each has one or more planted flaws)")
    print(f"\nActive tier: {tier.tier} ({tier.embedding} + {tier.llm})")
    print(f"Verifier mode: {'REAL LLM' if args.real_llm else 'MOCK (rerank-only)'}")
    if not args.real_llm:
        print("  Contradiction detection is weaker in MOCK mode. Use --real-llm for")
        print("  the full pipeline. Mock mode validates missing_section detection.")

    case_id = f"shield-eval-{uuid.uuid4()}"
    print(f"\nIndexing source corpus into scratch Qdrant ...")
    _index_corpus(corpus, case_id)

    # Run Shield on each report
    print("\n" + "-" * 70)
    print(f"{'ID':<4} {'Hits/Flaws':<12} {'Trust':<6}  Report")
    print("-" * 70)

    all_hits = 0
    all_flaws = 0
    report_pass_count = 0
    report_detail = []

    for rep in reports:
        detected = _detect_report_flaws(
            rep["text"], rep["doc_type"], case_id, args.real_llm
        )
        hits_in_report = sum(1 for p in rep["planted_flaws"] if _hit(p, detected))
        flaws_in_report = len(rep["planted_flaws"])
        all_hits += hits_in_report
        all_flaws += flaws_in_report
        passed = hits_in_report >= 1  # at least ONE planted flaw caught per report
        if passed:
            report_pass_count += 1

        report_detail.append({
            "id": rep["id"],
            "name": rep["name"],
            "planted": rep["planted_flaws"],
            "detected": {
                "missing_sections": detected["missing_sections"],
                "contradiction_count": detected["contradiction_count"],
                "unverified_count": detected["unverified_count"],
            },
            "trust_score": detected["trust_score"],
            "band": detected["band"],
            "hits": hits_in_report,
            "flaws": flaws_in_report,
            "passed": passed,
        })

        mark = "PASS" if passed else "FAIL"
        print(f"{rep['id']:<4} {hits_in_report}/{flaws_in_report:<10} "
              f"{detected['trust_score']:<6}  {mark}  {rep['name']}")

    # Overall report
    print("\n" + "=" * 70)
    print("OVERALL")
    print("=" * 70)
    print(f"  Reports with >=1 planted flaw detected: {report_pass_count}/{len(reports)}")
    print(f"  Total planted flaws detected:           {all_hits}/{all_flaws}")
    print(f"  Target: {TARGET_HIT_COUNT}/{len(reports)} reports with any detection.")

    # Per-report detail
    print("\nPER-REPORT DETAIL:")
    for d in report_detail:
        print(f"\n  [{d['id']}] {d['name']}  trust={d['trust_score']}/100 ({d['band']})")
        for i, p in enumerate(d["planted"]):
            caught = _hit(p, {"missing_sections": d["detected"]["missing_sections"],
                              "contradiction_count": d["detected"]["contradiction_count"]})
            tag = "CAUGHT" if caught else "MISSED"
            desc = p.get("section") if p["type"] == "missing_section" else p.get("about")
            print(f"    [{tag}] planted: {p['type']} ({desc})")
        print(f"    detected: missing={d['detected']['missing_sections']}")
        print(f"              contradictions={d['detected']['contradiction_count']} "
              f"unverified={d['detected']['unverified_count']}")

    # Clean up scratch Qdrant
    shutil.rmtree(_SCRATCH_QDRANT, ignore_errors=True)

    # Mode-aware pass criteria:
    #   mock mode: contradiction-only reports cannot pass (no LLM) — only
    #              enforce missing-section detection reports.
    #   real-llm:  full target of 4/5 reports caught.
    if args.real_llm:
        if report_pass_count >= TARGET_HIT_COUNT:
            print(f"\nPASS (real-llm) - {report_pass_count}/{len(reports)} reports caught.\n")
            return 0
        print(f"\nFAIL (real-llm) - only {report_pass_count}/{len(reports)} caught, target {TARGET_HIT_COUNT}.\n")
        return 1

    # Mock mode: count only reports that have at least one missing_section flaw
    testable_reports = [r for r in reports
                        if any(p["type"] == "missing_section" for p in r["planted_flaws"])]
    testable_passed = sum(1 for d in report_detail
                          if d["passed"]
                          and any(p["type"] == "missing_section" for p in d["planted"]))
    if testable_reports and testable_passed == len(testable_reports):
        print(f"\nPASS (mock) - all {len(testable_reports)} missing-section-testable reports caught.")
        print(f"             contradiction-only reports (r3, r4) require --real-llm.\n")
        return 0
    print(f"\nFAIL (mock) - {testable_passed}/{len(testable_reports)} missing-section reports caught.\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
