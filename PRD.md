# LegalVault AI — Product Requirements Document

**Version:** 1.0 · **Owner:** Bhavesh Khaple (Team Pied Piper) · **Last updated:** 22 Aug 2026

---

## 1. Product overview

LegalVault AI is an on-premise, evidence-first document assistant for Indian legal and BFSI teams. It combines semantic search across mixed PDF and audio archives with a reverse-RAG audit verifier (Report Shield) that validates finished reports against source documents before submission.

**One-line pitch:** "Google Search for your legal documents, with cited proof and a report audit shield."

**Team:** Bhavesh Khaple (lead), Shubham, Vijay, Tejas Bagal.
**Delivery target:** DIPEX 2025 demo, then paid pilots.
**Hard constraint:** Runs offline. No cloud APIs. Client data never leaves the machine.

---

## 2. Problem statement

Legal and BFSI professionals lose 3–5 hours per week per person hunting for clauses, decisions, and evidence spread across PDFs and meeting recordings. Ctrl+F fails on synonyms: `termination` never returns `exit clause` or `Article 9.2`. Cloud LLM tools like ChatGPT and Claude send privileged documents to third-party servers, which violates both SEBI/RBI confidentiality rules and client privilege. Enterprise tools like Relativity cost ₹1.5L+ per month and are out of reach for 95% of Indian firms.

Compliance teams also submit audit reports where individual claims are supported by evidence in source files. But nobody has time to manually verify every citation. Missing or contradicted claims sometimes ship to regulators, and the consequence is censure, fines, or license review.

## 3. Target users

Four segments, ranked by expected willingness-to-pay:

1. **Small law firms (2–20 lawyers)** — Corporate and contract-heavy practices. Lose 3–5 hrs/week per lawyer. Budget exists. Report Shield reduces pre-filing risk.
2. **BFSI compliance teams** — NBFCs, cooperative banks, insurance firms. RBI/SEBI audit prep drains 2 weeks per cycle. Already used to buying compliance tooling.
3. **CA / tax advisory firms** — High document volume, subscription-native, daily regulatory text search need.
4. **Real estate legal (RERA)** — Builder agreements, buyer contracts, RERA compliance docs. High volume, high stakes, entirely manual today.

## 4. Personas and top user stories

### Persona A — Corporate lawyer at a small firm
- "As a lawyer, I want to ask 'what are all the penalty clauses across Client X contracts,' so I can prepare an argument in minutes instead of hours."
- "As a lawyer, I want every answer to link back to the exact PDF page, so I can show the source in court without doubt."
- "As a lawyer, I want to search meeting recordings alongside contracts, so I can prove what was agreed verbally as well as in writing."

### Persona B — BFSI compliance officer
- "As a compliance officer, I want to upload our final audit report and see which claims lack supporting evidence, so I can fix gaps before filing with the regulator."
- "As a compliance officer, I want the system to flag missing standard clauses (indemnity, force majeure, governing law), so we do not ship non-compliant reports."
- "As a compliance officer, I want to export an evidence pack ZIP for a submission, so the regulator gets structured supporting material with the report."

### Persona C — CA firm partner
- "As a CA, I want to search across every client agreement and notice in one query, so I can answer 'what was our stance on Section 194J for Client Y' without opening ten folders."

### Persona D — RERA practitioner
- "As a RERA lawyer, I want to compare a new builder agreement against templates and flag missing mandatory clauses, so I do not miss compliance items before signing."

## 5. Core features (v1)

Priority: **P0** = launch blocker, **P1** = important, **P2** = deferrable.

| ID | Feature | Priority | Description |
|----|---------|----------|-------------|
| F1 | Multimodal ingestion | P0 | Parse PDFs (clause-aware) and transcribe audio (Whisper with timestamps). Async via ARQ. |
| F2 | Semantic query with citations | P0 | Ask a natural-language question, return an answer plus source citations (page or timestamp). |
| F3 | Report Shield (reverse-RAG audit) | P0 flagship | Upload a report, get each claim classified as Verified / Unverified / Contradicted, plus a Trust Score and gap list. |
| F4 | Evidence export | P1 | Bundle answer + highlighted sources as a ZIP. Export Report Shield findings as a formatted PDF. |
| F5 | Multi-case isolation | P0 | Documents from Case A never appear in Case B queries. Metadata-filtered per case_id. |
| F6 | On-premise deployment | P0 | Docker Compose stack runs entirely offline on client hardware. |
| F7 | Tiered model config | P1 | One env var switches Qwen3 MoE / Phi-4 / Llama 3.2 based on device tier. |
| F8 | Auth + RBAC | P1 | JWT-based auth. Roles: Admin, Analyst, Viewer. Case ownership enforced. |
| F9 | Immutable audit log | P1 | Every query, upload, export logged with user, timestamp, IP. Append-only. |

## 6. Success metrics

- **Retrieval quality:** Recall@5 above 75% on a 20-query annotated legal test set.
- **Latency:** End-to-end query response under 3 seconds on the CPU tier.
- **Report Shield precision:** At least 80% of flags catch real gaps or contradictions when tested against known-flaw sample reports.
- **Offline verification:** Zero external network calls during a full demo run (measured with tcpdump).
- **Adoption:** 3 paying pilot firms within 6 months of DIPEX submission.
- **Time saved:** Pilot users report 70% reduction in evidence-retrieval time compared to their prior workflow (self-reported, tracked monthly).

## 7. Non-goals

The following are explicitly out of scope for v1. Say no when asked:

- **No document drafting or generation.** This is a retrieval and verification tool, not a contract generator.
- **No Hindi document support at launch.** English and code-mixed English only. Hindi is a M10-12 candidate.
- **No mobile app.** Desktop web only.
- **No integration with existing case management systems** (Odoo, Zoho Legal, Manupatra). Standalone in v1.
- **No cloud-hosted SaaS option.** On-prem only in v1. SaaS is a M12+ decision.
- **No live legal research** (case law lookup, statute search). Only searches documents the client uploads.
- **No automatic contract classification into legal ontologies.** Free-text search only, no schema induction.

## 8. Business model

| Plan | ₹/month | Users | Docs | Includes |
|---|---|---|---|---|
| Starter | 3,000 | 1 | 100 | F1, F2, F4 |
| Professional | 8,000 | 5 | 1,000 | + F3 Report Shield |
| Enterprise | 15,000+ | Unlimited | Unlimited | + on-prem installer, priority support, SLA |

ROI story: Professional at ₹8,000/month for 5 lawyers works out to ₹1,600 per lawyer per month. Even 3 hours of retrieval time saved per lawyer per month puts the client in positive ROI at any Indian billable rate.

## 9. Roadmap

- **Now:** DIPEX 2025 demo, validate Report Shield with domain experts.
- **Month 1–2:** Report Shield v1 with clause library for Indian contracts. First feature no competitor has.
- **Month 2–4:** Free 4-week pilots with 2–3 firms. Measure real time saved. Refine chunking on Indian legal docs.
- **Month 4–6:** Convert pilots to paid. Professional plan becomes primary pitch.
- **Month 7–9:** BFSI-focused release with SEBI/RBI regulation packs.
- **Month 10–12:** Enterprise on-prem packaging with SLA. Hindi document support. Seed funding conversation with 6 months of usage data.

## 10. Assumptions

- Client hardware supports 8GB+ RAM at minimum. Verified before install.
- Client accepts on-disk storage of vector indexes and metadata (unencrypted at rest in v1; encryption is a P1 for M4).
- Client's users can operate a modern browser and drag-drop upload.
- English is the working language of the interface. Documents may be code-mixed.

## 11. Constraints

- **Legal privilege:** Documents are subject to attorney-client privilege. No third-party network access allowed at query time.
- **Regulatory:** SEBI Data Protection Circular, RBI IT Framework for NBFCs. Data residency is on-prem.
- **Team size:** 4 people, part-time (college major project). Ten-week build window before DIPEX. No paid infrastructure.
- **Hardware baseline:** Must run on 8GB RAM Windows/Linux laptops without a GPU (Tier 2).

## 12. Open questions

- Should search results include a "confidence" score in the UI, or hide it and only show it in exports? (Team vote by Sprint 2.)
- Do we ship a default clause library for Indian contracts, or ask each firm to seed their own? (Ship a starter library plus editable list.)
- Should the CI-CD pipeline auto-update client environments? (No, v1. Client controls their own upgrades.)
- Where does Whisper transcription cache live? (In client's Docker volume, not in Qdrant.)

## 13. Related documents

- `TRD.md` — Technical Requirements Document (architecture, schema, APIs).
- `RULES.md` — Team development rules and AI coding guardrails.
- `Major project Tracker.xlsx` — Sprint-level task tracker (Google Sheets).
- Pitch deck (DIPEX 2025) — Product narrative and visual system.
