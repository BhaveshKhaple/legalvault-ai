# Phase 3 — Metadata filters + widen recall

## What this fixes

Pre-Phase-3 LegalVault had **zero** metadata on documents. If a case contained
multiple versions of the same RBI circular (v1 and v2), both would appear in
retrieval results, and the LLM could cite the superseded one as current. That's
"confidently wrong" territory — the exact failure mode a legal tool must avoid.

## What's new

### Document fields (SQLModel)

Four optional fields on `Document`, auto-migrated by the existing SQLite helper
on next startup:

| Field | Type | Example |
|---|---|---|
| `effective_date` | timezone-aware datetime | `2026-01-15T00:00:00+00:00` |
| `jurisdiction` | str, index | `"India"`, `"Maharashtra"`, `"Delhi HC"` |
| `version_tag` | str | `"v2"`, `"2026-01-15"` |
| `regulator` | str, index | `"RBI"`, `"SEBI"`, `"MCA"` |

### Upload — optional Form fields

`POST /v1/cases/{case_id}/documents` now accepts four optional multipart form
fields. If supplied, they're persisted on the `Document` row AND stamped on
every chunk's Qdrant payload at insert time.

```bash
curl -F "file=@contract.pdf" \
     -F "jurisdiction=India" \
     -F "effective_date=2026-01-15" \
     -F "regulator=RBI" \
     -F "version_tag=v2" \
     -H "Authorization: Bearer $TOKEN" \
     https://.../v1/cases/$CASE_ID/documents
```

### Metadata PATCH endpoint

For docs already uploaded (or when metadata changes mid-case):

```
PATCH /v1/cases/{case_id}/documents/{doc_id}/metadata
Body: { "effective_date": "...", "jurisdiction": "...", "version_tag": "...", "regulator": "..." }
```

Only the fields you supply get updated. The endpoint also rewrites the
payload on every Qdrant point for this doc (payload-only, no re-embed —
fast even for large docs).

### Query filters

`POST /v1/cases/{case_id}/query` body gains an optional `filters` block:

```json
{
  "question": "What are the latest KYC requirements?",
  "filters": {
    "regulator": "RBI",
    "date_after": "2026-01-01",
    "jurisdiction": "India"
  }
}
```

All five filters are optional: `date_after`, `date_before`, `jurisdiction`,
`version_tag`, `regulator`. The `date_*` fields take ISO date strings and
are converted to unix timestamps internally so Qdrant's `Range` filter can
compare them. Omitted fields = no restriction on that dimension.

### Widened first-stage recall

`FIRST_STAGE_K` constant in `rag_service.py` bumped from `20` → `100`. The
cross-encoder reranker narrows to top-5 after, so LLM latency is unchanged
(the reranker handles 100 pairs in ~200ms). This catches chunks that were
previously ranked 21-50 by the first-stage retrievers but would have made
it to top-5 after reranking.

## Qdrant filter construction

New helper `build_metadata_filter(case_id, **optional_filters)` in
`backend/app/retrieval/vector_store.py` composes the right Qdrant `Filter`
object. The mandatory `case_id` filter is **always** added first — security:
if that ever gets dropped, cross-tenant leakage. There's a dedicated test
for this invariant (`test_case_id_always_present_even_with_other_filters`).

## Numbers

| Gate | Baseline | Phase 3 | Target | Verdict |
|---|---|---|---|---|
| Retrieval R@5 (no filters) | 91.3% | 91.3% | ≥80% | ✅ no regression |
| Tests | 304 | **320** | — | ✅ +16 Phase 3 tests |
| Latency (first-stage) | ~20ms | ~25ms | <2s | ✅ (widen cost < 5ms) |

## Known limitations

- The eval corpus (`tests/eval/corpus/`) doesn't yet carry metadata, so the
  filter path isn't exercised in `eval_corpus.py`. Unit tests
  (`tests/retrieval/test_metadata_filter_builder.py` and
  `tests/api/test_metadata_filters.py`) cover the behaviour.
- Date comparisons use unix timestamp (second resolution). Fine for legal
  effective dates; not for intraday ordering.
- The PATCH endpoint rewrites payload keys only — if a user clears a field
  (sends `null`), the key gets removed from the Qdrant payload. Not a bug,
  but worth noting.

## Follow-ups for a later branch

- Extend `corpus_manifest.json` with `metadata` on each doc + a
  `filter_queries` section so `eval_corpus.py` can measure filter correctness.
- Add `GET /v1/cases/{case_id}/documents?regulator=RBI&date_after=...`
  (filter the document list, not just query results).
