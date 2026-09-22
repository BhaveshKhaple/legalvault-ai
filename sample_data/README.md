# sample_data/

Non-sensitive sample documents used for local development and unit tests.

## Rules

- **NO client data.** Ever. This folder is public.
- **NO real names or identifiable information.** Use fabricated parties: "Acme Corp", "John Doe".
- **Small files only** (< 5MB each). Larger fixtures go in `tests/e2e/fixtures/` if needed.
- **License must permit redistribution.** Public government contract templates, RBI publications, and hand-written examples are all fine.

## What lives here

Each dev on the team should have access to at least:

- `short_contract.pdf` — a 2-page contract with clear clauses (for chunker tests)
- `long_contract.pdf` — 20+ page contract (for retrieval tests)
- `scanned_contract.pdf` — image-only PDF (for graceful-failure test)
- `contract_with_tables.pdf` — has a fee schedule table (for `pdfplumber` test)
- `meeting_short.mp3` — 5-minute mock meeting audio (for Whisper test)

These are ignored from git via `../.gitignore` — coordinate their distribution over Drive.
