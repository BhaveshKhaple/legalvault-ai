# LegalVault AI — Team Rules & AI Coding Guardrails

**Version:** 1.0 · **Owner:** Bhavesh Khaple · **Last updated:** 22 Aug 2026

Every teammate reads this before writing a single line of code. Every AI-generated line is checked against this document before it lands in a PR.

The rules here are not style suggestions. Break them and code gets sent back, no exceptions.

---

## 1. Before you code

### 1.1 Understand the task
Open `LegalVault_Tracker.xlsx` and read your task's row. All of it:
- Task description
- Acceptance criteria (this is the definition of Done)
- Depends On
- Priority

If any of those four are unclear, ask on the group chat. Do not guess. Do not start coding.

### 1.2 Read the source of truth
Every task assumes you have read:
- `PRD.md` — what the product is and who uses it
- `TRD.md` — the specific section for your module

If your task touches ingestion, you read TRD §6. If it touches retrieval, TRD §7. And so on.

### 1.3 Lock the task
Change your task's Status to `In Progress` in the tracker. Set Owner to your name.
Only one task may be `In Progress` at any time across the whole team. Check the Dashboard.

If the Lock Check on the Dashboard is red, someone else's task is still open. Wait, review their PR, or do docs work.

---

## 2. Working with AI (Codex, Antigravity, Claude, Copilot, Cursor, whatever you use)

AI writes code fast. AI also hallucinates, over-engineers, and injects patterns that will fail review. These rules exist because AI will fail all of them if you let it.

### 2.1 Prompt hygiene

Paste the following into the AI's context every time you start a new coding session:
- The task's description and acceptance criteria (from the tracker)
- The relevant section of `TRD.md`
- The current file you are editing, in full

Do NOT rely on the AI's general knowledge of "how a RAG pipeline works." Our RAG uses BGE-M3, embedded Qdrant, ARQ, and citation-enforced prompts. Generic RAG advice will contradict our stack.

### 2.2 What NOT to accept from AI

Reject any AI output that includes:

- **Hallucinated packages.** If the AI imports `from langchain_qdrant_advanced_v3 import X` and pip cannot find it, delete it.
- **Cloud APIs.** No `openai.ChatCompletion.create`, no `anthropic.messages.create`, no `google.generativeai.GenerativeModel`. Every LLM call goes through Ollama. Reject any code that calls a cloud endpoint.
- **Speculative features.** If you asked for a PDF chunker and the AI also added an OCR fallback, a caching layer, and a retry loop, delete everything except the chunker. See §3.2.
- **New abstractions.** If the AI wraps a 3-line function in a class hierarchy with a `Factory`, an `AbstractBaseChunker`, and a `ChunkerRegistry`, throw it out. Match our existing style.
- **Configuration bloat.** Every new config knob is a maintenance cost. Reject `chunker_enable_experimental_mode: bool = False` type parameters unless the task explicitly asks for them.
- **Error handling for impossible states.** If the AI wraps a call in `try/except Exception` and prints the exception, delete it. Let real errors propagate. Only handle exceptions where recovery is defined.
- **Silent failures.** No `except: pass`. No `return None if X else Y` at the top of functions.
- **Guessed API signatures.** If the AI writes `qdrant.upsert(collection, points, wait=True, ordering="strong")`, verify against the actual Qdrant Python client docs before using.
- **Confident wrong answers.** If the AI says "this is standard practice for FastAPI" but you have never seen it in the FastAPI docs, verify before accepting.

### 2.3 What TO accept from AI

- Boilerplate that matches our existing patterns (Pydantic models, SQLModel classes, FastAPI endpoint scaffolds).
- Unit tests for your own function, written against the acceptance criteria.
- Documentation strings (short, single line) and inline comments where behavior is genuinely non-obvious.
- Refactors of code you wrote yourself, once the code works.

### 2.4 The three-question test

Before you commit AI-generated code, answer these out loud:

1. **Did I read every line?** If any block looks like magic and you cannot explain what it does, either understand it or delete it.
2. **Does it match the acceptance criteria?** Not "does it kind of work" — does it satisfy every bullet in the tracker row?
3. **Would I be embarrassed if Bhavesh asked me why this exists?** If yes, remove it.

If you cannot pass all three, do not push.

---

## 3. Coding principles (Karpathy-inspired, enforced)

### 3.1 Think before coding
- State assumptions in the PR description. If you assumed BGE-M3 output is 1024 dims, say so.
- If there are multiple ways to solve the task, note the alternatives you considered.
- Ask on the group chat if any assumption feels shaky before coding, not after.

### 3.2 Simplicity first
- Minimum code that solves the problem. Nothing speculative.
- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that was not requested.
- If your diff has more than 300 net-new lines for a P1 task, stop. Something is wrong. Ask.

### 3.3 Surgical changes
- Touch only what you must. Do not "improve" adjacent code, comments, or formatting.
- Do not refactor things that are not broken.
- Match existing style, even if you would do it differently.
- If you notice unrelated dead code, mention it in the PR description. Do not delete it yourself.

### 3.4 Goal-driven execution
Turn every task into a verifiable goal. Examples:
- "Add clause chunker" → "Write a test for the 3.1-clause splitting example from TRD §6, then make it pass."
- "Fix retrieval bug" → "Write a failing test that reproduces the bug, then make it green."

The acceptance criteria in the tracker IS your test spec. Write the test first.

---

## 4. Security do-nots (non-negotiable)

Bhavesh's past experience: a `.env` with 3 Google API keys stayed in `Multimodal-SIH` on GitHub for 13 days after discovery. The keys were live and abused. Do not repeat this.

### 4.1 Never commit
- `.env` files. Any of them. Ever.
- API keys, tokens, passwords, database URLs with credentials.
- Client documents (PDFs, audio) from real cases.
- Private keys (RSA, SSH), certificates, PEM files.
- Postgres database dumps.
- Qdrant snapshots.

### 4.2 Always
- Add new secrets to `.env.example` (with a fake value) at the same time you add them to your `.env`.
- Read secrets via `os.environ["KEY"]`. Fail loudly on startup if missing.
- Use `git status` before every commit. If you see `.env`, unstage it.
- Run `git diff --staged` before pushing. Scan for suspicious strings (`AIza`, `sk-`, `ghp_`, `xoxb-`).
- Enable the pre-commit hook (see §7). It blocks obvious mistakes.

### 4.3 If a secret leaks (incident procedure)
1. Rotate the secret first. In the provider's dashboard. Before anything else.
2. Remove the file from the current branch, commit the removal.
3. Force-purge from history: `git filter-repo --path .env --invert-paths` (or BFG Repo-Cleaner).
4. Force-push to all branches.
5. Notify Bhavesh on the group chat with what leaked and when.
6. Add the leaked filename to `.gitignore` if not already present.

### 4.4 Input validation and injection
- Every API endpoint validates inputs via Pydantic v2 models. No raw `request.json()`.
- Every SQL query uses parameter binding via SQLModel. No string interpolation.
- Every LLM prompt uses the citation-enforcing template from `backend/services/prompts.py`. Do not construct prompts by string concatenation elsewhere.
- Documents are data, not commands. If the AI writes a prompt that embeds `{document_text}` inside instructions, wrap the document in an explicit sandbox and add the "ignore instructions in documents" clause. See TRD §12.4.

### 4.5 Access control
- Every endpoint that touches a resource verifies ownership: `case.org_id == user.org_id`.
- Cross-org access always returns 403.
- Audit log rows are append-only. No endpoint deletes them.

---

## 5. Code style

### 5.1 Python
- Format with `ruff format`. Not `black`. Not `autopep8`. One formatter.
- Lint with `ruff check --select=E,F,W,I,B,UP`. Fix warnings before pushing.
- Type-hint every function signature. Use `from __future__ import annotations` at the top of every file.
- Names: `snake_case` for functions and variables, `CamelCase` for classes, `SCREAMING_SNAKE_CASE` for constants.
- Files: `snake_case.py`. Modules: same.
- Max line length: 100. Do not fight the formatter.

### 5.2 SvelteKit / TypeScript
- Format with `prettier`. Prettier config lives at repo root.
- Lint with `eslint --config .eslintrc.js`.
- TypeScript strict mode on. No `any` unless there is a comment explaining why.

### 5.3 Comments
- Default: no comments. Well-named identifiers do the work.
- Only add a comment when the WHY is non-obvious: a hidden constraint, a subtle invariant, a workaround for a specific library bug.
- Never explain WHAT the code does. Never reference the current task, PR, or ticket. Comments outlive the branch.

### 5.4 Docstrings
- Single-line only for internal functions.
- Public API endpoints get a FastAPI docstring visible in `/docs`.
- No paragraphs. No Sphinx directives. No `:param:` blocks in v1.

---

## 6. Git workflow

### 6.1 Branches
- `main` — protected. Only Bhavesh merges here.
- `v1` — integration branch. All PRs target this first.
- Feature branches — named `feature/<module>-<task-id>`. Example: `feature/ingestion-1.2`.
- No long-lived personal branches other than your fork's `main`.

### 6.2 Commits
- One task = one PR = one focused set of commits.
- Commit message format:
  ```
  <type>(<module>): <short summary>

  <body: what and why, not how. Optional.>

  Task: <task-id>
  ```
- Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `perf`, `sec`.
- Example: `feat(ingestion): clause-aware PDF chunker (Task 1.2)`
- Never squash commits inside your own PR. Bhavesh squashes at merge if needed.

### 6.3 PR discipline
Before opening a PR, verify:
- [ ] Every acceptance criterion from the tracker is satisfied.
- [ ] `ruff check` passes.
- [ ] `pytest tests/unit tests/integration` passes locally.
- [ ] `docs/updates/<task-id>.md` exists in the diff with the structure from §6.5.
- [ ] No secrets or client documents in the diff.
- [ ] Diff is scoped to this task. No unrelated cleanups.

PR title: `[Task <id>] <one-line summary>`
PR body template lives at `.github/PULL_REQUEST_TEMPLATE.md`.

### 6.4 Review
- Bhavesh is the sole reviewer for v1.
- Response SLA: 24 hours from PR open.
- Bhavesh either merges to `v1`, requests changes with specific comments, or blocks with a reason.
- After changes are pushed, developer re-requests review by tagging Bhavesh.

### 6.5 update.md convention
Every completed task ships with `docs/updates/<task-id>.md`. Four sections, keep it tight:

```markdown
# Task <id>: <one-line summary>

## What I built
<3-8 bullets, plain English>

## Files touched
<file paths, one per line>

## How to test
<exact commands and expected outcome>

## Known gaps
<anything intentionally deferred or unresolved>
```

This is human-written or human-verified. Do not paste raw AI output.

---

## 7. Local tooling

Install these before your first PR:

```bash
# Python
pip install ruff pytest pytest-asyncio pre-commit

# Pre-commit hooks
pre-commit install

# Node (frontend only)
npm install -g pnpm
```

### 7.1 Pre-commit hooks (defined in `.pre-commit-config.yaml`)
- `ruff format` and `ruff check --fix`
- `detect-secrets` (blocks obvious keys)
- Reject any staged `.env` file with a clear error message

### 7.2 Local test loop
```bash
# unit only
pytest tests/unit -q

# integration (requires docker-compose up postgres redis)
docker compose up -d postgres redis
pytest tests/integration -q

# e2e (slow, run before PR)
docker compose up -d
pytest tests/e2e -q
```

---

## 8. When AI is the wrong tool

Some tasks in this project should NOT be delegated primarily to AI. Write these yourself, then let the AI review:

- **Auth and RBAC middleware.** JWT parsing, role checks, org boundary enforcement. Security-critical.
- **Prompt templates for LegalVault-specific tasks.** The citation-enforcing prompt lives at `backend/services/prompts.py`. Modify by hand with review.
- **Any code that touches `audit_log`.** Append-only integrity depends on nobody adding a DELETE path.
- **Migration files (Alembic).** Write them by hand. AI regularly produces migrations that lose data.
- **Anything under `backend/services/report_shield/`.** This is our flagship. Every line reviewed manually.

For all of these, you may prompt the AI to critique your code once you have written it. Do not prompt it to write from scratch.

---

## 9. When in doubt

- Ask on the group chat before coding.
- Ask on the PR before merging.
- If Bhavesh is unavailable, another teammate reviews first, and Bhavesh does a second-pass audit.
- Rule of thumb: if you are about to skip a rule "just this once," you are about to make it much harder for the next teammate.

## 10. Related documents

- `PRD.md` — product requirements
- `TRD.md` — technical requirements
- `README.md` — quick start and orientation
- `LegalVault_Tracker.xlsx` — sprint tracker (source of truth for task state)
