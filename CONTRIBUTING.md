# Contributing to LegalVault AI

Read this before you open your first PR. This flow is enforced by branch protection — direct pushes to `main` will be rejected.

---

## Branch model

```
main ← v1 ← feature/1.2-clause-chunker
                 (your fork)
```

- **`main`** — released, reviewed, tested. Protected. Nobody pushes directly.
- **`v1`** — integration branch. Bhavesh reviews every PR here, tests, then merges to `main` when a milestone is ready.
- **Your feature branches** — live in *your own fork*, not the main repo.

## Setup (do this once)

1. **Fork** `BhaveshKhaple/legalvault-ai` on GitHub (top-right "Fork" button).
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/<your-username>/legalvault-ai.git
   cd legalvault-ai
   ```
3. **Add upstream remote** so you can pull Bhavesh's updates:
   ```bash
   git remote add upstream https://github.com/BhaveshKhaple/legalvault-ai.git
   ```
4. **Create the backend venv** (see `README.md` for the exact command).
5. **Verify** you can see both remotes:
   ```bash
   git remote -v
   # origin    https://github.com/<you>/legalvault-ai.git  (fetch/push)
   # upstream  https://github.com/BhaveshKhaple/legalvault-ai.git (fetch/push)
   ```

## Working on a task

1. **Pick a task from the [tracker](https://docs.google.com/spreadsheets/d/1_JkYecHFrZVnllXicZ68Q_sEtrpWVICmhMLb3ZTXtLA/edit)** with Status = `Ready` and Owner empty.
2. **Claim it**: put your name in the Owner column, change Status to `In Progress`. One task at a time per person.
3. **Sync `v1` from upstream**:
   ```bash
   git fetch upstream
   git checkout v1
   git merge upstream/v1
   ```
4. **Branch from `v1`** with the task ID in the name:
   ```bash
   git checkout -b feature/1.2-clause-chunker
   ```
   Naming pattern: `feature/<task-id>-<short-slug>`. Examples: `feature/2.1-bge-wrapper`, `feature/6.3-query-endpoint`.
5. **Touch only the paths listed in your task's "Where in Codebase" column**. Do not "clean up" adjacent files. Do not refactor anything not in your scope.
6. **Write tests** for what you built. If your task lives in `backend/app/foo/bar.py`, tests go in `tests/foo/test_bar.py`.
7. **Commit small, commit often**. Message format:
   ```
   <task-id>: <short imperative>

   Optional body explaining the "why", not the "what".
   ```
   Example: `1.2: add clause-aware chunker for legal PDFs`
8. **Push to your fork**:
   ```bash
   git push origin feature/1.2-clause-chunker
   ```
9. **Open a PR into `v1`** (NOT `main`). The PR title should be your task ID + short description. The description should:
   - Link back to the tracker row.
   - List acceptance criteria and tick each one off with `[x]`.
   - Include screenshots for UI tasks.
   - Note any deviations from the task spec (with justification).
10. **Wait for Bhavesh's review**. Address comments. Do not merge yourself.

## Code rules (short version — full list in `docs/RULES.md`)

- **Python 3.11+** everywhere. Type hints on public functions.
- **Format with `ruff format`. Lint with `ruff check`.** Both run in CI.
- **No secrets in code.** Ever. If you leak one, tell Bhavesh immediately — do not try to fix it silently.
- **No new dependencies without discussion.** Post in the WhatsApp group first.
- **Tests must pass locally** before you push (`pytest`).
- **No AI code you don't understand.** If you used Claude/ChatGPT to generate it, you must be able to explain every line in review.

## PR review checklist (what Bhavesh looks for)

- [ ] Only files in the task's assigned folder were touched.
- [ ] Acceptance criteria from the tracker are all met.
- [ ] Tests exist and pass.
- [ ] No secrets, keys, or client data committed.
- [ ] Commit messages follow the format.
- [ ] Docs updated if the change touches public API or config.

## When you're stuck

- **Under 30 min stuck**: try one more thing, then post in WhatsApp with what you've tried.
- **Over 30 min stuck**: comment on your PR (or open a draft PR) tagging Bhavesh with the specific error.
- **Never** silently abandon a task. Mark it `Blocked` on the tracker so someone else can pick it up.
