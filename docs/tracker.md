# LegalVault AI — Task Tracker

The live sprint tracker is a Google Sheet:

**[Major project Tracker](https://docs.google.com/spreadsheets/d/1_JkYecHFrZVnllXicZ68Q_sEtrpWVICmhMLb3ZTXtLA/edit)**

## Tabs

| Tab | Purpose |
|-----|---------|
| **Backlog** | The 39-task list. Columns: Task ID, Module, Component, Task Description, Acceptance Criteria, Learn First, Where in Codebase, Depends On, Priority, Status, Owner, Progress, Started. |
| **Team** | Roster + roles. Bhavesh (lead), Shubham / Vijay / Tejas (devs). |
| **Docs** | Links to all planning docs + Drive folder. |
| **Change Log** | Manual log of major decisions. |
| **Dashboard** | Roll-up view (WIP). |

## Working rules (see also `../CONTRIBUTING.md` and `RULES.md`)

- **Sequential work only**: at most one task in `In Progress` per person.
- **Assigned folder only**: touch only the paths listed in your task's `Where in Codebase` column.
- **PR into `v1`, never `main`**: Bhavesh reviews on `v1` and merges to `main`.
- **Update tracker on start + finish**: set Status to `In Progress` when you start, `Done` when the PR is merged.

## Task status legend

- `Ready` — dependencies met, unassigned, safe to pick up.
- `Not Started` — deps not met yet.
- `In Progress` — someone is actively working on it (owner filled in).
- `Blocked` — needs help. Post in WhatsApp group + comment on the tracker row.
- `Done` — PR merged into `v1`.
- `Shipped` — merged into `main`.
