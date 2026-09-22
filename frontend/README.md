# frontend/

SvelteKit UI compiled to static files, served by FastAPI at `/`.

See `../docs/TRD.md` §9 for architecture, `../docs/architecture.md` for topology.

## Planned layout

```
frontend/
├── src/
│   ├── routes/                 # SvelteKit pages
│   │   ├── login/
│   │   ├── cases/
│   │   └── cases/[id]/
│   │       ├── query/          # main chat UI (Task 7.2)
│   │       └── shield/         # Report Shield dashboard (Task 7.4)
│   └── lib/
│       ├── api/                # fetch wrapper with JWT injection
│       └── components/         # EvidenceCard, TrustScore, QueryBox
├── static/
├── build/                      # svelte-kit output — gitignored
├── package.json
└── svelte.config.js
```

## Setup (Task 7.1 will scaffold this)

```bash
cd frontend
npm install
npm run dev       # http://localhost:5173 in dev
npm run build     # produces build/ served by FastAPI in prod
```
