"""
Generate LegalVault AI — Contributor Guide PDF.

Run from repo root:
    backend/venv/Scripts/python scripts/build_contributor_guide.py

Output: docs/CONTRIBUTOR_GUIDE.pdf
"""

from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

OUT = Path("docs/CONTRIBUTOR_GUIDE.pdf")
OUT.parent.mkdir(parents=True, exist_ok=True)

NAVY = colors.HexColor("#1b2a4e")
GOLD = colors.HexColor("#c9a84c")
IVORY = colors.HexColor("#faf8f2")
GREY = colors.HexColor("#6b7280")
LIGHT = colors.HexColor("#f3f4f6")
RED = colors.HexColor("#dc2626")
GREEN = colors.HexColor("#16a34a")

styles = getSampleStyleSheet()

TITLE_STYLE = ParagraphStyle(
    "DocTitle",
    parent=styles["Title"],
    fontName="Helvetica-Bold",
    fontSize=26,
    leading=32,
    textColor=colors.white,
    alignment=TA_CENTER,
    spaceAfter=6,
)
SUBTITLE_STYLE = ParagraphStyle(
    "DocSubtitle",
    parent=styles["Normal"],
    fontName="Helvetica",
    fontSize=12,
    leading=16,
    textColor=GOLD,
    alignment=TA_CENTER,
    spaceAfter=4,
)
H1 = ParagraphStyle(
    "H1",
    parent=styles["Heading1"],
    fontName="Helvetica-Bold",
    fontSize=15,
    leading=20,
    textColor=NAVY,
    spaceBefore=18,
    spaceAfter=6,
    borderPad=4,
)
H2 = ParagraphStyle(
    "H2",
    parent=styles["Heading2"],
    fontName="Helvetica-Bold",
    fontSize=12,
    leading=16,
    textColor=NAVY,
    spaceBefore=12,
    spaceAfter=4,
)
BODY = ParagraphStyle(
    "Body",
    parent=styles["BodyText"],
    fontName="Helvetica",
    fontSize=10,
    leading=15,
    textColor=colors.black,
    alignment=TA_JUSTIFY,
    spaceAfter=6,
)
BULLET = ParagraphStyle(
    "Bullet",
    parent=BODY,
    fontName="Helvetica",
    fontSize=10,
    leading=14,
    leftIndent=14,
    spaceAfter=4,
    bulletIndent=4,
)
CODE = ParagraphStyle(
    "Code",
    parent=styles["Code"],
    fontName="Courier",
    fontSize=9,
    leading=13,
    textColor=NAVY,
    backColor=LIGHT,
    leftIndent=10,
    rightIndent=10,
    spaceBefore=4,
    spaceAfter=4,
    borderPad=6,
)
WARN = ParagraphStyle(
    "Warn",
    parent=BODY,
    fontName="Helvetica-BoldOblique",
    fontSize=10,
    leading=14,
    textColor=RED,
    spaceAfter=6,
)
NOTE = ParagraphStyle(
    "Note",
    parent=BODY,
    fontName="Helvetica-Oblique",
    fontSize=9,
    leading=13,
    textColor=GREY,
    spaceAfter=4,
)


def rule():
    return HRFlowable(
        width="100%", thickness=1, color=GOLD, spaceAfter=6, spaceBefore=2
    )


def sp(n=6):
    return Spacer(1, n)


def h1(text):
    return Paragraph(text, H1)


def h2(text):
    return Paragraph(text, H2)


def body(text):
    return Paragraph(text, BODY)


def bullet(text):
    return Paragraph(f"• {text}", BULLET)


def code(text):
    return Paragraph(
        text.replace(" ", "&nbsp;").replace("<", "&lt;").replace(">", "&gt;"), CODE
    )


def warn(text):
    return Paragraph(f"⚠ {text}", WARN)


def note(text):
    return Paragraph(text, NOTE)


def cover_table():
    cover = Table(
        [
            [
                Paragraph("LegalVault AI", TITLE_STYLE),
                Paragraph(
                    "Contributor Guide &amp; Pull Request Manual", SUBTITLE_STYLE
                ),
                Paragraph("Team Pied Piper · October 2026", NOTE),
            ]
        ],
        colWidths=[170 * mm],
    )
    cover.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), NAVY),
                ("TOPPADDING", (0, 0), (-1, -1), 18),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 18),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("ROWBACKGROUNDS", (0, 0), (-1, -1), [NAVY]),
            ]
        )
    )
    return cover


def branch_table():
    data = [
        ["Branch", "Purpose", "Who merges"],
        ["main", "Production-ready milestone releases", "Bhavesh only"],
        ["v1", "Integration — all PRs target this", "Bhavesh after review"],
        [
            "feature/<task-id>-<slug>",
            "One task, one branch, from your fork",
            "You (PR → v1)",
        ],
    ]
    tbl = Table(data, colWidths=[50 * mm, 75 * mm, 45 * mm])
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [IVORY, colors.white]),
                ("GRID", (0, 0), (-1, -1), 0.5, GREY),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return tbl


def role_table():
    data = [
        ["Name", "Role", "GitHub handle", "Focus area"],
        ["Bhavesh Khaple", "Lead / Reviewer", "BhaveshKhaple", "All modules, merges"],
        ["Shubham", "Backend Dev", "—", "API, ingestion"],
        ["Vijay", "Frontend Dev", "AI-Vijay", "SvelteKit UI (7.x tasks)"],
        ["Tejas", "QA / Testing", "—", "Eval, test suites"],
    ]
    tbl = Table(data, colWidths=[40 * mm, 30 * mm, 40 * mm, 60 * mm])
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [IVORY, colors.white]),
                ("GRID", (0, 0), (-1, -1), 0.5, GREY),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return tbl


def roadmap_table():
    data = [
        ["Phase", "Branch", "What", "Leads to"],
        [
            "1 — Docling parser",
            "feature/docling-parser",
            "Layout-aware PDF/DOCX parsing",
            "Phase 2",
        ],
        [
            "2 — Parent-child chunking",
            "feature/parent-child-chunking",
            "Clause child + section parent",
            "Phase 3, 4",
        ],
        [
            "3 — Metadata filters",
            "feature/metadata-filters",
            "Date, jurisdiction, version tags",
            "Phase 4",
        ],
        [
            "4 — Contextual retrieval",
            "feature/contextual-retrieval",
            "Doc-level BM25 context prefix",
            "Final eval",
        ],
        [
            "5 — VLM multimodal",
            "feature/vlm-multimodal",
            "WhisperX, diagram captioning",
            "Other workstation",
        ],
    ]
    tbl = Table(data, colWidths=[38 * mm, 50 * mm, 52 * mm, 30 * mm])
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [IVORY, colors.white]),
                ("GRID", (0, 0), (-1, -1), 0.5, GREY),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return tbl


def checklist_table():
    checks = [
        ("Only files in the task's assigned folder were touched", "Code review"),
        ("All acceptance criteria from the tracker are met with [x]", "Code review"),
        ("Tests exist and pass locally (pytest)", "CI + code review"),
        ("No secrets, API keys, or client data committed", "CI + security scan"),
        ("Commit messages follow <task-id>: <imperative> format", "Git hygiene"),
        (
            "docs/updates/<task-id>.md created describing what you built",
            "Documentation",
        ),
        (
            "No new dependencies added without WhatsApp group discussion",
            "Dependency audit",
        ),
        (
            "AI-generated code fully understood and explained-able in review",
            "Code quality",
        ),
    ]
    data = [["Checklist item", "Verified by"]] + list(checks)
    tbl = Table(data, colWidths=[120 * mm, 50 * mm])
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [IVORY, colors.white]),
                ("GRID", (0, 0), (-1, -1), 0.5, GREY),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return tbl


def build():
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="LegalVault AI — Contributor Guide",
        author="Team Pied Piper",
    )

    story = []

    # ── Cover ────────────────────────────────────────────────────────────────
    story.append(cover_table())
    story.append(sp(20))
    story.append(
        body(
            "This document is the single source of truth for how to contribute code to "
            "LegalVault AI. Read it once end-to-end before you open your first pull request. "
            "Every rule here exists because we learned the hard way what happens when it is "
            "ignored — merge conflicts, broken tests, and wasted review cycles."
        )
    )
    story.append(rule())

    # ── Section 1 — Team ─────────────────────────────────────────────────────
    story.append(h1("1. Team Roles"))
    story.append(role_table())
    story.append(sp(8))
    story.append(
        note(
            "Vijay: your open PR (AI-Vijay:adding-my-code) targets main — the wrong base. "
            "Retarget it to v1 before review can begin. Steps are in Section 5."
        )
    )
    story.append(rule())

    # ── Section 2 — Repository layout ────────────────────────────────────────
    story.append(h1("2. Repository Layout"))
    story.append(body("GitHub: github.com/BhaveshKhaple/legalvault-ai"))
    story.append(sp(4))
    story.append(branch_table())
    story.append(sp(8))
    story.append(
        body(
            "The sequential rule: only ONE task may have Status = In Progress across the whole "
            "team at any time. If someone else is In Progress, wait, do a code review, or work "
            "on documentation. This is intentional — it prevents merge conflicts and "
            "half-finished dependencies reaching v1."
        )
    )
    story.append(rule())

    # ── Section 3 — One-time setup ───────────────────────────────────────────
    story.append(h1("3. One-Time Local Setup"))
    story.append(h2("3.1 Fork and clone"))
    story.append(
        bullet("Go to github.com/BhaveshKhaple/legalvault-ai → click Fork (top right).")
    )
    story.append(bullet("Clone your fork:"))
    story.append(code("git clone https://github.com/<YOUR-USERNAME>/legalvault-ai.git"))
    story.append(bullet("Add the upstream remote (Bhavesh's repo):"))
    story.append(
        code(
            "git remote add upstream https://github.com/BhaveshKhaple/legalvault-ai.git"
        )
    )
    story.append(bullet("Verify both remotes exist:"))
    story.append(code("git remote -v"))

    story.append(h2("3.2 Python environment"))
    story.append(
        bullet("Python 3.11+ required. Install it from python.org if missing.")
    )
    story.append(bullet("Create the venv from inside the backend/ folder:"))
    story.append(
        code(
            "cd backend\n"
            "python -m venv venv\n"
            "venv\\Scripts\\activate          # Windows\n"
            "source venv/bin/activate        # macOS / Linux\n"
            "pip install -r requirements.txt"
        )
    )
    story.append(bullet("Copy the environment template:"))
    story.append(
        code(
            "cp backend/.env.example backend/.env\n"
            "# Edit backend/.env — set MODEL_TIER, JWT_SECRET_KEY, etc."
        )
    )

    story.append(h2("3.3 Ollama (local LLM)"))
    story.append(bullet("Install Ollama from ollama.com/download."))
    story.append(bullet("Pull the model that matches your hardware tier:"))
    story.append(
        code(
            "ollama pull phi3:mini       # Tier 2 — office laptop (default)\n"
            "ollama pull qwen3:30b       # Tier 1 — GPU workstation"
        )
    )
    story.append(bullet("Start the server: ollama serve (leave this terminal open)."))

    story.append(h2("3.4 Run the backend"))
    story.append(
        code(
            "cd D:\\projects\\Legalvalult AI\\legalvault-ai\n"
            "backend\\venv\\Scripts\\python -m uvicorn backend.app.main:app --reload --port 8000"
        )
    )
    story.append(
        note(
            "Folder name 'Legalvalult' has a typo. That is intentional — do not rename it."
        )
    )
    story.append(rule())

    # ── Section 4 — Daily workflow ────────────────────────────────────────────
    story.append(h1("4. Daily Contribution Workflow"))
    story.append(h2("Step 1 — Pick a task"))
    story.append(
        body(
            "Open the tracker spreadsheet. Find a task where Status = Ready and Owner is empty. "
            "Read the task description, acceptance criteria, and the 'Where in Codebase' column. "
            "If anything is unclear, ask in WhatsApp before you start."
        )
    )
    story.append(
        note(
            "Tracker: docs.google.com/spreadsheets/d/1_JkYecHFrZVnllXicZ68Q_sEtrpWVICmhMLb3ZTXtLA"
        )
    )

    story.append(h2("Step 2 — Claim the task"))
    story.append(bullet("Set Status = In Progress."))
    story.append(bullet("Set Owner = your name."))
    story.append(bullet("Add today's date to the Started column."))
    story.append(
        warn(
            "Only one task In Progress at a time across the whole team. Check before you claim."
        )
    )

    story.append(h2("Step 3 — Sync v1 from upstream"))
    story.append(
        code("git fetch upstream\n" "git checkout v1\n" "git merge upstream/v1")
    )

    story.append(h2("Step 4 — Create your feature branch"))
    story.append(code("git checkout -b feature/<task-id>-<short-slug>"))
    story.append(body("Examples:"))
    story.append(
        code(
            "feature/7.1-upload-ui\n"
            "feature/9.1-backend-dockerfile\n"
            "feature/docling-parser"
        )
    )
    story.append(
        body(
            "Branch names are lowercase, hyphens only, no spaces. "
            "The task ID (e.g. 7.1) must appear first so the tracker can match it."
        )
    )

    story.append(h2("Step 5 — Write code"))
    story.append(
        bullet("Touch ONLY the files listed in your task's 'Where in Codebase' column.")
    )
    story.append(
        bullet("Do not clean up adjacent files or refactor things not in your scope.")
    )
    story.append(
        bullet("Match existing code style — indentation, naming, comment style.")
    )
    story.append(
        bullet("Type-hint all public functions: def foo(x: str) -> list[dict]:")
    )
    story.append(bullet("Write tests in tests/<module>/test_<file>.py before pushing."))

    story.append(h2("Step 6 — Commit your work"))
    story.append(body("Commit message format (mandatory):"))
    story.append(
        code(
            "<task-id>: <short imperative sentence>\n\n"
            "Optional body explaining WHY, not what."
        )
    )
    story.append(body("Examples:"))
    story.append(
        code(
            "7.1: add drag-and-drop upload UI with progress bar\n\n"
            "9.1: backend Dockerfile — multi-stage, non-root user, <1.5GB image"
        )
    )
    story.append(bullet("Commit small and often. One logical change per commit."))
    story.append(
        bullet("Never commit backend/.env — it is gitignored and contains secrets.")
    )
    story.append(
        warn(
            "If you accidentally commit a secret, tell Bhavesh immediately. Do not try to fix it silently."
        )
    )

    story.append(h2("Step 7 — Push to your fork"))
    story.append(code("git push origin feature/<task-id>-<slug>"))

    story.append(h2("Step 8 — Open a pull request"))
    story.append(bullet("Go to your fork on GitHub."))
    story.append(bullet("Click 'Compare & pull request'."))
    story.append(
        bullet("Set base repository: BhaveshKhaple/legalvault-ai, base branch: v1.")
    )
    story.append(
        bullet("Set head repository: <your-fork>, compare: feature/<your-branch>.")
    )
    story.append(body("Do NOT target main. Only v1."))
    story.append(rule())

    # ── Section 5 — PR template ───────────────────────────────────────────────
    story.append(PageBreak())
    story.append(h1("5. Pull Request Template"))
    story.append(
        body("Copy-paste this template into the PR description. Fill every section.")
    )
    story.append(sp(4))

    pr_template = Table(
        [
            [
                Paragraph(
                    "## Summary\n\n"
                    "_One paragraph: what this PR does and why._\n\n"
                    "## Tracker task\n\n"
                    "- Task ID: (e.g. 7.1)\n"
                    "- Tracker row: [link to spreadsheet row]\n\n"
                    "## Acceptance criteria\n\n"
                    "- [ ] &lt;criterion 1 from tracker&gt;\n"
                    "- [ ] &lt;criterion 2 from tracker&gt;\n\n"
                    "## Files changed\n\n"
                    "_List only the files in the task's assigned folder. "
                    "If you changed anything outside, explain why._\n\n"
                    "## How to test\n\n"
                    "1. Pull this branch.\n"
                    "2. Run: `pytest tests/&lt;module&gt;/`\n"
                    "3. Manually verify: &lt;specific UI action or API call&gt;\n\n"
                    "## Screenshots (if UI task)\n\n"
                    "_Attach before/after screenshots._\n\n"
                    "## Known gaps / deviations\n\n"
                    "_Anything you deliberately left out, and why._",
                    BODY,
                )
            ]
        ],
        colWidths=[170 * mm],
    )
    pr_template.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), IVORY),
                ("GRID", (0, 0), (-1, -1), 0.5, GOLD),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    story.append(pr_template)
    story.append(sp(10))

    story.append(h2("5.1 How to retarget an existing PR (Vijay — action required)"))
    story.append(
        body(
            "Your open PR targets main instead of v1. GitHub does not allow changing the base "
            "of a PR to a different repo, so the easiest fix is:"
        )
    )
    story.append(
        code(
            "# On your local machine, inside your fork:\n"
            "git fetch upstream\n"
            "git checkout adding-my-code\n"
            "git rebase upstream/v1\n"
            "git push origin adding-my-code --force-with-lease"
        )
    )
    story.append(
        body(
            "Then close the old PR and open a new one targeting v1. "
            "Paste the PR template above into the new PR description."
        )
    )
    story.append(rule())

    # ── Section 6 — Code rules ────────────────────────────────────────────────
    story.append(h1("6. Code Rules"))
    story.append(h2("6.1 What to do"))
    for item in [
        "Python 3.11+. Type hints on all public functions.",
        "Run ruff format . and ruff check . before every push.",
        "Write tests first (or at minimum alongside), never after review.",
        "Add a docs/updates/<task-id>.md describing what you built, files touched, how to test.",
        "Keep commits atomic: one logical change per commit.",
        "Understand every line you push — if you used AI to generate it, be ready to explain it.",
    ]:
        story.append(bullet(item))

    story.append(h2("6.2 What not to do"))
    for item in [
        "Do not push directly to v1 or main. Ever.",
        "Do not add a new pip dependency without posting in WhatsApp first.",
        "Do not change files outside your task's assigned folder without a justification comment.",
        "Do not commit .env, credentials, client PDFs, or any file not tracked by git.",
        "Do not merge your own PR. Wait for Bhavesh.",
        "Do not mark a task Done on the tracker until the PR is merged to v1.",
    ]:
        story.append(bullet(item))
    story.append(rule())

    # ── Section 7 — PR review checklist ──────────────────────────────────────
    story.append(h1("7. PR Review Checklist"))
    story.append(
        body(
            "Bhavesh checks these before merging. Self-review against this list before you "
            "request review — it saves at least one round-trip."
        )
    )
    story.append(sp(4))
    story.append(checklist_table())
    story.append(rule())

    # ── Section 8 — RAG optimization roadmap ─────────────────────────────────
    story.append(h1("8. Active RAG Optimization Roadmap"))
    story.append(
        body(
            "These are the next planned feature branches. Pick from this list when the "
            "tracker shows them as Ready. Do not start a roadmap branch before the "
            "preceding phase is merged to v1."
        )
    )
    story.append(sp(4))
    story.append(roadmap_table())
    story.append(sp(6))
    story.append(
        note(
            "Phase 5 (VLM multimodal) requires a GPU with >= 8GB VRAM. "
            "Do not attempt it on the Legion GTX 1650 laptop. "
            "It will be executed on the team's secondary workstation."
        )
    )
    story.append(rule())

    # ── Section 9 — When you're stuck ────────────────────────────────────────
    story.append(h1("9. When You Are Stuck"))
    story.append(
        bullet(
            "Under 30 minutes stuck: try one more thing, then post in WhatsApp with what you tried."
        )
    )
    story.append(
        bullet(
            "Over 30 minutes stuck: open a Draft PR and tag Bhavesh with the specific error."
        )
    )
    story.append(
        bullet(
            "Never silently abandon a task. Set Status = Blocked on the tracker so it can be picked up."
        )
    )
    story.append(
        bullet(
            "If you break something in v1, tell Bhavesh immediately. Do not try to silently fix history."
        )
    )
    story.append(sp(8))
    story.append(
        body(
            "Questions about task specs, codebase decisions, or whether something is in scope "
            "belong in the WhatsApp group — not as GitHub comments after you have already built it. "
            "Ask before you build."
        )
    )
    story.append(rule())

    # ── Footer note ───────────────────────────────────────────────────────────
    story.append(sp(8))
    story.append(
        note(
            "Document generated 2 October 2026 · LegalVault AI — Team Pied Piper · "
            "github.com/BhaveshKhaple/legalvault-ai"
        )
    )

    doc.build(story)
    print(f"Built: {OUT.resolve()}")


if __name__ == "__main__":
    build()
