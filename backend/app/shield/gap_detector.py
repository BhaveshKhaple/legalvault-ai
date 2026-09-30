"""
Task 5.3 — Missing-section detector for Report Shield.

Checks an uploaded document's text against a YAML template of required
sections for its document type. Uses keyword matching — simple and fast,
no model inference needed, no false negatives for standard clause labels.

Each template lists required sections with keyword lists. A section is
considered PRESENT if any of its keywords appear anywhere in the full
document text (case-insensitive). Missing = none of the keywords found.

Templates live in backend/app/shield/templates/<doc_type>.yaml.
Available doc types: contract, audit_report, agreement.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).parent / "templates"


@dataclass
class SectionResult:
    name: str
    present: bool
    matched_keyword: str | None   # first keyword that matched (None if missing)
    description: str = ""


@dataclass
class GapReport:
    doc_type: str
    present: list[SectionResult] = field(default_factory=list)
    missing: list[SectionResult] = field(default_factory=list)

    @property
    def missing_count(self) -> int:
        return len(self.missing)

    @property
    def present_count(self) -> int:
        return len(self.present)

    def to_dict(self) -> dict:
        return {
            "doc_type": self.doc_type,
            "present_count": self.present_count,
            "missing_count": self.missing_count,
            "present": [{"name": s.name, "matched_keyword": s.matched_keyword} for s in self.present],
            "missing": [{"name": s.name, "description": s.description} for s in self.missing],
        }


def _load_template(doc_type: str) -> dict | None:
    """Load the YAML template for a document type. Returns None if unknown."""
    path = _TEMPLATES_DIR / f"{doc_type}.yaml"
    if not path.exists():
        logger.warning("No template found for doc_type='%s' at %s", doc_type, path)
        return None
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _keyword_present(text_lower: str, keywords: list[str]) -> str | None:
    """Return the first matching keyword, or None."""
    for kw in keywords:
        pattern = re.compile(re.escape(kw.lower()))
        if pattern.search(text_lower):
            return kw
    return None


def detect_gaps(full_text: str, doc_type: str) -> GapReport:
    """Check full_text for required sections defined in the doc_type template.

    Args:
        full_text: Complete extracted text of the document.
        doc_type:  One of 'contract', 'audit_report', 'agreement'.

    Returns:
        GapReport with lists of present and missing sections.
    """
    report = GapReport(doc_type=doc_type)
    template = _load_template(doc_type)

    if template is None:
        # Unknown doc type — return empty report (no sections to check)
        return report

    text_lower = full_text.lower()
    for section in template.get("required_sections", []):
        name = section["name"]
        keywords = section.get("keywords", [])
        description = section.get("description", "")

        match = _keyword_present(text_lower, keywords)
        result = SectionResult(
            name=name,
            present=match is not None,
            matched_keyword=match,
            description=description,
        )
        if result.present:
            report.present.append(result)
        else:
            report.missing.append(result)

    logger.debug(
        "Gap detection for doc_type=%s: %d present, %d missing",
        doc_type, report.present_count, report.missing_count,
    )
    return report


def detect_gaps_from_pages(pages: list[dict], doc_type: str) -> GapReport:
    """Convenience wrapper: extract text from pdf_extractor page dicts."""
    full_text = "\n".join(p.get("text", "") for p in pages)
    return detect_gaps(full_text, doc_type)
