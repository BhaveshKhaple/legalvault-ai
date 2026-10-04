"""Phase 2 smoke test — exercise hierarchical_chunker on synthetic Docling pages."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.app.ingestion.hierarchical_chunker import chunk_hierarchical

pages = [{
    "text": "x",
    "page": 1,
    "doc_id": "d",
    "items": [
        {"heading_level": "H1", "text": "Section 1 - Terms", "page": 1},
        {"heading_level": "body", "text": "The parties agree to pay Rs. 50,000 per month.", "page": 1},
        {"heading_level": "body", "text": "Payment is due by the 5th of each month.", "page": 1},
        {"heading_level": "H1", "text": "Section 2 - Termination", "page": 2},
        {"heading_level": "body", "text": "Either party may terminate with 60 days notice.", "page": 2},
        {"heading_level": "table", "text": "Service | Fee\nSetup | 10000\nMonthly | 5000", "page": 3},
    ],
}]
r = chunk_hierarchical(pages)
print(f"parents: {len(r['parents'])}, children: {len(r['children'])}")
for i, p in enumerate(r["parents"]):
    print(f"  parent[{i}]: {p['section_title']!r} ({len(p['content'])} chars) is_table={p.get('is_table')}")
for c in r["children"]:
    print(f"  child -> parent[{c['parent_index']}] p{c['page']}: {c['content'][:50]!r}")
