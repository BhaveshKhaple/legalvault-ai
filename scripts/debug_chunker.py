"""Debug what the hierarchical chunker produces for d06 and d09."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.ingestion.docling_extractor import extract_docling
from backend.app.ingestion.hierarchical_chunker import chunk_hierarchical

for name in ["d06_fee_schedule.pdf", "d09_faq.docx"]:
    path = Path("tests/eval/corpus") / name
    print(f"\n{'='*72}\n{name}\n{'='*72}")
    try:
        pages = extract_docling(str(path))
    except Exception as exc:
        print(f"Docling failed: {exc}")
        continue
    print(f"Docling pages: {len(pages)}")
    for p in pages[:2]:
        print(f"  page {p['page']}: {len(p.get('items', []))} items")
        for it in (p.get("items") or [])[:8]:
            label = it.get("heading_level") or it.get("label")
            text = (it.get("text") or "")[:80]
            print(f"    [{label}] {text!r}")

    r = chunk_hierarchical(pages)
    print(f"\nparents: {len(r['parents'])}, children: {len(r['children'])}")
    for i, p in enumerate(r["parents"][:5]):
        print(f"  parent[{i}] is_table={p['is_table']} ({len(p['content'])} chars): {p['content'][:100]!r}")
    for c in r["children"][:5]:
        print(f"  child -> parent[{c['parent_index']}]: {c['content'][:100]!r}")
    # Check whether target substrings appear in any child content
    for needle in ["1,25,000", "6,500", "No. All processing", "8 GB RAM"]:
        hits_child = sum(1 for c in r["children"] if needle.lower() in c["content"].lower())
        hits_parent = sum(1 for p in r["parents"] if needle.lower() in p["content"].lower())
        print(f"  needle {needle!r}: children={hits_child}, parents={hits_parent}")
