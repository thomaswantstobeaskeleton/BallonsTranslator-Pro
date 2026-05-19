from __future__ import annotations

import csv
from typing import Dict, List


def export_glossary_csv(entries: List[Dict[str, str]], out_path: str) -> int:
    rows = list(entries or [])
    with open(out_path, 'w', encoding='utf-8-sig', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=['source', 'target'])
        wr.writeheader()
        for row in rows:
            wr.writerow({'source': str((row or {}).get('source', '') or ''), 'target': str((row or {}).get('target', '') or '')})
    return len(rows)


def import_glossary_csv(path: str) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        rd = csv.DictReader(f)
        for row in rd:
            s = str((row or {}).get('source', '') or '').strip()
            t = str((row or {}).get('target', '') or '').strip()
            if s:
                out.append({'source': s, 'target': t})
    return out


def preview_glossary_merge(existing: List[Dict[str, str]], incoming: List[Dict[str, str]], *, mode: str = "merge") -> Dict[str, object]:
    mode = str(mode or "merge").strip().lower()
    cur = list(existing or [])
    if mode == "replace":
        base = []
    else:
        base = cur
    seen = {str((r or {}).get('source', '') or '').strip() for r in base}
    added = []
    skipped = []
    for row in incoming or []:
        src = str((row or {}).get('source', '') or '').strip()
        tgt = str((row or {}).get('target', '') or '').strip()
        if not src:
            skipped.append({'source': src, 'target': tgt, 'reason': 'empty_source'})
            continue
        if src in seen:
            skipped.append({'source': src, 'target': tgt, 'reason': 'duplicate_source'})
            continue
        added.append({'source': src, 'target': tgt})
        seen.add(src)
    return {
        'mode': mode,
        'existing_count': len(cur),
        'incoming_count': len(list(incoming or [])),
        'added_count': len(added),
        'skipped_count': len(skipped),
        'added_preview': added[:20],
        'skipped_preview': skipped[:20],
        'result_count': len(base) + len(added),
    }
