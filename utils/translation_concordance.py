from __future__ import annotations

from typing import Any, Dict, List


def build_concordance_from_project(project) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for page, blks in (getattr(project, 'pages', {}) or {}).items():
        for idx, blk in enumerate(blks or []):
            src = (getattr(blk, 'get_text', lambda: '')() or '').strip()
            tgt = (getattr(blk, 'translation', '') or '').strip()
            if not src and not tgt:
                continue
            rows.append({
                'source': src,
                'target': tgt,
                'page': str(page or ''),
                'index': int(idx),
                'block_id': str(getattr(blk, 'api_block_id', '') or ''),
            })
    return rows


def query_concordance(rows: List[Dict[str, Any]], query: str, *, in_target: bool = True, limit: int = 50) -> List[Dict[str, Any]]:
    needle = str(query or '').strip().lower()
    if not needle:
        return []
    out: List[Dict[str, Any]] = []
    for row in rows or []:
        src = str((row or {}).get('source', '') or '')
        tgt = str((row or {}).get('target', '') or '')
        hay = src.lower() + ('\n' + tgt.lower() if in_target else '')
        if needle in hay:
            out.append(dict(row))
    return out[:max(1, int(limit))]
