from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass
class ReplaceHit:
    page: str
    index: int
    field: str
    before: str
    after: str
    count: int


def _compile(pattern: str, use_regex: bool, case_sensitive: bool):
    flags = 0 if case_sensitive else re.IGNORECASE
    if use_regex:
        return re.compile(pattern, flags)
    return re.compile(re.escape(pattern), flags)


def preview_batch_find_replace(project, pattern: str, replacement: str, *, use_regex: bool = True, case_sensitive: bool = False, target: str = "translation", pages: List[str] | None = None) -> Dict[str, Any]:
    if not pattern:
        return {"hits": [], "count": 0}
    rx = _compile(pattern, use_regex, case_sensitive)
    hits: List[Dict[str, Any]] = []
    page_list = pages or list((getattr(project, "pages", {}) or {}).keys())
    for page in page_list:
        blks = (getattr(project, "pages", {}) or {}).get(page, []) or []
        for idx, blk in enumerate(blks):
            before = (getattr(blk, "translation", "") if target == "translation" else getattr(blk, "get_text", lambda: "")()) or ""
            after, n = rx.subn(replacement, before)
            if n > 0 and after != before:
                hits.append({"page": page, "index": idx, "field": target, "before": before, "after": after, "count": n})
    return {"hits": hits, "count": len(hits)}


def apply_batch_find_replace(project, preview_payload: Dict[str, Any]) -> Tuple[int, List[Dict[str, Any]]]:
    changed = 0
    applied: List[Dict[str, Any]] = []
    for hit in preview_payload.get("hits", []) or []:
        page = str(hit.get("page", "") or "")
        idx = int(hit.get("index", -1))
        field = str(hit.get("field", "translation") or "translation")
        after = str(hit.get("after", "") or "")
        blks = (getattr(project, "pages", {}) or {}).get(page, []) or []
        if idx < 0 or idx >= len(blks):
            continue
        blk = blks[idx]
        if field == "translation":
            if (getattr(blk, "translation", "") or "") != after:
                blk.translation = after
                changed += 1
                applied.append({"page": page, "index": idx, "field": field})
    return changed, applied
