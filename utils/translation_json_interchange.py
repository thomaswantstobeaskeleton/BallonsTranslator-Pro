from __future__ import annotations

from typing import Any, Dict, List, Tuple
from utils.api_edit_ops import ensure_block_stable_id

SCHEMA = "ballonstranslator.translation_json.v1"


def export_translation_json(project) -> Dict[str, Any]:
    pages_payload: List[Dict[str, Any]] = []
    for page_name, blks in (getattr(project, "pages", {}) or {}).items():
        rows = []
        for idx, blk in enumerate(blks or []):
            rows.append({
                "index": idx,
                "block_id": ensure_block_stable_id(blk),
                "source": (getattr(blk, "get_text", lambda: "")() or ""),
                "translation": (getattr(blk, "translation", "") or ""),
            })
        pages_payload.append({"page": str(page_name), "blocks": rows})
    return {"schema": SCHEMA, "pages": pages_payload}


def import_translation_json(project, payload: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    pages = getattr(project, "pages", {}) or {}
    matched_pages, missing_pages, unmatched_pages = set(), [], []
    for page_rec in (payload.get("pages") or []):
        page = str((page_rec or {}).get("page", "") or "")
        if page not in pages:
            missing_pages.append(page)
            continue
        matched_pages.add(page)
        src_blocks = list((page_rec or {}).get("blocks") or [])
        dst_blocks = pages.get(page, []) or []
        if len(src_blocks) != len(dst_blocks):
            unmatched_pages.append(page)
        id_to_idx = {ensure_block_stable_id(b): i for i, b in enumerate(dst_blocks)}
        for i, rec in enumerate(src_blocks):
            target_idx = None
            bid = str((rec or {}).get("block_id", "") or "").strip()
            if bid and bid in id_to_idx:
                target_idx = id_to_idx[bid]
            elif i < len(dst_blocks):
                target_idx = i
            if target_idx is None:
                continue
            dst_blocks[target_idx].translation = str((rec or {}).get("translation", "") or "")
    ok = not missing_pages and not unmatched_pages
    return ok, {
        "matched_pages": matched_pages,
        "missing_pages": missing_pages,
        "unmatched_pages": unmatched_pages,
        "unexpected_pages": [],
    }
