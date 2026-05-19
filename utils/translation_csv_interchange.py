from __future__ import annotations

import csv
from io import StringIO
from typing import Any, Dict, List, Tuple

from utils.api_edit_ops import ensure_block_stable_id

SCHEMA = "ballonstranslator.translation_csv.v1"
FIELDS = ["page", "index", "block_id", "source", "translation"]


def export_translation_csv_text(project) -> str:
    bio = StringIO()
    w = csv.DictWriter(bio, fieldnames=FIELDS)
    w.writeheader()
    for page, blks in (getattr(project, "pages", {}) or {}).items():
        for idx, blk in enumerate(blks or []):
            w.writerow({
                "page": str(page),
                "index": int(idx),
                "block_id": ensure_block_stable_id(blk),
                "source": (getattr(blk, "get_text", lambda: "")() or ""),
                "translation": (getattr(blk, "translation", "") or ""),
            })
    return bio.getvalue()


def import_translation_csv_text(project, text: str) -> Tuple[bool, Dict[str, Any]]:
    rows = list(csv.DictReader(StringIO(text or "")))
    pages = getattr(project, "pages", {}) or {}
    matched_pages, missing_pages, unmatched_pages = set(), [], []

    by_page: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        page = str((row or {}).get("page", "") or "")
        by_page.setdefault(page, []).append(row)

    for page, page_rows in by_page.items():
        if page not in pages:
            missing_pages.append(page)
            continue
        matched_pages.add(page)
        dst_blocks = pages.get(page, []) or []
        id_to_idx = {ensure_block_stable_id(b): i for i, b in enumerate(dst_blocks)}
        touched = 0
        for row in page_rows:
            idx = None
            bid = str((row or {}).get("block_id", "") or "").strip()
            if bid and bid in id_to_idx:
                idx = id_to_idx[bid]
            else:
                try:
                    idx = int((row or {}).get("index", ""))
                except Exception:
                    idx = None
            if idx is None or idx < 0 or idx >= len(dst_blocks):
                continue
            dst_blocks[idx].translation = str((row or {}).get("translation", "") or "")
            touched += 1
        if touched != len(dst_blocks):
            unmatched_pages.append(page)

    ok = not missing_pages and not unmatched_pages
    return ok, {
        "schema": SCHEMA,
        "matched_pages": matched_pages,
        "missing_pages": missing_pages,
        "unmatched_pages": unmatched_pages,
        "unexpected_pages": [],
    }
