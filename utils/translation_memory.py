from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any, Dict, List

SCHEMA = "ballonstranslator.tm.v1"


def normalize_text(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def add_tm_entry(store: List[Dict[str, str]], source: str, target: str, *, page: str = "", block_id: str = "") -> None:
    src = (source or "").strip()
    tgt = (target or "").strip()
    if not src or not tgt:
        return
    for row in store:
        if normalize_text(row.get("source", "")) == normalize_text(src) and normalize_text(row.get("target", "")) == normalize_text(tgt):
            return
    store.append({"source": src, "target": tgt, "page": page or "", "block_id": block_id or ""})


def build_tm_from_project(project) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for page, blks in (getattr(project, "pages", {}) or {}).items():
        for blk in blks or []:
            source = (getattr(blk, "get_text", lambda: "")() or "")
            target = (getattr(blk, "translation", "") or "")
            add_tm_entry(rows, source, target, page=str(page), block_id=str(getattr(blk, "api_block_id", "") or ""))
    return rows


def query_tm(store: List[Dict[str, str]], source: str, *, min_score: float = 0.65, limit: int = 5) -> List[Dict[str, Any]]:
    needle = normalize_text(source)
    if not needle:
        return []
    out: List[Dict[str, Any]] = []
    for row in store or []:
        cand = normalize_text(row.get("source", ""))
        if not cand:
            continue
        score = SequenceMatcher(None, needle, cand).ratio()
        if score >= float(min_score):
            out.append({**row, "score": round(score, 4)})
    out.sort(key=lambda r: r.get("score", 0.0), reverse=True)
    return out[: max(1, int(limit))]


def export_tm_payload(store: List[Dict[str, str]]) -> Dict[str, Any]:
    return {"schema": SCHEMA, "entries": list(store or [])}


def import_tm_payload(payload: Dict[str, Any]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for row in (payload.get("entries") or []):
        add_tm_entry(out, str((row or {}).get("source", "") or ""), str((row or {}).get("target", "") or ""), page=str((row or {}).get("page", "") or ""), block_id=str((row or {}).get("block_id", "") or ""))
    return out
