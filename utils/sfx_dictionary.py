from __future__ import annotations

import csv
import json
from typing import Dict, List

_DEFAULT = [
    {"source": "ドン", "target": "THUD", "style": "impact"},
    {"source": "バン", "target": "BANG", "style": "impact"},
    {"source": "ザアア", "target": "SHHHH", "style": "ambient"},
    {"source": "キラ", "target": "SPARKLE", "style": "fx"},
]


def default_sfx_dictionary() -> List[Dict[str, str]]:
    return [dict(r) for r in _DEFAULT]


def _norm(v: str) -> str:
    return str(v or "").strip().lower()


def query_sfx_dictionary(entries: List[Dict[str, str]], query: str, *, limit: int = 50) -> List[Dict[str, str]]:
    needle = _norm(query)
    if not needle:
        return []
    out: List[Dict[str, str]] = []
    for row in entries or []:
        src = str((row or {}).get("source", "") or "")
        tgt = str((row or {}).get("target", "") or "")
        sty = str((row or {}).get("style", "") or "")
        hay = f"{src}\n{tgt}\n{sty}".lower()
        if needle in hay:
            out.append({"source": src, "target": tgt, "style": sty})
    return out[: max(1, int(limit))]


def merge_sfx_entries(existing: List[Dict[str, str]], incoming: List[Dict[str, str]]) -> Dict[str, object]:
    merged = [dict(r) for r in (existing or [])]
    by_key = {_norm(r.get("source", "")): i for i, r in enumerate(merged) if _norm(r.get("source", ""))}
    added = 0
    updated = 0
    for row in incoming or []:
        src = str((row or {}).get("source", "") or "").strip()
        if not src:
            continue
        tgt = str((row or {}).get("target", "") or "").strip()
        sty = str((row or {}).get("style", "") or "").strip()
        k = _norm(src)
        payload = {"source": src, "target": tgt, "style": sty}
        if k in by_key:
            merged[by_key[k]] = payload
            updated += 1
        else:
            by_key[k] = len(merged)
            merged.append(payload)
            added += 1
    return {"entries": merged, "added": added, "updated": updated, "count": len(merged)}


def export_sfx_dictionary(entries: List[Dict[str, str]], out_path: str, fmt: str = "json") -> int:
    fmt = str(fmt or "json").strip().lower()
    rows = list(entries or [])
    if fmt == "csv":
        with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=["source", "target", "style"])
            wr.writeheader()
            for r in rows:
                wr.writerow({"source": r.get("source", ""), "target": r.get("target", ""), "style": r.get("style", "")})
    else:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
    return len(rows)


def import_sfx_dictionary(path: str) -> List[Dict[str, str]]:
    low = path.lower()
    if low.endswith(".csv"):
        out = []
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                src = str((row or {}).get("source", "") or "").strip()
                if src:
                    out.append({"source": src, "target": str((row or {}).get("target", "") or "").strip(), "style": str((row or {}).get("style", "") or "").strip()})
        return out
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for row in data or []:
        src = str((row or {}).get("source", "") or "").strip()
        if src:
            out.append({"source": src, "target": str((row or {}).get("target", "") or "").strip(), "style": str((row or {}).get("style", "") or "").strip()})
    return out
