"""Stable-ish JSON operation protocol for project text edits."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ProjectOpSession:
    pages: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    undo_stack: List[List[Dict[str, Any]]] = field(default_factory=list)
    redo_stack: List[List[Dict[str, Any]]] = field(default_factory=list)

    def snapshot(self) -> Dict[str, Any]:
        return {"pages": self.pages}


def apply_ops(session: ProjectOpSession, ops: List[Dict[str, Any]]) -> Dict[str, Any]:
    applied = []
    for op in ops or []:
        t = str(op.get("op", "")).strip().lower()
        if t == "updatetext":
            page = str(op.get("page", "")).strip()
            idx = int(op.get("index", -1))
            text = str(op.get("text", ""))
            blks = session.pages.get(page) or []
            if 0 <= idx < len(blks):
                prev = str(blks[idx].get("translation", ""))
                blks[idx]["translation"] = text
                applied.append({"op": "UpdateText", "page": page, "index": idx, "prev": prev, "text": text})
        elif t == "batch":
            sub = op.get("ops") if isinstance(op.get("ops"), list) else []
            sub_result = apply_ops(session, sub)
            applied.extend(sub_result.get("applied", []))
    if applied:
        session.undo_stack.append(applied)
        session.redo_stack.clear()
    return {"applied": applied, "count": len(applied)}


def undo(session: ProjectOpSession) -> int:
    if not session.undo_stack:
        return 0
    chunk = session.undo_stack.pop()
    for op in reversed(chunk):
        if op.get("op") == "UpdateText":
            page = op["page"]
            idx = int(op["index"])
            blks = session.pages.get(page) or []
            if 0 <= idx < len(blks):
                blks[idx]["translation"] = op.get("prev", "")
    session.redo_stack.append(chunk)
    return len(chunk)


def redo(session: ProjectOpSession) -> int:
    if not session.redo_stack:
        return 0
    chunk = session.redo_stack.pop()
    for op in chunk:
        if op.get("op") == "UpdateText":
            page = op["page"]
            idx = int(op["index"])
            blks = session.pages.get(page) or []
            if 0 <= idx < len(blks):
                blks[idx]["translation"] = op.get("text", "")
    session.undo_stack.append(chunk)
    return len(chunk)
