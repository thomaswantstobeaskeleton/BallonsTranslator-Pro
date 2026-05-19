from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Mapping
import uuid


@dataclass
class EditValidationError(Exception):
    code: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": self.details}


@dataclass
class TextboxOperation:
    op: str
    index: Optional[int] = None
    text: Optional[str] = None
    rect: Optional[List[float]] = None
    page: Optional[str] = None
    block_id: Optional[str] = None


def _as_int(value: Any, field_name: str) -> int:
    try:
        return int(value)
    except Exception as e:
        raise EditValidationError("invalid_field_type", f"{field_name} must be integer", {"field": field_name}) from e


def validate_textbox_operation(raw: Dict[str, Any]) -> TextboxOperation:
    if not isinstance(raw, dict):
        raise EditValidationError("invalid_payload", "operation payload must be an object")
    op = str(raw.get("op", "") or "").strip().lower()
    if op not in {"add_textbox", "update_textbox", "delete_textbox", "undo", "redo"}:
        raise EditValidationError("unsupported_operation", "unsupported operation", {"op": op})
    item = TextboxOperation(op=op, page=str(raw.get("page", "") or "").strip() or None)
    if op in {"update_textbox", "delete_textbox"}:
        if "index" in raw:
            item.index = _as_int(raw.get("index"), "index")
        elif "block_id" in raw and str(raw.get("block_id","")).strip():
            item.block_id = str(raw.get("block_id", "")).strip()
        else:
            raise EditValidationError("missing_field", "index or block_id is required", {"field": "index|block_id", "op": op})
    if op == "update_textbox":
        if "text" not in raw:
            raise EditValidationError("missing_field", "text is required for update_textbox", {"field": "text"})
        item.text = str(raw.get("text", ""))
    if op == "add_textbox":
        rect = raw.get("rect")
        if rect is not None:
            if not isinstance(rect, (list, tuple)) or len(rect) != 4:
                raise EditValidationError("invalid_field_type", "rect must be [x, y, w, h]", {"field": "rect"})
            item.rect = [float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3])]
        item.text = str(raw.get("text", "") or "")
    return item


def validate_batch_payload(body: Dict[str, Any]) -> List[TextboxOperation]:
    if not isinstance(body, dict):
        raise EditValidationError("invalid_payload", "request body must be an object")
    ops = body.get("ops")
    if not isinstance(ops, list) or not ops:
        raise EditValidationError("missing_field", "ops must be a non-empty list", {"field": "ops"})
    out: List[TextboxOperation] = []
    for i, raw in enumerate(ops):
        try:
            out.append(validate_textbox_operation(raw))
        except EditValidationError as e:
            raise EditValidationError(e.code, e.message, {**e.details, "op_index": i}) from e
    return out



def ensure_block_stable_id(block: Any) -> str:
    """Return a stable automation id for one textbox block.

    IDs are persisted on the block object when possible so subsequent API calls
    can target the same textbox even if list indexes move.
    """
    for key in ("api_block_id", "block_id", "id"):
        try:
            value = getattr(block, key, None)
        except Exception:
            value = None
        if isinstance(value, str) and value.strip():
            return value.strip()
    new_id = f"tbx_{uuid.uuid4().hex[:16]}"
    try:
        setattr(block, "api_block_id", new_id)
    except Exception:
        pass
    return new_id


def find_block_index_by_stable_id(blocks: List[Any], block_id: str) -> int:
    target = str(block_id or "").strip()
    if not target:
        raise EditValidationError("missing_field", "block_id is required", {"field": "block_id"})
    for idx, block in enumerate(blocks or []):
        if ensure_block_stable_id(block) == target:
            return idx
    raise EditValidationError("not_found", "textbox not found", {"block_id": target})


def describe_block_ref(page: str, index: int, block: Any) -> Dict[str, Any]:
    return {
        "page": str(page or ""),
        "index": int(index),
        "block_id": ensure_block_stable_id(block),
    }
