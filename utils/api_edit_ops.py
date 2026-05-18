from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


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
        if "index" not in raw:
            raise EditValidationError("missing_field", "index is required", {"field": "index", "op": op})
        item.index = _as_int(raw.get("index"), "index")
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
