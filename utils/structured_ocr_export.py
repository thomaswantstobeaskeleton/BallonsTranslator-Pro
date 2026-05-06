from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, Optional


def _as_list(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [float(v) if isinstance(v, (int, float)) else v for v in value]
    return value


def _font_dict(block) -> dict:
    fmt = getattr(block, "fontformat", None)
    if fmt is None:
        return {}
    return {
        "family": getattr(fmt, "font_family", ""),
        "size_px": float(getattr(fmt, "font_size", 0.0) or 0.0),
        "alignment": int(getattr(fmt, "alignment", 0) or 0),
        "vertical": bool(getattr(fmt, "vertical", False)),
        "bold": bool(getattr(fmt, "bold", False)),
        "italic": bool(getattr(fmt, "italic", False)),
        "stroke_width": float(getattr(fmt, "stroke_width", 0.0) or 0.0),
    }


def build_structured_ocr_export(project, pages: Optional[Iterable[str]] = None) -> dict:
    """Build a stable JSON-serializable OCR/layout export for LLM and QA tools.

    The schema intentionally mirrors project state instead of re-running OCR:
    page metadata, block order, geometry, source/translation text, and render font
    hints are exported exactly as currently stored.
    """
    page_names = list(pages) if pages is not None else list(getattr(project, "pages", {}) or {})
    out_pages = []
    image_info = getattr(project, "_image_info", {}) or {}
    for page_index, page_name in enumerate(page_names):
        blocks = list((getattr(project, "pages", {}) or {}).get(page_name, []) or [])
        info = image_info.get(page_name, {}) if isinstance(image_info, dict) else {}
        out_blocks = []
        for block_index, block in enumerate(blocks):
            text = block.get_text() if hasattr(block, "get_text") else getattr(block, "text", "")
            out_blocks.append(
                {
                    "index": block_index,
                    "xyxy": _as_list(getattr(block, "xyxy", None)),
                    "lines": _as_list(getattr(block, "lines", None)),
                    "angle": float(getattr(block, "angle", 0.0) or 0.0),
                    "source_text": text or "",
                    "translation": getattr(block, "translation", "") or "",
                    "font": _font_dict(block),
                    "label": getattr(block, "label", "") or "",
                    "confidence": float(getattr(block, "confidence", 0.0) or 0.0),
                }
            )
        completion = "todo"
        if hasattr(project, "get_page_completion_state"):
            completion = project.get_page_completion_state(page_name)
        out_pages.append(
            {
                "index": page_index,
                "name": page_name,
                "width": int(info.get("width", 0) or 0),
                "height": int(info.get("height", 0) or 0),
                "ignored": bool(info.get("ignored", False)),
                "finish_code": int(info.get("finish_code", 0) or 0),
                "completion_state": completion,
                "blocks": out_blocks,
            }
        )
    return {
        "schema": "ballonstranslator.structured_ocr.v1",
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "project": getattr(project, "directory", "") or "",
        "current_page": getattr(project, "current_img", None),
        "pages": out_pages,
    }
