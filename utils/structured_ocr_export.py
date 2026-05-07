from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, Optional

from .text_rendering import resolve_writing_mode, sort_blocks_for_reading_order, vertical_layout_plan


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
        "line_spacing": float(getattr(fmt, "line_spacing", 1.0) or 1.0),
        "letter_spacing": float(getattr(fmt, "letter_spacing", 1.0) or 1.0),
        "text_padding": float(getattr(fmt, "text_padding", 0.0) or 0.0),
        "writing_mode": getattr(fmt, "writing_mode", "auto"),
        "fit_mode": getattr(fmt, "fit_mode", "shrink"),
        "line_break_strategy": getattr(fmt, "line_break_strategy", "auto"),
        "fallback_font_chain": getattr(fmt, "fallback_font_chain", ""),
        "manga_preset": getattr(fmt, "manga_preset", ""),
    }


def _render_hints(block, text: str) -> dict:
    fmt = getattr(block, "fontformat", None)
    xyxy = list(getattr(block, "xyxy", []) or [])
    box = (0.0, 0.0)
    if len(xyxy) >= 4:
        box = (max(0.0, float(xyxy[2]) - float(xyxy[0])), max(0.0, float(xyxy[3]) - float(xyxy[1])))
    writing_mode = getattr(fmt, "writing_mode", "auto") if fmt is not None else "auto"
    resolved = resolve_writing_mode(writing_mode, text or "", box)
    hints = {"resolved_writing_mode": resolved, "box_size": [box[0], box[1]]}
    if resolved == "vertical_rl":
        font_size = float(getattr(fmt, "font_size", 24.0) or 24.0) if fmt is not None else 24.0
        letter_spacing = float(getattr(fmt, "letter_spacing", 1.0) or 1.0) if fmt is not None else 1.0
        line_spacing = float(getattr(fmt, "line_spacing", 1.1) or 1.1) if fmt is not None else 1.1
        max_chars = max(1, int(max(1.0, box[1]) / max(1.0, font_size * max(0.1, letter_spacing))))
        plan = vertical_layout_plan(text or "", max_chars, font_size=font_size, line_spacing=line_spacing, letter_spacing=letter_spacing, strategy=getattr(fmt, "line_break_strategy", "cjk_strict") if fmt is not None else "cjk_strict")
        hints.update({
            "vertical_columns": plan.get("columns", []),
            "tate_chu_yoko_groups": plan.get("tate_chu_yoko_groups", []),
            "bracket_pairs": plan.get("bracket_pairs", []),
            "punctuation_hang": plan.get("punctuation_hang", True),
        })
    return hints


def build_structured_ocr_export(project, pages: Optional[Iterable[str]] = None, reading_order: str = "auto") -> dict:
    """Build a stable JSON-serializable OCR/layout export for LLM and QA tools.

    The schema intentionally mirrors project state instead of re-running OCR:
    page metadata, block order, geometry, source/translation text, and render font
    hints are exported exactly as currently stored.
    """
    page_names = list(pages) if pages is not None else list(getattr(project, "pages", {}) or {})
    out_pages = []
    image_info = getattr(project, "_image_info", {}) or {}
    for page_index, page_name in enumerate(page_names):
        original_blocks = list((getattr(project, "pages", {}) or {}).get(page_name, []) or [])
        blocks, resolved_reading_order = sort_blocks_for_reading_order(original_blocks, reading_order)
        original_index = {id(block): i for i, block in enumerate(original_blocks)}
        info = image_info.get(page_name, {}) if isinstance(image_info, dict) else {}
        out_blocks = []
        for block_index, block in enumerate(blocks):
            text = block.get_text() if hasattr(block, "get_text") else getattr(block, "text", "")
            out_blocks.append(
                {
                    "index": block_index,
                    "source_index": original_index.get(id(block), block_index),
                    "reading_order": resolved_reading_order,
                    "xyxy": _as_list(getattr(block, "xyxy", None)),
                    "lines": _as_list(getattr(block, "lines", None)),
                    "angle": float(getattr(block, "angle", 0.0) or 0.0),
                    "source_text": text or "",
                    "translation": getattr(block, "translation", "") or "",
                    "font": _font_dict(block),
                    "render_hints": _render_hints(block, getattr(block, "translation", "") or text or ""),
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
                "reading_order": resolved_reading_order,
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
