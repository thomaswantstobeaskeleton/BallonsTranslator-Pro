from __future__ import annotations

import html
import json
import os
import os.path as osp
from datetime import datetime, timezone
from typing import Dict, Optional, Sequence, Tuple

from .rendering_qa import analyze_text_block
from .text_rendering import resolve_writing_mode, vertical_columns


def _page_size(project, page_name: str) -> Tuple[int, int]:
    info = getattr(project, "_image_info", {}) or {}
    if isinstance(info, dict) and page_name in info:
        w = int((info.get(page_name) or {}).get("width", 0) or 0)
        h = int((info.get(page_name) or {}).get("height", 0) or 0)
        if w > 0 and h > 0:
            return w, h
    try:
        from .io_utils import imread
        img = imread(osp.join(getattr(project, "directory", "") or "", page_name))
        if img is not None:
            h, w = img.shape[:2]
            return int(w), int(h)
    except Exception:
        pass
    return 1, 1


def _rgb(value: Sequence[int] | None, default=(0, 0, 0)) -> str:
    vals = list(value or default)[:3]
    while len(vals) < 3:
        vals.append(0)
    return "#{:02x}{:02x}{:02x}".format(*(max(0, min(255, int(v))) for v in vals))


def _block_text(blk) -> str:
    return str(getattr(blk, "translation", "") or getattr(blk, "rich_text", "") or "\n".join(getattr(blk, "text", []) or []) or "")


def _text_svg_lines(text: str, mode: str, x1: float, y1: float, width: float, height: float, font_size: float, line_spacing: float) -> str:
    escaped_lines = []
    if mode == "vertical_rl":
        max_chars = max(1, int(max(1.0, height) / max(1.0, font_size)))
        cols = vertical_columns(text, max_chars)
        col_advance = font_size * max(0.1, line_spacing)
        for col_idx, col in enumerate(cols):
            x = x1 + width - (col_idx + 0.5) * col_advance
            y = y1 + font_size
            escaped_lines.append(f'<tspan x="{x:.2f}" y="{y:.2f}">{html.escape(col)}</tspan>')
    else:
        lines = (text or "").splitlines() or [text or ""]
        for idx, line in enumerate(lines):
            escaped_lines.append(f'<tspan x="{x1:.2f}" y="{(y1 + font_size + idx * font_size * max(0.1, line_spacing)):.2f}">{html.escape(line)}</tspan>')
    return "\n".join(escaped_lines)


def build_svg_text_handoff(project, page_name: str, out_dir: str, final_image_path: Optional[str] = None) -> Dict:
    """Write an SVG handoff with editable translated text and a JSON manifest.

    This is not a native PSD replacement; it is an interop handoff for vector
    editors and automation clients that need editable text placement.
    """
    os.makedirs(out_dir, exist_ok=True)
    page_w, page_h = _page_size(project, page_name)
    blocks = list((getattr(project, "pages", {}) or {}).get(page_name, []) or [])
    base_href = final_image_path or osp.join(getattr(project, "directory", "") or "", page_name)
    warnings = []
    layer_entries = []
    svg = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{page_w}" height="{page_h}" viewBox="0 0 {page_w} {page_h}">',
        '  <metadata><![CDATA[' + json.dumps({"format": "ballonstranslator.svg_text_handoff.v1", "page": page_name}, ensure_ascii=False) + ']]></metadata>',
    ]
    if base_href and osp.exists(base_href):
        svg.append(f'  <image id="base_image" href="{html.escape(osp.abspath(base_href))}" x="0" y="0" width="{page_w}" height="{page_h}" preserveAspectRatio="xMidYMid meet"/>')
    else:
        warnings.append("Base image path was not found; SVG contains text layers only.")
    svg.append('  <g id="translated_text_layers">')
    for idx, blk in enumerate(blocks):
        xyxy = list(getattr(blk, "xyxy", [0, 0, 0, 0]) or [0, 0, 0, 0])
        if len(xyxy) < 4:
            warnings.append(f"Block {idx} has invalid geometry and was skipped.")
            continue
        x1, y1, x2, y2 = [float(v or 0.0) for v in xyxy[:4]]
        width, height = max(1.0, x2 - x1), max(1.0, y2 - y1)
        fmt = getattr(blk, "fontformat", None)
        text = _block_text(blk)
        font_size = float(getattr(fmt, "font_size", 24.0) if fmt else 24.0 or 24.0)
        line_spacing = float(getattr(fmt, "line_spacing", 1.15) if fmt else 1.15 or 1.15)
        mode = resolve_writing_mode(getattr(fmt, "writing_mode", "auto") if fmt else "auto", text, (width, height))
        fill = _rgb(getattr(fmt, "frgb", None) if fmt else None)
        stroke = _rgb(getattr(fmt, "srgb", None) if fmt else None)
        stroke_width = max(0.0, font_size * float(getattr(fmt, "stroke_width", 0.0) if fmt else 0.0 or 0.0))
        anchor = "middle" if int(getattr(fmt, "alignment", 0) if fmt else 0 or 0) == 1 else ("end" if int(getattr(fmt, "alignment", 0) if fmt else 0 or 0) == 2 else "start")
        attrs = [
            f'id="text_{idx + 1:03d}"',
            f'font-family="{html.escape(str(getattr(fmt, "font_family", "") if fmt else ""))}"',
            f'font-size="{font_size:.2f}"',
            f'fill="{fill}"',
            f'stroke="{stroke}"',
            f'stroke-width="{stroke_width:.2f}"',
            f'text-anchor="{anchor}"',
            f'data-writing-mode="{mode}"',
        ]
        if mode == "vertical_rl":
            attrs.append('writing-mode="vertical-rl"')
            attrs.append('glyph-orientation-vertical="0"')
        if mode == "rtl":
            attrs.append('direction="rtl"')
        svg.append('    <text ' + " ".join(attrs) + '>')
        svg.append(_text_svg_lines(text, mode, x1, y1, width, height, font_size, line_spacing))
        svg.append('    </text>')
        try:
            diagnostics = analyze_text_block(blk, page_name, idx, config_obj=object())
        except Exception as exc:
            diagnostics = {"warnings": ["svg_diagnostics_unavailable"], "error": str(exc)}
        layer_entries.append({
            "index": idx,
            "text": text,
            "xyxy": [x1, y1, x2, y2],
            "writing_mode": mode,
            "font_family": getattr(fmt, "font_family", "") if fmt else "",
            "font_size": font_size,
            "diagnostics": diagnostics,
        })
        warnings.extend([f"Block {idx}: {w}" for w in diagnostics.get("warnings", [])])
    svg.append('  </g>')
    svg.append('</svg>')
    svg_path = osp.join(out_dir, osp.splitext(osp.basename(page_name))[0] + ".svg")
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))
    manifest = {
        "format": "ballonstranslator.svg_text_handoff.v1",
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "page": page_name,
        "svg_path": svg_path,
        "base_image": base_href,
        "text_layers": layer_entries,
        "warnings": warnings,
    }
    manifest_path = osp.join(out_dir, osp.splitext(osp.basename(page_name))[0] + ".svg_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    manifest["manifest_path"] = manifest_path
    return manifest
