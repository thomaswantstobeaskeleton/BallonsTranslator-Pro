from __future__ import annotations

import html
import json
import os
import os.path as osp
from datetime import datetime, timezone
from typing import Dict, Optional, Sequence, Tuple

from .rendering_qa import analyze_text_block
from .text_rendering import (
    resolve_writing_mode, lettering_proof_metrics,
    vertical_layout_plan, normalize_vertical_punctuation, font_fallback_runs, merge_font_fallback_chain,
)


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


def _font_run_entries(text: str, primary_family: str, config_obj=None, override_chain: str = "") -> list:
    entries = []
    for start, end, family in font_fallback_runs(text or "", primary_family or "", config_obj, override_chain):
        if end > start:
            entries.append({"start": start, "end": end, "family": family, "text": (text or "")[start:end]})
    return entries


def _all_font_runs(text: str, primary_family: str, config_obj=None, override_chain: str = "") -> list:
    text = text or ""
    fallback = _font_run_entries(text, primary_family, config_obj, override_chain)
    if not text:
        return []
    families = merge_font_fallback_chain(primary_family or "", text, config_obj, override_chain)
    primary = families[0] if families else (primary_family or "")
    breakpoints = {0, len(text)}
    for run in fallback:
        breakpoints.add(int(run["start"]))
        breakpoints.add(int(run["end"]))
    ordered = sorted(breakpoints)
    out = []
    for start, end in zip(ordered, ordered[1:]):
        if end <= start:
            continue
        family = primary
        for run in fallback:
            if int(run["start"]) <= start and end <= int(run["end"]):
                family = str(run.get("family") or primary)
                break
        out.append({"start": start, "end": end, "family": family, "text": text[start:end], "fallback": family != primary})
    return out


def _text_svg_lines(
    text: str,
    mode: str,
    x1: float,
    y1: float,
    width: float,
    height: float,
    font_size: float,
    line_spacing: float,
    letter_spacing: float = 1.0,
    line_break_strategy: str = "auto",
    primary_family: str = "",
    fallback_chain: str = "",
    config_obj=None,
) -> tuple[str, dict]:
    """Return editable SVG text tspans plus handoff metadata.

    Vertical CJK is emitted as positioned glyph tspans from the same renderer-neutral
    vertical layout plan used by QA.  That preserves top-to-bottom/right-to-left
    column order, punctuation offsets, rotation hints, tate-chu-yoko groups, and
    per-glyph fallback font choices instead of relying on one browser-dependent
    vertical text string.
    """
    escaped_lines = []
    text = text or ""
    primary_family = primary_family or ""
    fallback_entries = _font_run_entries(text, primary_family, config_obj, fallback_chain)
    meta = {"fallback_runs": fallback_entries, "font_runs": _all_font_runs(text, primary_family, config_obj, fallback_chain)}
    if mode == "vertical_rl":
        normalized = normalize_vertical_punctuation(text)
        max_chars = max(1, int(max(1.0, height) / max(1.0, font_size * max(0.1, letter_spacing))))
        plan = vertical_layout_plan(
            normalized,
            max_chars,
            font_size=font_size,
            line_spacing=line_spacing,
            letter_spacing=letter_spacing,
            strategy=line_break_strategy,
        )
        meta["vertical_layout_plan"] = plan
        glyphs = list(plan.get("glyphs", []) or [])
        groups = list(plan.get("tate_chu_yoko_groups", []) or [])
        group_by_start = {int(g.get("start", -1)): (i, g) for i, g in enumerate(groups)}
        group_ranges = []
        for i, g in enumerate(groups):
            group_ranges.append((i, int(g.get("start", -1)), int(g.get("end", -1))))
        col_adv = float(plan.get("column_advance", font_size * max(0.1, line_spacing)) or font_size)
        for glyph in glyphs:
            logical_index = int(glyph.get("index", -1))
            # Render compact tate-chu-yoko groups once at their first glyph.
            in_consumed_group = False
            for group_idx, start, end in group_ranges:
                if start < logical_index < end:
                    in_consumed_group = True
                    break
            if in_consumed_group:
                continue
            group_info = group_by_start.get(logical_index)
            ch = str(glyph.get("char", "") or "")
            text_to_draw = ch
            tcy_attr = ""
            if group_info is not None:
                group_idx, group = group_info
                text_to_draw = str(group.get("text", "") or ch)
                tcy_attr = ' data-tate-chu-yoko="true" textLength="{:.2f}" lengthAdjust="spacingAndGlyphs"'.format(font_size * 0.92)
            dx = float((glyph.get("offset", {}) or {}).get("dx", 0.0) or 0.0)
            dy = float((glyph.get("offset", {}) or {}).get("dy", 0.0) or 0.0)
            gx = x1 + width - (0.5 * col_adv) + float(glyph.get("x", 0.0) or 0.0) + dx
            gy = y1 + font_size + float(glyph.get("y", 0.0) or 0.0) + dy
            family = primary_family
            for run in meta.get("font_runs", []) or []:
                if int(run.get("start", -1)) <= logical_index < int(run.get("end", -1)):
                    family = str(run.get("family", "") or primary_family)
                    break
            rotate = float(glyph.get("rotate_degrees", 0.0) or (90.0 if glyph.get("rotate") else 0.0) or 0.0)
            transform = f' transform="rotate({rotate:.1f} {gx:.2f} {gy:.2f})"' if rotate else ""
            escaped_lines.append(
                f'<tspan x="{gx:.2f}" y="{gy:.2f}" font-family="{html.escape(str(family or primary_family))}" '
                f'data-column="{int(glyph.get("column", 0))}" data-row="{int(glyph.get("row", 0))}" '
                f'data-punctuation="{html.escape(str(glyph.get("punctuation_class", "")))}"{tcy_attr}{transform}>{html.escape(text_to_draw)}</tspan>'
            )
    else:
        cursor_index = 0
        lines = text.splitlines() or [text]
        runs = meta.get("font_runs") or []
        for idx, line in enumerate(lines):
            line_start = cursor_index
            line_end = line_start + len(line)
            y = y1 + font_size + idx * font_size * max(0.1, line_spacing)
            line_runs = [r for r in runs if int(r.get("end", 0)) > line_start and int(r.get("start", 0)) < line_end]
            if not line_runs:
                escaped_lines.append(f'<tspan x="{x1:.2f}" y="{y:.2f}">{html.escape(line)}</tspan>')
            else:
                escaped_lines.append(f'<tspan x="{x1:.2f}" y="{y:.2f}">')
                for run in line_runs:
                    start = max(line_start, int(run.get("start", 0)))
                    end = min(line_end, int(run.get("end", 0)))
                    part = text[start:end]
                    escaped_lines.append(f'<tspan font-family="{html.escape(str(run.get("family", primary_family) or primary_family))}">{html.escape(part)}</tspan>')
                escaped_lines.append('</tspan>')
            cursor_index = line_end + 1
    return "\n".join(escaped_lines), meta


def build_svg_text_handoff(project, page_name: str, out_dir: str, final_image_path: Optional[str] = None, config_obj=None) -> Dict:
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
        secondary_stroke_width = max(0.0, font_size * float(getattr(fmt, "secondary_stroke_width", 0.0) if fmt else 0.0 or 0.0))
        secondary_stroke = _rgb(getattr(fmt, "secondary_srgb", [255, 255, 255]) if fmt else [255, 255, 255])
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
            attrs.append('data-vertical-layout="top-to-bottom-right-to-left"')
        if mode == "rtl":
            attrs.append('direction="rtl"')
        if secondary_stroke_width > 0:
            back_attrs = list(attrs)
            back_attrs = [a for a in back_attrs if not a.startswith('id=') and not a.startswith('stroke=') and not a.startswith('stroke-width=') and not a.startswith('fill=')]
            back_attrs.insert(0, f'id="text_{idx + 1:03d}_back_outline"')
            back_attrs.extend([f'fill="none"', f'stroke="{secondary_stroke}"', f'stroke-width="{secondary_stroke_width:.2f}"'])
            svg.append('    <text ' + " ".join(back_attrs) + '>')
            back_markup, _back_meta = _text_svg_lines(text, mode, x1, y1, width, height, font_size, line_spacing, float(getattr(fmt, "letter_spacing", 1.0) if fmt else 1.0 or 1.0), getattr(fmt, "line_break_strategy", "auto") if fmt else "auto", getattr(fmt, "font_family", "") if fmt else "", getattr(fmt, "fallback_font_chain", "") if fmt else "", config_obj)
            svg.append(back_markup)
            svg.append('    </text>')
        svg.append('    <text ' + " ".join(attrs) + '>')
        text_markup, svg_text_meta = _text_svg_lines(text, mode, x1, y1, width, height, font_size, line_spacing, float(getattr(fmt, "letter_spacing", 1.0) if fmt else 1.0 or 1.0), getattr(fmt, "line_break_strategy", "auto") if fmt else "auto", getattr(fmt, "font_family", "") if fmt else "", getattr(fmt, "fallback_font_chain", "") if fmt else "", config_obj)
        svg.append(text_markup)
        svg.append('    </text>')
        try:
            diagnostics = analyze_text_block(blk, page_name, idx, config_obj=object())
        except Exception as exc:
            diagnostics = {"warnings": ["svg_diagnostics_unavailable"], "error": str(exc)}
        proof_metrics = lettering_proof_metrics(
            text, font_size, (width, height), mode, line_spacing=line_spacing,
            letter_spacing=float(getattr(fmt, "letter_spacing", 1.0) if fmt else 1.0 or 1.0),
            padding=float(getattr(fmt, "text_padding", 0.0) if fmt else 0.0 or 0.0),
            stroke_width=float(getattr(fmt, "stroke_width", 0.0) if fmt else 0.0 or 0.0),
            secondary_stroke_width=float(getattr(fmt, "secondary_stroke_width", 0.0) if fmt else 0.0 or 0.0),
            shadow_radius=float(getattr(fmt, "shadow_radius", 0.0) if fmt else 0.0 or 0.0),
            shadow_offset=getattr(fmt, "shadow_offset", [0.0, 0.0]) if fmt else [0.0, 0.0],
            line_break_strategy=getattr(fmt, "line_break_strategy", "auto") if fmt else "auto",
            sample_limit=64,
        )
        layer_entries.append({
            "index": idx,
            "text": text,
            "xyxy": [x1, y1, x2, y2],
            "writing_mode": mode,
            "font_family": getattr(fmt, "font_family", "") if fmt else "",
            "font_size": font_size,
            "secondary_stroke_width": secondary_stroke_width,
            "diagnostics": diagnostics,
            "proof_metrics": proof_metrics,
            "fallback_runs": svg_text_meta.get("fallback_runs", []),
            "font_runs": svg_text_meta.get("font_runs", []),
            "vertical_layout_plan": svg_text_meta.get("vertical_layout_plan", {}),
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
