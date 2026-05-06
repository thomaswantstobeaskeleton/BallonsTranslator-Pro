from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .fontformat import FontFormat
from .text_rendering import (
    FIT_MODE_BALANCE,
    FIT_MODE_EXPAND,
    FIT_MODE_PRESERVE,
    FIT_MODE_SHRINK,
    LINE_BREAK_CJK_STRICT,
    estimate_text_bounds,
    fallback_chain_for_text,
    fit_font_size_to_box,
    merge_font_fallback_chain,
    missing_glyphs_after_fallback,
    resolve_writing_mode,
)


def _block_text(blk) -> str:
    return str(getattr(blk, "translation", "") or getattr(blk, "rich_text", "") or "\n".join(getattr(blk, "text", []) or []) or "")


def _box_size(blk) -> Tuple[float, float]:
    xyxy = getattr(blk, "xyxy", [0, 0, 0, 0]) or [0, 0, 0, 0]
    if len(xyxy) < 4:
        return 1.0, 1.0
    return max(1.0, float(xyxy[2]) - float(xyxy[0])), max(1.0, float(xyxy[3]) - float(xyxy[1]))


def analyze_text_block(blk, page: str, index: int, config_obj=None) -> Dict:
    """Return renderer QA diagnostics for a project TextBlock without needing scene items."""
    if config_obj is None:
        from .config import pcfg as config_obj
    fmt: FontFormat = getattr(blk, "fontformat", None) or FontFormat()
    text = _block_text(blk)
    box_w, box_h = _box_size(blk)
    mode = resolve_writing_mode(getattr(fmt, "writing_mode", "auto"), text, (box_w, box_h))
    measured = estimate_text_bounds(
        text,
        float(getattr(fmt, "font_size", 24.0) or 24.0),
        mode,
        box_w,
        box_h,
        float(getattr(fmt, "line_spacing", 1.15) or 1.15),
        float(getattr(fmt, "letter_spacing", 1.0) or 1.0),
        float(getattr(fmt, "text_padding", 0.0) or 0.0),
        float(getattr(fmt, "stroke_width", 0.0) or 0.0),
        float(getattr(fmt, "shadow_radius", 0.0) or 0.0),
        getattr(fmt, "shadow_offset", [0.0, 0.0]) or [0.0, 0.0],
        line_break_strategy=getattr(fmt, "line_break_strategy", "auto"),
    )
    missing = missing_glyphs_after_fallback(
        getattr(fmt, "font_family", ""), text, config_obj, getattr(fmt, "fallback_font_chain", "")
    ) if text else []
    overflow = measured[0] > box_w or measured[1] > box_h
    warnings: List[str] = []
    suggestions: List[Dict] = []
    if overflow:
        warnings.append("overflow")
        suggestions.append({"action": "shrink_to_fit", "reason": "Measured text bounds exceed textbox bounds."})
    if missing:
        warnings.append("missing_glyphs")
        chain = fallback_chain_for_text(text, config_obj)
        suggestions.append({"action": "set_fallback_chain", "fallback_chain": chain, "reason": "Configured fonts still miss glyphs."})
    if mode == "vertical_rl" and getattr(fmt, "line_break_strategy", "auto") not in (LINE_BREAK_CJK_STRICT, "balanced"):
        warnings.append("weak_vertical_line_break_strategy")
        suggestions.append({"action": "set_line_break_strategy", "line_break_strategy": LINE_BREAK_CJK_STRICT, "reason": "Vertical CJK should use strict kinsoku wrapping."})
    if mode == "rtl" and getattr(fmt, "alignment", 0) == 0:
        warnings.append("rtl_left_alignment")
        suggestions.append({"action": "set_alignment", "alignment": 2, "reason": "RTL text is usually easier to edit/export right-aligned."})
    if float(getattr(fmt, "stroke_width", 0.0) or 0.0) > 0 and float(getattr(fmt, "text_padding", 0.0) or 0.0) < 1.0:
        warnings.append("low_padding_with_stroke")
        suggestions.append({"action": "increase_padding", "padding": 2.0, "reason": "Outlined text can clip without inset padding."})

    return {
        "page": page,
        "index": index,
        "text_length": len(text),
        "box": [box_w, box_h],
        "measured": [measured[0], measured[1]],
        "line_count": measured[2],
        "column_count": measured[3],
        "writing_mode": getattr(fmt, "writing_mode", "auto"),
        "resolved_writing_mode": mode,
        "fit_mode": getattr(fmt, "fit_mode", FIT_MODE_SHRINK),
        "line_break_strategy": getattr(fmt, "line_break_strategy", "auto"),
        "font_family": getattr(fmt, "font_family", ""),
        "fallback_chain": merge_font_fallback_chain(getattr(fmt, "font_family", ""), text, config_obj, getattr(fmt, "fallback_font_chain", "")),
        "missing_glyphs": missing,
        "overflow": overflow,
        "warnings": warnings,
        "suggestions": suggestions,
    }


def build_project_rendering_qa(project, pages: Optional[Sequence[str]] = None, include_ok: bool = False, config_obj=None) -> Dict:
    """Build a page/textbox renderer QA report for automation and UI export."""
    if project is None:
        return {"pages": [], "summary": {"pages": 0, "textboxes": 0, "issues": 0}}
    selected_pages = list(pages or getattr(project, "pages", {}).keys())
    page_entries = []
    total_boxes = 0
    total_issues = 0
    issue_counts: Dict[str, int] = {}
    for page in selected_pages:
        blocks = list((getattr(project, "pages", {}) or {}).get(page, []) or [])
        block_entries = []
        page_issue_count = 0
        for idx, blk in enumerate(blocks):
            total_boxes += 1
            diag = analyze_text_block(blk, page, idx, config_obj=config_obj)
            if diag["warnings"]:
                total_issues += 1
                page_issue_count += 1
                for warning in diag["warnings"]:
                    issue_counts[warning] = issue_counts.get(warning, 0) + 1
            if include_ok or diag["warnings"]:
                block_entries.append(diag)
        page_entries.append({"page": page, "textboxes": len(blocks), "issues": page_issue_count, "blocks": block_entries})
    return {
        "summary": {
            "pages": len(selected_pages),
            "textboxes": total_boxes,
            "issues": total_issues,
            "issue_counts": issue_counts,
        },
        "pages": page_entries,
    }



def flatten_rendering_qa_rows(report: Dict) -> List[Dict]:
    """Flatten a QA report into table/export rows."""
    rows: List[Dict] = []
    for page in report.get("pages", []) or []:
        page_name = page.get("page", "")
        for block in page.get("blocks", []) or []:
            warnings = list(block.get("warnings", []) or [])
            severity = "ok"
            if "overflow" in warnings or "missing_glyphs" in warnings:
                severity = "warning"
            if len(warnings) >= 3:
                severity = "error"
            rows.append({
                "page": page_name,
                "index": block.get("index", -1),
                "severity": severity,
                "warnings": warnings,
                "suggestions": [s.get("action", "") for s in block.get("suggestions", []) or []],
                "writing_mode": block.get("resolved_writing_mode", block.get("writing_mode", "")),
                "fit_mode": block.get("fit_mode", ""),
                "line_break_strategy": block.get("line_break_strategy", ""),
                "font_family": block.get("font_family", ""),
                "missing_glyphs": "".join(block.get("missing_glyphs", []) or []),
                "box": block.get("box", []),
                "measured": block.get("measured", []),
                "text_length": block.get("text_length", 0),
            })
    return rows


def rendering_qa_to_markdown(report: Dict) -> str:
    """Convert a QA report to a compact Markdown summary for reviews/PR handoff."""
    summary = report.get("summary", {}) or {}
    lines = [
        "# Typography QA Report",
        "",
        f"- Pages: {summary.get('pages', 0)}",
        f"- Text boxes: {summary.get('textboxes', 0)}",
        f"- Issue text boxes: {summary.get('issues', 0)}",
    ]
    counts = summary.get("issue_counts", {}) or {}
    if counts:
        lines.append("- Issue counts: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    rows = flatten_rendering_qa_rows(report)
    if rows:
        lines += [
            "",
            "| Page | # | Severity | Warnings | Suggestions | Mode | Fit | Break | Missing |",
            "| --- | ---: | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for row in rows:
            warnings = ", ".join(row.get("warnings", []))
            suggestions = ", ".join(row.get("suggestions", []))
            lines.append(
                f"| {row.get('page', '')} | {row.get('index', '')} | {row.get('severity', '')} | "
                f"{warnings} | {suggestions} | {row.get('writing_mode', '')} | "
                f"{row.get('fit_mode', '')} | {row.get('line_break_strategy', '')} | {row.get('missing_glyphs', '')} |"
            )
    return "\n".join(lines) + "\n"

def apply_project_rendering_fixes(project, pages: Optional[Sequence[str]] = None, config_obj=None) -> Dict:
    """Apply conservative typography fixes directly to project TextBlocks."""
    if config_obj is None:
        from .config import pcfg as config_obj
    report = build_project_rendering_qa(project, pages=pages, include_ok=False, config_obj=config_obj)
    applied: List[Dict] = []
    page_set = set(pages or (getattr(project, "pages", {}) or {}).keys())
    for page in page_set:
        blocks = list((getattr(project, "pages", {}) or {}).get(page, []) or [])
        for idx, blk in enumerate(blocks):
            diag = analyze_text_block(blk, page, idx, config_obj=config_obj)
            if not diag["warnings"]:
                continue
            fmt: FontFormat = getattr(blk, "fontformat", None) or FontFormat()
            text = _block_text(blk)
            box_w, box_h = _box_size(blk)
            changed: List[str] = []
            if "weak_vertical_line_break_strategy" in diag["warnings"]:
                fmt.line_break_strategy = LINE_BREAK_CJK_STRICT
                changed.append("line_break_strategy")
            if "rtl_left_alignment" in diag["warnings"]:
                fmt.alignment = 2
                changed.append("alignment")
            if "low_padding_with_stroke" in diag["warnings"]:
                fmt.text_padding = max(float(getattr(fmt, "text_padding", 0.0) or 0.0), 2.0)
                changed.append("text_padding")
            if "missing_glyphs" in diag["warnings"] and not getattr(fmt, "fallback_font_chain", ""):
                chain = fallback_chain_for_text(text, config_obj)
                if chain:
                    fmt.fallback_font_chain = chain
                    changed.append("fallback_font_chain")
            if "overflow" in diag["warnings"]:
                fit_mode = getattr(fmt, "fit_mode", FIT_MODE_SHRINK)
                if fit_mode in (FIT_MODE_SHRINK, FIT_MODE_EXPAND, FIT_MODE_PRESERVE, FIT_MODE_BALANCE):
                    new_size, new_text, fit_diag = fit_font_size_to_box(
                        text,
                        float(getattr(fmt, "font_size", 24.0) or 24.0),
                        (box_w, box_h),
                        FIT_MODE_SHRINK,
                        getattr(fmt, "writing_mode", "auto"),
                        padding=float(getattr(fmt, "text_padding", 0.0) or 0.0),
                        stroke_width=float(getattr(fmt, "stroke_width", 0.0) or 0.0),
                        line_spacing=float(getattr(fmt, "line_spacing", 1.15) or 1.15),
                        letter_spacing=float(getattr(fmt, "letter_spacing", 1.0) or 1.0),
                        line_break_strategy=getattr(fmt, "line_break_strategy", "auto"),
                    )
                    if new_size < float(getattr(fmt, "font_size", 24.0) or 24.0) - 0.2:
                        fmt.font_size = new_size
                        fmt.fit_mode = FIT_MODE_SHRINK
                        changed.append("font_size")
                    if new_text and new_text != text and "\n" in new_text:
                        blk.translation = new_text
                        changed.append("balanced_text")
            if changed:
                blk.fontformat = fmt
                applied.append({"page": page, "index": idx, "changed": changed})
    after = build_project_rendering_qa(project, pages=pages, include_ok=False, config_obj=config_obj)
    return {"applied": applied, "applied_count": len(applied), "before": report["summary"], "after": after["summary"]}
