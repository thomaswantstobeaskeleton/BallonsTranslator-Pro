from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple

from .fontformat import FontFormat
from .text_masking import masked_text_warnings, mask_effective_box
from .textbox_masking import centered_resize_xyxy, mask_aware_textbox_diagnostics
from .text_rendering import (
    FIT_MODE_BALANCE,
    FIT_MODE_EXPAND,
    FIT_MODE_PRESERVE,
    FIT_MODE_SHRINK,
    LINE_BREAK_CJK_STRICT,
    estimate_text_bounds,
    fallback_chain_for_text,
    lettering_quality_score,
    safe_inner_bounds,
    fit_font_size_to_box,
    smart_fit_text_to_box,
    plan_typography_cleanup,
    lettering_proof_metrics,
    line_break_opportunities,
    normalize_vertical_punctuation,
    optimal_kinsoku_wrap,
    resolve_fit_font_size_bounds,
    suggest_manga_effects_for_background,
    suggested_back_outline_rgb,
    vertical_layout_plan,
    merge_font_fallback_chain,
    missing_glyphs_after_fallback,
    resolve_writing_mode,
    recommended_tight_letter_spacing,
)


def _block_text(blk) -> str:
    return str(
        getattr(blk, "translation", "")
        or getattr(blk, "rich_text", "")
        or "\n".join(getattr(blk, "text", []) or [])
        or ""
    )


def _box_size(blk) -> Tuple[float, float]:
    xyxy = getattr(blk, "xyxy", [0, 0, 0, 0]) or [0, 0, 0, 0]
    if len(xyxy) < 4:
        return 1.0, 1.0
    return (
        max(1.0, float(xyxy[2]) - float(xyxy[0])),
        max(1.0, float(xyxy[3]) - float(xyxy[1])),
    )


def _merge_mask_diagnostics(mask_warning_diag: Dict, textbox_mask_diag: Dict) -> Dict:
    """Keep both legacy mask-safe warnings and newer textbox-mask diagnostics."""
    merged = dict(textbox_mask_diag or {})

    # PR-side masked_text_warnings fields.
    for key in (
        "coverage",
        "warning",
        "recommended_padding",
        "safe_insets",
        "safe_rect",
        "fully_masked",
        "narrow_safe_area",
    ):
        if key in (mask_warning_diag or {}):
            merged[key] = mask_warning_diag.get(key)

    # Main-side diagnostics commonly expose visible_area_ratio.
    if "visible_area_ratio" not in merged:
        if "coverage" in merged:
            merged["visible_area_ratio"] = merged.get("coverage")
        else:
            merged["visible_area_ratio"] = 1.0

    return merged


def analyze_text_block(blk, page: str, index: int, config_obj=None) -> Dict:
    """Return renderer QA diagnostics for a project TextBlock without needing scene items."""
    if config_obj is None:
        from .config import pcfg as config_obj

    fmt: FontFormat = getattr(blk, "fontformat", None) or FontFormat()
    text = _block_text(blk)
    box_w, box_h = _box_size(blk)

    mode = resolve_writing_mode(
        getattr(fmt, "writing_mode", "auto"),
        text,
        (box_w, box_h),
    )
    measured = estimate_text_bounds(
        text,
        float(getattr(fmt, "font_size", 24.0) or 24.0),
        mode,
        box_w,
        box_h,
        float(getattr(fmt, "line_spacing", 1.15) or 1.15),
        float(getattr(fmt, "letter_spacing", 1.0) or 1.0),
        padding=float(getattr(fmt, "text_padding", 0.0) or 0.0),
        stroke_width=float(getattr(fmt, "stroke_width", 0.0) or 0.0),
        secondary_stroke_width=float(getattr(fmt, "secondary_stroke_width", 0.0) or 0.0),
        shadow_radius=float(getattr(fmt, "shadow_radius", 0.0) or 0.0),
        shadow_offset=getattr(fmt, "shadow_offset", [0.0, 0.0]) or [0.0, 0.0],
        line_break_strategy=getattr(fmt, "line_break_strategy", "auto"),
    )

    missing = (
        missing_glyphs_after_fallback(
            getattr(fmt, "font_family", ""),
            text,
            config_obj,
            getattr(fmt, "fallback_font_chain", ""),
        )
        if text
        else []
    )

    style_min, style_max = resolve_fit_font_size_bounds(
        getattr(fmt, "fit_font_size_min", 0.0),
        getattr(fmt, "fit_font_size_max", 0.0),
        getattr(getattr(config_obj, "module", None), "layout_font_size_min", 6.0),
        getattr(getattr(config_obj, "module", None), "layout_font_size_max", 96.0),
    )

    fitted_size, fitted_text, fit_diag = fit_font_size_to_box(
        text,
        float(getattr(fmt, "font_size", 24.0) or 24.0),
        (box_w, box_h),
        getattr(fmt, "fit_mode", FIT_MODE_SHRINK),
        getattr(fmt, "writing_mode", "auto"),
        min_font_size=style_min,
        max_font_size=style_max,
        line_spacing=float(getattr(fmt, "line_spacing", 1.15) or 1.15),
        letter_spacing=float(getattr(fmt, "letter_spacing", 1.0) or 1.0),
        padding=float(getattr(fmt, "text_padding", 0.0) or 0.0),
        stroke_width=float(getattr(fmt, "stroke_width", 0.0) or 0.0),
        secondary_stroke_width=float(getattr(fmt, "secondary_stroke_width", 0.0) or 0.0),
        line_break_strategy=getattr(fmt, "line_break_strategy", "auto"),
        shadow_radius=float(getattr(fmt, "shadow_radius", 0.0) or 0.0),
        shadow_offset=getattr(fmt, "shadow_offset", [0.0, 0.0]) or [0.0, 0.0],
    )

    overflow = measured[0] > box_w or measured[1] > box_h
    inner_bounds, effect_margin = safe_inner_bounds(
        (box_w, box_h),
        float(getattr(fmt, "font_size", 24.0) or 24.0),
        stroke_width=float(getattr(fmt, "stroke_width", 0.0) or 0.0),
        secondary_stroke_width=float(getattr(fmt, "secondary_stroke_width", 0.0) or 0.0),
        shadow_radius=float(getattr(fmt, "shadow_radius", 0.0) or 0.0),
        shadow_offset=getattr(fmt, "shadow_offset", [0.0, 0.0]) or [0.0, 0.0],
        padding=float(getattr(fmt, "text_padding", 0.0) or 0.0),
    )

    explicit_horizontal_on_tall_cjk = (
        getattr(fmt, "writing_mode", "auto") == "horizontal_ltr"
        and resolve_writing_mode("auto", text, (box_w, box_h)) == "vertical_rl"
    )

    warnings: List[str] = []
    suggestions: List[Dict] = []

    effect_suggestion = suggest_manga_effects_for_background(
        getattr(fmt, "frgb", [0, 0, 0]) or [0, 0, 0],
        [255, 255, 255],
        float(getattr(fmt, "stroke_width", 0.0) or 0.0),
    )

    if overflow:
        warnings.append("overflow")
        suggestions.append(
            {
                "action": "shrink_to_fit",
                "reason": "Measured text bounds exceed textbox bounds.",
            }
        )
        if measured[0] > box_w and float(getattr(fmt, "letter_spacing", 1.0) or 1.0) > 0.92:
            suggestions.append(
                {
                    "action": "tighten_letter_spacing",
                    "letter_spacing": recommended_tight_letter_spacing(
                        float(getattr(fmt, "letter_spacing", 1.0) or 1.0),
                        measured[0] / max(1.0, box_w),
                    ),
                    "reason": "Wide lettering can often fit by tightening tracking before shrinking the font.",
                }
            )

    if missing:
        warnings.append("missing_glyphs")
        chain = fallback_chain_for_text(text, config_obj)
        suggestions.append(
            {
                "action": "set_fallback_chain",
                "fallback_chain": chain,
                "reason": "Configured fonts still miss glyphs.",
            }
        )

    if mode == "vertical_rl" and getattr(fmt, "line_break_strategy", "auto") not in (
        LINE_BREAK_CJK_STRICT,
        "balanced",
    ):
        warnings.append("weak_vertical_line_break_strategy")
        suggestions.append(
            {
                "action": "set_line_break_strategy",
                "line_break_strategy": LINE_BREAK_CJK_STRICT,
                "reason": "Vertical CJK should use strict kinsoku wrapping.",
            }
        )

    if explicit_horizontal_on_tall_cjk:
        warnings.append("horizontal_cjk_in_tall_box")
        suggestions.append(
            {
                "action": "switch_writing_mode",
                "writing_mode": "vertical_rl",
                "reason": "CJK text in a tall manga bubble usually letters better as vertical RL.",
            }
        )

    if mode == "vertical_rl" and text != normalize_vertical_punctuation(text):
        warnings.append("vertical_punctuation_needs_normalization")
        suggestions.append(
            {
                "action": "normalize_vertical_punctuation",
                "reason": "Repeated ?!/!!/?? marks can use compact vertical punctuation forms.",
            }
        )

    if mode == "rtl" and getattr(fmt, "alignment", 0) == 0:
        warnings.append("rtl_left_alignment")
        suggestions.append(
            {
                "action": "set_alignment",
                "alignment": 2,
                "reason": "RTL text is usually easier to edit/export right-aligned.",
            }
        )

    if float(getattr(fmt, "stroke_width", 0.0) or 0.0) > 0 and float(
        getattr(fmt, "text_padding", 0.0) or 0.0
    ) < 1.0:
        warnings.append("low_padding_with_stroke")
        suggestions.append(
            {
                "action": "increase_padding",
                "padding": 2.0,
                "reason": "Outlined text can clip without inset padding.",
            }
        )

    text_mask = getattr(blk, "text_mask", None)
    mask_warning_diag = masked_text_warnings(
        text_mask,
        float(getattr(fmt, "text_padding", 0.0) or 0.0),
    )
    textbox_mask_diag = mask_aware_textbox_diagnostics(
        (box_w, box_h),
        (measured[0], measured[1]),
        text_mask=text_mask,
        effect_margin=effect_margin,
        padding=float(getattr(fmt, "text_padding", 0.0) or 0.0),
    )
    mask_diag = _merge_mask_diagnostics(mask_warning_diag, textbox_mask_diag)
    effective_box = mask_effective_box(
        text_mask,
        (box_w, box_h),
        float(getattr(fmt, "text_padding", 0.0) or 0.0),
    )
    mask_overflow = bool(effective_box.get("uses_mask")) and (
        measured[0] > float(effective_box.get("width", box_w))
        or measured[1] > float(effective_box.get("height", box_h))
    )

    if mask_warning_diag.get("fully_masked") or mask_warning_diag.get("narrow_safe_area"):
        warnings.append("mask_safe_area")
        suggestions.append(
            {
                "action": "increase_padding",
                "padding": mask_warning_diag.get("recommended_padding", 2.0),
                "mask_safe_insets": mask_warning_diag.get("safe_insets", []),
                "reason": "Text mask leaves a constrained safe area; add inset before render/export to avoid clipped lettering.",
            }
        )
        suggestions.append(
            {
                "action": "recenter",
                "reason": "Recenter ink in the mask-safe area.",
            }
        )

    if mask_overflow:
        warnings.append("mask_safe_overflow")
        suggestions.append(
            {
                "action": "shrink_to_mask_safe_area",
                "effective_box": [
                    round(float(effective_box.get("width", box_w)), 2),
                    round(float(effective_box.get("height", box_h)), 2),
                ],
                "reason": "Measured lettering exceeds the visible mask-safe area even if it fits the full text box.",
            }
        )

    if textbox_mask_diag.get("mask_overflow") and "mask_visible_area_overflow" not in warnings:
        warnings.append("mask_visible_area_overflow")
        suggestions.append(
            {
                "action": "resize_to_mask_safe_box",
                "recommended_box_size": textbox_mask_diag.get(
                    "recommended_box_size",
                    [box_w, box_h],
                ),
                "mask_visible_rect": (textbox_mask_diag.get("mask", {}) or {}).get(
                    "visible_rect",
                    [],
                ),
                "mask_overflow_axes": textbox_mask_diag.get("mask_overflow_axes", []),
                "reason": "Resize the textbox so measured lettering fits inside the visible text-mask area with effect margins.",
            }
        )

    mask_info = textbox_mask_diag.get("mask", {}) or {}
    if mask_info.get("has_mask"):
        hidden_ratio = float(mask_info.get("hidden_ratio", 0.0) or 0.0)
        if hidden_ratio > 0.25:
            warnings.append("text_mask_restricts_visible_area")
            suggestions.append(
                {
                    "action": "review_text_mask",
                    "hidden_ratio": hidden_ratio,
                    "reason": "The text eraser mask hides a large part of the textbox; check whether lettering is being clipped.",
                }
            )
        if mask_info.get("edge_hidden"):
            warnings.append("text_mask_erases_edge")
            suggestions.append(
                {
                    "action": "increase_padding",
                    "padding": max(
                        2.0,
                        float(getattr(fmt, "text_padding", 0.0) or 0.0) + 1.0,
                    ),
                    "reason": "The text mask erases the textbox edge; add inset so stroke/shadow does not hit the erased boundary.",
                }
            )

    if effect_suggestion.get("needs_effect") and float(
        getattr(fmt, "stroke_width", 0.0) or 0.0
    ) <= 0:
        warnings.append("low_contrast_no_effect")
        suggestions.append(
            {
                "action": "apply_contrast_stroke",
                **effect_suggestion,
                "reason": "Light lettering on a light bubble/page may need an outline or soft shadow.",
            }
        )

    preset_suggestion = getattr(fit_diag, "preset_suggestion", "")
    if (
        preset_suggestion == "sfx_bold"
        and float(getattr(fmt, "secondary_stroke_width", 0.0) or 0.0) <= 0
    ):
        back_rgb = suggested_back_outline_rgb(
            getattr(fmt, "frgb", [0, 0, 0]) or [0, 0, 0],
            getattr(fmt, "srgb", [0, 0, 0]) or [0, 0, 0],
        )
        warnings.append("sfx_missing_back_outline")
        suggestions.append(
            {
                "action": "apply_double_outline",
                "secondary_stroke_width": 0.20,
                "secondary_srgb": back_rgb,
                "reason": "SFX-style lettering is more readable with a wider back outline.",
            }
        )

    recommended_box = list(getattr(fit_diag, "recommended_box_size", (box_w, box_h)))
    if overflow and (
        recommended_box[0] > box_w * 1.02 or recommended_box[1] > box_h * 1.02
    ):
        suggestions.append(
            {
                "action": "resize_to_recommended_box",
                "recommended_box_size": recommended_box,
                "box_scale_hint": getattr(fit_diag, "box_scale_hint", 1.0),
                "reason": "Effect-aware fit diagnostics recommend a larger text box.",
            }
        )

    if getattr(fit_diag, "ink_clip_risk", False):
        warnings.append("ink_clip_risk")
        suggestions.append(
            {
                "action": "increase_padding",
                "padding": max(
                    2.0,
                    float(getattr(fmt, "text_padding", 0.0) or 0.0) + 1.0,
                ),
                "reason": "Estimated ink bounds leave too little room for stroke/shadow at the edge.",
            }
        )

    if (inner_bounds[0] < box_w * 0.62 or inner_bounds[1] < box_h * 0.62) and effect_margin > 0:
        warnings.append("unsafe_effect_bounds")
        suggestions.append(
            {
                "action": "increase_box_or_reduce_effect",
                "effect_margin": effect_margin,
                "safe_inner_bounds": list(inner_bounds),
                "recommended_box_size": recommended_box,
                "reason": "Stroke/shadow/padding leave little safe text area inside this box.",
            }
        )

    preset_suggestion = getattr(fit_diag, "preset_suggestion", "") or ""
    if preset_suggestion and preset_suggestion != getattr(fmt, "preset", ""):
        if preset_suggestion in {"vertical_cjk_bubble", "sfx_bold", "caption_box"}:
            suggestions.append(
                {
                    "action": "apply_manga_preset",
                    "preset": preset_suggestion,
                    "reason": "Script/geometry heuristics suggest this manga lettering preset.",
                }
            )

    quality_score = lettering_quality_score(
        overflow,
        missing,
        float(effect_suggestion.get("contrast_ratio", 7.0) or 7.0),
        effect_margin,
        (box_w, box_h),
    )

    vertical_plan = None
    break_ops = []
    if mode == "vertical_rl":
        max_chars = max(
            1,
            int(max(1.0, box_h) / max(1.0, float(getattr(fmt, "font_size", 24.0) or 24.0))),
        )
        vertical_plan = vertical_layout_plan(
            text,
            max_chars,
            getattr(fmt, "font_size", 24.0),
            getattr(fmt, "line_spacing", 1.15),
            getattr(fmt, "letter_spacing", 1.0),
            getattr(fmt, "line_break_strategy", "auto"),
            rotate_latin=bool(getattr(config_obj, "vertical_cjk_rotate_latin", True)),
            punctuation_hang=bool(
                getattr(config_obj, "vertical_cjk_punctuation_hang", True)
            ),
        )
    else:
        break_ops = line_break_opportunities(
            text,
            getattr(fmt, "line_break_strategy", "auto"),
        )[:24]

    typography_cleanup = plan_typography_cleanup(
        text,
        float(getattr(fmt, "font_size", 24.0) or 24.0),
        (box_w, box_h),
        getattr(fmt, "writing_mode", "auto"),
        getattr(fmt, "fit_mode", FIT_MODE_SHRINK),
        getattr(fmt, "line_break_strategy", "auto"),
        line_spacing=float(getattr(fmt, "line_spacing", 1.15) or 1.15),
        letter_spacing=float(getattr(fmt, "letter_spacing", 1.0) or 1.0),
        text_padding=float(getattr(fmt, "text_padding", 0.0) or 0.0),
        font_family=getattr(fmt, "font_family", ""),
        fallback_font_chain=getattr(fmt, "fallback_font_chain", ""),
        config_obj=config_obj,
    )
    if typography_cleanup.actions:
        suggestions.append(
            {
                "action": "polish_typography",
                "typography_cleanup": typography_cleanup.to_dict(),
                "reason": "Normalize writing mode, line breaks, vertical punctuation, padding, and fallback fonts before fitting/export.",
            }
        )

    proof_metrics = lettering_proof_metrics(
        text,
        float(getattr(fmt, "font_size", 24.0) or 24.0),
        (box_w, box_h),
        getattr(fmt, "writing_mode", "auto"),
        line_spacing=float(getattr(fmt, "line_spacing", 1.15) or 1.15),
        letter_spacing=float(getattr(fmt, "letter_spacing", 1.0) or 1.0),
        padding=float(getattr(fmt, "text_padding", 0.0) or 0.0),
        stroke_width=float(getattr(fmt, "stroke_width", 0.0) or 0.0),
        secondary_stroke_width=float(getattr(fmt, "secondary_stroke_width", 0.0) or 0.0),
        shadow_radius=float(getattr(fmt, "shadow_radius", 0.0) or 0.0),
        shadow_offset=getattr(fmt, "shadow_offset", [0.0, 0.0]) or [0.0, 0.0],
        line_break_strategy=getattr(fmt, "line_break_strategy", "auto"),
        sample_limit=64,
    )
    line_quality = proof_metrics.get("line_break_quality", {}) or {}
    if line_quality.get("needs_balance") and mode != "vertical_rl":
        if line_quality.get("has_widow"):
            warnings.append("line_break_widow")
        elif line_quality.get("has_kinsoku_violation"):
            warnings.append("line_break_kinsoku_violation")
        else:
            warnings.append("ragged_line_breaks")
        suggestions.append(
            {
                "action": "balance_lines",
                "line_break_quality": line_quality,
                "recommended_strategy": line_quality.get("recommended_strategy", "balanced"),
                "reason": "Wrapped lettering has a widow, kinsoku violation, or uneven line lengths; rebalance before final render.",
            }
        )
    if proof_metrics.get("density", 0) > 0.94 or any(
        float(v) > 0 for v in proof_metrics.get("overflow_pixels", [0, 0])
    ):
        suggestions.append(
            {
                "action": "export_lettering_proof",
                "proof_metrics": proof_metrics,
                "reason": "Crowded or overflowing lettering should be reviewed in the proof handoff before final export.",
            }
        )

    smart_fit = smart_fit_text_to_box(
        text,
        float(getattr(fmt, "font_size", 24.0) or 24.0),
        (box_w, box_h),
        getattr(fmt, "writing_mode", "auto"),
        getattr(fmt, "fit_mode", FIT_MODE_SHRINK),
        min_font_size=style_min,
        max_font_size=style_max,
        line_spacing=float(getattr(fmt, "line_spacing", 1.15) or 1.15),
        letter_spacing=float(getattr(fmt, "letter_spacing", 1.0) or 1.0),
        padding=float(getattr(fmt, "text_padding", 0.0) or 0.0),
        stroke_width=float(getattr(fmt, "stroke_width", 0.0) or 0.0),
        secondary_stroke_width=float(getattr(fmt, "secondary_stroke_width", 0.0) or 0.0),
        line_break_strategy=getattr(fmt, "line_break_strategy", "auto"),
        shadow_radius=float(getattr(fmt, "shadow_radius", 0.0) or 0.0),
        shadow_offset=getattr(fmt, "shadow_offset", [0.0, 0.0]) or [0.0, 0.0],
        effective_box_size=(
            float(effective_box.get("width", box_w)),
            float(effective_box.get("height", box_h)),
        )
        if effective_box.get("uses_mask")
        else None,
    )
    if smart_fit.actions and (
        overflow
        or mask_overflow
        or smart_fit.font_size < float(getattr(fmt, "font_size", 24.0) or 24.0) - 0.2
    ):
        suggestions.append(
            {
                "action": "smart_fit",
                "smart_fit": smart_fit.to_dict(),
                "reason": "Apply balanced wrapping, tracking/leading tightening, writing-mode correction, and font fitting in a safe order.",
            }
        )

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
        "fit_font_size_min": getattr(fmt, "fit_font_size_min", 0.0),
        "fit_font_size_max": getattr(fmt, "fit_font_size_max", 0.0),
        "resolved_fit_font_size_min": style_min,
        "resolved_fit_font_size_max": style_max,
        "line_break_strategy": getattr(fmt, "line_break_strategy", "auto"),
        "font_family": getattr(fmt, "font_family", ""),
        "fallback_chain": merge_font_fallback_chain(
            getattr(fmt, "font_family", ""),
            text,
            config_obj,
            getattr(fmt, "fallback_font_chain", ""),
        ),
        "missing_glyphs": missing,
        "overflow": overflow,
        "fit_diagnostics": fit_diag.to_dict(),
        "safe_inner_bounds": [inner_bounds[0], inner_bounds[1]],
        "effect_margin": effect_margin,
        "quality_score": quality_score,
        "suggested_font_size": fitted_size,
        "smart_fit": smart_fit.to_dict(),
        "recommended_box_size": recommended_box,
        "box_scale_hint": getattr(fit_diag, "box_scale_hint", 1.0),
        "ink_clip_risk": bool(getattr(fit_diag, "ink_clip_risk", False)),
        "mask_diagnostics": mask_diag,
        "textbox_mask_diagnostics": textbox_mask_diag,
        "mask_effective_box": effective_box,
        "preset_suggestion": preset_suggestion,
        "suggested_text": fitted_text if fitted_text != text else "",
        "contrast_effect": effect_suggestion,
        "vertical_layout_plan": vertical_plan,
        "line_break_opportunities": break_ops,
        "warnings": warnings,
        "typography_cleanup": typography_cleanup.to_dict(),
        "proof_metrics": proof_metrics,
        "suggestions": suggestions,
    }


def build_project_rendering_qa(
    project,
    pages: Optional[Sequence[str]] = None,
    include_ok: bool = False,
    config_obj=None,
) -> Dict:
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

        page_entries.append(
            {
                "page": page,
                "textboxes": len(blocks),
                "issues": page_issue_count,
                "blocks": block_entries,
            }
        )

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

            mask_diag = block.get("mask_diagnostics", {}) or {}
            rows.append(
                {
                    "page": page_name,
                    "index": block.get("index", -1),
                    "severity": severity,
                    "warnings": warnings,
                    "suggestions": [
                        s.get("action", "") for s in block.get("suggestions", []) or []
                    ],
                    "writing_mode": block.get(
                        "resolved_writing_mode",
                        block.get("writing_mode", ""),
                    ),
                    "fit_mode": block.get("fit_mode", ""),
                    "line_break_strategy": block.get("line_break_strategy", ""),
                    "font_family": block.get("font_family", ""),
                    "missing_glyphs": "".join(block.get("missing_glyphs", []) or []),
                    "box": block.get("box", []),
                    "measured": block.get("measured", []),
                    "text_length": block.get("text_length", 0),
                    "quality_score": block.get("quality_score", 1.0),
                    "effect_margin": block.get("effect_margin", 0.0),
                    "safe_inner_bounds": block.get("safe_inner_bounds", []),
                    "mask_coverage": mask_diag.get(
                        "coverage",
                        mask_diag.get("visible_area_ratio", 1.0),
                    ),
                    "mask_warning": mask_diag.get("warning", ""),
                    "mask_effective_box": block.get("mask_effective_box", {}),
                    "mask_visible_area_ratio": mask_diag.get("visible_area_ratio", 1.0),
                    "smart_fit_actions": (block.get("smart_fit", {}) or {}).get(
                        "actions",
                        [],
                    ),
                }
            )

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
            "| Page | # | Severity | Quality | Warnings | Suggestions | Mode | Fit | Break | Missing |",
            "| --- | ---: | --- | ---: | --- | --- | --- | --- | --- | --- |",
        ]
        for row in rows:
            warnings = ", ".join(row.get("warnings", []))
            suggestions = ", ".join(row.get("suggestions", []))
            lines.append(
                f"| {row.get('page', '')} | {row.get('index', '')} | {row.get('severity', '')} | {row.get('quality_score', '')} | "
                f"{warnings} | {suggestions} | {row.get('writing_mode', '')} | "
                f"{row.get('fit_mode', '')} | {row.get('line_break_strategy', '')} | {row.get('missing_glyphs', '')} |"
            )

    return "\n".join(lines) + "\n"


def summarize_suggested_actions(report: Dict) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in flatten_rendering_qa_rows(report):
        for action in row.get("suggestions", []) or []:
            counts[action] = counts.get(action, 0) + 1
    return counts


def _selected_fix_map(
    selected_fixes: Optional[Sequence[Dict]],
) -> Optional[Dict[Tuple[str, int], Set[str]]]:
    if selected_fixes is None:
        return None

    out: Dict[Tuple[str, int], Set[str]] = {}
    for item in selected_fixes or []:
        try:
            page = str(item.get("page", ""))
            index = int(item.get("index", -1))
        except Exception:
            continue
        if not page or index < 0:
            continue
        actions = set(str(a) for a in (item.get("actions") or []) if str(a))
        out[(page, index)] = actions or {"*"}

    return out


def _first_suggestion(diag: Dict, action: str) -> Dict:
    for suggestion in diag.get("suggestions", []) or []:
        if suggestion.get("action") == action:
            return suggestion
    return {}


def _resize_block_centered(blk, size: Sequence[float], fallback_size: Tuple[float, float]) -> bool:
    if len(size or []) < 2:
        return False
    try:
        rec_w = max(float(fallback_size[0]), float(size[0]))
        rec_h = max(float(fallback_size[1]), float(size[1]))
    except Exception:
        return False
    if rec_w <= fallback_size[0] * 1.02 and rec_h <= fallback_size[1] * 1.02:
        return False

    blk.xyxy = centered_resize_xyxy(
        getattr(blk, "xyxy", [0, 0, fallback_size[0], fallback_size[1]]),
        (rec_w, rec_h),
    )
    return True


def apply_project_rendering_fixes(
    project,
    pages: Optional[Sequence[str]] = None,
    config_obj=None,
    selected_fixes: Optional[Sequence[Dict]] = None,
) -> Dict:
    """Apply conservative typography fixes directly to project TextBlocks.

    `selected_fixes` is a list of `{page, index, actions}` rows from the QA
    preview. When omitted all conservative fixes are eligible; when provided
    only checked row/action combinations are mutated, making the UI reviewable
    instead of an all-or-nothing project rewrite.
    """
    if config_obj is None:
        from .config import pcfg as config_obj

    report = build_project_rendering_qa(
        project,
        pages=pages,
        include_ok=False,
        config_obj=config_obj,
    )
    applied: List[Dict] = []
    selected_map = _selected_fix_map(selected_fixes)
    page_set = set(pages or (getattr(project, "pages", {}) or {}).keys())

    for page in page_set:
        blocks = list((getattr(project, "pages", {}) or {}).get(page, []) or [])
        for idx, blk in enumerate(blocks):
            diag = analyze_text_block(blk, page, idx, config_obj=config_obj)
            if not diag["warnings"]:
                continue

            allowed = None if selected_map is None else selected_map.get((page, idx))
            if selected_map is not None and allowed is None:
                continue

            def wants(action_name: str) -> bool:
                return allowed is None or "*" in allowed or action_name in allowed

            fmt: FontFormat = getattr(blk, "fontformat", None) or FontFormat()
            text = _block_text(blk)
            box_w, box_h = _box_size(blk)
            changed: List[str] = []

            if "weak_vertical_line_break_strategy" in diag["warnings"] and wants(
                "set_line_break_strategy"
            ):
                fmt.line_break_strategy = LINE_BREAK_CJK_STRICT
                changed.append("line_break_strategy")

            if "rtl_left_alignment" in diag["warnings"] and wants("set_alignment"):
                fmt.alignment = 2
                changed.append("alignment")

            if (
                "low_padding_with_stroke" in diag["warnings"]
                or "ink_clip_risk" in diag["warnings"]
                or "mask_safe_area" in diag["warnings"]
            ) and wants("increase_padding"):
                recommended_padding = 2.0
                if "mask_safe_area" in diag["warnings"]:
                    recommended_padding = float(
                        (diag.get("mask_diagnostics", {}) or {}).get(
                            "recommended_padding",
                            2.0,
                        )
                        or 2.0
                    )
                fmt.text_padding = max(
                    float(getattr(fmt, "text_padding", 0.0) or 0.0),
                    recommended_padding,
                )
                changed.append("text_padding")

            if "horizontal_cjk_in_tall_box" in diag["warnings"] and wants(
                "switch_writing_mode"
            ):
                fmt.writing_mode = "vertical_rl"
                fmt.vertical = True
                changed.append("writing_mode")

            if "vertical_punctuation_needs_normalization" in diag["warnings"] and wants(
                "normalize_vertical_punctuation"
            ):
                blk.translation = normalize_vertical_punctuation(
                    getattr(blk, "translation", "") or text
                )
                changed.append("vertical_punctuation")

            if (
                any(w in diag["warnings"] for w in ("line_break_widow", "line_break_kinsoku_violation", "ragged_line_breaks"))
                and wants("balance_lines")
            ):
                avg = max(1.0, float(getattr(fmt, "font_size", 24.0) or 24.0) * 0.56 * max(0.1, float(getattr(fmt, "letter_spacing", 1.0) or 1.0)))
                max_chars = max(2, int(max(1.0, box_w - 2 * float(getattr(fmt, "text_padding", 0.0) or 0.0)) / avg))
                balanced = "\n".join(optimal_kinsoku_wrap(text.replace("\n", ""), max_chars, getattr(fmt, "line_break_strategy", "balanced")))
                if balanced and balanced != text:
                    blk.translation = balanced
                    fmt.line_break_strategy = "balanced"
                    changed.append("balance_lines")

            if "low_contrast_no_effect" in diag["warnings"] and wants(
                "apply_contrast_stroke"
            ):
                eff = diag.get("contrast_effect", {}) or {}
                fmt.stroke_width = max(
                    float(getattr(fmt, "stroke_width", 0.0) or 0.0),
                    float(eff.get("recommended_stroke_width", 0.06) or 0.06),
                )
                if hasattr(fmt, "srgb"):
                    fmt.srgb = list(eff.get("recommended_stroke_rgb", [255, 255, 255]))
                changed.append("contrast_stroke")

            if "sfx_missing_back_outline" in diag["warnings"] and wants("apply_double_outline"):
                suggestion = next((s for s in diag.get("suggestions", []) if s.get("action") == "apply_double_outline"), {})
                fmt.secondary_stroke_width = max(
                    float(getattr(fmt, "secondary_stroke_width", 0.0) or 0.0),
                    float(suggestion.get("secondary_stroke_width", 0.20) or 0.20),
                )
                fmt.secondary_srgb = list(suggestion.get("secondary_srgb", [255, 255, 255]) or [255, 255, 255])[:3]
                changed.append("double_outline")

            if (
                "missing_glyphs" in diag["warnings"]
                and not getattr(fmt, "fallback_font_chain", "")
                and wants("set_fallback_chain")
            ):
                chain = fallback_chain_for_text(text, config_obj)
                if chain:
                    fmt.fallback_font_chain = chain
                    changed.append("fallback_font_chain")

            if "mask_visible_area_overflow" in diag["warnings"] and wants(
                "resize_to_mask_safe_box"
            ):
                rec = (
                    (diag.get("textbox_mask_diagnostics", {}) or {}).get(
                        "recommended_box_size",
                        [],
                    )
                    or _first_suggestion(diag, "resize_to_mask_safe_box").get(
                        "recommended_box_size",
                        [],
                    )
                )
                if _resize_block_centered(blk, rec, (box_w, box_h)):
                    changed.append("mask_safe_textbox_size")

            if "overflow" in diag["warnings"] and wants("resize_to_recommended_box"):
                rec = diag.get("recommended_box_size", []) or []
                if _resize_block_centered(blk, rec, (box_w, box_h)):
                    changed.append("textbox_size")

            if "overflow" in diag["warnings"] and wants("tighten_letter_spacing"):
                suggestion = _first_suggestion(diag, "tighten_letter_spacing")
                if suggestion:
                    new_spacing = max(
                        0.80,
                        min(
                            float(getattr(fmt, "letter_spacing", 1.0) or 1.0),
                            float(suggestion.get("letter_spacing", 0.94) or 0.94),
                        ),
                    )
                    if new_spacing < float(getattr(fmt, "letter_spacing", 1.0) or 1.0):
                        fmt.letter_spacing = new_spacing
                        changed.append("letter_spacing")

            if wants("smart_fit"):
                smart = diag.get("smart_fit", {}) or {}
                smart_actions = set(smart.get("actions", []) or [])
                if smart_actions and (
                    "overflow" in diag["warnings"]
                    or "mask_safe_overflow" in diag["warnings"]
                    or "horizontal_cjk_in_tall_box" in diag["warnings"]
                ):
                    try:
                        smart_size = float(smart.get("font_size", 0.0) or 0.0)
                    except Exception:
                        smart_size = 0.0

                    if smart_size > 0 and abs(
                        smart_size - float(getattr(fmt, "font_size", 24.0) or 24.0)
                    ) > 0.2:
                        fmt.font_size = smart_size
                        changed.append("smart_font_size")

                    if "tighten_letter_spacing" in smart_actions:
                        try:
                            fmt.letter_spacing = float(
                                smart.get(
                                    "letter_spacing",
                                    getattr(fmt, "letter_spacing", 1.0),
                                )
                                or getattr(fmt, "letter_spacing", 1.0)
                            )
                            changed.append("smart_letter_spacing")
                        except Exception:
                            pass

                    if "tighten_line_spacing" in smart_actions:
                        try:
                            fmt.line_spacing = float(
                                smart.get(
                                    "line_spacing",
                                    getattr(fmt, "line_spacing", 1.15),
                                )
                                or getattr(fmt, "line_spacing", 1.15)
                            )
                            changed.append("smart_line_spacing")
                        except Exception:
                            pass

                    if "switch_writing_mode" in smart_actions:
                        fmt.writing_mode = str(
                            smart.get(
                                "writing_mode",
                                getattr(fmt, "writing_mode", "auto"),
                            )
                            or "auto"
                        )
                        fmt.vertical = fmt.writing_mode == "vertical_rl"
                        changed.append("smart_writing_mode")

                    new_text = str(smart.get("text", "") or "")
                    if new_text and new_text != text and (
                        "balance_lines" in smart_actions
                        or "normalize_vertical_punctuation" in smart_actions
                    ):
                        blk.translation = new_text
                        text = new_text
                        changed.append("smart_text")

                    if smart_actions:
                        fmt.fit_mode = FIT_MODE_SHRINK

            if wants("polish_typography"):
                cleanup = diag.get("typography_cleanup", {}) or {}
                cleanup_actions = set(cleanup.get("actions", []) or [])
                if cleanup_actions:
                    if "switch_writing_mode" in cleanup_actions:
                        fmt.writing_mode = str(
                            cleanup.get(
                                "writing_mode",
                                getattr(fmt, "writing_mode", "auto"),
                            )
                            or "auto"
                        )
                        fmt.vertical = fmt.writing_mode == "vertical_rl"
                        changed.append("polish_writing_mode")

                    if "set_line_break_strategy" in cleanup_actions:
                        fmt.line_break_strategy = str(
                            cleanup.get(
                                "line_break_strategy",
                                getattr(fmt, "line_break_strategy", "auto"),
                            )
                            or "auto"
                        )
                        changed.append("polish_line_breaks")

                    if "increase_padding" in cleanup_actions:
                        try:
                            fmt.text_padding = max(
                                float(getattr(fmt, "text_padding", 0.0) or 0.0),
                                float(cleanup.get("text_padding", 0.0) or 0.0),
                            )
                            changed.append("polish_padding")
                        except Exception:
                            pass

                    if "apply_font_fallback" in cleanup_actions:
                        chain = str(cleanup.get("fallback_font_chain", "") or "")
                        if chain:
                            fmt.fallback_font_chain = chain
                            changed.append("polish_fallback")

                    new_text = str(cleanup.get("text", "") or "")
                    if new_text and new_text != text:
                        blk.translation = new_text
                        text = new_text
                        changed.append("polish_text")

            if "mask_safe_overflow" in diag["warnings"] and wants(
                "shrink_to_mask_safe_area"
            ):
                eff = diag.get("mask_effective_box", {}) or {}
                try:
                    eff_box = (
                        max(1.0, float(eff.get("width", box_w))),
                        max(1.0, float(eff.get("height", box_h))),
                    )
                except Exception:
                    eff_box = (box_w, box_h)

                new_size, _new_text, _fit_diag = fit_font_size_to_box(
                    text,
                    float(getattr(fmt, "font_size", 24.0) or 24.0),
                    eff_box,
                    FIT_MODE_SHRINK,
                    getattr(fmt, "writing_mode", "auto"),
                    padding=float(getattr(fmt, "text_padding", 0.0) or 0.0),
                    stroke_width=float(getattr(fmt, "stroke_width", 0.0) or 0.0),
                    secondary_stroke_width=float(getattr(fmt, "secondary_stroke_width", 0.0) or 0.0),
                    line_spacing=float(getattr(fmt, "line_spacing", 1.15) or 1.15),
                    letter_spacing=float(getattr(fmt, "letter_spacing", 1.0) or 1.0),
                    line_break_strategy=getattr(fmt, "line_break_strategy", "auto"),
                )
                if new_size < float(getattr(fmt, "font_size", 24.0) or 24.0) - 0.2:
                    fmt.font_size = new_size
                    fmt.fit_mode = FIT_MODE_SHRINK
                    changed.append("mask_safe_font_size")

            if "overflow" in diag["warnings"] and (
                wants("shrink_to_fit") or wants("balance_lines")
            ):
                fit_mode = getattr(fmt, "fit_mode", FIT_MODE_SHRINK)
                if fit_mode in (
                    FIT_MODE_SHRINK,
                    FIT_MODE_EXPAND,
                    FIT_MODE_PRESERVE,
                    FIT_MODE_BALANCE,
                ):
                    new_size, new_text, _fit_diag = fit_font_size_to_box(
                        text,
                        float(getattr(fmt, "font_size", 24.0) or 24.0),
                        (box_w, box_h),
                        FIT_MODE_SHRINK,
                        getattr(fmt, "writing_mode", "auto"),
                        padding=float(getattr(fmt, "text_padding", 0.0) or 0.0),
                        stroke_width=float(getattr(fmt, "stroke_width", 0.0) or 0.0),
                        secondary_stroke_width=float(getattr(fmt, "secondary_stroke_width", 0.0) or 0.0),
                        line_spacing=float(getattr(fmt, "line_spacing", 1.15) or 1.15),
                        letter_spacing=float(getattr(fmt, "letter_spacing", 1.0) or 1.0),
                        line_break_strategy=getattr(fmt, "line_break_strategy", "auto"),
                    )
                    if wants("shrink_to_fit") and new_size < float(
                        getattr(fmt, "font_size", 24.0) or 24.0
                    ) - 0.2:
                        fmt.font_size = new_size
                        fmt.fit_mode = FIT_MODE_SHRINK
                        changed.append("font_size")
                    if wants("balance_lines") and new_text and new_text != text and "\n" in new_text:
                        blk.translation = new_text
                        changed.append("balanced_text")

            if changed:
                blk.fontformat = fmt
                applied.append({"page": page, "index": idx, "changed": changed})

    after = build_project_rendering_qa(
        project,
        pages=pages,
        include_ok=False,
        config_obj=config_obj,
    )
    return {
        "applied": applied,
        "applied_count": len(applied),
        "before": report["summary"],
        "after": after["summary"],
    }
