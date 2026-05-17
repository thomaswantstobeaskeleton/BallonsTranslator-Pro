from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from .rendering_qa import build_project_rendering_qa, flatten_rendering_qa_rows, summarize_suggested_actions


CORE_FIX_ACTIONS = {
    "polish_typography",
    "smart_fit",
    "atomic_bubble_fit",
    "shrink_to_fit",
    "shrink_to_mask_safe_area",
    "switch_writing_mode",
    "normalize_vertical_punctuation",
    "set_line_break_strategy",
    "increase_padding",
    "tighten_letter_spacing",
    "resize_to_recommended_box",
    "resize_to_mask_safe_box",
    "recenter",
    "apply_manga_preset",
    "apply_contrast_stroke",
}


def _severity_score(row: Dict) -> int:
    warnings = set(row.get("warnings") or [])
    score = 0
    if "overflow" in warnings or "mask_safe_overflow" in warnings:
        score += 40
    if "missing_glyphs" in warnings:
        score += 35
    if "horizontal_cjk_in_tall_box" in warnings or "vertical_punctuation_needs_normalization" in warnings:
        score += 25
    if "ink_clip_risk" in warnings or "unsafe_effect_bounds" in warnings:
        score += 20
    score += min(30, len(warnings) * 5)
    try:
        score += int((1.0 - float(row.get("quality_score", 1.0))) * 25)
    except Exception:
        pass
    return score


def _suggested_fix_payload(report: Dict, max_items: int = 200) -> List[Dict]:
    fixes: List[Dict] = []
    for page in report.get("pages", []) or []:
        page_name = page.get("page", "")
        for block in page.get("blocks", []) or []:
            actions = []
            for suggestion in block.get("suggestions", []) or []:
                action = str(suggestion.get("action", "") or "")
                if action in CORE_FIX_ACTIONS and action not in actions:
                    actions.append(action)
            if actions:
                fixes.append({"page": page_name, "index": block.get("index", -1), "actions": actions})
                if len(fixes) >= max_items:
                    return fixes
    return fixes


def build_lettering_workflow_plan(
    project,
    pages: Optional[Sequence[str]] = None,
    config_obj=None,
    include_ok: bool = False,
    max_focus_items: int = 12,
) -> Dict:
    """Build a one-click manga lettering workflow plan for UI/API users.

    The plan turns detailed rendering QA into an ordered workflow: polish low-risk
    typography, smart-fit/resize problem boxes, optionally run layout review when
    high-risk issues remain, then export a proof pack or final render.  It is
    intentionally side-effect free so it can power menus, status panels, tests,
    and the local automation API.
    """
    report = build_project_rendering_qa(project, pages=pages, include_ok=include_ok, config_obj=config_obj)
    rows = flatten_rendering_qa_rows(report)
    rows = sorted(rows, key=_severity_score, reverse=True)
    issue_rows = [row for row in rows if row.get("warnings")]
    action_counts = summarize_suggested_actions(report)
    core_actions = {k: v for k, v in action_counts.items() if k in CORE_FIX_ACTIONS}

    high_risk_warnings = {
        "overflow",
        "mask_safe_overflow",
        "missing_glyphs",
        "horizontal_cjk_in_tall_box",
        "ink_clip_risk",
        "unsafe_effect_bounds",
    }
    needs_layout_review = any(high_risk_warnings.intersection(set(row.get("warnings") or [])) for row in issue_rows)
    needs_proof = bool(issue_rows) or any(k in action_counts for k in ("export_lettering_proof", "apply_manga_preset"))

    steps: List[Dict] = []
    if core_actions.get("polish_typography") or core_actions.get("normalize_vertical_punctuation") or core_actions.get("set_line_break_strategy"):
        steps.append({
            "id": "polish_typography",
            "label": "Polish typography",
            "apply_order": 10,
            "reason": "Resolve writing mode, vertical punctuation, line-break policy, padding, and fallback chains before fitting.",
            "affected_textboxes": max(core_actions.get("polish_typography", 0), core_actions.get("normalize_vertical_punctuation", 0), core_actions.get("set_line_break_strategy", 0)),
        })
    if core_actions.get("atomic_bubble_fit") or core_actions.get("smart_fit") or core_actions.get("shrink_to_fit") or core_actions.get("tighten_letter_spacing") or core_actions.get("resize_to_recommended_box"):
        steps.append({
            "id": "smart_fit",
            "label": "Atomic/smart fit textboxes",
            "apply_order": 20,
            "reason": "Apply atomic bubble fitting, balanced wrapping, tracking/leading tightening, and fit-to-box sizing.",
            "affected_textboxes": max(core_actions.get("atomic_bubble_fit", 0), core_actions.get("smart_fit", 0), core_actions.get("shrink_to_fit", 0), core_actions.get("resize_to_recommended_box", 0)),
        })
    if needs_layout_review:
        steps.append({
            "id": "layout_review",
            "label": "Run layout review",
            "apply_order": 30,
            "reason": "High-risk overflow, mask, glyph, or placement issues remain and should use the review agent/fallback heuristics.",
            "affected_textboxes": len(issue_rows),
        })
    if needs_proof:
        steps.append({
            "id": "export_lettering_proof",
            "label": "Export lettering proof pack",
            "apply_order": 40,
            "reason": "Generate QA JSON/Markdown/SVG/PSD-helper handoff before final lettering export.",
            "affected_textboxes": len(issue_rows),
        })
    steps.append({
        "id": "render_current_page",
        "label": "Render current page",
        "apply_order": 50,
        "reason": "Refresh final composite after text formatting fixes.",
        "affected_textboxes": len(rows),
    })

    return {
        "ok": True,
        "summary": report.get("summary", {}),
        "action_summary": action_counts,
        "core_action_summary": core_actions,
        "steps": sorted(steps, key=lambda x: x.get("apply_order", 0)),
        "focus_items": issue_rows[: max(0, int(max_focus_items or 0))],
        "selected_fixes": _suggested_fix_payload(report),
        "has_issues": bool(issue_rows),
        "needs_layout_review": bool(needs_layout_review),
        "needs_proof": bool(needs_proof),
    }


def next_rendering_issue(report: Dict, after_page: str = "", after_index: int = -1) -> Dict:
    """Return the next issue row after the current page/index, wrapping once."""
    rows = []
    for page in report.get("pages", []) or []:
        for block in page.get("blocks", []) or []:
            if block.get("warnings"):
                rows.append(block)
    if not rows:
        return {"found": False}
    keyed = [(str(row.get("page", "")), int(row.get("index", -1)), row) for row in rows]
    for page, idx, row in keyed:
        if page > after_page or (page == after_page and idx > int(after_index)):
            return {"found": True, "page": page, "index": idx, "issue": row}
    page, idx, row = keyed[0]
    return {"found": True, "page": page, "index": idx, "issue": row, "wrapped": True}
