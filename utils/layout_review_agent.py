from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Protocol, Sequence, Tuple

ActionType = Literal[
    "move",
    "resize",
    "resize_to_recommended_box",
    "resize_to_mask_safe_box",
    "set_font_size",
    "auto_fit",
    "center_in_bubble",
    "resize_to_fit_content",
    "shrink_to_fit",
    "switch_writing_mode",
    "recenter",
    "increase_padding",
    "tighten_letter_spacing",
    "balance_lines",
    "apply_manga_preset",
    "set_line_break_strategy",
    "apply_font_fallback",
    "flag_missing_glyphs",
    "normalize_vertical_punctuation",
    "set_alignment",
    "apply_contrast_stroke",
]


@dataclass
class ReviewIssue:
    """A deterministic issue emitted by the review pass."""

    code: str
    severity: Literal["info", "warning", "error"]
    message: str
    score_penalty: float = 0.0


@dataclass
class ReviewAction:
    """Action that can be safely replayed and undone by caller command stack."""

    action: ActionType
    block_index: int
    args: Dict[str, float | int | str | bool] = field(default_factory=dict)
    reason: str = ""


@dataclass
class BlockSnapshot:
    """Minimal geometry/style snapshot used by the planner.

    This is intentionally UI-agnostic so callers can map TextBlkItem/TextBlock into it.
    """

    block_index: int
    xyxy: Tuple[float, float, float, float]
    text: str
    font_size: float
    bubble_center: Optional[Tuple[float, float]] = None
    est_text_size: Optional[Tuple[float, float]] = None
    writing_mode: str = "auto"
    resolved_writing_mode: str = "horizontal_ltr"
    overflow_status: bool = False
    measured_bounds: Optional[Tuple[float, float]] = None
    text_style: Dict[str, Any] = field(default_factory=dict)
    font_fallback_warning: str = ""
    quality_score: float = 1.0


@dataclass
class BlockReviewResult:
    block_index: int
    score_before: float
    score_after: float
    issues: List[ReviewIssue]
    actions: List[ReviewAction]


@dataclass
class PageReviewResult:
    page_name: str
    blocks: List[BlockReviewResult]


@dataclass
class ReviewModelConfig:
    """Model/provider controls for prompt and generation parameters."""

    provider: Literal["heuristic", "local_api", "cloud_api"] = "heuristic"
    model_name: str = ""
    prompt: str = (
        "Review text boxes for overflow, off-center placement, and readability issues. "
        "Return deterministic fix actions only."
    )
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 512
    include_page_screenshot: bool = True
    screenshot_max_side: int = 1280
    extra_params: Dict[str, Any] = field(default_factory=dict)


class ReviewProvider(Protocol):
    def review(self, page_name: str, blocks: Sequence[BlockSnapshot], config: ReviewModelConfig) -> PageReviewResult:
        ...


class LayoutReviewPlanner:
    """Heuristic first-pass planner for text box clean-up.

    The planner is conservative: it proposes small fixes that can be executed through
    existing commands (move/resize/auto-fit/center). A higher-level caller can run
    these actions, re-render, then feed updated snapshots back for another pass.
    """

    def review_page(self, page_name: str, blocks: Sequence[BlockSnapshot]) -> PageReviewResult:
        reviewed: List[BlockReviewResult] = []
        for blk in blocks:
            issues, actions, before, after = self._review_block(blk)
            reviewed.append(
                BlockReviewResult(
                    block_index=blk.block_index,
                    score_before=before,
                    score_after=after,
                    issues=issues,
                    actions=actions,
                )
            )
        return PageReviewResult(page_name=page_name, blocks=reviewed)

    def _review_block(self, blk: BlockSnapshot):
        issues: List[ReviewIssue] = []
        actions: List[ReviewAction] = []

        x1, y1, x2, y2 = blk.xyxy
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        box_area = bw * bh

        est_w, est_h = blk.est_text_size or (0.0, 0.0)
        overflow_x = est_w > bw * 0.98
        overflow_y = est_h > bh * 0.98
        if blk.overflow_status:
            overflow_x = overflow_x or est_w > bw * 0.92
            overflow_y = overflow_y or est_h > bh * 0.92

        score_before = 1.0

        if overflow_x or overflow_y:
            issues.append(
                ReviewIssue(
                    code="text_overflow",
                    severity="warning",
                    message="Estimated text bounds exceed text box bounds.",
                    score_penalty=0.25,
                )
            )
            recommended = (blk.text_style or {}).get("recommended_box_size") or ()
            try:
                rec_w = float(recommended[0])
                rec_h = float(recommended[1])
            except Exception:
                rec_w = rec_h = 0.0
            if rec_w > bw * 1.02 or rec_h > bh * 1.02:
                actions.append(
                    ReviewAction(
                        action="resize_to_recommended_box",
                        block_index=blk.block_index,
                        args={"width": max(bw, rec_w), "height": max(bh, rec_h)},
                        reason="Use effect-aware fit diagnostics to resize the box just enough for the measured lettering.",
                    )
                )
            actions.append(
                ReviewAction(
                    action="shrink_to_fit",
                    block_index=blk.block_index,
                    reason="Reduce font size using the selected fit policy before resizing.",
                )
            )
            actions.append(
                ReviewAction(
                    action="auto_fit",
                    block_index=blk.block_index,
                    reason="Compatibility action: reduce font size to fit current box before resizing.",
                )
            )
            letter_spacing = float((blk.text_style or {}).get("letter_spacing", 1.0) or 1.0)
            if overflow_x and letter_spacing > 0.92:
                from .text_rendering import recommended_tight_letter_spacing
                actions.append(
                    ReviewAction(
                        action="tighten_letter_spacing",
                        block_index=blk.block_index,
                        args={"letter_spacing": recommended_tight_letter_spacing(letter_spacing, max(est_w / max(1.0, bw), 1.0))},
                        reason="Tighten tracking slightly before shrinking/resizing wide manga lettering.",
                    )
                )
            if (blk.text_style or {}).get("fit_mode") == "balance" or (blk.text_style or {}).get("line_break_strategy") != "balanced":
                actions.append(
                    ReviewAction(
                        action="balance_lines",
                        block_index=blk.block_index,
                        reason="Rebalance line breaks to reduce overflow without changing wording.",
                    )
                )
            actions.append(
                ReviewAction(
                    action="resize_to_fit_content",
                    block_index=blk.block_index,
                    reason="Expand box if auto-fit alone is insufficient.",
                )
            )
            score_before -= 0.25


        # Mask-visible area can overflow even when the raw rectangle does not.
        mask_diag = (blk.text_style or {}).get("mask_diagnostics") or {}
        if bool(mask_diag.get("mask_overflow")) and not any(i.code == "mask_visible_area_overflow" for i in issues):
            issues.append(
                ReviewIssue(
                    code="mask_visible_area_overflow",
                    severity="warning",
                    message="Measured lettering exceeds the visible area left by the text mask.",
                    score_penalty=0.14,
                )
            )
            actions.append(
                ReviewAction(
                    action="resize_to_mask_safe_box",
                    block_index=blk.block_index,
                    args={"width": float((mask_diag.get("recommended_box_size") or [bw, bh])[0]), "height": float((mask_diag.get("recommended_box_size") or [bw, bh])[1])},
                    reason="Grow the textbox enough that rendered lettering fits inside the mask-visible area.",
                )
            )
            score_before -= 0.14
        mask_info = mask_diag.get("mask") or {}
        if bool(mask_info.get("edge_hidden")) and not any(i.code == "text_mask_erases_edge" for i in issues):
            issues.append(
                ReviewIssue(
                    code="text_mask_erases_edge",
                    severity="info",
                    message="The text eraser mask hides part of the textbox edge, risking stroke/shadow clipping.",
                    score_penalty=0.04,
                )
            )
            actions.append(
                ReviewAction(
                    action="increase_padding",
                    block_index=blk.block_index,
                    args={"padding": max(2.0, float((blk.text_style or {}).get("text_padding", 0.0) or 0.0) + 1.0)},
                    reason="Inset lettering away from erased mask edges.",
                )
            )
            score_before -= 0.04

        if blk.bubble_center is not None:
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            bcx, bcy = blk.bubble_center
            dx, dy = abs(cx - bcx), abs(cy - bcy)
            if dx > bw * 0.2 or dy > bh * 0.2:
                issues.append(
                    ReviewIssue(
                        code="off_center",
                        severity="info",
                        message="Text box center is far from bubble center.",
                        score_penalty=0.1,
                    )
                )
                actions.append(
                    ReviewAction(
                        action="center_in_bubble",
                        block_index=blk.block_index,
                        reason="Recenters text while preserving style.",
                    )
                )
                score_before -= 0.1

        tiny_area = box_area < 64 * 64
        if tiny_area and len((blk.text or "").strip()) > 20:
            issues.append(
                ReviewIssue(
                    code="tiny_box_dense_text",
                    severity="warning",
                    message="Dense text in very small box likely unreadable.",
                    score_penalty=0.15,
                )
            )
            actions.append(
                ReviewAction(
                    action="resize",
                    block_index=blk.block_index,
                    args={"scale": 1.1},
                    reason="Slightly increase box area before final fit pass.",
                )
            )
            score_before -= 0.15


        if blk.font_fallback_warning:
            issues.append(
                ReviewIssue(
                    code="missing_glyphs",
                    severity="warning",
                    message=f"Selected font may not render: {blk.font_fallback_warning}",
                    score_penalty=0.12,
                )
            )
            actions.append(
                ReviewAction(
                    action="flag_missing_glyphs",
                    block_index=blk.block_index,
                    args={"glyphs": blk.font_fallback_warning},
                    reason="Flag missing glyphs for font fallback or font-family correction.",
                )
            )
            chain = str((blk.text_style or {}).get("fallback_chain", "") or "")
            if chain:
                actions.append(
                    ReviewAction(
                        action="apply_font_fallback",
                        block_index=blk.block_index,
                        args={"fallback_chain": chain},
                        reason="Apply the configured fallback font chain to unsupported glyph runs.",
                    )
                )
            score_before -= 0.12


        if blk.resolved_writing_mode == "vertical_rl" and any(seq in (blk.text or "") for seq in ("?!", "!?", "!!", "??", "...")):
            issues.append(
                ReviewIssue(
                    code="vertical_punctuation_normalization",
                    severity="info",
                    message="Vertical CJK punctuation can use compact vertical forms.",
                    score_penalty=0.03,
                )
            )
            actions.append(
                ReviewAction(
                    action="normalize_vertical_punctuation",
                    block_index=blk.block_index,
                    reason="Normalize repeated punctuation for vertical manga lettering.",
                )
            )
            score_before -= 0.03

        if blk.resolved_writing_mode == "rtl" and int((blk.text_style or {}).get("alignment", 0) or 0) == 0:
            issues.append(
                ReviewIssue(
                    code="rtl_alignment",
                    severity="info",
                    message="RTL text is currently left aligned.",
                    score_penalty=0.05,
                )
            )
            actions.append(
                ReviewAction(
                    action="set_alignment",
                    block_index=blk.block_index,
                    args={"alignment": 2},
                    reason="Right-align RTL text for more predictable editing and export.",
                )
            )
            score_before -= 0.05

        if float((blk.text_style or {}).get("contrast_ratio", 7.0) or 7.0) < 4.5 and float((blk.text_style or {}).get("stroke_width", 0.0) or 0.0) <= 0:
            issues.append(
                ReviewIssue(
                    code="low_contrast_no_effect",
                    severity="warning",
                    message="Text/background contrast is low and no outline is set.",
                    score_penalty=0.10,
                )
            )
            actions.append(
                ReviewAction(
                    action="apply_contrast_stroke",
                    block_index=blk.block_index,
                    args={"stroke_width": 0.06},
                    reason="Add a conservative manga outline for low-contrast lettering.",
                )
            )
            score_before -= 0.10

        if bool((blk.text_style or {}).get("ink_clip_risk", False)):
            issues.append(
                ReviewIssue(
                    code="ink_clip_risk",
                    severity="warning",
                    message="Estimated ink bounds may clip stroke, glow, or shadow at the text box edge.",
                    score_penalty=0.06,
                )
            )
            actions.append(
                ReviewAction(
                    action="increase_padding",
                    block_index=blk.block_index,
                    args={"padding": max(2.0, float((blk.text_style or {}).get("text_padding", 0.0) or 0.0) + 1.0)},
                    reason="Increase inset before final render so manga effects have safe ink margin.",
                )
            )
            score_before -= 0.06

        suggested_preset = str((blk.text_style or {}).get("preset_suggestion", "") or "")
        if suggested_preset and suggested_preset != str((blk.text_style or {}).get("preset", "") or ""):
            if suggested_preset in {"vertical_cjk_bubble", "sfx_bold", "caption_box"}:
                issues.append(
                    ReviewIssue(
                        code="preset_suggestion",
                        severity="info",
                        message=f"Renderer diagnostics suggest the {suggested_preset} manga lettering preset.",
                        score_penalty=0.03,
                    )
                )
                actions.append(
                    ReviewAction(
                        action="apply_manga_preset",
                        block_index=blk.block_index,
                        args={"preset": suggested_preset},
                        reason="Apply the preset suggested by script, geometry, and lettering diagnostics.",
                    )
                )
                score_before -= 0.03

        if float(getattr(blk, "quality_score", 1.0) or 1.0) < 0.7:
            issues.append(
                ReviewIssue(
                    code="low_lettering_quality_score",
                    severity="info",
                    message="Renderer diagnostics assigned a low lettering quality score.",
                    score_penalty=0.04,
                )
            )
            score_before -= 0.04

        style = blk.text_style or {}
        padding = float(style.get("text_padding", 0.0) or 0.0)
        stroke_width = float(style.get("stroke_width", 0.0) or 0.0)
        if blk.resolved_writing_mode == "vertical_rl" and style.get("line_break_strategy") not in ("cjk_strict", "balanced"):
            issues.append(
                ReviewIssue(
                    code="vertical_cjk_line_break_policy",
                    severity="info",
                    message="Vertical CJK text should use strict kinsoku line breaking.",
                    score_penalty=0.04,
                )
            )
            actions.append(
                ReviewAction(
                    action="set_line_break_strategy",
                    block_index=blk.block_index,
                    args={"line_break_strategy": "cjk_strict"},
                    reason="Use strict CJK punctuation guards for vertical lettering.",
                )
            )
            score_before -= 0.04

        if blk.resolved_writing_mode == "vertical_rl" and style.get("fit_mode") in (None, "", "preserve"):
            issues.append(
                ReviewIssue(
                    code="vertical_manga_preset_recommended",
                    severity="info",
                    message="Vertical CJK text would benefit from the vertical manga bubble preset.",
                    score_penalty=0.04,
                )
            )
            actions.append(
                ReviewAction(
                    action="apply_manga_preset",
                    block_index=blk.block_index,
                    args={"preset": "vertical_cjk_bubble"},
                    reason="Apply vertical CJK bubble spacing, stroke, alignment, and padding.",
                )
            )
            score_before -= 0.04
        if stroke_width > 0 and padding < 1.0:
            issues.append(
                ReviewIssue(
                    code="low_padding_with_stroke",
                    severity="info",
                    message="Outlined text has little inset and may clip at the box edge.",
                    score_penalty=0.05,
                )
            )
            actions.append(
                ReviewAction(
                    action="increase_padding",
                    block_index=blk.block_index,
                    args={"padding": 2.0},
                    reason="Add inset so stroke/shadow does not clip.",
                )
            )
            score_before -= 0.05

        # Mode sanity: auto-resolved vertical/RTL but persisted style is forced differently.
        if blk.writing_mode not in ("auto", blk.resolved_writing_mode):
            if blk.resolved_writing_mode in ("vertical_rl", "rtl"):
                issues.append(
                    ReviewIssue(
                        code="writing_mode_mismatch",
                        severity="info",
                        message=f"Text appears better suited to {blk.resolved_writing_mode} writing mode.",
                        score_penalty=0.08,
                    )
                )
                actions.append(
                    ReviewAction(
                        action="switch_writing_mode",
                        block_index=blk.block_index,
                        args={"writing_mode": blk.resolved_writing_mode},
                        reason="Switch writing mode to match script and box geometry.",
                    )
                )
                score_before -= 0.08

        score_after = max(score_before + 0.2 * (1 if actions else 0), 0.0)
        return issues, actions, max(score_before, 0.0), min(score_after, 1.0)


def collect_actions_by_type(page_result: PageReviewResult) -> Dict[ActionType, List[ReviewAction]]:
    """Group proposed actions by action type."""
    grouped: Dict[ActionType, List[ReviewAction]] = {
        "move": [],
        "resize": [],
        "resize_to_recommended_box": [],
        "resize_to_mask_safe_box": [],
        "set_font_size": [],
        "auto_fit": [],
        "center_in_bubble": [],
        "resize_to_fit_content": [],
        "shrink_to_fit": [],
        "switch_writing_mode": [],
        "recenter": [],
        "increase_padding": [],
        "tighten_letter_spacing": [],
        "balance_lines": [],
        "apply_manga_preset": [],
        "set_line_break_strategy": [],
        "apply_font_fallback": [],
        "flag_missing_glyphs": [],
        "normalize_vertical_punctuation": [],
        "set_alignment": [],
        "apply_contrast_stroke": [],
    }
    for blk in page_result.blocks:
        for action in blk.actions:
            grouped[action.action].append(action)
    return grouped


def collect_actions_from_list(actions: Sequence[ReviewAction]) -> Dict[ActionType, List[ReviewAction]]:
    """Group already-flattened actions by action type."""
    grouped: Dict[ActionType, List[ReviewAction]] = {
        "move": [],
        "resize": [],
        "resize_to_recommended_box": [],
        "resize_to_mask_safe_box": [],
        "set_font_size": [],
        "auto_fit": [],
        "center_in_bubble": [],
        "resize_to_fit_content": [],
        "shrink_to_fit": [],
        "switch_writing_mode": [],
        "recenter": [],
        "increase_padding": [],
        "tighten_letter_spacing": [],
        "balance_lines": [],
        "apply_manga_preset": [],
        "set_line_break_strategy": [],
        "apply_font_fallback": [],
        "flag_missing_glyphs": [],
        "normalize_vertical_punctuation": [],
        "set_alignment": [],
        "apply_contrast_stroke": [],
    }
    for action in actions:
        grouped[action.action].append(action)
    return grouped


def filter_actions_by_block_indices(
    actions: Sequence[ReviewAction], block_indices: Sequence[int]
) -> List[ReviewAction]:
    """Keep only actions targeting the requested block indices."""
    target_ids = set(block_indices)
    return [a for a in actions if a.block_index in target_ids]


class HeuristicReviewProvider:
    """Default deterministic provider backed by LayoutReviewPlanner."""

    def __init__(self):
        self.planner = LayoutReviewPlanner()

    def review(self, page_name: str, blocks: Sequence[BlockSnapshot], config: ReviewModelConfig) -> PageReviewResult:
        return self.planner.review_page(page_name, blocks)


class ExternalReviewProvider:
    """Adapter for local/cloud API review handlers.

    `handler` should accept `(page_name, blocks, config)` and return `PageReviewResult`.
    """

    def __init__(self, handler: Callable[[str, Sequence[BlockSnapshot], ReviewModelConfig], PageReviewResult]):
        self.handler = handler

    def review(self, page_name: str, blocks: Sequence[BlockSnapshot], config: ReviewModelConfig) -> PageReviewResult:
        return self.handler(page_name, blocks, config)


def iterative_review(
    page_name: str,
    initial_blocks: Sequence[BlockSnapshot],
    apply_actions: Callable[[List[ReviewAction]], Sequence[BlockSnapshot]],
    max_iters: int = 3,
) -> Tuple[PageReviewResult, int]:
    """Run review -> apply -> re-review loop.

    Caller provides `apply_actions`, which maps actions into actual UI/model mutations
    and returns the new block snapshots from the updated render state.
    """

    planner = LayoutReviewPlanner()
    current = list(initial_blocks)
    last = planner.review_page(page_name, current)

    for i in range(max_iters):
        pending = [a for b in last.blocks for a in b.actions]
        if not pending:
            return last, i
        current = list(apply_actions(pending))
        last = planner.review_page(page_name, current)

    return last, max_iters
