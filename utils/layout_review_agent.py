from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Protocol, Sequence, Tuple

ActionType = Literal[
    "move",
    "resize",
    "set_font_size",
    "auto_fit",
    "center_in_bubble",
    "resize_to_fit_content",
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
            actions.append(
                ReviewAction(
                    action="auto_fit",
                    block_index=blk.block_index,
                    reason="Reduce font size to fit current box before resizing.",
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

        score_after = max(score_before + 0.2 * (1 if actions else 0), 0.0)
        return issues, actions, max(score_before, 0.0), min(score_after, 1.0)


def collect_actions_by_type(page_result: PageReviewResult) -> Dict[ActionType, List[ReviewAction]]:
    """Group proposed actions by action type."""
    grouped: Dict[ActionType, List[ReviewAction]] = {
        "move": [],
        "resize": [],
        "set_font_size": [],
        "auto_fit": [],
        "center_in_bubble": [],
        "resize_to_fit_content": [],
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
        "set_font_size": [],
        "auto_fit": [],
        "center_in_bubble": [],
        "resize_to_fit_content": [],
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
