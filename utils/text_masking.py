from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class MaskSafetyDiagnostics:
    """Safe lettering area derived from an eraser/bubble mask.

    `safe_rect` is `(left, top, right, bottom)` in mask-local pixels and
    `safe_insets` is `(left, top, right, bottom)` distance from the full text box.
    Coverage is the ratio of visible mask pixels in the box.
    """

    coverage: float = 1.0
    safe_rect: Tuple[int, int, int, int] = (0, 0, 0, 0)
    safe_insets: Tuple[int, int, int, int] = (0, 0, 0, 0)
    fully_masked: bool = False
    narrow_safe_area: bool = False
    warning: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "coverage": round(float(self.coverage), 4),
            "safe_rect": list(self.safe_rect),
            "safe_insets": list(self.safe_insets),
            "fully_masked": bool(self.fully_masked),
            "narrow_safe_area": bool(self.narrow_safe_area),
            "warning": self.warning,
        }


def _as_uint8_mask(mask: np.ndarray | None) -> np.ndarray | None:
    if mask is None:
        return None
    arr = np.asarray(mask)
    if arr.size == 0:
        return None
    if arr.ndim == 3:
        arr = arr[..., 0]
    return np.ascontiguousarray(arr, dtype=np.uint8)


def mask_safe_rect(mask: np.ndarray | None, threshold: int = 8) -> MaskSafetyDiagnostics:
    """Return visible coverage and largest simple safe rect from a textbox mask.

    BallonsTranslator text masks use 255 for visible text and 0 for erased holes.
    For live UI diagnostics we deliberately use a conservative bounding rectangle
    of visible pixels rather than a costly maximal-rectangle algorithm. This is
    fast, deterministic, and good enough to warn about masks that would clip ink.
    """
    arr = _as_uint8_mask(mask)
    if arr is None:
        return MaskSafetyDiagnostics()
    h, w = arr.shape[:2]
    if w <= 0 or h <= 0:
        return MaskSafetyDiagnostics(coverage=0.0, fully_masked=True, warning="mask has no area")
    visible = arr > int(threshold)
    count = int(visible.sum())
    coverage = count / float(max(1, w * h))
    if count <= 0:
        return MaskSafetyDiagnostics(
            coverage=0.0,
            safe_rect=(0, 0, 0, 0),
            safe_insets=(w, h, w, h),
            fully_masked=True,
            narrow_safe_area=True,
            warning="text mask hides the entire text box",
        )
    ys, xs = np.where(visible)
    left, right = int(xs.min()), int(xs.max()) + 1
    top, bottom = int(ys.min()), int(ys.max()) + 1
    insets = (left, top, max(0, w - right), max(0, h - bottom))
    safe_w = max(0, right - left)
    safe_h = max(0, bottom - top)
    narrow = coverage < 0.72 or safe_w < w * 0.72 or safe_h < h * 0.72
    warning = ""
    if narrow:
        warning = "text mask leaves a small or off-center safe lettering area"
    return MaskSafetyDiagnostics(
        coverage=coverage,
        safe_rect=(left, top, right, bottom),
        safe_insets=insets,
        fully_masked=False,
        narrow_safe_area=bool(narrow),
        warning=warning,
    )


def bubble_safe_text_rect(
    mask: np.ndarray | None,
    region_rect: Sequence[float] | None = None,
    threshold: int = 220,
    min_coverage: float = 0.88,
    min_edge_coverage: float = 0.70,
) -> Dict[str, object]:
    """Find a conservative mask-local rectangle for lettering inside a bubble.

    `extract_ballon_region()` gives a bounding rectangle for the bubble contour,
    but a bounding rectangle still includes unsafe corners for ovals, diamonds,
    and pointed balloons.  This helper searches shrink variants of that rectangle
    and returns the largest candidate whose filled-pixel coverage and edge
    coverage are high enough for text placement.  It is intentionally NumPy-only
    so layout tests and headless environments do not depend on OpenCV/libGL.
    """
    arr = _as_uint8_mask(mask)
    if arr is None:
        return {
            "rect": list(region_rect or [0.0, 0.0, 0.0, 0.0]),
            "coverage": 1.0,
            "edge_coverage": 1.0,
            "used_mask": False,
        }
    h, w = arr.shape[:2]
    if w <= 0 or h <= 0:
        return {
            "rect": [0.0, 0.0, 0.0, 0.0],
            "coverage": 0.0,
            "edge_coverage": 0.0,
            "used_mask": False,
        }
    if region_rect and len(region_rect) >= 4 and region_rect[2] > 0 and region_rect[3] > 0:
        rx, ry, rw, rh = [float(v) for v in region_rect[:4]]
    else:
        visible = arr > int(threshold)
        if not np.any(visible):
            return {
                "rect": [0.0, 0.0, float(w), float(h)],
                "coverage": 0.0,
                "edge_coverage": 0.0,
                "used_mask": False,
            }
        ys, xs = np.where(visible)
        rx, ry = float(xs.min()), float(ys.min())
        rw, rh = float(xs.max() + 1 - xs.min()), float(ys.max() + 1 - ys.min())

    x1 = max(0.0, min(float(w - 1), rx))
    y1 = max(0.0, min(float(h - 1), ry))
    x2 = max(x1 + 1.0, min(float(w), rx + rw))
    y2 = max(y1 + 1.0, min(float(h), ry + rh))
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    base_w, base_h = max(1.0, x2 - x1), max(1.0, y2 - y1)
    visible = arr > int(threshold)

    best = None
    # Prefer larger rectangles but include non-uniform shrink candidates for
    # pointed and diamond balloons where horizontal/vertical safe space differs.
    width_fracs = (1.0, 0.96, 0.92, 0.88, 0.84, 0.78, 0.72, 0.66, 0.58)
    height_fracs = (1.0, 0.96, 0.92, 0.88, 0.84, 0.78, 0.72, 0.66, 0.58)
    for wf in width_fracs:
        for hf in height_fracs:
            cw = max(1.0, base_w * wf)
            ch = max(1.0, base_h * hf)
            lx = int(round(cx - cw / 2.0))
            ty = int(round(cy - ch / 2.0))
            rx2 = int(round(cx + cw / 2.0))
            by = int(round(cy + ch / 2.0))
            lx = max(0, min(w - 1, lx))
            ty = max(0, min(h - 1, ty))
            rx2 = max(lx + 1, min(w, rx2))
            by = max(ty + 1, min(h, by))
            roi = visible[ty:by, lx:rx2]
            if roi.size <= 0:
                continue
            coverage = float(np.mean(roi))
            edge = np.concatenate([roi[0, :], roi[-1, :], roi[:, 0], roi[:, -1]])
            edge_coverage = float(np.mean(edge)) if edge.size else coverage
            area = float((rx2 - lx) * (by - ty))
            if coverage >= float(min_coverage) and edge_coverage >= float(min_edge_coverage):
                score = area * (0.75 + 0.25 * coverage) * (0.85 + 0.15 * edge_coverage)
                if best is None or score > best[0]:
                    best = (score, [float(lx), float(ty), float(rx2 - lx), float(by - ty)], coverage, edge_coverage)
    if best is None:
        return {
            "rect": [float(x1), float(y1), float(base_w), float(base_h)],
            "coverage": 0.0,
            "edge_coverage": 0.0,
            "used_mask": False,
        }
    return {
        "rect": best[1],
        "coverage": round(best[2], 4),
        "edge_coverage": round(best[3], 4),
        "used_mask": True,
    }


def recommended_padding_for_mask(mask: np.ndarray | None, current_padding: float = 0.0, min_padding: float = 2.0) -> float:
    """Suggest padding that keeps rendered ink away from mask edges/holes."""
    diag = mask_safe_rect(mask)
    if diag.fully_masked:
        return max(float(current_padding or 0.0), float(min_padding))
    if not diag.narrow_safe_area:
        return max(float(current_padding or 0.0), float(min_padding))
    left, top, right, bottom = diag.safe_insets
    edge_inset = max(left, top, right, bottom)
    # Cap so a noisy mask cannot request absurd inset; layout review can resize separately.
    return max(float(current_padding or 0.0), min(24.0, max(float(min_padding), float(edge_inset) + 1.0)))


def masked_text_warnings(mask: np.ndarray | None, current_padding: float = 0.0) -> Dict[str, object]:
    diag = mask_safe_rect(mask)
    payload = diag.to_dict()
    payload["recommended_padding"] = round(recommended_padding_for_mask(mask, current_padding), 2)
    return payload


def mask_effective_box(mask: np.ndarray | None, box_size: Tuple[float, float], current_padding: float = 0.0) -> Dict[str, object]:
    """Return effective fitting bounds after accounting for a textbox mask.

    The returned width/height are conservative: for normal/full masks they match
    the original box, while narrow/off-center masks use the visible safe rect
    minus the recommended padding. This lets QA/review detect overflow that would
    be invisible in the final masked render even when the full textbox looks big
    enough.
    """
    bw, bh = box_size or (0.0, 0.0)
    bw = max(1.0, float(bw or 0.0))
    bh = max(1.0, float(bh or 0.0))
    diag = mask_safe_rect(mask)
    if mask is None or (not diag.narrow_safe_area and not diag.fully_masked):
        return {
            "width": bw,
            "height": bh,
            "offset": [0.0, 0.0],
            "uses_mask": False,
            "coverage": diag.coverage,
            "safe_rect": list(diag.safe_rect),
            "recommended_padding": round(max(float(current_padding or 0.0), 0.0), 2),
        }
    if diag.fully_masked:
        return {
            "width": 1.0,
            "height": 1.0,
            "offset": [0.0, 0.0],
            "uses_mask": True,
            "coverage": diag.coverage,
            "safe_rect": list(diag.safe_rect),
            "recommended_padding": recommended_padding_for_mask(mask, current_padding),
        }
    left, top, right, bottom = diag.safe_rect
    pad = recommended_padding_for_mask(mask, current_padding)
    safe_w = max(1.0, float(right - left) - 2.0 * float(pad or 0.0))
    safe_h = max(1.0, float(bottom - top) - 2.0 * float(pad or 0.0))
    return {
        "width": min(bw, safe_w),
        "height": min(bh, safe_h),
        "offset": [float(left) + float(pad or 0.0), float(top) + float(pad or 0.0)],
        "uses_mask": True,
        "coverage": diag.coverage,
        "safe_rect": list(diag.safe_rect),
        "recommended_padding": round(float(pad or 0.0), 2),
    }
