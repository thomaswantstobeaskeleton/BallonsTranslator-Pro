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
