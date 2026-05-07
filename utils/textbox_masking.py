from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np


def _as_mask_array(mask) -> np.ndarray | None:
    if mask is None:
        return None
    arr = np.asarray(mask)
    if arr.size == 0:
        return None
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.uint8, copy=False)


def _scale_rect(rect: Tuple[float, float, float, float], src_size: Tuple[int, int], dst_size: Tuple[float, float]) -> Tuple[float, float, float, float]:
    src_w, src_h = max(1, int(src_size[0])), max(1, int(src_size[1]))
    dst_w, dst_h = max(1.0, float(dst_size[0])), max(1.0, float(dst_size[1]))
    sx = dst_w / src_w
    sy = dst_h / src_h
    x1, y1, x2, y2 = rect
    return x1 * sx, y1 * sy, x2 * sx, y2 * sy


def visible_mask_rect(mask, box_size: Tuple[float, float], threshold: int = 127) -> Dict[str, object]:
    """Return visible/in-bounds area for a per-textbox text mask.

    Text eraser masks use 255=visible and 0=erased.  This helper converts the
    non-zero area into textbox coordinates so overflow checks can use the actual
    visible lettering area instead of the raw rectangle.
    """
    arr = _as_mask_array(mask)
    box_w, box_h = max(1.0, float(box_size[0])), max(1.0, float(box_size[1]))
    if arr is None:
        return {
            "has_mask": False,
            "coverage": 1.0,
            "hidden_ratio": 0.0,
            "visible_rect": [0.0, 0.0, box_w, box_h],
            "visible_size": [box_w, box_h],
            "edge_hidden": False,
            "edge_hidden_ratio": 0.0,
        }
    visible = arr > int(threshold)
    coverage = float(np.count_nonzero(visible)) / float(max(1, visible.size))
    if not np.any(visible):
        rect = (0.0, 0.0, 0.0, 0.0)
    else:
        ys, xs = np.where(visible)
        rect = (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
    scaled = _scale_rect(rect, (arr.shape[1], arr.shape[0]), (box_w, box_h))
    edge = np.concatenate([visible[0, :], visible[-1, :], visible[:, 0], visible[:, -1]]) if visible.size else np.array([], dtype=bool)
    # edge_hidden means the mask erases part of the edge; this often clips stroke/shadow.
    edge_hidden_ratio = 1.0 - (float(np.count_nonzero(edge)) / float(max(1, edge.size)))
    return {
        "has_mask": True,
        "mask_size": [int(arr.shape[1]), int(arr.shape[0])],
        "coverage": round(coverage, 4),
        "hidden_ratio": round(1.0 - coverage, 4),
        "visible_rect": [round(v, 2) for v in scaled],
        "visible_size": [round(max(0.0, scaled[2] - scaled[0]), 2), round(max(0.0, scaled[3] - scaled[1]), 2)],
        "edge_hidden": bool(edge_hidden_ratio > 0.03),
        "edge_hidden_ratio": round(edge_hidden_ratio, 4),
    }


def mask_safe_inner_size(box_size: Tuple[float, float], text_mask=None, padding: float = 0.0) -> Tuple[float, float, Dict[str, object]]:
    diag = visible_mask_rect(text_mask, box_size)
    vis_w, vis_h = diag.get("visible_size", [box_size[0], box_size[1]])
    pad = max(0.0, float(padding or 0.0))
    return max(1.0, float(vis_w) - 2 * pad), max(1.0, float(vis_h) - 2 * pad), diag


def mask_aware_textbox_diagnostics(
    box_size: Tuple[float, float],
    measured_bounds: Tuple[float, float],
    text_mask=None,
    effect_margin: float = 0.0,
    padding: float = 0.0,
) -> Dict[str, object]:
    safe_w, safe_h, mask_diag = mask_safe_inner_size(box_size, text_mask=text_mask, padding=padding)
    measured_w, measured_h = max(0.0, float(measured_bounds[0])), max(0.0, float(measured_bounds[1]))
    overflow_x = measured_w + max(0.0, float(effect_margin or 0.0)) > safe_w
    overflow_y = measured_h + max(0.0, float(effect_margin or 0.0)) > safe_h
    box_w, box_h = max(1.0, float(box_size[0])), max(1.0, float(box_size[1]))
    recommended_w = max(box_w, measured_w + 2 * max(float(effect_margin or 0.0), float(padding or 0.0)))
    recommended_h = max(box_h, measured_h + 2 * max(float(effect_margin or 0.0), float(padding or 0.0)))
    return {
        "mask": mask_diag,
        "safe_inner_size": [round(safe_w, 2), round(safe_h, 2)],
        "mask_overflow": bool(overflow_x or overflow_y),
        "mask_overflow_axes": [axis for axis, flag in (("x", overflow_x), ("y", overflow_y)) if flag],
        "recommended_box_size": [round(recommended_w, 2), round(recommended_h, 2)],
        "visible_area_ratio": round((safe_w * safe_h) / max(1.0, box_w * box_h), 4),
    }


def centered_resize_xyxy(xyxy: Sequence[float], target_size: Sequence[float]) -> list:
    vals = list(xyxy or [0, 0, 0, 0])
    vals = (vals + [0, 0, 0, 0])[:4]
    x1, y1, x2, y2 = [float(v or 0.0) for v in vals]
    cur_w, cur_h = max(1.0, x2 - x1), max(1.0, y2 - y1)
    target_w = max(cur_w, float((target_size or [cur_w, cur_h])[0] or cur_w))
    target_h = max(cur_h, float((target_size or [cur_w, cur_h])[1] or cur_h))
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    return [cx - target_w / 2.0, cy - target_h / 2.0, cx + target_w / 2.0, cy + target_h / 2.0]
