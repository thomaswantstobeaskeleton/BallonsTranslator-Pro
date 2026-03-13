"""
Layout judge: nudge text box toward bubble center, keep it away from bubble edges,
and clamp so the box (and thus text) never goes outside the bubble.
Optional small/fast image model can be used to adjust strength (default: geometric only).
"""

from typing import List, Tuple, Optional
import numpy as np

# Default small, fast models for optional model-assisted judge (image classification).
# These are lightweight and run quickly; leave empty to use geometric-only judge.
DEFAULT_JUDGE_MODEL_IDS = [
    "google/mobilenet_v2_1.0_224",   # ~3.5M params, fast
    "qualcomm/MobileNet-v3-Small",    # ~2.5M params, very fast
]


def judge_textbox_in_bubble(
    box_xywh: List[float],
    bubble_im_xyxy: Tuple[float, float, float, float],
    margin_ratio: float = 0.06,
    center_strength: float = 0.7,
    clamp_overflow: bool = True,
) -> Tuple[float, float, float, float]:
    """
    Suggest a new (x, y, w, h) for the text box so it is closer to the bubble center,
    stays at least margin_ratio away from bubble edges, and never extends outside the bubble.

    Args:
        box_xywh: [x, y, w, h] text box in image coordinates.
        bubble_im_xyxy: (x1, y1, x2, y2) bubble region in image coordinates.
        margin_ratio: Min margin from bubble edges as fraction of min(bubble_w, bubble_h). e.g. 0.06 = 6%.
        center_strength: 0 = no nudge toward center, 1 = full nudge to center.
        clamp_overflow: If True, shrink box to fit inside (bubble - margin) when it would extend out.

    Returns:
        (new_x, new_y, new_w, new_h). Size is clamped so the box never goes out of the bubble.
    """
    if len(box_xywh) < 4:
        return (box_xywh[0], box_xywh[1], max(1, box_xywh[2]), max(1, box_xywh[3]))

    x1, y1, x2, y2 = bubble_im_xyxy
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    margin = margin_ratio * min(bw, bh)
    inner_x1 = x1 + margin
    inner_y1 = y1 + margin
    inner_x2 = x2 - margin
    inner_y2 = y2 - margin
    inner_w = max(1, inner_x2 - inner_x1)
    inner_h = max(1, inner_y2 - inner_y1)

    bx = box_xywh[0]
    by = box_xywh[1]
    bw_box = max(1, box_xywh[2])
    bh_box = max(1, box_xywh[3])

    # Clamp size so box never extends outside the bubble (fix overflow)
    if clamp_overflow:
        if bw_box > inner_w or bh_box > inner_h:
            bw_box = min(bw_box, inner_w)
            bh_box = min(bh_box, inner_h)
            bw_box = max(1, bw_box)
            bh_box = max(1, bh_box)

    # Position: max so box fits inside inner region
    max_x = inner_x2 - bw_box
    max_y = inner_y2 - bh_box
    if max_x < inner_x1 or max_y < inner_y1:
        # Box larger than inner region: center it and use inner size (already clamped above)
        new_x = inner_x1 + (inner_w - bw_box) / 2
        new_y = inner_y1 + (inner_h - bh_box) / 2
        new_x = max(inner_x1, min(new_x, inner_x2 - bw_box))
        new_y = max(inner_y1, min(new_y, inner_y2 - bh_box))
        return (new_x, new_y, float(bw_box), float(bh_box))

    if margin_ratio <= 0 and center_strength <= 0:
        # Only clamp position to inner region
        new_x = max(inner_x1, min(bx, max_x))
        new_y = max(inner_y1, min(by, max_y))
        return (new_x, new_y, float(bw_box), float(bh_box))

    bubble_cx = (x1 + x2) / 2
    bubble_cy = (y1 + y2) / 2
    box_cx = bx + bw_box / 2
    box_cy = by + bh_box / 2

    # Nudge toward bubble center (capped by center_strength)
    dx = (bubble_cx - box_cx) * center_strength
    dy = (bubble_cy - box_cy) * center_strength
    new_x = bx + dx
    new_y = by + dy

    # Clamp to inner region so we don't reach corners
    new_x = max(inner_x1, min(new_x, max_x))
    new_y = max(inner_y1, min(new_y, max_y))
    return (new_x, new_y, float(bw_box), float(bh_box))


def judge_with_model(
    crop_rgb: np.ndarray,
    model_id: str,
) -> Optional[float]:
    """
    Optional: run a small image-classification model on the bubble crop.
    Returns a score in [0, 1] (e.g. confidence or normalized logit) that callers
    can use to scale center_strength or margin. Returns None if model unavailable or fails.

    Default models (fast, small): google/mobilenet_v2_1.0_224, qualcomm/MobileNet-v3-Small.
    """
    if not model_id or not crop_rgb.size or crop_rgb.ndim < 2:
        return None
    try:
        from PIL import Image
        from transformers import pipeline as hf_pipeline
    except ImportError:
        return None
    # Module-level cache per model_id
    cache = getattr(judge_with_model, "_pipe_cache", None)
    if cache is None or cache[0] != model_id:
        try:
            pipe = hf_pipeline("image-classification", model=model_id)
            judge_with_model._pipe_cache = (model_id, pipe)
        except Exception:
            return None
    _, pipe = judge_with_model._pipe_cache
    try:
        if crop_rgb.ndim == 2:
            pil_img = Image.fromarray(crop_rgb).convert("RGB")
        else:
            if crop_rgb.shape[2] == 4:
                crop_rgb = crop_rgb[:, :, :3]
            pil_img = Image.fromarray(crop_rgb).convert("RGB")
        out = pipe(pil_img, top_k=1)
        if not out or not isinstance(out, list) or not out[0]:
            return None
        score = float(out[0].get("score", 0.5))
        return max(0.0, min(1.0, score))
    except Exception:
        return None
