"""
Optional model-based bubble shape classification for layout_balloon_shape == "auto".
Uses Hugging Face image-classification pipeline (e.g. geometric-shape or bubble-shape models).
Falls back to contour or aspect-ratio when model is disabled or fails.
"""
from typing import Optional, List, Tuple
import numpy as np

# Module-level cache: (model_id, pipeline) so we load once per model_id
_bubble_shape_pipeline_cache: Optional[Tuple[str, object]] = None

# Map common HF model labels to our shape names (round, elongated, narrow, square, diamond, pentagon, bevel, point)
LABEL_TO_SHAPE = {
    "circle": "round",
    "oval": "elongated",
    "ellipse": "round",
    "round": "round",
    "rectangle": "elongated",
    "rect": "elongated",
    "square": "square",
    "diamond": "diamond",
    "rhombus": "diamond",
    "pentagon": "pentagon",
    "hexagon": "bevel",
    "triangle": "point",
    "star": "point",
    "trapezoid": "bevel",
    "kite": "point",
}


def get_bubble_shape_from_model(
    mask: np.ndarray,
    mask_xyxy: List[int],
    img: np.ndarray,
    model_id: str,
) -> Optional[str]:
    """
    Run optional image-classification model on bubble crop to get shape.
    Returns our shape name (round, square, diamond, etc.) or None on failure.
    """
    if not model_id or not img.size or len(mask_xyxy) < 4:
        return None
    x1, y1, x2, y2 = int(mask_xyxy[0]), int(mask_xyxy[1]), int(mask_xyxy[2]), int(mask_xyxy[3])
    h_img, w_img = img.shape[:2]
    x1 = max(0, min(x1, w_img - 1))
    y1 = max(0, min(y1, h_img - 1))
    x2 = max(x1 + 1, min(x2, w_img))
    y2 = max(y1 + 1, min(y2, h_img))
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    try:
        from PIL import Image
        from transformers import pipeline as hf_pipeline
    except ImportError:
        return None
    global _bubble_shape_pipeline_cache
    if _bubble_shape_pipeline_cache is None or _bubble_shape_pipeline_cache[0] != model_id:
        try:
            pipe = hf_pipeline("image-classification", model=model_id)
            _bubble_shape_pipeline_cache = (model_id, pipe)
        except Exception:
            return None
    _, pipe = _bubble_shape_pipeline_cache
    try:
        if crop.ndim == 2:
            pil_img = Image.fromarray(crop).convert("RGB")
        else:
            if crop.shape[2] == 4:
                crop = crop[:, :, :3]
            pil_img = Image.fromarray(crop).convert("RGB")
        crop_h, crop_w = crop.shape[0], crop.shape[1]
        is_wide = crop_w > crop_h * 1.2 if (crop_h and crop_w) else False
        out = pipe(pil_img, top_k=1)
        if not out or not isinstance(out, list) or not out[0]:
            return None
        label = out[0].get("label") or out[0].get("label_str") or ""
        if isinstance(label, str):
            label = label.lower().strip()
        shape = LABEL_TO_SHAPE.get(label) or LABEL_TO_SHAPE.get(label.replace(" ", "_"))
        if shape is None:
            return None
        # Wide crops (oval-like): treat model "square" as elongated so bubbles draw as oval
        if is_wide and shape == "square":
            return "elongated"
        return shape
    except Exception:
        return None
