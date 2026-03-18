"""
Optional model-based check: is the text box (from detection) too large or too small for the bubble?
Uses a Hugging Face image-classification model on the bubble crop. Model should output one of:
  too_large / too large / oversized / large -> box is too big, caller will shrink to bubble
  too_small / too small / undersized / small -> box is too small, caller may expand
  ok / good / fit -> no change
When model_id is "builtin", uses zero-shot CLIP (no custom training). When empty or model fails, returns None.
"""
from typing import Optional, List
import numpy as np

_box_size_check_cache: Optional[tuple] = None  # (model_id, pipeline or "zero_shot" pipe)
_BUILTIN_ZERO_SHOT_LABELS = [
    "text box too large for the speech bubble",
    "text box too small for the speech bubble",
    "text box fits well in the speech bubble",
]
_BUILTIN_LABEL_TO_RESULT = ("too_large", "too_small", "ok")

# Map common HF labels to our result
LABEL_TO_RESULT = {
    "too_large": "too_large",
    "too large": "too_large",
    "oversized": "too_large",
    "large": "too_large",
    "too_small": "too_small",
    "too small": "too_small",
    "undersized": "too_small",
    "small": "too_small",
    "ok": "ok",
    "good": "ok",
    "fit": "ok",
    "correct": "ok",
}


def check_box_size_from_model(
    img: np.ndarray,
    mask_xyxy: List[int],
    model_id: str,
) -> Optional[str]:
    """
    Run optional image-classification model on bubble crop to check if text box is too large/small.
    Returns "too_large", "too_small", "ok", or None on failure/disabled.
    When model_id is "builtin", uses zero-shot CLIP (openai/clip-vit-base-patch32).
    """
    raw_id = (model_id or "").strip()
    if not raw_id or not img.size or len(mask_xyxy) < 4:
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
    if crop.ndim == 2:
        pil_img = Image.fromarray(crop).convert("RGB")
    else:
        if crop.shape[2] == 4:
            crop = crop[:, :, :3]
        pil_img = Image.fromarray(crop).convert("RGB")

    use_builtin = raw_id.lower() == "builtin"
    global _box_size_check_cache
    if _box_size_check_cache is None or _box_size_check_cache[0] != raw_id:
        try:
            if use_builtin:
                pipe = hf_pipeline(
                    "zero-shot-image-classification",
                    model="openai/clip-vit-base-patch32",
                )
            else:
                pipe = hf_pipeline("image-classification", model=raw_id)
            _box_size_check_cache = (raw_id, pipe)
        except Exception:
            return None
    _, pipe = _box_size_check_cache
    try:
        if use_builtin:
            out = pipe(pil_img, candidate_labels=_BUILTIN_ZERO_SHOT_LABELS)
            if not out or not isinstance(out, list) or not out[0]:
                return None
            top_label = out[0].get("label") or ""
            try:
                idx = _BUILTIN_ZERO_SHOT_LABELS.index(top_label)
                return _BUILTIN_LABEL_TO_RESULT[idx]
            except (ValueError, IndexError):
                return None
        out = pipe(pil_img, top_k=1)
        if not out or not isinstance(out, list) or not out[0]:
            return None
        label = out[0].get("label") or out[0].get("label_str") or ""
        if isinstance(label, str):
            label = label.lower().strip().replace("-", " ")
        return LABEL_TO_RESULT.get(label) or LABEL_TO_RESULT.get(label.replace(" ", "_")) or None
    except Exception:
        return None
