from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from utils.text_masking import bubble_safe_text_rect
except Exception:  # pragma: no cover
    bubble_safe_text_rect = None

MaskRunner = Callable[[np.ndarray, str], Sequence[np.ndarray]]
BoxRunner = Callable[[np.ndarray, str], Sequence[Sequence[float]]]

SAM3_REFINEMENT_SETTINGS_KEY = "sam3_refinement"
SAM3_REFINEMENT_MODEL_DEFAULT = "facebook/sam3"
SAM3_REFINEMENT_MODEL_CUSTOM = "Custom SAM 3 model id"
SAM3_REFINEMENT_MODEL_OPTIONS: Tuple[str, ...] = (
    SAM3_REFINEMENT_MODEL_DEFAULT,
    SAM3_REFINEMENT_MODEL_CUSTOM,
)
SAM3_REFINEMENT_BASE_DETECTOR_OPTIONS: Tuple[str, ...] = (
    "ctd",
    "hf_object_det",
    "ysgyolo",
    "paddle_det",
    "paddle_det_v5",
    "easyocr_det",
    "craft_det",
)


def curated_sam3_refinement_models() -> Tuple[str, ...]:
    """Model choices shown in UI for SAM-based refinement.

    Keep this intentionally short: these are only models that should help mask
    refinement.  General OCR/detector/LLM models are deliberately excluded.
    """
    return SAM3_REFINEMENT_MODEL_OPTIONS


def resolve_refinement_model_id(choice: str, custom_model_id: str = "") -> str:
    choice = (choice or SAM3_REFINEMENT_MODEL_DEFAULT).strip()
    if choice == SAM3_REFINEMENT_MODEL_CUSTOM:
        return (custom_model_id or "").strip() or SAM3_REFINEMENT_MODEL_DEFAULT
    if choice not in SAM3_REFINEMENT_MODEL_OPTIONS:
        return SAM3_REFINEMENT_MODEL_DEFAULT
    return choice


def default_refinement_settings() -> Dict[str, Any]:
    return {
        "enabled": False,
        "base_detector": "ctd",
        "refinement_model": SAM3_REFINEMENT_MODEL_DEFAULT,
        "custom_model_id": "",
        "prompt": "speech bubble",
        "fallback_prompt": "text",
        "merge_mode": "union",
        "min_mask_area": 64,
        "max_mask_area_ratio": 0.75,
        "min_iou_with_initial": 0.02,
        "expand_px": 0,
        "safe_rect_min_coverage": 0.88,
        "attach_safe_areas": True,
    }


def get_persistent_refinement_settings(module_cfg: Any) -> Dict[str, Any]:
    """Read persistent settings from pcfg.module.textdetector_params.

    This avoids adding another custom config save path: settings live beside the
    normal detector params and round-trip through existing config serialization.
    """
    params = getattr(module_cfg, "textdetector_params", None) or {}
    stored = params.get(SAM3_REFINEMENT_SETTINGS_KEY, {}) or {}
    out = default_refinement_settings()
    if isinstance(stored, dict):
        for key in out:
            val = stored.get(key, out[key])
            if isinstance(val, dict) and "value" in val:
                val = val["value"]
            out[key] = val
    return out


@dataclass(frozen=True)
class SAM3RefinementOptions:
    enabled: bool = False
    prompt: str = "speech bubble"
    fallback_prompt: str = "text"
    merge_mode: str = "union"  # union | intersect | replace_if_nonempty
    min_mask_area: int = 64
    max_mask_area_ratio: float = 0.75
    min_iou_with_initial: float = 0.02
    expand_px: int = 0
    safe_rect_min_coverage: float = 0.88

    @classmethod
    def from_settings(cls, settings: Mapping[str, Any]) -> "SAM3RefinementOptions":
        return cls(
            enabled=bool(settings.get("enabled", False)),
            prompt=str(settings.get("prompt", "speech bubble") or "speech bubble"),
            fallback_prompt=str(settings.get("fallback_prompt", "text") or "text"),
            merge_mode=str(settings.get("merge_mode", "union") or "union"),
            min_mask_area=int(settings.get("min_mask_area", 64) or 64),
            max_mask_area_ratio=float(settings.get("max_mask_area_ratio", 0.75) or 0.75),
            min_iou_with_initial=float(settings.get("min_iou_with_initial", 0.02) or 0.02),
            expand_px=int(settings.get("expand_px", 0) or 0),
            safe_rect_min_coverage=float(settings.get("safe_rect_min_coverage", 0.88) or 0.88),
        )


@dataclass(frozen=True)
class SAM3RefinementResult:
    mask: np.ndarray
    used_sam3: bool
    prompt: str
    merge_mode: str
    warnings: Tuple[str, ...] = tuple()
    candidate_count: int = 0
    selected_candidate_count: int = 0
    safe_areas: Tuple[Dict[str, Any], ...] = tuple()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "used_sam3": self.used_sam3,
            "prompt": self.prompt,
            "merge_mode": self.merge_mode,
            "warnings": list(self.warnings),
            "candidate_count": self.candidate_count,
            "selected_candidate_count": self.selected_candidate_count,
            "safe_areas": list(self.safe_areas),
        }


def as_binary_mask(mask: np.ndarray | None, shape: Tuple[int, int] | None = None) -> np.ndarray:
    if mask is None:
        return np.zeros(shape[:2], dtype=np.uint8) if shape else np.zeros((0, 0), dtype=np.uint8)
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr = (arr > 0).astype(np.uint8) * 255
    if shape is None or arr.shape[:2] == tuple(shape[:2]):
        return np.ascontiguousarray(arr)
    out = np.zeros(tuple(shape[:2]), dtype=np.uint8)
    h = min(out.shape[0], arr.shape[0])
    w = min(out.shape[1], arr.shape[1])
    if h > 0 and w > 0:
        out[:h, :w] = arr[:h, :w]
    return out


def mask_area(mask: np.ndarray | None) -> int:
    return int(np.count_nonzero(as_binary_mask(mask) > 0))


def mask_iou(a: np.ndarray | None, b: np.ndarray | None) -> float:
    aa = as_binary_mask(a)
    bb = as_binary_mask(b, aa.shape if aa.size else None)
    if aa.size == 0 or bb.size == 0:
        return 0.0
    inter = np.logical_and(aa > 0, bb > 0).sum()
    union = np.logical_or(aa > 0, bb > 0).sum()
    return float(inter) / float(union) if union else 0.0


def dilate_mask_numpy(mask: np.ndarray, radius: int) -> np.ndarray:
    arr = as_binary_mask(mask)
    r = max(0, int(radius or 0))
    if r <= 0 or arr.size == 0:
        return arr
    padded = np.pad(arr > 0, ((r, r), (r, r)), mode="constant", constant_values=False)
    out = np.zeros(arr.shape, dtype=bool)
    for dy in range(0, 2 * r + 1):
        for dx in range(0, 2 * r + 1):
            out |= padded[dy:dy + arr.shape[0], dx:dx + arr.shape[1]]
    return out.astype(np.uint8) * 255


def _masks_from_boxes(boxes: Sequence[Sequence[float]], shape: Tuple[int, int]) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    h, w = shape[:2]
    for box in boxes or []:
        if len(box) < 4:
            continue
        x1, y1, x2, y2 = [int(round(float(v))) for v in box[:4]]
        x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
        y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            continue
        m = np.zeros((h, w), dtype=np.uint8)
        m[y1:y2, x1:x2] = 255
        out.append(m)
    return out


def filter_sam3_candidates(initial_mask: np.ndarray, candidates: Sequence[np.ndarray], opts: SAM3RefinementOptions) -> Tuple[List[np.ndarray], List[str]]:
    base = as_binary_mask(initial_mask)
    page_area = max(1, base.shape[0] * base.shape[1]) if base.size else 1
    selected: List[np.ndarray] = []
    warnings: List[str] = []
    for cand in candidates or []:
        cm = as_binary_mask(cand, base.shape if base.size else None)
        area = mask_area(cm)
        if area < int(opts.min_mask_area):
            continue
        if opts.max_mask_area_ratio > 0 and area > float(opts.max_mask_area_ratio) * page_area:
            warnings.append("ignored oversized SAM3 candidate")
            continue
        if mask_area(base) > 0 and mask_iou(base, cm) < float(opts.min_iou_with_initial):
            continue
        selected.append(cm)
    return selected, warnings


def combine_refined_mask(initial_mask: np.ndarray, refined: Sequence[np.ndarray], opts: SAM3RefinementOptions) -> np.ndarray:
    base = as_binary_mask(initial_mask)
    if not refined:
        return base
    stack = np.zeros_like(base)
    for cm in refined:
        stack = np.maximum(stack, as_binary_mask(cm, base.shape))
    mode = (opts.merge_mode or "union").strip().lower()
    if mode == "intersect":
        merged = np.where(np.logical_and(base > 0, stack > 0), 255, 0).astype(np.uint8)
    elif mode == "replace_if_nonempty":
        merged = stack if mask_area(stack) > 0 else base
    else:
        merged = np.maximum(base, stack)
    return dilate_mask_numpy(merged, opts.expand_px) if int(opts.expand_px or 0) > 0 else merged


def refine_mask_with_sam3(image: np.ndarray, initial_mask: np.ndarray, opts: SAM3RefinementOptions, *, mask_runner: Optional[MaskRunner] = None, box_runner: Optional[BoxRunner] = None) -> SAM3RefinementResult:
    base = as_binary_mask(initial_mask, np.asarray(image).shape[:2])
    if not opts.enabled:
        return SAM3RefinementResult(base, False, opts.prompt, opts.merge_mode)
    if mask_runner is None and box_runner is None:
        return SAM3RefinementResult(base, False, opts.prompt, opts.merge_mode, ("SAM3 enabled but no runner was provided",))
    prompt = opts.prompt or "speech bubble"
    raw: List[np.ndarray] = []
    warnings: List[str] = []
    try:
        if mask_runner is not None:
            raw.extend(as_binary_mask(m, base.shape) for m in (mask_runner(image, prompt) or []))
        if box_runner is not None:
            raw.extend(_masks_from_boxes(box_runner(image, prompt) or [], base.shape))
    except Exception as exc:
        warnings.append(f"SAM3 refinement failed: {exc}")
    if not raw and opts.fallback_prompt and opts.fallback_prompt != prompt:
        prompt = opts.fallback_prompt
        try:
            if mask_runner is not None:
                raw.extend(as_binary_mask(m, base.shape) for m in (mask_runner(image, prompt) or []))
            if box_runner is not None:
                raw.extend(_masks_from_boxes(box_runner(image, prompt) or [], base.shape))
        except Exception as exc:
            warnings.append(f"SAM3 fallback failed: {exc}")
    selected, filter_warnings = filter_sam3_candidates(base, raw, opts)
    warnings.extend(filter_warnings)
    merged = combine_refined_mask(base, selected, opts)
    safe_areas = tuple(sam3_safe_areas_from_mask(merged, opts)) if selected else tuple()
    return SAM3RefinementResult(merged, bool(selected), prompt, opts.merge_mode, tuple(warnings), len(raw), len(selected), safe_areas)


def sam3_safe_areas_from_mask(mask: np.ndarray, opts: SAM3RefinementOptions | None = None) -> List[Dict[str, Any]]:
    arr = as_binary_mask(mask)
    if arr.size == 0 or not np.any(arr > 0):
        return []
    ys, xs = np.where(arr > 0)
    rect = [float(xs.min()), float(ys.min()), float(xs.max() + 1 - xs.min()), float(ys.max() + 1 - ys.min())]
    if bubble_safe_text_rect is not None:
        safe = bubble_safe_text_rect(arr, rect, min_coverage=float((opts.safe_rect_min_coverage if opts else 0.88) or 0.88))
    else:
        safe = {"rect": rect, "coverage": 1.0, "edge_coverage": 1.0, "used_mask": False}
    return [{"kind": "sam3_bubble_safe_area", "bbox": rect, "safe_rect": list(safe.get("rect", rect)), "coverage": float(safe.get("coverage", 0.0)), "edge_coverage": float(safe.get("edge_coverage", 0.0)), "used_mask": bool(safe.get("used_mask", False))}]


def should_use_sam3_for_live_translation(profile: str | Mapping[str, Any] | None) -> bool:
    if profile is None:
        return False
    if isinstance(profile, Mapping):
        if bool(profile.get("use_sam3")):
            quality = str(profile.get("quality_mode", profile.get("mode", ""))).lower()
            return quality in {"high_quality", "quality", "slower", "slow", "hq"}
        profile = str(profile.get("name", profile.get("preset", profile.get("mode", ""))))
    text = str(profile or "").strip().lower().replace("-", "_").replace(" ", "_")
    return any(t in text for t in ("high_quality", "hq", "slower", "slow")) and "fast" not in text


def cleanup_only_mode_plan(detector: str = "sam3_refiner", sam_prompt: str = "speech bubble", inpainter: str = "lama_manga_onnx", export_clean_raws: bool = True) -> Dict[str, Any]:
    return {"mode": "cleanup_only", "run_detection": True, "run_ocr": False, "run_translation": False, "run_inpaint": True, "run_render": False, "detector": detector, "sam_prompt": sam_prompt, "inpainter": inpainter, "export_clean_raws": bool(export_clean_raws)}
