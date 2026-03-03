"""
Image upscaling and pipeline scaling helpers (Section 6 & 6.1: quality + OCR assist, per-stage resizing).

- Lanczos upscale for pre-OCR crops, initial page upscale, and final output.
- processing_scale from image area for consistent params across resolutions.
- Per-stage policy: none, lanczos (model/model_lite reserved for future).
"""
from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np


# Interpolation for quality upscale (Lanczos is good for 2x; CUBIC is fallback)
def _lanczos_interp():
    try:
        return cv2.INTER_LANCZOS4
    except AttributeError:
        return cv2.INTER_CUBIC


def upscale_lanczos(img: np.ndarray, factor: float) -> np.ndarray:
    """Upscale image by factor using Lanczos (or CUBIC). factor > 1."""
    if img is None or img.size == 0 or factor <= 1.0:
        return img
    h, w = img.shape[:2]
    nw = max(2, int(round(w * factor)))
    nh = max(2, int(round(h * factor)))
    return cv2.resize(img, (nw, nh), interpolation=_lanczos_interp())


def downscale_to_size(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Downscale image to exact target size. Uses INTER_AREA for downscaling."""
    if img is None or img.size == 0:
        return img
    h, w = img.shape[:2]
    if w == target_w and h == target_h:
        return img
    return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)


def processing_scale(
    width: int,
    height: int,
    ref_area: float = 1_000_000.0,
    min_scale: float = 0.5,
    max_scale: float = 2.0,
) -> float:
    """
    Compute scale factor from image area so pipeline params behave consistently
    across low-res vs high-res pages. Used to scale fonts, padding, morphology, area thresholds.

    processing_scale = sqrt((w * h) / ref_area), clamped to [min_scale, max_scale].
    ref_area 1e6 => 1Mpx reference; scale 1.0 at ~1000x1000.
    """
    if width <= 0 or height <= 0:
        return 1.0
    area = width * height
    s = math.sqrt(area / ref_area)
    return float(np.clip(s, min_scale, max_scale))


def resize_with_policy(
    img: np.ndarray,
    policy: str,
    factor: Optional[float] = None,
    target_long_side: Optional[int] = None,
) -> np.ndarray:
    """
    Resize image by policy: 'none' (return as-is), 'lanczos' (upscale/downscale by factor or to target_long_side).
    'model' / 'model_lite' reserved for future AI upscaler; currently fall back to lanczos.
    """
    if img is None or img.size == 0:
        return img
    policy = (policy or "none").strip().lower()
    if policy == "none" or (factor is None and target_long_side is None):
        return img

    h, w = img.shape[:2]
    long_side = max(h, w)

    if target_long_side is not None and target_long_side > 0 and long_side != target_long_side:
        factor = target_long_side / long_side
    if factor is None or factor == 1.0:
        return img

    nw = max(2, int(round(w * factor)))
    nh = max(2, int(round(h * factor)))
    if factor > 1:
        interp = _lanczos_interp()
    else:
        interp = cv2.INTER_AREA
    return cv2.resize(img, (nw, nh), interpolation=interp)


def apply_upscale_final(img: np.ndarray, factor: float, policy: str = "lanczos") -> np.ndarray:
    """Apply final output upscale (e.g. 2x). policy: lanczos (or none to skip)."""
    if factor is None or factor <= 1.0 or policy == "none":
        return img
    return resize_with_policy(img, policy, factor=factor)


def apply_initial_upscale(img: np.ndarray, factor: float, policy: str = "lanczos") -> np.ndarray:
    """Apply initial page upscale before detection/OCR. Returns upscaled image."""
    if factor is None or factor <= 1.0 or policy == "none":
        return img
    return resize_with_policy(img, policy, factor=factor)
