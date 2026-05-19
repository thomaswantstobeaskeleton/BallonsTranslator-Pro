from __future__ import annotations

import numpy as np


def _dilate_binary(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    m = (mask > 0).astype(np.uint8)
    out = m.copy()
    for _ in range(int(radius)):
        up = np.pad(out[:-1, :], ((1, 0), (0, 0)), constant_values=0)
        dn = np.pad(out[1:, :], ((0, 1), (0, 0)), constant_values=0)
        lf = np.pad(out[:, :-1], ((0, 0), (1, 0)), constant_values=0)
        rg = np.pad(out[:, 1:], ((0, 0), (0, 1)), constant_values=0)
        out = np.maximum.reduce([out, up, dn, lf, rg])
    return (out * 255).astype(np.uint8)


def adaptive_mask_expand(mask: np.ndarray, *, inside_radius: int = 1, outside_radius: int = 2) -> np.ndarray:
    """Simple bubble-aware proxy: mild inside dilation + stronger outside edge dilation.

    Without explicit bubble contour, this approximates quality cleanup by combining two
    dilation passes and keeping the larger coverage edge ring from outside_radius.
    """
    base = (np.asarray(mask) > 0).astype(np.uint8) * 255
    in_d = _dilate_binary(base, int(max(0, inside_radius)))
    out_d = _dilate_binary(base, int(max(0, outside_radius)))
    ring = ((out_d > 0) & (in_d == 0)).astype(np.uint8) * 255
    merged = np.maximum(in_d, ring)
    return merged.astype(np.uint8)


def merge_masks_with_confidence(primary: np.ndarray, secondary: np.ndarray | None = None, *, confidence: float = 1.0) -> np.ndarray:
    p = (np.asarray(primary) > 0).astype(np.uint8) * 255
    if secondary is None:
        return p
    s = (np.asarray(secondary) > 0).astype(np.uint8) * 255
    c = float(max(0.0, min(1.0, confidence)))
    if c >= 0.75:
        return np.maximum(p, s)
    if c <= 0.25:
        return p
    # medium confidence: only keep overlap + core
    overlap = ((p > 0) & (s > 0)).astype(np.uint8) * 255
    return np.maximum(p, overlap)
