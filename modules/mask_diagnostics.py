import cv2
import numpy as np


def build_mask_diagnostics(mask: np.ndarray, threshold: int = 127, dilate_iter: int = 1):
    if mask is None:
        return None
    if len(mask.shape) == 3:
        src = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        src = mask.copy()
    th = max(0, min(255, int(threshold)))
    otsu_th, otsu_mask = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, thresh = cv2.threshold(src, th, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=max(0, int(dilate_iter)))
    edges = cv2.Canny(src, 50, 150)
    ring = cv2.morphologyEx(dilated, cv2.MORPH_GRADIENT, kernel)
    halo = cv2.bitwise_and(edges, ring)
    halo_ratio = float(np.count_nonzero(halo)) / float(max(1, np.count_nonzero(ring)))
    return {
        "raw": src,
        "thresholded": thresh,
        "thresholded_otsu": otsu_mask,
        "dilated": dilated,
        "edge_halo": halo,
        "stats": {
            "otsu_threshold": int(round(float(otsu_th))),
            "manual_threshold": th,
            "mask_fill_ratio": float(np.count_nonzero(dilated)) / float(dilated.size),
            "edge_halo_ratio": halo_ratio,
        },
    }
