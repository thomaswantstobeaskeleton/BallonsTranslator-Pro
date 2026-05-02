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
    _, thresh = cv2.threshold(src, th, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=max(0, int(dilate_iter)))
    return {"raw": src, "thresholded": thresh, "dilated": dilated}
