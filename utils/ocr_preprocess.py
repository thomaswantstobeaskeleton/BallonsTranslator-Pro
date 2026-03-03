from typing import Optional

import cv2
import numpy as np


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


def preprocess_for_ocr(
    crop_rgb: np.ndarray,
    recipe: str = "none",
    upscale_min_side: int = 0,
    clahe_clip_limit: float = 2.0,
    clahe_grid: int = 8,
) -> np.ndarray:
    """
    Lightweight OCR preprocessing "recipes" for low-contrast / noisy crops.

    Returns an image compatible with common OCR engines (RGB or grayscale).
    """
    if crop_rgb is None or crop_rgb.size == 0:
        return crop_rgb

    img = crop_rgb
    h, w = img.shape[:2]
    if upscale_min_side and max(h, w) > 0 and max(h, w) < int(upscale_min_side):
        scale = float(upscale_min_side) / float(max(h, w))
        nw = max(2, int(round(w * scale)))
        nh = max(2, int(round(h * scale)))
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC)

    r = (recipe or "none").strip().lower()
    if r in {"none", "off", ""}:
        return img

    if r == "clahe":
        g = _to_gray(img)
        grid = max(2, min(32, int(clahe_grid)))
        clip = max(0.5, min(8.0, float(clahe_clip_limit)))
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
        g2 = clahe.apply(g)
        return g2

    if r == "clahe+sharpen":
        g = preprocess_for_ocr(img, recipe="clahe", upscale_min_side=0, clahe_clip_limit=clahe_clip_limit, clahe_grid=clahe_grid)
        if g is None or g.size == 0:
            return g
        # unsharp mask
        blur = cv2.GaussianBlur(g, (0, 0), 1.2)
        sharp = cv2.addWeighted(g, 1.6, blur, -0.6, 0)
        return sharp

    if r == "otsu":
        g = _to_gray(img)
        g = cv2.GaussianBlur(g, (3, 3), 0)
        _t, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th

    if r == "adaptive":
        g = _to_gray(img)
        g = cv2.GaussianBlur(g, (3, 3), 0)
        # blockSize must be odd and >=3
        block = 31 if min(g.shape[:2]) >= 64 else 15
        th = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, 5)
        return th

    if r == "denoise":
        # keep RGB; good for some recognizers
        if img.ndim == 2:
            return cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
        if img.ndim == 3:
            return cv2.fastNlMeansDenoisingColored(img, None, 7, 7, 7, 21)
        return img

    return img

