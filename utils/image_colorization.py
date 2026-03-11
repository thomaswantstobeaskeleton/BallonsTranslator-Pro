import numpy as np
import cv2


def _is_grayscale_like(img: np.ndarray, atol: int = 2) -> bool:
    """Return True if a 3/4-channel RGB(A) image is effectively grayscale."""
    if img.ndim != 3 or img.shape[2] < 3:
        return False
    rgb = img[..., :3]
    c0, c1, c2 = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return (
        np.allclose(c0, c1, atol=atol)
        and np.allclose(c1, c2, atol=atol)
    )


def apply_colorization(img: np.ndarray, strength: float = 0.6) -> np.ndarray:
    """
    Lightweight page colorization for grayscale manga pages.

    - If the image is grayscale (2D) or RGB(A) but effectively grayscale, apply a soft colormap.
    - If the image is already colored, return it unchanged.
    - strength in [0,1]: 0 = no effect, 1 = full colormap.
    """
    if img is None:
        return img
    if not isinstance(img, np.ndarray):
        return img
    if strength <= 0.0:
        return img
    strength = float(max(0.0, min(1.0, strength)))

    arr = img
    has_alpha = arr.ndim == 3 and arr.shape[2] == 4

    if arr.ndim == 2:
        gray = arr
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        if not _is_grayscale_like(arr):
            return img
        gray = cv2.cvtColor(arr[..., :3], cv2.COLOR_RGB2GRAY)
    else:
        return img

    # Use a relatively gentle colormap to avoid neon artifacts.
    color_bgr = cv2.applyColorMap(gray, cv2.COLORMAP_TWILIGHT)
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

    # Blend between original (grayscale as RGB) and colorized version.
    base_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    blended_rgb = cv2.addWeighted(color_rgb, strength, base_rgb, 1.0 - strength, 0.0)

    if has_alpha:
        alpha = arr[..., 3:4]
        blended = np.concatenate([blended_rgb, alpha], axis=-1)
    else:
        blended = blended_rgb

    return blended

