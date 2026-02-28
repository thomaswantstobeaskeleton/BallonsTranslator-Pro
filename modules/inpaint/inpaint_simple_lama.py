"""
Optional inpainter using the LaMa model via the simple-lama-inpainting or simple-lama pip package.
Install separately: pip install simple-lama-inpainting  (or  pip install simple-lama)
Best for: high-quality text removal when you want an alternative to built-in lama_mpe / lama_large_512px.
"""
import numpy as np
import cv2
from typing import List

from .base import InpainterBase, register_inpainter, TextBlock

_SIMPLE_LAMA_AVAILABLE = False
_SimpleLama = None
try:
    from simple_lama_inpainting import SimpleLama
    _SimpleLama = SimpleLama
    _SIMPLE_LAMA_AVAILABLE = True
except ImportError:
    try:
        from simple_lama import SimpleLama
        _SimpleLama = SimpleLama
        _SIMPLE_LAMA_AVAILABLE = True
    except ImportError:
        import logging
        logging.getLogger("BallonTranslator").debug(
            "Simple LaMa not available. Install: pip install simple-lama-inpainting  or  pip install simple-lama"
        )


if _SIMPLE_LAMA_AVAILABLE:

    @register_inpainter("simple_lama")
    class SimpleLamaInpainter(InpainterBase):
        """
        LaMa inpainting via simple-lama-inpainting or simple-lama pip package.
        Good alternative to built-in lama_mpe; install the package separately.
        """
        inpaint_by_block = False  # full-image inpainting
        check_need_inpaint = True

        params = {
            "description": "LaMa via pip (simple-lama-inpainting or simple-lama). Install the package first.",
        }
        _load_model_keys = {"model"}

        def __init__(self, **params) -> None:
            super().__init__(**params)
            self.model = None

        def _load_model(self):
            if self.model is None:
                self.model = _SimpleLama()

        def _inpaint(self, img: np.ndarray, mask: np.ndarray, textblock_list: List[TextBlock] = None) -> np.ndarray:
            if img.ndim == 3 and img.shape[2] == 4:
                img = img[:, :, :3]
            if mask.ndim == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask_bin = (mask > 127).astype(np.uint8) * 255
            try:
                from PIL import Image
                pil_img = Image.fromarray(img)
                pil_mask = Image.fromarray(mask_bin).convert("L")
                result = self.model(pil_img, pil_mask)
                if hasattr(result, "save"):
                    out = np.array(result)
                else:
                    out = result
                if out.ndim == 2:
                    out = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)
                return out.astype(np.uint8)
            except Exception as e:
                self.logger.error(f"Simple LaMa inpainting failed: {e}")
                return img.copy()
