"""
EasyOCR recognition on crops – use EasyOCR recognizer for per-block text.
Pairs with easyocr_det for full EasyOCR pipeline, or use with any detector.
Requires: pip install easyocr (PyTorch).
"""
from typing import List
import numpy as np
import cv2

from .base import OCRBase, register_OCR, DEVICE_SELECTOR, TextBlock
from utils.ocr_preprocess import preprocess_for_ocr

_EASYOCR_AVAILABLE = False
try:
    import easyocr
    _EASYOCR_AVAILABLE = True
except ImportError:
    import logging
    logging.getLogger("BallonTranslator").debug(
        "EasyOCR not available for OCR. Install: pip install easyocr"
    )


if _EASYOCR_AVAILABLE:

    @register_OCR("easyocr_ocr")
    class EasyOCROCR(OCRBase):
        """
        EasyOCR recognition on detector crops. Good for Chinese, English, ja, ko.
        Use with easyocr_det for full EasyOCR pipeline, or any other detector.
        """
        params = {
            "language": {
                "type": "selector",
                "options": ["ch_sim", "ch_sim+en", "en", "ja", "ko"],
                "value": "ch_sim+en",
                "description": "Language(s): ch_sim, ch_sim+en, en, ja, ko.",
            },
            "gpu": {
                "type": "checkbox",
                "value": True,
                "description": "Use GPU if available.",
            },
            "crop_padding": {
                "type": "line_editor",
                "value": 4,
                "description": "Pixels to add around each box when cropping (0–24).",
            },
            "preprocess_recipe": {
                "type": "selector",
                "options": ["none", "clahe", "clahe+sharpen", "otsu", "adaptive", "denoise"],
                "value": "none",
                "description": "Optional preprocessing for hard OCR cases (contrast/noise).",
            },
            "upscale_min_side": {
                "type": "line_editor",
                "value": 0,
                "description": "If >0, upscale crop so its longer side >= this (e.g. 512). Helps tiny text. 0 = off.",
            },
            "description": "EasyOCR recognition on crops (pair with easyocr_det or any detector).",
        }
        _load_model_keys = {"reader"}

        def __init__(self, **params) -> None:
            super().__init__(**params)
            self.reader = None

        def _load_model(self):
            if self.reader is not None:
                return
            lang_val = self.params["language"]["value"]
            if lang_val == "ch_sim+en":
                lang_list = ["ch_sim", "en"]
            else:
                lang_list = [lang_val]
            gpu = self.params["gpu"]["value"]
            self.reader = easyocr.Reader(
                lang_list,
                gpu=gpu,
                detector=True,
                recognizer=True,
                verbose=False,
            )

        def _ocr_blk_list(self, img: np.ndarray, blk_list: List[TextBlock], *args, **kwargs) -> None:
            if self.reader is None:
                return
            im_h, im_w = img.shape[:2]
            if img.ndim == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            pad = max(0, min(24, int(self.params.get("crop_padding", {}).get("value", 4))))
            recipe = (self.params.get("preprocess_recipe", {}) or {}).get("value", "none") or "none"
            try:
                upscale_min_side = int((self.params.get("upscale_min_side", {}) or {}).get("value", 0) or 0)
            except (TypeError, ValueError):
                upscale_min_side = 0
            if upscale_min_side <= 0:
                try:
                    from utils.config import pcfg
                    upscale_min_side = int(getattr(pcfg.module, "ocr_upscale_min_side", 0) or 0)
                except Exception:
                    pass
            for blk in blk_list:
                x1, y1, x2, y2 = blk.xyxy
                if pad > 0:
                    x1 = max(0, x1 - pad)
                    y1 = max(0, y1 - pad)
                    x2 = min(im_w, x2 + pad)
                    y2 = min(im_h, y2 + pad)
                if not (0 <= x1 < x2 <= im_w and 0 <= y1 < y2 <= im_h):
                    blk.text = [""]
                    continue
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    blk.text = [""]
                    continue
                crop = preprocess_for_ocr(crop, recipe=recipe, upscale_min_side=upscale_min_side)
                try:
                    results = self.reader.readtext(crop)
                except Exception as e:
                    self.logger.debug(f"EasyOCR readtext failed on crop: {e}")
                    blk.text = [""]
                    continue
                parts = []
                for (_bbox, text, _conf) in (results or []):
                    if text and isinstance(text, str):
                        parts.append(text.strip())
                blk.text = ["\n".join(parts)] if parts else [""]

        def ocr_img(self, img: np.ndarray) -> str:
            if self.reader is None:
                return ""
            if img.ndim == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            try:
                results = self.reader.readtext(img)
            except Exception as e:
                self.logger.debug(f"EasyOCR readtext failed: {e}")
                return ""
            parts = []
            for (_bbox, text, _conf) in (results or []):
                if text and isinstance(text, str):
                    parts.append(text.strip())
            return "\n".join(parts)
