"""
TrOCR – Transformer-based OCR (Microsoft, Hugging Face).
Best for: printed or handwritten English / Latin text (e.g. Western comics, Latin text in manga).
Requires: pip install transformers torch pillow
Models: microsoft/trocr-base-printed, trocr-small-printed, trocr-base-handwritten.
"""
from typing import List
import numpy as np
import cv2
from .base import OCRBase, register_OCR, DEFAULT_DEVICE, DEVICE_SELECTOR, TextBlock

_TROCR_AVAILABLE = False
try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    from PIL import Image
    _TROCR_AVAILABLE = True
except ImportError:
    import logging
    logging.getLogger("BallonTranslator").debug(
        "TrOCR not available. Install: pip install transformers torch pillow"
    )


def _cv2_to_pil_rgb(img: np.ndarray) -> "Image.Image":
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)


if _TROCR_AVAILABLE:

    @register_OCR("trocr")
    class TrOCROCR(OCRBase):
        """
        TrOCR: transformer-based OCR for printed or handwritten English/Latin.
        Good for Western comics or Latin text in manga. Not for CJK.
        """
        params = {
            "model_type": {
                "type": "selector",
                "options": [
                    "microsoft/trocr-small-printed",
                    "microsoft/trocr-base-printed",
                    "microsoft/trocr-large-printed",
                    "microsoft/trocr-base-handwritten",
                ],
                "value": "microsoft/trocr-base-printed",
                "description": "Printed (documents) or handwritten. Base = good balance.",
            },
            "device": DEVICE_SELECTOR(),
            "crop_padding": {
                "type": "line_editor",
                "value": 4,
                "description": "Pixels to add around each box when cropping (0–24).",
            },
            "description": "TrOCR – printed/handwritten English (Hugging Face).",
        }
        _load_model_keys = {"processor", "model"}

        def __init__(self, **params) -> None:
            super().__init__(**params)
            self.device = self.params["device"]["value"]
            self.processor = None
            self.model = None
            self._model_name = None

        def _load_model(self):
            model_name = self.params.get("model_type", {}).get("value", "microsoft/trocr-base-printed")
            if self.processor is not None and self._model_name == model_name:
                return
            self._model_name = model_name
            self.processor = TrOCRProcessor.from_pretrained(model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
            self.model.to(self.device)

        def _ocr_blk_list(self, img: np.ndarray, blk_list: List[TextBlock], *args, **kwargs) -> None:
            im_h, im_w = img.shape[:2]
            pad = 0
            cp = self.params.get("crop_padding", {})
            if isinstance(cp, dict):
                val = cp.get("value", 0)
            else:
                val = 0
            try:
                pad = max(0, min(24, int(val)))
            except (TypeError, ValueError):
                pass
            for blk in blk_list:
                x1, y1, x2, y2 = blk.xyxy
                x1 = max(0, min(int(round(float(x1))), im_w - 1))
                y1 = max(0, min(int(round(float(y1))), im_h - 1))
                x2 = max(x1 + 1, min(int(round(float(x2))), im_w))
                y2 = max(y1 + 1, min(int(round(float(y2))), im_h))
                if pad > 0:
                    x1 = max(0, x1 - pad)
                    y1 = max(0, y1 - pad)
                    x2 = min(im_w, x2 + pad)
                    y2 = min(im_h, y2 + pad)
                if not (x1 < x2 and y1 < y2 and x2 <= im_w and y2 <= im_h and x1 >= 0 and y1 >= 0):
                    blk.text = [""]
                    continue
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    blk.text = [""]
                    continue
                pil_img = _cv2_to_pil_rgb(crop)
                try:
                    pixel_values = self.processor(pil_img, return_tensors="pt").pixel_values
                    pixel_values = pixel_values.to(self.model.device)
                    generated_ids = self.model.generate(pixel_values)
                    text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                    blk.text = [text if text else ""]
                except Exception as e:
                    self.logger.warning(f"TrOCR failed for block: {e}")
                    blk.text = [""]

        def ocr_img(self, img: np.ndarray) -> str:
            pil_img = _cv2_to_pil_rgb(img)
            pixel_values = self.processor(pil_img, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.model.device)
            generated_ids = self.model.generate(pixel_values)
            return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        def updateParam(self, param_key: str, param_content):
            super().updateParam(param_key, param_content)
            if param_key == "device":
                self.device = self.params["device"]["value"]
                if self.model is not None:
                    self.model.to(self.device)
            elif param_key == "model_type":
                self.processor = None
                self.model = None
                self._model_name = None
