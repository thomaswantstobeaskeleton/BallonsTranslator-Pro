"""
Donut – OCR-free document understanding (NAVER CLOVA, Hugging Face).
Image + task prompt -> text. Use DocVQA checkpoint for "read text" or CORD for receipts.
Requires: pip install transformers torch pillow
Models: naver-clova-ix/donut-base-finetuned-docvqa, donut-base-finetuned-cord-v2, etc.
"""
from typing import List
import numpy as np
import cv2
from .base import OCRBase, register_OCR, DEFAULT_DEVICE, DEVICE_SELECTOR, TextBlock

_DONUT_AVAILABLE = False
try:
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from PIL import Image
    _DONUT_AVAILABLE = True
except ImportError as e:
    import logging
    logging.getLogger("BallonTranslator").debug(
        f"Donut not available: {e}. Install: pip install transformers torch pillow"
    )


def _cv2_to_pil_rgb(img: np.ndarray) -> "Image.Image":
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)


if _DONUT_AVAILABLE:

    @register_OCR("donut")
    class DonutOCR(OCRBase):
        """
        Donut: OCR-free document understanding. Use DocVQA for "read text" or CORD for receipts/forms.
        """
        params = {
            "model_name": {
                "type": "selector",
                "options": [
                    "naver-clova-ix/donut-base-finetuned-docvqa",
                    "naver-clova-ix/donut-base-finetuned-cord-v2",
                    "naver-clova-ix/donut-base",
                ],
                "value": "naver-clova-ix/donut-base-finetuned-docvqa",
                "description": "DocVQA = read text; CORD = receipt/form parsing.",
            },
            "task_prompt": {
                "type": "line_editor",
                "value": "What is the text in this image?",
                "description": "Task prompt (DocVQA: question; CORD: leave default or empty).",
            },
            "device": DEVICE_SELECTOR(),
            "crop_padding": {
                "type": "line_editor",
                "value": 4,
                "description": "Pixels to add around each box when cropping (0–24).",
            },
            "max_length": {
                "type": "line_editor",
                "value": 512,
                "description": "Max generation length.",
            },
            "description": "Donut – OCR-free document understanding (Hugging Face).",
        }
        _load_model_keys = {"processor", "model"}

        def __init__(self, **params) -> None:
            super().__init__(**params)
            self.device = self.params["device"]["value"]
            self.processor = None
            self.model = None
            self._model_name = None

        def _load_model(self):
            model_name = (self.params.get("model_name") or {}).get("value", "naver-clova-ix/donut-base-finetuned-docvqa") or "naver-clova-ix/donut-base-finetuned-docvqa"
            if self.processor is not None and self._model_name == model_name:
                return
            self._model_name = model_name
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForImageTextToText.from_pretrained(model_name)
            self.model.to(self.device)

        def _run_one(self, pil_img: "Image.Image") -> str:
            task_prompt = (self.params.get("task_prompt") or {}).get("value", "What is the text in this image?") or "What is the text in this image?"
            max_len = 512
            ml = self.params.get("max_length", {})
            if isinstance(ml, dict):
                try:
                    max_len = max(64, min(1024, int(ml.get("value", 512))))
                except (TypeError, ValueError):
                    pass
            inputs = self.processor(pil_img, task_prompt, return_tensors="pt")
            inputs = inputs.to(self.device)
            if hasattr(inputs, "input_ids"):
                input_ids = inputs.input_ids
                pixel_values = inputs.pixel_values
            else:
                input_ids = inputs["input_ids"]
                pixel_values = inputs["pixel_values"]
            out = self.model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                max_length=max_len,
            )
            text = self.processor.decode(out[0], skip_special_tokens=True).strip()
            return text

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
                    text = self._run_one(pil_img)
                    blk.text = [text if text else ""]
                except Exception as e:
                    self.logger.warning(f"Donut failed for block: {e}")
                    blk.text = [""]

        def ocr_img(self, img: np.ndarray) -> str:
            pil_img = _cv2_to_pil_rgb(img)
            return self._run_one(pil_img)

        def updateParam(self, param_key: str, param_content):
            super().updateParam(param_key, param_content)
            if param_key == "device":
                self.device = self.params["device"]["value"]
                if self.model is not None:
                    self.model.to(self.device)
            elif param_key == "model_name":
                self.processor = None
                self.model = None
                self._model_name = None
