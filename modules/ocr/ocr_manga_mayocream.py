"""
Mayocream Manga OCR — alternative manga OCR using mayocream/manga-ocr on Hugging Face.
Based on kha-white/manga-ocr-base with additional fine-tuning. Auto-downloads via transformers.
Requires: pip install transformers torch
"""
import numpy as np
from typing import List

from .base import register_OCR, OCRBase, TextBlock, DEFAULT_DEVICE, DEVICE_SELECTOR
from utils.logger import logger as _LOGGER

_MAYO_MANGA_OCR_AVAILABLE = False
try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    import torch
    _MAYO_MANGA_OCR_AVAILABLE = True
except ImportError as _e:
    _LOGGER.warning("Mayocream Manga OCR dependencies missing: %s", _e)


HF_REPO_ID = "mayocream/manga-ocr"


@register_OCR("manga_ocr_mayocream")
class MangaOCRMayocream(OCRBase):
    """
    Mayocream fine-tuned Manga OCR. Auto-downloads from Hugging Face.
    Alternative to the base kha-white/manga-ocr-base.
    """
    params = {
        "device": DEVICE_SELECTOR(),
        "crop_padding": {
            "type": "line_editor",
            "value": 6,
            "description": "Pixels to add around each box when cropping for OCR (0–24).",
        },
    }
    device = DEFAULT_DEVICE

    download_file_on_load = True
    _load_model_keys = {"model", "processor"}

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.device = self.params["device"]["value"]
        self.model = None
        self.processor = None

    def _load_model(self):
        if not _MAYO_MANGA_OCR_AVAILABLE:
            raise RuntimeError(
                "Mayocream Manga OCR requires transformers and torch. "
                "Install: pip install transformers torch"
            )
        _LOGGER.info("Mayocream Manga OCR: loading model from %s...", HF_REPO_ID)
        self.processor = TrOCRProcessor.from_pretrained(HF_REPO_ID)
        self.model = VisionEncoderDecoderModel.from_pretrained(HF_REPO_ID)
        self.model.to(self.device)
        self.model.eval()
        _LOGGER.info("Mayocream Manga OCR: model loaded on %s", self.device)

    def ocr_img(self, img: np.ndarray) -> str:
        if self.model is None or self.processor is None:
            return ""
        from PIL import Image
        pil_img = Image.fromarray(img)
        pixel_values = self.processor(pil_img, return_tensors="pt").pixel_values.to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values)
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text

    def _ocr_blk_list(self, img: np.ndarray, blk_list: List[TextBlock], *args, **kwargs):
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
                x1a = max(0, x1 - pad)
                y1a = max(0, y1 - pad)
                x2a = min(im_w, x2 + pad)
                y2a = min(im_h, y2 + pad)
                if getattr(blk, "src_is_vertical", False):
                    extra_bottom = max(pad, min(int((y2 - y1) * 0.12), 24))
                    y2a = min(im_h, y2a + extra_bottom)
                x1, y1, x2, y2 = x1a, y1a, x2a, y2a
            if y2 <= im_h and x2 <= im_w and x1 >= 0 and y1 >= 0 and x1 < x2 and y1 < y2:
                region = img[y1:y2, x1:x2]
                blk.text = [self.ocr_img(region)]
            else:
                self.logger.warning("invalid textbbox to target img")
                blk.text = [""]

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        device = self.params["device"]["value"]
        if self.device != device and self.model is not None:
            self.device = device
            self.model.to(device)
