"""
MiniCPM-o – OpenBMB vision-language OCR (Hugging Face). Tops OCRBench for models under 25B.
Vision-only mode (no TTS/audio). Requires: pip install transformers torch pillow
"""
from typing import List
import numpy as np
import cv2
from .base import OCRBase, register_OCR, DEVICE_SELECTOR, TextBlock

_MINICPM_AVAILABLE = False
try:
    from transformers import AutoModel, AutoTokenizer
    from PIL import Image
    import torch
    _MINICPM_AVAILABLE = True
except ImportError as e:
    import logging
    logging.getLogger("BallonTranslator").debug(
        f"MiniCPM-o OCR not available: {e}. Install: pip install transformers torch pillow"
    )


def _cv2_to_pil_rgb(img: np.ndarray) -> "Image.Image":
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)


OCR_PROMPT = "What is the text in this image?"


if _MINICPM_AVAILABLE:

    @register_OCR("minicpm_ocr")
    class MiniCPMOCROCR(OCRBase):
        """
        MiniCPM-o 2.6: strong OCR (OCRBench 897). Vision-only; 8B or int4 for less VRAM.
        Use with any detector.
        """
        params = {
            "model_name": {
                "type": "selector",
                "options": [
                    "openbmb/MiniCPM-o-2_6",
                    "openbmb/MiniCPM-o-2_6-int4",
                ],
                "value": "openbmb/MiniCPM-o-2_6-int4",
                "description": "MiniCPM-o (int4 = ~7GB VRAM, full = higher quality).",
            },
            "device": DEVICE_SELECTOR(),
            "crop_padding": {
                "type": "line_editor",
                "value": 4,
                "description": "Pixels to add around each box when cropping (0–24).",
            },
            "max_new_tokens": {
                "type": "line_editor",
                "value": 256,
                "description": "Max tokens per crop (128–512).",
            },
            "description": "MiniCPM-o (HF). Vision-only OCR; quality over speed.",
        }
        _load_model_keys = {"model", "tokenizer"}

        def __init__(self, **params) -> None:
            super().__init__(**params)
            self.device = (self.params.get("device") or {}).get("value", "cpu")
            self.model = None
            self.tokenizer = None
            self._model_name = None

        def _load_model(self):
            model_name = (self.params.get("model_name") or {}).get("value", "openbmb/MiniCPM-o-2_6-int4") or "openbmb/MiniCPM-o-2_6-int4"
            device = (self.params.get("device") or {}).get("value", "cpu")
            if device in ("cuda", "gpu") and torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
            if self.model is not None and self._model_name == model_name:
                if hasattr(self.model, "to"):
                    self.model.to(device)
                self.device = device
                return
            self._model_name = model_name
            kwargs = {"trust_remote_code": True, "torch_dtype": torch.bfloat16 if device == "cuda" else torch.float32}
            try:
                self.model = AutoModel.from_pretrained(
                    model_name,
                    init_vision=True,
                    init_audio=False,
                    init_tts=False,
                    **kwargs,
                )
            except TypeError:
                self.model = AutoModel.from_pretrained(model_name, **kwargs)
            self.model.to(device)
            self.model.eval()
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.device = device

        def _run_one(self, pil_img: "Image.Image") -> str:
            if pil_img.size[0] == 0 or pil_img.size[1] == 0:
                return ""
            try:
                msgs = [{"role": "user", "content": [pil_img, OCR_PROMPT]}]
                res = self.model.chat(msgs=msgs, tokenizer=self.tokenizer)
                return (res or "").strip()
            except Exception as e:
                self.logger.warning(f"MiniCPM-o OCR failed: {e}")
                return ""

        def _ocr_blk_list(self, img: np.ndarray, blk_list: List[TextBlock], *args, **kwargs) -> None:
            im_h, im_w = img.shape[:2]
            padding = 0
            try:
                p = self.params.get("crop_padding", {})
                padding = max(0, min(24, int(p.get("value", 0) if isinstance(p, dict) else p)))
            except (TypeError, ValueError):
                pass
            for blk in blk_list:
                x1, y1, x2, y2 = blk.xyxy
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(im_w, x2 + padding)
                y2 = min(im_h, y2 + padding)
                if x2 <= x1 or y2 <= y1:
                    blk.text = [""]
                    continue
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    blk.text = [""]
                    continue
                pil_crop = _cv2_to_pil_rgb(crop)
                text = self._run_one(pil_crop)
                blk.text = [text if text else ""]

        def ocr_img(self, img: np.ndarray) -> str:
            if img.ndim == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            pil_img = _cv2_to_pil_rgb(img)
            return self._run_one(pil_img)

        def updateParam(self, param_key: str, param_content):
            super().updateParam(param_key, param_content)
            if param_key in ("model_name", "device"):
                self.model = None
                self.tokenizer = None
                self._model_name = None
