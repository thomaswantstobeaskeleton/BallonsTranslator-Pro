"""
Florence-2 – Microsoft vision foundation model for OCR (Hugging Face).
Prompt-based OCR; supports <OCR> for text extraction. Good quality/speed on GPU.
Requires: pip install transformers torch pillow
"""
from typing import List
import numpy as np
import cv2
from .base import OCRBase, register_OCR, DEVICE_SELECTOR, TextBlock

_FLORENCE2_AVAILABLE = False
try:
    from transformers import AutoProcessor, AutoModelForCausalLM
    from PIL import Image
    import torch
    _FLORENCE2_AVAILABLE = True
except ImportError as e:
    import logging
    logging.getLogger("BallonTranslator").debug(
        f"Florence-2 OCR not available: {e}. Install: pip install transformers torch pillow"
    )


FLORENCE2_OCR_TASK = "<OCR>"


def _cv2_to_pil_rgb(img: np.ndarray) -> "Image.Image":
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)


if _FLORENCE2_AVAILABLE:

    @register_OCR("florence2_ocr")
    class Florence2OCROCR(OCRBase):
        """
        Florence-2: Microsoft vision model for OCR. Use with any detector.
        Base (faster) or Large (higher quality). Task <OCR> extracts text per crop.
        """
        params = {
            "model_name": {
                "type": "selector",
                "options": [
                    "microsoft/Florence-2-base",
                    "microsoft/Florence-2-large",
                ],
                "value": "microsoft/Florence-2-base",
                "description": "Florence-2 model (base = less VRAM, large = better quality).",
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
                "description": "Max tokens to generate per crop (128–512).",
            },
            "description": "Florence-2 (HF). OCR per crop with <OCR> task.",
        }
        _load_model_keys = {"model", "processor"}

        def __init__(self, **params) -> None:
            super().__init__(**params)
            self.device = (self.params.get("device") or {}).get("value", "cpu")
            self.model = None
            self.processor = None
            self._model_name = None

        def _load_model(self):
            model_name = (
                (self.params.get("model_name") or {}).get("value", "microsoft/Florence-2-base")
                or "microsoft/Florence-2-base"
            )
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
            self.processor = AutoProcessor.from_pretrained(
                model_name, trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            )
            self.model.to(device)
            self.model.eval()
            self.device = device

        def _run_one(self, pil_img: "Image.Image") -> str:
            if pil_img.size[0] == 0 or pil_img.size[1] == 0:
                return ""
            try:
                inputs = self.processor(
                    text=FLORENCE2_OCR_TASK,
                    images=pil_img,
                    return_tensors="pt",
                )
                device = next(self.model.parameters()).device
                inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
                max_new_tokens = 256
                try:
                    mt = self.params.get("max_new_tokens", {})
                    if isinstance(mt, dict):
                        max_new_tokens = max(64, min(512, int(mt.get("value", 256))))
                    else:
                        max_new_tokens = max(64, min(512, int(mt)))
                except (TypeError, ValueError):
                    pass
                with torch.inference_mode():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                    )
                generated_text = self.processor.batch_decode(
                    generated_ids, skip_special_tokens=False
                )[0]
                w, h = pil_img.size
                parsed = self.processor.post_process_generation(
                    generated_text,
                    task=FLORENCE2_OCR_TASK,
                    image_size=(w, h),
                )
                if parsed and FLORENCE2_OCR_TASK in parsed:
                    out = parsed[FLORENCE2_OCR_TASK]
                    if isinstance(out, dict) and "labels" in out:
                        return "\n".join(out["labels"]).strip()
                    if isinstance(out, str):
                        return out.strip()
                return generated_text.strip() or ""
            except Exception as e:
                self.logger.warning(f"Florence-2 OCR failed: {e}")
                return ""

        def _ocr_blk_list(self, img: np.ndarray, blk_list: List[TextBlock], *args, **kwargs) -> None:
            im_h, im_w = img.shape[:2]
            padding = 0
            try:
                p = self.params.get("crop_padding", {})
                if isinstance(p, dict):
                    padding = max(0, min(24, int(p.get("value", 0))))
                else:
                    padding = max(0, min(24, int(p)))
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
                self.processor = None
                self._model_name = None
