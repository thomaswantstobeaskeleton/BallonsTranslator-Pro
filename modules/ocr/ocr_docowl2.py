"""
DocOwl2 – 9B OCR-free document understanding (Hugging Face, trust_remote_code).
Multi-page, tables, layout; single image or crop → text. Quality over speed.
Requires: pip install transformers torch pillow accelerate
"""
from typing import List
import os
import tempfile
import numpy as np
import cv2
from .base import OCRBase, register_OCR, DEVICE_SELECTOR, TextBlock

_DOCOWL2_AVAILABLE = False
try:
    from transformers import AutoTokenizer, AutoModel
    from PIL import Image
    import torch
    _DOCOWL2_AVAILABLE = True
except ImportError as e:
    import logging
    logging.getLogger("BallonTranslator").debug(
        f"DocOwl2 OCR not available: {e}. Install: pip install transformers torch pillow accelerate"
    )


def _cv2_to_pil_rgb(img: np.ndarray) -> "Image.Image":
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)


if _DOCOWL2_AVAILABLE:

    @register_OCR("docowl2_ocr")
    class DocOwl2OCROCR(OCRBase):
        """
        DocOwl2 9B: OCR-free document understanding. Tables, layout, multi-script.
        Use for document/crop text extraction. Heavyweight; quality over speed.
        """
        params = {
            "model_name": {
                "type": "line_editor",
                "value": "mPLUG/DocOwl2",
                "description": "Hugging Face model id (trust_remote_code).",
            },
            "device": DEVICE_SELECTOR(),
            "crop_padding": {
                "type": "line_editor",
                "value": 4,
                "description": "Pixels to add around each box when cropping (0–24).",
            },
            "prompt": {
                "type": "line_editor",
                "value": "Extract all text from this image. Preserve layout and line breaks.",
                "description": "Query for text extraction.",
            },
            "basic_image_size": {
                "type": "line_editor",
                "value": 504,
                "description": "Base image size for processor (e.g. 504).",
            },
            "use_bf16": {
                "type": "checkbox",
                "value": True,
                "description": "Use bfloat16 when available.",
            },
            "description": "DocOwl2 9B (HF, trust_remote_code). Document understanding.",
        }
        _load_model_keys = {"model", "tokenizer"}

        def __init__(self, **params) -> None:
            super().__init__(**params)
            self.device = self.params["device"]["value"]
            self.model = None
            self.tokenizer = None
            self._model_name = None

        def _load_model(self):
            model_name = (self.params.get("model_name") or {}).get("value", "mPLUG/DocOwl2") or "mPLUG/DocOwl2"
            if self.model is not None and self._model_name == model_name:
                return
            self._model_name = model_name
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
            use_bf16 = self.params.get("use_bf16", {}).get("value", True)
            dtype = torch.bfloat16 if (use_bf16 and torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)()) else torch.float16
            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )
            if hasattr(self.model, "init_processor"):
                bis = 504
                p = self.params.get("basic_image_size", {})
                if isinstance(p, dict):
                    try:
                        bis = max(224, min(1024, int(p.get("value", 504))))
                    except (TypeError, ValueError):
                        pass
                self.model.init_processor(tokenizer=self.tokenizer, basic_image_size=bis, crop_anchors="grid_12")
            self.model = self.model.to(self.device)
            self.model.eval()

        def _run_one(self, pil_img: "Image.Image") -> str:
            tmp_path = None
            try:
                fd, tmp_path = tempfile.mkstemp(suffix=".png")
                os.close(fd)
                pil_img.save(tmp_path)
                images = [tmp_path]
                prompt = (self.params.get("prompt") or {}).get("value", "Extract all text from this image.") or "Extract all text from this image."
                messages = [{"role": "USER", "content": "<|image|>" * len(images) + prompt}]
                with torch.no_grad():
                    out = self.model.chat(messages=messages, images=images, tokenizer=self.tokenizer)
                if out is None:
                    return ""
                return (out if isinstance(out, str) else str(out)).strip()
            except Exception as e:
                self.logger.warning(f"DocOwl2 OCR failed: {e}")
                return ""
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

        def _ocr_blk_list(self, img: np.ndarray, blk_list: List[TextBlock], *args, **kwargs) -> None:
            im_h, im_w = img.shape[:2]
            pad = 0
            cp = self.params.get("crop_padding", {})
            if isinstance(cp, dict):
                try:
                    pad = max(0, min(24, int(cp.get("value", 0))))
                except (TypeError, ValueError):
                    pass
            for blk in blk_list:
                x1, y1, x2, y2 = blk.xyxy
                if pad > 0:
                    x1 = max(0, x1 - pad)
                    y1 = max(0, y1 - pad)
                    x2 = min(im_w, x2 + pad)
                    y2 = min(im_h, y2 + pad)
                if not (x1 < x2 and y1 < y2):
                    blk.text = [""]
                    continue
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    blk.text = [""]
                    continue
                pil_img = _cv2_to_pil_rgb(crop)
                text = self._run_one(pil_img)
                blk.text = [text if text else ""]

        def ocr_img(self, img: np.ndarray) -> str:
            pil_img = _cv2_to_pil_rgb(img)
            return self._run_one(pil_img)

        def updateParam(self, param_key: str, param_content):
            super().updateParam(param_key, param_content)
            if param_key == "device":
                self.device = self.params["device"]["value"]
                if self.model is not None:
                    self.model = self.model.to(self.device)
            elif param_key in ("model_name", "use_bf16", "basic_image_size"):
                self.model = None
                self.tokenizer = None
                self._model_name = None
