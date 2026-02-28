"""
PaddleOCR-VL – SOTA document OCR via Hugging Face Transformers (0.9B).
109 languages; text, tables, formulas, charts. Chat-style: image + "OCR:" -> text.
Requires: pip install transformers torch pillow (transformers 5.x for PaddleOCR-VL).
Model: PaddlePaddle/PaddleOCR-VL

Comparison with PaddleOCRVLManga:
- paddleocr_vl_hf: General PaddleOCR-VL 0.9B from HF (PaddlePaddle/PaddleOCR-VL). 109 languages,
  document parsing (tables, formulas, charts). Configurable model id, bf16, crop padding.
- PaddleOCRVLManga: Manga-tuned model (jzhang533/PaddleOCR-VL-For-Manga), local/data dir.
  Optimized for comics/manga; uses same "OCR:" chat style but different architecture and shims for transformers 5.x.
"""
from typing import List
import os
import tempfile
import numpy as np
import cv2
from .base import OCRBase, register_OCR, DEFAULT_DEVICE, DEVICE_SELECTOR, TextBlock

_PADDLEOCR_VL_AVAILABLE = False
try:
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from PIL import Image
    import torch
    _PADDLEOCR_VL_AVAILABLE = True
except ImportError as e:
    import logging
    logging.getLogger("BallonTranslator").debug(
        f"PaddleOCR-VL (HF) not available: {e}. Install: pip install transformers torch pillow"
    )


def _cv2_to_pil_rgb(img: np.ndarray) -> "Image.Image":
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)


if _PADDLEOCR_VL_AVAILABLE:

    @register_OCR("paddleocr_vl_hf")
    class PaddleOCRVLHFOCR(OCRBase):
        """
        PaddleOCR-VL 0.9B via Hugging Face (element-level recognition).
        109 languages; text, table, formula, chart. Prompt "OCR:" for text.
        """
        params = {
            "model_name": {
                "type": "line_editor",
                "value": "PaddlePaddle/PaddleOCR-VL",
                "description": "Hugging Face model id (requires transformers 5.x for PaddleOCR-VL).",
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
                "description": "Max tokens per block.",
            },
            "use_bf16": {
                "type": "checkbox",
                "value": True,
                "description": "Use bfloat16 when available (recommended).",
            },
            "description": "PaddleOCR-VL 0.9B (HF transformers). 109 languages, document parsing.",
        }
        _load_model_keys = {"processor", "model"}

        def __init__(self, **params) -> None:
            super().__init__(**params)
            self.device = self.params["device"]["value"]
            self.processor = None
            self.model = None
            self._model_name = None

        def _load_model(self):
            model_name = (self.params.get("model_name") or {}).get("value", "PaddlePaddle/PaddleOCR-VL") or "PaddlePaddle/PaddleOCR-VL"
            if self.processor is not None and self._model_name == model_name:
                return
            self._model_name = model_name
            self.processor = AutoProcessor.from_pretrained(model_name)
            use_bf16 = self.params.get("use_bf16", {}).get("value", True)
            dtype = torch.bfloat16 if (use_bf16 and torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)()) else torch.float16
            try:
                self.model = AutoModelForImageTextToText.from_pretrained(model_name, torch_dtype=dtype)
            except Exception as e:
                self.logger.warning(f"PaddleOCR-VL load with dtype failed: {e}; trying default dtype.")
                self.model = AutoModelForImageTextToText.from_pretrained(model_name)
            self.model.to(self.device)

        def _run_one(self, pil_img: "Image.Image") -> str:
            tmp_path = None
            try:
                fd, tmp_path = tempfile.mkstemp(suffix=".png")
                os.close(fd)
                pil_img.save(tmp_path)
                ocr_prompt = "OCR (Chinese):" if self.params.get("chinese_only", {}).get("value", False) else "OCR:"
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "url": tmp_path},
                            {"type": "text", "text": ocr_prompt},
                        ],
                    }
                ]
                inputs = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                inputs = inputs.to(self.model.device)
                max_tokens = 256
                mt = self.params.get("max_new_tokens", {})
                if isinstance(mt, dict):
                    try:
                        max_tokens = max(32, min(1024, int(mt.get("value", 256))))
                    except (TypeError, ValueError):
                        pass
                generated_ids = self.model.generate(**inputs, max_new_tokens=max_tokens)
                in_ids = inputs["input_ids"]
                out_ids = generated_ids[0]
                generated_ids_trimmed = out_ids[in_ids.shape[-1]:]
                text = self.processor.decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
                return text
            except Exception as e:
                self.logger.warning(f"PaddleOCR-VL HF failed: {e}")
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
                    self.model.to(self.device)
            elif param_key in ("model_name", "use_bf16", "chinese_only"):
                if param_key != "chinese_only":
                    self.processor = None
                    self.model = None
                    self._model_name = None
