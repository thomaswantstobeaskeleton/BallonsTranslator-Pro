"""
GOT-OCR2 – General OCR (Hugging Face, 580M).
Unified OCR for plain/scene/formatted text; supports region-level via cropped image.
Requires: pip install transformers torch accelerate pillow
Model: stepfun-ai/GOT-OCR-2.0-hf
Fully implements: single-image and batched inference, dtype option, stop_strings, decode slice.
"""
from typing import List
import numpy as np
import cv2
from .base import OCRBase, register_OCR, DEFAULT_DEVICE, DEVICE_SELECTOR, TextBlock

_GOT_OCR2_AVAILABLE = False
try:
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from PIL import Image
    import torch
    _GOT_OCR2_AVAILABLE = True
except ImportError as e:
    import logging
    logging.getLogger("BallonTranslator").debug(
        f"GOT-OCR2 not available: {e}. Install: pip install transformers torch accelerate pillow"
    )


def _cv2_to_pil_rgb(img: np.ndarray) -> "Image.Image":
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)


if _GOT_OCR2_AVAILABLE:

    @register_OCR("got_ocr2")
    class GOTOCR2OCR(OCRBase):
        """
        GOT-OCR2: unified OCR (plain/scene/formatted). Good for documents and comics.
        Per-block: each crop is sent as one image. Heavier than TrOCR; use when quality matters.
        """
        params = {
            "model_name": {
                "type": "line_editor",
                "value": "stepfun-ai/GOT-OCR-2.0-hf",
                "description": "Hugging Face model id (default: stepfun-ai/GOT-OCR-2.0-hf).",
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
                "description": "Max tokens per block (256 is enough for a line; increase for long paragraphs).",
            },
            "use_fast_processor": {
                "type": "checkbox",
                "value": True,
                "description": "Use fast tokenizer/processor (recommended).",
            },
            "use_bf16": {
                "type": "checkbox",
                "value": False,
                "description": "Use bfloat16 (saves VRAM on supported GPUs).",
            },
            "batch_size": {
                "type": "line_editor",
                "value": 1,
                "description": "Batch size (1 = safe for 11GB VRAM; 4–8 = faster, needs more VRAM). OOM falls back to 1.",
            },
            "description": "GOT-OCR2 – unified OCR (Hugging Face, 580M).",
        }
        _load_model_keys = {"processor", "model"}

        def __init__(self, **params) -> None:
            super().__init__(**params)
            self.device = self.params["device"]["value"]
            self.processor = None
            self.model = None
            self._model_name = None

        def _load_model(self):
            model_name = (self.params.get("model_name") or {}).get("value", "stepfun-ai/GOT-OCR-2.0-hf") or "stepfun-ai/GOT-OCR-2.0-hf"
            if self.processor is not None and self._model_name == model_name:
                return
            self._model_name = model_name
            use_fast = self.params.get("use_fast_processor", {}).get("value", True)
            self.processor = AutoProcessor.from_pretrained(model_name, use_fast=use_fast)
            dtype = torch.float32
            if self.params.get("use_bf16", {}).get("value", False) and torch.cuda.is_available():
                try:
                    if torch.cuda.is_bf16_supported():
                        dtype = torch.bfloat16
                except Exception:
                    dtype = torch.float16
            elif self.device == "cuda":
                dtype = torch.float16
            self.model = AutoModelForImageTextToText.from_pretrained(model_name, torch_dtype=dtype)
            self.model.to(self.device)

        def _get_max_tokens(self):
            max_tokens = 256
            mt = self.params.get("max_new_tokens", {})
            if isinstance(mt, dict):
                try:
                    max_tokens = max(32, min(4096, int(mt.get("value", 256))))
                except (TypeError, ValueError):
                    pass
            return max_tokens

        def _generate_one(self, pil_img, max_tokens: int) -> str:
            inputs = self.processor(pil_img, return_tensors="pt")
            inputs = inputs.to(self.device)
            gen_kw = {
                **inputs,
                "do_sample": False,
                "tokenizer": self.processor.tokenizer,
                "max_new_tokens": max_tokens,
            }
            try:
                out = self.model.generate(**gen_kw, stop_strings="")
            except TypeError:
                out = self.model.generate(**gen_kw)
            prompt_len = inputs["input_ids"].shape[1]
            return self.processor.decode(out[0, prompt_len:], skip_special_tokens=True).strip()

        def _generate_batch(self, pil_imgs: List, max_tokens: int) -> List[str]:
            if not pil_imgs:
                return []
            inputs = self.processor(pil_imgs, return_tensors="pt")
            inputs = inputs.to(self.device)
            gen_kw = {
                **inputs,
                "do_sample": False,
                "tokenizer": self.processor.tokenizer,
                "max_new_tokens": max_tokens,
            }
            try:
                out = self.model.generate(**gen_kw, stop_strings="")
            except TypeError:
                out = self.model.generate(**gen_kw)
            prompt_len = inputs["input_ids"].shape[1]
            texts = self.processor.batch_decode(out[:, prompt_len:], skip_special_tokens=True)
            return [t.strip() for t in texts]

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
            max_tokens = self._get_max_tokens()
            batch_size = 1
            bs = self.params.get("batch_size", {})
            if isinstance(bs, dict):
                try:
                    batch_size = max(1, min(16, int(bs.get("value", 1))))
                except (TypeError, ValueError):
                    pass
            valid_indices = []
            valid_pils = []
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
                valid_indices.append(blk)
                valid_pils.append(_cv2_to_pil_rgb(crop))
            if batch_size <= 1 or len(valid_pils) <= 1:
                for blk, pil_img in zip(valid_indices, valid_pils):
                    try:
                        text = self._generate_one(pil_img, max_tokens)
                        blk.text = [text if text else ""]
                    except Exception as e:
                        self.logger.warning(f"GOT-OCR2 failed for block: {e}")
                        blk.text = [""]
                return
            for i in range(0, len(valid_pils), batch_size):
                batch_blks = valid_indices[i : i + batch_size]
                batch_pils = valid_pils[i : i + batch_size]
                try:
                    texts = self._generate_batch(batch_pils, max_tokens)
                    for blk, text in zip(batch_blks, texts):
                        blk.text = [text if text else ""]
                except Exception as e:
                    self.logger.warning(f"GOT-OCR2 batch failed: {e}")
                    # OOM or other error: free cache and fall back to one-by-one so blocks get text instead of empty
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    for blk, pil_img in zip(batch_blks, batch_pils):
                        try:
                            text = self._generate_one(pil_img, max_tokens)
                            blk.text = [text if text else ""]
                        except Exception as e2:
                            self.logger.warning(f"GOT-OCR2 fallback failed for block: {e2}")
                            blk.text = [""]

        def ocr_img(self, img: np.ndarray) -> str:
            pil_img = _cv2_to_pil_rgb(img)
            return self._generate_one(pil_img, self._get_max_tokens())

        def updateParam(self, param_key: str, param_content):
            super().updateParam(param_key, param_content)
            if param_key == "device":
                self.device = self.params["device"]["value"]
                if self.model is not None:
                    self.model.to(self.device)
            elif param_key in ("model_name", "use_fast_processor", "use_bf16"):
                self.processor = None
                self.model = None
                self._model_name = None
