"""
Nanonets-OCR2-3B – document OCR (Hugging Face). Image-to-markdown, tables, LaTeX, 86.8% table tests.
Requires: pip install transformers torch pillow accelerate
"""
from typing import List
import os
import tempfile
import numpy as np
import cv2
from .base import OCRBase, register_OCR, DEVICE_SELECTOR, TextBlock

_NANONETS_AVAILABLE = False
try:
    from transformers import AutoProcessor, AutoModelForImageTextToText, AutoTokenizer
    from PIL import Image
    import torch
    _NANONETS_AVAILABLE = True
except ImportError as e:
    import logging
    logging.getLogger("BallonTranslator").debug(
        f"Nanonets-OCR2 not available: {e}. Install: pip install transformers torch pillow accelerate"
    )


def _cv2_to_pil_rgb(img: np.ndarray) -> "Image.Image":
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)


if _NANONETS_AVAILABLE:

    @register_OCR("nanonets_ocr")
    class NanonetsOCROCR(OCRBase):
        """
        Nanonets-OCR2-3B: document OCR. Tables, LaTeX, markdown; strong on table tests.
        Quality over speed. Optional flash_attention_2 for faster inference.
        """
        params = {
            "model_name": {
                "type": "selector",
                "options": ["nanonets/Nanonets-OCR2-3B", "nanonets/Nanonets-OCR2-1.5B-exp"],
                "value": "nanonets/Nanonets-OCR2-3B",
                "description": "3B (default) or 1.5B-exp.",
            },
            "device": DEVICE_SELECTOR(),
            "crop_padding": {
                "type": "line_editor",
                "value": 4,
                "description": "Pixels to add around each box when cropping (0–24).",
            },
            "max_new_tokens": {
                "type": "line_editor",
                "value": 2048,
                "description": "Max tokens per block (tables need more).",
            },
            "prompt": {
                "type": "line_editor",
                "value": "Extract the text from the above document as if you were reading it naturally. Return the tables in html format. Return the equations in LaTeX representation.",
                "description": "Nanonets-style extraction prompt.",
            },
            "use_bf16": {
                "type": "checkbox",
                "value": True,
                "description": "Use bfloat16 when available.",
            },
            "use_flash_attn": {
                "type": "checkbox",
                "value": False,
                "description": "Use Flash Attention 2 (requires flash-attn).",
            },
            "description": "Nanonets-OCR2-3B (HF). Tables, LaTeX, markdown.",
        }
        _load_model_keys = {"processor", "model"}

        def __init__(self, **params) -> None:
            super().__init__(**params)
            self.device = self.params["device"]["value"]
            self.processor = None
            self.model = None
            self._model_name = None

        def _load_model(self):
            model_name = (self.params.get("model_name") or {}).get("value", "nanonets/Nanonets-OCR2-3B") or "nanonets/Nanonets-OCR2-3B"
            if self.processor is not None and self._model_name == model_name:
                return
            self._model_name = model_name
            self.processor = AutoProcessor.from_pretrained(model_name)
            use_bf16 = self.params.get("use_bf16", {}).get("value", True)
            dtype = torch.bfloat16 if (use_bf16 and torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)()) else torch.float16
            kwargs = {"torch_dtype": dtype}
            if self.params.get("use_flash_attn", {}).get("value", False):
                try:
                    kwargs["attn_implementation"] = "flash_attention_2"
                except Exception:
                    pass
            try:
                self.model = AutoModelForImageTextToText.from_pretrained(model_name, **kwargs)
            except Exception as e:
                self.logger.warning(f"Nanonets load with attn failed: {e}; retrying without flash_attn.")
                kwargs.pop("attn_implementation", None)
                self.model = AutoModelForImageTextToText.from_pretrained(model_name, **kwargs)
            self.model = self.model.to(self.device).eval()

        def _run_one(self, pil_img: "Image.Image") -> str:
            tmp_path = None
            try:
                fd, tmp_path = tempfile.mkstemp(suffix=".png")
                os.close(fd)
                pil_img.save(tmp_path)
                prompt = (self.params.get("prompt") or {}).get("value", "Extract the text from the above document.") or "Extract the text from the above document."
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": f"file://{os.path.abspath(tmp_path).replace(os.sep, '/')}"},
                            {"type": "text", "text": prompt},
                        ],
                    },
                ]
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.processor(text=[text], images=[pil_img], padding=True, return_tensors="pt")
                inputs = {k: v.to(self.model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
                max_tokens = 2048
                mt = self.params.get("max_new_tokens", {})
                if isinstance(mt, dict):
                    try:
                        max_tokens = max(256, min(4096, int(mt.get("value", 2048))))
                    except (TypeError, ValueError):
                        pass
                output_ids = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
                generated_ids_trimmed = [output_ids[i, inputs["input_ids"].shape[1]:] for i in range(output_ids.size(0))]
                output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                return (output_text[0] if output_text else "").strip()
            except Exception as e:
                self.logger.warning(f"Nanonets-OCR2 failed: {e}")
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
                x1 = max(0, min(int(round(float(x1))), im_w - 1))
                y1 = max(0, min(int(round(float(y1))), im_h - 1))
                x2 = max(x1 + 1, min(int(round(float(x2))), im_w))
                y2 = max(y1 + 1, min(int(round(float(y2))), im_h))
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
            elif param_key in ("model_name", "use_bf16", "use_flash_attn"):
                self.processor = None
                self.model = None
                self._model_name = None
