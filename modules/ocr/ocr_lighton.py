"""
LightOnOCR-2-1B – high-performance 1B OCR (Hugging Face). Quality/speed balance.
83.2% OlmOCR-Bench; faster than OlmOCR/Chandra. Per-block or full-page.
Requires: pip install transformers torch pillow
"""
from typing import List
import os
import tempfile
import numpy as np
import cv2
from .base import OCRBase, register_OCR, DEVICE_SELECTOR, TextBlock

_LIGHTON_AVAILABLE = False
try:
    from transformers import pipeline, AutoProcessor, AutoModelForImageTextToText
    from PIL import Image
    import torch
    _LIGHTON_AVAILABLE = True
except ImportError as e:
    import logging
    logging.getLogger("BallonTranslator").debug(
        f"LightOnOCR not available: {e}. Install: pip install transformers torch pillow"
    )


def _cv2_to_pil_rgb(img: np.ndarray) -> "Image.Image":
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)


if _LIGHTON_AVAILABLE:

    @register_OCR("lighton_ocr")
    class LightOnOCROCR(OCRBase):
        """
        LightOnOCR-2-1B: 1B-parameter OCR. Strong quality, faster than 7B VLMs.
        Use for document/crop OCR when you want quality without heavyweight VRAM.
        """
        params = {
            "model_name": {
                "type": "selector",
                "options": [
                    "lightonai/LightOnOCR-2-1B",
                    "lightonai/LightOnOCR-2-1B-bbox",
                ],
                "value": "lightonai/LightOnOCR-2-1B",
                "description": "OCR-only or bbox variant.",
            },
            "device": DEVICE_SELECTOR(),
            "crop_padding": {
                "type": "line_editor",
                "value": 4,
                "description": "Pixels to add around each box when cropping (0–24).",
            },
            "max_new_tokens": {
                "type": "line_editor",
                "value": 512,
                "description": "Max tokens per block.",
            },
            "use_bf16": {
                "type": "checkbox",
                "value": True,
                "description": "Use bfloat16 when available.",
            },
            "description": "LightOnOCR-2-1B (HF). High performance, 1B params.",
        }
        _load_model_keys = {"model"}

        def __init__(self, **params) -> None:
            super().__init__(**params)
            self.device = self.params["device"]["value"]
            self.pipe = None
            self.model = None
            self.processor = None
            self._model_name = None

        def _load_model(self):
            model_name = (self.params.get("model_name") or {}).get("value", "lightonai/LightOnOCR-2-1B") or "lightonai/LightOnOCR-2-1B"
            if self.model is not None and self._model_name == model_name:
                return
            self._model_name = model_name
            self.pipe = None
            self.processor = None
            use_bf16 = self.params.get("use_bf16", {}).get("value", True)
            dtype = torch.bfloat16 if (use_bf16 and torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)()) else torch.float16
            try:
                self.pipe = pipeline(
                    "image-text-to-text",
                    model=model_name,
                    device=self.device if self.device != "cpu" else -1,
                    torch_dtype=dtype,
                )
                self.model = getattr(self.pipe, "model", self.pipe)
            except Exception as e:
                self.logger.warning(f"LightOnOCR pipeline failed: {e}; trying with processor+model.")
                self._load_model_fallback(model_name, dtype)

        def _load_model_fallback(self, model_name: str, dtype):
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForImageTextToText.from_pretrained(model_name, torch_dtype=dtype, trust_remote_code=True)
            self.model.to(self.device)
            self.pipe = None

        def _run_one(self, pil_img: "Image.Image") -> str:
            try:
                if self.pipe is not None:
                    out = self.pipe(pil_img, max_new_tokens=self._max_tokens())
                    if isinstance(out, list) and len(out) > 0:
                        item = out[0]
                        if isinstance(item, dict) and "generated_text" in item:
                            return (item["generated_text"] or "").strip()
                        if isinstance(item, str):
                            return item.strip()
                    return ""
                # Fallback: processor + model (chat-style if needed)
                processor = getattr(self, "processor", None)
                model = getattr(self, "model", None)
                if processor is None or model is None:
                    return ""
                if hasattr(processor, "apply_chat_template"):
                    import os as _os
                    fd, path = tempfile.mkstemp(suffix=".png")
                    _os.close(fd)
                    pil_img.save(path)
                    try:
                        messages = [{"role": "user", "content": [{"type": "image", "url": path}, {"type": "text", "text": "OCR:"}]}]
                        inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
                        if hasattr(inputs, "to"):
                            inputs = inputs.to(model.device)
                        else:
                            inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
                        gen = model.generate(**inputs, max_new_tokens=self._max_tokens())
                        prompt_len = inputs["input_ids"].shape[1]
                        text = processor.decode(gen[0, prompt_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
                        return text
                    finally:
                        if _os.path.exists(path):
                            _os.unlink(path)
                inputs = processor(images=pil_img, return_tensors="pt")
                inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
                gen = model.generate(**inputs, max_new_tokens=self._max_tokens())
                text = processor.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                return (text[0] if text else "").strip()
            except Exception as e:
                self.logger.warning(f"LightOnOCR failed: {e}")
                return ""

        def _max_tokens(self):
            mt = self.params.get("max_new_tokens", {})
            if isinstance(mt, dict):
                try:
                    return max(64, min(2048, int(mt.get("value", 512))))
                except (TypeError, ValueError):
                    pass
            return 512

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
                if self.pipe is not None:
                    self.pipe.device = self.device if self.device != "cpu" else -1
                elif getattr(self, "model", None) is not None:
                    self.model.to(self.device)
            elif param_key in ("model_name", "use_bf16"):
                self.pipe = None
                self._model_name = None
