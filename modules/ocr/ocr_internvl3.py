"""
InternVL3 – OpenGVLab VLM OCR (Hugging Face). Native transformers image-text-to-text.
Strong multimodal; use for document/OCR. Requires: pip install transformers torch pillow
"""
from typing import List
import os
import tempfile
import numpy as np
import cv2
from .base import OCRBase, register_OCR, DEVICE_SELECTOR, TextBlock

_INTERNVL3_AVAILABLE = False
try:
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from PIL import Image
    import torch
    _INTERNVL3_AVAILABLE = True
except ImportError as e:
    import logging
    logging.getLogger("BallonTranslator").debug(
        f"InternVL3 OCR not available: {e}. Install: pip install transformers torch pillow"
    )


def _cv2_to_pil_rgb(img: np.ndarray) -> "Image.Image":
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)


if _INTERNVL3_AVAILABLE:

    @register_OCR("internvl3_ocr")
    class InternVL3OCROCR(OCRBase):
        """
        InternVL3: OpenGVLab VLM via native transformers. Use for document/OCR.
        Supports 1B/2B/8B; 8B = best quality, 1B = less VRAM.
        """
        params = {
            "model_name": {
                "type": "selector",
                "options": [
                    "OpenGVLab/InternVL3-1B-hf",
                    "OpenGVLab/InternVL3-2B-hf",
                    "OpenGVLab/InternVL3-8B-hf",
                ],
                "value": "OpenGVLab/InternVL3-2B-hf",
                "description": "InternVL3 (1B/2B = less VRAM, 8B = best quality).",
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
                "description": "OCR prompt.",
            },
            "max_new_tokens": {
                "type": "line_editor",
                "value": 512,
                "description": "Max new tokens per block (128–2048).",
            },
            "use_bf16": {
                "type": "checkbox",
                "value": True,
                "description": "Use bfloat16 when available.",
            },
            "description": "InternVL3 (HF) – document/OCR via image-text-to-text.",
        }
        _load_model_keys = {"model", "processor"}

        def __init__(self, **params) -> None:
            super().__init__(**params)
            self.device = (self.params.get("device") or {}).get("value", "cpu")
            self.model = None
            self.processor = None
            self._model_name = None

        def _load_model(self):
            model_name = (self.params.get("model_name") or {}).get("value", "OpenGVLab/InternVL3-2B-hf") or "OpenGVLab/InternVL3-2B-hf"
            dev = (self.params.get("device") or {}).get("value", "cpu")
            if dev in ("cuda", "gpu") and torch.cuda.is_available():
                dev = "cuda"
            else:
                dev = "cpu"
            if self.model is not None and self._model_name == model_name:
                if hasattr(self.model, "to"):
                    self.model.to(dev)
                self.device = dev
                return
            self._model_name = model_name
            self.device = dev
            use_bf16 = self.params.get("use_bf16", {}).get("value", True)
            dtype = torch.bfloat16
            if not (torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)()):
                dtype = torch.float16
            if not use_bf16:
                dtype = torch.float16
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map=dev if dev == "cuda" else None,
            )
            if dev == "cpu":
                self.model = self.model.to(dev)
            self.model.eval()

        def _run_one(self, pil_img: "Image.Image") -> str:
            if pil_img.size[0] == 0 or pil_img.size[1] == 0:
                return ""
            tmp_path = None
            try:
                prompt = (self.params.get("prompt") or {}).get("value", "Extract all text from this image.") or "Extract all text from this image."
                max_nt = 512
                mt = self.params.get("max_new_tokens", {})
                if isinstance(mt, dict):
                    try:
                        max_nt = max(64, min(2048, int(mt.get("value", 512))))
                    except (TypeError, ValueError):
                        pass
                # Prefer temp file for compatibility (processors often expect url/path)
                fd, tmp_path = tempfile.mkstemp(suffix=".png")
                os.close(fd)
                pil_img.save(tmp_path)
                img_path = os.path.abspath(tmp_path)
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "url": img_path},
                            {"type": "text", "text": prompt},
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
                inputs = {k: v.to(self.model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
                if "pixel_values" in inputs and hasattr(inputs["pixel_values"], "to"):
                    inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)
                with torch.inference_mode():
                    out = self.model.generate(**inputs, max_new_tokens=max_nt)
                input_len = inputs["input_ids"].shape[1]
                text = self.processor.decode(out[0, input_len:], skip_special_tokens=True)
                return (text or "").strip()
            except Exception as e:
                self.logger.warning(f"InternVL3 OCR failed: {e}")
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
                x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
                x2, y2 = min(im_w, x2 + pad), min(im_h, y2 + pad)
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
            if img.ndim == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            return self._run_one(_cv2_to_pil_rgb(img))

        def updateParam(self, param_key: str, param_content):
            super().updateParam(param_key, param_content)
            if param_key in ("model_name", "device"):
                self.model = None
                self.processor = None
                self._model_name = None
            elif param_key == "device":
                self.device = (self.params.get("device") or {}).get("value", "cpu")
