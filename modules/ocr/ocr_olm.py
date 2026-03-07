"""
OlmOCR 7B – Allen AI document OCR (Hugging Face). Issue #872.
Uses allenai/olmOCR-2-7B-1025 with language-aware prompt:
"Give me text from image, writen in {lang} language, nothing else."
Requires: pip install transformers torch pillow accelerate
"""
from typing import List
import os
import tempfile
import numpy as np
import cv2
from .base import OCRBase, register_OCR, DEVICE_SELECTOR, TextBlock

_OLM_AVAILABLE = False
try:
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from PIL import Image
    import torch
    _OLM_AVAILABLE = True
except ImportError:
    import logging
    logging.getLogger("BallonsTranslator").debug(
        "OlmOCR not available. Install: pip install transformers torch pillow accelerate"
    )


# Language names for the prompt (as in issue #872)
OLM_SOURCE_LANGUAGES = [
    "English", "Japanese", "Korean", "Chinese", "Russian",
    "French", "German", "Spanish", "Italian", "Portuguese",
    "Arabic", "Hindi", "Vietnamese", "Thai", "Indonesian",
]

MODEL_ID = "allenai/olmOCR-2-7B-1025"
PROCESSOR_ID = "Qwen/Qwen2.5-VL-7B-Instruct"  # OlmOCR is based on Qwen2.5-VL


def _cv2_to_pil_rgb(img: np.ndarray) -> "Image.Image":
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)


def _prompt_for_lang(lang: str) -> str:
    return f"Give me text from image, writen in {lang} language, nothing else."


if _OLM_AVAILABLE:

    @register_OCR("olm_ocr")
    class OlmOCROCR(OCRBase):
        """
        OlmOCR 7B (Allen AI). Best-in-class OCR per benchmarks; use language prompt for best results.
        Prompt: "Give me text from image, writen in {lang} language, nothing else."
        """
        params = {
            "source_language": {
                "type": "selector",
                "options": OLM_SOURCE_LANGUAGES,
                "value": "Japanese",
                "description": "Language of the text in the image (for OCR prompt).",
            },
            "device": DEVICE_SELECTOR(),
            "crop_padding": {
                "type": "line_editor",
                "value": 4,
                "description": "Pixels around each crop (0–24).",
            },
            "max_new_tokens": {
                "type": "line_editor",
                "value": 512,
                "description": "Max tokens per block (128–2048).",
            },
            "use_bf16": {
                "type": "checkbox",
                "value": True,
                "description": "Use bfloat16 (saves VRAM).",
            },
            "low_vram": {
                "type": "checkbox",
                "value": True,
                "description": "Use device_map=auto (CPU offload).",
            },
            "description": "OlmOCR 7B – Allen AI document OCR; language-aware prompt.",
        }
        _load_model_keys = {"processor", "model"}

        def __init__(self, **params) -> None:
            super().__init__(**params)
            self.device = (self.params.get("device") or {}).get("value", "cpu")
            self.processor = None
            self.model = None
            self._device_for_inputs = None

        def _load_model(self):
            dev = (self.params.get("device") or {}).get("value", "cpu")
            if dev in ("cuda", "gpu") and torch.cuda.is_available():
                dev = "cuda"
            else:
                dev = "cpu"
            if self.processor is not None and self.model is not None:
                if not getattr(self, "_device_for_inputs", None) and hasattr(self.model, "to"):
                    self.model.to(dev)
                self.device = dev
                return
            self.device = dev
            use_bf16 = self.params.get("use_bf16", {}).get("value", True)
            dtype = torch.bfloat16
            if not (torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)()):
                dtype = torch.float16
            if not use_bf16:
                dtype = torch.float16
            low_vram = self.params.get("low_vram", {}).get("value", True)
            self.processor = AutoProcessor.from_pretrained(PROCESSOR_ID)
            load_kw = {"torch_dtype": dtype}
            if low_vram:
                load_kw["device_map"] = "auto"
                self.model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, **load_kw)
                self._device_for_inputs = next(self.model.parameters()).device
            else:
                self.model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, **load_kw)
                self.model.to(dev)
                self._device_for_inputs = None
            self.model.eval()

        def _run_one(self, pil_img: "Image.Image") -> str:
            if pil_img.size[0] == 0 or pil_img.size[1] == 0:
                return ""
            tmp_path = None
            try:
                lang = (self.params.get("source_language") or {}).get("value", "Japanese") or "Japanese"
                prompt = _prompt_for_lang(lang)
                fd, tmp_path = tempfile.mkstemp(suffix=".png")
                os.close(fd)
                pil_img.save(tmp_path)
                img_ref = os.path.abspath(tmp_path)
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "url": img_ref},
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
                inputs.pop("token_type_ids", None)
                inp_device = self._device_for_inputs if self._device_for_inputs is not None else self.model.device
                if hasattr(inputs, "to"):
                    inputs = inputs.to(inp_device)
                else:
                    inputs = {k: (v.to(inp_device) if hasattr(v, "to") else v) for k, v in inputs.items()}
                max_tokens = 512
                mt = self.params.get("max_new_tokens", {})
                if isinstance(mt, dict):
                    try:
                        max_tokens = max(64, min(2048, int(mt.get("value", 512))))
                    except (TypeError, ValueError):
                        pass
                with torch.inference_mode():
                    out = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
                input_len = inputs["input_ids"].shape[1]
                gen = out[0, input_len:]
                text = self.processor.decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
                return text
            except Exception as e:
                self.logger.warning(f"OlmOCR failed: {e}")
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
            if param_key == "device":
                self.device = (self.params.get("device") or {}).get("value", "cpu")
                if self.model is not None and not getattr(self, "_device_for_inputs", None):
                    try:
                        self.model.to(self.device)
                    except Exception:
                        pass
