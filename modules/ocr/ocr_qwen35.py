"""
Qwen3.5 OCR – Image-Text-to-Text models from the Qwen3.5 collection.
https://huggingface.co/collections/Qwen/qwen35
Implements all sizes under 9B plus 9B: 0.8B, 2B, 4B, 9B (Instruct and Base).
Requires: pip install transformers torch pillow accelerate
Use with any detector (e.g. hf_object_det).
"""
from typing import List
import os
import re
import tempfile
import numpy as np
import cv2
from .base import OCRBase, register_OCR, DEVICE_SELECTOR, TextBlock

_QWEN35_AVAILABLE = False
try:
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from PIL import Image
    import torch
    _QWEN35_AVAILABLE = True
except ImportError as e:
    import logging
    logging.getLogger("BallonTranslator").debug(
        f"Qwen3.5 OCR not available: {e}. Install: pip install transformers torch pillow accelerate"
    )


def _cv2_to_pil_rgb(img: np.ndarray) -> "Image.Image":
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)


def _is_no_text_placeholder(text: str) -> bool:
    """Treat model 'no text' / placeholder / role leak / HTML-only responses as empty."""
    if not text or not isinstance(text, str):
        return True
    raw = text.strip()
    if not raw:
        return True
    # Normalize for phrase matching: single line, collapsed spaces, upper
    t = raw.upper().replace("\n", " ").replace("\r", " ")
    t = " ".join(t.split())
    if not t:
        return True
    # Punctuation/whitespace only (e.g. "." or " . ")
    if re.sub(r"[\s\.\,\-\;\:\!\?\'\"\-\–\—\u3000\u00a0]+", "", t) == "":
        return True
    # "THERE IS NO TEXT YOU CAN EXTRACT..." and variants
    no_text_phrases = (
        "THERE IS NO TEXT",
        "NO TEXT YOU CAN EXTRACT",
        "NO TEXT TO EXTRACT",
        "CANNOT EXTRACT",
        "UNABLE TO EXTRACT",
        "NO EXTRACTABLE TEXT",
        "NO VISIBLE TEXT",
        "IMAGE CONTAINS NO TEXT",
        "NO READABLE TEXT",
    )
    for phrase in no_text_phrases:
        if phrase in t:
            return True
    # Role/chat leak: "assistant", "<think>...</think>", "<tool_call>", etc.
    tag_removed = re.sub(r"<[^>]*>", " ", t, flags=re.IGNORECASE)
    tag_removed = " ".join(tag_removed.split()).strip()
    if tag_removed in ("", "ASSISTANT", "USER", "SYSTEM"):
        return True
    # Only role labels (e.g. "user user assistant" with optional think tags)
    role_only = {"USER", "ASSISTANT", "SYSTEM"}
    words = tag_removed.split()
    if words and all(w in role_only for w in words):
        return True
    if tag_removed.startswith("ASSISTANT") and len(tag_removed) < 50:
        rest = tag_removed[9:].strip()
        if not rest or re.sub(r"[\s\.\,\-\;\:\!\?]+", "", rest) == "":
            return True
    # HTML/XML-only (e.g. <html><body><p></p></html> or <P><DIV></DIV></P>)
    stripped = t.replace(" ", "")
    if stripped.startswith("<") and ">" in stripped:
        remainder = re.sub(r"<[^>]+>", "", stripped)
        if not remainder or remainder in ("", "/"):
            return True
        # Allow only punctuation/whitespace after stripping tags
        if re.sub(r"[\s\.\,\-\;\:\!\?\/]+", "", remainder) == "":
            return True
    # LaTeX/math placeholder only (e.g. $$\TEXT{ }$$ or $$ $$ or $${}$$
    if "$$" in raw or ("$" in raw and raw.strip().startswith("$")):
        # Explicit empty \TEXT{ } pattern (e.g. $$\TEXT\n{ }$$)
        if "\\TEXT" in raw.upper() and re.search(r"\\TEXT\s*\{\s*\}", raw, re.IGNORECASE | re.DOTALL):
            return True
        # Single block $$...$$ or $...$ with empty or placeholder content
        math_block = re.sub(r"\s+", " ", raw).strip()
        if re.match(r"^\$\$.*\$\$$", math_block) or (
            re.match(r"^\$[^$].*\$$", math_block) and "$$" not in math_block
        ):
            inner = math_block
            if inner.startswith("$$"):
                inner = inner[2:].lstrip()
            if inner.endswith("$$"):
                inner = inner[:-2].rstrip()
            if inner.startswith("$") and inner.endswith("$"):
                inner = inner[1:-1].strip()
            # Empty or only backslash-commands and braces/whitespace
            if not inner or re.sub(r"[\\\s{}]+", "", inner) == "":
                return True
    return False


# Qwen3.5 collection: all models under 9B + 9B (Instruct and Base)
# https://huggingface.co/collections/Qwen/qwen35
QWEN35_MODEL_OPTIONS = [
    "Qwen/Qwen3.5-0.8B",
    "Qwen/Qwen3.5-0.8B-Base",
    "Qwen/Qwen3.5-2B",
    "Qwen/Qwen3.5-2B-Base",
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-4B-Base",
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.5-9B-Base",
]


if _QWEN35_AVAILABLE:

    @register_OCR("qwen35_ocr")
    class Qwen35OCR(OCRBase):
        """
        Qwen3.5 Image-Text-to-Text: 0.8B, 2B, 4B, 9B (Instruct and Base).
        From https://huggingface.co/collections/Qwen/qwen35. Use with any detector.
        """
        params = {
            "model_name": {
                "type": "selector",
                "options": QWEN35_MODEL_OPTIONS,
                "value": "Qwen/Qwen3.5-2B",
                "description": "Model: 0.8B (smallest) to 9B. Base = no chat tuning, Instruct = chat.",
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
                "description": "Max tokens per block (128–2048).",
            },
            "prompt": {
                "type": "line_editor",
                "value": "Extract all text from this image. Preserve layout and line breaks.",
                "description": "OCR prompt for the model.",
            },
            "use_bf16": {
                "type": "checkbox",
                "value": True,
                "description": "Use bfloat16 when available (saves VRAM).",
            },
            "low_vram": {
                "type": "checkbox",
                "value": False,
                "description": "Use device_map=auto (CPU offload). Enable on limited VRAM; slower.",
            },
            "attn_implementation": {
                "type": "selector",
                "options": ["sdpa", "eager", "flash_attention_2"],
                "value": "sdpa",
                "description": "Attention backend. flash_attention_2 is fastest if installed.",
            },
            "description": "Qwen3.5 OCR – 0.8B/2B/4B/9B (Instruct & Base). Use with hf_object_det or other detector.",
        }
        _load_model_keys = {"processor", "model"}

        def __init__(self, **params) -> None:
            super().__init__(**params)
            self.device = (self.params.get("device") or {}).get("value", "cpu")
            self.processor = None
            self.model = None
            self._model_name = None
            self._device_for_inputs = None

        def _load_model(self):
            import torch
            model_name = (self.params.get("model_name") or {}).get("value", QWEN35_MODEL_OPTIONS[0]) or QWEN35_MODEL_OPTIONS[0]
            dev = (self.params.get("device") or {}).get("value", "cpu")
            if dev in ("cuda", "gpu") and torch.cuda.is_available():
                dev = "cuda"
            else:
                dev = "cpu"
            if self.processor is not None and self._model_name == model_name:
                if self.model is not None and hasattr(self.model, "to") and dev != "cpu":
                    try:
                        self.model.to(dev)
                    except Exception:
                        pass
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
            low_vram = self.params.get("low_vram", {}).get("value", False)
            attn_impl = (self.params.get("attn_implementation") or {}).get("value", "sdpa") or "sdpa"
            if attn_impl == "flash_attention_2":
                try:
                    import torch.utils.checkpoint
                except Exception:
                    attn_impl = "sdpa"
            self.processor = AutoProcessor.from_pretrained(model_name)
            load_kw = {"torch_dtype": dtype}
            if attn_impl and attn_impl != "eager":
                load_kw["attn_implementation"] = attn_impl
            if low_vram:
                load_kw["device_map"] = "auto"
                self.model = AutoModelForImageTextToText.from_pretrained(model_name, **load_kw)
                self._device_for_inputs = next(self.model.parameters()).device
            else:
                self.model = AutoModelForImageTextToText.from_pretrained(model_name, **load_kw)
                self.model.to(dev)
                self._device_for_inputs = None
            self.model.eval()

        def _run_one(self, pil_img: "Image.Image") -> str:
            if pil_img.size[0] == 0 or pil_img.size[1] == 0:
                return ""
            tmp_path = None
            try:
                fd, tmp_path = tempfile.mkstemp(suffix=".png")
                os.close(fd)
                pil_img.save(tmp_path)
                img_ref = os.path.abspath(tmp_path)
                prompt = (self.params.get("prompt") or {}).get("value", "Extract all text from this image.") or "Extract all text from this image."
                # Qwen3.5 may expect "image" with "url" or "image_url" with "url"; try common formats
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
                if _is_no_text_placeholder(text):
                    text = ""
                return text
            except Exception as e:
                self.logger.warning(f"Qwen3.5 OCR failed: {e}")
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
                if _is_no_text_placeholder(text):
                    text = ""
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
            elif param_key in ("model_name", "use_bf16", "low_vram", "attn_implementation"):
                self.processor = None
                self.model = None
                self._model_name = None
                self._device_for_inputs = None
