"""
Qwen2.5-VL 7B / OlmOCR 7B – heavyweight OCR (Hugging Face). Quality over speed.
Uses Qwen2.5-VL-7B-Instruct or allenai/olmOCR-2-7B-1025 with "Extract the text" prompt.
Requires: pip install transformers torch pillow accelerate (and ~16GB+ VRAM for 7B bf16).
"""
from typing import List
import os
import tempfile
import numpy as np
import cv2
from .base import OCRBase, register_OCR, DEFAULT_DEVICE, DEVICE_SELECTOR, TextBlock

_QWEN2VL_AVAILABLE = False
try:
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from PIL import Image
    import torch
    _QWEN2VL_AVAILABLE = True
except ImportError as e:
    import logging
    logging.getLogger("BallonTranslator").debug(
        f"Qwen2.5-VL OCR not available: {e}. Install: pip install transformers torch pillow accelerate"
    )


def _cv2_to_pil_rgb(img: np.ndarray) -> "Image.Image":
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)


if _QWEN2VL_AVAILABLE:

    @register_OCR("qwen2vl_7b")
    class Qwen2VL7BOCR(OCRBase):
        """
        Qwen2.5-VL 7B or OlmOCR 7B: heavyweight OCR. Prefer quality over speed.
        Use Qwen2.5-VL-7B-Instruct for general VLM OCR; olmOCR-2-7B for document-tuned.
        """
        params = {
            "model_name": {
                "type": "selector",
                "options": [
                    "Qwen/Qwen2.5-VL-7B-Instruct",
                    "allenai/olmOCR-2-7B-1025",
                ],
                "value": "Qwen/Qwen2.5-VL-7B-Instruct",
                "description": "7B VLM: Qwen = general; OlmOCR = document-tuned.",
            },
            "processor_name": {
                "type": "line_editor",
                "value": "",
                "description": "Processor (blank = use Qwen base for OlmOCR).",
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
                "description": "Max tokens per block (512 for paragraphs).",
            },
            "prompt": {
                "type": "line_editor",
                "value": "Extract all text from this image. Preserve layout and line breaks.",
                "description": "User prompt for OCR (used as text in chat).",
            },
            "use_bf16": {
                "type": "checkbox",
                "value": True,
                "description": "Use bfloat16 (recommended for 7B; saves VRAM).",
            },
            "low_vram": {
                "type": "checkbox",
                "value": True,
                "description": "Use CPU offload (device_map=auto). Enable on 11GB GPUs to avoid OOM; slower.",
            },
            "description": "Qwen2.5-VL 7B / OlmOCR 7B – heavyweight OCR (quality over speed).",
        }
        _load_model_keys = {"processor", "model"}

        def __init__(self, **params) -> None:
            super().__init__(**params)
            self.device = self.params["device"]["value"]
            self.processor = None
            self.model = None
            self._model_name = None
            self._processor_name = None
            self._device_for_inputs = None  # when using device_map="auto"

        def _load_model(self):
            model_name = (self.params.get("model_name") or {}).get("value", "Qwen/Qwen2.5-VL-7B-Instruct") or "Qwen/Qwen2.5-VL-7B-Instruct"
            proc_name = (self.params.get("processor_name") or {}).get("value", "").strip()
            if not proc_name:
                proc_name = "Qwen/Qwen2.5-VL-7B-Instruct" if "olmOCR" in model_name else model_name
            low_vram = self.params.get("low_vram", {}).get("value", True)
            if self.processor is not None and self._model_name == model_name and self._processor_name == proc_name:
                return
            self._model_name = model_name
            self._processor_name = proc_name
            self.processor = AutoProcessor.from_pretrained(proc_name)
            use_bf16 = self.params.get("use_bf16", {}).get("value", True)
            dtype = torch.bfloat16
            if not (torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)()):
                dtype = torch.float16
            if not use_bf16:
                dtype = torch.float16
            if low_vram:
                # device_map="auto" offloads layers to CPU when GPU is full (for 11GB cards).
                try:
                    self.model = AutoModelForImageTextToText.from_pretrained(
                        model_name, torch_dtype=dtype, device_map="auto"
                    )
                except Exception:
                    self.model = AutoModelForImageTextToText.from_pretrained(model_name, device_map="auto")
                self._device_for_inputs = next(self.model.parameters()).device
            else:
                try:
                    self.model = AutoModelForImageTextToText.from_pretrained(
                        model_name, torch_dtype=dtype, device_map=None
                    )
                except Exception:
                    self.model = AutoModelForImageTextToText.from_pretrained(model_name, device_map=None)
                self.model.to(self.device)
                self._device_for_inputs = None

        def _run_one(self, pil_img: "Image.Image") -> str:
            tmp_path = None
            try:
                fd, tmp_path = tempfile.mkstemp(suffix=".png")
                os.close(fd)
                pil_img.save(tmp_path)
                prompt = (self.params.get("prompt") or {}).get("value", "Extract all text from this image.") or "Extract all text from this image."
                # Qwen2.5-VL expects "url" (local path or file:// works in many versions)
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
                inp_device = self._device_for_inputs if self._device_for_inputs is not None else self.model.device
                if hasattr(inputs, "to"):
                    inputs = inputs.to(inp_device)
                else:
                    inputs = {k: v.to(inp_device) if hasattr(v, "to") else v for k, v in inputs.items()}
                max_tokens = 512
                mt = self.params.get("max_new_tokens", {})
                if isinstance(mt, dict):
                    try:
                        max_tokens = max(64, min(2048, int(mt.get("value", 512))))
                    except (TypeError, ValueError):
                        pass
                out = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
                prompt_len = inputs["input_ids"].shape[1]
                gen = out[0, prompt_len:]
                text = self.processor.decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
                return text
            except Exception as e:
                self.logger.warning(f"Qwen2.5-VL OCR failed: {e}")
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
            elif param_key in ("model_name", "processor_name", "use_bf16", "low_vram"):
                self.processor = None
                self.model = None
                self._model_name = None
                self._processor_name = None
                self._device_for_inputs = None
