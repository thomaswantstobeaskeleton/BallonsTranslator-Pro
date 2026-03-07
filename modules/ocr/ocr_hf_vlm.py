"""
Generic Hugging Face VLM OCR – use any HF vision-language model that supports
image + text chat (e.g. Qwen2-VL, Qwen3-VL, InternVL, OlmOCR) via model ID.
Requires: pip install transformers torch pillow accelerate
Use with any detector (e.g. hf_object_det). For Qwen3-VL sizes use qwen3vl_ocr for a dedicated selector.
"""
from typing import List
import os
import tempfile
import numpy as np
import cv2
from .base import OCRBase, register_OCR, DEVICE_SELECTOR, TextBlock

_HF_VLM_AVAILABLE = False
try:
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from PIL import Image
    import torch
    _HF_VLM_AVAILABLE = True
except ImportError as e:
    import logging
    logging.getLogger("BallonTranslator").debug(
        f"HF VLM OCR not available: {e}. Install: pip install transformers torch pillow accelerate"
    )


def _cv2_to_pil_rgb(img: np.ndarray) -> "Image.Image":
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)


if _HF_VLM_AVAILABLE:

    @register_OCR("hf_vlm_ocr")
    class HFVLMOCR(OCRBase):
        """
        Generic HF VLM OCR: set any Hugging Face model ID that supports image+text chat
        (Qwen2-VL, Qwen3-VL, InternVL, OlmOCR, etc.). Use with any detector.
        """
        params = {
            "model_id": {
                "type": "line_editor",
                "value": "Qwen/Qwen3-VL-2B-Instruct",
                "description": "Hugging Face model id (e.g. Qwen/Qwen3-VL-2B-Instruct, OpenGVLab/InternVL3-2B-hf, allenai/olmOCR-2-7B-1025).",
            },
            "processor_id": {
                "type": "line_editor",
                "value": "",
                "description": "Processor id (blank = same as model_id). Set if processor repo differs.",
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
                "description": "OCR prompt (chat user message).",
            },
            "use_bf16": {
                "type": "checkbox",
                "value": True,
                "description": "Use bfloat16 when available.",
            },
            "low_vram": {
                "type": "checkbox",
                "value": False,
                "description": "Use device_map=auto (CPU offload). Slower, for limited VRAM.",
            },
            "description": "Generic HF VLM OCR – any model id with image+text chat. Use with hf_object_det or other detector.",
        }
        _load_model_keys = {"processor", "model"}

        def __init__(self, **params) -> None:
            super().__init__(**params)
            self.device = (self.params.get("device") or {}).get("value", "cpu")
            self.processor = None
            self.model = None
            self._model_id = None
            self._processor_id = None
            self._device_for_inputs = None

        def _load_model(self):
            model_id = (self.params.get("model_id") or {}).get("value", "").strip() or "Qwen/Qwen3-VL-2B-Instruct"
            proc_id = (self.params.get("processor_id") or {}).get("value", "").strip() or model_id
            dev = (self.params.get("device") or {}).get("value", "cpu")
            if dev in ("cuda", "gpu") and torch.cuda.is_available():
                dev = "cuda"
            else:
                dev = "cpu"
            if self.processor is not None and self._model_id == model_id and self._processor_id == proc_id:
                if self.model is not None and hasattr(self.model, "to") and dev != "cpu":
                    try:
                        self.model.to(dev)
                    except Exception:
                        pass
                self.device = dev
                return
            self._model_id = model_id
            self._processor_id = proc_id
            self.device = dev
            use_bf16 = self.params.get("use_bf16", {}).get("value", True)
            dtype = torch.bfloat16
            if not (torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)()):
                dtype = torch.float16
            if not use_bf16:
                dtype = torch.float16
            low_vram = self.params.get("low_vram", {}).get("value", False)
            self.processor = AutoProcessor.from_pretrained(proc_id)
            load_kw = {"torch_dtype": dtype}
            if low_vram:
                load_kw["device_map"] = "auto"
                self.model = AutoModelForImageTextToText.from_pretrained(model_id, **load_kw)
                self._device_for_inputs = next(self.model.parameters()).device
            else:
                self.model = AutoModelForImageTextToText.from_pretrained(model_id, **load_kw)
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
                self.logger.warning(f"HF VLM OCR failed: {e}")
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
            elif param_key in ("model_id", "processor_id", "use_bf16", "low_vram"):
                self.processor = None
                self.model = None
                self._model_id = None
                self._processor_id = None
                self._device_for_inputs = None
