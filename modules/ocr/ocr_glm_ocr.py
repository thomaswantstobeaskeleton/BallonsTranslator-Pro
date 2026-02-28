"""
GLM-OCR – lightweight document OCR (Z.ai, Hugging Face, 0.9B).
Multimodal: text, formula, table recognition. Chat-style API: image + "Text Recognition:" -> text.
Requires: pip install transformers torch pillow (accelerate optional).
Model: zai-org/GLM-OCR
"""
from typing import List
import os
import tempfile
import numpy as np
import cv2
from .base import OCRBase, register_OCR, DEFAULT_DEVICE, DEVICE_SELECTOR, TextBlock

_GLM_OCR_AVAILABLE = False
try:
    from transformers import AutoProcessor, GlmOcrForConditionalGeneration
    from PIL import Image
    _GLM_OCR_AVAILABLE = True
except ImportError as e:
    import logging
    logging.getLogger("BallonTranslator").debug(
        f"GLM-OCR not available: {e}. Install: pip install transformers torch pillow"
    )


def _cv2_to_pil_rgb(img: np.ndarray) -> "Image.Image":
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)


if _GLM_OCR_AVAILABLE:

    @register_OCR("glm_ocr")
    class GLMOCROCR(OCRBase):
        """
        GLM-OCR: 0.9B document OCR (text, formula, table). Chat-style; good for documents and mixed content.
        """
        params = {
            "model_name": {
                "type": "line_editor",
                "value": "zai-org/GLM-OCR",
                "description": "Hugging Face model id.",
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
                "description": "Use bfloat16 when available (saves VRAM).",
            },
            "description": "GLM-OCR – document OCR 0.9B (Hugging Face).",
        }
        _load_model_keys = {"processor", "model"}

        def __init__(self, **params) -> None:
            super().__init__(**params)
            self.device = self.params["device"]["value"]
            self.processor = None
            self.model = None
            self._model_name = None

        def _load_model(self):
            model_name = (self.params.get("model_name") or {}).get("value", "zai-org/GLM-OCR") or "zai-org/GLM-OCR"
            if self.processor is not None and self._model_name == model_name:
                return
            self._model_name = model_name
            self.processor = AutoProcessor.from_pretrained(model_name)
            use_bf16 = self.params.get("use_bf16", {}).get("value", True)
            import torch
            dtype = torch.bfloat16 if (use_bf16 and torch.cuda.is_bf16_supported()) else torch.float16
            self.model = GlmOcrForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map=None,
            )
            self.model.to(self.device)

        def _run_ocr_on_crop(self, pil_img: "Image.Image") -> str:
            """Run GLM-OCR on a single PIL image; returns recognized text."""
            tmp_path = None
            try:
                fd, tmp_path = tempfile.mkstemp(suffix=".png")
                os.close(fd)
                pil_img.save(tmp_path)
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "url": tmp_path},
                            {"type": "text", "text": "Text Recognition:"},
                        ],
                    }
                ]
                inputs = self.processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                inputs = inputs.to(self.model.device)
                inputs.pop("token_type_ids", None)
                max_tokens = 256
                mt = self.params.get("max_new_tokens", {})
                if isinstance(mt, dict):
                    try:
                        max_tokens = max(32, min(1024, int(mt.get("value", 256))))
                    except (TypeError, ValueError):
                        pass
                out = self.model.generate(**inputs, max_new_tokens=max_tokens)
                prompt_len = inputs["input_ids"].shape[1]
                text = self.processor.decode(out[0, prompt_len:], skip_special_tokens=True)
                if "Text Recognition:" in text:
                    text = text.split("Text Recognition:")[-1]
                for sep in ["assistant", "Assistant"]:
                    if sep in text:
                        text = text.split(sep)[-1]
                return text.strip()
            except Exception as e:
                self.logger.warning(f"GLM-OCR crop failed: {e}")
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
                text = self._run_ocr_on_crop(pil_img)
                blk.text = [text if text else ""]

        def ocr_img(self, img: np.ndarray) -> str:
            pil_img = _cv2_to_pil_rgb(img)
            return self._run_ocr_on_crop(pil_img)

        def updateParam(self, param_key: str, param_content):
            super().updateParam(param_key, param_content)
            if param_key == "device":
                self.device = self.params["device"]["value"]
                if self.model is not None:
                    self.model.to(self.device)
            elif param_key in ("model_name", "use_bf16"):
                self.processor = None
                self.model = None
                self._model_name = None
