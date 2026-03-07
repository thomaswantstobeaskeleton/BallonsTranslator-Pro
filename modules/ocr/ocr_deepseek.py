"""
DeepSeek-OCR – heavyweight document OCR (Hugging Face, trust_remote_code).
Multilingual, document layout, tables. Quality over speed. Prompt "<image>\\nFree OCR. " for plain text.
Requires: pip install transformers torch pillow accelerate (flash-attn optional).
"""
from typing import List
import os
import re
import tempfile
import numpy as np
import cv2
from .base import OCRBase, register_OCR, DEVICE_SELECTOR, TextBlock

_DEEPSEEK_OCR_AVAILABLE = False
try:
    from transformers import AutoModel, AutoTokenizer
    from PIL import Image
    import torch
    _DEEPSEEK_OCR_AVAILABLE = True
except ImportError as e:
    import logging
    logging.getLogger("BallonTranslator").debug(
        f"DeepSeek-OCR not available: {e}. Install: pip install transformers torch pillow accelerate"
    )


def _cv2_to_pil_rgb(img: np.ndarray) -> "Image.Image":
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)


def _clean_deepseek_ocr_text(s: str) -> str:
    """Strip DeepSeek-OCR special tokens for plain text."""
    if not s or not isinstance(s, str):
        return ""
    # Remove ref/det/other tags: <|ref|>...<|/ref|>, <|det|>...<|/det|>, etc.
    s = re.sub(r"<\|[^|]+\|[^>]*>", "", s)
    s = re.sub(r"<\|[^|]+\|>", "", s)
    return s.strip()


if _DEEPSEEK_OCR_AVAILABLE:

    @register_OCR("deepseek_ocr")
    class DeepSeekOCROCR(OCRBase):
        """
        DeepSeek-OCR: heavyweight document OCR. Multilingual, layout, tables.
        Use 'Free OCR' prompt for plain text per region. Prefer quality over speed.
        """
        params = {
            "model_name": {
                "type": "line_editor",
                "value": "deepseek-ai/DeepSeek-OCR",
                "description": "Hugging Face model id (requires trust_remote_code).",
            },
            "device": DEVICE_SELECTOR(),
            "crop_padding": {
                "type": "line_editor",
                "value": 4,
                "description": "Pixels to add around each box when cropping (0–24).",
            },
            "prompt": {
                "type": "line_editor",
                "value": "<image>\nFree OCR. ",
                "description": "Prompt for OCR (use 'Free OCR.' for plain text).",
            },
            "base_size": {
                "type": "line_editor",
                "value": 640,
                "description": "Base size for inference (512/640/1024; smaller = faster for crops).",
            },
            "image_size": {
                "type": "line_editor",
                "value": 640,
                "description": "Image size for inference.",
            },
            "use_bf16": {
                "type": "checkbox",
                "value": True,
                "description": "Use bfloat16 when available (recommended).",
            },
            "use_flash_attn": {
                "type": "checkbox",
                "value": False,
                "description": "Use Flash Attention 2 (faster; requires flash-attn).",
            },
            "description": "DeepSeek-OCR (HF, trust_remote_code). Multilingual document OCR.",
        }
        _load_model_keys = {"model", "tokenizer"}

        def __init__(self, **params) -> None:
            super().__init__(**params)
            self.device = self.params["device"]["value"]
            self.model = None
            self.tokenizer = None
            self._model_name = None

        def _load_model(self):
            model_name = (self.params.get("model_name") or {}).get("value", "deepseek-ai/DeepSeek-OCR") or "deepseek-ai/DeepSeek-OCR"
            if self.model is not None and self._model_name == model_name:
                return
            self._model_name = model_name
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            use_bf16 = self.params.get("use_bf16", {}).get("value", True)
            use_flash = self.params.get("use_flash_attn", {}).get("value", False)
            dtype = torch.bfloat16 if (use_bf16 and torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)()) else torch.float16
            kwargs = {"torch_dtype": dtype, "trust_remote_code": True}
            if use_flash:
                try:
                    kwargs["attn_implementation"] = "flash_attention_2"
                except Exception:
                    pass
            try:
                self.model = AutoModel.from_pretrained(model_name, **kwargs)
            except Exception as e:
                self.logger.warning(f"DeepSeek-OCR load with attn failed: {e}; retrying without flash_attn.")
                kwargs.pop("attn_implementation", None)
                self.model = AutoModel.from_pretrained(model_name, **kwargs)
            self.model = self.model.eval().to(self.device)
            if self.device == "cuda" and dtype == torch.bfloat16:
                self.model = self.model.to(torch.bfloat16)

        def _run_one(self, pil_img: "Image.Image") -> str:
            tmp_path = None
            try:
                fd, tmp_path = tempfile.mkstemp(suffix=".png")
                os.close(fd)
                pil_img.save(tmp_path)
                prompt = (self.params.get("prompt") or {}).get("value", "<image>\nFree OCR. ") or "<image>\nFree OCR. "
                base_size = 640
                image_size = 640
                for key, default in (("base_size", 640), ("image_size", 640)):
                    p = self.params.get(key, {})
                    if isinstance(p, dict):
                        try:
                            val = int(p.get("value", default))
                            if key == "base_size":
                                base_size = max(256, min(1280, val))
                            else:
                                image_size = max(256, min(1280, val))
                        except (TypeError, ValueError):
                            pass
                out_dir = tempfile.mkdtemp()
                try:
                    res = self.model.infer(
                        self.tokenizer,
                        prompt=prompt,
                        image_file=tmp_path,
                        output_path=out_dir,
                        base_size=base_size,
                        image_size=image_size,
                        crop_mode=False,
                        save_results=False,
                        test_compress=False,
                        eval_mode=True,
                    )
                except TypeError:
                    res = self.model.infer(
                        self.tokenizer,
                        prompt=prompt,
                        image_file=tmp_path,
                        output_path=out_dir,
                        base_size=base_size,
                        image_size=image_size,
                        crop_mode=False,
                        save_results=False,
                    )
                if res is None:
                    return ""
                text = res if isinstance(res, str) else getattr(res, "text", str(res))
                return _clean_deepseek_ocr_text(text)
            except Exception as e:
                self.logger.warning(f"DeepSeek-OCR failed: {e}")
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
                self.model = None
                self.tokenizer = None
                self._model_name = None
