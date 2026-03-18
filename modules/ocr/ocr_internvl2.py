"""
InternVL2 – 8B/2B VLM OCR (Hugging Face). Strong document/chart/OCR (e.g. OCRBench 794).
Quality over speed. Requires: pip install transformers torch pillow accelerate
"""
from typing import List
import numpy as np
import cv2
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image

from .base import OCRBase, register_OCR, DEVICE_SELECTOR, TextBlock

_INTERNVL_AVAILABLE = False
try:
    from transformers import AutoModel, AutoTokenizer
    _INTERNVL_AVAILABLE = True
except ImportError as e:
    import logging
    logging.getLogger("BallonTranslator").debug(
        f"InternVL2 OCR not available: {e}. Install: pip install transformers torch pillow accelerate"
    )


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _cv2_to_pil_rgb(img: np.ndarray) -> "Image.Image":
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)


def _build_transform(input_size: int = 448):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _dynamic_preprocess(image: "Image.Image", image_size: int = 448, max_num: int = 6):
    """Adapted from InternVL2 README: tile image for variable resolution."""
    import math
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(1, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1)
        if i * j <= max_num and i * j >= 1
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
    target_width = image_size * best_ratio[0]
    target_height = image_size * best_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed = []
    for i in range(best_ratio[0] * best_ratio[1]):
        box = (
            (i % best_ratio[0]) * image_size,
            (i // best_ratio[0]) * image_size,
            ((i % best_ratio[0]) + 1) * image_size,
            ((i // best_ratio[0]) + 1) * image_size,
        )
        processed.append(resized_img.crop(box))
    return processed


if _INTERNVL_AVAILABLE:

    @register_OCR("internvl2_ocr")
    class InternVL2OCROCR(OCRBase):
        """
        InternVL2 8B or 2B: strong document/chart/OCR (OCRBench 794). Quality over speed.
        Uses OpenGVLab/InternVL2-8B or InternVL2-2B; trust_remote_code.
        """
        params = {
            "model_name": {
                "type": "selector",
                "options": [
                    "OpenGVLab/InternVL2-8B",
                    "OpenGVLab/InternVL2-2B",
                    "OpenGVLab/InternVL2-4B",
                ],
                "value": "OpenGVLab/InternVL2-8B",
                "description": "InternVL2 model (8B = best quality, 2B = less VRAM).",
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
            "image_size": {
                "type": "line_editor",
                "value": 448,
                "description": "Vision input size (448 default for InternVL2).",
            },
            "max_num_tiles": {
                "type": "line_editor",
                "value": 6,
                "description": "Max tiles for dynamic preprocessing (1 = single crop).",
            },
            "max_new_tokens": {
                "type": "line_editor",
                "value": 512,
                "description": "Max new tokens per block.",
            },
            "use_bf16": {
                "type": "checkbox",
                "value": True,
                "description": "Use bfloat16 when available.",
            },
            "description": "InternVL2 8B/2B – document/chart/OCR (HF, trust_remote_code).",
        }
        _load_model_keys = {"model", "tokenizer"}

        def __init__(self, **params) -> None:
            super().__init__(**params)
            self.device = self.params["device"]["value"]
            self.model = None
            self.tokenizer = None
            self._model_name = None
            self._transform = None

        def _load_model(self):
            model_name = (self.params.get("model_name") or {}).get("value", "OpenGVLab/InternVL2-8B") or "OpenGVLab/InternVL2-8B"
            if self.model is not None and self._model_name == model_name:
                return
            self._model_name = model_name
            # Prefer fast tokenizer when available (user preference).
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            use_bf16 = self.params.get("use_bf16", {}).get("value", True)
            dtype = torch.bfloat16
            if not (torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)()):
                dtype = torch.float16
            if not use_bf16:
                dtype = torch.float16
            self.model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            try:
                self.model = self.model.eval().to(self.device)
            except Exception:
                self.model = self.model.to(self.device).eval()
            isize = 448
            p = self.params.get("image_size", {})
            if isinstance(p, dict):
                try:
                    isize = max(224, min(448, int(p.get("value", 448))))
                except (TypeError, ValueError):
                    pass
            self._transform = _build_transform(isize)

        def _run_one(self, pil_img: "Image.Image") -> str:
            try:
                max_num = 6
                mn = self.params.get("max_num_tiles", {})
                if isinstance(mn, dict):
                    try:
                        max_num = max(1, min(12, int(mn.get("value", 6))))
                    except (TypeError, ValueError):
                        pass
                image_size = 448
                p = self.params.get("image_size", {})
                if isinstance(p, dict):
                    try:
                        image_size = max(224, min(448, int(p.get("value", 448))))
                    except (TypeError, ValueError):
                        pass
                tiles = _dynamic_preprocess(pil_img, image_size=image_size, max_num=max_num)
                pixel_values = torch.stack([self._transform(t) for t in tiles])
                pixel_values = pixel_values.to(self.model.dtype).to(self.device)
                prompt = (self.params.get("prompt") or {}).get("value", "Extract all text from this image.") or "Extract all text from this image."
                question = "<image>\n" + prompt
                max_new_tokens = 512
                mt = self.params.get("max_new_tokens", {})
                if isinstance(mt, dict):
                    try:
                        max_new_tokens = max(64, min(2048, int(mt.get("value", 512))))
                    except (TypeError, ValueError):
                        pass
                generation_config = dict(max_new_tokens=max_new_tokens, do_sample=False)
                with torch.inference_mode():
                    response, _ = self.model.chat(
                        self.tokenizer, pixel_values, question,
                        generation_config=generation_config,
                        history=None, return_history=True,
                    )
                return (response or "").strip()
            except Exception as e:
                self.logger.warning(f"InternVL2 OCR failed: {e}")
                return ""

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
            elif param_key in ("model_name", "use_bf16", "image_size"):
                self.model = None
                self.tokenizer = None
                self._model_name = None
