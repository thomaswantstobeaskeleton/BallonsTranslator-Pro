"""
Chandra OCR – 9B document OCR (chandra-ocr package). Top olmOCR benchmark; layout, tables, math.
Requires: pip install chandra-ocr transformers torch pillow
Heavyweight; prefer quality over speed.
"""
from typing import List
import logging
import numpy as np
import cv2
from .base import OCRBase, register_OCR, DEVICE_SELECTOR, TextBlock

# Loggers that emit noisy warnings during Chandra load (tie_weights, tensor_parallel, big_modeling).
_CHANDRA_LOAD_LOGGERS = (
    "transformers.modeling_utils",
    "transformers",
    "accelerate.tensor_parallel",
    "accelerate.big_modeling",
    "accelerate",
    "tensor_parallel",  # also used as logger name in some versions
)

_CHANDRA_AVAILABLE = False
try:
    from chandra.model import InferenceManager
    from chandra.model.schema import BatchInputItem
    from PIL import Image
    _CHANDRA_AVAILABLE = True
except ImportError as e:
    import logging
    logging.getLogger("BallonTranslator").debug(
        f"Chandra OCR not available: {e}. Install: pip install chandra-ocr transformers torch pillow"
    )


def _cv2_to_pil_rgb(img: np.ndarray) -> "Image.Image":
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)


if _CHANDRA_AVAILABLE:

    @register_OCR("chandra_ocr")
    class ChandraOCROCR(OCRBase):
        """
        Chandra 9B: document OCR (layout, tables, math, 40+ languages).
        Uses chandra-ocr package; top olmOCR benchmark. Quality over speed.
        """
        params = {
            "prompt_type": {
                "type": "selector",
                "options": ["ocr_layout", "ocr_plain"],
                "value": "ocr_layout",
                "description": "ocr_layout = markdown/layout; ocr_plain = plain text.",
            },
            "device": DEVICE_SELECTOR(),
            "crop_padding": {
                "type": "line_editor",
                "value": 4,
                "description": "Pixels to add around each box when cropping (0–24).",
            },
            "use_bf16": {
                "type": "checkbox",
                "value": True,
                "description": "Use bfloat16 when available (saves VRAM).",
            },
            "description": "Chandra 9B OCR (chandra-ocr). Layout, tables, math; install: pip install chandra-ocr",
        }
        _load_model_keys = {"manager"}

        def __init__(self, **params) -> None:
            super().__init__(**params)
            self.manager = None
            self._chandra_first_inference = True

        def _load_model(self):
            if self.manager is not None:
                return
            # Suppress noisy warnings from Chandra's deps (tie_weights, tensor_parallel, big_modeling).
            saved = []
            for name in _CHANDRA_LOAD_LOGGERS:
                log = logging.getLogger(name)
                saved.append((log, log.getEffectiveLevel()))
                log.setLevel(logging.ERROR)
            try:
                self.manager = InferenceManager(method="hf")
            finally:
                for log, level in saved:
                    log.setLevel(level)

        def _run_one(self, pil_img: "Image.Image") -> str:
            try:
                if self._chandra_first_inference:
                    self.logger.info(
                        "Chandra OCR: first inference running (may take several minutes on 11GB VRAM with offload)."
                    )
                prompt_type = (self.params.get("prompt_type") or {}).get("value", "ocr_layout") or "ocr_layout"
                # chandra-ocr PROMPT_MAPPING uses "ocr_layout" and "ocr" (not "ocr_plain")
                if prompt_type == "ocr_plain":
                    prompt_type = "ocr"
                batch = [BatchInputItem(image=pil_img, prompt_type=prompt_type)]
                results = self.manager.generate(batch)
                if self._chandra_first_inference:
                    self.logger.info("Chandra OCR: first inference done.")
                    self._chandra_first_inference = False
                if not results:
                    return ""
                return (results[0].markdown or "").strip()
            except Exception as e:
                self.logger.warning(f"Chandra OCR failed: {e}")
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
            if param_key in ("prompt_type",):
                pass
