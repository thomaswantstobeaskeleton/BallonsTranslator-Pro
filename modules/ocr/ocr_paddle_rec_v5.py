"""
PaddleOCR PP-OCRv5 text recognition only – use with any detector (e.g. paddle_det_v5).
PaddleOCR 3.x TextRecognition module; PP-OCRv5_mobile_rec or PP-OCRv5_server_rec.
Chinese, English, Traditional Chinese, Japanese; handwriting, vertical text.
Requires: paddleocr 3.x, paddlepaddle. Pair with paddle_det_v5 for full PP-OCRv5 pipeline.
"""
import os
import tempfile
import numpy as np
import cv2
from typing import List

from .base import OCRBase, register_OCR, DEVICE_SELECTOR, TextBlock
from utils.ocr_preprocess import preprocess_for_ocr

os.environ.setdefault("PPOCR_HOME", os.path.join("data", "models", "paddle-ocr"))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_PADDLE_REC_V5_AVAILABLE = False
try:
    from paddleocr import TextRecognition
    _PADDLE_REC_V5_AVAILABLE = True
except ImportError:
    import logging
    logging.getLogger("BallonTranslator").debug(
        "PaddleOCR TextRecognition (PP-OCRv5) not available. Install paddleocr 3.x and paddlepaddle."
    )


if _PADDLE_REC_V5_AVAILABLE:

    @register_OCR("paddle_rec_v5")
    class PaddleRecV5OCR(OCRBase):
        """
        PP-OCRv5 recognition only. Use with paddle_det_v5 (or any detector) for full pipeline.
        Chinese, English, Traditional Chinese, Japanese; handwriting, vertical.
        """
        params = {
            "model_name": {
                "type": "selector",
                "options": ["PP-OCRv5_mobile_rec", "PP-OCRv5_server_rec"],
                "value": "PP-OCRv5_mobile_rec",
                "description": "PP-OCRv5 mobile (faster) or server (higher accuracy).",
            },
            "device": DEVICE_SELECTOR(),
            "drop_score": {
                "type": "line_editor",
                "value": 0.5,
                "description": "Min confidence for recognized text (0–1).",
            },
            "crop_padding": {
                "type": "line_editor",
                "value": 4,
                "description": "Pixels to add around each crop (0–24). Like Ocean OCR; helps recognizer see full text.",
            },
            "description": "PaddleOCR PP-OCRv5 recognition only (paddleocr 3.x). Pair with paddle_det_v5.",
        }
        _load_model_keys = {"model"}

        def __init__(self, **params) -> None:
            super().__init__(**params)
            self.model = None
            self._model_name = None
            self._device = None
            self.crop_padding = max(0, min(24, int(self.params.get("crop_padding", {}).get("value", 4) or 4)))

        def _load_model(self):
            model_name = (
                (self.params.get("model_name") or {}).get("value", "PP-OCRv5_mobile_rec")
                or "PP-OCRv5_mobile_rec"
            )
            dev = (self.params.get("device") or {}).get("value", "cpu")
            device = "gpu:0" if dev in ("cuda", "gpu") else "cpu"
            if self.model is not None and self._model_name == model_name and self._device == device:
                return
            self._model_name = model_name
            self._device = device
            try:
                self.model = TextRecognition(model_name=model_name, device=device)
            except Exception as e:
                self.logger.error(f"Paddle OCR v5 rec load failed: {e}")
                raise

        def _run_one(self, img: np.ndarray) -> str:
            if img.size == 0:
                return ""
            try:
                output = list(self.model.predict(input=img, batch_size=1))
                if not output:
                    return ""
                res = output[0]
                data = res.json() if hasattr(res, "json") and callable(getattr(res, "json")) else getattr(res, "json", res)
                if isinstance(data, dict):
                    inner = data.get("res", data)
                    if isinstance(inner, dict):
                        text = inner.get("rec_text", "")
                        score = inner.get("rec_score", 1.0)
                        drop = 0.5
                        try:
                            d = self.params.get("drop_score", {})
                            if isinstance(d, dict):
                                drop = float(d.get("value", 0.5))
                        except (TypeError, ValueError):
                            pass
                        if score >= drop:
                            return text or ""
                        return text or ""
                return ""
            except Exception as e:
                self.logger.warning(f"Paddle rec v5 failed: {e}")
                return ""

        def _ocr_blk_list(self, img: np.ndarray, blk_list: List[TextBlock], *args, **kwargs) -> None:
            im_h, im_w = img.shape[:2]
            upscale_min_side = 0
            try:
                from utils.config import pcfg
                upscale_min_side = int(getattr(pcfg.module, "ocr_upscale_min_side", 0) or 0)
            except Exception:
                pass
            if upscale_min_side <= 0:
                upscale_min_side = 32
            pad = max(0, min(24, int(getattr(self, "crop_padding", 4) or 4)))
            for blk in blk_list:
                if getattr(blk, "lines", None) and len(blk.lines) > 1:
                    text_parts = []
                    for line_pts in blk.lines:
                        if not isinstance(line_pts, (list, tuple)) or len(line_pts) < 4:
                            continue
                        xs = [p[0] for p in line_pts if isinstance(p, (list, tuple)) and len(p) >= 2]
                        ys = [p[1] for p in line_pts if isinstance(p, (list, tuple)) and len(p) >= 2]
                        if not xs or not ys:
                            continue
                        x1 = max(0, int(min(xs)) - pad)
                        y1 = max(0, int(min(ys)) - pad)
                        x2 = min(im_w, int(max(xs)) + pad)
                        y2 = min(im_h, int(max(ys)) + pad)
                        if not (0 <= x1 < x2 <= im_w and 0 <= y1 < y2 <= im_h):
                            text_parts.append("")
                            continue
                        crop = img[y1:y2, x1:x2]
                        if crop.size == 0:
                            text_parts.append("")
                            continue
                        if crop.ndim == 3 and crop.shape[2] == 4:
                            crop = cv2.cvtColor(crop, cv2.COLOR_RGBA2RGB)
                        crop = np.ascontiguousarray(crop)
                        crop = preprocess_for_ocr(crop, recipe="none", upscale_min_side=upscale_min_side)
                        t = self._run_one(crop)
                        text_parts.append(t if t else "")
                    blk.text = text_parts if text_parts else [""]
                    continue
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
                if not (0 <= x1 < x2 <= im_w and 0 <= y1 < y2 <= im_h):
                    blk.text = [""]
                    continue
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    blk.text = [""]
                    continue
                if crop.ndim == 3 and crop.shape[2] == 4:
                    crop = cv2.cvtColor(crop, cv2.COLOR_RGBA2RGB)
                crop = np.ascontiguousarray(crop)
                crop = preprocess_for_ocr(crop, recipe="none", upscale_min_side=upscale_min_side)
                text = self._run_one(crop)
                blk.text = [text if text else ""]

        def ocr_img(self, img: np.ndarray) -> str:
            if img.ndim == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            return self._run_one(img)

        def updateParam(self, param_key: str, param_content):
            super().updateParam(param_key, param_content)
            if param_key == "crop_padding":
                self.crop_padding = max(0, min(24, int(self.params.get("crop_padding", {}).get("value", 4) or 4)))
            if param_key in ("model_name", "device"):
                self.model = None
                self._model_name = None
                self._device = None
