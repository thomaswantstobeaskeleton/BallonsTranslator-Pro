"""
MMOCR text recognition on crops – use MMOCR recognizer for per-block text.
Pairs with mmocr_det for full MMOCR pipeline (DBNet + CRNN/SAR etc.).
Requires: pip install openmim && mim install mmengine mmcv mmdet mmocr
"""
from typing import List
import numpy as np
import cv2

from .base import OCRBase, register_OCR, DEVICE_SELECTOR, TextBlock

_MMOCR_REC_AVAILABLE = False
try:
    from mmocr.apis import TextRecInferencer
    _MMOCR_REC_AVAILABLE = True
except ImportError:
    import logging
    logging.getLogger("BallonTranslator").debug(
        "MMOCR TextRecInferencer not available. Install: mim install mmengine mmcv mmdet mmocr"
    )


if _MMOCR_REC_AVAILABLE:

    @register_OCR("mmocr_ocr")
    class MMOCROCR(OCRBase):
        """
        MMOCR recognition on detector crops. Use with mmocr_det for full pipeline.
        Same dependencies as mmocr_det (mmengine, mmcv, mmdet, mmocr).
        """
        params = {
            "rec_model": {
                "type": "selector",
                "options": ["SAR", "CRNN", "CRNN_TPS", "ABINet", "NRTR"],
                "value": "SAR",
                "description": "MMOCR recognition model. SAR/CRNN common for document.",
            },
            "device": DEVICE_SELECTOR(),
            "crop_padding": {
                "type": "line_editor",
                "value": 4,
                "description": "Pixels to add around each box when cropping (0–24).",
            },
            "description": "MMOCR recognition on crops (pair with mmocr_det). Install: mim install mmengine mmcv mmdet mmocr",
        }
        _load_model_keys = {"inferencer"}

        def __init__(self, **params) -> None:
            super().__init__(**params)
            self.inferencer = None
            self._rec_model = None
            self._device = None

        def _load_model(self):
            rec_model = (self.params.get("rec_model") or {}).get("value", "SAR") or "SAR"
            device = (self.params.get("device") or {}).get("value", "cpu")
            if device in ("cuda", "gpu"):
                device = "cuda:0"
            if self.inferencer is not None and self._rec_model == rec_model and self._device == device:
                return
            self._rec_model = rec_model
            self._device = device
            try:
                self.inferencer = TextRecInferencer(model=rec_model, device=device)
            except (ImportError, OSError) as e:
                err_msg = str(e)
                if "DLL load failed" in err_msg or "_ext" in err_msg or "procedure could not be found" in err_msg:
                    raise RuntimeError(
                        "MMOCR rec failed: mmcv was built for a different PyTorch version. "
                        "Use another OCR (e.g. surya_ocr, paddle_rec_v5). See doc/INSTALL_MMOCR.md."
                    ) from e
                raise

        def _ocr_blk_list(self, img: np.ndarray, blk_list: List[TextBlock], *args, **kwargs) -> None:
            if self.inferencer is None:
                return
            im_h, im_w = img.shape[:2]
            if img.ndim == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            pad = max(0, min(24, int(self.params.get("crop_padding", {}).get("value", 4))))
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
                if not (0 <= x1 < x2 <= im_w and 0 <= y1 < y2 <= im_h):
                    blk.text = [""]
                    continue
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    blk.text = [""]
                    continue
                crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR) if crop.ndim == 3 else crop
                try:
                    result = self.inferencer(crop_bgr, return_datasamples=False)
                except Exception as e:
                    self.logger.debug(f"MMOCR rec failed on crop: {e}")
                    blk.text = [""]
                    continue
                preds = result.get("predictions") or []
                if not preds:
                    blk.text = [""]
                    continue
                first = preds[0]
                rec_texts = first.get("rec_texts") or first.get("rec_texts_") or []
                if isinstance(rec_texts, str):
                    rec_texts = [rec_texts]
                parts = [str(t).strip() for t in rec_texts if t]
                blk.text = ["\n".join(parts)] if parts else [""]

        def ocr_img(self, img: np.ndarray) -> str:
            if self.inferencer is None:
                return ""
            if img.ndim == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            try:
                result = self.inferencer(img, return_datasamples=False)
            except Exception as e:
                self.logger.debug(f"MMOCR rec failed: {e}")
                return ""
            preds = result.get("predictions") or []
            if not preds:
                return ""
            rec_texts = preds[0].get("rec_texts") or preds[0].get("rec_texts_") or []
            if isinstance(rec_texts, str):
                rec_texts = [rec_texts]
            return "\n".join(str(t).strip() for t in rec_texts if t)

        def updateParam(self, param_key: str, param_content):
            super().updateParam(param_key, param_content)
            if param_key in ("rec_model", "device"):
                self.inferencer = None
                self._rec_model = None
                self._device = None
