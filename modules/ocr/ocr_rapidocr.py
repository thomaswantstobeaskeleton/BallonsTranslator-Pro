"""
RapidOCR recognition on detector crops. Pairs with RapidOCR detector or any detector.
Uses perspective warp on quad regions (EasyScanlate-style) for better recognition.
Requires: pip install rapidocr-onnxruntime (or rapidocr with onnxruntime).
Optional: place rec model and dict in data/models/rapidocr/ for custom paths (e.g. Korean PP-OCRv5).
"""
import os
from typing import List

import cv2
import numpy as np

from .base import OCRBase, register_OCR, TextBlock
from utils.ocr_preprocess import preprocess_for_ocr

_RAPIDOCR_AVAILABLE = False
_RapidOCR = None
_EngineType = None
try:
    from rapidocr import RapidOCR, EngineType
    _RapidOCR = RapidOCR
    _EngineType = EngineType
    _RAPIDOCR_AVAILABLE = True
except ImportError:
    try:
        from rapidocr_onnxruntime import RapidOCR
        _RapidOCR = RapidOCR
        _RAPIDOCR_AVAILABLE = True
    except ImportError:
        import logging
        logging.getLogger("BallonsTranslator").debug(
            "RapidOCR not available for OCR. Install: pip install rapidocr-onnxruntime"
        )


def _get_rotate_crop_image(img: np.ndarray, points) -> np.ndarray:
    """
    Crop and warp image by quad (perspective transform). Same idea as EasyScanlate's get_rotate_crop_image.
    """
    points = np.array(points, dtype=np.float32)
    if points.ndim == 1 or points.size < 8:
        x1, y1 = int(points[0]), int(points[1])
        x2, y2 = int(points[2]), int(points[3])
        return img[max(0, y1):min(img.shape[0], y2), max(0, x1):min(img.shape[1], x2)]
    if points.shape[0] != 4:
        points = points.reshape(-1, 2)
    x_sorted = points[np.argsort(points[:, 0]), :]
    left_most, right_most = x_sorted[:2, :], x_sorted[2:, :]
    left_most = left_most[np.argsort(left_most[:, 1]), :]
    tl, bl = left_most[0], left_most[1]
    right_most = right_most[np.argsort(right_most[:, 1]), :]
    tr, br = right_most[0], right_most[1]
    pts = np.array([tl, tr, br, bl], dtype=np.float32)
    width_A = np.linalg.norm(br - bl)
    width_B = np.linalg.norm(tr - tl)
    max_width = max(int(width_A), int(width_B))
    height_A = np.linalg.norm(tr - br)
    height_B = np.linalg.norm(tl - bl)
    max_height = max(int(height_A), int(height_B))
    dst_pts = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts, dst_pts)
    warped = cv2.warpPerspective(
        img, M, (max_width, max_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    return warped


def _looks_like_points(obj) -> bool:
    """True if obj looks like polygon points [[x,y], [x,y], ...] rather than text."""
    if not isinstance(obj, (list, tuple)) or len(obj) < 4:
        return False
    for p in obj[:4]:
        if not isinstance(p, (list, tuple)) or len(p) < 2:
            return False
        try:
            float(p[0])
            float(p[1])
        except (TypeError, ValueError):
            return False
    return True


def _parse_rec_output(rec_out) -> tuple:
    """Return (text, score) from RapidOCR recognition output. Handles (text, score), (score, text), or (points, text, score)."""
    text, score = "", 0.0
    try:
        if isinstance(rec_out, tuple) and rec_out[0] and len(rec_out[0]) > 0:
            item = rec_out[0][0]
            if isinstance(item, (list, tuple)):
                text, score = _item_to_text_score(item)
            else:
                text = str(item)
        elif hasattr(rec_out, "txts") and rec_out.txts and len(rec_out.txts) > 0:
            text = rec_out.txts[0]
            score = getattr(rec_out, "scores", [0.0])[0]
        elif isinstance(rec_out, list) and len(rec_out) > 0:
            item = rec_out[0]
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                text, score = _item_to_text_score(item)
            elif isinstance(item, (list, tuple)) and len(item) > 0:
                text = item[0]
            else:
                text = str(item) if item else ""
    except (ValueError, TypeError, IndexError):
        # Fallback: avoid treating (points, text, score) as text=points
        if isinstance(rec_out, tuple) and rec_out[0] and len(rec_out[0]) > 0:
            item = rec_out[0][0]
            if isinstance(item, (list, tuple)) and len(item) >= 3 and _looks_like_points(item[0]):
                text = str(item[1]).strip() if item[1] else ""
                try:
                    score = float(item[2])
                except (TypeError, ValueError):
                    score = 0.0
            else:
                text = item[0] if isinstance(item, (list, tuple)) and len(item) > 0 else str(item)
                score = 0.0
        elif isinstance(rec_out, list) and len(rec_out) > 0:
            item = rec_out[0]
            if isinstance(item, (list, tuple)) and len(item) >= 3 and _looks_like_points(item[0]):
                text = str(item[1]).strip() if item[1] else ""
                try:
                    score = float(item[2])
                except (TypeError, ValueError):
                    score = 0.0
            else:
                text = item[0] if isinstance(item, (list, tuple)) and len(item) > 0 else str(item)
                score = 0.0
        else:
            score = 0.0
    return (str(text).strip() if text else "", float(score))


def _item_to_text_score(item) -> tuple:
    """From [a, b] or [points, text, score], return (text, score). Handles (text, score) or (score, text) order."""
    if len(item) >= 3 and _looks_like_points(item[0]):
        # (points, text, score) format: use text and score, not points as text
        try:
            return (str(item[1]).strip() if item[1] else "", float(item[2]))
        except (TypeError, ValueError):
            return (str(item[1]).strip() if item[1] else "", 0.0)
    if len(item) < 2:
        return (item[0] if len(item) > 0 else "", 0.0)
    a, b = item[0], item[1]
    try:
        s = float(b)
        return (str(a).strip(), s)
    except (TypeError, ValueError):
        pass
    try:
        s = float(a)
        return (str(b).strip(), s)
    except (TypeError, ValueError):
        pass
    return (str(a).strip(), 0.0)


if _RAPIDOCR_AVAILABLE and _RapidOCR is not None:

    @register_OCR("rapidocr")
    class RapidOCROCR(OCRBase):
        """
        RapidOCR recognition on detector crops. Uses perspective warp for quads.
        Good for Korean/Chinese/English when paired with rapidocr_det or any detector.
        """
        params = {
            "rec_model_path": {
                "type": "line_editor",
                "value": "",
                "description": "Path to recognition ONNX model. Empty = use built-in.",
            },
            "rec_keys_path": {
                "type": "line_editor",
                "value": "",
                "description": "Path to recognition dict (e.g. korean_dict.txt). Empty = use built-in.",
            },
            "crop_padding": {
                "type": "line_editor",
                "value": 4,
                "description": "Pixels to add around each box when cropping (0–24).",
            },
            "preprocess_recipe": {
                "type": "selector",
                "options": ["none", "clahe", "clahe+sharpen", "otsu", "adaptive", "denoise"],
                "value": "none",
                "description": "Optional preprocessing for hard OCR cases.",
            },
            "upscale_min_side": {
                "type": "line_editor",
                "value": 0,
                "description": "If >0, upscale crop so longer side >= this (e.g. 512). 0 = off.",
            },
            "min_confidence": {
                "value": 0.2,
                "description": "Minimum recognition confidence (0–1). Results below this are cleared. EasyScanlate default: 0.2.",
            },
            "description": "RapidOCR recognition on crops (ONNX). Install: pip install rapidocr-onnxruntime",
        }
        _load_model_keys = {"rec_engine"}

        def __init__(self, **params) -> None:
            super().__init__(**params)
            self.rec_engine = None

        def _load_model(self):
            if self.rec_engine is not None:
                return
            rec_path = (self.params.get("rec_model_path") or {}).get("value") or ""
            keys_path = (self.params.get("rec_keys_path") or {}).get("value") or ""
            params = {
                "Global.use_det": False,
                "Global.use_rec": True,
                "Global.use_cls": False,
            }
            if _EngineType is not None:
                params["Rec.engine_type"] = _EngineType.ONNXRUNTIME
            if rec_path and os.path.isfile(rec_path):
                params["Rec.model_path"] = rec_path
            if keys_path and os.path.isfile(keys_path):
                params["Rec.rec_keys_path"] = keys_path
            try:
                self.rec_engine = _RapidOCR(params=params)
            except Exception as e:
                self.logger.warning("RapidOCR rec init with custom paths failed, trying default: %s", e)
                self.rec_engine = _RapidOCR(params={"Global.use_det": False, "Global.use_rec": True, "Global.use_cls": False})

        def _ocr_blk_list(self, img: np.ndarray, blk_list: List[TextBlock], *args, **kwargs) -> None:
            if self.rec_engine is None:
                return
            im_h, im_w = img.shape[:2]
            if img.ndim == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            pad = max(0, min(24, int((self.params.get("crop_padding") or {}).get("value", 4))))
            recipe = (self.params.get("preprocess_recipe") or {}).get("value", "none") or "none"
            try:
                upscale_min_side = int((self.params.get("upscale_min_side") or {}).get("value", 0) or 0)
            except (TypeError, ValueError):
                upscale_min_side = 0
            if upscale_min_side <= 0:
                try:
                    from utils.config import pcfg
                    upscale_min_side = int(getattr(pcfg.module, "ocr_upscale_min_side", 0) or 0)
                except Exception:
                    pass

            for blk in blk_list:
                # Build a crop that covers all lines in the block, not just the first one.
                if blk.lines:
                    xs, ys = [], []
                    for line in blk.lines:
                        for p in line:
                            if not isinstance(p, (list, tuple)) or len(p) < 2:
                                continue
                            xs.append(p[0])
                            ys.append(p[1])
                    if xs and ys:
                        x1 = max(0, int(min(xs)) - pad)
                        y1 = max(0, int(min(ys)) - pad)
                        x2 = min(im_w, int(max(xs)) + pad)
                        y2 = min(im_h, int(max(ys)) + pad)
                    else:
                        x1, y1, x2, y2 = blk.xyxy
                        x1 = max(0, x1 - pad)
                        y1 = max(0, y1 - pad)
                        x2 = min(im_w, x2 + pad)
                        y2 = min(im_h, y2 + pad)
                else:
                    x1, y1, x2, y2 = blk.xyxy
                    x1 = max(0, x1 - pad)
                    y1 = max(0, y1 - pad)
                    x2 = min(im_w, x2 + pad)
                    y2 = min(im_h, y2 + pad)
                if not (0 <= x1 < x2 <= im_w and 0 <= y1 < y2 <= im_h):
                    blk.text = [""]
                    continue
                # If there is exactly one quad for this block, allow perspective warp.
                # For multi-line bubbles, use an axis-aligned crop so all lines are included.
                if blk.lines and len(blk.lines) == 1 and len(blk.lines[0]) >= 4:
                    quad_rel = [[float(p[0]) - x1, float(p[1]) - y1] for p in blk.lines[0]]
                    crop = _get_rotate_crop_image(img[y1:y2, x1:x2], quad_rel)
                else:
                    crop = img[y1:y2, x1:x2]
                if crop is None or crop.size == 0:
                    blk.text = [""]
                    continue
                crop = preprocess_for_ocr(crop, recipe=recipe, upscale_min_side=upscale_min_side)
                if crop.ndim == 2:
                    crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
                try:
                    rec_out = self.rec_engine(crop)
                except Exception as e:
                    self.logger.debug("RapidOCR rec failed on crop: %s", e)
                    blk.text = [""]
                    continue
                text, _score = _parse_rec_output(rec_out)
                min_conf = float((self.params.get("min_confidence") or {}).get("value", 0.2) or 0)
                if min_conf > 0 and _score < min_conf:
                    text = ""
                blk.text = [text] if text else [""]

        def ocr_img(self, img: np.ndarray) -> str:
            if self.rec_engine is None:
                return ""
            if img.ndim == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            try:
                rec_out = self.rec_engine(img)
            except Exception as e:
                self.logger.debug("RapidOCR rec failed: %s", e)
                return ""
            text, _ = _parse_rec_output(rec_out)
            return text
