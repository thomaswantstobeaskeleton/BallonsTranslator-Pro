"""
RapidOCR text detection only. Produces text regions for use with RapidOCR OCR or any recognizer.
Inspired by EasyScanlate (https://github.com/Liiesl/EasyScanlate); uses the same Det→Rec pipeline split.
Requires: pip install rapidocr-onnxruntime (or rapidocr with onnxruntime).
Optional: place PP-OCRv5 det/rec models in data/models/rapidocr/ for custom paths.
"""
import os
from typing import Tuple, List

import cv2
import numpy as np

from .base import register_textdetectors, TextDetectorBase, TextBlock, ProjImgTrans
from .box_utils import expand_blocks
from utils.textblock import sort_regions, mit_merge_textlines

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
            "RapidOCR not available for detector. Install: pip install rapidocr-onnxruntime"
        )


def _apply_contrast(img: np.ndarray, factor: float) -> np.ndarray:
    """Apply contrast enhancement. factor is added to 1.0 (e.g. 0.5 -> 1.5). 0 = no change."""
    if factor is None or abs(factor) < 1e-6:
        return img
    try:
        from PIL import Image, ImageEnhance
        if img.ndim == 2:
            pil = Image.fromarray(img)
        else:
            pil = Image.fromarray(img)
        if pil.mode != "RGB":
            pil = pil.convert("RGB")
        enhancer = ImageEnhance.Contrast(pil)
        out = enhancer.enhance(max(0.1, 1.0 + float(factor)))
        return np.array(out)
    except Exception:
        alpha = max(0.1, 1.0 + float(factor))
        return np.clip((img.astype(np.float32) - 127.5) * alpha + 127.5, 0, 255).astype(np.uint8)


def _bbox_distance_px(blk_a: TextBlock, blk_b: TextBlock) -> float:
    """Minimum distance between two axis-aligned boxes; 0 if overlapping."""
    x1_a, y1_a, x2_a, y2_a = blk_a.xyxy
    x1_b, y1_b, x2_b, y2_b = blk_b.xyxy
    dx = max(0, max(x1_a, x1_b) - min(x2_a, x2_b))
    dy = max(0, max(y1_a, y1_b) - min(y2_a, y2_b))
    return (dx * dx + dy * dy) ** 0.5


def _merge_nearby_blocks(blk_list: List[TextBlock], gap_px: int) -> List[TextBlock]:
    """Merge blocks whose bounding box centers are within gap_px (EasyScanlate-style distance merge)."""
    if gap_px <= 0 or len(blk_list) <= 1:
        return blk_list
    merged: List[TextBlock] = []
    for blk in blk_list:
        combined = False
        for m in merged:
            if _bbox_distance_px(m, blk) <= gap_px:
                m.lines.extend(blk.lines)
                m.adjust_bbox()
                if getattr(blk, "_detected_font_size", -1) > 0:
                    m._detected_font_size = max(getattr(m, "_detected_font_size", 0), blk._detected_font_size)
                combined = True
                break
        if not combined:
            merged.append(blk)
    return merged


def _extract_boxes_from_det_output(det_output) -> List:
    """Extract box list from RapidOCR detection output (handles different API shapes)."""
    boxes = []
    if hasattr(det_output, "boxes") and det_output.boxes is not None:
        boxes = det_output.boxes
    elif isinstance(det_output, (list, tuple)):
        if len(det_output) > 0 and det_output[0] is not None:
            if isinstance(det_output[0], (list, np.ndarray)):
                boxes = det_output[0]
            elif isinstance(det_output[0], tuple):
                boxes = [x[0] for x in det_output]
    elif hasattr(det_output, "dt_boxes"):
        boxes = det_output.dt_boxes
    if boxes is None or (isinstance(boxes, np.ndarray) and boxes.size == 0):
        boxes = []
    if not boxes:
        return []
    return list(boxes) if not isinstance(boxes, list) else boxes


def _box_to_pts(box) -> np.ndarray:
    """
    Convert a single box from RapidOCR (various formats) to a 4x2 float32 array of points.
    Returns array of shape (4, 2) or None if invalid.
    Handles: (points, text, score) tuple, .box attribute, list of 4 [x,y], nested lists.
    """
    if box is None:
        return None
    raw = None
    if hasattr(box, "box"):
        raw = box.box
    elif isinstance(box, (list, tuple)) and len(box) >= 1:
        # (points, text, score) or just points
        first = box[0]
        if isinstance(first, (list, tuple)) and len(first) >= 2:
            raw = first if (len(first) == 4 or (len(first) > 2 and np.isscalar(first[0]))) else box
        else:
            raw = box
    else:
        raw = box
    if raw is None:
        return None
    try:
        pts = np.array(raw, dtype=np.float32)
    except (ValueError, TypeError):
        try:
            # Inhomogeneous: e.g. list of [x,y], [x,y], ... with different nesting
            out = []
            for p in raw:
                if hasattr(p, "__len__") and len(p) >= 2:
                    out.append([float(p[0]), float(p[1])])
                else:
                    return None
            if len(out) < 4:
                return None
            pts = np.array(out[:4], dtype=np.float32)
        except (ValueError, TypeError):
            return None
    if pts.ndim == 1 and pts.size >= 8:
        pts = pts.reshape(-1, 2)
    if pts.ndim != 2 or pts.shape[0] < 4 or pts.shape[1] < 2:
        return None
    if pts.shape[0] > 4:
        pts = pts[:4]
    return pts


if _RAPIDOCR_AVAILABLE and _RapidOCR is not None:

    @register_textdetectors("rapidocr_det")
    class RapidOCRDetector(TextDetectorBase):
        """
        Text detection using RapidOCR (ONNX). Lightweight, no GPU required.
        Pairs well with RapidOCR OCR for an EasyScanlate-like pipeline.
        """
        params = {
            "det_model_path": {
                "type": "line_editor",
                "value": "",
                "description": "Path to detection ONNX model. Empty = use built-in model.",
            },
            "merge_text_lines": {
                "type": "checkbox",
                "value": True,
                "description": "Merge nearby lines into one bubble (recommended for comics).",
            },
            "merge_gap_px": {
                "value": 50,
                "description": "Merge blocks within this many pixels (center-to-center).",
            },
            "box_padding": {
                "type": "line_editor",
                "value": 5,
                "description": "Pixels to add around each detected box (0–24).",
            },
            "min_text_height": {
                "value": 8,
                "description": "Minimum text region height (px). Smaller regions are dropped.",
            },
            "max_text_height": {
                "value": 0,
                "description": "Maximum text region height (px). Larger regions dropped. 0 = no limit.",
            },
            "adjust_contrast": {
                "value": 0.0,
                "description": "Contrast adjustment before detection (e.g. 0.5 = +50%). 0 = off. Helps faint text.",
            },
            "resize_threshold": {
                "value": 0,
                "description": "If image width > this (px), resize to this before detection. 0 = no resize. Speeds up large pages.",
            },
            "description": "RapidOCR detection only (ONNX). Install: pip install rapidocr-onnxruntime",
        }
        _load_model_keys = {"det_engine"}

        def __init__(self, **params) -> None:
            super().__init__(**params)
            self.det_engine = None

        def _load_model(self):
            if self.det_engine is not None:
                return
            model_path = (self.params.get("det_model_path") or {}).get("value") or ""
            params = {
                "Global.use_det": True,
                "Global.use_rec": False,
                "Global.use_cls": True,
            }
            if _EngineType is not None:
                params["Det.engine_type"] = _EngineType.ONNXRUNTIME
            if model_path and os.path.isfile(model_path):
                params["Det.model_path"] = model_path
            try:
                self.det_engine = _RapidOCR(params=params)
            except Exception as e:
                self.logger.warning("RapidOCR det init with custom path failed, trying default: %s", e)
                self.det_engine = _RapidOCR(params={"Global.use_det": True, "Global.use_rec": False, "Global.use_cls": True})

        def _detect(self, img: np.ndarray, proj: ProjImgTrans = None) -> Tuple[np.ndarray, List[TextBlock]]:
            if img.ndim == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            orig_h, orig_w = img.shape[:2]

            # EasyScanlate-style preprocessing: contrast then optional resize
            adjust = float((self.params.get("adjust_contrast") or {}).get("value", 0) or 0)
            if abs(adjust) > 1e-6:
                img = _apply_contrast(img, adjust)
            resize_thresh = int((self.params.get("resize_threshold") or {}).get("value", 0) or 0)
            scale_x, scale_y = 1.0, 1.0
            if resize_thresh > 0 and orig_w > resize_thresh:
                ratio = resize_thresh / orig_w
                new_w = resize_thresh
                new_h = max(2, int(orig_h * ratio))
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                scale_x = orig_w / new_w
                scale_y = orig_h / new_h

            h, w = img.shape[:2]
            mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
            blk_list: List[TextBlock] = []

            try:
                det_out = self.det_engine(img)
            except Exception as e:
                self.logger.error("RapidOCR det failed: %s", e)
                return mask, blk_list

            boxes = _extract_boxes_from_det_output(det_out)
            min_h = max(0, int((self.params.get("min_text_height") or {}).get("value", 8)))
            max_h_val = int((self.params.get("max_text_height") or {}).get("value", 0) or 0)

            pts_list = []
            for box in boxes:
                pts = _box_to_pts(box)
                if pts is None or pts.size < 8:
                    continue
                if pts.shape[0] != 4:
                    pts = pts.reshape(-1, 2)
                if pts.shape[0] < 4:
                    continue
                # Scale coordinates back to original image if we resized
                if scale_x != 1.0 or scale_y != 1.0:
                    pts[:, 0] *= scale_x
                    pts[:, 1] *= scale_y
                y_coords = pts[:, 1]
                height = float(np.max(y_coords) - np.min(y_coords))
                if height < min_h:
                    continue
                if max_h_val > 0 and height > max_h_val:
                    continue
                pts_int = np.clip(pts.astype(np.int32), 0, [orig_w - 1, orig_h - 1])
                x1, y1 = int(pts_int[:, 0].min()), int(pts_int[:, 1].min())
                x2, y2 = int(pts_int[:, 0].max()), int(pts_int[:, 1].max())
                if x2 <= x1 or y2 <= y1:
                    continue
                if pts_int.shape[0] == 4:
                    pts_list.append(pts_int.tolist())
                else:
                    pts_list.append([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])

            if self.params.get("merge_text_lines", {}).get("value", True) and len(pts_list) > 0:
                blk_list = mit_merge_textlines(pts_list, width=orig_w, height=orig_h)
                for blk in blk_list:
                    for line_pts in blk.lines:
                        pts_np = np.array(line_pts, dtype=np.int32)
                        cv2.fillPoly(mask, [pts_np], 255)
            else:
                for pts in pts_list:
                    pts_np = np.array(pts, dtype=np.int32)
                    if pts_np.ndim == 1:
                        pts_np = pts_np.reshape(-1, 2)
                    x1 = int(pts_np[:, 0].min())
                    y1 = int(pts_np[:, 1].min())
                    x2 = int(pts_np[:, 0].max())
                    y2 = int(pts_np[:, 1].max())
                    blk = TextBlock(xyxy=[x1, y1, x2, y2], lines=[pts_np.tolist()])
                    blk.language = "unknown"
                    blk._detected_font_size = max(y2 - y1, 12)
                    blk_list.append(blk)
                    cv2.fillPoly(mask, [pts_np], 255)

            merge_gap = int(self.params.get("merge_gap_px", {}).get("value", 50))
            blk_list = _merge_nearby_blocks(blk_list, merge_gap)
            blk_list = sort_regions(blk_list)

            pad_val = 0
            bp = self.params.get("box_padding", {})
            if isinstance(bp, dict):
                v = bp.get("value", 5)
                try:
                    pad_val = max(0, min(24, int(v) if v not in (None, "") else 5))
                except (TypeError, ValueError):
                    pad_val = 5
            if pad_val > 0:
                blk_list = expand_blocks(blk_list, pad_val, orig_w, orig_h)

            return mask, blk_list
