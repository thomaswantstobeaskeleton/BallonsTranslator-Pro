"""
Surya text detection – line-level detection (90+ languages) via surya-ocr.
Uses vikp/surya_det (Segformer). Quality-focused; good for document/comic text.
Requires: pip install surya-ocr  (Python 3.10+, PyTorch).
"""
import numpy as np
import cv2
from typing import Tuple, List, Any

from .base import register_textdetectors, TextDetectorBase, TextBlock, ProjImgTrans
from .box_utils import expand_blocks
from ..base import DEVICE_SELECTOR
from utils.textblock import sort_regions, mit_merge_textlines

_SURYA_DET_AVAILABLE = False
_SURYA_LEGACY_API = False
_SURYA_USE_PREDICTOR_ONLY = False  # DetectionPredictor only (avoids surya.models / ocr_error)
try:
    from surya.models import load_predictors
    from PIL import Image
    _SURYA_DET_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    try:
        from surya.detection import DetectionPredictor
        from PIL import Image
        _SURYA_DET_AVAILABLE = True
        _SURYA_USE_PREDICTOR_ONLY = True
    except (ImportError, ModuleNotFoundError):
        try:
            from surya.detection import batch_text_detection
            from surya.model.detection.segformer import load_model, load_processor
            from PIL import Image
            _SURYA_DET_AVAILABLE = True
            _SURYA_LEGACY_API = True
        except ImportError:
            try:
                from surya.detection.batch_detection import batch_text_detection
                from surya.model.detection.model import load_model
                from surya.model.detection.processor import load_processor
                from PIL import Image
                _SURYA_DET_AVAILABLE = True
                _SURYA_LEGACY_API = True
            except ImportError:
                import logging
                logging.getLogger("BallonTranslator").debug(
                    "Surya detection not available. Install: pip install surya-ocr (Python 3.10+, PyTorch)."
                )


def _bbox_distance_px(blk_a: TextBlock, blk_b: TextBlock) -> float:
    """Minimum distance between two axis-aligned boxes; 0 if overlapping."""
    x1_a, y1_a, x2_a, y2_a = blk_a.xyxy
    x1_b, y1_b, x2_b, y2_b = blk_b.xyxy
    dx = max(0, max(x1_a, x1_b) - min(x2_a, x2_b))
    dy = max(0, max(y1_a, y1_b) - min(y2_a, y2_b))
    return (dx * dx + dy * dy) ** 0.5


def _merge_nearby_blocks(blk_list: List[TextBlock], gap_px: int) -> List[TextBlock]:
    """Merge blocks whose bounding boxes are within gap_px (or overlapping) into one per bubble."""
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


def _predictions_to_mask_blocks(
    h: int, w: int,
    predictions: List[Any],
    min_score: float,
) -> Tuple[np.ndarray, List[TextBlock]]:
    """Convert Surya detection predictions to mask and TextBlock list."""
    mask = np.zeros((h, w), dtype=np.uint8)
    blk_list: List[TextBlock] = []
    if not predictions:
        return mask, blk_list
    # One prediction per image; we passed one image
    pred = predictions[0]
    boxes = getattr(pred, "bboxes", None) or getattr(pred, "text_lines", None)
    if boxes is None and isinstance(pred, (list, tuple)):
        boxes = pred
    elif boxes is None and isinstance(pred, dict):
        boxes = pred.get("bboxes", pred.get("text_lines", []))
    if boxes is None:
        boxes = []
    if hasattr(boxes, "__iter__") and not isinstance(boxes, (list, tuple)):
        boxes = list(boxes)
    for item in boxes:
        pts = None
        score = 1.0
        if isinstance(item, (list, tuple)):
            pts = np.array(item, dtype=np.int32)
        elif hasattr(item, "bbox"):
            b = item.bbox
            if hasattr(b, "tolist"):
                b = b.tolist()
            b = list(b)
            if len(b) == 4:
                x1, y1, x2, y2 = b
                pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
            else:
                pts = np.array(b, dtype=np.int32)
        elif hasattr(item, "polygon"):
            p = item.polygon
            pts = np.array(p if hasattr(p, "__len__") else p.tolist(), dtype=np.int32)
        else:
            continue
        if pts.ndim == 1:
            if len(pts) == 4:
                x1, y1, x2, y2 = pts[0], pts[1], pts[2], pts[3]
                pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
            else:
                pts = pts.reshape(-1, 2)
        if pts.size < 6:
            continue
        if hasattr(item, "confidence"):
            score = float(item.confidence)
        elif hasattr(item, "score"):
            score = float(item.score)
        if score < min_score:
            continue
        x1 = int(pts[:, 0].min())
        y1 = int(pts[:, 1].min())
        x2 = int(pts[:, 0].max())
        y2 = int(pts[:, 1].max())
        if x2 <= x1 or y2 <= y1:
            continue
        blk = TextBlock(xyxy=[x1, y1, x2, y2], lines=[pts.tolist()])
        blk.language = "unknown"
        blk._detected_font_size = max(y2 - y1, 12)
        blk_list.append(blk)
        cv2.fillPoly(mask, [pts.astype(np.int32)], 255)
    return mask, blk_list


if _SURYA_DET_AVAILABLE:

    @register_textdetectors("surya_det")
    class SuryaDetector(TextDetectorBase):
        """
        Surya line-level text detection (90+ languages). Quality over speed.
        Uses surya-ocr detection (Segformer). Good for document/comic when CTD is not used.
        """
        params = {
            "device": DEVICE_SELECTOR(),
            "det_score_thresh": {
                "type": "line_editor",
                "value": 0.3,
                "description": "Min detection score (0.2–0.6).",
            },
            "merge_text_lines": {
                "type": "checkbox",
                "value": True,
                "description": "Merge nearby lines into one bubble (recommended for comics / dense text). Disable to get one box per line.",
            },
            "merge_gap_px": {
                "type": "line_editor",
                "value": 12,
                "description": "Merge blocks only when within this many pixels. 0 = no merging (one box per detected line; keeps close bubbles separate). 10–15 = merge only very close lines.",
            },
            "box_padding": {
                "type": "line_editor",
                "value": 0,
                "description": "Pixels to add around each detected box (all sides). Reduces clipped punctuation (?, !) and character edges. Recommended 4–6.",
            },
            "description": "Surya text detection (line-level). Install: pip install surya-ocr",
        }
        _load_model_keys = {"det_model", "det_processor"}

        def __init__(self, **params) -> None:
            super().__init__(**params)
            self.det_model = None
            self.det_processor = None

        def _load_model(self):
            if self.det_model is not None:
                return
            dev = (self.params.get("device") or {}).get("value", "cpu")
            if dev in ("cuda", "gpu"):
                dev = "cuda"
            if _SURYA_LEGACY_API:
                self.det_model = load_model()
                self.det_processor = load_processor()
                if dev == "cuda" and hasattr(self.det_model, "to"):
                    self.det_model = self.det_model.to(dev)
            elif _SURYA_USE_PREDICTOR_ONLY:
                self.det_model = DetectionPredictor(device=dev)
                self.det_processor = None
            else:
                preds = load_predictors(device=dev)
                self.det_model = preds["detection"]
                self.det_processor = None

        def _detect(self, img: np.ndarray, proj: ProjImgTrans = None) -> Tuple[np.ndarray, List[TextBlock]]:
            if img.ndim == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            h, w = img.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            blk_list: List[TextBlock] = []
            try:
                pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                if _SURYA_LEGACY_API:
                    predictions = batch_text_detection([pil_img], self.det_model, self.det_processor)
                else:
                    predictions = self.det_model([pil_img])
                    if not isinstance(predictions, list):
                        predictions = [predictions]
            except Exception as e:
                self.logger.error(f"Surya detection failed: {e}")
                return mask, blk_list
            min_score = 0.3
            ps = self.params.get("det_score_thresh", {})
            if isinstance(ps, dict):
                try:
                    min_score = max(0.0, min(1.0, float(ps.get("value", 0.3))))
                except (TypeError, ValueError):
                    pass
            mask, blk_list = _predictions_to_mask_blocks(h, w, predictions, min_score)
            if not blk_list:
                return mask, blk_list
            merge_gap = 12
            mg = self.params.get("merge_gap_px", {})
            if isinstance(mg, dict):
                try:
                    merge_gap = max(0, int(float(mg.get("value", 12))))
                except (TypeError, ValueError):
                    pass
            merge_lines = self.params.get("merge_text_lines", {}).get("value", True)
            # When merge_gap_px is 0, skip line grouping so each detection line stays one box
            # (avoids two close bubbles being merged by mit_merge_textlines).
            if merge_lines and merge_gap > 0 and len(blk_list) > 0:
                pts_list = [line_pts for blk in blk_list for line_pts in blk.lines]
                if pts_list:
                    blk_list = mit_merge_textlines(pts_list, width=w, height=h)
                    mask = np.zeros((h, w), dtype=np.uint8)
                    for blk in blk_list:
                        for line_pts in blk.lines:
                            pts = np.array(line_pts, dtype=np.int32)
                            if pts.ndim == 1:
                                pts = pts.reshape(-1, 2)
                            cv2.fillPoly(mask, [pts], 255)
            blk_list = _merge_nearby_blocks(blk_list, merge_gap)
            blk_list = sort_regions(blk_list)
            pad_val = 0
            bp = self.params.get("box_padding", {})
            if isinstance(bp, dict):
                try:
                    v = bp.get("value", 0)
                    pad_val = max(0, min(24, int(v) if v not in (None, '') else 0))
                except (TypeError, ValueError):
                    pass
            if pad_val > 0:
                blk_list = expand_blocks(blk_list, pad_val, w, h)
                mask = np.zeros((h, w), dtype=np.uint8)
                for blk in blk_list:
                    if blk.lines:
                        pts = np.array(blk.lines[0], dtype=np.int32)
                    else:
                        x1, y1, x2, y2 = blk.xyxy
                        pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
                    cv2.fillPoly(mask, [pts], 255)
            return mask, blk_list

        def updateParam(self, param_key: str, param_content):
            super().updateParam(param_key, param_content)
            if param_key == "device":
                self.det_model = None
