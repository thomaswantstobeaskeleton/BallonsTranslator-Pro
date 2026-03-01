"""
EasyOCR text detection only – CRAFT or DBNet18 detector.
Good for Chinese, English, and many languages. Uses EasyOCR's detect() so only the detection model is used at runtime.
Requires: pip install easyocr (PyTorch). Set recognizer=False to load detector only (optional).
"""
import numpy as np
import cv2
from typing import Tuple, List

from .base import register_textdetectors, TextDetectorBase, TextBlock, ProjImgTrans
from .box_utils import expand_blocks
from utils.textblock import sort_regions, mit_merge_textlines


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

_EASYOCR_AVAILABLE = False
try:
    import easyocr
    _EASYOCR_AVAILABLE = True
except ImportError:
    import logging
    logging.getLogger("BallonTranslator").debug(
        "EasyOCR not available for detector. Install: pip install easyocr"
    )


if _EASYOCR_AVAILABLE:

    @register_textdetectors("easyocr_det")
    class EasyOCRDetector(TextDetectorBase):
        """
        Text detection using EasyOCR's detector (CRAFT or DBNet18).
        Good for Chinese, English, and multilingual. Use when CTD or paddle_det miss text.
        """
        params = {
            "language": {
                "type": "selector",
                "options": ["ch_sim", "ch_sim+en", "en", "ja", "ko"],
                "value": "ch_sim+en",
                "description": "Language(s): ch_sim (Chinese simplified), ch_sim+en, en, ja, ko.",
            },
            "gpu": {
                "type": "checkbox",
                "value": True,
                "description": "Use GPU if available.",
            },
            "detect_network": {
                "type": "selector",
                "options": ["craft", "dbnet18"],
                "value": "craft",
                "description": "Backbone: craft or dbnet18. If Chinese in bubbles is missed, try switching to dbnet18.",
            },
            "text_threshold": {
                "value": 0.35,
                "description": "Detection threshold. Lower = more regions. If Chinese missed: try 0.25–0.3.",
            },
            "link_threshold": {
                "value": 0.3,
                "description": "Link threshold. Lower = more grouping. If text missed: try 0.25.",
            },
            "min_size": {
                "value": 6,
                "description": "Min box size (px). Lower = catch smaller text. If Chinese missed: try 4–6.",
            },
            "merge_text_lines": {
                "type": "checkbox",
                "value": True,
                "description": "Merge nearby lines into one bubble (recommended for comics).",
            },
            "merge_gap_px": {
                "value": 50,
                "description": "Merge blocks within this many pixels (reduces overlapping text in one bubble).",
            },
            "box_padding": {
                "type": "line_editor",
                "value": 5,
                "description": "Pixels to add around each detected box (all sides). Reduces clipped punctuation (?, !) and character edges. Recommended 4–6.",
            },
            "description": "EasyOCR detection only (CRAFT/DBNet18).",
        }
        _load_model_keys = {"reader"}

        def __init__(self, **params) -> None:
            super().__init__(**params)
            self.reader = None

        def _load_model(self):
            if self.reader is not None:
                return
            lang_val = self.params["language"]["value"]
            if lang_val == "ch_sim+en":
                lang_list = ["ch_sim", "en"]
            else:
                lang_list = [lang_val]
            gpu = self.params["gpu"]["value"]
            detect_network = self.params["detect_network"]["value"]
            self.reader = easyocr.Reader(
                lang_list,
                gpu=gpu,
                detector=True,
                recognizer=False,
                verbose=False,
                detect_network=detect_network,
            )

        def _detect(self, img: np.ndarray, proj: ProjImgTrans = None) -> Tuple[np.ndarray, List[TextBlock]]:
            if img.ndim == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            h, w = img.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            blk_list: List[TextBlock] = []

            text_threshold = float(self.params["text_threshold"]["value"])
            link_threshold = float(self.params["link_threshold"]["value"])
            min_size = int(self.params["min_size"]["value"])
            detect_kw = dict(
                min_size=min_size,
                text_threshold=text_threshold,
                link_threshold=link_threshold,
                low_text=0.4,
                canvas_size=2560,
                mag_ratio=1.0,
            )

            try:
                horizontal_list_agg, free_list_agg = self.reader.detect(img, **detect_kw)
            except Exception as e:
                err_msg = str(e)
                # Windows + CUDA: deform_conv_cuda often not available; fall back to CPU
                if ("deform_conv" in err_msg or "not imported successfully" in err_msg) and "cuda" in err_msg.lower():
                    self.logger.warning(
                        "EasyOCR CUDA failed (deform_conv not available on this system). Falling back to CPU for detection."
                    )
                    self.reader = None
                    self.params["gpu"]["value"] = False
                    self._load_model()
                    try:
                        horizontal_list_agg, free_list_agg = self.reader.detect(img, **detect_kw)
                    except Exception as e2:
                        self.logger.error(f"EasyOCR det failed: {e2}")
                        return mask, blk_list
                else:
                    self.logger.error(f"EasyOCR det failed: {e}")
                    return mask, blk_list

            horizontal_list = horizontal_list_agg[0] if horizontal_list_agg else []
            free_list = free_list_agg[0] if free_list_agg else []

            pts_list = []
            for bbox in horizontal_list:
                if len(bbox) < 4:
                    continue
                x1, x2 = int(bbox[0]), int(bbox[1])
                y1, y2 = int(bbox[2]), int(bbox[3])
                if x2 <= x1 or y2 <= y1:
                    continue
                pts = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                pts_list.append(pts)

            for poly in free_list:
                if len(poly) < 4:
                    continue
                pts = np.array(poly, dtype=np.int32)
                if pts.ndim == 1:
                    pts = pts.reshape(-1, 2)
                if pts.shape[0] < 4:
                    continue
                x1, y1 = int(pts[:, 0].min()), int(pts[:, 1].min())
                x2, y2 = int(pts[:, 0].max()), int(pts[:, 1].max())
                if x2 <= x1 or y2 <= y1:
                    continue
                # Quadrilateral expects 4 points; use bbox corners if poly has more
                if pts.shape[0] == 4:
                    pts_list.append(pts.tolist())
                else:
                    pts_list.append([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])

            if self.params.get("merge_text_lines", {}).get("value", True) and len(pts_list) > 0:
                blk_list = mit_merge_textlines(pts_list, width=w, height=h)
                for blk in blk_list:
                    for line_pts in blk.lines:
                        pts = np.array(line_pts, dtype=np.int32)
                        cv2.fillPoly(mask, [pts], 255)
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
                blk_list = expand_blocks(blk_list, pad_val, w, h)

            return mask, blk_list
