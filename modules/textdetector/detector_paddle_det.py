"""
PaddleOCR text detection only – no recognition.
Uses PP-OCR det model (DB/DB++) for text region detection. Strong for Chinese, English, and document/comic text.
Use when CTD misses text or for non-comic layouts. Requires: paddleocr, paddlepaddle.
When PyTorch is already loaded (e.g. Ocean OCR), detection runs in a subprocess to avoid "_gpuDeviceProperties is already registered".
"""
import os
import sys
import json
import tempfile
import subprocess
import numpy as np
import cv2
from typing import Tuple, List

from .base import register_textdetectors, TextDetectorBase, TextBlock, ProjImgTrans
from utils.textblock import sort_regions, mit_merge_textlines

os.environ.setdefault("PPOCR_HOME", os.path.join("data", "models", "paddle-ocr"))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# Disable oneDNN to avoid ConvertPirAttribute2RuntimeAttribute error on Windows (must be set before Paddle import)
os.environ["FLAGS_use_mkldnn"] = "0"

_PADDLE_AVAILABLE = False
try:
    from paddleocr import PaddleOCR
    _PADDLE_AVAILABLE = True
except ImportError:
    import logging
    logging.getLogger("BallonTranslator").debug(
        "PaddleOCR not available for detector. Install: pip install paddleocr paddlepaddle"
    )


def _shrink_block(blk: TextBlock, margin_px: int, img_w: int, img_h: int) -> None:
    """Shrink block bbox and polygon lines inward by margin_px so boxes sit inside bubbles. In-place."""
    if margin_px <= 0:
        return
    x1, y1, x2, y2 = blk.xyxy
    w, h = x2 - x1, y2 - y1
    if w <= 2 * margin_px or h <= 2 * margin_px:
        return
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    x1n = min(x1 + margin_px, int(cx))
    y1n = min(y1 + margin_px, int(cy))
    x2n = max(x2 - margin_px, int(cx) + 1)
    y2n = max(y2 - margin_px, int(cy) + 1)
    x1n = max(0, x1n)
    y1n = max(0, y1n)
    x2n = min(img_w, x2n)
    y2n = min(img_h, y2n)
    if x2n <= x1n or y2n <= y1n:
        return
    blk.xyxy = [x1n, y1n, x2n, y2n]
    new_lines = []
    for line in blk.lines:
        pts = np.array(line, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts.reshape(-1, 2)
        out = []
        for i in range(len(pts)):
            px, py = pts[i, 0], pts[i, 1]
            dx = cx - px
            dy = cy - py
            norm = (dx * dx + dy * dy) ** 0.5
            if norm > 1e-6:
                move = min(margin_px, norm)
                px = px + move * (dx / norm)
                py = py + move * (dy / norm)
            px = np.clip(px, x1n, x2n)
            py = np.clip(py, y1n, y2n)
            out.append([int(round(px)), int(round(py))])
        if len(out) >= 3:
            new_lines.append(out)
    if new_lines:
        blk.lines = new_lines


def _bbox_distance_px(blk_a: TextBlock, blk_b: TextBlock) -> float:
    """Minimum distance between two axis-aligned boxes; 0 if overlapping."""
    x1_a, y1_a, x2_a, y2_a = blk_a.xyxy
    x1_b, y1_b, x2_b, y2_b = blk_b.xyxy
    dx = max(0, max(x1_a, x1_b) - min(x2_a, x2_b))
    dy = max(0, max(y1_a, y1_b) - min(y2_a, y2_b))
    return (dx * dx + dy * dy) ** 0.5


def _bbox_vertical_overlap_ratio(blk_a: TextBlock, blk_b: TextBlock) -> float:
    """Fraction of the shorter box's height that overlaps in y. 0 = no overlap, 1 = full overlap."""
    _, y1_a, _, y2_a = blk_a.xyxy
    _, y1_b, _, y2_b = blk_b.xyxy
    overlap = max(0, min(y2_a, y2_b) - max(y1_a, y1_b))
    ha = max(1, y2_a - y1_a)
    hb = max(1, y2_b - y1_b)
    return overlap / min(ha, hb)


def _merge_nearby_blocks(
    blk_list: List[TextBlock],
    gap_px: int,
    same_line_only: bool = True,
    overlap_ratio: float = 0.35,
) -> List[TextBlock]:
    """Merge blocks within gap_px. If same_line_only, only merge when vertical overlap >= overlap_ratio."""
    if gap_px <= 0 or len(blk_list) <= 1:
        return blk_list
    merged: List[TextBlock] = []
    for blk in blk_list:
        combined = False
        for m in merged:
            if _bbox_distance_px(m, blk) > gap_px:
                continue
            if same_line_only and _bbox_vertical_overlap_ratio(m, blk) < overlap_ratio:
                continue
            m.lines.extend(blk.lines)
            m.adjust_bbox()
            if getattr(blk, "_detected_font_size", -1) > 0:
                m._detected_font_size = max(getattr(m, "_detected_font_size", 0), blk._detected_font_size)
            combined = True
            break
        if not combined:
            merged.append(blk)
    return merged


if _PADDLE_AVAILABLE:

    @register_textdetectors("paddle_det")
    class PaddleDetector(TextDetectorBase):
        """
        Text detection using PaddleOCR's detection model only (DB/DB++).
        Good for Chinese, English, and general document/comic text. Use as alternative to CTD when CTD misses regions.
        """
        params = {
            "language": {
                "type": "selector",
                "options": ["ch", "en"],
                "value": "ch",
                "description": "Language for detection (ch = Chinese, en = English).",
            },
            "device": {
                "type": "selector",
                "options": ["cpu"],
                "value": "cpu",
                "description": "Device (CPU only; use with Ocean/PyTorch OCR).",
            },
            "det_limit_side_len": {
                "value": 960,
                "description": "Max side length for detection input. With Ocean OCR (subprocess/CPU), capped at 960 to avoid timeout. Higher = better box tightness when run in-process.",
            },
            "strict_bubble_mode": {
                "type": "checkbox",
                "value": True,
                "description": "Stricter settings for comics: higher box threshold, larger min area, inward shrink, aspect filter. Turn off if missing text.",
            },
            "det_db_thresh": {
                "value": 0.4,
                "description": "Detection binarization threshold.",
            },
            "det_db_box_thresh": {
                "value": 0.72,
                "description": "Box score threshold. Higher = fewer boxes outside bubbles (try 0.75–0.8 in strict mode).",
            },
            "min_detection_area": {
                "type": "line_editor",
                "value": 200,
                "description": "Drop detections smaller than this area (px²). Reduces noise outside bubbles.",
            },
            "max_aspect_ratio": {
                "type": "line_editor",
                "value": 10,
                "description": "Drop boxes with width/height or height/width above this (removes thin strips). 0 = disable.",
            },
            "box_shrink_px": {
                "type": "line_editor",
                "value": 4,
                "description": "Shrink each box inward by this many pixels so it sits inside the bubble. 0 = off.",
            },
            "merge_text_lines": {
                "type": "checkbox",
                "value": True,
                "description": "Merge nearby lines into one bubble (recommended for comics / dense text).",
            },
            "merge_gap_px": {
                "type": "line_editor",
                "value": 50,
                "description": "Merge blocks within this many pixels.",
            },
            "merge_same_line_only": {
                "type": "checkbox",
                "value": True,
                "description": "Only merge blocks on the same line (y-overlap). Prevents merging boxes from different bubbles.",
            },
            "merge_line_overlap_ratio": {
                "type": "line_editor",
                "value": 0.35,
                "description": "Min vertical overlap (0–1) to merge two blocks. Higher = stricter same-line.",
            },
            "description": "PaddleOCR text detection only (PP-OCR det).",
        }
        _load_model_keys = {"model"}

        def __init__(self, **params) -> None:
            super().__init__(**params)
            self.model = None
            self._use_subprocess = False  # True when torch already loaded to avoid paddle/torch CUDA type conflict

        def _load_model(self):
            if self.model is not None:
                return
            # If PyTorch is already loaded, Paddle's import will raise "_gpuDeviceProperties is already registered".
            # Run detection in a subprocess that only loads Paddle.
            if "torch" in sys.modules:
                self._use_subprocess = True
                self.model = True  # mark as "loaded" so all_model_loaded() passes
                return
            device = "cpu"
            lang = self.params["language"]["value"]
            try:
                self.model = PaddleOCR(
                    use_angle_cls=False,
                    lang=lang,
                    device=device,
                    enable_mkldnn=False,  # Avoid oneDNN ConvertPirAttribute2RuntimeAttribute error on Windows
                    det_limit_side_len=int(self.params["det_limit_side_len"]["value"]),
                    det_db_thresh=float(self.params["det_db_thresh"]["value"]),
                    det_db_box_thresh=float(self.params["det_db_box_thresh"]["value"]),
                )
            except Exception as e:
                if "already registered" in str(e) or "_gpuDeviceProperties" in str(e):
                    self._use_subprocess = True
                    self.model = True
                    return
                raise

        def _detect(self, img: np.ndarray, proj: ProjImgTrans = None) -> Tuple[np.ndarray, List[TextBlock]]:
            if img.ndim == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            h, w = img.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            blk_list: List[TextBlock] = []

            if self._use_subprocess:
                # Run Paddle in subprocess so it never shares the process with PyTorch
                try:
                    with tempfile.TemporaryDirectory(suffix="paddle_det") as tmpdir:
                        img_path = os.path.join(tmpdir, "img.png")
                        cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                        box_thresh = float(self.params.get("det_db_box_thresh", {}).get("value", 0.72))
                        det_side = int(self.params["det_limit_side_len"]["value"])
                        if self.params.get("strict_bubble_mode", {}).get("value", True):
                            box_thresh = max(box_thresh, 0.75)
                            det_side = max(det_side, 1280)
                        # Subprocess runs on CPU only; cap side len to avoid timeout (1280+ can take 5+ min/page)
                        det_side = min(det_side, 960)
                        params = {
                            "device": "cpu",
                            "lang": self.params["language"]["value"],
                            "det_limit_side_len": det_side,
                            "det_db_thresh": float(self.params["det_db_thresh"]["value"]),
                            "det_db_box_thresh": box_thresh,
                        }
                        params_path = os.path.join(tmpdir, "params.json")
                        with open(params_path, "w", encoding="utf-8") as f:
                            json.dump(params, f)
                        cmd = [
                            sys.executable, "-m", "modules.textdetector.paddle_det_runner",
                            img_path, params_path,
                        ]
                        result = subprocess.run(
                            cmd,
                            cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
                            capture_output=True,
                            text=True,
                            timeout=300,
                            env={**os.environ, "FLAGS_use_mkldnn": "0", "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "1"},
                        )
                        if result.returncode != 0:
                            self.logger.error(f"Paddle det subprocess failed: {result.stderr or result.stdout}")
                            return mask, blk_list
                        out = json.loads(result.stdout)
                        if "error" in out:
                            self.logger.error(out["error"])
                            return mask, blk_list
                        strict = self.params.get("strict_bubble_mode", {}).get("value", True)
                        min_area = 200
                        try:
                            min_area = max(0, int(float(self.params.get("min_detection_area", {}).get("value", 200))))
                        except (TypeError, ValueError):
                            pass
                        if strict:
                            min_area = max(min_area, 250)
                        max_ar = 10
                        try:
                            max_ar = float(self.params.get("max_aspect_ratio", {}).get("value", 10) or 0)
                        except (TypeError, ValueError):
                            pass
                        for b in out.get("blocks", []):
                            xyxy = b["xyxy"]
                            x1, y1, x2, y2 = xyxy
                            area = (x2 - x1) * (y2 - y1)
                            if area < min_area:
                                continue
                            if max_ar > 0:
                                bw, bh = x2 - x1, y2 - y1
                                if min(bw, bh) <= 0 or max(bw / bh, bh / bw) > max_ar:
                                    continue
                            lines = b.get("lines", [])
                            blk = TextBlock(xyxy=xyxy, lines=lines)
                            blk.language = "unknown"
                            blk._detected_font_size = b.get("font_size", 12)
                            blk_list.append(blk)
                            for line_pts in lines:
                                pts = np.array(line_pts, dtype=np.int32)
                                cv2.fillPoly(mask, [pts], 255)
                except (subprocess.TimeoutExpired, json.JSONDecodeError, KeyError) as e:
                    self.logger.error(f"Paddle det subprocess error: {e}")
                    return mask, blk_list
                if not blk_list:
                    return mask, blk_list
                merge_lines = self.params.get("merge_text_lines", {}).get("value", True)
                if merge_lines:
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
                merge_gap = 50
                mg = self.params.get("merge_gap_px", {})
                if isinstance(mg, dict):
                    try:
                        merge_gap = max(0, int(float(mg.get("value", 50))))
                    except (TypeError, ValueError):
                        pass
                same_line_only = self.params.get("merge_same_line_only", {}).get("value", True)
                overlap_ratio = 0.35
                try:
                    overlap_ratio = float(self.params.get("merge_line_overlap_ratio", {}).get("value", 0.35) or 0.35)
                    overlap_ratio = max(0, min(1, overlap_ratio))
                except (TypeError, ValueError):
                    pass
                if strict:
                    overlap_ratio = max(overlap_ratio, 0.5)
                blk_list = _merge_nearby_blocks(blk_list, merge_gap, same_line_only=same_line_only, overlap_ratio=overlap_ratio)
                blk_list = sort_regions(blk_list)
                shrink_px = 4
                try:
                    shrink_px = max(0, int(float(self.params.get("box_shrink_px", {}).get("value", 4) or 0)))
                except (TypeError, ValueError):
                    pass
                if strict:
                    shrink_px = max(shrink_px, 5)
                if shrink_px > 0:
                    for blk in blk_list:
                        _shrink_block(blk, shrink_px, w, h)
                    mask = np.zeros((h, w), dtype=np.uint8)
                    for blk in blk_list:
                        for line_pts in blk.lines:
                            pts = np.array(line_pts, dtype=np.int32)
                            if pts.ndim == 1:
                                pts = pts.reshape(-1, 2)
                            cv2.fillPoly(mask, [pts], 255)
                return mask, blk_list

            try:
                box_thresh = float(self.params.get("det_db_box_thresh", {}).get("value", 0.72))
                det_side = int(self.params["det_limit_side_len"]["value"])
                if self.params.get("strict_bubble_mode", {}).get("value", True):
                    box_thresh = max(box_thresh, 0.75)
                    det_side = max(det_side, 1280)
                # New PaddleOCR pipeline API: predict() only (no det/rec/cls args)
                result = self.model.predict(
                    img,
                    use_textline_orientation=False,
                    text_det_limit_side_len=det_side,
                    text_det_thresh=float(self.params["det_db_thresh"]["value"]),
                    text_det_box_thresh=box_thresh,
                )
            except Exception as e:
                self.logger.error(f"Paddle det failed: {e}")
                return mask, blk_list

            if not result or len(result) == 0:
                return mask, blk_list

            strict = self.params.get("strict_bubble_mode", {}).get("value", True)
            page = result[0]
            # New API: result items have "rec_polys" (list of polygon boxes)
            try:
                polys = page["rec_polys"] if "rec_polys" in page else []
            except (TypeError, KeyError):
                polys = []
            if hasattr(polys, "tolist"):
                polys = polys.tolist()

            min_area = 200
            try:
                min_area = max(0, int(float(self.params.get("min_detection_area", {}).get("value", 200))))
            except (TypeError, ValueError):
                pass
            if strict:
                min_area = max(min_area, 250)
            max_ar = 10
            try:
                max_ar = float(self.params.get("max_aspect_ratio", {}).get("value", 10) or 0)
            except (TypeError, ValueError):
                pass

            for box in polys:
                if box is None or (hasattr(box, "__len__") and len(box) < 4):
                    continue
                pts = np.array(box, dtype=np.int32)
                if pts.ndim == 1:
                    pts = pts.reshape(-1, 2)
                if pts.shape[0] < 4:
                    continue
                x1 = int(pts[:, 0].min())
                y1 = int(pts[:, 1].min())
                x2 = int(pts[:, 0].max())
                y2 = int(pts[:, 1].max())
                if x2 <= x1 or y2 <= y1:
                    continue
                if (x2 - x1) * (y2 - y1) < min_area:
                    continue
                if max_ar > 0:
                    bw, bh = x2 - x1, y2 - y1
                    if min(bw, bh) > 0 and max(bw / bh, bh / bw) > max_ar:
                        continue
                blk = TextBlock(xyxy=[x1, y1, x2, y2], lines=[pts.tolist()])
                blk.language = "unknown"
                blk._detected_font_size = max(y2 - y1, 12)
                blk_list.append(blk)
                cv2.fillPoly(mask, [pts], 255)

            if not blk_list:
                return mask, blk_list
            merge_lines = self.params.get("merge_text_lines", {}).get("value", True)
            if merge_lines and len(blk_list) > 0:
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
            merge_gap = 50
            mg = self.params.get("merge_gap_px", {})
            if isinstance(mg, dict):
                try:
                    merge_gap = max(0, int(float(mg.get("value", 50))))
                except (TypeError, ValueError):
                    pass
            same_line_only = self.params.get("merge_same_line_only", {}).get("value", True)
            overlap_ratio = 0.35
            try:
                overlap_ratio = float(self.params.get("merge_line_overlap_ratio", {}).get("value", 0.35) or 0.35)
                overlap_ratio = max(0, min(1, overlap_ratio))
            except (TypeError, ValueError):
                pass
            if strict:
                overlap_ratio = max(overlap_ratio, 0.5)
            blk_list = _merge_nearby_blocks(blk_list, merge_gap, same_line_only=same_line_only, overlap_ratio=overlap_ratio)
            blk_list = sort_regions(blk_list)
            shrink_px = 4
            try:
                shrink_px = max(0, int(float(self.params.get("box_shrink_px", {}).get("value", 4) or 0)))
            except (TypeError, ValueError):
                pass
            if strict:
                shrink_px = max(shrink_px, 5)
            if shrink_px > 0:
                for blk in blk_list:
                    _shrink_block(blk, shrink_px, w, h)
                mask = np.zeros((h, w), dtype=np.uint8)
                for blk in blk_list:
                    for line_pts in blk.lines:
                        pts = np.array(line_pts, dtype=np.int32)
                        if pts.ndim == 1:
                            pts = pts.reshape(-1, 2)
                        cv2.fillPoly(mask, [pts], 255)
            return mask, blk_list
