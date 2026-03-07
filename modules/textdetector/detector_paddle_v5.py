"""
PaddleOCR PP-OCRv5 text detection only – detection module with latest v5 models.
Uses TextDetection(model_name="PP-OCRv5_mobile_det") when available (PaddleOCR 3.x).
Strong for handwriting, vertical, rotated, curved text; multiple languages.
Requires: paddleocr (3.x for PP-OCRv5), paddlepaddle.
"""
import os
import tempfile
import numpy as np
import cv2
from typing import Tuple, List

from .base import register_textdetectors, TextDetectorBase, TextBlock, ProjImgTrans
from .box_utils import expand_blocks
from ..base import DEVICE_SELECTOR
from utils.textblock import mit_merge_textlines

try:
    from utils.split_text_region import split_textblock
except ImportError:
    split_textblock = None

os.environ.setdefault("PPOCR_HOME", os.path.join("data", "models", "paddle-ocr"))
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# Disable oneDNN/MKLDNN to avoid "ConvertPirAttribute2RuntimeAttribute not support" with PP-OCRv5_server_det on some Paddle versions
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_use_mkldnn_in_fc"] = "0"


def _bbox_distance_px(blk_a: TextBlock, blk_b: TextBlock) -> float:
    """Minimum distance between two axis-aligned boxes; 0 if overlapping."""
    x1_a, y1_a, x2_a, y2_a = blk_a.xyxy
    x1_b, y1_b, x2_b, y2_b = blk_b.xyxy
    dx = max(0, max(x1_a, x1_b) - min(x2_a, x2_b))
    dy = max(0, max(y1_a, y1_b) - min(y2_a, y2_b))
    return (dx * dx + dy * dy) ** 0.5


def _overlap_min_side(blk_a: TextBlock, blk_b: TextBlock) -> int:
    """When overlapping, return min(overlap_width, overlap_height) in px; else 0."""
    x1_a, y1_a, x2_a, y2_a = blk_a.xyxy
    x1_b, y1_b, x2_b, y2_b = blk_b.xyxy
    ow = max(0, min(x2_a, x2_b) - max(x1_a, x1_b))
    oh = max(0, min(y2_a, y2_b) - max(y1_a, y1_b))
    return min(ow, oh)


def _merge_nearby_blocks(
    blk_list: List[TextBlock], gap_px: int, min_overlap_px: int = 0
) -> List[TextBlock]:
    """Merge blocks within gap_px. If min_overlap_px > 0, only merge overlapping boxes whose overlap is at least that (keeps touching/barely overlapping separate)."""
    if gap_px < 0 or len(blk_list) <= 1:
        return blk_list
    merged: List[TextBlock] = []
    for blk in blk_list:
        combined = False
        for m in merged:
            if _bbox_distance_px(m, blk) > gap_px:
                continue
            if min_overlap_px > 0 and _overlap_min_side(m, blk) < min_overlap_px:
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


def _split_block_by_image_gap(
    img: np.ndarray, blk: TextBlock, min_block_height: int = 40, min_span_height: int = 15
) -> List[TextBlock]:
    """If the block appears to span two regions (e.g. two bubbles), split by vertical gap in the crop. Returns [blk] if no split."""
    if split_textblock is None:
        return [blk]
    im_h, im_w = img.shape[:2]
    x1, y1, x2, y2 = blk.xyxy
    x1 = max(0, min(int(round(float(x1))), im_w - 1))
    y1 = max(0, min(int(round(float(y1))), im_h - 1))
    x2 = max(x1, min(int(round(float(x2))), im_w - 1))
    y2 = max(y1, min(int(round(float(y2))), im_h - 1))
    bw, bh = x2 - x1, y2 - y1
    if bh < min_block_height or bw < 30:
        return [blk]
    try:
        crop = img[y1:y2 + 1, x1:x2 + 1]
        if crop.size == 0:
            return [blk]
        if crop.ndim == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        else:
            gray = np.asarray(crop, dtype=np.uint8)
        span_list, _ = split_textblock(
            gray, crop_ratio=0, discard=True, shrink=True, recheck=False
        )
        if not span_list or len(span_list) < 2:
            return [blk]
        out = []
        for s in span_list:
            sh = (s.bottom - s.top) if s.bottom is not None and s.top is not None else 0
            if sh < min_span_height:
                continue
            left_im = x1 + (s.left or 0)
            right_im = x1 + (s.right or bw)
            top_im = y1 + (s.top or 0)
            bottom_im = y1 + (s.bottom or bh)
            if bottom_im - top_im < min_span_height:
                continue
            pts = [
                [left_im, top_im],
                [right_im, top_im],
                [right_im, bottom_im],
                [left_im, bottom_im],
            ]
            new_blk = TextBlock(xyxy=[left_im, top_im, right_im, bottom_im], lines=[pts])
            new_blk.language = getattr(blk, "language", "unknown")
            new_blk._detected_font_size = max(bottom_im - top_im, 12)
            out.append(new_blk)
        return out if len(out) >= 2 else [blk]
    except Exception:
        return [blk]


_PADDLE_V5_AVAILABLE = False
try:
    from paddleocr import TextDetection
    _PADDLE_V5_AVAILABLE = True
except ImportError:
    import logging
    logging.getLogger("BallonTranslator").debug(
        "PaddleOCR TextDetection (PP-OCRv5) not available. Install paddleocr 3.x and paddlepaddle."
    )


if _PADDLE_V5_AVAILABLE:

    @register_textdetectors("paddle_det_v5")
    class PaddleDetectorV5(TextDetectorBase):
        """
        Text detection using PaddleOCR PP-OCRv5 (TextDetection module only).
        Handwriting, vertical, rotated, curved text; Chinese, English, Japanese, etc.
        For narrative/vertical text: try limit_side_len 1280–1600, lower thresholds, merge_min_overlap_px 0, merge_overlapping_gap_px 15–30. For white-on-dark manga text, CTD or Surya may work better.
        """
        params = {
            "model_name": {
                "type": "selector",
                "options": ["PP-OCRv5_mobile_det", "PP-OCRv5_server_det"],
                "value": "PP-OCRv5_mobile_det",
                "description": "PP-OCRv5 mobile (faster) or server. If either fails with oneDNN/PIR error, install paddlepaddle==3.2.0 with paddleocr 3.3.x (see doc/INSTALL_PADDLEOCR.md).",
            },
            "device": DEVICE_SELECTOR(),
            "det_score_thresh": {
                "type": "line_editor",
                "value": 0.25,
                "description": "Pixel score threshold. Lower = more sensitive (try 0.2 or 0.15 for narrative/non-bubble text).",
            },
            "det_box_thresh": {
                "type": "line_editor",
                "value": 0.5,
                "description": "Box score threshold. Lower = more detections (try 0.4 for narrative/non-bubble text).",
            },
            "min_box_side_px": {
                "type": "line_editor",
                "value": 24,
                "description": "Drop boxes smaller than this (px). Avoids tiny crops that Ocean OCR rejects (e.g. 14x14).",
            },
            "limit_side_len": {
                "type": "line_editor",
                "value": 1600,
                "description": "Max side length sent to detector (0 = model default). 1600 balanced; 2000–2400 best quality (slower, more VRAM).",
            },
            "merge_text_lines": {
                "type": "checkbox",
                "value": True,
                "description": "Merge nearby lines into one bubble. When 'Merge overlapping blocks' is off and gap is 0, this is skipped so each line stays one box (keeps two close bubbles separate).",
            },
            "merge_overlapping_blocks": {
                "type": "checkbox",
                "value": True,
                "description": "Merge only overlapping boxes into one (fixes 2 boxes in 1 bubble). Off + gap 0 = no merging at all (one box per detected line).",
            },
            "merge_overlapping_gap_px": {
                "type": "line_editor",
                "value": 0,
                "description": "Max gap (px) to merge. 0 = no line grouping (one box per detected line; keeps two close bubbles separate). Use 15–30 only for narrative text.",
            },
            "merge_min_overlap_px": {
                "type": "line_editor",
                "value": 50,
                "description": "Only merge if overlap ≥ this (px). For narrative text use 0 so vertical chars in a line merge.",
            },
            "split_cross_bubble_lines": {
                "type": "checkbox",
                "value": True,
                "description": "Split a single detection that spans two bubbles into two blocks (uses image gap detection). Turn off if it splits normal blocks.",
            },
            "box_padding": {
                "type": "line_editor",
                "value": 0,
                "description": "Pixels to add around each detected box (all sides). Reduces clipped punctuation (?, !) and character edges. Recommended 4–6.",
            },
            "description": "PaddleOCR PP-OCRv5 text detection only (requires paddleocr 3.x).",
        }
        _load_model_keys = {"model"}

        def __init__(self, **params) -> None:
            super().__init__(**params)
            self.model = None
            self._model_name = None
            self._device = None

        def _load_model(self):
            model_name = (self.params.get("model_name") or {}).get("value", "PP-OCRv5_mobile_det") or "PP-OCRv5_mobile_det"
            dev = (self.params.get("device") or {}).get("value", "cpu")
            device = "gpu:0" if dev in ("cuda", "gpu") else "cpu"
            if self.model is not None and self._model_name == model_name and self._device == device:
                return
            self._model_name = model_name
            self._device = device
            try:
                try:
                    import paddle
                    paddle.set_flags({"FLAGS_use_mkldnn": False})
                except Exception:
                    pass
                self.model = TextDetection(model_name=model_name, device=device)
            except ImportError as e:
                if "already registered" in str(e) or "_gpuDeviceProperties" in str(e):
                    raise RuntimeError(
                        "PaddlePaddle and PyTorch conflict in this process (CUDA type already registered). "
                        "Restart the app and run detection with paddle_det_v5 before opening other GPU models, "
                        "or use CTD or Surya detection instead."
                    ) from e
                raise

        def _detect(self, img: np.ndarray, proj: ProjImgTrans = None) -> Tuple[np.ndarray, List[TextBlock]]:
            if img.ndim == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            h, w = img.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            blk_list: List[TextBlock] = []
            tmp_path = None
            try:
                fd, tmp_path = tempfile.mkstemp(suffix=".png")
                os.close(fd)
                cv2.imwrite(tmp_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                thresh = 0.25
                box_thresh = 0.5
                try:
                    pt = self.params.get("det_score_thresh", {})
                    if isinstance(pt, dict):
                        thresh = float(pt.get("value", 0.25))
                    bt = self.params.get("det_box_thresh", {})
                    if isinstance(bt, dict):
                        box_thresh = float(bt.get("value", 0.5))
                except (TypeError, ValueError):
                    pass
                limit_side = None
                try:
                    ls = self.params.get("limit_side_len", {})
                    if isinstance(ls, dict):
                        v = ls.get("value", 0)
                        if v is not None and int(float(v)) > 0:
                            limit_side = int(float(v))
                except (TypeError, ValueError):
                    pass
                predict_kw = dict(input=tmp_path, batch_size=1, thresh=thresh, box_thresh=box_thresh)
                if limit_side is not None:
                    predict_kw["limit_side_len"] = limit_side
                output = self.model.predict(**predict_kw)
                # PaddleOCR 3.x may return a generator or list of result objects (with .json), not raw dicts
                output = list(output) if output is not None else []
            except Exception as e:
                err_msg = str(e)
                self.logger.error(f"Paddle v5 det failed: {e}")
                if "ConvertPirAttribute2RuntimeAttribute" in err_msg or "onednn_instruction" in err_msg:
                    self.logger.warning(
                        "PP-OCRv5 hit Paddle/oneDNN bug (both mobile and server). Fix: pip install paddlepaddle==3.2.0 paddleocr==3.3.0 then restart. See PaddleOCR discussion #17350."
                    )
                return mask, blk_list
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
            if not output:
                return mask, blk_list
            for item in output:
                # PaddleOCR 3.x returns result objects with .json (dict); fallback to item if already dict
                if hasattr(item, "json"):
                    data = item.json() if callable(getattr(item, "json")) else item.json
                elif isinstance(item, dict):
                    data = item
                else:
                    continue
                if not isinstance(data, dict):
                    continue
                res = data.get("res")
                if not isinstance(res, dict):
                    res = data if isinstance(data, dict) and "dt_polys" in data else None
                if not isinstance(res, dict):
                    continue
                polys = res.get("dt_polys")
                scores = res.get("dt_scores", [])
                if polys is None:
                    continue
                if hasattr(polys, "tolist"):
                    polys = polys.tolist()
                if not isinstance(polys, list):
                    continue
                for idx, box in enumerate(polys):
                    if box is None or (hasattr(box, "__len__") and len(box) < 4):
                        continue
                    pts = np.array(box, dtype=np.int32)
                    if pts.ndim == 1:
                        pts = pts.reshape(-1, 2)
                    if pts.shape[0] < 3:
                        continue
                    score = float(scores[idx]) if idx < len(scores) else 1.0
                    score_thresh = 0.25
                    pt = self.params.get("det_score_thresh", {})
                    if isinstance(pt, dict):
                        try:
                            score_thresh = float(pt.get("value", 0.25))
                        except (TypeError, ValueError):
                            pass
                    if score < score_thresh:
                        continue
                    x1 = int(pts[:, 0].min())
                    y1 = int(pts[:, 1].min())
                    x2 = int(pts[:, 0].max())
                    y2 = int(pts[:, 1].max())
                    if x2 <= x1 or y2 <= y1:
                        continue
                    min_side = 24
                    try:
                        ms = self.params.get("min_box_side_px", {})
                        if isinstance(ms, dict):
                            min_side = max(0, int(float(ms.get("value", 24))))
                    except (TypeError, ValueError):
                        pass
                    if min(x2 - x1, y2 - y1) < min_side:
                        continue
                    blk = TextBlock(xyxy=[x1, y1, x2, y2], lines=[pts.tolist()])
                    blk.language = "unknown"
                    blk._detected_font_size = max(y2 - y1, 12)
                    blk_list.append(blk)
                    cv2.fillPoly(mask, [pts], 255)
            if not blk_list:
                return mask, blk_list
            merge_overlap = self.params.get("merge_overlapping_blocks", {}).get("value", True)
            merge_gap = 0
            min_overlap = 50
            try:
                g = self.params.get("merge_overlapping_gap_px", {})
                if isinstance(g, dict):
                    merge_gap = max(0, int(float(g.get("value", 0))))
                mo = self.params.get("merge_min_overlap_px", {})
                if isinstance(mo, dict):
                    min_overlap = max(0, int(float(mo.get("value", 50))))
            except (TypeError, ValueError):
                pass
            merge_lines = self.params.get("merge_text_lines", {}).get("value", True)
            # Only run line grouping when gap > 0. When gap is 0 we keep one box per detected line
            # so two close bubbles are never merged by mit_merge_textlines (even if overlap merge is on).
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
            if merge_overlap and len(blk_list) > 1:
                gap = merge_gap
                blk_list = _merge_nearby_blocks(blk_list, gap, min_overlap_px=min_overlap)
                mask = np.zeros((h, w), dtype=np.uint8)
                for blk in blk_list:
                    for line_pts in blk.lines:
                        pts = np.array(line_pts, dtype=np.int32)
                        if pts.ndim == 1:
                            pts = pts.reshape(-1, 2)
                        cv2.fillPoly(mask, [pts], 255)
            split_cross = self.params.get("split_cross_bubble_lines", {}).get("value", True)
            if split_cross and split_textblock is not None and len(blk_list) > 0:
                new_list = []
                for blk in blk_list:
                    split_blks = _split_block_by_image_gap(img, blk)
                    new_list.extend(split_blks)
                if len(new_list) != len(blk_list):
                    blk_list = new_list
                    mask = np.zeros((h, w), dtype=np.uint8)
                    for blk in blk_list:
                        for line_pts in blk.lines:
                            pts = np.array(line_pts, dtype=np.int32)
                            if pts.ndim == 1:
                                pts = pts.reshape(-1, 2)
                            cv2.fillPoly(mask, [pts], 255)
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
                    for line_pts in blk.lines:
                        pts = np.array(line_pts, dtype=np.int32)
                        if pts.ndim == 1:
                            pts = pts.reshape(-1, 2)
                        cv2.fillPoly(mask, [pts], 255)
            return mask, blk_list

        def updateParam(self, param_key: str, param_content):
            super().updateParam(param_key, param_content)
            if param_key in ("model_name", "device"):
                self.model = None
                self._model_name = None
                self._device = None
