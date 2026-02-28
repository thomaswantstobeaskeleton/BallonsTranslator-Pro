"""
Hugging Face object-detection detector – generic detector using any HF object-detection model.
Default: ogkalu/comic-text-and-bubble-detector (RT-DETR fine-tuned for comic text and bubbles).
Use model_id for other DETR/RT-DETR-style models. Requires: pip install transformers torch
"""
import numpy as np
import cv2
from typing import Tuple, List

from .base import register_textdetectors, TextDetectorBase, TextBlock, DEVICE_SELECTOR
from utils.textblock import sort_regions


def _iou_xyxy(a, b):
    """Intersection over union for two boxes [x1, y1, x2, y2]."""
    ax1, ay1, ax2, ay2 = a[0], a[1], a[2], a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[2], b[3]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _merge_overlapping_blocks(blk_list, iou_threshold):
    """Merge text blocks that overlap (e.g. bubble + text_bubble for same region)."""
    if not blk_list or iou_threshold <= 0:
        return blk_list
    blks = list(blk_list)
    while True:
        merged_any = False
        for i in range(len(blks)):
            for j in range(i + 1, len(blks)):
                if _iou_xyxy(blks[i].xyxy, blks[j].xyxy) >= iou_threshold:
                    a, b = blks[i], blks[j]
                    x1 = min(a.xyxy[0], b.xyxy[0])
                    y1 = min(a.xyxy[1], b.xyxy[1])
                    x2 = max(a.xyxy[2], b.xyxy[2])
                    y2 = max(a.xyxy[3], b.xyxy[3])
                    pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
                    font_size = max(
                        getattr(a, "_detected_font_size", 12),
                        getattr(b, "_detected_font_size", 12),
                    )
                    merged = TextBlock(xyxy=[x1, y1, x2, y2], lines=[pts.tolist()])
                    merged._detected_font_size = font_size
                    blks = [bx for k, bx in enumerate(blks) if k != i and k != j] + [merged]
                    merged_any = True
                    break
            if merged_any:
                break
        if not merged_any:
            break
    return blks


_HF_DET_AVAILABLE = False
try:
    from transformers import pipeline
    import torch
    from PIL import Image as PILImage
    _HF_DET_AVAILABLE = True
except ImportError:
    import logging
    logging.getLogger("BallonTranslator").debug(
        "HF object-detection detector not available. Install: pip install transformers torch"
    )


if _HF_DET_AVAILABLE:

    @register_textdetectors("hf_object_det")
    class HFObjectDetector(TextDetectorBase):
        """
        Generic text/bubble detection using Hugging Face object-detection pipeline.
        Default: ogkalu/comic-text-and-bubble-detector (bubble, text_bubble, text_free).
        Set model_id to any HF object-detection model (e.g. DETR, RT-DETR).
        """
        params = {
            "model_id": {
                "type": "line_editor",
                "value": "ogkalu/comic-text-and-bubble-detector",
                "description": "Hugging Face model id. Default: comic text+bubble detector (RT-DETR).",
            },
            "score_threshold": {
                "type": "line_editor",
                "value": 0.4,
                "description": "Min detection score for bubble/text_bubble (0.2–0.5). Lower = more boxes.",
            },
            "score_threshold_text_free": {
                "type": "line_editor",
                "value": 0.2,
                "description": "Threshold for text_free (sound effects, captions). Lower (e.g. 0.05) = catch more, more false positives.",
            },
            "labels_include": {
                "type": "line_editor",
                "value": "bubble,text_bubble,text_free",
                "description": "Comma-separated labels to keep (default: all comic classes). Empty = all.",
            },
            "merge_overlap_iou": {
                "type": "line_editor",
                "value": 0.35,
                "description": "Merge only when IoU >= this (0.3–0.5). Higher = don't merge adjacent bubbles; lower = merge more (risk merging 2 bubbles).",
            },
            "detect_min_side": {
                "type": "line_editor",
                "value": 1280,
                "description": "Upscale image so longer side >= this before detection (e.g. 1280). 0 = no upscale.",
            },
            "detect_max_side": {
                "type": "line_editor",
                "value": 1920,
                "description": "Downscale image so longer side <= this (e.g. 1920). Helps free/small text on large pages. 0 = no downscale.",
            },
            "tile_size": {
                "type": "line_editor",
                "value": 0,
                "description": "If > 0, run on overlapping tiles (e.g. 512 or 384). Smaller = catch more tiny text, slower. 0 = off.",
            },
            "tile_overlap": {
                "type": "line_editor",
                "value": 0.5,
                "description": "Tile overlap ratio 0–0.8 (e.g. 0.5 = 50% overlap). Used when tile_size > 0.",
            },
            "device": DEVICE_SELECTOR(),
            "description": "HF object-detection. Default: ogkalu comic text+bubble detector. Install: pip install transformers torch",
        }
        _load_model_keys = {"pipe"}

        def __init__(self, **params) -> None:
            super().__init__(**params)
            self.pipe = None
            self._model_id = None
            self._device = None

        def _load_model(self):
            model_id = (self.params.get("model_id") or {}).get("value", "ogkalu/comic-text-and-bubble-detector") or "ogkalu/comic-text-and-bubble-detector"
            device = (self.params.get("device") or {}).get("value", "cpu")
            if device == "gpu":
                device = "cuda"
            if device == "cuda" and not torch.cuda.is_available():
                device = "cpu"
            if self.pipe is not None and self._model_id == model_id and self._device == device:
                return
            self._model_id = model_id
            self._device = device
            self.pipe = pipeline(
                "object-detection",
                model=model_id,
                device=0 if device == "cuda" else -1,
            )

        def _detect(self, img: np.ndarray, proj=None) -> Tuple[np.ndarray, List[TextBlock]]:
            if img.ndim == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            h, w = img.shape[:2]
            scale = 1.0
            detect_img = img
            h_det, w_det = h, w
            min_side_val = 0
            max_side_val = 0
            dms = self.params.get("detect_min_side", {})
            if isinstance(dms, dict):
                try:
                    min_side_val = int(dms.get("value", 0) or 0)
                except (TypeError, ValueError):
                    pass
            dms_max = self.params.get("detect_max_side", {})
            if isinstance(dms_max, dict):
                try:
                    max_side_val = int(dms_max.get("value", 0) or 0)
                except (TypeError, ValueError):
                    pass
            long_side = max(h, w)
            if min_side_val > 0 and long_side < min_side_val:
                scale = min_side_val / long_side
            elif max_side_val > 0 and long_side > max_side_val:
                scale = max_side_val / long_side
            if scale != 1.0:
                new_w = int(round(w * scale))
                new_h = int(round(h * scale))
                detect_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                h_det, w_det = detect_img.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            blk_list: List[TextBlock] = []

            tile_size_val = 0
            tile_overlap_val = 0.5
            ts = self.params.get("tile_size", {})
            if isinstance(ts, dict):
                try:
                    tile_size_val = int(ts.get("value", 0) or 0)
                except (TypeError, ValueError):
                    pass
            to = self.params.get("tile_overlap", {})
            if isinstance(to, dict):
                try:
                    tile_overlap_val = max(0.0, min(0.95, float(to.get("value", 0.5))))
                except (TypeError, ValueError):
                    pass

            score_thr = 0.4
            st = self.params.get("score_threshold", {})
            if isinstance(st, dict):
                try:
                    score_thr = max(0.0, min(1.0, float(st.get("value", 0.4))))
                except (TypeError, ValueError):
                    pass
            score_thr_text_free = score_thr
            stf = self.params.get("score_threshold_text_free", {})
            if isinstance(stf, dict):
                try:
                    score_thr_text_free = max(0.0, min(1.0, float(stf.get("value", 0.2))))
                except (TypeError, ValueError):
                    pass
            labels_include = (self.params.get("labels_include") or {}).get("value", "") or ""
            allowed = set(s.strip().lower() for s in labels_include.split(",") if s.strip()) if labels_include else None

            def run_on_image(det_img, offset_x=0, offset_y=0):
                out = []
                try:
                    pil_img = PILImage.fromarray(det_img)
                    results = self.pipe(pil_img)
                except Exception as e:
                    self.logger.warning(f"HF object-detection failed: {e}")
                    return out
                if not results:
                    return out
                tw_det, th_det = det_img.shape[1], det_img.shape[0]
                for item in results:
                    if not isinstance(item, dict):
                        continue
                    score = item.get("score", 0)
                    label = (item.get("label") or "").strip().lower()
                    thr = score_thr_text_free if label == "text_free" else score_thr
                    if score < thr:
                        continue
                    if allowed is not None and label not in allowed:
                        continue
                    box = item.get("box")
                    if not box:
                        continue
                    xmin = int(box.get("xmin", 0))
                    ymin = int(box.get("ymin", 0))
                    xmax = int(box.get("xmax", 0))
                    ymax = int(box.get("ymax", 0))
                    x1 = max(0, min(xmin, xmax)) + offset_x
                    x2 = min(tw_det, max(xmin, xmax)) + offset_x
                    y1 = max(0, min(ymin, ymax)) + offset_y
                    y2 = min(th_det, max(ymin, ymax)) + offset_y
                    if x2 <= x1 or y2 <= y1:
                        continue
                    if scale != 1.0:
                        x1 = int(round(x1 / scale))
                        y1 = int(round(y1 / scale))
                        x2 = int(round(x2 / scale))
                        y2 = int(round(y2 / scale))
                        x1 = max(0, min(x1, w))
                        x2 = max(0, min(x2, w))
                        y1 = max(0, min(y1, h))
                        y2 = max(0, min(y2, h))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
                    blk = TextBlock(xyxy=[x1, y1, x2, y2], lines=[pts.tolist()])
                    blk._detected_font_size = max(y2 - y1, 12)
                    out.append(blk)
                return out

            if tile_size_val > 0 and h_det > 0 and w_det > 0:
                stride = max(1, int(tile_size_val * (1.0 - tile_overlap_val)))
                tile_w = min(tile_size_val, w_det)
                tile_h = min(tile_size_val, h_det)
                for ty in range(0, h_det, stride):
                    for tx in range(0, w_det, stride):
                        x1 = tx
                        y1 = ty
                        x2 = min(tx + tile_w, w_det)
                        y2 = min(ty + tile_h, h_det)
                        if x2 <= x1 or y2 <= y1:
                            continue
                        crop = detect_img[y1:y2, x1:x2]
                        if crop.size == 0:
                            continue
                        blk_list.extend(run_on_image(crop, x1, y1))
            else:
                blk_list = run_on_image(detect_img, 0, 0)

            for blk in blk_list:
                if blk.lines:
                    pts = np.array(blk.lines[0], dtype=np.int32)
                else:
                    x1, y1, x2, y2 = blk.xyxy
                    pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
                cv2.fillPoly(mask, [pts], 255)
            merge_iou = 0.35
            mo = self.params.get("merge_overlap_iou", {})
            if isinstance(mo, dict):
                try:
                    merge_iou = max(0.0, min(1.0, float(mo.get("value", 0.35))))
                except (TypeError, ValueError):
                    pass
            blk_list = _merge_overlapping_blocks(blk_list, merge_iou)
            blk_list = sort_regions(blk_list)
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
            if param_key in ("model_id", "device"):
                self.pipe = None
                self._model_id = None
                self._device = None
