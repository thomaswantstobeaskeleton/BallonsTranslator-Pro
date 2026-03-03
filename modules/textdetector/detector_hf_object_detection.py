"""
Hugging Face object-detection detector – generic detector using any HF object-detection model,
or a YOLO .pt as primary (no HF). Use model_id: HF id (e.g. ogkalu/...) or path to .pt
(e.g. data/models/ysgyolo_comic_text_segmenter_v8m.pt). Requires: pip install transformers torch
(and ultralytics for YOLO primary).
"""
import copy
import os.path as osp
import numpy as np
import cv2
from typing import Tuple, List

from .base import register_textdetectors, TextDetectorBase, TextBlock, DEVICE_SELECTOR
from .box_utils import expand_blocks
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


def _inset_blocks(blk_list: List[TextBlock], ratio: float, img_w: int, img_h: int) -> List[TextBlock]:
    """Shrink each block inward by ratio of its width/height."""
    if ratio <= 0:
        return blk_list
    import copy as copy_mod
    out = []
    for blk in blk_list:
        x1, y1, x2, y2 = blk.xyxy
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            out.append(blk)
            continue
        dx = max(0, int(w * ratio))
        dy = max(0, int(h * ratio))
        x1n = min(x1 + dx, x2 - 1)
        y1n = min(y1 + dy, y2 - 1)
        x2n = max(x2 - dx, x1 + 1)
        y2n = max(y2 - dy, y1 + 1)
        x1n = max(0, min(x1n, img_w - 1))
        y1n = max(0, min(y1n, img_h - 1))
        x2n = max(0, min(x2n, img_w))
        y2n = max(0, min(y2n, img_h))
        if x2n <= x1n or y2n <= y1n:
            out.append(blk)
            continue
        new_blk = copy.copy(blk)
        new_blk.xyxy = [x1n, y1n, x2n, y2n]
        new_blk.lines = [[[x1n, y1n], [x2n, y1n], [x2n, y2n], [x1n, y2n]]]
        out.append(new_blk)
    return out


def _contains_xyxy(outer, inner):
    """True if outer fully contains inner [x1,y1,x2,y2]."""
    o1, o2, o3, o4 = outer[0], outer[1], outer[2], outer[3]
    i1, i2, i3, i4 = inner[0], inner[1], inner[2], inner[3]
    return o1 <= i1 and o2 <= i2 and o3 >= i3 and o4 >= i4


def _dedup_blocks_by_iou(blk_list: List["TextBlock"], iou_threshold: float) -> List["TextBlock"]:
    """Merge/deduplicate blocks with IoU >= iou_threshold (keep larger or first). Returns new list."""
    if not blk_list or iou_threshold <= 0:
        return list(blk_list)
    kept = []
    for blk in blk_list:
        xyxy = blk.xyxy
        merged = False
        for i, k in enumerate(kept):
            if _iou_xyxy(xyxy, k.xyxy) >= iou_threshold:
                # Merge: use union box
                x1 = min(xyxy[0], k.xyxy[0])
                y1 = min(xyxy[1], k.xyxy[1])
                x2 = max(xyxy[2], k.xyxy[2])
                y2 = max(xyxy[3], k.xyxy[3])
                pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
                new_blk = TextBlock(xyxy=[x1, y1, x2, y2], lines=[pts.tolist()])
                new_blk._detected_font_size = max(getattr(blk, "_detected_font_size", 12), getattr(k, "_detected_font_size", 12))
                new_blk.label = getattr(k, "label", None) or getattr(blk, "label", None)
                kept[i] = new_blk
                merged = True
                break
        if not merged:
            kept.append(copy.copy(blk))
    return kept


def _clip_blocks_to_page(blk_list: List[TextBlock], img_w: int, img_h: int) -> List[TextBlock]:
    """Clip every block's xyxy and lines to page bounds [0, img_w] x [0, img_h].
    Polygon points are clamped to valid mask indices [0, img_w-1] x [0, img_h-1] for fillPoly.
    Returns only non-degenerate blocks so mask and inpainter get consistent, page-sized data."""
    if not blk_list or img_w <= 0 or img_h <= 0:
        return blk_list
    out = []
    for blk in blk_list:
        x1, y1, x2, y2 = blk.xyxy
        try:
            x1 = max(0, min(int(round(float(x1))), img_w - 1))
            x2 = max(0, min(int(round(float(x2))), img_w))
            y1 = max(0, min(int(round(float(y1))), img_h - 1))
            y2 = max(0, min(int(round(float(y2))), img_h))
        except (TypeError, ValueError):
            out.append(blk)
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        new_blk = copy.copy(blk)
        new_blk.xyxy = [x1, y1, x2, y2]
        if getattr(blk, 'lines', None) and len(blk.lines) > 0:
            try:
                pts = np.array(blk.lines[0], dtype=np.float64)
                if pts.ndim == 2 and pts.shape[0] >= 3 and pts.shape[1] >= 2:
                    pts[:, 0] = np.clip(pts[:, 0], 0, img_w - 1)
                    pts[:, 1] = np.clip(pts[:, 1], 0, img_h - 1)
                    new_blk.lines = [pts.astype(np.int32).tolist()]
                else:
                    new_blk.lines = [[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]]
            except (TypeError, ValueError, IndexError):
                new_blk.lines = [[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]]
        else:
            new_blk.lines = [[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]]
        out.append(new_blk)
    return out


def _merge_overlapping_blocks(blk_list, iou_threshold):
    """Merge text blocks that overlap (e.g. bubble + text_bubble for same region).
    Merges when IoU >= threshold or when one box fully contains the other (nested bubble+text_bubble)."""
    if not blk_list or iou_threshold <= 0:
        return blk_list
    blks = list(blk_list)
    while True:
        merged_any = False
        for i in range(len(blks)):
            for j in range(i + 1, len(blks)):
                a, b = blks[i], blks[j]
                iou = _iou_xyxy(a.xyxy, b.xyxy)
                contained = _contains_xyxy(a.xyxy, b.xyxy) or _contains_xyxy(b.xyxy, a.xyxy)
                if iou >= iou_threshold or contained:
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
                "value": "data/models/ysgyolo_comic_text_segmenter_v8m.pt",
                "description": "Hugging Face model id (e.g. ogkalu/comic-text-and-bubble-detector) OR path to YOLO .pt to use YOLO as primary (no HF). E.g. data/models/ysgyolo_comic_text_segmenter_v8m.pt",
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
                "description": "Comma-separated labels to keep: bubble, text_bubble, text_free. Typo 'bble' is treated as 'bubble'. Empty = all.",
            },
            "merge_overlap_iou": {
                "type": "line_editor",
                "value": 0.35,
                "description": "Merge only when IoU >= this (0.3–0.5). Higher = don't merge adjacent bubbles; 1.0 = disable merge (one box per detection).",
            },
            "box_inset_ratio": {
                "type": "line_editor",
                "value": 0,
                "description": "Shrink each box inward by this ratio of its width/height (0–0.2). E.g. 0.05 = 5% per side, box becomes 90% size. Does not split combined bubbles.",
            },
            "box_padding": {
                "type": "line_editor",
                "value": 5,
                "description": "Pixels to add around each detected box (all sides). Reduces clipped punctuation (?, !) and bubble edges. Recommended 4–6; 0 = tight box.",
            },
            "detect_min_side": {
                "type": "line_editor",
                "value": 1280,
                "description": "Upscale image so longer side >= this before detection (e.g. 1280). 0 = no upscale. Output mask/blocks are always in page (original) size.",
            },
            "detect_max_side": {
                "type": "line_editor",
                "value": 1920,
                "description": "Downscale image so longer side <= this for detection (e.g. 1920). 0 = no downscale. Output mask/blocks are always in page (original) size.",
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
            "enable_conjoined_secondary": {
                "type": "checkbox",
                "value": False,
                "description": "Run one or more conjoined models to split merged bubbles. Use conjoined_yolo_paths and/or conjoined_model_ids (multiple allowed).",
            },
            "conjoined_backend": {
                "type": "selector",
                "options": ["yolo", "hf", "both"],
                "value": "yolo",
                "description": "Which conjoined sources to run: yolo = YOLO paths only; hf = HF model ids only; both = run both lists and merge.",
            },
            "conjoined_yolo_paths": {
                "type": "line_editor",
                "value": "data/models/ysgyolo_comic_text_segmenter_v8m.pt\ndata/models/ysgyolo_comic_speech_bubble_v8m.pt",
                "description": "YOLO .pt paths: one per line or comma-separated. Up to 10. All run and results merged. Best quality: use 3–5 models (see docs/CONJOINED_MODELS.md).",
            },
            "conjoined_yolo_path": {
                "type": "line_editor",
                "value": "",
                "description": "(Legacy) Single YOLO path. Ignored if conjoined_yolo_paths has entries.",
            },
            "conjoined_model_ids": {
                "type": "line_editor",
                "value": "",
                "description": "HF object-detection model ids: comma-separated. Optional; run when conjoined_backend=hf or both. E.g. ogkalu/comic-text-and-bubble-detector",
            },
            "conjoined_model_id": {
                "type": "line_editor",
                "value": "",
                "description": "(Legacy) Single HF model id. Ignored if conjoined_model_ids has entries.",
            },
            "conjoined_score_threshold": {
                "type": "line_editor",
                "value": 0.4,
                "description": "Min detection score for secondary conjoined model.",
            },
            "conjoined_labels_include": {
                "type": "line_editor",
                "value": "bubble,text_bubble,text_free",
                "description": "Comma-separated labels to keep for conjoined output: bubble, text_bubble, text_free. Include text_free to detect captions/SFX outside bubbles. Empty = all.",
            },
            "conjoined_min_boxes_in_primary": {
                "type": "line_editor",
                "value": 2,
                "description": "Only replace a primary box if the conjoined model finds at least this many boxes inside it.",
            },
            "conjoined_min_box_area": {
                "type": "line_editor",
                "value": 2500,
                "description": "Ignore conjoined detections smaller than this area (px). Lower (e.g. 500–1000) to keep small SFX/captions.",
            },
            "conjoined_dedup_iou": {
                "type": "line_editor",
                "value": 0.85,
                "description": "When using multiple conjoined models: merge boxes with IoU >= this (0 = off, 0.8–0.95). Reduces duplicates.",
            },
            "description": "HF object-detection. Default: ogkalu comic text+bubble detector. Install: pip install transformers torch",
        }
        _load_model_keys = {"pipe", "model_primary", "pipe_conjoined", "pipe_conjoined_list", "model_conjoined", "models_conjoined"}

        def __init__(self, **params) -> None:
            super().__init__(**params)
            self.pipe = None
            self.model_primary = None
            self.pipe_conjoined = None
            self.pipe_conjoined_list: List = []
            self.model_conjoined = None
            self.models_conjoined: List = []
            self._model_id = None
            self._device = None
            self._conjoined_model_id = None
            self._conjoined_model_ids: List[str] = []
            self._conjoined_yolo_path = None
            self._conjoined_yolo_paths: List[str] = []
            self._conjoined_device = None

        @staticmethod
        def _parse_multiline_paths(s) -> List[str]:
            """Parse comma or newline separated paths; return non-empty stripped list (max 10)."""
            if not s or not isinstance(s, str):
                return []
            parts = s.replace(",", "\n").splitlines()
            out = [p.strip() for p in parts if p and p.strip()]
            return out[:10]

        @staticmethod
        def _parse_comma_ids(s) -> List[str]:
            """Parse comma-separated model ids; return non-empty stripped list (max 5)."""
            if not s or not isinstance(s, str):
                return []
            parts = s.split(",")
            out = [p.strip() for p in parts if p and p.strip()]
            return out[:5]

        def _normalize_model_id(self, model_id: str) -> str:
            model_id = (model_id or "").strip()
            if model_id and "/" not in model_id:
                if model_id.lower() in ("text-and-bubble-detector", "comic-text-and-bubble-detector"):
                    return "ogkalu/comic-text-and-bubble-detector"
            return model_id

        @staticmethod
        def _repo_root() -> str:
            """Project root (folder containing launch.py)."""
            return osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)), "..", ".."))

        def _resolve_model_path(self, model_id: str) -> str:
            """Return path to use for loading; resolve data/... relative to repo root if not found in cwd."""
            s = (model_id or "").strip()
            if not s or s.startswith("http"):
                return s
            if osp.isfile(s):
                return s
            root = self._repo_root()
            candidate = osp.normpath(osp.join(root, s))
            if osp.isfile(candidate):
                return candidate
            return s

        def _is_yolo_primary_path(self, model_id: str) -> bool:
            """True if model_id looks like a path to a YOLO .pt file that exists."""
            if not model_id or not isinstance(model_id, str):
                return False
            s = self._resolve_model_path(model_id)
            if not s or s.startswith("http"):
                return False
            if "/" in s or "\\" in s or s.endswith(".pt") or s.endswith(".pth") or "data" in s:
                return osp.isfile(s)
            return False

        def _run_yolo_primary(
            self,
            model,
            detect_img: np.ndarray,
            scale: float,
            w: int,
            h: int,
            score_thr: float,
            score_thr_text_free: float,
            allowed: set,
        ) -> List[TextBlock]:
            """Run primary YOLO model; return TextBlocks in page coords (same label filtering as HF)."""
            if model is None:
                return []
            def _predict():
                return model.predict(
                    source=detect_img,
                    save=False,
                    show=False,
                    verbose=False,
                    conf=0.01,
                    iou=0.5,
                    agnostic_nms=True,
                )
            try:
                results = _predict()
            except Exception as e:
                err_str = str(e).lower()
                if "out of memory" in err_str:
                    try:
                        model.to("cpu")
                        results = _predict()
                        self.logger.info("Primary YOLO ran on CPU after GPU OOM.")
                    except Exception as e2:
                        self.logger.warning("Primary YOLO predict failed (GPU OOM, CPU retry failed): %s", e2)
                        return []
                else:
                    self.logger.warning("Primary YOLO predict failed: %s", e)
                    return []
            if not results or len(results) == 0:
                return []
            result = results[0]
            dets = result.boxes
            if dets is None or len(dets.cls) == 0:
                return []
            raw_names = getattr(result, "names", None)
            if isinstance(raw_names, (list, tuple)):
                names = {i: (raw_names[i] if i < len(raw_names) else str(i)) for i in range(max(len(raw_names), 1))}
            else:
                names = raw_names or {}
            out = []
            for i in range(len(dets.cls)):
                cls_idx = int(dets.cls[i])
                label = (names.get(cls_idx, str(cls_idx)) or str(cls_idx)).strip().lower()
                if label in ("text", "text_region", "text_block", "character", "char"):
                    label = "text_bubble"
                elif label in ("balloon", "speech_bubble"):
                    label = "bubble"
                elif label in ("onomatopoeia", "caption", "sfx", "sound_effect", "soundeffect"):
                    label = "text_free"
                if label not in ("bubble", "text_bubble", "text_free") and (allowed is None or "text_bubble" in (allowed or set())):
                    label = "text_bubble"
                thr = score_thr_text_free if label == "text_free" else score_thr
                if dets.conf is not None and i < len(dets.conf) and float(dets.conf[i]) < thr:
                    continue
                if allowed is not None and label not in allowed:
                    continue
                xyxy = dets.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
                if scale != 1.0 and scale > 0:
                    x1, y1, x2, y2 = x1 / scale, y1 / scale, x2 / scale, y2 / scale
                x1 = max(0, min(int(round(x1)), w))
                x2 = max(0, min(int(round(x2)), w))
                y1 = max(0, min(int(round(y1)), h))
                y2 = max(0, min(int(round(y2)), h))
                if x2 <= x1 or y2 <= y1:
                    continue
                pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
                blk = TextBlock(xyxy=[x1, y1, x2, y2], lines=[pts.tolist()])
                blk._detected_font_size = max(y2 - y1, 12)
                blk.label = label
                out.append(blk)
            return out

        def _run_yolo_conjoined(
            self,
            model,
            detect_img: np.ndarray,
            scale: float,
            w: int,
            h: int,
            conj_thr: float,
            conj_allowed: set,
            min_area_px: int,
        ) -> List[TextBlock]:
            """Run one YOLO conjoined model on image and return TextBlocks (page coords)."""
            if model is None:
                return []
            def _predict():
                return model.predict(
                    source=detect_img,
                    save=False,
                    show=False,
                    verbose=False,
                    conf=0.05,
                    iou=0.5,
                    agnostic_nms=True,
                )
            try:
                results = _predict()
            except Exception as e:
                err_str = str(e).lower()
                if "out of memory" in err_str:
                    try:
                        model.to("cpu")
                        results = _predict()
                    except Exception as e2:
                        self.logger.warning("Conjoined YOLO predict failed (GPU OOM, CPU retry failed): %s", e2)
                        return []
                else:
                    self.logger.warning("Conjoined YOLO predict failed: %s", e)
                    return []
            if not results or len(results) == 0:
                return []
            result = results[0]
            dets = result.boxes
            if dets is None or len(dets.cls) == 0:
                return []
            names = getattr(result, "names", {}) or {}
            out = []
            for i in range(len(dets.cls)):
                cls_idx = int(dets.cls[i])
                label = (names.get(cls_idx, "unknown") or "unknown").strip().lower()
                if conj_allowed is not None:
                    if label not in conj_allowed:
                        # Map common YOLO names to HF-style labels
                        if label in ("text", "text_region", "text_block", "character", "char") and "text_bubble" in conj_allowed:
                            label = "text_bubble"
                        elif label in ("balloon", "bubble", "speech_bubble") and "bubble" in conj_allowed:
                            label = "bubble"
                        elif label in ("onomatopoeia", "caption", "sfx", "sound_effect", "soundeffect") and "text_free" in conj_allowed:
                            label = "text_free"
                        elif label not in conj_allowed:
                            continue
                if dets.conf is not None and i < len(dets.conf) and float(dets.conf[i]) < conj_thr:
                    continue
                xyxy = dets.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
                if scale != 1.0 and scale > 0:
                    x1 = x1 / scale
                    y1 = y1 / scale
                    x2 = x2 / scale
                    y2 = y2 / scale
                x1 = max(0, min(int(round(x1)), w))
                x2 = max(0, min(int(round(x2)), w))
                y1 = max(0, min(int(round(y1)), h))
                y2 = max(0, min(int(round(y2)), h))
                if x2 <= x1 or y2 <= y1:
                    continue
                area = (x2 - x1) * (y2 - y1)
                if area < min_area_px:
                    continue
                pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
                blk = TextBlock(xyxy=[x1, y1, x2, y2], lines=[pts.tolist()])
                blk._detected_font_size = max(y2 - y1, 12)
                blk.label = label
                out.append(blk)
            return out

        def _load_model(self):
            model_id = (self.params.get("model_id") or {}).get("value", "data/models/ysgyolo_comic_text_segmenter_v8m.pt") or "data/models/ysgyolo_comic_text_segmenter_v8m.pt"
            model_id = (model_id or "").strip()
            model_id = self._normalize_model_id(model_id) or model_id
            if not model_id:
                model_id = "ogkalu/comic-text-and-bubble-detector"
            device = (self.params.get("device") or {}).get("value", "cpu")
            if device == "gpu":
                device = "cuda"
            if device == "cuda" and not torch.cuda.is_available():
                device = "cpu"

            use_yolo_primary = self._is_yolo_primary_path(model_id)
            if use_yolo_primary:
                if self.model_primary is not None and self._model_id == model_id and self._device == device:
                    pass
                else:
                    self._model_id = model_id
                    self._device = device
                    self.pipe = None
                    try:
                        from ultralytics import YOLO as YOLOCls
                        path = self._resolve_model_path(model_id)
                        self.model_primary = YOLOCls(path)
                        if device == "cuda":
                            try:
                                self.model_primary.to("cuda")
                            except Exception as oom:
                                if "out of memory" in str(oom).lower():
                                    self.logger.warning("GPU OOM loading primary YOLO; using CPU for detector.")
                                    self.model_primary.to("cpu")
                                    self._device = "cpu"
                                else:
                                    raise
                        self.logger.info("Primary detector: YOLO from %s", path)
                    except Exception as e:
                        self.logger.warning("Failed to load primary YOLO %s: %s", model_id, e)
                        self.model_primary = None
            else:
                if self.pipe is not None and self._model_id == model_id and self._device == device:
                    pass
                else:
                    self._model_id = model_id
                    self._device = device
                    self.model_primary = None
                    resolved = self._resolve_model_path(model_id)
                    looks_like_path = resolved.endswith(".pt") or "/" in resolved or "\\" in resolved
                    if looks_like_path and not osp.isfile(resolved):
                        self.logger.warning("model_id looks like a path but file not found; primary disabled. Set model_id to an HF id or a valid .pt path.")
                        self.pipe = None
                    else:
                        try:
                            self.pipe = pipeline(
                                "object-detection",
                                model=model_id,
                                device=0 if device == "cuda" else -1,
                            )
                            self.logger.info("Primary detector: HF pipeline %s", model_id)
                        except Exception as e:
                            self.logger.warning("Failed to load HF model %s: %s", model_id, e)

            # Optional conjoined model(s): multiple YOLO and/or HF
            enable_conj = bool((self.params.get("enable_conjoined_secondary") or {}).get("value", False))
            conj_backend = (self.params.get("conjoined_backend") or {}).get("value", "yolo") or "yolo"
            yolo_paths = self._parse_multiline_paths((self.params.get("conjoined_yolo_paths") or {}).get("value", "") or "")
            if not yolo_paths:
                single_yolo = (self.params.get("conjoined_yolo_path") or {}).get("value", "") or ""
                if isinstance(single_yolo, str) and single_yolo.strip():
                    yolo_paths = [single_yolo.strip()]
            hf_ids = self._parse_comma_ids((self.params.get("conjoined_model_ids") or {}).get("value", "") or "")
            if not hf_ids:
                single_hf = (self.params.get("conjoined_model_id") or {}).get("value", "") or ""
                if isinstance(single_hf, str) and single_hf.strip():
                    single_hf = self._normalize_model_id(single_hf.strip())
                    if single_hf:
                        hf_ids = [single_hf]
            device = (self.params.get("device") or {}).get("value", "cpu")
            if device == "gpu":
                device = "cuda"
            if device == "cuda" and not torch.cuda.is_available():
                device = "cpu"

            if not enable_conj:
                self.pipe_conjoined = None
                self.pipe_conjoined_list = []
                self.model_conjoined = None
                self.models_conjoined = []
                self._conjoined_model_id = None
                self._conjoined_model_ids = []
                self._conjoined_yolo_path = None
                self._conjoined_yolo_paths = []
                self._conjoined_device = None
                return

            run_yolo = conj_backend in ("yolo", "both") and len(yolo_paths) > 0
            run_hf = conj_backend in ("hf", "both") and len(hf_ids) > 0
            cache_ok = (
                self._conjoined_device == device
                and self._conjoined_yolo_paths == yolo_paths
                and self._conjoined_model_ids == hf_ids
            )
            if cache_ok and (not run_yolo or len(self.models_conjoined) == len(yolo_paths)) and (not run_hf or len(self.pipe_conjoined_list) == len(hf_ids)):
                return

            if run_yolo:
                self._conjoined_yolo_paths = yolo_paths
                self.models_conjoined = []
                for path in yolo_paths:
                    path = path.strip()
                    if not path:
                        continue
                    resolved_path = self._resolve_model_path(path)
                    if not osp.isfile(resolved_path):
                        continue
                    try:
                        from ultralytics import YOLO as YOLOCls
                        m = YOLOCls(resolved_path)
                        if device == "cuda":
                            try:
                                m.to("cuda")
                            except Exception as oom:
                                if "out of memory" in str(oom).lower():
                                    self.logger.warning("GPU OOM loading conjoined YOLO %s; using CPU.", resolved_path)
                                    m.to("cpu")
                                else:
                                    raise
                        self.models_conjoined.append(m)
                    except Exception as e:
                        self.logger.warning("Conjoined YOLO load failed for %s: %s", resolved_path, e)
            else:
                self.models_conjoined = []
                self._conjoined_yolo_paths = []

            if run_hf:
                self._conjoined_model_ids = hf_ids
                self.pipe_conjoined_list = []
                for mid in hf_ids:
                    mid = self._normalize_model_id(mid) if mid else ""
                    if not mid:
                        continue
                    try:
                        pipe = pipeline(
                            "object-detection",
                            model=mid,
                            device=0 if device == "cuda" else -1,
                        )
                        self.pipe_conjoined_list.append(pipe)
                    except Exception as e:
                        self.logger.warning("Conjoined HF load failed for %s: %s", mid, e)
                self.pipe_conjoined = self.pipe_conjoined_list[0] if len(self.pipe_conjoined_list) == 1 else None
            else:
                self.pipe_conjoined_list = []
                self.pipe_conjoined = None
                self._conjoined_model_ids = []

            self._conjoined_device = device
            self.model_conjoined = self.models_conjoined[0] if len(self.models_conjoined) == 1 else None

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
            raw_allowed = [s.strip().lower() for s in labels_include.split(",") if s.strip()]
            # Normalize common typo: "bble" -> "bubble" (ogkalu model outputs bubble, text_bubble, text_free)
            allowed = None
            if raw_allowed:
                allowed = set()
                for lab in raw_allowed:
                    allowed.add("bubble" if lab == "bble" else lab)

            def run_on_image(pipe_to_use, det_img, offset_x=0, offset_y=0, score_thr_override=None, allowed_override=None):
                out = []
                try:
                    pil_img = PILImage.fromarray(det_img)
                    results = pipe_to_use(pil_img)
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
                    thr0 = score_thr
                    if score_thr_override is not None:
                        thr0 = float(score_thr_override)
                    thr = score_thr_text_free if label == "text_free" else thr0
                    if score < thr:
                        continue
                    allowed_set = allowed
                    if allowed_override is not None:
                        allowed_set = allowed_override
                    if allowed_set is not None and label not in allowed_set:
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
                    # Preserve detector label so downstream can treat outside-text differently (e.g. text_free).
                    blk.label = label
                    out.append(blk)
                return out

            if self.model_primary is not None:
                blk_list = self._run_yolo_primary(
                    self.model_primary, detect_img, scale, w, h,
                    score_thr, score_thr_text_free, allowed,
                )
            elif tile_size_val > 0 and h_det > 0 and w_det > 0:
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
                        blk_list.extend(run_on_image(self.pipe, crop, x1, y1))
            else:
                blk_list = run_on_image(self.pipe, detect_img, 0, 0)

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
            if merge_iou < 0.99:
                blk_list = _merge_overlapping_blocks(blk_list, merge_iou)

            # Optional: conjoined-bubble detector(s) that split merged primary boxes (multiple YOLO and/or HF).
            try:
                enable_conj = bool((self.params.get("enable_conjoined_secondary") or {}).get("value", False))
                has_conjoined = (self.models_conjoined and len(self.models_conjoined) > 0) or (self.pipe_conjoined_list and len(self.pipe_conjoined_list) > 0) or self.pipe_conjoined is not None or self.model_conjoined is not None
                if enable_conj and has_conjoined:
                    conj_thr = 0.4
                    cst = self.params.get("conjoined_score_threshold", {})
                    if isinstance(cst, dict):
                        try:
                            conj_thr = max(0.0, min(1.0, float(cst.get("value", 0.4))))
                        except (TypeError, ValueError):
                            pass
                    conj_labels = (self.params.get("conjoined_labels_include") or {}).get("value", "") or ""
                    conj_allowed_raw = [s.strip().lower() for s in conj_labels.split(",") if s.strip()]
                    conj_allowed = None
                    if conj_allowed_raw:
                        conj_allowed = set()
                        for lab in conj_allowed_raw:
                            conj_allowed.add("bubble" if lab == "bble" else lab)
                    min_inside = 2
                    mii = self.params.get("conjoined_min_boxes_in_primary", {})
                    if isinstance(mii, dict):
                        try:
                            min_inside = max(1, int(mii.get("value", 2) or 2))
                        except (TypeError, ValueError):
                            pass
                    min_area_px = 2500
                    mapx = self.params.get("conjoined_min_box_area", {})
                    if isinstance(mapx, dict):
                        try:
                            min_area_px = max(0, int(mapx.get("value", 2500) or 2500))
                        except (TypeError, ValueError):
                            pass
                    dedup_iou = 0.0
                    ddx = self.params.get("conjoined_dedup_iou", {})
                    if isinstance(ddx, dict):
                        try:
                            dedup_iou = max(0.0, min(0.99, float(ddx.get("value", 0) or 0)))
                        except (TypeError, ValueError):
                            pass

                    conjoined_blks: List[TextBlock] = []
                    for model in self.models_conjoined:
                        conjoined_blks.extend(
                            self._run_yolo_conjoined(
                                model, detect_img, scale, w, h, conj_thr, conj_allowed, min_area_px
                            )
                        )
                    for pipe in self.pipe_conjoined_list:
                        conjoined_blks.extend(
                            run_on_image(
                                pipe,
                                detect_img,
                                0,
                                0,
                                score_thr_override=conj_thr,
                                allowed_override=conj_allowed,
                            )
                        )
                    if dedup_iou > 0 and len(conjoined_blks) > 1:
                        conjoined_blks = _dedup_blocks_by_iou(conjoined_blks, dedup_iou)
                    if conjoined_blks:
                        conjoined_blks = [b for b in conjoined_blks if (b.xyxy[2] - b.xyxy[0]) * (b.xyxy[3] - b.xyxy[1]) >= min_area_px]

                    if conjoined_blks:
                        # Assign each conjoined box to the tightest containing primary box (min area).
                        prim = list(blk_list)
                        prim_areas = [max(1, (p.xyxy[2] - p.xyxy[0]) * (p.xyxy[3] - p.xyxy[1])) for p in prim]
                        groups = {}
                        orphans: List[TextBlock] = []
                        for cb in conjoined_blks:
                            best_i = None
                            best_area = None
                            for i, pb in enumerate(prim):
                                if _contains_xyxy(pb.xyxy, cb.xyxy):
                                    a = prim_areas[i]
                                    if best_area is None or a < best_area:
                                        best_area = a
                                        best_i = i
                            if best_i is not None:
                                groups.setdefault(best_i, []).append(cb)
                            else:
                                orphans.append(cb)

                        replace_set = {i for i, lst in groups.items() if len(lst) >= min_inside}
                        if replace_set:
                            kept = [p for i, p in enumerate(prim) if i not in replace_set]
                            added = []
                            for i in sorted(replace_set):
                                added.extend(groups.get(i, []))
                            # Deduplicate: drop added boxes that strongly overlap kept boxes.
                            final_added = []
                            for a in added:
                                ok = True
                                for k in kept:
                                    if _iou_xyxy(a.xyxy, k.xyxy) >= 0.85:
                                        ok = False
                                        break
                                if ok:
                                    final_added.append(a)
                            blk_list = kept + final_added

                        # Add conjoined boxes that did not fall inside any primary (e.g. SFX/text the primary missed).
                        if orphans:
                            existing = list(blk_list)
                            for o in orphans:
                                overlap = False
                                for e in existing:
                                    if _iou_xyxy(o.xyxy, e.xyxy) >= 0.5:
                                        overlap = True
                                        break
                                if not overlap:
                                    blk_list.append(o)
                                    existing.append(o)
            except Exception:
                # Never fail detection due to optional secondary pass
                pass
            inset_ratio = 0.0
            bi = self.params.get("box_inset_ratio", {})
            if isinstance(bi, dict):
                try:
                    inset_ratio = max(0.0, min(0.2, float(bi.get("value", 0))))
                except (TypeError, ValueError):
                    pass
            if inset_ratio > 0:
                blk_list = _inset_blocks(blk_list, inset_ratio, w, h)
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
            blk_list = sort_regions(blk_list)
            # Clip all blocks to page bounds so mask and textblock_list match page size (im_h, im_w).
            blk_list = _clip_blocks_to_page(blk_list, w, h)
            mask = np.zeros((h, w), dtype=np.uint8)
            for blk in blk_list:
                drawn = False
                if getattr(blk, 'lines', None) and len(blk.lines) > 0:
                    for line in blk.lines:
                        try:
                            pts = np.array(line, dtype=np.int32)
                            if pts.ndim == 2 and pts.shape[0] >= 3 and pts.shape[1] == 2:
                                pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
                                pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
                                cv2.fillPoly(mask, [pts], 255)
                                drawn = True
                        except (TypeError, ValueError, IndexError):
                            pass
                if not drawn:
                    x1, y1, x2, y2 = blk.xyxy
                    pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
                    cv2.fillPoly(mask, [pts], 255)
            # Mask shape (h, w) and block coordinates are in page space; inpainter expects same.
            return mask, blk_list

        def updateParam(self, param_key: str, param_content):
            super().updateParam(param_key, param_content)
            if param_key in ("model_id", "device", "enable_conjoined_secondary", "conjoined_model_id", "conjoined_model_ids", "conjoined_backend", "conjoined_yolo_path", "conjoined_yolo_paths"):
                self.pipe = None
                self._model_id = None
                self._device = None
                self.pipe_conjoined = None
                self.pipe_conjoined_list = []
                self.model_conjoined = None
                self.models_conjoined = []
                self._conjoined_model_id = None
                self._conjoined_model_ids = []
                self._conjoined_yolo_path = None
                self._conjoined_yolo_paths = []
                self._conjoined_device = None
