"""
RT-DETRv2 detection using Hugging Face RTDetrV2ForObjectDetection.
Default model: ogkalu/comic-text-and-bubble-detector — fine-tuned for comic/manga
text and speech bubbles (classes: bubble, text_bubble, text_free).
If you use a COCO model (e.g. PekingU/rtdetr_v2_*), no text regions are returned.
Requires: pip install transformers torch
"""
import copy
import os.path as osp
import numpy as np
import cv2
from typing import Tuple, List, Optional

from .base import register_textdetectors, TextDetectorBase, TextBlock, DEVICE_SELECTOR
from .box_utils import expand_blocks
from utils.textblock import sort_regions

# Default comic text & bubble model; use this when model_id is empty so no model id is required.
DEFAULT_RTDETR_COMIC_MODEL = "ogkalu/comic-text-and-bubble-detector"

_RTDETR_AVAILABLE = False
try:
    from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor
    import torch
    from PIL import Image as PILImage
    _RTDETR_AVAILABLE = True
except ImportError:
    import logging
    logging.getLogger("BallonTranslator").debug(
        "RT-DETRv2 detector not available. Install: pip install transformers torch"
    )


def _clip_blocks_to_page(blk_list: List[TextBlock], img_w: int, img_h: int) -> List[TextBlock]:
    """Clip every block's xyxy and lines to page bounds."""
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
        new_blk.lines = [[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]]
        out.append(new_blk)
    return out


if _RTDETR_AVAILABLE:

    @register_textdetectors("rtdetr_comic")
    class RTDetrV2Detector(TextDetectorBase):
        """
        RT-DETRv2 object detection (Hugging Face). Default: comic text/bubble model.
        """
        params = {
            "model_id": {
                "type": "line_editor",
                "value": "",
                "description": "Hugging Face model id. Leave empty for comic text & bubble detector (default). PekingU/rtdetr_v2_* are COCO (not text).",
            },
            "score_threshold": {
                "type": "line_editor",
                "value": 0.4,
                "description": "Min detection score (0.2–0.6). Lower = more regions.",
            },
            "class_ids": {
                "type": "line_editor",
                "value": "",
                "description": "Optional: comma-separated class IDs to keep. Comic model: 0=bubble, 1=text_bubble, 2=text_free. Empty = all.",
            },
            "box_padding": {
                "type": "line_editor",
                "value": 5,
                "description": "Pixels to add around each box (0–24).",
            },
            "detect_min_side": {
                "type": "line_editor",
                "value": 640,
                "description": "Resize so longer side >= this before detection (RT-DETRv2 prefers 640). 0 = no upscale.",
            },
            "detect_max_side": {
                "type": "line_editor",
                "value": 1280,
                "description": "Resize so longer side <= this (0 = no limit).",
            },
            "max_area_ratio": {
                "type": "line_editor",
                "value": 0.4,
                "description": "Drop boxes covering more than this fraction of the image (0.2–0.6). Avoids huge COCO detections (e.g. table, bed).",
            },
            "min_area_px": {
                "type": "line_editor",
                "value": 400,
                "description": "Drop boxes smaller than this area in pixels (0 = no minimum). Reduces tiny random COCO detections.",
            },
            "min_side_px": {
                "type": "line_editor",
                "value": 14,
                "description": "Drop boxes whose width or height is smaller than this (0 = no minimum). Filters thin slivers and dots.",
            },
            "device": DEVICE_SELECTOR(),
            "description": "RT-DETRv2 (Hugging Face). Default: comic text & bubble detector. Install: pip install transformers torch",
        }
        _load_model_keys = {"model", "processor"}

        def __init__(self, **params) -> None:
            super().__init__(**params)
            self.model = None
            self.processor = None
            self._model_id = None
            self._device = None

        @staticmethod
        def _repo_root() -> str:
            return osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)), "..", ".."))

        def _load_model(self):
            raw = (self.params.get("model_id") or {}).get("value") or ""
            model_id = (raw if isinstance(raw, str) else "").strip() or DEFAULT_RTDETR_COMIC_MODEL
            device = (self.params.get("device") or {}).get("value", "cpu")
            if device == "gpu":
                device = "cuda"
            if device == "cuda" and not torch.cuda.is_available():
                device = "cpu"

            if self.model is not None and self._model_id == model_id and self._device == device:
                return

            self._model_id = model_id
            self._device = device
            try:
                self.processor = RTDetrImageProcessor.from_pretrained(model_id)
                self.model = RTDetrV2ForObjectDetection.from_pretrained(model_id)
                if device == "cuda":
                    try:
                        self.model.to("cuda")
                    except Exception as e:
                        if "out of memory" in str(e).lower():
                            self.logger.warning("RT-DETRv2 GPU OOM; using CPU.")
                            self.model.to("cpu")
                            self._device = "cpu"
                        else:
                            raise
                self.logger.info("RT-DETRv2 detector loaded: %s", model_id)
            except Exception as e:
                self.logger.warning("Failed to load RT-DETRv2 %s: %s", model_id, e)
                self.model = None
                self.processor = None

        def _parse_class_ids(self) -> Optional[List[int]]:
            raw = (self.params.get("class_ids") or {}).get("value", "") or ""
            if not raw or not isinstance(raw, str):
                return None
            out = []
            for s in raw.split(","):
                s = s.strip()
                if not s:
                    continue
                try:
                    out.append(int(s))
                except ValueError:
                    pass
            return out if out else None

        def _detect(self, img: np.ndarray, proj=None) -> Tuple[np.ndarray, List[TextBlock]]:
            if img.ndim == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            h, w = img.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            blk_list: List[TextBlock] = []

            if self.model is None or self.processor is None:
                return mask, blk_list

            raw = (self.params.get("model_id") or {}).get("value") or ""
            model_id = (raw if isinstance(raw, str) else "").strip() or DEFAULT_RTDETR_COMIC_MODEL
            class_ids = self._parse_class_ids()
            # Default PekingU models are COCO-trained (person, tree, car, etc.) — not text. Return no boxes and warn.
            if "PekingU/rtdetr_v2" in model_id and class_ids is None:
                if not getattr(self, "_coco_warned", False):
                    self.logger.warning(
                        "RT-DETRv2 model '%s' is trained on COCO (objects: person, tree, car, etc.), not text. "
                        "No text regions returned. Use a text detector (e.g. CTD, easyocr_det, hf_object_det) "
                        "or a fine-tuned RT-DETRv2 model for text/bubbles.", model_id
                    )
                    self._coco_warned = True
                return mask, blk_list
            scale = 1.0
            detect_img = img
            min_side_val = 0
            max_side_val = 0
            try:
                min_side_val = int((self.params.get("detect_min_side") or {}).get("value", 640) or 640)
            except (TypeError, ValueError):
                pass
            try:
                max_side_val = int((self.params.get("detect_max_side") or {}).get("value", 1280) or 1280)
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

            score_thr = 0.4
            try:
                score_thr = max(0.0, min(1.0, float((self.params.get("score_threshold") or {}).get("value", 0.4))))
            except (TypeError, ValueError):
                pass
            try:
                pil_img = PILImage.fromarray(detect_img)
                inputs = self.processor(images=pil_img, return_tensors="pt")
                if self._device == "cuda":
                    inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.model(**inputs)
                target_h, target_w = detect_img.shape[0], detect_img.shape[1]
                results = self.processor.post_process_object_detection(
                    outputs,
                    target_sizes=torch.tensor([[target_h, target_w]]),
                    threshold=float(score_thr),
                )
            except Exception as e:
                self.logger.warning("RT-DETRv2 inference failed: %s", e)
                return mask, blk_list

            if not results or len(results) == 0:
                return mask, blk_list

            result = results[0]
            boxes = result.get("boxes")
            scores = result.get("scores")
            labels = result.get("labels") or result.get("label")
            if boxes is None or len(boxes) == 0:
                return mask, blk_list

            for i in range(len(boxes)):
                box = boxes[i]
                score = float(scores[i]) if scores is not None and i < len(scores) else 1.0
                label_id = int(labels[i]) if labels is not None and i < len(labels) else 0
                if class_ids is not None and label_id not in class_ids:
                    continue
                if score < score_thr:
                    continue
                if hasattr(box, "cpu"):
                    box = box.cpu().numpy()
                x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                if scale != 1.0 and scale > 0:
                    x1, y1, x2, y2 = x1 / scale, y1 / scale, x2 / scale, y2 / scale
                x1 = max(0, min(int(round(x1)), w))
                x2 = max(0, min(int(round(x2)), w))
                y1 = max(0, min(int(round(y1)), h))
                y2 = max(0, min(int(round(y2)), h))
                if x2 <= x1 or y2 <= y1:
                    continue
                area = (x2 - x1) * (y2 - y1)
                page_area = w * h
                if page_area <= 0:
                    continue
                max_ratio = 0.4
                try:
                    max_ratio = max(0.1, min(0.8, float((self.params.get("max_area_ratio") or {}).get("value", 0.4))))
                except (TypeError, ValueError):
                    pass
                if area > max_ratio * page_area:
                    continue
                min_area_val = 400
                try:
                    min_area_val = max(0, int((self.params.get("min_area_px") or {}).get("value", 400) or 400))
                except (TypeError, ValueError):
                    pass
                if min_area_val > 0 and area < min_area_val:
                    continue
                min_side_val = 14
                try:
                    min_side_val = max(0, int((self.params.get("min_side_px") or {}).get("value", 14) or 14))
                except (TypeError, ValueError):
                    pass
                if min_side_val > 0:
                    bw, bh = x2 - x1, y2 - y1
                    if bw < min_side_val or bh < min_side_val:
                        continue
                pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
                blk = TextBlock(xyxy=[x1, y1, x2, y2], lines=[pts.tolist()])
                blk._detected_font_size = max(y2 - y1, 12)
                blk_list.append(blk)
                cv2.fillPoly(mask, [pts], 255)

            pad_val = 0
            try:
                v = (self.params.get("box_padding") or {}).get("value", 5)
                pad_val = max(0, min(24, int(v) if v not in (None, "") else 0))
            except (TypeError, ValueError):
                pass
            if pad_val > 0:
                blk_list = expand_blocks(blk_list, pad_val, w, h)
            blk_list = sort_regions(blk_list)
            blk_list = _clip_blocks_to_page(blk_list, w, h)

            mask = np.zeros((h, w), dtype=np.uint8)
            for blk in blk_list:
                x1, y1, x2, y2 = blk.xyxy
                pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
                cv2.fillPoly(mask, [pts], 255)

            return mask, blk_list

        def updateParam(self, param_key: str, param_content):
            super().updateParam(param_key, param_content)
            if param_key in ("model_id", "device"):
                self.model = None
                self.processor = None
                self._model_id = None
                self._device = None
