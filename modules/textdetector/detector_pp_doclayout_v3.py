"""
PP-DocLayoutV3 detector — PaddlePaddle document-layout text detector.
Identifies text regions (title, text, paragraph, etc.) in complex document/comic layouts.
Requires:  pip install transformers pillow  (or paddleocr / paddlepaddle)
Model:    https://huggingface.co/PaddlePaddle/PP-DocLayoutV3
"""
import os
from typing import Tuple, List

import numpy as np
import cv2

from .base import register_textdetectors, TextDetectorBase, TextBlock, DEVICE_SELECTOR
from .box_utils import expand_blocks
from utils.textblock import sort_regions, examine_textblk, sort_pnts
from utils.imgproc_utils import xywh2xyxypoly


_PP_DOCLAYOUT_AVAILABLE = False
_Backend = None  # 'transformers' or 'paddle'

# Try transformers first (easier to install, no Paddle C++ libs)
try:
    from transformers import AutoModelForVisualDocumentUnderstanding, AutoProcessor
    _PP_DOCLAYOUT_AVAILABLE = True
    _Backend = "transformers"
except Exception:
    pass

# Fallback to PaddleOCR LayoutDetection
try:
    if not _PP_DOCLAYOUT_AVAILABLE:
        from paddleocr import LayoutDetection
        _PP_DOCLAYOUT_AVAILABLE = True
        _Backend = "paddle"
except Exception:
    pass


HF_REPO_ID = "PaddlePaddle/PP-DocLayoutV3"

# Classes considered "text" regions we want to keep.
# V3 label names vary by backend; we include common synonyms.
_TEXT_LABELS = {
    "text", "paragraph", "title", "header", "footer", "caption",
    "list", "table", "equation", "chart",
    # Chinese labels (Paddle often emits Chinese class names)
    "文字", "文本", "段落", "标题", "页眉", "页脚", "题注", "列表",
}


def _is_text_label(label: str) -> bool:
    if not label:
        return False
    lo = label.strip().lower()
    # Direct match
    if lo in _TEXT_LABELS:
        return True
    # Partial match (e.g. "text_block")
    for tl in _TEXT_LABELS:
        if tl in lo or lo in tl:
            return True
    return False


@register_textdetectors("pp_doclayout_v3")
class PPDocLayoutV3Detector(TextDetectorBase):
    """
    PP-DocLayoutV3 — document layout detector that finds text regions in complex layouts
    (curved pages, mixed columns, manga with complex paneling, etc.).

    Uses transformers (preferred) or paddleocr as backend.
    Auto-downloads model weights from Hugging Face on first use.
    """
    params = {
        "confidence threshold": {
            "display_name": "Confidence threshold",
            "type": "line_editor",
            "value": 0.3,
        },
        "device": {**DEVICE_SELECTOR(), "display_name": "Device"},
        "font size multiplier": {
            "display_name": "Font size multiplier",
            "type": "line_editor",
            "value": 1.0,
        },
        "font size max": {
            "display_name": "Max font size",
            "type": "line_editor",
            "value": -1,
        },
        "font size min": {
            "display_name": "Min font size",
            "type": "line_editor",
            "value": -1,
        },
        "box_padding": {
            "type": "line_editor",
            "value": 4,
            "display_name": "Box padding",
            "description": "Pixels to add around each detected box.",
        },
        "description": "PP-DocLayoutV3 (PaddlePaddle). Document layout-aware text detection for complex page layouts and curved surfaces. Auto-downloads from HF.",
    }

    _load_model_keys = {"model", "processor"}

    def __init__(self, **params) -> None:
        super().__init__(**params)

    def _load_model(self):
        if not _PP_DOCLAYOUT_AVAILABLE:
            raise RuntimeError(
                "PP-DocLayoutV3 requires transformers (pip install transformers pillow) "
                "or paddleocr (pip install paddleocr)."
            )
        if _Backend == "transformers":
            self.processor = AutoProcessor.from_pretrained(HF_REPO_ID, trust_remote_code=True)
            self.model = AutoModelForVisualDocumentUnderstanding.from_pretrained(
                HF_REPO_ID, trust_remote_code=True
            )
            device = self.get_param_value("device")
            if device and device != "cpu":
                self.model = self.model.to(device)
        elif _Backend == "paddle":
            self.model = LayoutDetection(model_name="PP-DocLayout-L")
            self.processor = None

    def _detect(self, img: np.ndarray, proj=None) -> Tuple[np.ndarray, List[TextBlock]]:
        if not _PP_DOCLAYOUT_AVAILABLE:
            raise RuntimeError("PP-DocLayoutV3 backend not installed.")

        im_h, im_w = img.shape[:2]
        mask = np.zeros((im_h, im_w), dtype=np.uint8)
        detected_items = []
        conf_thresh = float(self.get_param_value("confidence threshold"))

        if _Backend == "transformers":
            # transformers pipeline
            from PIL import Image
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            inputs = self.processor(images=pil_img, return_tensors="pt")
            device = self.get_param_value("device")
            if device and device != "cpu":
                inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

            with np.errstate(all="ignore"):
                outputs = self.model(**inputs)

            # The output structure varies; try common patterns
            pred_boxes = None
            pred_labels = None
            pred_scores = None

            if hasattr(outputs, "pred_boxes") and outputs.pred_boxes is not None:
                pred_boxes = outputs.pred_boxes.cpu().numpy() if hasattr(outputs.pred_boxes, "cpu") else outputs.pred_boxes
            if hasattr(outputs, "pred_logits") and outputs.pred_logits is not None:
                logits = outputs.pred_logits
                if hasattr(logits, "cpu"):
                    logits = logits.cpu().numpy()
                pred_scores = np.max(logits, axis=-1)
                pred_labels = np.argmax(logits, axis=-1)

            # Some models return decoded dicts directly
            if pred_boxes is None and isinstance(outputs, dict):
                pred_boxes = outputs.get("boxes") or outputs.get("pred_boxes")
                pred_labels = outputs.get("labels") or outputs.get("pred_labels")
                pred_scores = outputs.get("scores") or outputs.get("pred_scores")

            if pred_boxes is not None and pred_labels is not None:
                id2label = getattr(self.model.config, "id2label", {}) if hasattr(self.model, "config") else {}
                for i in range(len(pred_boxes)):
                    box = pred_boxes[i]
                    score = float(pred_scores[i]) if pred_scores is not None and i < len(pred_scores) else 1.0
                    if score < conf_thresh:
                        continue
                    label_id = int(pred_labels[i]) if pred_labels is not None else -1
                    label_str = id2label.get(label_id, str(label_id)) if isinstance(id2label, dict) else str(label_id)
                    if not _is_text_label(label_str):
                        continue

                    # Box may be normalized [0,1] or pixel coords
                    if np.max(box) <= 1.0:
                        x1, y1, x2, y2 = (
                            int(box[0] * im_w),
                            int(box[1] * im_h),
                            int(box[2] * im_w),
                            int(box[3] * im_h),
                        )
                    else:
                        x1, y1, x2, y2 = map(int, box[:4])

                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(im_w, x2), min(im_h, y2)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                    pts = xywh2xyxypoly(np.array([[x1, y1, x2 - x1, y2 - y1]])).reshape(4, 2).tolist()
                    detected_items.append({"pts": pts, "label": "text"})

        elif _Backend == "paddle":
            # paddleocr LayoutDetection returns a list of dicts
            result = self.model.predict(img)
            if isinstance(result, list):
                for item in result:
                    if not isinstance(item, dict):
                        continue
                    score = float(item.get("score", 0))
                    if score < conf_thresh:
                        continue
                    label = item.get("label", "")
                    if not _is_text_label(label):
                        continue
                    bbox = item.get("bbox")
                    if bbox is None:
                        continue
                    x1, y1, x2, y2 = map(int, bbox[:4])
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(im_w, x2), min(im_h, y2)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                    pts = xywh2xyxypoly(np.array([[x1, y1, x2 - x1, y2 - y1]])).reshape(4, 2).tolist()
                    detected_items.append({"pts": pts, "label": "text"})

        blk_list = []
        for item in detected_items:
            pts_sorted, is_vertical = sort_pnts(item["pts"])
            blk = TextBlock(lines=[pts_sorted], src_is_vertical=is_vertical, label=item["label"])
            blk.vertical = is_vertical
            blk.adjust_bbox()
            examine_textblk(blk, im_w, im_h)
            blk_list.append(blk)

        blk_list = sort_regions(blk_list)

        fnt_rsz = self.get_param_value("font size multiplier")
        fnt_max = self.get_param_value("font size max")
        fnt_min = self.get_param_value("font size min")
        for blk in blk_list:
            sz = blk._detected_font_size * fnt_rsz
            if fnt_max > 0:
                sz = min(fnt_max, sz)
            if fnt_min > 0:
                sz = max(fnt_min, sz)
            blk.font_size = sz
            blk._detected_font_size = sz

        pad_val = 0
        bp = self.params.get("box_padding", {})
        if isinstance(bp, dict):
            v = bp.get("value", 4)
            try:
                pad_val = max(0, min(24, int(v) if v not in (None, "") else 4))
            except (TypeError, ValueError):
                pad_val = 4
        if pad_val > 0:
            blk_list = expand_blocks(blk_list, pad_val, im_w, im_h)

        return mask, blk_list
