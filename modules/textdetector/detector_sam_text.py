"""
SAM 2.1 / SAM 3 text and speech-bubble detection using Hugging Face.
- SAM 3: text-prompt segmentation (e.g. "text", "speech bubble") via Sam3Model.
  When SAM 3 is not available, the detector is not registered.
Requires: pip install transformers torch (transformers >= 5.3 for SAM 3).
"""
import numpy as np
import cv2
from typing import Tuple, List

from .base import register_textdetectors, TextDetectorBase, TextBlock, DEVICE_SELECTOR
from .box_utils import expand_blocks
from utils.textblock import sort_regions

_SAM3_AVAILABLE = False
try:
    from transformers import Sam3Model, Sam3Processor
    import torch
    from PIL import Image as PILImage
    _SAM3_AVAILABLE = True
except ImportError:
    pass


def _masks_to_blocks(masks, scores, img_w: int, img_h: int, score_thr: float, min_area: int, min_side: int, max_area_ratio: float):
    """Convert binary masks + scores to list of TextBlock (xyxy, lines)."""
    blk_list: List[TextBlock] = []
    page_area = img_w * img_h
    if page_area <= 0:
        return blk_list
    for i, mask in enumerate(masks):
        if scores is not None and i < len(scores) and float(scores[i]) < score_thr:
            continue
        if hasattr(mask, "cpu"):
            mask = mask.cpu().numpy()
        mask_bin = (mask > 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            x2, y2 = x + bw, y + bh
            area = bw * bh
            if min_area > 0 and area < min_area:
                continue
            if min_side > 0 and (bw < min_side or bh < min_side):
                continue
            if max_area_ratio > 0 and area > max_area_ratio * page_area:
                continue
            x1 = max(0, min(x, img_w - 1))
            y1 = max(0, min(y, img_h - 1))
            x2 = max(x1 + 1, min(x2, img_w))
            y2 = max(y1 + 1, min(y2, img_h))
            if x2 <= x1 or y2 <= y1:
                continue
            pts = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            blk = TextBlock(xyxy=[x1, y1, x2, y2], lines=[pts])
            blk._detected_font_size = max(y2 - y1, 12)
            blk_list.append(blk)
    return blk_list


if _SAM3_AVAILABLE:

    @register_textdetectors("sam_text_det")
    class SAMTextDetector(TextDetectorBase):
        """
        SAM 3 text-prompt segmentation for text/speech-bubble detection.
        Uses text prompt (e.g. 'text', 'speech bubble') to segment regions.
        """
        params = {
            "model_id": {
                "type": "line_editor",
                "value": "facebook/sam3",
                "description": "Hugging Face model id (facebook/sam3). SAM 3 is gated — request access at https://huggingface.co/facebook/sam3 if you get 403.",
            },
            "text_prompt": {
                "type": "line_editor",
                "value": "text",
                "description": "Text prompt for SAM 3 (e.g. 'text', 'speech bubble', 'caption'). Ignored for SAM 2.1.",
            },
            "score_threshold": {
                "type": "line_editor",
                "value": 0.5,
                "description": "Min mask score (0.3–0.7).",
            },
            "mask_threshold": {
                "type": "line_editor",
                "value": 0.5,
                "description": "Binarize mask at this value (0.3–0.7).",
            },
            "box_padding": {
                "type": "line_editor",
                "value": 5,
                "description": "Pixels to add around each box (0–24).",
            },
            "min_area_px": {
                "type": "line_editor",
                "value": 200,
                "description": "Drop regions smaller than this area (0 = no minimum).",
            },
            "min_side_px": {
                "type": "line_editor",
                "value": 12,
                "description": "Drop regions with width or height smaller than this (0 = no minimum).",
            },
            "max_area_ratio": {
                "type": "line_editor",
                "value": 0.5,
                "description": "Drop regions larger than this fraction of image (0 = no limit).",
            },
            "device": DEVICE_SELECTOR(),
            "description": "SAM 3 (text prompt) for text/bubble detection. Install: pip install transformers torch (>=5.3)",
        }
        _load_model_keys = {"model", "processor"}

        def __init__(self, **params) -> None:
            super().__init__(**params)
            self.model = None
            self.processor = None
            self._model_id = None
            self._device = None

        def _load_model(self):
            model_id = (self.params.get("model_id") or {}).get("value", "facebook/sam3") or "facebook/sam3"
            model_id = (model_id or "").strip()
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
                self.processor = Sam3Processor.from_pretrained(model_id)
                self.model = Sam3Model.from_pretrained(model_id)
                if device == "cuda":
                    try:
                        self.model.to("cuda")
                    except Exception as e:
                        if "out of memory" in str(e).lower():
                            self.logger.warning("SAM GPU OOM; using CPU.")
                            self.model.to("cpu")
                            self._device = "cpu"
                        else:
                            raise
                self.logger.info("SAM 3 text detector loaded: %s", model_id)
            except Exception as e:
                err_str = str(e).lower()
                if "gated" in err_str or "403" in err_str or "authorized list" in err_str or "access" in err_str:
                    self.logger.warning(
                        "SAM model '%s' is gated. Request access at https://huggingface.co/%s or use another detector (e.g. surya_det, ctd).",
                        model_id, model_id
                    )
                else:
                    self.logger.warning("Failed to load SAM %s: %s", model_id, e)
                self.model = None
                self.processor = None

        def _detect(self, img: np.ndarray, proj=None) -> Tuple[np.ndarray, List[TextBlock]]:
            h, w = img.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            blk_list: List[TextBlock] = []

            if self.model is None or self.processor is None:
                return mask, blk_list

            score_thr = 0.5
            mask_thr = 0.5
            try:
                score_thr = max(0.0, min(1.0, float((self.params.get("score_threshold") or {}).get("value", 0.5))))
            except (TypeError, ValueError):
                pass
            try:
                mask_thr = max(0.0, min(1.0, float((self.params.get("mask_threshold") or {}).get("value", 0.5))))
            except (TypeError, ValueError):
                pass
            min_area = 200
            try:
                min_area = max(0, int((self.params.get("min_area_px") or {}).get("value", 200) or 200))
            except (TypeError, ValueError):
                pass
            min_side = 12
            try:
                min_side = max(0, int((self.params.get("min_side_px") or {}).get("value", 12) or 12))
            except (TypeError, ValueError):
                pass
            max_area_ratio = 0.5
            try:
                max_area_ratio = max(0.0, min(1.0, float((self.params.get("max_area_ratio") or {}).get("value", 0.5) or 0.5)))
            except (TypeError, ValueError):
                pass

            try:
                pil_img = PILImage.fromarray(img)
                text_prompt = (self.params.get("text_prompt") or {}).get("value", "text") or "text"
                text_prompt = (text_prompt or "text").strip()
                inputs = self.processor(images=pil_img, text=text_prompt, return_tensors="pt")
                if self._device == "cuda":
                    inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.model(**inputs)
                orig_sizes = inputs.get("original_sizes")
                if orig_sizes is not None and hasattr(orig_sizes, "tolist"):
                    target_sizes = orig_sizes.tolist()
                else:
                    target_sizes = [[h, w]]
                results = self.processor.post_process_instance_segmentation(
                    outputs,
                    threshold=score_thr,
                    mask_threshold=mask_thr,
                    target_sizes=target_sizes
                )
                if results and len(results) > 0:
                    r = results[0]
                    boxes_r = r.get("boxes")
                    scores_r = r.get("scores")
                    if boxes_r is not None and len(boxes_r) > 0:
                        for i, box in enumerate(boxes_r):
                            if scores_r is not None and i < len(scores_r) and float(scores_r[i]) < score_thr:
                                continue
                            if hasattr(box, "cpu"):
                                box = box.cpu().numpy()
                            x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                            x1 = max(0, min(int(round(x1)), w - 1))
                            x2 = max(0, min(int(round(x2)), w))
                            y1 = max(0, min(int(round(y1)), h - 1))
                            y2 = max(0, min(int(round(y2)), h))
                            if x2 <= x1 or y2 <= y1:
                                continue
                            area = (x2 - x1) * (y2 - y1)
                            if min_area > 0 and area < min_area:
                                continue
                            if min_side > 0 and ((x2 - x1) < min_side or (y2 - y1) < min_side):
                                continue
                            if max_area_ratio > 0 and area > max_area_ratio * (w * h):
                                continue
                            pts = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                            blk = TextBlock(xyxy=[x1, y1, x2, y2], lines=[pts])
                            blk._detected_font_size = max(y2 - y1, 12)
                            blk_list.append(blk)
                    elif r.get("masks") is not None and len(r["masks"]) > 0:
                        blk_list = _masks_to_blocks(
                            r["masks"], r.get("scores"), w, h, score_thr, min_area, min_side, max_area_ratio
                        )
            except Exception as e:
                self.logger.warning("SAM text detection failed: %s", e)
                return mask, blk_list

            pad_val = 0
            try:
                v = (self.params.get("box_padding") or {}).get("value", 5)
                pad_val = max(0, min(24, int(v) if v not in (None, "") else 0))
            except (TypeError, ValueError):
                pass
            if pad_val > 0:
                blk_list = expand_blocks(blk_list, pad_val, w, h)
            blk_list = sort_regions(blk_list)

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
