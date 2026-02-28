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
                "value": 0.5,
                "description": "Min detection score (0.2–0.9). Lower = more boxes.",
            },
            "labels_include": {
                "type": "line_editor",
                "value": "bubble,text_bubble,text_free",
                "description": "Comma-separated labels to keep (default: all comic classes). Empty = all.",
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
            mask = np.zeros((h, w), dtype=np.uint8)
            blk_list: List[TextBlock] = []
            try:
                pil_img = PILImage.fromarray(img)
                results = self.pipe(pil_img)
            except Exception as e:
                self.logger.warning(f"HF object-detection failed: {e}")
                return mask, blk_list
            if not results:
                return mask, blk_list
            score_thr = 0.5
            st = self.params.get("score_threshold", {})
            if isinstance(st, dict):
                try:
                    score_thr = max(0.0, min(1.0, float(st.get("value", 0.5))))
                except (TypeError, ValueError):
                    pass
            labels_include = (self.params.get("labels_include") or {}).get("value", "") or ""
            allowed = set(s.strip().lower() for s in labels_include.split(",") if s.strip()) if labels_include else None
            for item in results:
                if not isinstance(item, dict):
                    continue
                score = item.get("score", 0)
                if score < score_thr:
                    continue
                label = (item.get("label") or "").strip().lower()
                if allowed is not None and label not in allowed:
                    continue
                box = item.get("box")
                if not box:
                    continue
                xmin = int(box.get("xmin", 0))
                ymin = int(box.get("ymin", 0))
                xmax = int(box.get("xmax", 0))
                ymax = int(box.get("ymax", 0))
                x1 = max(0, min(xmin, xmax))
                x2 = min(w, max(xmin, xmax))
                y1 = max(0, min(ymin, ymax))
                y2 = min(h, max(ymin, ymax))
                if x2 <= x1 or y2 <= y1:
                    continue
                pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
                blk = TextBlock(xyxy=[x1, y1, x2, y2], lines=[pts.tolist()])
                blk._detected_font_size = max(y2 - y1, 12)
                blk_list.append(blk)
                cv2.fillPoly(mask, [pts], 255)
            blk_list = sort_regions(blk_list)
            return mask, blk_list

        def updateParam(self, param_key: str, param_content):
            super().updateParam(param_key, param_content)
            if param_key in ("model_id", "device"):
                self.pipe = None
                self._model_id = None
                self._device = None
