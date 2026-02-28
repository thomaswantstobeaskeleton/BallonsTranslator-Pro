"""
Manga Whisperer (Magi) – CVPR 2024. Unified manga detection: panels, text boxes, characters, reading order.
Uses ragavsachdeva/magi (v1). Quality-focused for manga/comics. Requires: pip install transformers torch einops
"""
import numpy as np
import cv2
from typing import Tuple, List

from .base import register_textdetectors, TextDetectorBase, TextBlock, ProjImgTrans
from ..base import DEVICE_SELECTOR

_MAGI_AVAILABLE = False
try:
    from transformers import AutoModel
    import torch
    _MAGI_AVAILABLE = True
except ImportError:
    import logging
    logging.getLogger("BallonTranslator").debug(
        "Magi (Manga Whisperer) not available. Install: pip install transformers torch einops"
    )


if _MAGI_AVAILABLE:

    @register_textdetectors("magi_det")
    class MagiDetector(TextDetectorBase):
        """
        Manga Whisperer (Magi): panels, text boxes, characters, reading order (CVPR 2024).
        This detector returns only text boxes; use any OCR after. Heavyweight; quality for manga.
        """
        params = {
            "model_name": {
                "type": "line_editor",
                "value": "ragavsachdeva/magi",
                "description": "Hugging Face model (magi v1). magiv2/magiv3 have different APIs.",
            },
            "device": DEVICE_SELECTOR(),
            "text_detection_threshold": {
                "type": "line_editor",
                "value": 0.25,
                "description": "Min confidence for text boxes (0.2–0.5).",
            },
            "description": "Manga Whisperer (Magi) – manga text/panel detection. Install: pip install transformers einops",
        }
        _load_model_keys = {"model"}

        def __init__(self, **params) -> None:
            super().__init__(**params)
            self.model = None
            self._model_name = None
            self._device = None

        def _load_model(self):
            model_name = (self.params.get("model_name") or {}).get("value", "ragavsachdeva/magi") or "ragavsachdeva/magi"
            device = (self.params.get("device") or {}).get("value", "cuda")
            if device in ("cuda", "gpu"):
                device = "cuda" if torch.cuda.is_available() else "cpu"
            if self.model is not None and self._model_name == model_name and self._device == device:
                return
            self._model_name = model_name
            self._device = device
            try:
                self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            except ValueError as e:
                if "AutoBackbone" in str(e) or "Unrecognized configuration class" in str(e):
                    raise RuntimeError(
                        "Magi (Manga Whisperer) is incompatible with transformers 5.x: the model's backbone config "
                        "is not in the current AutoBackbone registry. Use CTD or Surya detection instead."
                    ) from e
                raise
            self.model = self.model.to(device)
            self.model.eval()

        def _detect(self, img: np.ndarray, proj: ProjImgTrans = None) -> Tuple[np.ndarray, List[TextBlock]]:
            if img.ndim == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            h, w = img.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            blk_list: List[TextBlock] = []
            # Magi expects RGB numpy (from PIL convert L then RGB in their demo)
            if img.shape[2] == 3:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = img
            try:
                thresh = 0.25
                tdt = self.params.get("text_detection_threshold", {})
                if isinstance(tdt, dict):
                    try:
                        thresh = max(0.1, min(0.8, float(tdt.get("value", 0.25))))
                    except (TypeError, ValueError):
                        pass
                with torch.no_grad():
                    results = self.model.predict_detections_and_associations(
                        [img_rgb],
                        text_detection_threshold=thresh,
                    )
                if not results or len(results) == 0:
                    return mask, blk_list
                texts = results[0].get("texts", [])
                if not texts:
                    return mask, blk_list
                for bbox in texts:
                    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                        x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                    elif hasattr(bbox, "tolist"):
                        pt = bbox.tolist()
                        x1, y1, x2, y2 = pt[0], pt[1], pt[2], pt[3]
                    else:
                        continue
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
                    blk = TextBlock(xyxy=[x1, y1, x2, y2], lines=[pts.tolist()])
                    blk.language = "unknown"
                    blk._detected_font_size = max(y2 - y1, 12)
                    blk_list.append(blk)
                    cv2.fillPoly(mask, [pts], 255)
            except Exception as e:
                self.logger.error(f"Magi detection failed: {e}")
            return mask, blk_list

        def updateParam(self, param_key: str, param_content):
            super().updateParam(param_key, param_content)
            if param_key in ("model_name", "device"):
                self.model = None
                self._model_name = None
                self._device = None
