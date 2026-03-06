"""
MMOCR text detection – DBNet (and other MMOCR det models). Quality-focused scene/document detection.
Requires: pip install openmim && mim install mmengine mmcv mmdet && mim install mmocr
Or: pip install mmocr (may pull mmengine/mmcv/mmdet). Heavier deps; use when you need MMOCR quality.
"""
import numpy as np
import cv2
from typing import Tuple, List, Any

from .base import register_textdetectors, TextDetectorBase, TextBlock, ProjImgTrans
from .box_utils import expand_blocks
from ..base import DEVICE_SELECTOR

_MMOCR_AVAILABLE = False
try:
    from mmocr.apis import TextDetInferencer
    _MMOCR_AVAILABLE = True
except ImportError:
    import logging
    logging.getLogger("BallonTranslator").debug(
        "MMOCR not available for detector. Install: openmim && mim install mmengine mmcv mmdet mmocr"
    )


def _polys_scores_to_mask_blocks(
    h: int, w: int,
    det_polygons: List[Any],
    det_scores: List[float],
    min_score: float,
) -> Tuple[np.ndarray, List[TextBlock]]:
    """Convert MMOCR det_polygons/det_scores to mask and TextBlock list."""
    mask = np.zeros((h, w), dtype=np.uint8)
    blk_list: List[TextBlock] = []
    if not det_polygons:
        return mask, blk_list
    scores = det_scores if det_scores is not None else [1.0] * len(det_polygons)
    for idx, poly in enumerate(det_polygons):
        if poly is None:
            continue
        if isinstance(poly, (list, tuple)):
            pts = np.array(poly, dtype=np.int32)
        else:
            pts = np.asarray(poly, dtype=np.int32)
        if pts.ndim == 1:
            pts = pts.reshape(-1, 2)
        if pts.size < 6:
            continue
        if pts.shape[0] < 3:
            continue
        score = float(scores[idx]) if idx < len(scores) else 1.0
        if score < min_score:
            continue
        x1 = int(pts[:, 0].min())
        y1 = int(pts[:, 1].min())
        x2 = int(pts[:, 0].max())
        y2 = int(pts[:, 1].max())
        if x2 <= x1 or y2 <= y1:
            continue
        blk = TextBlock(xyxy=[x1, y1, x2, y2], lines=[pts.tolist()])
        blk.language = "unknown"
        blk._detected_font_size = max(y2 - y1, 12)
        blk_list.append(blk)
        cv2.fillPoly(mask, [pts.astype(np.int32)], 255)
    return mask, blk_list


if _MMOCR_AVAILABLE:

    @register_textdetectors("mmocr_det")
    class MMOCRDetector(TextDetectorBase):
        """
        Text detection via MMOCR (DBNet by default). Scene/document text; polygon output.
        Heavier dependencies (mmengine, mmcv, mmdet, mmocr). Prefer quality over speed.
        """
        params = {
            "det_model": {
                "type": "selector",
                "options": ["DBNet", "DBNetpp", "FCENet", "PSENet", "TextSnake"],
                "value": "DBNet",
                "description": "MMOCR detection model (DBNet/DBNetpp strong for document).",
            },
            "device": DEVICE_SELECTOR(),
            "det_score_thresh": {
                "type": "line_editor",
                "value": 0.3,
                "description": "Min detection score (0.2–0.6).",
            },
            "box_padding": {
                "type": "line_editor",
                "value": 0,
                "description": "Pixels to add around each detected box (all sides). Reduces clipped punctuation (?, !) and character edges. Recommended 4–6.",
            },
            "description": "MMOCR text detection (DBNet etc.). Install: mim install mmengine mmcv mmdet mmocr",
        }
        _load_model_keys = {"inferencer"}

        def __init__(self, **params) -> None:
            super().__init__(**params)
            self.inferencer = None
            self._det_model = None
            self._device = None

        def _load_model(self):
            det_model = (self.params.get("det_model") or {}).get("value", "DBNet") or "DBNet"
            device = (self.params.get("device") or {}).get("value", "cpu")
            if device in ("cuda", "gpu"):
                device = "cuda:0"
            if self.inferencer is not None and self._det_model == det_model and self._device == device:
                return
            self._det_model = det_model
            self._device = device
            try:
                self.inferencer = TextDetInferencer(model=det_model, device=device)
            except (ImportError, OSError) as e:
                err_msg = str(e)
                # On Windows / mismatched mmcv, this can throw low-level DLL errors that would crash the app.
                # Log a clear message and disable MMOCR instead of raising.
                if "DLL load failed" in err_msg or "_ext" in err_msg or "procedure could not be found" in err_msg:
                    self.logger.error(
                        "MMOCR detector failed to initialize due to incompatible mmcv/PyTorch build. "
                        "Use CTD, Surya, RapidOCR, or Paddle detection instead. See docs/OPTIONAL_DEPENDENCIES.md. "
                        "Raw error: %s",
                        err_msg,
                    )
                else:
                    self.logger.error("MMOCR detector initialization failed: %s", err_msg)
                self.inferencer = None
            except Exception as e:
                # Any other unexpected init error: log and disable MMOCR, but don't crash the app.
                self.logger.error("MMOCR detector unexpected initialization error: %s", e)
                self.inferencer = None

        def _detect(self, img: np.ndarray, proj: ProjImgTrans = None) -> Tuple[np.ndarray, List[TextBlock]]:
            if img.ndim == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            h, w = img.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            blk_list: List[TextBlock] = []
            try:
                # MMOCR/mmcv typically expects BGR
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.shape[2] == 3 else img
                result = self.inferencer(img_bgr, return_datasamples=False)
            except Exception as e:
                self.logger.error(f"MMOCR det failed: {e}")
                return mask, blk_list
            preds = result.get("predictions") or []
            if not preds:
                return mask, blk_list
            first = preds[0]
            polygons = first.get("det_polygons") or first.get("det_polygons_") or []
            scores = first.get("det_scores") or first.get("det_scores_") or []
            min_score = 0.3
            ps = self.params.get("det_score_thresh", {})
            if isinstance(ps, dict):
                try:
                    min_score = max(0.0, min(1.0, float(ps.get("value", 0.3))))
                except (TypeError, ValueError):
                    pass
            return _polys_scores_to_mask_blocks(h, w, polygons, scores, min_score)

        def updateParam(self, param_key: str, param_content):
            super().updateParam(param_key, param_content)
            if param_key in ("det_model", "device"):
                self.inferencer = None
                self._det_model = None
                self._device = None
