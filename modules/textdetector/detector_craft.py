"""
Standalone CRAFT text detector – Character-Region Awareness For Text detection (pip: craft-text-detector).
Curved and oriented text; good for scene text. Pair with any OCR.
Requires: pip install craft-text-detector torch
"""
import numpy as np
import cv2
from typing import Tuple, List

from .base import register_textdetectors, TextDetectorBase, TextBlock, DEVICE_SELECTOR
from utils.textblock import sort_regions, mit_merge_textlines

_CRAFT_AVAILABLE = False
try:
    from craft_text_detector import (
        read_image,
        load_craftnet_model,
        load_refinenet_model,
        get_prediction,
        empty_cuda_cache,
    )
    # craft-text-detector 0.4.3 requires opencv-python<4.5.4.62; main requirements use >=4.8
    try:
        parts = cv2.__version__.split(".")[:4]
        v_tuple = tuple(int(x) if x.isdigit() else 0 for x in parts)
        v_tuple = (v_tuple + (0,) * 4)[:4]
        if v_tuple >= (4, 5, 4, 62):
            import logging
            logging.getLogger("BallonTranslator").warning(
                "craft_det: craft-text-detector requires opencv-python<4.5.4.62; you have %s. "
                "Use easyocr_det or mmocr_det, or see docs/OPTIONAL_DEPENDENCIES.md.",
                cv2.__version__,
            )
        else:
            _CRAFT_AVAILABLE = True
    except Exception:
        _CRAFT_AVAILABLE = True
except ImportError as e:
    import logging
    logging.getLogger("BallonTranslator").debug(
        f"CRAFT detector not available: {e}. Install: pip install craft-text-detector torch. "
        "If opencv version conflicts, see docs/OPTIONAL_DEPENDENCIES.md."
    )


if _CRAFT_AVAILABLE:

    @register_textdetectors("craft_det")
    class CraftDetector(TextDetectorBase):
        """
        Standalone CRAFT text detection (curved/oriented scene text).
        Use when you want CRAFT without EasyOCR. Pair with any OCR.
        """
        params = {
            "text_threshold": {
                "type": "line_editor",
                "value": 0.5,
                "description": "Character region threshold (0.3–0.8). Lower = more regions.",
            },
            "link_threshold": {
                "type": "line_editor",
                "value": 0.4,
                "description": "Affinity threshold (0.2–0.5). Lower = more grouping.",
            },
            "low_text": {
                "type": "line_editor",
                "value": 0.4,
                "description": "Low text threshold.",
            },
            "long_size": {
                "type": "line_editor",
                "value": 1280,
                "description": "Resize long side for inference (640–2560).",
            },
            "merge_text_lines": {
                "type": "checkbox",
                "value": True,
                "description": "Merge nearby lines into one bubble.",
            },
            "mask_dilate_size": {
                "type": "line_editor",
                "value": 2,
                "description": "Mask dilation (0–5).",
            },
            "device": {**DEVICE_SELECTOR(), "description": "Device."},
            "description": "CRAFT (craft-text-detector). Curved/scene text.",
        }
        _load_model_keys = {"craft_net", "refine_net"}

        def __init__(self, **params) -> None:
            super().__init__(**params)
            self.craft_net = None
            self.refine_net = None
            self._cuda = None

        def _load_model(self):
            cuda = (self.params.get("device") or {}).get("value", "cpu") in ("cuda", "gpu")
            if self.craft_net is not None and self._cuda == cuda:
                return
            self._cuda = cuda
            self.craft_net = load_craftnet_model(cuda=cuda)
            self.refine_net = load_refinenet_model(cuda=cuda)

        def _detect(self, img: np.ndarray, proj=None) -> Tuple[np.ndarray, List[TextBlock]]:
            im_h, im_w = img.shape[:2]
            mask = np.zeros((im_h, im_w), dtype=np.uint8)
            blk_list: List[TextBlock] = []
            try:
                text_threshold = float(self.params.get("text_threshold", {}).get("value", 0.5))
                link_threshold = float(self.params.get("link_threshold", {}).get("value", 0.4))
                low_text = float(self.params.get("low_text", {}).get("value", 0.4))
                long_size = int(self.params.get("long_size", {}).get("value", 1280))
                prediction_result = get_prediction(
                    image=img,
                    craft_net=self.craft_net,
                    refine_net=self.refine_net,
                    text_threshold=text_threshold,
                    link_threshold=link_threshold,
                    low_text=low_text,
                    cuda=self._cuda,
                    long_size=long_size,
                )
            except Exception as e:
                self.logger.warning(f"CRAFT detection failed: {e}")
                return mask, blk_list

            boxes = prediction_result.get("boxes") or []
            if not boxes:
                return mask, blk_list

            pts_list = []
            for box in boxes:
                if box is None or len(box) < 4:
                    continue
                pts = np.array(box, dtype=np.float32)
                if pts.ndim == 1:
                    pts = pts.reshape(-1, 2)
                if pts.shape[0] < 4:
                    continue
                if pts.shape[0] > 4:
                    x1, y1 = pts[:, 0].min(), pts[:, 1].min()
                    x2, y2 = pts[:, 0].max(), pts[:, 1].max()
                    pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
                else:
                    pts = np.round(pts).astype(np.int32)
                pts_list.append(pts.tolist())

            if self.params.get("merge_text_lines", {}).get("value", True) and pts_list:
                blk_list = mit_merge_textlines(pts_list, width=im_w, height=im_h)
                for blk in blk_list:
                    for line_pts in blk.lines:
                        pts = np.array(line_pts, dtype=np.int32)
                        if pts.ndim == 1:
                            pts = pts.reshape(-1, 2)
                        cv2.fillPoly(mask, [pts], 255)
            else:
                for pts in pts_list:
                    pts_np = np.array(pts, dtype=np.int32)
                    if pts_np.ndim == 1:
                        pts_np = pts_np.reshape(-1, 2)
                    x1 = int(pts_np[:, 0].min())
                    y1 = int(pts_np[:, 1].min())
                    x2 = int(pts_np[:, 0].max())
                    y2 = int(pts_np[:, 1].max())
                    blk = TextBlock(xyxy=[x1, y1, x2, y2], lines=[pts_np.tolist()])
                    blk._detected_font_size = max(y2 - y1, 12)
                    blk_list.append(blk)
                    cv2.fillPoly(mask, [pts_np], 255)

            blk_list = sort_regions(blk_list)

            ksize = 0
            try:
                ksize = max(0, min(5, int(self.params.get("mask_dilate_size", {}).get("value", 2))))
            except (TypeError, ValueError):
                pass
            if ksize > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ksize + 1, 2 * ksize + 1))
                mask = cv2.dilate(mask, kernel)

            return mask, blk_list

        def updateParam(self, param_key: str, param_content):
            super().updateParam(param_key, param_content)
            if param_key == "device":
                self._cuda = (self.params.get("device") or {}).get("value", "cpu") in ("cuda", "gpu")
                if self.craft_net is not None:
                    self.craft_net = None
                    self.refine_net = None
