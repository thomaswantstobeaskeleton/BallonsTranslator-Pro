"""
Nemotron OCR v1 – Full-page OCR with detection + recognition (NVIDIA).
Uses the nemotron-ocr PyPI package. Runs on full image; assigns text to blocks by bbox IoU.
Requires: pip install nemotron-ocr (Python 3.12+). Models download from HF on first run.
"""
import sys
import tempfile
import os
from typing import List, Tuple
import numpy as np
import cv2

from .base import OCRBase, register_OCR, DEFAULT_DEVICE, DEVICE_SELECTOR, TextBlock


def _iou_rect(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@register_OCR("nemotron_ocr_v1")
class NemotronOCRv1(OCRBase):
    """
    Nemotron OCR v1: full-page detection + recognition. Runs once per image;
    assigns text to blocks by bbox overlap. Requires nemotron-ocr (Python 3.12+).
    """
    params = {
        "merge_level": {
            "type": "selector",
            "options": ["word", "sentence", "paragraph"],
            "value": "paragraph",
            "description": "Merge level for detected text (word / sentence / paragraph).",
        },
        "iou_threshold": {
            "type": "line_editor",
            "value": "0.2",
            "description": "Min IoU to assign an OCR bbox to a block (0.1–0.5).",
        },
        "description": "Nemotron OCR v1 full-page OCR. Install: pip install nemotron-ocr (Python 3.12+).",
    }
    _load_model_keys = {"_ocr"}
    optional_install_hint = "pip install nemotron-ocr (requires Python 3.12+)"

    @classmethod
    def is_environment_compatible(cls) -> bool:
        """Only show in non-dev dropdown when Python is 3.12+ (nemotron-ocr requirement)."""
        return sys.version_info >= (3, 12)

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self._ocr = None

    def _load_model(self):
        if self._ocr is not None:
            return
        if sys.version_info < (3, 12):
            raise RuntimeError(
                "Nemotron OCR v1 requires Python 3.12 or newer (current: %s.%s). "
                "Install a Python 3.12+ environment and run: pip install nemotron-ocr"
                % (sys.version_info.major, sys.version_info.minor)
            )
        try:
            from nemotron_ocr.inference.pipeline import NemotronOCR
            self._ocr = NemotronOCR()
        except ImportError as e:
            raise RuntimeError(
                "Nemotron OCR v1 requires the nemotron-ocr package (Python 3.12+). "
                "Install with: pip install nemotron-ocr"
            ) from e

    def _run_page(self, img: np.ndarray) -> Tuple[List[Tuple[float, float, float, float]], List[str]]:
        """Run OCR on full image. Returns (bboxes in pixel coords, texts)."""
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        merge_level = "paragraph"
        try:
            ml = (self.params.get("merge_level") or {}).get("value", "paragraph")
            if ml in ("word", "sentence", "paragraph"):
                merge_level = ml
        except Exception:
            pass
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            try:
                cv2.imwrite(f.name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                predictions = self._ocr(f.name, merge_level=merge_level)
            finally:
                try:
                    os.unlink(f.name)
                except OSError:
                    pass
        bboxes = []
        texts = []
        for pred in predictions:
            text = (pred.get("text") or "").strip()
            left = pred.get("left", 0)
            upper = pred.get("upper", 0)
            right = pred.get("right", 0)
            lower = pred.get("lower", 0)
            # Normalized 0–1 → pixel coords
            x1 = left * w
            y1 = upper * h
            x2 = right * w
            y2 = lower * h
            bboxes.append((x1, y1, x2, y2))
            texts.append(text)
        return bboxes, texts

    def ocr_img(self, img: np.ndarray) -> str:
        if not self.all_model_loaded():
            self.load_model()
        _, texts = self._run_page(img)
        return "\n".join(texts) if texts else ""

    def _ocr_blk_list(self, img: np.ndarray, blk_list: List[TextBlock], *args, **kwargs) -> None:
        if not self.all_model_loaded():
            self.load_model()
        bboxes, texts = self._run_page(img)
        iou_thr = 0.2
        try:
            iou_thr = max(0.05, min(0.9, float((self.params.get("iou_threshold") or {}).get("value", "0.2"))))
        except (TypeError, ValueError):
            pass
        for blk in blk_list:
            x1, y1, x2, y2 = blk.xyxy
            blk_rect = (float(x1), float(y1), float(x2), float(y2))
            best_iou = 0.0
            best_text = ""
            for (bx1, by1, bx2, by2), txt in zip(bboxes, texts):
                iou = _iou_rect(blk_rect, (bx1, by1, bx2, by2))
                if iou >= iou_thr and iou > best_iou:
                    best_iou = iou
                    best_text = (txt or "").strip()
            blk.text = [best_text if best_text else ""]
