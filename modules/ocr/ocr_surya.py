"""
Surya OCR – multilingual recognition (90+ languages).
Uses pre-cropped regions from the text detector; good for Chinese, English, and mixed script.
Requires: pip install surya-ocr  (Python 3.10+, PyTorch).
Supports current surya-ocr API (RecognitionPredictor + FoundationPredictor).
"""
from typing import List
import re
import numpy as np
import cv2
from .base import OCRBase, register_OCR, DEFAULT_DEVICE, DEVICE_SELECTOR, TextBlock
from utils.io_utils import trim_ocr_repetition

# Unicode replacement character and common OCR garbage that renders as empty squares in many fonts
_REPLACEMENT_CHAR = "\uFFFD"
_VISIBLE_PLACEHOLDER = "\u25A1"  # □ (white square) - visible in CJK fonts


def _normalize_ocr_text(text: str, chinese_only: bool = False) -> str:
    """Replace replacement chars so they don't render as empty; optionally strip stray katakana when Chinese-only."""
    if not text:
        return text
    if _REPLACEMENT_CHAR in text:
        text = text.replace(_REPLACEMENT_CHAR, _VISIBLE_PLACEHOLDER)
    if chinese_only:
        # Strip stray katakana that OCR sometimes outputs for garbled/cut-off Chinese (e.g. "山クト" for "3")
        text = re.sub(r"[\u30A0-\u30FF]+", "", text)  # katakana
        text = re.sub(r"\s+", " ", text).strip()
    return text

_SURYA_AVAILABLE = False
_surya_predictor = None
_surya_task_name = None

try:
    from surya.recognition import RecognitionPredictor
    from surya.foundation import FoundationPredictor
    from surya.common.surya.schema import TaskNames
    from PIL import Image
    _SURYA_AVAILABLE = True
    _surya_predictor = RecognitionPredictor
    _surya_task_name = TaskNames.ocr_with_boxes
except ImportError as e:
    import logging
    logging.getLogger("BallonTranslator").debug(
        f"Surya OCR not available: {e}. Install with: pip install surya-ocr (Python 3.10+, PyTorch)."
    )


def _cv2_to_pil_rgb(img: np.ndarray) -> "Image.Image":
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)


if _SURYA_AVAILABLE:

    @register_OCR("surya_ocr")
    class SuryaOCR(OCRBase):
        """
        Surya OCR: 90+ languages, line-level recognition.
        Best for: Chinese, English, multilingual manhua/comics when you want an alternative to mit48px or PaddleOCR.
        """
        lang_map = {
            "Chinese (Simplified)": ["zh"],
            "Chinese (Traditional)": ["zh"],
            "Chinese + English": ["zh", "en"],
            "English": ["en"],
            "Japanese": ["ja"],
            "Korean": ["ko"],
            "Multilingual (zh, en, ja, ko)": ["zh", "en", "ja", "ko"],
        }
        # Common Latin misrecognitions when the actual script is Chinese (model has no language hint in API)
        _LATIN_TO_CHINESE_FIXES = {"Wg": "王", "Wo": "我", "Ol": "了", "On": "们", "Og": "公"}
        params = {
            "language": {
                "type": "selector",
                "options": list(lang_map.keys()),
                "value": "Chinese + English",
                "description": "Language for display; set to Chinese (Simplified) for Chinese-only to reduce Latin misreads (e.g. 'Wg'→王).",
            },
            "fix_latin_misread": {
                "type": "checkbox",
                "value": True,
                "description": "When language is Chinese-only, fix common Latin misrecognitions (e.g. Wg→王).",
            },
            "device": DEVICE_SELECTOR(),
            "batch_size": {
                "value": 16,
                "description": "Batch size for recognition (reduce if OOM).",
            },
            "crop_padding": {
                "type": "line_editor",
                "value": 6,
                "description": "Pixels to add around each box when cropping for OCR (0–24). Reduces clipped text at edges (e.g. with CTD).",
            },
            "description": "Surya OCR – 90+ languages (pip install surya-ocr)",
        }
        _load_model_keys = {"_recognizer"}

        def __init__(self, **params) -> None:
            super().__init__(**params)
            self.device = self.params["device"]["value"]
            self._recognizer = None

        def _load_model(self):
            if self._recognizer is not None:
                return
            try:
                foundation = FoundationPredictor(device=self.device)
                self._recognizer = _surya_predictor(foundation)
            except Exception as e:
                raise RuntimeError(
                    f"Surya OCR failed to load model: {e}. "
                    "Ensure surya-ocr is installed: pip install surya-ocr (Python 3.10+, PyTorch)."
                ) from e

        def _ocr_blk_list(self, img: np.ndarray, blk_list: List[TextBlock], *args, **kwargs) -> None:
            im_h, im_w = img.shape[:2]
            pad = max(0, int(self.params.get("crop_padding", {}).get("value", 6)))
            images_pil = []
            indices = []
            for idx, blk in enumerate(blk_list):
                x1, y1, x2, y2 = blk.xyxy
                if not (0 <= x1 < x2 <= im_w and 0 <= y1 < y2 <= im_h):
                    blk.text = [""]
                    continue
                # Pad crop so text at box edges (e.g. last line) is not clipped; helps when CTD bbox is tight
                x1a = max(0, x1 - pad)
                y1a = max(0, y1 - pad)
                x2a = min(im_w, x2 + pad)
                y2a = min(im_h, y2 + pad)
                # Vertical text is often clipped at bottom; add extra padding there
                if getattr(blk, "src_is_vertical", False):
                    extra_bottom = max(pad, min(int((y2 - y1) * 0.12), 24))
                    y2a = min(im_h, y2a + extra_bottom)
                crop = img[y1a:y2a, x1a:x2a]
                if crop.size == 0:
                    blk.text = [""]
                    continue
                images_pil.append(_cv2_to_pil_rgb(crop))
                indices.append(idx)

            if not images_pil:
                return
            batch_size = max(1, int(self.params.get("batch_size", {}).get("value", 16)))
            # One bbox per crop: full image so the recognizer processes the whole crop as one line.
            bboxes = [[[0, 0, im.size[0], im.size[1]]] for im in images_pil]
            task_names = [_surya_task_name] * len(images_pil)

            try:
                results = self._recognizer(
                    images_pil,
                    task_names=task_names,
                    bboxes=bboxes,
                    recognition_batch_size=batch_size,
                    drop_repeated_text=True,
                )
            except Exception as e:
                self.logger.error(f"Surya recognition error: {e}")
                for idx in indices:
                    blk_list[idx].text = [""]
                return

            lang_opt = self.params.get("language", {}).get("value", "Chinese + English")
            fix_latin = self.params.get("fix_latin_misread", {}).get("value", True)
            chinese_only = lang_opt in ("Chinese (Simplified)", "Chinese (Traditional)")

            for idx, result in zip(indices, results):
                parts = []
                if result.text_lines:
                    for line in result.text_lines:
                        t = (line.text or "").strip()
                        if t:
                            parts.append(t)
                text = "\n".join(parts) if parts else ""
                text = trim_ocr_repetition(text)
                text = _normalize_ocr_text(text, chinese_only=chinese_only)
                if chinese_only and fix_latin and text and text.isascii() and len(text) <= 4:
                    for lat, ch in SuryaOCR._LATIN_TO_CHINESE_FIXES.items():
                        text = text.replace(lat, ch)
                blk_list[idx].text = [text]

        def ocr_img(self, img: np.ndarray) -> str:
            blk = TextBlock(xyxy=[0, 0, img.shape[1], img.shape[0]])
            blk.lines = [[[0, 0], [img.shape[1], 0], [img.shape[1], img.shape[0]], [0, img.shape[0]]]]
            self._ocr_blk_list(img, [blk])
            return blk.get_text()
