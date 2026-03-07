from typing import Tuple, List, Dict, Union, Callable
import numpy as np
import cv2
from collections import OrderedDict
import re

from utils.textblock import TextBlock
from utils.registry import Registry
OCR = Registry('OCR')
register_OCR = OCR.register_module

from ..base import BaseModule, DEFAULT_DEVICE, DEVICE_SELECTOR, LOGGER
from utils.io_utils import normalize_line_breaks
from utils.imgproc_utils import xywh2xyxypoly

# Substrings that indicate OCR/API junk or watermarks; lines containing these are cleared from source.
SOURCE_JUNK_PHRASES = [
    "the quick brown fox jumps over the lazy dog",
    "the image is too blurry to recognize",
    "too blurry to recognize any text",
    "腾讯动漫",
    "[table_companyrpttype]",
    "the world in a book",
    "온라인",  # common in watermarks / error text
]
SOURCE_JUNK_PATTERN = re.compile(
    "|".join(re.escape(p) for p in SOURCE_JUNK_PHRASES),
    re.IGNORECASE
)


def _filter_source_junk(text_list: List[str]) -> None:
    """In-place: replace any line that contains known junk with empty string."""
    for i, line in enumerate(text_list):
        if not line or not line.strip():
            continue
        # Match ASCII case-insensitive; CJK and other phrases as-is
        if SOURCE_JUNK_PATTERN.search(line):
            text_list[i] = ""


def ensure_blocks_have_lines(blk_list: List[TextBlock]) -> None:
    """
    Ensure every block has at least one line (for OCR and downstream).
    When lines are missing or empty (e.g. loaded from project, or detector left box-only),
    derive one line from xyxy so block is valid for line-based OCR and cache.
    """
    added = 0
    for blk in blk_list:
        lines = getattr(blk, "lines", None)
        if lines is not None and len(lines) > 0:
            continue
        xyxy = getattr(blk, "xyxy", None)
        if not xyxy or len(xyxy) != 4:
            continue
        try:
            x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
            if x2 <= x1 or y2 <= y1:
                continue
            w, h = x2 - x1, y2 - y1
            xywh = np.array([[x1, y1, w, h]])
            poly = xywh2xyxypoly(xywh, to_int=False)
            blk.lines = poly.reshape(-1, 4, 2).tolist()
            added += 1
        except Exception:
            continue
    if added > 0:
        LOGGER.debug("ensure_blocks_have_lines: derived line(s) from xyxy for %d block(s) that had no lines", added)


def normalize_block_text(blk: TextBlock) -> None:
    """
    Normalize blk.text in-place so it is always a list of line strings.
    Handles: str (splitlines), tuple (convert to list), list with newlines in elements,
    empty list -> [""]. Call after OCR or when applying cached OCR so all paths stay consistent.
    """
    raw = getattr(blk, "text", None)
    if isinstance(raw, str):
        blk.text = [ln.strip() for ln in raw.splitlines()] if raw.strip() else [""]
    elif isinstance(raw, tuple):
        blk.text = list(raw)
    elif not isinstance(raw, list):
        blk.text = [str(raw)] if raw else [""]
    else:
        blk.text = list(raw)
    expanded = []
    for t in blk.text:
        if isinstance(t, str) and "\n" in t:
            expanded.extend(ln.strip() for ln in t.splitlines())
        else:
            expanded.append(str(t).strip() if t is not None else "")
    blk.text = expanded if expanded else [""]
    blk.text = [normalize_line_breaks(t) for t in blk.text]
    blk.text = [t.replace("\uFFFD", "\u25A1") for t in blk.text]
    _filter_source_junk(blk.text)


class OCRBase(BaseModule):

    _postprocess_hooks = OrderedDict()
    _preprocess_hooks = OrderedDict()
    _line_only: bool = False

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.name = ''
        for key in OCR.module_dict:
            if OCR.module_dict[key] == self.__class__:
                self.name = key
                break

    def run_ocr(self, img: np.ndarray, blk_list: List[TextBlock] = None, *args, **kwargs) -> Union[List[TextBlock], str]:

        if not self.all_model_loaded():
            self.load_model()

        if img.ndim == 3 and img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        if blk_list is None:
            text = self.ocr_img(img)
            return text
        elif isinstance(blk_list, TextBlock):
            blk_list = [blk_list]

        ensure_blocks_have_lines(blk_list)
        for blk in blk_list:
            if self.name != 'none_ocr':
                blk.text = []
                
        self._ocr_blk_list(img, blk_list, *args, **kwargs)
        for blk in blk_list:
            normalize_block_text(blk)
        self._register_spell_check_hook()
        for callback_name, callback in self._postprocess_hooks.items():
            callback(textblocks=blk_list, img=img, ocr_module=self)

        return blk_list

    def run_ocr_translate(self, img: np.ndarray, blk_list: List[TextBlock], source_lang: str, target_lang: str, *args, **kwargs) -> List[TextBlock]:
        """
        Optional: one-step VLM flow (image -> translation) implemented by specific OCR modules.
        Default: not supported.
        """
        raise NotImplementedError

    def _ocr_blk_list(self, img: np.ndarray, blk_list: List[TextBlock], *args, **kwargs) -> None:
        raise NotImplementedError

    def ocr_img(self, img: np.ndarray) -> str:
        raise NotImplementedError

    @classmethod
    def _register_spell_check_hook(cls):
        if 'spell_check' in cls._postprocess_hooks:
            return
        try:
            from utils.ocr_spellcheck import spell_check_textblocks
            cls._postprocess_hooks['spell_check'] = spell_check_textblocks
        except ImportError:
            pass


    def offload_to_cpu(self) -> None:
        """Move OCR model to CPU and free GPU memory. Used before inpainting when both OCR and inpainting use GPU."""
        model = getattr(self, "model", None)
        if model is not None and hasattr(model, "to"):
            import torch
            model.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def restore_to_device(self) -> None:
        """Move OCR model back to its configured device. Call before running OCR again after offload_to_cpu."""
        model = getattr(self, "model", None)
        device = getattr(self, "device", None)
        if model is not None and device is not None and hasattr(model, "to"):
            model.to(device)
