from typing import Tuple, List, Dict, Union, Callable
import numpy as np
import cv2
from collections import OrderedDict

from utils.textblock import TextBlock
from utils.registry import Registry
OCR = Registry('OCR')
register_OCR = OCR.register_module

from ..base import BaseModule, DEFAULT_DEVICE, DEVICE_SELECTOR, LOGGER
from utils.io_utils import normalize_line_breaks

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

        for blk in blk_list:
            if self.name != 'none_ocr':
                blk.text = []
                
        self._ocr_blk_list(img, blk_list, *args, **kwargs)
        for blk in blk_list:
            blk.text = [normalize_line_breaks(t) for t in blk.text]
            # Avoid empty squares: replace Unicode replacement char with visible placeholder (all OCRs)
            blk.text = [t.replace("\uFFFD", "\u25A1") for t in blk.text]
        self._register_spell_check_hook()
        for callback_name, callback in self._postprocess_hooks.items():
            callback(textblocks=blk_list, img=img, ocr_module=self)

        return blk_list

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
