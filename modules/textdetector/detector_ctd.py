import numpy as np
import cv2
import os
from typing import Tuple, List

from .base import register_textdetectors, TextDetectorBase, TextBlock, DEFAULT_DEVICE, DEVICE_SELECTOR, ProjImgTrans
from .ctd import CTDModel

CTD_ONNX_PATH = 'data/models/comictextdetector.pt.onnx'
CTD_TORCH_PATH = 'data/models/comictextdetector.pt'

def load_ctd_model(model_path, device, detect_size=1024) -> CTDModel:
    model = CTDModel(model_path, detect_size=detect_size, device=device)
    
    return model

@register_textdetectors('ctd')
class ComicTextDetector(TextDetectorBase):

    params = {
        'detect_size': {
            'type': 'selector',
            'options': [896, 1024, 1152, 1280, 1536, 1792, 2048, 2400],
            'value': 1536
        }, 
        'det_rearrange_max_batches': {
            'type': 'selector',
            'options': [1, 2, 4, 6, 8, 12, 16, 24, 32], 
            'value': 4
        },
        'device': DEVICE_SELECTOR(),
        'description': 'ComicTextDetector',
        'font size multiplier': 1.,
        'font size max': -1,
        'font size min': -1,
        'mask dilate size': 5,
        'merge font size tolerance': {
            'type': 'line_editor',
            'value': 3.0,
            'description': 'Merge scattered lines with similar font size. Higher = fewer boxes per bubble (e.g. 3.0).',
        },
        'box score threshold': {
            'type': 'line_editor',
            'value': 0.45,
            'description': 'Min confidence (0.35–0.7). Lower = more boxes (e.g. small out-of-bubble text).',
        },
        'min box area': {
            'type': 'line_editor',
            'value': 0,
            'description': 'Drop regions smaller than this (px²). 0=off. Use 100–300 to remove tiny false boxes.',
        },
        'custom_onnx_path': {
            'type': 'line_editor',
            'value': '',
            'description': 'Optional: path to a custom ONNX model (e.g. mayocream/comic-text-detector-onnx). Leave empty to use built-in CTD ONNX. Used when device is CPU or when forcing ONNX.',
        },
    }
    _load_model_keys = {'model'}
    download_file_list = [{
        'url': 'https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/',
        'files': ['data/models/comictextdetector.pt', 'data/models/comictextdetector.pt.onnx'],
        'sha256_pre_calculated': ['1f90fa60aeeb1eb82e2ac1167a66bf139a8a61b8780acd351ead55268540cccb', '1a86ace74961413cbd650002e7bb4dcec4980ffa21b2f19b86933372071d718f'],
        'concatenate_url_filename': 2,
    }]

    device = DEFAULT_DEVICE
    detect_size = 1024
    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.model: CTDModel = None

    @property
    def device(self):
        return self.params['device']['value']
    
    @property
    def detect_size(self):
        return int(self.params['detect_size']['value'])

    def _load_model(self):
        custom_onnx = (self.params.get('custom_onnx_path') or {}).get('value', '') or ''
        custom_onnx = custom_onnx.strip() if isinstance(custom_onnx, str) else ''
        if self.device != 'cpu':
            self.model = load_ctd_model(CTD_TORCH_PATH, self.device, self.detect_size)
        else:
            onnx_path = custom_onnx if custom_onnx and os.path.isfile(custom_onnx) else CTD_ONNX_PATH
            self.model = load_ctd_model(onnx_path, self.device, self.detect_size)

    def _detect(self, img: np.ndarray, proj: ProjImgTrans) -> Tuple[np.ndarray, List[TextBlock]]:
        merge_tol = self.get_param_value('merge font size tolerance')
        if merge_tol is None or (isinstance(merge_tol, str) and not merge_tol.strip()):
            merge_tol = 3.0
        else:
            try:
                merge_tol = float(merge_tol)
            except (TypeError, ValueError):
                merge_tol = 3.0
        self.model.merge_fntsize_tol_hor = merge_tol
        self.model.merge_fntsize_tol_ver = merge_tol
        box_thresh = self.get_param_value('box score threshold')
        try:
            box_thresh = float(box_thresh) if box_thresh not in (None, '') else 0.45
            box_thresh = max(0.35, min(0.8, box_thresh))
        except (TypeError, ValueError):
            box_thresh = 0.45
        self.model.box_thresh = box_thresh
        min_area = self.get_param_value('min box area')
        try:
            min_area = int(min_area) if min_area not in (None, '') else 0
            min_area = max(0, min_area)
        except (TypeError, ValueError):
            min_area = 0
        self.model.min_box_area = min_area
        _, mask, blk_list = self.model(img)
        
        fnt_rsz = self.get_param_value('font size multiplier')
        fnt_max = self.get_param_value('font size max')
        fnt_min = self.get_param_value('font size min')
        for blk in blk_list:
            sz = blk._detected_font_size * fnt_rsz
            if fnt_max > 0:
                sz = min(fnt_max, sz)
            if fnt_min > 0:
                sz = max(fnt_min, sz)
            blk.font_size = sz
            blk._detected_font_size = sz

        ksize = self.get_param_value('mask dilate size')
        if ksize > 0:
            element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ksize + 1, 2 * ksize + 1),(ksize, ksize))
            mask = cv2.dilate(mask, element)

        return mask, blk_list

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        device = self.device
        if self.model is not None:
            if self.model.device != device:
                self.model.device = device
                if device != 'cpu':
                    self.model.load_model(CTD_TORCH_PATH)
                else:
                    custom_onnx = (self.params.get('custom_onnx_path') or {}).get('value', '') or ''
                    custom_onnx = custom_onnx.strip() if isinstance(custom_onnx, str) else ''
                    onnx_path = custom_onnx if custom_onnx and os.path.isfile(custom_onnx) else CTD_ONNX_PATH
                    self.model.load_model(onnx_path)
            self.model.detect_size = self.detect_size