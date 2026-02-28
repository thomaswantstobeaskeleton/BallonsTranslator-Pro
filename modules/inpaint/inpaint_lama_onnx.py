"""
LaMa general ONNX – Optional inpainter using opencv/inpainting_lama (general-purpose LaMa ONNX, 512×512).
Download inpainting_lama_2025jan.onnx from Hugging Face and set model_path.
Requires: pip install onnxruntime
"""
import os
import numpy as np
import cv2
from typing import List

from .base import InpainterBase, register_inpainter, TextBlock
from utils.imgproc_utils import resize_keepasp

_LAMA_ONNX_AVAILABLE = False
try:
    import onnxruntime as ort
    _LAMA_ONNX_AVAILABLE = True
except ImportError:
    import logging
    logging.getLogger("BallonTranslator").debug(
        "LaMa ONNX not available. Install: pip install onnxruntime"
    )

LAMA_ONNX_SIZE = 512


if _LAMA_ONNX_AVAILABLE:

    @register_inpainter("lama_onnx")
    class LamaOnnxInpainter(InpainterBase):
        """
        LaMa general inpainting via ONNX (opencv/inpainting_lama).
        General-purpose; for manga-specific use lama_manga_onnx or lama_large_512px.
        """
        inpaint_by_block = True
        check_need_inpaint = True

        params = {
            "model_path": {
                "type": "line_editor",
                "value": "data/models/inpainting_lama_2025jan.onnx",
                "description": "Path to LaMa ONNX (download from Hugging Face opencv/inpainting_lama).",
            },
            "inpaint_size": {
                "type": "selector",
                "options": [512, 768, 1024],
                "value": 512,
                "description": "Model expects 512×512; max side for input before resize.",
            },
        }
        _load_model_keys = {"session"}

        def __init__(self, **params) -> None:
            super().__init__(**params)
            self.session = None
            self._input_names = []

        def _load_model(self):
            path = (self.params.get("model_path") or {}).get("value", "data/models/inpainting_lama_2025jan.onnx")
            if isinstance(path, str):
                path = path.strip()
            if not path or not os.path.isfile(path):
                self.logger.warning(
                    f"LaMa ONNX not found at {path}. "
                    "Download from https://huggingface.co/opencv/inpainting_lama/resolve/main/inpainting_lama_2025jan.onnx"
                )
                return
            self.session = ort.InferenceSession(
                path,
                providers=(
                    ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    if "CUDAExecutionProvider" in ort.get_available_providers()
                    else ["CPUExecutionProvider"]
                ),
            )
            self._input_names = [inp.name for inp in self.session.get_inputs()]

        def _inpaint(self, img: np.ndarray, mask: np.ndarray, textblock_list: List[TextBlock] = None) -> np.ndarray:
            if self.session is None or len(self._input_names) < 2:
                return img.copy()
            im_h, im_w = img.shape[:2]
            if img.ndim == 3 and img.shape[2] == 4:
                img = img[:, :, :3]
            if mask.ndim == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask_bin = (mask > 127).astype(np.float32)
            mask_orig = mask_bin[:, :, np.newaxis]
            inpaint_size = int((self.params.get("inpaint_size") or {}).get("value", 512))
            max_side = max(im_h, im_w)
            if max_side > inpaint_size:
                img_resized = resize_keepasp(img, inpaint_size, stride=None)
                mask_resized = resize_keepasp(
                    (mask_bin * 255).astype(np.uint8), inpaint_size, stride=None
                ).astype(np.float32) / 255.0
            else:
                img_resized = img
                mask_resized = mask_bin
            h, w = img_resized.shape[:2]
            img_512 = cv2.resize(img_resized, (LAMA_ONNX_SIZE, LAMA_ONNX_SIZE), interpolation=cv2.INTER_LANCZOS4)
            mask_512 = cv2.resize(mask_resized, (LAMA_ONNX_SIZE, LAMA_ONNX_SIZE), interpolation=cv2.INTER_NEAREST)
            img_chw = np.transpose(img_512.astype(np.float32) / 255.0, (2, 0, 1))[np.newaxis, ...]
            mask_chw = mask_512.astype(np.float32).reshape(1, 1, LAMA_ONNX_SIZE, LAMA_ONNX_SIZE)
            feeds = {}
            for name in self._input_names:
                if "mask" in name.lower():
                    feeds[name] = mask_chw
                else:
                    feeds[name] = img_chw
            out = self.session.run(None, feeds)
            result = out[0]
            if result.ndim == 4:
                result = result[0]
            if result.shape[0] == 3:
                result = np.transpose(result, (1, 2, 0))
            result = (np.clip(result, 0, 1) * 255).astype(np.uint8)
            result = cv2.resize(result, (w, h), interpolation=cv2.INTER_LANCZOS4)
            if (h, w) != (im_h, im_w):
                result = cv2.resize(result, (im_w, im_h), interpolation=cv2.INTER_LANCZOS4)
            return (result.astype(np.float32) * mask_orig + img.astype(np.float32) * (1 - mask_orig)).astype(np.uint8)
