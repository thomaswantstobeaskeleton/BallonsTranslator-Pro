"""
LaMa Manga ONNX – Optional inpainter using mayocream/lama-manga-onnx (ONNX export of AnimeMangaInpainting).
Download the ONNX model from Hugging Face and set the path. Requires: pip install onnxruntime
"""
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
        "LaMa manga ONNX not available. Install: pip install onnxruntime"
    )


if _LAMA_ONNX_AVAILABLE:

    @register_inpainter("lama_manga_onnx")
    class LamaMangaOnnxInpainter(InpainterBase):
        """
        LaMa manga inpainting via ONNX (e.g. mayocream/lama-manga-onnx).
        Download the .onnx from Hugging Face and set model_path.
        """
        inpaint_by_block = False
        check_need_inpaint = True

        params = {
            "model_path": {
                "type": "line_editor",
                "value": "data/models/lama_manga.onnx",
                "description": "Path to LaMa manga ONNX file (download from mayocream/lama-manga-onnx).",
            },
            "inpaint_size": {
                "type": "selector",
                "options": [512, 768, 1024, 1536, 2048],
                "value": 1024,
                "description": "Max side for inference.",
            },
        }
        _load_model_keys = {"session"}

        def __init__(self, **params) -> None:
            super().__init__(**params)
            self.session = None
            self._input_names = []
            self._output_names = []

        def _load_model(self):
            import os
            path = (self.params.get("model_path") or {}).get("value", "data/models/lama_manga.onnx")
            if isinstance(path, str):
                path = path.strip()
            if not path or not os.path.isfile(path):
                self.logger.warning(
                    f"LaMa manga ONNX not found at {path}. "
                    "Download from https://huggingface.co/mayocream/lama-manga-onnx and set model_path."
                )
                return
            self.session = ort.InferenceSession(
                path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
                if "CUDAExecutionProvider" in ort.get_available_providers()
                else ["CPUExecutionProvider"],
            )
            self._input_names = [inp.name for inp in self.session.get_inputs()]
            self._output_names = [out.name for out in self.session.get_outputs()]

        def _inpaint(self, img: np.ndarray, mask: np.ndarray, textblock_list: List[TextBlock] = None) -> np.ndarray:
            if self.session is None or len(self._input_names) < 2:
                return img.copy()
            im_h, im_w = img.shape[:2]
            if img.ndim == 3 and img.shape[2] == 4:
                img = img[:, :, :3]
            if mask.ndim == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask_bin = (mask > 127).astype(np.float32)
            inpaint_size = int((self.params.get("inpaint_size") or {}).get("value", 1024))
            max_side = max(im_h, im_w)
            if max_side > inpaint_size:
                img = resize_keepasp(img, inpaint_size, stride=64)
                mask_bin = resize_keepasp(
                    (mask_bin * 255).astype(np.uint8), inpaint_size, stride=64
                ).astype(np.float32) / 255.0
            h, w = img.shape[:2]
            img_f = img.astype(np.float32) / 255.0
            img_chw = np.transpose(img_f, (2, 0, 1))[np.newaxis, ...]
            mask_chw = mask_bin[np.newaxis, np.newaxis, ...]
            feeds = {}
            for name in self._input_names:
                if "mask" in name.lower() or name == self._input_names[1]:
                    feeds[name] = mask_chw.astype(np.float32)
                else:
                    feeds[name] = img_chw.astype(np.float32)
            out = self.session.run(self._output_names, feeds)
            result = out[0]
            if result.ndim == 4:
                result = result[0]
            if result.shape[0] == 3:
                result = np.transpose(result, (1, 2, 0))
            result = (np.clip(result, 0, 1) * 255).astype(np.uint8)
            if result.shape[:2] != (im_h, im_w):
                result = cv2.resize(result, (im_w, im_h), interpolation=cv2.INTER_LINEAR)
            mask_3 = (mask > 127).astype(np.float32)[:, :, np.newaxis]
            return (result * mask_3 + img.astype(np.float32) * (1 - mask_3)).astype(np.uint8)
