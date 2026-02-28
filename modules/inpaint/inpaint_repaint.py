"""
Optional inpainter using Diffusers RePaint (DDPM-based inpainting).
Uses a pretrained unconditional DDPM; good for arbitrary masks. Default: google/ddpm-ema-celebahq-256 (256x256).
Install: pip install diffusers transformers accelerate torch
"""
import numpy as np
import cv2
from typing import List

from ..base import DEVICE_SELECTOR
from .base import InpainterBase, register_inpainter, TextBlock
from utils.imgproc_utils import resize_keepasp

_REPAINT_AVAILABLE = False
try:
    from diffusers import RePaintPipeline, RePaintScheduler
    import torch
    from PIL import Image
    _REPAINT_AVAILABLE = True
except ImportError:
    import logging
    logging.getLogger("BallonTranslator").debug(
        "RePaint inpainter not available. Install: pip install diffusers transformers accelerate torch"
    )


if _REPAINT_AVAILABLE:

    @register_inpainter("repaint")
    class RePaintInpainter(InpainterBase):
        """
        RePaint: DDPM-based inpainting. Resizes crops to model size (default 256), then pastes back.
        Slower than LaMa; useful for diverse, high-quality fill with extreme masks.
        """
        inpaint_by_block = True
        check_need_inpaint = True

        params = {
            "model_name": {
                "type": "line_editor",
                "value": "google/ddpm-ema-celebahq-256",
                "description": "Pretrained DDPM model id (e.g. google/ddpm-ema-celebahq-256). Must be UNet2D/DDPM.",
            },
            "device": DEVICE_SELECTOR(),
            "inpaint_size": {
                "type": "line_editor",
                "value": 256,
                "description": "Model input size (256 for celebahq; resize crop to this).",
            },
            "num_inference_steps": {
                "type": "line_editor",
                "value": 250,
                "description": "Denoising steps (100–500; more = better quality, slower).",
            },
            "eta": {
                "type": "line_editor",
                "value": 0.0,
                "description": "DDIM eta (0 = deterministic).",
            },
            "jump_length": {
                "type": "line_editor",
                "value": 10,
                "description": "RePaint jump length.",
            },
            "jump_n_sample": {
                "type": "line_editor",
                "value": 10,
                "description": "RePaint jump n sample.",
            },
            "description": "RePaint (DDPM) inpainting. Install: pip install diffusers accelerate",
        }
        _load_model_keys = {"pipeline"}

        def __init__(self, **params) -> None:
            super().__init__(**params)
            self.device = (self.params.get("device") or {}).get("value", "cpu")
            if self.device == "gpu":
                self.device = "cuda"
            self.pipeline = None
            self._model_name = None

        def _load_model(self):
            model_name = (self.params.get("model_name") or {}).get("value", "google/ddpm-ema-celebahq-256") or "google/ddpm-ema-celebahq-256"
            if self.pipeline is not None and self._model_name == model_name:
                return
            self._model_name = model_name
            dtype = torch.float32
            if self.device == "cuda":
                dtype = torch.float16
            scheduler = RePaintScheduler.from_pretrained(model_name)
            self.pipeline = RePaintPipeline.from_pretrained(model_name, scheduler=scheduler)
            self.pipeline = self.pipeline.to(self.device, dtype=dtype)

        def _inpaint(self, img: np.ndarray, mask: np.ndarray, textblock_list: List[TextBlock] = None) -> np.ndarray:
            if img.ndim == 3 and img.shape[2] == 4:
                img = img[:, :, :3]
            if mask.ndim == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask_bin = (mask > 127).astype(np.uint8)
            h, w = img.shape[:2]
            try:
                size = 256
                vs = self.params.get("inpaint_size", {})
                if isinstance(vs, dict):
                    try:
                        size = max(64, min(256, int(vs.get("value", 256))))
                    except (TypeError, ValueError):
                        pass
                new_shape = size if max(h, w) > size else None
                img_resized = resize_keepasp(img, new_shape, stride=None)
                mask_resized = resize_keepasp((mask_bin * 255).astype(np.uint8), new_shape, stride=None)
                rh, rw = img_resized.shape[:2]
                if rh != size or rw != size:
                    img_resized = cv2.resize(img_resized, (size, size), interpolation=cv2.INTER_LANCZOS4)
                    mask_resized = cv2.resize(mask_resized, (size, size), interpolation=cv2.INTER_NEAREST)
                pil_img = Image.fromarray(img_resized)
                # RePaint: 0.0 = region to inpaint, 1.0 = keep. Our mask: 255 = inpaint.
                repaint_mask = (255 - mask_resized).astype(np.uint8)
                pil_mask = Image.fromarray(repaint_mask).convert("L")
                steps = 250
                st = self.params.get("num_inference_steps", {})
                if isinstance(st, dict):
                    try:
                        steps = max(50, min(500, int(st.get("value", 250))))
                    except (TypeError, ValueError):
                        pass
                eta = 0.0
                et = self.params.get("eta", {})
                if isinstance(et, dict):
                    try:
                        eta = float(et.get("value", 0.0))
                    except (TypeError, ValueError):
                        pass
                jlen = 10
                jl = self.params.get("jump_length", {})
                if isinstance(jl, dict):
                    try:
                        jlen = int(jl.get("value", 10))
                    except (TypeError, ValueError):
                        pass
                jn = 10
                jnparam = self.params.get("jump_n_sample", {})
                if isinstance(jnparam, dict):
                    try:
                        jn = int(jnparam.get("value", 10))
                    except (TypeError, ValueError):
                        pass
                out = self.pipeline(
                    image=pil_img,
                    mask_image=pil_mask,
                    num_inference_steps=steps,
                    eta=eta,
                    jump_length=jlen,
                    jump_n_sample=jn,
                    output_type="np",
                )
                out = (out.images[0] * 255).clip(0, 255).astype(np.uint8)
                if out.shape[0] != rh or out.shape[1] != rw:
                    out = cv2.resize(out, (rw, rh), interpolation=cv2.INTER_LANCZOS4)
                if (rh, rw) != (h, w):
                    out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LANCZOS4)
                mask_3 = (mask_bin[:, :, None]).astype(np.float32)
                result = (out.astype(np.float32) * mask_3 + img.astype(np.float32) * (1 - mask_3)).astype(np.uint8)
                return result
            except Exception as e:
                self.logger.error(f"RePaint inpainting failed: {e}")
                return img.copy()

        def updateParam(self, param_key: str, param_content):
            super().updateParam(param_key, param_content)
            if param_key == "device":
                self.device = (self.params.get("device") or {}).get("value", "cpu")
                if self.device == "gpu":
                    self.device = "cuda"
                if self.pipeline is not None:
                    self.pipeline = self.pipeline.to(self.device)
            elif param_key == "model_name":
                self.pipeline = None
                self._model_name = None
