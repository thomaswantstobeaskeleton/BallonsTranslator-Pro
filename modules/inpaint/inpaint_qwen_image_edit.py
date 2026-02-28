"""
Optional inpainter using Diffusers Qwen-Image-Edit (QwenImageInpaintPipeline).
Supports semantic/appearance editing and text removal. Model: Qwen/Qwen-Image-Edit or Qwen/Qwen-Image-Edit-2509.
Install: pip install diffusers transformers accelerate torch
"""
import numpy as np
import cv2
from typing import List

from ..base import DEVICE_SELECTOR
from .base import InpainterBase, register_inpainter, TextBlock
from utils.imgproc_utils import resize_keepasp

_QWEN_INPAINT_AVAILABLE = False
try:
    from diffusers import QwenImageInpaintPipeline
    import torch
    from PIL import Image
    _QWEN_INPAINT_AVAILABLE = True
except ImportError:
    try:
        from diffusers import QwenImageEditInpaintPipeline as QwenImageInpaintPipeline
        import torch
        from PIL import Image
        _QWEN_INPAINT_AVAILABLE = True
    except ImportError:
        import logging
        logging.getLogger("BallonTranslator").debug(
            "Qwen-Image-Edit inpainting not available. Install: pip install diffusers transformers accelerate"
        )


if _QWEN_INPAINT_AVAILABLE:

    @register_inpainter("qwen_image_edit")
    class QwenImageEditInpainter(InpainterBase):
        """
        Qwen-Image-Edit inpainting via Diffusers. Good for text removal and semantic editing.
        Heavy (Qwen2.5-VL backbone); use when quality matters more than speed.
        """
        inpaint_by_block = True
        check_need_inpaint = True

        params = {
            "model_name": {
                "type": "line_editor",
                "value": "Qwen/Qwen-Image-Edit",
                "description": "Model id: Qwen/Qwen-Image-Edit or Qwen/Qwen-Image-Edit-2509.",
            },
            "device": DEVICE_SELECTOR(),
            "inpaint_size": {
                "type": "line_editor",
                "value": 1024,
                "description": "Max side for inference (512–1024).",
            },
            "prompt": {
                "type": "line_editor",
                "value": "clean background, solid color, no text",
                "description": "Prompt for inpainting (text removal: neutral description).",
            },
            "negative_prompt": {
                "type": "line_editor",
                "value": " ",
                "description": "Negative prompt (space for CFG).",
            },
            "num_inference_steps": {
                "type": "line_editor",
                "value": 50,
                "description": "Denoising steps (35–50).",
            },
            "strength": {
                "type": "line_editor",
                "value": 1.0,
                "description": "Inpaint strength (1.0 = full replace).",
            },
            "description": "Qwen-Image-Edit inpainting. Install: pip install diffusers accelerate",
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
            model_name = (self.params.get("model_name") or {}).get("value", "Qwen/Qwen-Image-Edit") or "Qwen/Qwen-Image-Edit"
            if self.pipeline is not None and self._model_name == model_name:
                return
            self._model_name = model_name
            dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
            try:
                self.pipeline = QwenImageInpaintPipeline.from_pretrained(model_name, torch_dtype=dtype)
            except Exception as e:
                self.logger.warning(f"QwenImageInpaintPipeline failed, try float32: {e}")
                self.pipeline = QwenImageInpaintPipeline.from_pretrained(model_name, torch_dtype=torch.float32)
            self.pipeline = self.pipeline.to(self.device)
            if self.device == "cuda":
                try:
                    self.pipeline.enable_vae_slicing()
                except Exception:
                    pass

        def _inpaint(self, img: np.ndarray, mask: np.ndarray, textblock_list: List[TextBlock] = None) -> np.ndarray:
            if img.ndim == 3 and img.shape[2] == 4:
                img = img[:, :, :3]
            if mask.ndim == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask_bin = (mask > 127).astype(np.uint8) * 255
            h, w = img.shape[:2]
            img_orig = img.copy()
            try:
                size = 1024
                vs = self.params.get("inpaint_size", {})
                if isinstance(vs, dict):
                    try:
                        size = max(512, min(1024, int(vs.get("value", 1024))))
                    except (TypeError, ValueError):
                        pass
                if max(h, w) > size:
                    img = resize_keepasp(img, size, stride=None)
                    mask_bin = resize_keepasp(mask_bin, size, stride=None)
                pil_img = Image.fromarray(img)
                pil_mask = Image.fromarray(mask_bin).convert("L")
                prompt = (self.params.get("prompt") or {}).get("value", "clean background, no text") or "clean background, no text"
                negative = (self.params.get("negative_prompt") or {}).get("value", " ") or " "
                steps = 50
                st = self.params.get("num_inference_steps", {})
                if isinstance(st, dict):
                    try:
                        steps = max(20, min(80, int(st.get("value", 50))))
                    except (TypeError, ValueError):
                        pass
                strength = 1.0
                sval = self.params.get("strength", {})
                if isinstance(sval, dict):
                    try:
                        strength = max(0.5, min(1.0, float(sval.get("value", 1.0))))
                    except (TypeError, ValueError):
                        pass
                out = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative,
                    image=pil_img,
                    mask_image=pil_mask,
                    strength=strength,
                    num_inference_steps=steps,
                    output_type="np",
                )
                result = (out.images[0] * 255).clip(0, 255).astype(np.uint8)
                if result.shape[0] != h or result.shape[1] != w:
                    result = cv2.resize(result, (w, h), interpolation=cv2.INTER_LANCZOS4)
                mask_3 = (mask > 127).astype(np.float32)[:, :, np.newaxis]
                return (result.astype(np.float32) * mask_3 + img_orig.astype(np.float32) * (1 - mask_3)).astype(np.uint8)
            except Exception as e:
                self.logger.error(f"Qwen-Image-Edit inpainting failed: {e}")
                return img_orig.copy()

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
