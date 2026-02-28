"""
Optional inpainter using Hugging Face Diffusers (Stable Diffusion inpainting).
Install: pip install diffusers transformers accelerate
Best for: alternative inpainting when you prefer a diffusion model (e.g. runwayml/stable-diffusion-inpainting).
Heavier and slower than LaMa; use when you want SD-style inpainting.
"""
import numpy as np
import cv2
from typing import List

from ..base import DEVICE_SELECTOR
from .base import InpainterBase, register_inpainter, TextBlock

_DIFFUSERS_AVAILABLE = False
try:
    from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint import StableDiffusionInpaintPipeline
    import torch
    from PIL import Image
    _DIFFUSERS_AVAILABLE = True
except ImportError:
    import logging
    logging.getLogger("BallonTranslator").debug(
        "Diffusers inpainting not available. Install: pip install diffusers transformers accelerate"
    )


if _DIFFUSERS_AVAILABLE:

    @register_inpainter("diffusers_sd_inpaint")
    class DiffusersSDInpainter(InpainterBase):
        """
        Stable Diffusion inpainting via Hugging Face Diffusers.
        Uses a neutral prompt for text removal. Slower than LaMa; good for natural-looking fill.
        """
        inpaint_by_block = False
        check_need_inpaint = True

        params = {
            "model_name": {
                "type": "line_editor",
                "value": "runwayml/stable-diffusion-inpainting",
                "description": "Diffusers model id (e.g. runwayml/stable-diffusion-inpainting).",
            },
            "device": DEVICE_SELECTOR(),
            "inpaint_size": {
                "type": "line_editor",
                "value": 512,
                "description": "Size for SD (512 or 768; smaller = faster, lower quality).",
            },
            "prompt": {
                "type": "line_editor",
                "value": "clean background, solid color, no text",
                "description": "Prompt for inpainting (neutral for text removal).",
            },
            "negative_prompt": {
                "type": "line_editor",
                "value": "text, letters, words, watermark",
                "description": "Negative prompt.",
            },
            "num_inference_steps": {
                "type": "line_editor",
                "value": 25,
                "description": "Denoising steps (20–50).",
            },
            "description": "Stable Diffusion inpainting (Diffusers). Install: pip install diffusers accelerate",
        }
        _load_model_keys = {"pipeline"}

        def __init__(self, **params) -> None:
            super().__init__(**params)
            self.device = self.params["device"]["value"]
            self.pipeline = None
            self._model_name = None

        def _load_model(self):
            model_name = (self.params.get("model_name") or {}).get("value", "runwayml/stable-diffusion-inpainting") or "runwayml/stable-diffusion-inpainting"
            if self.pipeline is not None and self._model_name == model_name:
                return
            self._model_name = model_name
            import torch
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(model_name, torch_dtype=dtype)
            self.pipeline = self.pipeline.to(self.device)
            if self.device == "cuda":
                try:
                    self.pipeline.enable_attention_slicing()
                except Exception:
                    pass

        def _inpaint(self, img: np.ndarray, mask: np.ndarray, textblock_list: List[TextBlock] = None) -> np.ndarray:
            if img.ndim == 3 and img.shape[2] == 4:
                img = img[:, :, :3]
            if mask.ndim == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask_bin = (mask > 127).astype(np.uint8) * 255
            h, w = img.shape[:2]
            try:
                size = 512
                vs = self.params.get("inpaint_size", {})
                if isinstance(vs, dict):
                    try:
                        size = max(256, min(768, int(vs.get("value", 512))))
                    except (TypeError, ValueError):
                        pass
                pil_img = Image.fromarray(img)
                pil_mask = Image.fromarray(mask_bin).convert("L")
                if max(h, w) > size or min(h, w) != size:
                    scale = size / max(h, w)
                    new_w = max(64, int(w * scale))
                    new_h = max(64, int(h * scale))
                    pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    pil_mask = pil_mask.resize((new_w, new_h), Image.Resampling.NEAREST)
                prompt = (self.params.get("prompt") or {}).get("value", "clean background, no text") or "clean background, no text"
                negative = (self.params.get("negative_prompt") or {}).get("value", "text, letters") or "text, letters"
                steps = 25
                st = self.params.get("num_inference_steps", {})
                if isinstance(st, dict):
                    try:
                        steps = max(10, min(50, int(st.get("value", 25))))
                    except (TypeError, ValueError):
                        pass
                out = self.pipeline(
                    prompt=prompt,
                    image=pil_img,
                    mask_image=pil_mask,
                    negative_prompt=negative,
                    num_inference_steps=steps,
                ).images[0]
                out = np.array(out)
                if out.shape[0] != h or out.shape[1] != w:
                    out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LANCZOS4)
                return out.astype(np.uint8)
            except Exception as e:
                self.logger.error(f"Diffusers inpainting failed: {e}")
                return img.copy()

        def updateParam(self, param_key: str, param_content):
            super().updateParam(param_key, param_content)
            if param_key == "device":
                self.device = self.params["device"]["value"]
                if self.pipeline is not None:
                    self.pipeline = self.pipeline.to(self.device)
            elif param_key == "model_name":
                self.pipeline = None
                self._model_name = None
