"""
Optional inpainter using Kandinsky 2.1 Inpainting via Hugging Face Diffusers.
Install: pip install diffusers transformers accelerate
Model: kandinsky-community/kandinsky-2-1-inpaint (CLIP + diffusion prior; white mask = fill).
"""
import numpy as np
import cv2
from typing import List

from ..base import DEVICE_SELECTOR
from .base import InpainterBase, register_inpainter, TextBlock

_KANDINSKY_AVAILABLE = False
try:
    from diffusers.pipelines.kandinsky.pipeline_kandinsky_inpaint import KandinskyInpaintPipeline
    import torch
    from PIL import Image
    _KANDINSKY_AVAILABLE = True
except ImportError:
    import logging
    logging.getLogger("BallonTranslator").debug(
        "Kandinsky inpainting not available. Install: pip install diffusers transformers accelerate"
    )


if _KANDINSKY_AVAILABLE:

    @register_inpainter("kandinsky_inpaint")
    class KandinskyInpainter(InpainterBase):
        """
        Kandinsky 2.1 inpainting via Diffusers. White mask = region to fill.
        Alternative to SD inpainting; good for text removal with neutral prompt.
        """
        inpaint_by_block = False
        check_need_inpaint = True

        params = {
            "model_name": {
                "type": "line_editor",
                "value": "kandinsky-community/kandinsky-2-1-inpaint",
                "description": "Diffusers model id (Kandinsky 2.1 inpainting).",
            },
            "device": DEVICE_SELECTOR(),
            "inpaint_size": {
                "type": "line_editor",
                "value": 512,
                "description": "Max side length (512–768; smaller = less VRAM).",
            },
            "prompt": {
                "type": "line_editor",
                "value": "clean background, no text, solid color",
                "description": "Prompt for filled region.",
            },
            "negative_prompt": {
                "type": "line_editor",
                "value": "text, letters, words",
                "description": "Negative prompt.",
            },
            "num_inference_steps": {
                "type": "line_editor",
                "value": 25,
                "description": "Denoising steps (20–50).",
            },
            "guidance_scale": {
                "type": "line_editor",
                "value": 7.0,
                "description": "Guidance scale (e.g. 5–10).",
            },
            "description": "Kandinsky 2.1 inpainting (Diffusers). Install: pip install diffusers accelerate",
        }
        _load_model_keys = {"pipeline"}

        def __init__(self, **params) -> None:
            super().__init__(**params)
            self.device = self.params["device"]["value"]
            self.pipeline = None
            self._model_name = None

        def _load_model(self):
            model_name = (self.params.get("model_name") or {}).get("value", "kandinsky-community/kandinsky-2-1-inpaint") or "kandinsky-community/kandinsky-2-1-inpaint"
            if self.pipeline is not None and self._model_name == model_name:
                return
            self._model_name = model_name
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self.pipeline = KandinskyInpaintPipeline.from_pretrained(model_name, torch_dtype=dtype)
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
                if max(h, w) > size:
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
                guidance = 7.0
                gs = self.params.get("guidance_scale", {})
                if isinstance(gs, dict):
                    try:
                        guidance = max(1.0, min(20.0, float(gs.get("value", 7.0))))
                    except (TypeError, ValueError):
                        pass
                out = self.pipeline(
                    prompt=prompt,
                    image=pil_img,
                    mask_image=pil_mask,
                    negative_prompt=negative,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                ).images[0]
                out = np.array(out)
                if out.shape[0] != h or out.shape[1] != w:
                    out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LANCZOS4)
                return out.astype(np.uint8)
            except Exception as e:
                self.logger.error(f"Kandinsky inpainting failed: {e}")
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
