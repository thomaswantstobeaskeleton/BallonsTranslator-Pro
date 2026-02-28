"""
Fluently v4 inpainting – anime/comic-optimized SD inpainting via Hugging Face Diffusers.
Model: fluently/Fluently-v4-inpainting (based on runwayml/stable-diffusion-inpainting).
Good for anime and comic book style inpainting; use LaMa (lama_large_512px) for fastest text removal.
Install: pip install diffusers transformers accelerate
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
        "Fluently inpainting not available. Install: pip install diffusers transformers accelerate"
    )


if _DIFFUSERS_AVAILABLE:

    @register_inpainter("fluently_v4_inpaint")
    class FluentlyV4Inpainter(InpainterBase):
        """
        Fluently v4 inpainting via Diffusers. Tuned for anime & comic book style.
        Small parts and complex objects; good for manga speech bubbles.
        """
        inpaint_by_block = False
        check_need_inpaint = True

        params = {
            "model_name": {
                "type": "line_editor",
                "value": "fluently/Fluently-v4-inpainting",
                "description": "Fluently inpainting model (512; anime/comic).",
            },
            "device": DEVICE_SELECTOR(),
            "inpaint_size": {
                "type": "line_editor",
                "value": 512,
                "description": "Max side (512 default; 20–30 steps recommended).",
            },
            "prompt": {
                "type": "line_editor",
                "value": "manga comic panel, speech bubble interior, empty blank white space, uniform flat color, no text, no letters, seamless inpainting, clean smooth surface, anime comic book style, preserve bubble outline",
                "description": "Prompt for anime/comic bubble fill.",
            },
            "negative_prompt": {
                "type": "line_editor",
                "value": "text, letters, words, handwriting, strokes, scribbles, watermark, screentone, halftone, distorted, blurry, artifacts",
                "description": "Negative prompt.",
            },
            "num_inference_steps": {
                "type": "line_editor",
                "value": 25,
                "description": "Steps (20–30 recommended by author).",
            },
            "strength": {
                "type": "line_editor",
                "value": 0.99,
                "description": "Inpaint strength (0.95–1.0 to fully replace text).",
            },
            "guidance_scale": {
                "type": "line_editor",
                "value": 6.0,
                "description": "CFG scale (5–7 recommended by author).",
            },
            "description": "Fluently v4 inpainting (Diffusers). Anime/comic style; install diffusers.",
        }
        _load_model_keys = {"pipeline"}

        def __init__(self, **params) -> None:
            super().__init__(**params)
            self.device = self.params["device"]["value"]
            self.pipeline = None
            self._model_name = None

        def _load_model(self):
            model_name = (self.params.get("model_name") or {}).get("value", "fluently/Fluently-v4-inpainting") or "fluently/Fluently-v4-inpainting"
            if self.pipeline is not None and self._model_name == model_name:
                return
            self._model_name = model_name
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
                if max(h, w) != size:
                    scale = size / max(h, w)
                    new_w = max(64, int(w * scale))
                    new_h = max(64, int(h * scale))
                    pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    pil_mask = pil_mask.resize((new_w, new_h), Image.Resampling.NEAREST)
                prompt = (self.params.get("prompt") or {}).get("value", "") or "manga speech bubble empty no text"
                negative = (self.params.get("negative_prompt") or {}).get("value", "") or "text, letters"
                steps = 25
                st = self.params.get("num_inference_steps", {})
                if isinstance(st, dict):
                    try:
                        steps = max(15, min(50, int(st.get("value", 25))))
                    except (TypeError, ValueError):
                        pass
                strength = 0.99
                sval = self.params.get("strength", {})
                if isinstance(sval, dict):
                    try:
                        strength = max(0.8, min(1.0, float(sval.get("value", 0.99))))
                    except (TypeError, ValueError):
                        pass
                guidance = 6.0
                gval = self.params.get("guidance_scale", {})
                if isinstance(gval, dict):
                    try:
                        guidance = max(1.0, min(20.0, float(gval.get("value", 6.0))))
                    except (TypeError, ValueError):
                        pass
                out = self.pipeline(
                    prompt=prompt,
                    image=pil_img,
                    mask_image=pil_mask,
                    negative_prompt=negative,
                    num_inference_steps=steps,
                    strength=strength,
                    guidance_scale=guidance,
                ).images[0]
                out = np.array(out)
                if out.shape[0] != h or out.shape[1] != w:
                    out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LANCZOS4)
                return out.astype(np.uint8)
            except Exception as e:
                self.logger.error(f"Fluently v4 inpainting failed: {e}")
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
