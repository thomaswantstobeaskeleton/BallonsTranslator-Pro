"""
DreamShaper inpainting – SD inpainting via Hugging Face Diffusers.
Not recommended for manga/comic text removal: diffusion models often fill masks with
squiggly or text-like artifacts. Use LaMa (lama_large_512px, simple_lama, lama_mpe) instead.
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
        "DreamShaper inpainting not available. Install: pip install diffusers transformers accelerate"
    )


if _DIFFUSERS_AVAILABLE:

    @register_inpainter("dreamshaper_inpaint")
    class DreamShaperInpainter(InpainterBase):
        """
        DreamShaper 8 inpainting via Diffusers. Not recommended for text removal (use LaMa).
        """
        inpaint_by_block = False
        check_need_inpaint = True

        params = {
            "model_name": {
                "type": "line_editor",
                "value": "Lykon/dreamshaper-8-inpainting",
                "description": "DreamShaper inpainting model (512).",
            },
            "device": DEVICE_SELECTOR(),
            "inpaint_size": {
                "type": "line_editor",
                "value": 512,
                "description": "Max side (512 default; 768 for higher quality).",
            },
            "prompt": {
                "type": "line_editor",
                "value": "manhua Chinese comic panel, speech bubble interior only, empty blank white space inside bubble, uniform flat solid color fill, seamless inpainting, clean smooth surface, no text no letters no writing, no strokes no lines no marks, pure empty bubble area matching original bubble background color, soft white or off-white fill, preserve bubble shape and black outline, comic book manga style panel, no details no texture no patterns inside bubble, continuous smooth color, professional clean result",
                "description": "Prompt: manhua + speech bubble context + empty, uniform fill.",
            },
            "negative_prompt": {
                "type": "line_editor",
                "value": "text, letters, words, numbers, handwriting, font, typeface, alphabet, Chinese characters, Hanzi, Simplified Chinese, Traditional Chinese, Japanese characters, Korean characters, CJK, symbols, punctuation, writing, scribbles, squiggles, strokes, pen marks, ink lines, brush strokes, doodles, graffiti, watermark, logo, signature, subtitle, caption, sound effect, onomatopoeia, screentone, halftone dots, crosshatching, speed lines, manga effects, distorted, deformed, garbled, blurry, noisy, artifacts, jagged edges, color bleed, wrong color, grey box, colored patch, uneven fill, visible seam, leftover text, ghost text, faded letters, double exposure, duplicate shapes, extra bubbles, mutilated bubble, broken outline, low quality, jpeg artifacts, pixelated",
                "description": "Negative: all text, strokes, comic effects, and inpainting artifacts.",
            },
            "num_inference_steps": {
                "type": "line_editor",
                "value": 35,
                "description": "Denoising steps. Higher = cleaner fill, slower. 30–40 for manhua bubbles.",
            },
            "strength": {
                "type": "line_editor",
                "value": 0.99,
                "description": "Inpaint strength. Keep 0.95–1.0 to fully replace text. Lower = more original kept (risk leftover text).",
            },
            "guidance_scale": {
                "type": "line_editor",
                "value": 8.0,
                "description": "Prompt adherence. 7–9 for speech bubbles; higher = stronger 'empty, no text'; too high can cause artifacts.",
            },
            "description": "DreamShaper inpainting (Diffusers). Not for text removal—use LaMa (lama_large_512px, simple_lama).",
        }
        _load_model_keys = {"pipeline"}

        def __init__(self, **params) -> None:
            super().__init__(**params)
            self.device = self.params["device"]["value"]
            self.pipeline = None
            self._model_name = None

        def _load_model(self):
            model_name = (self.params.get("model_name") or {}).get("value", "Lykon/dreamshaper-8-inpainting") or "Lykon/dreamshaper-8-inpainting"
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
                _prompt_default = "manhua Chinese comic speech bubble interior, empty blank area, uniform flat color, no text, seamless clean fill"
                _neg_default = "text, letters, words, scribbles, squiggles, strokes, symbols, watermark, distorted, blurry, artifacts, screentone, halftone"
                prompt = (self.params.get("prompt") or {}).get("value", _prompt_default) or _prompt_default
                negative = (self.params.get("negative_prompt") or {}).get("value", _neg_default) or _neg_default
                steps = 35
                st = self.params.get("num_inference_steps", {})
                if isinstance(st, dict):
                    try:
                        steps = max(15, min(50, int(st.get("value", 35))))
                    except (TypeError, ValueError):
                        pass
                strength = 0.99
                sval = self.params.get("strength", {})
                if isinstance(sval, dict):
                    try:
                        strength = max(0.8, min(1.0, float(sval.get("value", 0.99))))
                    except (TypeError, ValueError):
                        pass
                guidance = 8.0
                gval = self.params.get("guidance_scale", {})
                if isinstance(gval, dict):
                    try:
                        guidance = max(1.0, min(20.0, float(gval.get("value", 8.0))))
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
                self.logger.error(f"DreamShaper inpainting failed: {e}")
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
