"""
Optional inpainter using FLUX.1-Fill (Black Forest Labs) via Diffusers.
Install: pip install diffusers transformers accelerate
Best for: high-quality inpainting/outpainting; no strength parameter.
Model: black-forest-labs/FLUX.1-Fill-dev (12B params; use GPU with sufficient VRAM or CPU offload).
"""
import numpy as np
import cv2
from typing import List

from ..base import DEVICE_SELECTOR
from .base import InpainterBase, register_inpainter, TextBlock
from utils.logger import logger as LOGGER

_FLUX_FILL_AVAILABLE = False
try:
    from diffusers import FluxFillPipeline
    import torch
    from PIL import Image
    _FLUX_FILL_AVAILABLE = True
except ImportError:
    import logging
    logging.getLogger("BallonTranslator").debug(
        "FLUX Fill not available. Install: pip install diffusers transformers accelerate"
    )


if _FLUX_FILL_AVAILABLE:

    @register_inpainter("flux_fill")
    class FluxFillInpainter(InpainterBase):
        """
        FLUX.1-Fill inpainting via Diffusers. High quality; 12B params.
        Use CPU offload if VRAM is limited. No strength parameter.
        """
        inpaint_by_block = False
        check_need_inpaint = True

        params = {
            "model_name": {
                "type": "line_editor",
                "value": "black-forest-labs/FLUX.1-Fill-dev",
                "description": "Diffusers model id.",
            },
            "device": DEVICE_SELECTOR(),
            "max_size": {
                "type": "line_editor",
                "value": 1024,
                "description": "Max side length (smaller = less VRAM; Flux supports variable size).",
            },
            "prompt": {
                "type": "line_editor",
                "value": "clean background, no text, solid color",
                "description": "Prompt for filled region (text removal: neutral description).",
            },
            "guidance_scale": {
                "type": "line_editor",
                "value": 30,
                "description": "Guidance scale (e.g. 20–40).",
            },
            "num_inference_steps": {
                "type": "line_editor",
                "value": 50,
                "description": "Denoising steps (28–50).",
            },
            "use_cpu_offload": {
                "type": "checkbox",
                "value": False,
                "description": "Offload to CPU to save VRAM (slower).",
            },
            "description": "FLUX.1-Fill inpainting (Diffusers). Install: pip install diffusers accelerate",
        }
        _load_model_keys = {"pipeline"}

        def __init__(self, **params) -> None:
            super().__init__(**params)
            self.device = self.params["device"]["value"]
            self.pipeline = None
            self._model_name = None

        def _load_model(self):
            model_name = (self.params.get("model_name") or {}).get("value", "black-forest-labs/FLUX.1-Fill-dev") or "black-forest-labs/FLUX.1-Fill-dev"
            if self.pipeline is not None and self._model_name == model_name:
                return
            self._model_name = model_name
            LOGGER.info("Loading FLUX.1-Fill model (this can take several minutes on first run)...")
            dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
            try:
                if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
                    dtype = torch.float16
            except Exception:
                dtype = torch.float16 if self.device == "cuda" else torch.float32
            try:
                self.pipeline = FluxFillPipeline.from_pretrained(model_name, torch_dtype=dtype)
            except Exception as e:
                err_str = str(e).lower()
                if "gated" in err_str or "401" in err_str or "access" in err_str or "authenticated" in err_str:
                    raise RuntimeError(
                        "FLUX.1-Fill is a gated model. Log in to Hugging Face (huggingface-cli login or set HF_TOKEN), "
                        "accept the model terms at https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev, "
                        "then retry. Or use another inpainter (e.g. lama_large_512px or simple_lama)."
                    ) from e
                raise
            if self.params.get("use_cpu_offload", {}).get("value", False):
                self.pipeline.enable_model_cpu_offload()
            else:
                self.pipeline = self.pipeline.to(self.device)
            LOGGER.info("FLUX.1-Fill model loaded.")

        def _inpaint(self, img: np.ndarray, mask: np.ndarray, textblock_list: List[TextBlock] = None) -> np.ndarray:
            if img.ndim == 3 and img.shape[2] == 4:
                img = img[:, :, :3]
            if mask.ndim == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask_bin = (mask > 127).astype(np.uint8) * 255
            orig_h, orig_w = img.shape[:2]
            try:
                max_side = 1024
                ms = self.params.get("max_size", {})
                if isinstance(ms, dict):
                    try:
                        max_side = max(256, min(2048, int(ms.get("value", 1024))))
                    except (TypeError, ValueError):
                        pass
                h, w = orig_h, orig_w
                if max(h, w) > max_side:
                    scale = max_side / max(h, w)
                    w, h = int(w * scale), int(h * scale)
                    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LANCZOS4)
                    mask_bin = cv2.resize(mask_bin, (w, h), interpolation=cv2.INTER_NEAREST)
                pil_img = Image.fromarray(img)
                pil_mask = Image.fromarray(mask_bin).convert("L")
                prompt = (self.params.get("prompt") or {}).get("value", "clean background, no text") or "clean background, no text"
                guidance = 30
                gs = self.params.get("guidance_scale", {})
                if isinstance(gs, dict):
                    try:
                        guidance = max(1.0, min(100, float(gs.get("value", 30))))
                    except (TypeError, ValueError):
                        pass
                steps = 50
                st = self.params.get("num_inference_steps", {})
                if isinstance(st, dict):
                    try:
                        steps = max(20, min(100, int(st.get("value", 50))))
                    except (TypeError, ValueError):
                        pass
                out = self.pipeline(
                    prompt=prompt,
                    image=pil_img,
                    mask_image=pil_mask,
                    height=h,
                    width=w,
                    guidance_scale=guidance,
                    num_inference_steps=steps,
                    max_sequence_length=512,
                ).images[0]
                out = np.array(out)
                if out.shape[0] != orig_h or out.shape[1] != orig_w:
                    out = cv2.resize(out, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)
                return out.astype(np.uint8)
            except Exception as e:
                self.logger.error(f"FLUX Fill inpainting failed: {e}")
                return img.copy()

        def updateParam(self, param_key: str, param_content):
            super().updateParam(param_key, param_content)
            if param_key == "device":
                self.device = self.params["device"]["value"]
                if self.pipeline is not None and not self.params.get("use_cpu_offload", {}).get("value", False):
                    self.pipeline = self.pipeline.to(self.device)
            elif param_key == "model_name":
                self.pipeline = None
                self._model_name = None
            elif param_key == "use_cpu_offload" and self.pipeline is not None:
                if self.params.get("use_cpu_offload", {}).get("value", False):
                    self.pipeline.enable_model_cpu_offload()
                else:
                    self.pipeline = self.pipeline.to(self.device)
