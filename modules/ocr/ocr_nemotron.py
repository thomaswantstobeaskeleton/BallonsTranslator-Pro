"""
Nemotron Parse v1.1 – Full-page document OCR with bboxes and classes (NVIDIA).
Runs on full image; assigns text to blocks by bbox overlap. Requires: pip install transformers accelerate torch albumentations timm; trust_remote_code.
Model: nvidia/NVIDIA-Nemotron-Parse-v1.1. Min resolution 1024×1280; best for documents, not per-crop manga.
"""
import os
import re
from typing import List, Tuple
import numpy as np
import cv2

from .base import OCRBase, register_OCR, DEFAULT_DEVICE, DEVICE_SELECTOR, TextBlock


def _cv2_to_pil(img: np.ndarray):
    from PIL import Image as PILImage
    if img.ndim == 2:
        return PILImage.fromarray(img).convert("RGB")
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return PILImage.fromarray(img)


def _iou_rect(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@register_OCR("nemotron_parse")
class NemotronParseOCR(OCRBase):
    """
    Nemotron Parse v1.1: full-page document OCR with bboxes and classes.
    Runs once per image; assigns text to blocks by bbox overlap. Best for documents (min 1024×1280).
    """
    params = {
        "model_id": {
            "type": "line_editor",
            "value": "nvidia/NVIDIA-Nemotron-Parse-v1.1",
            "description": "Hugging Face model id.",
        },
        "device": DEVICE_SELECTOR(),
        "min_resolution": {
            "type": "line_editor",
            "value": "1024",
            "description": "Min side for image (model expects 1024×1280–1648×2048).",
        },
        "iou_threshold": {
            "type": "line_editor",
            "value": "0.2",
            "description": "Min IoU to assign a parsed bbox to a block (0.1–0.5).",
        },
        "description": "Nemotron Parse full-page OCR (bbox + class). Install: transformers accelerate torch albumentations timm.",
    }
    _load_model_keys = {"model", "processor", "tokenizer"}
    optional_install_hint = "pip install transformers accelerate albumentations timm"

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.model = None
        self.processor = None
        self.tokenizer = None
        self._model_id = None
        self._device = None
        self._extract_classes_bboxes = None
        self._transform_bbox_to_original = None

    def _load_model(self):
        import torch
        from transformers import AutoModel, AutoProcessor, AutoTokenizer, GenerationConfig
        from PIL import Image as PILImage
        model_id = (self.params.get("model_id") or {}).get("value", "nvidia/NVIDIA-Nemotron-Parse-v1.1")
        device = (self.params.get("device") or {}).get("value", "cuda")
        if device == "gpu":
            device = "cuda"
        if not torch.cuda.is_available() and device == "cuda":
            device = "cpu"
        if self.model is not None and self._model_id == model_id and self._device == device:
            return
        try:
            from utils.model_manager import get_model_manager
            post_path = get_model_manager().hf_hub_download("nvidia/NVIDIA-Nemotron-Parse-v1.1", "postprocessing.py")
            import importlib.util
            spec = importlib.util.spec_from_file_location("nemotron_postprocessing", post_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self._extract_classes_bboxes = getattr(mod, "extract_classes_bboxes", None)
            self._transform_bbox_to_original = getattr(mod, "transform_bbox_to_original", None)
        except Exception as e:
            raise RuntimeError(
                "Nemotron Parse: could not load postprocessing from nvidia/NVIDIA-Nemotron-Parse-v1.1. "
                "Install: pip install transformers accelerate torch albumentations timm; ensure huggingface_hub is available."
            ) from e
        if not self._extract_classes_bboxes or not self._transform_bbox_to_original:
            raise RuntimeError(
                "Nemotron Parse: postprocessing.py did not expose extract_classes_bboxes/transform_bbox_to_original. "
                "Check the model repo nvidia/NVIDIA-Nemotron-Parse-v1.1."
            )
        self._model_id = model_id
        self._device = device
        self.model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        ).to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    def _run_page(self, img: np.ndarray) -> Tuple[List[str], List[Tuple[float, float, float, float]], List[str]]:
        import torch
        from transformers import GenerationConfig
        from PIL import Image as PILImage
        pil = _cv2_to_pil(img)
        w, h = pil.size
        min_res = int((self.params.get("min_resolution") or {}).get("value", "1024"))
        if min(w, h) < min_res:
            scale = min_res / min(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            pil = pil.resize((new_w, new_h), PILImage.Resampling.LANCZOS)
            w, h = pil.size
        task_prompt = "</s><s><predict_bbox><predict_classes><output_markdown>"
        inputs = self.processor(
            images=[pil],
            text=task_prompt,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(self._device)
        gen_config = GenerationConfig.from_pretrained(self._model_id, trust_remote_code=True)
        with torch.no_grad():
            out = self.model.generate(**inputs, generation_config=gen_config)
        generated = self.processor.batch_decode(out, skip_special_tokens=True)[0]
        classes, bboxes, texts = self._extract_classes_bboxes(generated)
        orig_bboxes = []
        for bbox in bboxes:
            try:
                ob = self._transform_bbox_to_original(bbox, w, h)
                orig_bboxes.append((float(ob[0]), float(ob[1]), float(ob[2]), float(ob[3])))
            except Exception:
                orig_bboxes.append((0.0, 0.0, 0.0, 0.0))
        return classes, orig_bboxes, texts

    def ocr_img(self, img: np.ndarray) -> str:
        if not self.all_model_loaded():
            self.load_model()
        _, _, texts = self._run_page(img)
        return "\n".join(texts) if texts else ""

    def _ocr_blk_list(self, img: np.ndarray, blk_list: List[TextBlock], *args, **kwargs) -> None:
        if not self.all_model_loaded():
            self.load_model()
        im_h, im_w = img.shape[:2]
        _, bboxes, texts = self._run_page(img)
        iou_thr = 0.2
        try:
            iou_thr = max(0.05, min(0.9, float((self.params.get("iou_threshold") or {}).get("value", "0.2"))))
        except (TypeError, ValueError):
            pass
        for blk in blk_list:
            x1, y1, x2, y2 = blk.xyxy
            blk_rect = (float(x1), float(y1), float(x2), float(y2))
            best_iou = 0.0
            best_text = ""
            for (bx1, by1, bx2, by2), txt in zip(bboxes, texts):
                iou = _iou_rect(blk_rect, (bx1, by1, bx2, by2))
                if iou >= iou_thr and iou > best_iou:
                    best_iou = iou
                    best_text = (txt or "").strip()
            blk.text = [best_text if best_text else ""]
