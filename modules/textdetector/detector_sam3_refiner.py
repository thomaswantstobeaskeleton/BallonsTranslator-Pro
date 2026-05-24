"""
SAM 3 mask refinement detector.

This module behaves like a normal BallonsTranslator text detector in the UI:
users pick `sam3_refiner`, choose a curated SAM refinement model, choose a base
text detector, and the normal config persistence path stores the parameters.

Flow:
1. Run a curated base detector (CTD/YOLO/Paddle/etc.).
2. Run SAM 3 with a prompt such as "speech bubble".
3. Filter SAM candidates against the base detector mask.
4. Merge the refined mask and attach safe-area metadata for auto-layout.

It is disabled-by-default at the app level because users must explicitly select
this detector. No global detector behavior changes.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .base import TEXTDETECTORS, register_textdetectors, TextDetectorBase, TextBlock, DEVICE_SELECTOR
from utils.sam3_refinement import (
    SAM3_REFINEMENT_BASE_DETECTOR_OPTIONS,
    SAM3_REFINEMENT_MODEL_CUSTOM,
    curated_sam3_refinement_models,
    resolve_refinement_model_id,
    SAM3RefinementOptions,
    refine_mask_with_sam3,
)


@register_textdetectors("sam3_refiner")
class SAM3RefinerDetector(TextDetectorBase):
    """Base-detector + SAM 3 refinement wrapper.

    This is intentionally a wrapper detector instead of a global hidden hook so
    the feature has the same selection/config persistence behavior as other
    detectors.
    """

    params = {
        "base_detector": {
            "type": "selector",
            "options": list(SAM3_REFINEMENT_BASE_DETECTOR_OPTIONS),
            "value": "ctd",
            "description": "Detector to run first. Use CTD for default manga, hf_object_det/ysgyolo for comic bubble models, or Paddle for Chinese/general text.",
        },
        "refinement_model": {
            "type": "selector",
            "options": list(curated_sam3_refinement_models()),
            "value": "facebook/sam3",
            "description": "Curated models that actually help mask refinement. Use Custom only for SAM 3-compatible models.",
        },
        "custom_model_id": {
            "type": "line_editor",
            "value": "",
            "description": "Used only when refinement_model is Custom SAM 3 model id. Must be SAM 3-compatible.",
        },
        "prompt": {
            "type": "selector",
            "options": ["speech bubble", "text", "caption", "sound effect", "panel"],
            "value": "speech bubble",
            "description": "SAM 3 text prompt. speech bubble is best for bubble safe areas; text can refine text masks.",
        },
        "fallback_prompt": {
            "type": "selector",
            "options": ["text", "speech bubble", "caption", "sound effect", ""],
            "value": "text",
            "description": "Prompt used if the primary prompt returns no usable mask.",
        },
        "merge_mode": {
            "type": "selector",
            "options": ["union", "intersect", "replace_if_nonempty"],
            "value": "union",
            "description": "How to combine base detector mask and SAM mask. union = safer coverage; intersect = strict cleanup; replace_if_nonempty = whole SAM bubble.",
        },
        "min_iou_with_initial": {
            "type": "line_editor",
            "value": 0.02,
            "description": "Reject SAM candidates with IoU below this against the base detector mask. Lower catches more, higher is safer.",
        },
        "min_mask_area": {
            "type": "line_editor",
            "value": 64,
            "description": "Reject SAM candidates smaller than this area in pixels.",
        },
        "max_mask_area_ratio": {
            "type": "line_editor",
            "value": 0.75,
            "description": "Reject candidates larger than this fraction of the page.",
        },
        "expand_px": {
            "type": "line_editor",
            "value": 0,
            "description": "Optional final mask dilation in pixels after merging.",
        },
        "safe_rect_min_coverage": {
            "type": "line_editor",
            "value": 0.88,
            "description": "Minimum bubble-mask coverage used when deriving auto-layout safe rectangles.",
        },
        "attach_safe_areas": {
            "type": "checkbox",
            "value": True,
            "description": "Attach SAM-derived safe-area metadata to detected blocks for future auto-layout/bubble fitting.",
        },
        "device": DEVICE_SELECTOR(),
        "description": "Run CTD/YOLO/Paddle first, then refine the mask with a curated SAM 3 model. Normal detector settings are persisted like other modules.",
    }
    _load_model_keys = {"_sam_detector"}

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self._base_detector = None
        self._base_detector_key = None
        self._sam_detector = None
        self._sam_detector_key = None
        self._sam_model_id = None
        self._sam_device = None

    def _param(self, key: str, default=None):
        try:
            return self.get_param_value(key)
        except Exception:
            return default

    def _base_params(self, detector_key: str) -> dict:
        try:
            from utils.config import pcfg
            return (getattr(pcfg.module, "textdetector_params", {}) or {}).get(detector_key, {}) or {}
        except Exception:
            return {}

    def _make_detector(self, detector_key: str, params: dict | None = None):
        if detector_key not in TEXTDETECTORS.module_dict:
            raise KeyError(detector_key)
        cls = TEXTDETECTORS.module_dict[detector_key]
        return cls(**(params or {}))

    def _load_base_detector(self):
        detector_key = str(self._param("base_detector", "ctd") or "ctd").strip()
        if detector_key == "sam3_refiner":
            detector_key = "ctd"
        if detector_key not in SAM3_REFINEMENT_BASE_DETECTOR_OPTIONS:
            self.logger.warning("SAM3 refiner rejected unsupported base detector %s; using ctd", detector_key)
            detector_key = "ctd"
        if self._base_detector is not None and self._base_detector_key == detector_key:
            return
        self._base_detector_key = detector_key
        self._base_detector = self._make_detector(detector_key, self._base_params(detector_key))
        self._base_detector.load_model()

    def _load_sam_detector(self):
        choice = str(self._param("refinement_model", "facebook/sam3") or "facebook/sam3")
        custom = str(self._param("custom_model_id", "") or "")
        model_id = resolve_refinement_model_id(choice, custom)
        device = str(self._param("device", "cpu") or "cpu")
        if device == "Default":
            device = ""
        if self._sam_detector is not None and self._sam_model_id == model_id and self._sam_device == device:
            return
        if "sam_text_det" not in TEXTDETECTORS.module_dict:
            raise RuntimeError("sam_text_det is unavailable. Install transformers/torch with SAM 3 support and accept the model access terms if required.")
        sam_cls = TEXTDETECTORS.module_dict["sam_text_det"]
        self._sam_model_id = model_id
        self._sam_device = device
        self._sam_detector_key = "sam_text_det"
        self._sam_detector = sam_cls(
            model_id=model_id,
            text_prompt=str(self._param("prompt", "speech bubble") or "speech bubble"),
            device=device or "cpu",
            box_padding=0,
            min_area_px=int(float(self._param("min_mask_area", 64) or 64)),
            max_area_ratio=float(self._param("max_mask_area_ratio", 0.75) or 0.75),
        )
        self._sam_detector.load_model()

    def _load_model(self):
        self._load_base_detector()
        self._load_sam_detector()

    def _sam_box_runner(self, image: np.ndarray, prompt: str):
        self._load_sam_detector()
        if self._sam_detector is None:
            return []
        if hasattr(self._sam_detector, "set_param_value"):
            self._sam_detector.set_param_value("text_prompt", prompt)
        _mask, blocks = self._sam_detector.detect(image, None)
        return [getattr(blk, "xyxy", []) for blk in (blocks or [])]

    def _opts(self) -> SAM3RefinementOptions:
        return SAM3RefinementOptions(
            enabled=True,
            prompt=str(self._param("prompt", "speech bubble") or "speech bubble"),
            fallback_prompt=str(self._param("fallback_prompt", "text") or "text"),
            merge_mode=str(self._param("merge_mode", "union") or "union"),
            min_mask_area=int(float(self._param("min_mask_area", 64) or 64)),
            max_mask_area_ratio=float(self._param("max_mask_area_ratio", 0.75) or 0.75),
            min_iou_with_initial=float(self._param("min_iou_with_initial", 0.02) or 0.02),
            expand_px=int(float(self._param("expand_px", 0) or 0)),
            safe_rect_min_coverage=float(self._param("safe_rect_min_coverage", 0.88) or 0.88),
        )

    def _detect(self, img: np.ndarray, proj=None) -> Tuple[np.ndarray, List[TextBlock]]:
        self._load_base_detector()
        mask, blocks = self._base_detector.detect(img, proj)
        opts = self._opts()
        result = refine_mask_with_sam3(img, mask, opts, box_runner=self._sam_box_runner)
        attach = bool(self._param("attach_safe_areas", True))
        if attach and blocks and result.safe_areas:
            meta = result.to_dict()
            for blk in blocks:
                try:
                    rid = getattr(blk, "region_inpaint_dict", None) or {}
                    rid["sam3_refinement"] = meta
                    blk.region_inpaint_dict = rid
                except Exception:
                    pass
        for blk in blocks or []:
            try:
                blk.det_model = f"{self.name}:{self._base_detector_key}+sam3"
            except Exception:
                pass
        return result.mask, blocks or []

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        if param_key == "base_detector":
            self._base_detector = None
            self._base_detector_key = None
        if param_key in {"refinement_model", "custom_model_id", "device"}:
            self._sam_detector = None
            self._sam_model_id = None
            self._sam_device = None
