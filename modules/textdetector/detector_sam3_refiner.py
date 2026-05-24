from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .base import TEXTDETECTORS, register_textdetectors, TextDetectorBase, TextBlock, DEVICE_SELECTOR
from utils.sam3_refinement import (
    SAM3_REFINEMENT_MODEL_CUSTOM,
    all_base_detector_options,
    curated_refiner_detector_options,
    curated_sam3_refinement_models,
    resolve_refinement_model_id,
    SAM3RefinementOptions,
    refine_mask_with_sam3,
)


def _registered_detector_names() -> list[str]:
    return [k for k in TEXTDETECTORS.module_dict.keys() if k != "sam3_refiner"]


@register_textdetectors("sam3_refiner")
class SAM3RefinerDetector(TextDetectorBase):
    """Detector wrapper: run any base detector, then a curated mask refiner."""

    params = {
        "base_detector": {
            "type": "selector",
            "options": list(all_base_detector_options()),
            "value": "ctd",
            "description": "First-pass detector. The UI refreshes this to all registered detectors except sam3_refiner.",
        },
        "refiner_detector": {
            "type": "selector",
            "options": list(curated_refiner_detector_options()),
            "value": "sam_text_det",
            "description": "Second-pass detector used only for mask refinement. Curated to detectors likely to improve bubble/text masks.",
        },
        "refinement_model": {
            "type": "selector",
            "options": list(curated_sam3_refinement_models()),
            "value": "facebook/sam3",
            "description": "SAM 3 model for sam_text_det. Use Custom only for SAM 3-compatible models.",
        },
        "custom_model_id": {
            "type": "line_editor",
            "value": "",
            "description": "Used only when refinement_model is Custom SAM 3 model id.",
        },
        "prompt": {
            "type": "selector",
            "options": ["speech bubble", "text", "caption", "sound effect", "panel"],
            "value": "speech bubble",
            "description": "Prompt for SAM-style refiners. speech bubble is best for bubble safe areas; text is stricter for text masks.",
        },
        "fallback_prompt": {
            "type": "selector",
            "options": ["text", "speech bubble", "caption", "sound effect", ""],
            "value": "text",
            "description": "Prompt used if the primary prompt returns no usable candidate.",
        },
        "merge_mode": {
            "type": "selector",
            "options": ["union", "intersect", "replace_if_nonempty"],
            "value": "union",
            "description": "How to combine base and refiner masks. union = coverage; intersect = strict; replace_if_nonempty = whole refiner mask.",
        },
        "min_iou_with_initial": {
            "type": "line_editor",
            "value": 0.02,
            "description": "Reject refiner candidates with IoU below this against the base detector mask.",
        },
        "min_mask_area": {"type": "line_editor", "value": 64, "description": "Reject candidates smaller than this area in pixels."},
        "max_mask_area_ratio": {"type": "line_editor", "value": 0.75, "description": "Reject candidates larger than this fraction of the page."},
        "expand_px": {"type": "line_editor", "value": 0, "description": "Optional final mask dilation in pixels after merging."},
        "safe_rect_min_coverage": {"type": "line_editor", "value": 0.88, "description": "Minimum mask coverage used when deriving auto-layout safe rectangles."},
        "attach_safe_areas": {"type": "checkbox", "value": True, "description": "Attach refinement safe-area metadata to detected blocks."},
        "device": DEVICE_SELECTOR(),
        "description": "Run a selected base detector first, then refine the mask with a curated second detector such as SAM 3.",
    }
    _load_model_keys = {"_base_detector", "_refiner_detector"}

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self._refresh_selector_options()
        self._base_detector = None
        self._base_detector_key = None
        self._refiner_detector = None
        self._refiner_detector_key = None
        self._refiner_model_id = None
        self._refiner_device = None

    @classmethod
    def _refresh_class_selector_options(cls):
        names = _registered_detector_names()
        if "base_detector" in cls.params:
            opts = list(all_base_detector_options(names)) or list(all_base_detector_options())
            cls.params["base_detector"]["options"] = opts
            if cls.params["base_detector"].get("value") not in opts:
                cls.params["base_detector"]["value"] = "ctd" if "ctd" in opts else (opts[0] if opts else "")
        if "refiner_detector" in cls.params:
            opts = list(curated_refiner_detector_options(names)) or list(curated_refiner_detector_options())
            cls.params["refiner_detector"]["options"] = opts
            if cls.params["refiner_detector"].get("value") not in opts:
                cls.params["refiner_detector"]["value"] = opts[0] if opts else "sam_text_det"

    def _refresh_selector_options(self):
        self.__class__._refresh_class_selector_options()
        if self.params:
            for key in ("base_detector", "refiner_detector"):
                if key in self.__class__.params and key in self.params:
                    try:
                        self.params[key]["options"] = list(self.__class__.params[key].get("options", []))
                    except Exception:
                        pass

    def _param(self, key: str, default=None):
        try:
            return self.get_param_value(key)
        except Exception:
            return default

    def _saved_params_for(self, detector_key: str) -> dict:
        try:
            from utils.config import pcfg
            return (getattr(pcfg.module, "textdetector_params", {}) or {}).get(detector_key, {}) or {}
        except Exception:
            return {}

    def _make_detector(self, detector_key: str, params: dict | None = None):
        if detector_key not in TEXTDETECTORS.module_dict:
            raise KeyError(detector_key)
        return TEXTDETECTORS.module_dict[detector_key](**(params or {}))

    def _load_base_detector(self):
        self._refresh_selector_options()
        detector_key = str(self._param("base_detector", "ctd") or "ctd").strip()
        if detector_key == "sam3_refiner":
            detector_key = "ctd"
        if detector_key not in TEXTDETECTORS.module_dict:
            fallback = "ctd" if "ctd" in TEXTDETECTORS.module_dict else next(iter(TEXTDETECTORS.module_dict.keys()))
            if fallback == "sam3_refiner":
                raise RuntimeError("No usable base detector is registered for sam3_refiner.")
            self.logger.warning("SAM3 refiner base detector %s unavailable; using %s", detector_key, fallback)
            detector_key = fallback
        if self._base_detector is not None and self._base_detector_key == detector_key:
            return
        self._base_detector_key = detector_key
        self._base_detector = self._make_detector(detector_key, self._saved_params_for(detector_key))
        self._base_detector.load_model()

    def _refiner_params(self, detector_key: str) -> dict:
        params = self._saved_params_for(detector_key)
        if detector_key == "sam_text_det":
            choice = str(self._param("refinement_model", "facebook/sam3") or "facebook/sam3")
            custom = str(self._param("custom_model_id", "") or "")
            model_id = resolve_refinement_model_id(choice, custom)
            params = dict(params)
            params.update({
                "model_id": model_id,
                "text_prompt": str(self._param("prompt", "speech bubble") or "speech bubble"),
                "device": str(self._param("device", "cpu") or "cpu"),
                "box_padding": 0,
                "min_area_px": int(float(self._param("min_mask_area", 64) or 64)),
                "max_area_ratio": float(self._param("max_mask_area_ratio", 0.75) or 0.75),
            })
        return params

    def _load_refiner_detector(self):
        self._refresh_selector_options()
        detector_key = str(self._param("refiner_detector", "sam_text_det") or "sam_text_det").strip()
        allowed = set(curated_refiner_detector_options(_registered_detector_names()))
        if detector_key not in allowed:
            fallback = "sam_text_det" if "sam_text_det" in allowed else (next(iter(allowed)) if allowed else "")
            if not fallback:
                raise RuntimeError("No supported mask refiner detector is registered.")
            self.logger.warning("SAM3 refiner rejected unsupported refiner %s; using %s", detector_key, fallback)
            detector_key = fallback
        marker_model = resolve_refinement_model_id(str(self._param("refinement_model", "facebook/sam3") or "facebook/sam3"), str(self._param("custom_model_id", "") or ""))
        marker_device = str(self._param("device", "cpu") or "cpu")
        if self._refiner_detector is not None and self._refiner_detector_key == detector_key and self._refiner_model_id == marker_model and self._refiner_device == marker_device:
            return
        self._refiner_detector_key = detector_key
        self._refiner_model_id = marker_model
        self._refiner_device = marker_device
        self._refiner_detector = self._make_detector(detector_key, self._refiner_params(detector_key))
        self._refiner_detector.load_model()

    def _load_model(self):
        self._load_base_detector()
        self._load_refiner_detector()

    def _refiner_box_runner(self, image: np.ndarray, prompt: str):
        self._load_refiner_detector()
        if self._refiner_detector is None:
            return []
        if hasattr(self._refiner_detector, "set_param_value") and self._refiner_detector_key == "sam_text_det":
            self._refiner_detector.set_param_value("text_prompt", prompt)
        _mask, blocks = self._refiner_detector.detect(image, None)
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
        result = refine_mask_with_sam3(img, mask, self._opts(), box_runner=self._refiner_box_runner)
        if bool(self._param("attach_safe_areas", True)) and blocks and result.safe_areas:
            meta = result.to_dict()
            meta["base_detector"] = self._base_detector_key
            meta["refiner_detector"] = self._refiner_detector_key
            for blk in blocks:
                try:
                    rid = getattr(blk, "region_inpaint_dict", None) or {}
                    rid["sam3_refinement"] = meta
                    blk.region_inpaint_dict = rid
                except Exception:
                    pass
        for blk in blocks or []:
            try:
                blk.det_model = f"{self.name}:{self._base_detector_key}+{self._refiner_detector_key}"
            except Exception:
                pass
        return result.mask, blocks or []

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        if param_key == "base_detector":
            self._base_detector = None
            self._base_detector_key = None
        if param_key in {"refiner_detector", "refinement_model", "custom_model_id", "device"}:
            self._refiner_detector = None
            self._refiner_detector_key = None
            self._refiner_model_id = None
            self._refiner_device = None
