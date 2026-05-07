from __future__ import annotations

from typing import Dict, Iterable


WORKFLOW_PRESETS: Dict[str, Dict[str, object]] = {
    "full": {
        "label": "Full manga pipeline",
        "description": "Detect, OCR, translate, inpaint, and render/export-ready text.",
        "enable_detect": True,
        "enable_ocr": True,
        "enable_translate": True,
        "enable_inpaint": True,
        "run_preset_name": "Full",
    },
    "detect_ocr": {
        "label": "Detect + OCR only",
        "description": "Find text/bubbles and OCR source text without modifying translations or clean art.",
        "enable_detect": True,
        "enable_ocr": True,
        "enable_translate": False,
        "enable_inpaint": False,
        "run_preset_name": "Detect+OCR",
    },
    "translate": {
        "label": "Translate only",
        "description": "Reuse existing boxes/OCR and run translation only.",
        "enable_detect": False,
        "enable_ocr": False,
        "enable_translate": True,
        "enable_inpaint": False,
        "run_preset_name": "Translate",
    },
    "inpaint": {
        "label": "Inpaint only",
        "description": "Reuse existing boxes/masks and regenerate clean art only.",
        "enable_detect": False,
        "enable_ocr": False,
        "enable_translate": False,
        "enable_inpaint": True,
        "run_preset_name": "Inpaint",
    },
    "lettering_review": {
        "label": "Lettering QA pass",
        "description": "Skip AI stages and focus on rendering QA/layout review/export checks.",
        "enable_detect": False,
        "enable_ocr": False,
        "enable_translate": False,
        "enable_inpaint": False,
        "run_preset_name": "Lettering QA",
    },
}

ALIASES = {
    "detect+ocr": "detect_ocr",
    "detect-ocr": "detect_ocr",
    "ocr": "detect_ocr",
    "full_manga_pipeline": "full",
    "qa": "lettering_review",
    "lettering": "lettering_review",
}


def normalize_workflow_preset(preset_id: str) -> str:
    key = str(preset_id or "").strip().lower().replace(" ", "_").replace("-", "_")
    key = ALIASES.get(key, key)
    if key not in WORKFLOW_PRESETS:
        raise ValueError(f"unknown workflow preset: {preset_id}")
    return key


def list_workflow_presets() -> Dict[str, Dict[str, object]]:
    return {key: dict(value) for key, value in WORKFLOW_PRESETS.items()}


def apply_workflow_preset(module_config, preset_id: str) -> Dict[str, object]:
    key = normalize_workflow_preset(preset_id)
    preset = WORKFLOW_PRESETS[key]
    for attr in ("enable_detect", "enable_ocr", "enable_translate", "enable_inpaint"):
        if hasattr(module_config, attr):
            setattr(module_config, attr, bool(preset[attr]))
    if hasattr(module_config, "run_preset_name"):
        setattr(module_config, "run_preset_name", str(preset["run_preset_name"]))
    return {"preset_id": key, **dict(preset)}


def workflow_stage_vector(module_config) -> Dict[str, bool]:
    return {
        "enable_detect": bool(getattr(module_config, "enable_detect", False)),
        "enable_ocr": bool(getattr(module_config, "enable_ocr", False)),
        "enable_translate": bool(getattr(module_config, "enable_translate", False)),
        "enable_inpaint": bool(getattr(module_config, "enable_inpaint", False)),
        "run_preset_name": getattr(module_config, "run_preset_name", ""),
    }
