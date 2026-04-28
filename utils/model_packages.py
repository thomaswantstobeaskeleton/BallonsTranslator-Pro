"""
Model packages and user-facing presets for first-launch selective download.

`MODEL_PACKAGES` remains the low-level module grouping consumed by download logic.
`MODEL_PACKAGE_PRESETS` adds user-centered curated bundles (fast start, balanced,
OCR-heavy, etc.) so the UI can explain tradeoffs and dependency hints clearly.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Any

# Registry name -> module_key. pkuseg is a special key (handled in prepare_local_files).
MODEL_PACKAGES = {
    "core": [
        ("textdetector", "ctd"),
        ("inpaint", "aot"),
        ("ocr", "manga_ocr"),
        ("ocr", "mit48px_ctc"),
        ("pkuseg", None),
    ],
    "advanced_ocr": [
        ("ocr", "PaddleOCRVLManga"),
        ("ocr", "mit48px"),
        ("ocr", "mit32px"),
    ],
    "advanced_inpaint": [
        ("inpaint", "lama_mpe"),
        ("inpaint", "lama_large_512px"),
        ("inpaint", "lama_onnx"),
        ("inpaint", "lama_manga_onnx"),
        ("inpaint", "patchmatch"),
    ],
    "optional_onnx": [
        # Handled separately in prepare_local_files (optional_onnx_models list)
        ("_optional_onnx", None),
    ],
}

# Human-readable labels and short descriptions for package-level advanced/custom mode.
PACKAGE_LABELS = {
    "core": ("Core package", "Text detection, inpainting, OCR, pkuseg — minimal to run"),
    "advanced_ocr": ("Advanced OCR", "PaddleOCR-VL for manga (~1.7 GB), MIT 48px/32px"),
    "advanced_inpaint": ("Advanced inpainting", "LaMa variants, ONNX, PatchMatch"),
    "optional_onnx": ("Optional ONNX inpainting", "Lama 2025 / lama-manga ONNX (smaller, CPU-friendly)"),
}

# User-centered presets surfaced in the first-launch dialog.
# package_ids is intentionally explicit to keep setup reproducible.
MODEL_PACKAGE_PRESETS: Dict[str, Dict[str, Any]] = {
    "core_minimal": {
        "label": "Core minimal (fast start)",
        "intended_use": "Smallest useful setup for quick first run and low disk usage.",
        "approx_size": "~0.7 GB",
        "package_ids": ["core"],
        "dependency_hints": "No extra runtime dependencies expected beyond base install.",
    },
    "balanced_default": {
        "label": "Balanced default",
        "intended_use": "Recommended for most users: solid OCR + inpaint quality without huge downloads.",
        "approx_size": "~2.5 GB",
        "package_ids": ["core", "advanced_inpaint"],
        "dependency_hints": "Large download; ONNX inpaint variants run best with onnxruntime(-gpu) when available.",
    },
    "ocr_heavy": {
        "label": "OCR-heavy",
        "intended_use": "Best OCR coverage/accuracy for diverse manga text styles and mixed languages.",
        "approx_size": "~4.2 GB",
        "package_ids": ["core", "advanced_ocr"],
        "dependency_hints": "Very large download. PaddleOCR-VL may require extra runtime wheels depending on platform/device.",
    },
    "inpaint_heavy": {
        "label": "Inpaint-heavy",
        "intended_use": "More inpainting backends for harder backgrounds, speed/quality experiments, and CPU fallback.",
        "approx_size": "~3.0 GB",
        "package_ids": ["core", "advanced_inpaint", "optional_onnx"],
        "dependency_hints": "Large download. ONNX variants benefit from onnxruntime(-gpu); PatchMatch is CPU-heavy on big pages.",
    },
    "video_focused_optional": {
        "label": "Video-focused optional",
        "intended_use": "Optional add-on for video/comic workflows that favor broader OCR + inpaint compatibility.",
        "approx_size": "~5.0 GB",
        "package_ids": ["core", "advanced_ocr", "advanced_inpaint", "optional_onnx"],
        "dependency_hints": "Largest download. Recommended only with stable bandwidth/storage and optional GPU runtimes.",
    },
}

DEFAULT_MODEL_PACKAGE_PRESET_ID = "balanced_default"


def get_package_ids_for_preset(preset_id: str) -> List[str]:
    preset = MODEL_PACKAGE_PRESETS.get(preset_id)
    if preset is None:
        return []
    return list(preset.get("package_ids") or [])


def get_module_classes_for_packages(
    package_ids: Optional[List[str]],
    *,
    registries: Optional[Any] = None,
) -> List[Any]:
    """
    Return the list of module classes to download for the given package IDs.
    If package_ids is None (legacy), returns None to mean "all modules".
    Otherwise returns only classes for (registry, module_key) in the selected packages.
    """
    if package_ids is None:
        return None  # caller treats as "download all"
    if not package_ids:
        return []  # download nothing at startup

    if registries is None:
        from modules import INPAINTERS, TEXTDETECTORS, OCR, TRANSLATORS
        registries = {
            "textdetector": TEXTDETECTORS,
            "ocr": OCR,
            "inpaint": INPAINTERS,
            "translator": TRANSLATORS,
        }

    result = []
    seen: set[tuple[str, Optional[str]]] = set()  # (registry, key) to avoid duplicates  # (registry, key) to avoid duplicates
    for pid in package_ids:
        entries = MODEL_PACKAGES.get(pid, [])
        for registry_name, module_key in entries:
            if (registry_name, module_key) in seen:
                continue
            if registry_name == "pkuseg" or registry_name == "_optional_onnx":
                # Handled separately in prepare_local_files
                continue
            seen.add((registry_name, module_key))
            reg = registries.get(registry_name)
            if reg is None:
                continue
            cls = reg.get(module_key) if module_key else None
            if cls is not None and getattr(cls, "download_file_list", None) is not None:
                result.append(cls)
    return result


def package_ids_include_pkuseg(package_ids: Optional[List[str]]) -> bool:
    """True if pkuseg should be prepared (core package or legacy)."""
    if package_ids is None:
        return True
    return "core" in package_ids


def package_ids_include_optional_onnx(package_ids: Optional[List[str]]) -> bool:
    """True if optional ONNX inpainter models should be downloaded."""
    if package_ids is None:
        return True
    return "optional_onnx" in package_ids
