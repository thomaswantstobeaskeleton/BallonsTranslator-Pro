"""
Model packages and user-facing presets for first-launch selective download.

`MODEL_PACKAGES` remains the low-level module grouping consumed by download logic.
`MODEL_PACKAGE_PRESETS` adds user-centered curated bundles (fast start, balanced,
OCR-heavy, etc.) so the UI can explain tradeoffs and dependency hints clearly.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from qtpy.QtCore import QT_TRANSLATE_NOOP

# Registry name -> module_key. pkuseg is a special key (handled in prepare_local_files).
FALLBACK_MODEL_PACKAGES = {
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

# Package tier controls first-run presentation order and labels.
PACKAGE_TIERS = {
    "core": "Stable",
    "advanced_inpaint": "Beta",
    "advanced_ocr": "External dependency heavy",
    "optional_onnx": "Experimental",
}

# Human-readable labels and short descriptions for the first-launch dialog
FALLBACK_PACKAGE_LABELS = {
    "core": ("Core (recommended)", "Text detection, inpainting, OCR, pkuseg — minimal to run"),
    "advanced_ocr": ("Advanced OCR", "PaddleOCR-VL for manga (~1.7 GB), MIT 48px/32px"),
    "advanced_inpaint": ("Advanced inpainting", "LaMa variants, ONNX, PatchMatch"),
    "optional_onnx": ("Optional ONNX inpainting", "Lama 2025 / lama-manga ONNX (smaller, CPU-friendly)"),
}

# Translation extraction catalog for package labels/descriptions.
PACKAGE_LABELS = {
    "core": (
        QT_TRANSLATE_NOOP("ModelPackageCatalog", "Core package"),
        QT_TRANSLATE_NOOP("ModelPackageCatalog", "Text detection, inpainting, OCR, pkuseg — minimal to run"),
    ),
    "advanced_ocr": (
        QT_TRANSLATE_NOOP("ModelPackageCatalog", "Advanced OCR"),
        QT_TRANSLATE_NOOP("ModelPackageCatalog", "PaddleOCR-VL for manga (~1.7 GB), MIT 48px/32px"),
    ),
    "advanced_inpaint": (
        QT_TRANSLATE_NOOP("ModelPackageCatalog", "Advanced inpainting"),
        QT_TRANSLATE_NOOP("ModelPackageCatalog", "LaMa variants, ONNX, PatchMatch"),
    ),
    "optional_onnx": (
        QT_TRANSLATE_NOOP("ModelPackageCatalog", "Optional ONNX inpainting"),
        QT_TRANSLATE_NOOP("ModelPackageCatalog", "Lama 2025 / lama-manga ONNX (smaller, CPU-friendly)"),
    ),
}

FALLBACK_MODULE_MANIFEST = {
    "ctd": {
        "module_key": "ctd",
        "category": "textdetector",
        "size_estimate": "~260 MB",
        "required_deps": [],
        "support_tier": "core",
    },
    "aot": {
        "module_key": "aot",
        "category": "inpaint",
        "size_estimate": "~220 MB",
        "required_deps": [],
        "support_tier": "core",
    },
    "manga_ocr": {
        "module_key": "manga_ocr",
        "category": "ocr",
        "size_estimate": "~130 MB",
        "required_deps": [],
        "support_tier": "core",
    },
    "mit48px_ctc": {
        "module_key": "mit48px_ctc",
        "category": "ocr",
        "size_estimate": "~160 MB",
        "required_deps": [],
        "support_tier": "core",
    },
    "pkuseg": {
        "module_key": "pkuseg",
        "category": "pkuseg",
        "size_estimate": "~50 MB",
        "required_deps": [],
        "support_tier": "core",
    },
    "PaddleOCRVLManga": {
        "module_key": "PaddleOCRVLManga",
        "category": "ocr",
        "size_estimate": "~1.7 GB",
        "required_deps": ["paddlepaddle"],
        "support_tier": "advanced",
    },
    "mit48px": {
        "module_key": "mit48px",
        "category": "ocr",
        "size_estimate": "~180 MB",
        "required_deps": [],
        "support_tier": "advanced",
    },
    "mit32px": {
        "module_key": "mit32px",
        "category": "ocr",
        "size_estimate": "~140 MB",
        "required_deps": [],
        "support_tier": "advanced",
    },
    "lama_mpe": {
        "module_key": "lama_mpe",
        "category": "inpaint",
        "size_estimate": "~160 MB",
        "required_deps": [],
        "support_tier": "advanced",
    },
    "lama_large_512px": {
        "module_key": "lama_large_512px",
        "category": "inpaint",
        "size_estimate": "~220 MB",
        "required_deps": [],
        "support_tier": "advanced",
    },
    "lama_onnx": {
        "module_key": "lama_onnx",
        "category": "inpaint",
        "size_estimate": "~160 MB",
        "required_deps": ["onnxruntime"],
        "support_tier": "optional",
    },
    "lama_manga_onnx": {
        "module_key": "lama_manga_onnx",
        "category": "inpaint",
        "size_estimate": "~170 MB",
        "required_deps": ["onnxruntime"],
        "support_tier": "optional",
    },
    "patchmatch": {
        "module_key": "patchmatch",
        "category": "inpaint",
        "size_estimate": "~30 MB",
        "required_deps": [],
        "support_tier": "advanced",
    },
}

_LOGGER = logging.getLogger(__name__)
_MANIFEST_PATH = Path(__file__).resolve().parents[1] / "data" / "model_manifest.json"
_MANIFEST_CACHE: Optional[Dict[str, Any]] = None


def _fallback_manifest() -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "packages": [
            {
                "id": package_id,
                "label": label,
                "description": desc,
                "modules": [
                    {"category": category, "module_key": module_key}
                    for category, module_key in entries
                ],
            }
            for package_id, entries in FALLBACK_MODEL_PACKAGES.items()
            for label, desc in [FALLBACK_PACKAGE_LABELS.get(package_id, (package_id, ""))]
        ],
        "modules": list(FALLBACK_MODULE_MANIFEST.values()),
    }


def _validate_manifest_data(manifest_data: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if not isinstance(manifest_data, dict):
        return ["Manifest root must be a JSON object."]

    if not isinstance(manifest_data.get("schema_version"), int):
        errors.append("schema_version must be an integer.")

    modules = manifest_data.get("modules")
    if not isinstance(modules, list):
        errors.append("modules must be a list.")
        modules = []
    module_keys: set[str] = set()
    for idx, module in enumerate(modules):
        ctx = f"modules[{idx}]"
        if not isinstance(module, dict):
            errors.append(f"{ctx} must be an object.")
            continue
        for req in ("module_key", "category", "size_estimate", "required_deps", "support_tier"):
            if req not in module:
                errors.append(f"{ctx}.{req} is required.")
        key = module.get("module_key")
        if not isinstance(key, str) or not key:
            errors.append(f"{ctx}.module_key must be a non-empty string.")
        elif key in module_keys:
            errors.append(f"{ctx}.module_key '{key}' is duplicated.")
        else:
            module_keys.add(key)
        if not isinstance(module.get("required_deps"), list) or any(not isinstance(d, str) for d in module.get("required_deps", [])):
            errors.append(f"{ctx}.required_deps must be a list of strings.")

    packages = manifest_data.get("packages")
    if not isinstance(packages, list):
        errors.append("packages must be a list.")
        packages = []
    package_ids: set[str] = set()
    for idx, package in enumerate(packages):
        ctx = f"packages[{idx}]"
        if not isinstance(package, dict):
            errors.append(f"{ctx} must be an object.")
            continue
        pid = package.get("id")
        if not isinstance(pid, str) or not pid:
            errors.append(f"{ctx}.id must be a non-empty string.")
        elif pid in package_ids:
            errors.append(f"{ctx}.id '{pid}' is duplicated.")
        else:
            package_ids.add(pid)
        if not isinstance(package.get("label"), str):
            errors.append(f"{ctx}.label must be a string.")
        if not isinstance(package.get("description"), str):
            errors.append(f"{ctx}.description must be a string.")
        refs = package.get("modules")
        if not isinstance(refs, list):
            errors.append(f"{ctx}.modules must be a list.")
            continue
        for ref_idx, ref in enumerate(refs):
            ref_ctx = f"{ctx}.modules[{ref_idx}]"
            if not isinstance(ref, dict):
                errors.append(f"{ref_ctx} must be an object.")
                continue
            if not isinstance(ref.get("category"), str):
                errors.append(f"{ref_ctx}.category must be a string.")
            module_key = ref.get("module_key")
            if module_key is not None and not isinstance(module_key, str):
                errors.append(f"{ref_ctx}.module_key must be null or string.")
            if isinstance(module_key, str) and module_key not in module_keys and ref.get("category") not in {"_optional_onnx"}:
                errors.append(f"{ref_ctx}.module_key '{module_key}' is missing from modules.")
    return errors


def _parse_manifest_data(manifest_data: Dict[str, Any]) -> Dict[str, Any]:
    package_map: Dict[str, List[Tuple[str, Optional[str]]]] = {}
    package_labels: Dict[str, Tuple[str, str]] = {}
    modules_by_key: Dict[str, Dict[str, Any]] = {}
    for module in manifest_data.get("modules", []):
        if isinstance(module, dict) and isinstance(module.get("module_key"), str):
            modules_by_key[module["module_key"]] = module
    for package in manifest_data.get("packages", []):
        if not isinstance(package, dict):
            continue
        package_id = package.get("id")
        if not isinstance(package_id, str):
            continue
        package_labels[package_id] = (
            str(package.get("label", package_id)),
            str(package.get("description", "")),
        )
        refs: List[Tuple[str, Optional[str]]] = []
        for ref in package.get("modules", []):
            if not isinstance(ref, dict):
                continue
            category = ref.get("category")
            module_key = ref.get("module_key")
            if not isinstance(category, str):
                continue
            refs.append((category, module_key if isinstance(module_key, str) else None))
        package_map[package_id] = refs
    return {
        "schema_version": manifest_data.get("schema_version", 1),
        "model_packages": package_map,
        "package_labels": package_labels,
        "modules_by_key": modules_by_key,
        "manifest_path": str(_MANIFEST_PATH),
        "source": "manifest",
    }


def _load_model_manifest() -> Dict[str, Any]:
    global _MANIFEST_CACHE
    if _MANIFEST_CACHE is not None:
        return _MANIFEST_CACHE

    if _MANIFEST_PATH.exists():
        try:
            manifest_data = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
            errors = _validate_manifest_data(manifest_data)
            if errors:
                raise ValueError("; ".join(errors))
            _MANIFEST_CACHE = _parse_manifest_data(manifest_data)
            return _MANIFEST_CACHE
        except Exception as e:
            _LOGGER.warning("Invalid model manifest at %s: %s. Falling back to hardcoded package data.", _MANIFEST_PATH, e)
    _MANIFEST_CACHE = _parse_manifest_data(_fallback_manifest())
    _MANIFEST_CACHE["source"] = "fallback"
    return _MANIFEST_CACHE


def validate_manifest_on_startup() -> bool:
    """Validate manifest schema at app startup. Returns True when valid or intentionally missing."""
    if not _MANIFEST_PATH.exists():
        _LOGGER.info("Model manifest not found at %s, using hardcoded fallback package data.", _MANIFEST_PATH)
        return True
    try:
        manifest_data = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        _LOGGER.error("Failed to read model manifest %s: %s", _MANIFEST_PATH, e)
        return False
    errors = _validate_manifest_data(manifest_data)
    if errors:
        _LOGGER.error("Model manifest schema validation failed (%s): %s", _MANIFEST_PATH, " | ".join(errors))
        return False
    return True


def get_module_manifest_metadata(module_key: str) -> Dict[str, Any]:
    """Return manifest metadata for a module key, or an empty dict."""
    manifest = _load_model_manifest()
    data = manifest["modules_by_key"].get(module_key)
    return data.copy() if isinstance(data, dict) else {}


# Backward-compatible module-level constants consumed by existing UI code.
_MANIFEST = _load_model_manifest()
MODEL_PACKAGES = _MANIFEST["model_packages"]
PACKAGE_LABELS = _MANIFEST["package_labels"]
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
