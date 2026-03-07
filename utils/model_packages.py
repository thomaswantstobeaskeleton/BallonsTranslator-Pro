"""
Model packages for selective download at first launch (Issue #15).

Packages group modules so users can choose e.g. "Core" only (~hundreds MB)
instead of auto-downloading everything including PaddleOCR-VL (~1.7 GB).
None = legacy "download all" for existing installs; ["core"] = minimal default for new users.
"""
from __future__ import annotations

from typing import List, Optional, Tuple, Any

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

# Human-readable labels and short descriptions for the first-launch dialog
PACKAGE_LABELS = {
    "core": ("Core (recommended)", "Text detection, inpainting, OCR, pkuseg — minimal to run"),
    "advanced_ocr": ("Advanced OCR", "PaddleOCR-VL for manga (~1.7 GB), MIT 48px/32px"),
    "advanced_inpaint": ("Advanced inpainting", "LaMa variants, ONNX, PatchMatch"),
    "optional_onnx": ("Optional ONNX inpainting", "Lama 2025 / lama-manga ONNX (smaller, CPU-friendly)"),
}


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
