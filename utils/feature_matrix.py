from __future__ import annotations
import importlib.util
import os.path as osp
from typing import List, Dict


def _exists_module(rel_path: str) -> bool:
    return osp.isfile(rel_path)


def _has_pkg(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def build_feature_matrix() -> List[Dict[str, str]]:
    rows = [
        {"feature": "Detection", "detail": "Speech bubble detection + segmentation (YOLO + SAM2/3)",
         "status": "available" if _exists_module('modules/textdetector/detector_hf_object_detection.py') else "missing"},
        {"feature": "Cleaning", "detail": "Inpaint speech bubbles / OSB text (Flux / OpenCV / others)",
         "status": "available" if _exists_module('modules/inpaint/inpaint_flux_fill.py') else "partial"},
        {"feature": "Translation", "detail": "LLM OCR + translation multi-language",
         "status": "available" if _exists_module('modules/ocr/ocr_paddleocr_vl_hf.py') else "partial"},
        {"feature": "Rendering", "detail": "Alignment + custom font packs",
         "status": "available"},
        {"feature": "Upscaling", "detail": "Real-CUGAN / ESRGAN style upscalers",
         "status": "available" if _exists_module('modules/upscaling/esrgan_upscaler.py') else "partial"},
        {"feature": "Processing", "detail": "Single + batch + ZIP workflows",
         "status": "available"},
        {"feature": "Interfaces", "detail": "Desktop UI + automation API (+ optional web/CLI sidecar)",
         "status": "available"},
        {"feature": "Automation", "detail": "One-click pipeline + local API",
         "status": "available"},
        {"feature": "Models: PaddleOCR", "detail": "PaddleOCR / PaddleOCR-VL / MangaOCR variants",
         "status": "available" if _exists_module('modules/ocr/ocr_paddleVL_manga.py') else "partial"},
        {"feature": "Models: Segmentation", "detail": "SAM2/SAM3 refinement toggle",
         "status": "available" if _has_pkg('transformers') else "optional_dep_missing"},
        {"feature": "Models: Community showcase", "detail": "YSG segmenters, MangaLens bubble segmentation, Flux variants, PaddleOCR-VL-1.5, Real-CUGAN/AnimeSharp",
         "status": "available" if _exists_module('modules/textdetector/detector_hf_object_detection.py') else "partial"},
    ]
    return rows


def feature_matrix_text() -> str:
    lines = ["Feature Matrix (runtime capability)"]
    for row in build_feature_matrix():
        lines.append(f"- {row['feature']}: {row['detail']} [{row['status']}]")
    return "\n".join(lines)
