"""
Tier metadata for module stability and dependency weight.

Used by docs/UI to show consistent labels:
- Stable
- Beta
- Experimental
- External dependency heavy
"""
from __future__ import annotations

from typing import Dict, Optional

TIER_STABLE = "Stable"
TIER_BETA = "Beta"
TIER_EXPERIMENTAL = "Experimental"
TIER_EXTERNAL_HEAVY = "External dependency heavy"

TIER_BADGES = {
    TIER_STABLE: "🟢 Stable",
    TIER_BETA: "🟡 Beta",
    TIER_EXPERIMENTAL: "🟠 Experimental",
    TIER_EXTERNAL_HEAVY: "🔵 External dependency heavy",
}

# Canonical per-module labels used by UI tooltips and docs.
# Missing modules default to Stable in UI.
MODULE_TIERS: Dict[str, str] = {
    # Detectors
    "ctd": TIER_STABLE,
    "paddle_det": TIER_STABLE,
    "easyocr_det": TIER_BETA,
    "ysgyolo": TIER_BETA,
    "hf_object_det": TIER_BETA,
    "mmocr_det": TIER_EXTERNAL_HEAVY,
    "surya_det": TIER_BETA,
    "paddle_det_v5": TIER_BETA,
    "magi_det": TIER_BETA,
    "craft_det": TIER_EXTERNAL_HEAVY,
    "rapidocr_det": TIER_BETA,
    "dptext_detr": TIER_EXTERNAL_HEAVY,
    "swintextspotter_v2": TIER_EXTERNAL_HEAVY,
    "hunyuan_ocr_det": TIER_EXPERIMENTAL,
    "textmamba_det": TIER_EXPERIMENTAL,
    # OCR
    "manga_ocr": TIER_STABLE,
    "paddle_ocr": TIER_STABLE,
    "mit48px_ctc": TIER_STABLE,
    "mit48px": TIER_BETA,
    "mit32px": TIER_BETA,
    "easyocr_ocr": TIER_BETA,
    "mmocr_ocr": TIER_EXTERNAL_HEAVY,
    "surya_ocr": TIER_BETA,
    "paddle_rec_v5": TIER_BETA,
    "rapidocr": TIER_BETA,
    "PaddleOCRVLManga": TIER_EXTERNAL_HEAVY,
    "paddle_vl": TIER_EXTERNAL_HEAVY,
    "paddleocr_vl_hf": TIER_EXTERNAL_HEAVY,
    "qwen2vl_7b": TIER_EXTERNAL_HEAVY,
    "internvl2_ocr": TIER_EXTERNAL_HEAVY,
    "internvl3_ocr": TIER_EXTERNAL_HEAVY,
    "nemotron_ocr": TIER_EXTERNAL_HEAVY,
    "hunyuan_ocr": TIER_EXTERNAL_HEAVY,
    "manga_ocr_mobile": TIER_BETA,
    # Inpainters
    "aot": TIER_STABLE,
    "lama_large_512px": TIER_BETA,
    "lama_mpe": TIER_BETA,
    "lama_onnx": TIER_BETA,
    "lama_manga_onnx": TIER_BETA,
    "patchmatch": TIER_STABLE,
    "simple_lama": TIER_EXTERNAL_HEAVY,
    "mat": TIER_EXTERNAL_HEAVY,
    "flux_fill": TIER_EXTERNAL_HEAVY,
    # Translators
    "google": TIER_STABLE,
    "DeepL": TIER_STABLE,
    "ChatGPT": TIER_BETA,
    "LLM_API_Translator": TIER_BETA,
    "Sakura": TIER_BETA,
    "nllb200": TIER_BETA,
    "m2m100": TIER_BETA,
    "text-generation-webui": TIER_EXPERIMENTAL,
}


def get_module_tier(module_key: str) -> str:
    return MODULE_TIERS.get(module_key, TIER_STABLE)


def get_module_tier_badge(module_key: str) -> str:
    return TIER_BADGES.get(get_module_tier(module_key), TIER_BADGES[TIER_STABLE])


def format_module_tier_tooltip(module_key: str, *, show_badge: bool = True) -> Optional[str]:
    if not show_badge:
        return None
    tier = get_module_tier(module_key)
    badge = TIER_BADGES.get(tier, tier)
    return f"{module_key} — {badge}"

