"""
Config and input validation (Section 9).

- Central numeric constraint clamping for settings.
- Zip/dir batch input normalization and validation.
- API key and HuggingFace token format validation.
"""
from __future__ import annotations

import os
import os.path as osp
import re
from typing import Any, Dict, Optional, Tuple, Union

from .logger import logger as LOGGER

# -----------------------------------------------------------------------------
# Numeric constraints: key path (dot-separated for nested) -> (min, max)
# Apply to pcfg.module.* and pcfg.* and to nested param dicts.
# -----------------------------------------------------------------------------
SETTING_CONSTRAINTS: Dict[str, Tuple[Union[int, float], Union[int, float]]] = {
    # ModuleConfig
    "module.inpaint_tile_size": (0, 2048),
    "module.inpaint_tile_overlap": (0, 256),
    "module.osb_group_gap_px": (0, 500),
    "module.osb_exclude_bubble_iou": (0.0, 1.0),
    "module.osb_page_number_margin_ratio": (0.0, 0.5),
    "module.layout_collision_min_mask_ratio": (0.0, 1.0),
    "module.layout_collision_max_retries": (0, 20),
    "module.ocr_upscale_min_side": (0, 4096),
    "module.image_upscale_initial_factor": (1.0, 4.0),
    "module.image_upscale_final_factor": (1.0, 4.0),
    "module.inpaint_spill_to_disk_after_blocks": (0, 128),
    # DrawPanelConfig
    "drawpanel.pentool_width": (0.1, 500.0),
    "drawpanel.inpainter_width": (0.1, 500.0),
    "drawpanel.inpaint_hardness": (0, 100),
    "drawpanel.recttool_dilate_ksize": (0, 31),
    "drawpanel.recttool_erode_ksize": (0, 31),
    "drawpanel.sam_maskrefine_padding_px": (0, 128),
    # ProgramConfig (top-level)
    "mask_transparency": (0.0, 1.0),
    "original_transparency": (0.0, 1.0),
    "recent_proj_list_max": (0, 100),
    "imgsave_quality": (1, 101),
    "supersampling_factor": (1, 8),
    "config_panel_font_scale": (0.5, 2.0),
    "unload_after_idle_minutes": (0, 1440),
    "manga_source_request_delay": (0.0, 60.0),
}


def _get_nested(obj: Any, key_path: str) -> Optional[Any]:
    """Get nested attribute or dict value. key_path e.g. 'module.inpaint_tile_size'."""
    try:
        for part in key_path.split("."):
            if hasattr(obj, part):
                obj = getattr(obj, part)
            elif isinstance(obj, dict) and part in obj:
                obj = obj[part]
            else:
                return None
        return obj
    except Exception:
        return None


def _set_nested(obj: Any, key_path: str, value: Any) -> bool:
    """Set nested attribute or dict value. Returns True if set."""
    try:
        parts = key_path.split(".")
        for part in parts[:-1]:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            elif isinstance(obj, dict) and part in obj:
                obj = obj[part]
            else:
                return False
        last = parts[-1]
        if hasattr(obj, last):
            setattr(obj, last, value)
            return True
        if isinstance(obj, dict) and last in obj:
            obj[last] = value
            return True
        return False
    except Exception:
        return False


def _is_numeric(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, (int, float)):
        return True
    if isinstance(v, dict) and "value" in v:
        return isinstance(v["value"], (int, float))
    return False


def _to_number(v: Any) -> Optional[Union[int, float]]:
    if v is None:
        return None
    if isinstance(v, dict) and "value" in v:
        v = v["value"]
    if isinstance(v, (int, float)):
        return v
    try:
        if isinstance(v, str) and v.strip() != "":
            if "." in v:
                return float(v)
            return int(v)
    except (ValueError, TypeError):
        pass
    return None


def clamp_settings(obj: Any, constraints: Optional[Dict[str, Tuple[Union[int, float], Union[int, float]]]] = None) -> None:
    """
    Apply min/max constraints to numeric settings in place.
    obj can be pcfg (ProgramConfig) or a dict with same structure (e.g. from UI).
    Uses SETTING_CONSTRAINTS by default.
    """
    constraints = constraints or SETTING_CONSTRAINTS
    for key_path, (lo, hi) in constraints.items():
        val = _get_nested(obj, key_path)
        if not _is_numeric(val):
            continue
        num = _to_number(val)
        if num is None:
            continue
        if num < lo or num > hi:
            clamped = max(lo, min(hi, num))
            _set_nested(obj, key_path, clamped)
            LOGGER.debug("Clamped setting %s from %s to %s", key_path, num, clamped)


def clamp_module_params_dict(params: Dict[str, Any], module_name: str) -> None:
    """
    Clamp known numeric values inside a module's params dict (e.g. translator_params["trans_llm_api"]).
    """
    param_bounds: Dict[str, Tuple[Union[int, float], Union[int, float]]] = {
        "temperature": (0.0, 2.0),
        "top_p": (0.0, 1.0),
        "max_tokens": (1, 131072),
        "timeout": (1, 600),
    }
    for pk, pv in list(params.items()):
        if pk not in param_bounds:
            continue
        num = _to_number(pv)
        if num is None:
            continue
        lo, hi = param_bounds[pk]
        if num < lo or num > hi:
            clamped = max(lo, min(hi, num))
            if isinstance(pv, dict) and "value" in pv:
                params[pk]["value"] = clamped
            else:
                params[pk] = clamped
            LOGGER.debug("Clamped %s.%s from %s to %s", module_name, pk, num, clamped)


# -----------------------------------------------------------------------------
# Zip / batch input normalization and validation
# -----------------------------------------------------------------------------

def normalize_zip_file_input(path_or_file: Union[str, Any]) -> str:
    """
    Normalize file-like or string input to a single path string.
    If path_or_file has a .name attribute (e.g. QFileDialog selected file), use it.
    """
    if path_or_file is None:
        return ""
    if hasattr(path_or_file, "name") and getattr(path_or_file, "name", None):
        return osp.normpath(osp.abspath(str(path_or_file.name)))
    s = str(path_or_file).strip()
    return osp.normpath(osp.abspath(s)) if s else ""


def validate_batch_input_path(path: str) -> Tuple[str, str]:
    """
    Validate that path exists and is either a directory or a .zip file.
    Returns (normalized_absolute_path, "dir" | "zip").
    Raises ValueError with a clear message if invalid.
    """
    if not path or not str(path).strip():
        raise ValueError("Batch input path is empty.")
    path = osp.normpath(osp.abspath(str(path).strip()))
    if not osp.exists(path):
        raise ValueError(f"Path does not exist: {path}")
    if osp.isfile(path):
        if path.lower().endswith(".zip"):
            return path, "zip"
        raise ValueError(f"File is not a directory or .zip: {path}")
    if osp.isdir(path):
        return path, "dir"
    raise ValueError(f"Path is neither a directory nor a .zip file: {path}")


# -----------------------------------------------------------------------------
# API key / token format validation (heuristic)
# -----------------------------------------------------------------------------

def validate_huggingface_token(token: Optional[str]) -> Tuple[bool, Optional[str]]:
    """
    Check HF token shape. Valid tokens are typically "hf_..." (length >= 10).
    Returns (True, None) if valid or empty; (False, message) if invalid.
    """
    if not token or not str(token).strip():
        return True, None
    t = str(token).strip()
    if len(t) < 10:
        return False, "HuggingFace token is too short (expected at least 10 characters)."
    if not re.match(r"^hf_[A-Za-z0-9_-]+$", t):
        return False, "HuggingFace token should start with 'hf_' and contain only letters, numbers, hyphens, and underscores."
    return True, None


_API_KEY_HEURISTICS: Dict[str, Tuple[Tuple[str, ...], int]] = {
    "OpenAI": (("sk-", "sk-proj-"), 20),
    "Google": (("AIza",), 20),
    "OpenRouter": (("sk-or-",), 20),
    "Grok": (("xai-",), 20),
    "LLM Studio": ((), 1),
    "Anthropic": (("sk-ant-",), 20),
    "DeepL": ((), 30),
    "Youdao": ((), 10),
    "Yandex": (("trnsl.", "AQVN"), 10),
}


def validate_api_key(provider: Optional[str], key: Optional[str], strict: bool = False) -> Tuple[bool, Optional[str]]:
    """
    Heuristic validation of provider API key format (prefix/length).
    Returns (True, None) if valid or key is empty; (False, message) if invalid.
    When strict=False, empty key is valid (optional); when strict=True, empty key returns invalid.
    """
    key = (key or "").strip()
    if not key:
        return (False, "API key is required for this provider.") if strict else (True, None)
    provider = (provider or "").strip()
    if not provider:
        return True, None
    heuristics = _API_KEY_HEURISTICS.get(provider)
    if not heuristics:
        return True, None
    prefixes, min_len = heuristics
    if len(key) < min_len:
        return False, f"API key is too short (expected at least {min_len} characters for {provider})."
    if prefixes:
        if not any(key.startswith(p) for p in prefixes):
            return False, f"API key format may be wrong for {provider} (expected prefix like {prefixes[0]}...)."
    return True, None
