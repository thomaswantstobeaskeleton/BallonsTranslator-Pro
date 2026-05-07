from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Dict, Iterable, Mapping, Optional, Tuple

from .text_rendering import custom_manga_presets, manga_presets, preset_id_from_label, sanitize_manga_preset

PRESET_PACK_FORMAT = "ballonstranslator.rendering_presets.v1"


def build_preset_pack(config_obj, include_builtins: bool = False, source: str = "BallonsTranslator-Pro") -> Dict[str, object]:
    """Build a portable JSON pack for custom manga lettering presets."""
    presets = manga_presets(config_obj) if include_builtins else custom_manga_presets(config_obj)
    return {
        "format": PRESET_PACK_FORMAT,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "include_builtins": bool(include_builtins),
        "presets": {pid: sanitize_manga_preset(preset) for pid, preset in presets.items()},
    }


def write_preset_pack(config_obj, path: str, include_builtins: bool = False) -> Dict[str, object]:
    path = os.path.normpath(str(path or ""))
    if not path:
        raise ValueError("path is required")
    if not path.lower().endswith(".json"):
        path += ".json"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pack = build_preset_pack(config_obj, include_builtins=include_builtins)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(pack, f, ensure_ascii=False, indent=2)
    pack["path"] = path
    return pack


def _preset_entries_from_pack(data: Mapping[str, object]) -> Dict[str, Dict[str, object]]:
    if not isinstance(data, Mapping):
        raise ValueError("preset pack must be a JSON object")
    presets = data.get("presets", data)
    if not isinstance(presets, Mapping):
        raise ValueError("preset pack does not contain a presets object")
    out: Dict[str, Dict[str, object]] = {}
    for preset_id, preset in presets.items():
        if not isinstance(preset, Mapping):
            continue
        label = str(preset.get("label") or preset_id or "Imported preset")
        pid = str(preset_id or "").strip()
        if not pid or pid == "label":
            pid = preset_id_from_label(label, out.keys())
        if not pid.startswith("custom:"):
            pid = "custom:" + pid
        out[pid] = sanitize_manga_preset(dict(preset), label=label)
    return out


def read_preset_pack(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {"path": path, "presets": _preset_entries_from_pack(data), "format": data.get("format", "unknown") if isinstance(data, Mapping) else "unknown"}


def import_preset_pack(config_obj, path: str, overwrite: bool = False) -> Dict[str, object]:
    loaded = read_preset_pack(path)
    incoming: Dict[str, Dict[str, object]] = dict(loaded.get("presets", {}) or {})
    current: Dict[str, Dict[str, object]] = dict(getattr(config_obj, "render_custom_manga_presets", {}) or {})
    imported: Dict[str, str] = {}
    skipped: Dict[str, str] = {}
    for preset_id, preset in incoming.items():
        target_id = preset_id
        if target_id in current and not overwrite:
            label = str(preset.get("label") or target_id.split(":", 1)[-1])
            target_id = preset_id_from_label(label, list(current.keys()) + list(imported.values()))
        if target_id in current and overwrite:
            current[target_id] = preset
            imported[preset_id] = target_id
        elif target_id not in current:
            current[target_id] = preset
            imported[preset_id] = target_id
        else:
            skipped[preset_id] = "duplicate"
    config_obj.render_custom_manga_presets = current
    return {"path": path, "imported": imported, "skipped": skipped, "imported_count": len(imported), "total_custom_presets": len(current)}


def delete_custom_preset(config_obj, preset_id: str) -> Dict[str, object]:
    preset_id = str(preset_id or "").strip()
    current = dict(getattr(config_obj, "render_custom_manga_presets", {}) or {})
    existed = preset_id in current
    if existed:
        current.pop(preset_id, None)
        config_obj.render_custom_manga_presets = current
    return {"preset_id": preset_id, "deleted": bool(existed), "total_custom_presets": len(current)}


def preset_font_diagnostics(presets: Mapping[str, Mapping[str, object]], available_fonts: Optional[Iterable[str]] = None) -> Dict[str, object]:
    """Return missing-font diagnostics for preset libraries without requiring Qt imports."""
    available = {str(f).lower() for f in (available_fonts or []) if str(f or "").strip()}
    if not available:
        return {"checked": False, "missing": {}, "used_fonts": []}
    used = []
    missing = {}
    for preset_id, preset in (presets or {}).items():
        family = str((preset or {}).get("font_family", "") or "").strip()
        if not family:
            continue
        used.append(family)
        if family.lower() not in available:
            missing[str(preset_id)] = family
    return {"checked": True, "missing": missing, "used_fonts": sorted(set(used), key=str.lower)}
