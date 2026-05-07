from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

from .text_rendering import MANGA_PRESETS, normalize_fit_mode, normalize_line_break_strategy, normalize_writing_mode


def _target_pages(project, pages: Optional[Iterable[str]]) -> List[str]:
    all_pages = list((getattr(project, "pages", {}) or {}).keys())
    if pages is None:
        return all_pages
    wanted = [str(p) for p in pages]
    return [p for p in wanted if p in (getattr(project, "pages", {}) or {})]


def normalize_text_style_updates(updates: Dict) -> Dict:
    """Return a safe, renderer-style update dict for project-wide style application."""
    updates = dict(updates or {})
    out: Dict = {}
    if updates.get("preset") in MANGA_PRESETS:
        out["preset"] = str(updates.get("preset"))
    for key in ("font_family", "fallback_font_chain"):
        val = str(updates.get(key, "") or "").strip()
        if val:
            out[key] = val
    if "font_size" in updates or "font_size_px" in updates:
        val = updates.get("font_size", updates.get("font_size_px"))
        try:
            val = float(val)
            if val > 0:
                out["font_size"] = val
        except Exception:
            pass
    for key in ("fit_font_size_min", "fit_font_size_max", "text_padding", "stroke_width", "shadow_radius", "line_spacing", "letter_spacing"):
        if key in updates:
            try:
                val = float(updates.get(key))
                if val >= 0:
                    out[key] = val
            except Exception:
                pass
    if "alignment" in updates:
        try:
            val = int(updates.get("alignment"))
            if val in (0, 1, 2):
                out["alignment"] = val
        except Exception:
            pass
    if updates.get("writing_mode"):
        out["writing_mode"] = normalize_writing_mode(updates.get("writing_mode"))
    if updates.get("fit_mode"):
        out["fit_mode"] = normalize_fit_mode(updates.get("fit_mode"))
    if updates.get("line_break_strategy"):
        out["line_break_strategy"] = normalize_line_break_strategy(updates.get("line_break_strategy"))
    if "auto_fit_font_size" in updates:
        out["auto_fit_font_size"] = bool(updates.get("auto_fit_font_size"))
    return out


def apply_text_style_batch(
    project,
    updates: Dict,
    pages: Optional[Iterable[str]] = None,
    indices: Optional[Sequence[int]] = None,
    only_auto_sized: bool = False,
    dry_run: bool = False,
) -> Dict:
    """Apply a safe batch text-style override to project blocks.

    This utility backs both the UI batch style dialog and headless automation, so
    Koharu-style project-wide font/alignment/fitting requests do not need to be
    reimplemented in mainwindow code.
    """
    updates = normalize_text_style_updates(updates)
    selected_pages = _target_pages(project, pages)
    index_filter = None if indices is None else {int(i) for i in indices}
    changed = 0
    skipped = 0
    touched_pages: List[str] = []
    for page in selected_pages:
        page_changed = 0
        for idx, blk in enumerate((getattr(project, "pages", {}) or {}).get(page, []) or []):
            if index_filter is not None and idx not in index_filter:
                skipped += 1
                continue
            fmt = getattr(blk, "fontformat", None)
            if fmt is None:
                skipped += 1
                continue
            if only_auto_sized and not bool(getattr(fmt, "auto_fit_font_size", False)):
                skipped += 1
                continue
            if not dry_run:
                preset_id = updates.get("preset")
                if preset_id in MANGA_PRESETS:
                    for key, value in MANGA_PRESETS[preset_id].items():
                        if key != "label" and hasattr(fmt, key):
                            setattr(fmt, key, value)
                    fmt.manga_preset = preset_id
                for key, value in updates.items():
                    if key == "preset":
                        continue
                    if hasattr(fmt, key):
                        setattr(fmt, key, value)
                if updates.get("writing_mode"):
                    fmt.vertical = updates.get("writing_mode") == "vertical_rl"
            changed += 1
            page_changed += 1
        if page_changed:
            touched_pages.append(page)
    return {
        "updates": updates,
        "pages": selected_pages,
        "changed": changed,
        "skipped": skipped,
        "touched_pages": touched_pages,
        "dry_run": bool(dry_run),
    }
