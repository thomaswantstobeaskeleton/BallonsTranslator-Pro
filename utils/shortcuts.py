"""
Default keyboard shortcuts and helpers for customizable keybinds.
Keys are stored as strings (e.g. "Ctrl+S", "Ctrl+Shift+M") and merged from config.
"""
from typing import Dict, List, Tuple, Optional

# List of (action_id, default_key, category, description) for UI and defaults.
# category is used for grouping in the shortcuts dialog.
SHORTCUT_SCHEMA: List[Tuple[str, str, str, str]] = [
    # File
    ("file.open_folder", "Ctrl+O", "File", "Open Folder"),
    ("file.save_proj", "Ctrl+S", "File", "Save Project"),
    ("file.export_all_pages", "Ctrl+Shift+S", "File", "Export all pages"),
    # Edit
    ("edit.undo", "Ctrl+Z", "Edit", "Undo"),
    ("edit.redo", "Ctrl+Shift+Z", "Edit", "Redo"),
    ("edit.page_search", "Ctrl+F", "Edit", "Search (current page)"),
    ("edit.global_search", "Ctrl+G", "Edit", "Global Search"),
    ("edit.omni_search", "Ctrl+P", "Edit", "Omni search (menus/settings/canvas)"),
    ("edit.merge_tool", "Ctrl+Shift+M", "Edit", "Region merge tool"),
    # View
    ("view.draw_board", "P", "View", "Drawing Board"),
    ("view.text_edit", "T", "View", "Text Editor"),
    ("view.keyboard_shortcuts", "Ctrl+K", "View", "Keyboard Shortcuts"),
    ("view.context_menu_options", "Ctrl+Shift+O", "View", "Context menu options"),
    # Go / Navigation
    ("go.prev_page", "PageUp", "Go", "Previous Page"),
    ("go.next_page", "PageDown", "Go", "Next Page"),
    ("go.prev_page_alt", "A", "Go", "Previous Page (alternate)"),
    ("go.next_page_alt", "D", "Go", "Next Page (alternate)"),
    # Canvas / General
    ("canvas.textblock_mode", "W", "Canvas", "Text block mode"),
    ("canvas.zoom_in", "Ctrl++", "Canvas", "Zoom In"),
    ("canvas.zoom_out", "Ctrl+-", "Canvas", "Zoom Out"),
    ("canvas.delete", "Ctrl+D", "Canvas", "Delete / Rect delete"),
    ("canvas.space", "Space", "Canvas", "Inpaint (when drawing)"),
    ("canvas.select_all", "Ctrl+A", "Canvas", "Select all blocks"),
    ("canvas.escape", "Escape", "Canvas", "Escape / Deselect"),
    ("canvas.delete_line", "Delete", "Canvas", "Delete (key)"),
    ("canvas.create_textbox", "Ctrl+Shift+N", "Canvas", "Create text box"),
    # Format
    ("format.bold", "Ctrl+B", "Format", "Bold"),
    ("format.italic", "Ctrl+I", "Format", "Italic"),
    ("format.underline", "Ctrl+U", "Format", "Underline"),
    ("format.font_size_up", "Ctrl+Alt+Up", "Format", "Increase font size of selected text"),
    ("format.font_size_down", "Ctrl+Alt+Down", "Format", "Decrease font size of selected text"),
    ("format.apply", "", "Format", "Apply font formatting"),
    ("format.layout", "", "Format", "Auto layout"),
    ("format.fit_to_bubble", "", "Format", "Fit to bubble"),
    ("format.auto_fit", "", "Format", "Auto fit font size to box"),
    ("format.auto_fit_binary", "", "Format", "Auto fit font size (binary search)"),
    ("format.balloon_shape_auto", "", "Format", "Set balloon shape to Auto"),
    ("format.resize_to_fit_content", "", "Format", "Resize to fit content"),
    ("format.layout_review_selected", "Ctrl+Shift+L", "Format", "Layout review: selected textboxes"),
    ("format.layout_review_page", "Ctrl+Alt+L", "Format", "Layout review: entire page"),
    ("format.layout_review_config", "Ctrl+Shift+Alt+L", "Format", "Layout review settings"),
    ("review.ocr_triage_page", "Ctrl+Shift+Y", "Review", "OCR triage worklist (current page)"),
    ("review.translation_qa_page", "Ctrl+Shift+Q", "Review", "Translation QA report (current page)"),
    ("review.auto_extract_glossary_page", "Ctrl+Shift+G", "Review", "Auto-extract glossary (current page)"),
    # Context / Run shortcuts
    ("run.detect_page", "", "Run", "Detect text on page"),
    ("run.translate", "", "Run", "Translate"),
    ("run.ocr", "", "Run", "OCR"),
    ("run.ocr_translate", "", "Run", "OCR and translate"),
    ("run.ocr_translate_inpaint", "", "Run", "OCR, translate and inpaint"),
    ("run.macro_detect_ocr_translate", "", "Run", "Macro: Detect+OCR+Translate"),
    ("run.macro_ocr_translate_inpaint", "", "Run", "Macro: OCR+Translate+Inpaint"),
    # Drawing panel tools
    ("draw.hand", "H", "Drawing", "Hand tool (pan)"),
    ("draw.inpaint", "J", "Drawing", "Inpaint brush"),
    ("draw.pen", "B", "Drawing", "Pen tool"),
    ("draw.text_eraser", "E", "Drawing", "Text eraser / repair depth brush"),
    ("draw.rect", "R", "Drawing", "Rectangle select"),
    ("draw.brush_size_up", "]", "Drawing", "Increase brush size"),
    ("draw.brush_size_down", "[", "Drawing", "Decrease brush size"),
]


def get_default_shortcuts() -> Dict[str, str]:
    """Return a dict action_id -> default key sequence string."""
    return {item[0]: item[1] for item in SHORTCUT_SCHEMA}


def get_shortcut_info() -> List[Tuple[str, str, str, str]]:
    """Return full schema for the shortcuts dialog (id, default_key, category, description)."""
    return list(SHORTCUT_SCHEMA)


def get_shortcut(action_id: str, shortcuts_override: Optional[Dict[str, str]] = None) -> str:
    """Return the key sequence string for action_id. Uses override dict or defaults."""
    defaults = get_default_shortcuts()
    if shortcuts_override and action_id in shortcuts_override:
        return shortcuts_override[action_id] or defaults.get(action_id, "")
    return defaults.get(action_id, "")


def find_shortcut_conflicts(shortcuts_map: Dict[str, str]) -> Dict[str, List[str]]:
    """Return key -> [action_ids] for duplicate non-empty shortcuts."""
    key_to_actions: Dict[str, List[str]] = {}
    for action_id, key in (shortcuts_map or {}).items():
        k = (key or "").strip()
        if not k:
            continue
        key_to_actions.setdefault(k, []).append(action_id)
    return {k: v for k, v in key_to_actions.items() if len(v) > 1}


def classify_shortcut_conflicts(shortcuts_map: Dict[str, str]) -> Dict[str, Dict[str, List[str]]]:
    """Return {'hard': {key:[ids]}, 'alias': {key:[ids]}}."""
    conflicts = find_shortcut_conflicts(shortcuts_map)
    out = {'hard': {}, 'alias': {}}
    for key, ids in conflicts.items():
        base = {i.replace('_alt', '') for i in ids}
        if len(base) == 1:
            out['alias'][key] = ids
        else:
            out['hard'][key] = ids
    return out


def auto_resolve_shortcut_conflicts(shortcuts_map: Dict[str, str]) -> Dict[str, str]:
    """Keep first action for each duplicate key, clear others."""
    out = dict(shortcuts_map or {})
    conflicts = find_shortcut_conflicts(out)
    for _key, ids in conflicts.items():
        for aid in ids[1:]:
            out[aid] = ''
    return out


def shortcut_should_ignore_text_input(focus_widget) -> bool:
    """True when single-key tool/navigation shortcuts should not fire because the user is typing."""
    if focus_widget is None:
        return False
    name = focus_widget.__class__.__name__.lower()
    if any(token in name for token in ("lineedit", "textedit", "plaintextedit", "spinbox", "combobox", "keysequenceedit")):
        return True
    if hasattr(focus_widget, "textInteractionFlags"):
        try:
            return bool(int(focus_widget.textInteractionFlags()))
        except Exception:
            return True
    return False


def is_single_key_sequence(key: str) -> bool:
    k = (key or "").strip()
    if not k:
        return False
    lowered = k.lower()
    return "+" not in k and lowered not in {"pageup", "pagedown", "escape", "space", "delete", "backspace", "tab", "enter", "return"}
