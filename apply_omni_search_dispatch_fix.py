#!/usr/bin/env python3
"""Patch ui/mainwindowbars.py so the top-bar omni search actually dispatches.

Run from the repository root:

    python apply_omni_search_dispatch_fix.py

Or pass a custom file path:

    python apply_omni_search_dispatch_fix.py path/to/ui/mainwindowbars.py

The script keeps a .bak copy and is idempotent.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

START = "    # >>> OMNI_SEARCH_DISPATCH_FIX_START\n"
END = "    # <<< OMNI_SEARCH_DISPATCH_FIX_END\n"

METHOD_BLOCK = r'''
    # >>> OMNI_SEARCH_DISPATCH_FIX_START
    def _setup_omni_search(self):
        """Create/wire a deterministic omni command palette.

        The previous omni box could open visually, but selected items did not
        reliably execute. This version builds a display-label -> callable map,
        wires Enter and completer activation, and calls the selected feature or
        menu directly.
        """
        if getattr(self, "_omni_search_ready", False):
            self._rebuild_omni_commands()
            self._refresh_omni_model()
            return

        box = getattr(self, "omniSearch", None)
        if box is None:
            box = getattr(self, "omniSearchEdit", None)
        if box is None:
            box = QLineEdit(self)
            box.setObjectName("OmniSearch")
            box.setPlaceholderText(self.tr("Search menus, settings, actions..."))
            try:
                box.setClearButtonEnabled(True)
            except Exception:
                pass
            layout = self.layout()
            if layout is not None and layout.indexOf(box) < 0:
                layout.addWidget(box)
            self.omniSearch = box

        self._searchExpanded = bool(getattr(self, "_searchExpanded", False))
        if not self._searchExpanded:
            box.setFixedWidth(0)
            box.hide()

        self._omni_command_map = {}
        self._rebuild_omni_commands()

        self._omni_model = QStandardItemModel(self)
        self._refresh_omni_model()

        self._omni_proxy = QSortFilterProxyModel(self)
        self._omni_proxy.setSourceModel(self._omni_model)
        self._omni_proxy.setFilterKeyColumn(0)
        try:
            self._omni_proxy.setDynamicSortFilter(True)
        except Exception:
            pass
        try:
            self._omni_proxy.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        except Exception:
            try:
                self._omni_proxy.setFilterCaseSensitivity(Qt.CaseInsensitive)
            except Exception:
                pass

        comp = QCompleter(self._omni_proxy, self)
        try:
            comp.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        except Exception:
            comp.setCompletionMode(QCompleter.PopupCompletion)
        try:
            comp.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        except Exception:
            try:
                comp.setCaseSensitivity(Qt.CaseInsensitive)
            except Exception:
                pass
        try:
            comp.setFilterMode(Qt.MatchFlag.MatchContains)
        except Exception:
            try:
                comp.setFilterMode(Qt.MatchContains)
            except Exception:
                pass
        box.setCompleter(comp)
        self.omniCompleter = comp

        # Remove stale/old handlers if they exist, then install deterministic ones.
        for signal in (box.textEdited, box.returnPressed):
            try:
                signal.disconnect()
            except Exception:
                pass
        box.textEdited.connect(self._on_omni_text_edited)
        box.returnPressed.connect(self._activate_current_omni_command)

        for sig_getter in (
            lambda: comp.activated[str],
            lambda: comp.activated[QModelIndex],
            lambda: comp.activated,
        ):
            try:
                sig_getter().disconnect()
            except Exception:
                pass

        try:
            comp.activated[str].connect(self._activate_omni_command)
        except Exception:
            try:
                comp.activated.connect(self._activate_omni_command)
            except Exception:
                pass
        try:
            comp.activated[QModelIndex].connect(self._activate_omni_command)
        except Exception:
            pass

        self._omni_search_ready = True

    def _add_omni_command(self, label: str, callback):
        if not label or not callable(callback):
            return
        self._omni_command_map[str(label)] = callback

    def _rebuild_omni_commands(self):
        """Build display-label -> callable command map."""
        self._omni_command_map = {}
        mw = getattr(self, "mainwindow", None)
        if mw is None:
            return

        def add(label, callback):
            self._add_omni_command(label, callback)

        def add_method(label, owner, method_name):
            if owner is None:
                return
            fn = getattr(owner, method_name, None)
            if callable(fn):
                add(label, fn)

        def add_signal(label, signal_owner, signal_attr):
            sig = getattr(signal_owner, signal_attr, None) if signal_owner is not None else None
            if sig is not None and hasattr(sig, "emit"):
                add(label, sig.emit)

        def show_menu(btn):
            def _show():
                try:
                    if btn is not None and hasattr(btn, "showMenu"):
                        btn.showMenu()
                except Exception:
                    pass
            return _show

        left = getattr(mw, "leftBar", None)

        # Menus / primary panels.
        if left is not None:
            add("Menu: File / Open", show_menu(getattr(left, "openBtn", None)))
        for label, attr in (
            ("Menu: Edit", "editToolBtn"),
            ("Menu: View", "viewToolBtn"),
            ("Menu: Go", "goToolBtn"),
            ("Menu: Tools", "toolsToolBtn"),
            ("Menu: Pipeline", "pipelineToolBtn"),
            ("Menu: Format", "formatToolBtn"),
            ("Menu: Review", "reviewToolBtn"),
        ):
            btn = getattr(self, attr, None)
            if btn is not None:
                add(label, show_menu(btn))

        # File.
        add_method("File: Open folder...", left, "onOpenFolder")
        add_method("File: Open images...", left, "onOpenImages")
        add_method("File: Open ACBF/CBZ...", left, "onOpenACBFCBZ")
        add_method("File: Open CBR...", left, "onOpenCBR")
        add_method("File: Open project JSON...", left, "onOpenProj")
        add_method("File: Close project and show welcome", mw, "close_project_and_show_welcome")
        add_method("File: Save project", mw, "manual_save")
        add_method("File: Export all pages", mw, "on_batch_export")
        add_method("File: Export all pages as...", mw, "on_batch_export_as")
        add_method("File: Export current page as...", mw, "on_export_current_page_as")
        add_method("File: Export source text as TXT", mw, "on_export_src_txt")
        add_method("File: Export translation as TXT", mw, "on_export_trans_txt")
        add_method("File: Import translation from TXT/markdown", mw, "on_import_trans_txt")

        # View / navigation.
        add_method("View: Welcome screen", mw, "_show_welcome_screen")
        add_method("View: Canvas / project workspace", mw, "setupImgTransUI")
        add_method("View: Config", mw, "setupConfigUI")
        add_method("View: Text editor", mw, "shortcutTextedit")
        add_method("View: Drawing board", mw, "shortcutDrawboard")
        add_method("View: Spell check panel", mw, "shortcutSpellCheckPanel")
        add_method("View: Keyboard shortcuts", mw, "open_shortcuts_dialog")
        add_method("View: Theme and UI customizer", mw, "open_theme_customizer")
        add_method("View: Context menu options", mw, "shortcutContextMenuOptions")
        add_method("View: Global search panel", mw, "on_global_search")
        add_method("View: Page search", mw, "on_page_search")

        # Edit / search.
        add_method("Edit: Undo", mw, "on_undo")
        add_method("Edit: Redo", mw, "on_redo")
        add_method("Edit: Search current page", mw, "on_page_search")
        add_method("Edit: Global search", mw, "on_global_search")
        add_method("Edit: Translation context", mw, "show_translation_context_dialog")
        add_method("Edit: OCR keyword substitution", mw, "show_OCR_keyword_window")
        add_method("Edit: Pre-MT keyword substitution", mw, "show_pre_MT_keyword_window")
        add_method("Edit: MT keyword substitution", mw, "show_MT_keyword_window")

        # Pipeline / tools.
        add_method("Pipeline: Run", mw, "run_imgtrans")
        add_method("Pipeline: Full preset", mw, "on_run_preset_full")
        add_method("Pipeline: Detect + OCR preset", mw, "on_run_preset_detect_ocr")
        add_method("Pipeline: Translate preset", mw, "on_run_preset_translate")
        add_method("Pipeline: Inpaint preset", mw, "on_run_preset_inpaint")
        add_method("Pipeline: Re-run detection only", mw, "on_re_run_detection_only")
        add_method("Pipeline: Re-run OCR only", mw, "on_re_run_ocr_only")
        add_method("Tools: Region merge tool", mw, "on_open_merge_tool")
        add_method("Tools: Batch queue", mw, "on_open_batch_queue")
        add_method("Tools: Manga / Comic source", mw, "on_open_manga_source")
        add_method("Tools: Manage models", mw, "on_open_manage_models")
        add_method("Tools: Retry model downloads", mw, "on_retry_model_downloads")
        add_method("Tools: Runtime resource summary", mw, "on_runtime_resource_summary")
        add_method("Tools: Release model caches", mw, "on_release_model_caches")
        add_method("Tools: Clear pipeline caches", mw, "on_clear_pipeline_caches")
        add_method("Tools: Environment doctor", mw, "on_environment_doctor")
        add_method("Tools: Video translator", mw, "on_video_translator")
        add_method("Tools: Subtitle file translator", mw, "on_subtitle_file_translator")
        add_method("Tools: Video subtitle editor", mw, "on_video_subtitle_editor")

        # Format / review.
        add_method("Format: Apply formatting", mw, "shortcutFormatApply")
        add_method("Format: Auto layout", mw, "shortcutFormatLayout")
        add_method("Format: Fit to bubble", mw, "shortcutFitToBubble")
        add_method("Format: Auto fit font size", mw, "shortcutFormatAutoFit")
        add_method("Format: Auto fit font size binary", mw, "shortcutFormatAutoFitBinary")
        add_method("Format: Resize to fit content", mw, "shortcutResizeToFitContent")
        add_method("Review: OCR triage", mw, "on_open_ocr_triage_current_page")
        add_method("Review: Translation QA report", mw, "on_translation_qa_report_current_page")
        add_method("Review: Auto extract glossary", mw, "on_auto_extract_glossary_current_page")
        add_method("Review: Layout review selected", mw, "shortcutLayoutReviewSelected")
        add_method("Review: Layout review page", mw, "shortcutLayoutReviewPage")
        add_method("Review: Layout review settings", mw, "shortcutLayoutReviewConfig")

        # Theme/help.
        add_method("Theme: Light", mw, "_on_theme_light_triggered")
        add_method("Theme: Dark", mw, "_on_theme_dark_triggered")
        add_method("Help: Documentation", mw, "on_help_documentation")
        add_method("Help: About", mw, "on_help_about")
        add_method("Help: Update from GitHub", mw, "on_update_from_github")

        # Also register QAction trigger signals when present, so menu-backed actions
        # that do not have public MainWindow methods still work.
        for attr in dir(self):
            if not attr.endswith("_trigger"):
                continue
            sig = getattr(self, attr, None)
            if sig is not None and hasattr(sig, "emit"):
                label = "Action: " + attr[:-8].replace("_", " ").strip().title()
                add(label, sig.emit)

    def _refresh_omni_model(self):
        model = getattr(self, "_omni_model", None)
        if model is None:
            return
        model.clear()
        for label in sorted((getattr(self, "_omni_command_map", {}) or {}).keys(), key=lambda s: s.lower()):
            item = QStandardItem(label)
            model.appendRow(item)

    def _set_omni_filter(self, text: str):
        proxy = getattr(self, "_omni_proxy", None)
        if proxy is None:
            return
        pattern = QRegularExpression.escape(text or "")
        try:
            proxy.setFilterRegularExpression(QRegularExpression(pattern))
        except Exception:
            try:
                proxy.setFilterRegExp(pattern)
            except Exception:
                pass

    def _on_omni_text_edited(self, text: str):
        self._set_omni_filter(text)
        comp = getattr(self, "omniCompleter", None)
        if comp is None:
            return
        try:
            popup = comp.popup()
            if popup is not None:
                popup.setMinimumWidth(max(320, self.omniSearch.width()))
        except Exception:
            pass
        comp.complete()

    def _omni_label_from_value(self, value) -> str:
        if isinstance(value, QModelIndex):
            try:
                return str(value.data() or "")
            except Exception:
                return ""
        return str(value or "")

    def _activate_current_omni_command(self):
        text = str(self.omniSearch.text() or "").strip()
        if text in self._omni_command_map:
            self._activate_omni_command(text)
            return

        comp = getattr(self, "omniCompleter", None)
        if comp is not None:
            try:
                current = str(comp.currentCompletion() or "").strip()
                if current:
                    self._activate_omni_command(current)
                    return
            except Exception:
                pass

        proxy = getattr(self, "_omni_proxy", None)
        if proxy is not None and proxy.rowCount() > 0:
            try:
                first = str(proxy.index(0, 0).data() or "").strip()
                if first:
                    self._activate_omni_command(first)
                    return
            except Exception:
                pass

    def _activate_omni_command(self, value):
        label = self._omni_label_from_value(value).strip()
        commands = getattr(self, "_omni_command_map", {}) or {}
        fn = commands.get(label)

        if fn is None:
            lowered = label.lower()
            for key, candidate in commands.items():
                if key.lower() == lowered:
                    label, fn = key, candidate
                    break

        if fn is None:
            needle = label.lower()
            for key, candidate in commands.items():
                if needle and needle in key.lower():
                    label, fn = key, candidate
                    break

        if fn is None:
            return

        self._hide_omni_search(clear=True)
        QTimer.singleShot(0, fn)

    def _show_omni_search(self):
        if getattr(self, "omniSearch", None) is None:
            self._setup_omni_search()
        self._searchExpanded = True
        self._rebuild_omni_commands()
        self._refresh_omni_model()
        self._set_omni_filter(self.omniSearch.text())
        self.omniSearch.show()
        self.omniSearch.setFixedWidth(320)
        self.omniSearch.setFocus(Qt.FocusReason.ShortcutFocusReason)
        try:
            self.omniSearch.selectAll()
        except Exception:
            pass
        comp = getattr(self, "omniCompleter", None)
        if comp is not None:
            QTimer.singleShot(0, comp.complete)

    def _hide_omni_search(self, clear: bool = False):
        if getattr(self, "omniSearch", None) is None:
            return
        if clear:
            self.omniSearch.clear()
        comp = getattr(self, "omniCompleter", None)
        if comp is not None:
            try:
                comp.popup().hide()
            except Exception:
                pass
        self._searchExpanded = False
        self.omniSearch.clearFocus()
        self.omniSearch.setFixedWidth(0)
        self.omniSearch.hide()

    def _toggle_omni_search(self):
        if getattr(self, "omniSearch", None) is None:
            self._setup_omni_search()
        if self.omniSearch.isVisible() and self.omniSearch.width() > 8:
            self._hide_omni_search(clear=False)
        else:
            self._show_omni_search()
    # <<< OMNI_SEARCH_DISPATCH_FIX_END
'''


def ensure_name_in_import_tuple(text: str, module: str, name: str) -> str:
    pattern = rf"from {re.escape(module)} import \((.*?)\)"
    m = re.search(pattern, text, flags=re.S)
    if not m:
        return text
    block = m.group(1)
    if re.search(rf"\b{re.escape(name)}\b", block):
        return text
    insert = block.rstrip() + f",\n    {name},\n"
    return text[:m.start(1)] + insert + text[m.end(1):]


def ensure_imports(text: str) -> str:
    for name in ("QLineEdit", "QCompleter"):
        text = ensure_name_in_import_tuple(text, "qtpy.QtWidgets", name)
    for name in ("QSortFilterProxyModel", "QModelIndex", "QRegularExpression", "QTimer"):
        text = ensure_name_in_import_tuple(text, "qtpy.QtCore", name)
    for name in ("QStandardItemModel", "QStandardItem"):
        text = ensure_name_in_import_tuple(text, "qtpy.QtGui", name)
    return text


def remove_old_block(text: str) -> str:
    start = text.find(START)
    if start == -1:
        return text
    end = text.find(END, start)
    if end == -1:
        return text
    end += len(END)
    return text[:start] + text[end:]


def find_titlebar_bounds(text: str) -> tuple[int, int]:
    class_match = re.search(r"^class TitleBar\b.*?:\n", text, flags=re.M)
    if not class_match:
        raise RuntimeError("Could not find class TitleBar")
    start = class_match.start()
    next_class = re.search(r"^class \w+\b.*?:\n", text[class_match.end():], flags=re.M)
    if next_class:
        end = class_match.end() + next_class.start()
    else:
        end = len(text)
    return start, end


def insert_setup_call(text: str) -> str:
    if "self._setup_omni_search()" in text:
        return text

    class_start, class_end = find_titlebar_bounds(text)
    sub = text[class_start:class_end]
    init_match = re.search(r"^    def __init__\(.*?\):\n", sub, flags=re.M)
    if not init_match:
        raise RuntimeError("Could not find TitleBar.__init__")

    init_body_start = class_start + init_match.end()
    next_method = re.search(r"^    def \w+\(", text[init_body_start:class_end], flags=re.M)
    if not next_method:
        raise RuntimeError("Could not find end of TitleBar.__init__")
    insert_at = init_body_start + next_method.start()

    call = "        # Build omni search after menus/actions are created.\n        self._setup_omni_search()\n\n"
    return text[:insert_at] + call + text[insert_at:]


def insert_methods(text: str) -> str:
    text = remove_old_block(text)
    class_start, class_end = find_titlebar_bounds(text)
    return text[:class_end] + "\n" + METHOD_BLOCK + "\n" + text[class_end:]


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("ui/mainwindowbars.py")
    if not path.exists():
        print(f"ERROR: {path} does not exist", file=sys.stderr)
        return 2

    text = path.read_text(encoding="utf-8")
    original = text
    text = ensure_imports(text)
    text = insert_setup_call(text)
    text = insert_methods(text)

    if text == original:
        print(f"No changes needed: {path}")
        return 0

    backup = path.with_suffix(path.suffix + ".bak")
    if not backup.exists():
        backup.write_text(original, encoding="utf-8")
    path.write_text(text, encoding="utf-8")
    print(f"Patched {path}")
    print(f"Backup: {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
