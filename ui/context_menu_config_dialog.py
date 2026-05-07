"""
Context menu options dialog: show/hide canvas right-click menu items by category.
Pinned items are shown at the top of the menu; drag in the pinned list to reorder.
"""
from typing import Dict, List, Optional, Tuple

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QScrollArea,
    QWidget,
    QLabel,
    QCheckBox,
    QPushButton,
    QGroupBox,
    QFrame,
    QListWidget,
    QListWidgetItem,
    QAbstractItemView,
    QSizePolicy,
    QMenu,
    QPlainTextEdit,
    QLineEdit,
)

from utils.config import pcfg, save_config, CONTEXT_MENU_DEFAULT, parse_context_run_macros


def _all_keys_with_labels() -> List[Tuple[str, str]]:
    """Flat list of (key, label) for all context menu items."""
    out: List[Tuple[str, str]] = []
    for _cat, items in CONTEXT_MENU_ITEMS:
        out.extend(items)
    return out


# (category_title, items: [(key, label), ...])
CONTEXT_MENU_ITEMS: List[Tuple[str, List[Tuple[str, str]]]] = [
    ("Edit", [
        ("edit_copy", "Copy"),
        ("edit_paste", "Paste"),
        ("edit_copy_trans", "Copy translation"),
        ("edit_paste_trans", "Paste translation"),
        ("edit_copy_src", "Copy source text"),
        ("edit_paste_src", "Paste source text"),
        ("edit_delete", "Delete"),
        ("edit_delete_recover", "Delete and Recover removed text"),
        ("edit_clear_src", "Clear source text"),
        ("edit_clear_trans", "Clear translation"),
        ("edit_select_all", "Select all"),
    ]),
    ("Text", [
        ("text_spell_src", "Spell check source text"),
        ("text_spell_trans", "Spell check translation"),
        ("text_trim", "Trim whitespace"),
        ("text_upper", "To uppercase"),
        ("text_lower", "To lowercase"),
        ("text_strikethrough", "Toggle strikethrough"),
        ("text_gradient", "Gradient type (submenu)"),
        ("text_on_path", "Text on path (submenu)"),
    ]),
    ("Block", [
        ("block_merge", "Merge selected blocks"),
        ("block_split", "Split selected region(s)"),
        ("block_move_up", "Move block(s) up"),
        ("block_move_down", "Move block(s) down"),
        ("block_triage_add", "Add selected to triage worklist"),
        ("block_triage_mark_reviewed", "Mark selected as reviewed"),
        ("create_textbox", "Create text box"),
    ]),
    ("Image / Overlay", [
        ("overlay_import", "Import Image"),
        ("overlay_clear", "Clear overlay image"),
    ]),
    ("Transform", [
        ("transform_free", "Free Transform"),
        ("transform_reset_warp", "Reset warp"),
        ("transform_warp_preset", "Warp preset (submenu)"),
    ]),
    ("Order", [
        ("order_bring_front", "Bring to front"),
        ("order_send_back", "Send to back"),
    ]),
    ("Format", [
        ("format_apply", "Apply font formatting"),
        ("format_layout", "Auto layout"),
        ("format_fit_to_bubble", "Fit to bubble"),
        ("format_auto_fit", "Auto fit font size to box"),
        ("format_auto_fit_binary", "Auto fit font size (binary search)"),
        ("format_balloon_shape", "Balloon shape (submenu)"),
        ("format_resize_to_fit_content", "Resize to fit content"),
        ("format_fit_to_mask_safe_box", "Fit box to visible mask area"),
        ("format_center_in_bubble", "Center in bubble"),
        ("format_writing_mode", "Writing mode (submenu)"),
        ("format_recenter_text_in_box", "Recenter text in box"),
        ("format_angle", "Reset Angle"),
        ("format_squeeze", "Squeeze"),
        ("format_layout_review_selected", "Layout review selected textboxes"),
        ("format_layout_review_page", "Layout review entire page"),
        ("format_layout_review_config", "Layout review settings"),
    ]),
    ("Run", [
        ("run_detect_region", "Detect text in region"),
        ("run_detect_page", "Detect text on page"),
        ("run_translate", "Translate"),
        ("run_ocr", "OCR"),
        ("run_ocr_translate", "OCR and translate"),
        ("run_ocr_translate_inpaint", "OCR, translate and inpaint"),
        ("run_inpaint", "Inpaint"),
        ("review_ocr_triage_page", "OCR triage worklist (current page)"),
        ("review_translation_qa_page", "Translation QA report (current page)"),
        ("review_auto_extract_glossary_page", "Auto-extract glossary (current page)"),
    ]),
    ("Download image", [
        ("download_image", "Download image (submenu)"),
    ]),
]

KEY_TO_LABEL = dict(_all_keys_with_labels())


class ContextMenuConfigDialog(QDialog):
    """Configure which actions appear in the canvas right-click menu. Pinned items show at top."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Context Menu Options"))
        self.setMinimumSize(460, 560)
        self.resize(520, 620)

        self._checkboxes: Dict[str, QCheckBox] = {}
        self._pinned_keys: List[str] = list(getattr(pcfg, 'context_menu_pinned', None) or [])
        # Keep only keys that exist
        valid_keys = set(KEY_TO_LABEL)
        self._pinned_keys = [k for k in self._pinned_keys if k in valid_keys]

        layout = QVBoxLayout(self)

        hint = QLabel(self.tr(
            "Choose which actions appear in the canvas right-click menu. "
            "Checked = visible, unchecked = hidden. Pin items to the list below to show them at the top of the menu (drag to reorder)."
        ))
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray;")
        layout.addWidget(hint)

        # Pinned (at top of menu)
        pinned_group = QGroupBox(self.tr("Shown at top of menu (drag to reorder)"))
        pinned_layout = QVBoxLayout(pinned_group)
        self._pinned_list = QListWidget()
        self._pinned_list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self._pinned_list.setDefaultDropAction(Qt.DropAction.MoveAction)
        self._pinned_list.setMaximumHeight(120)
        self._pinned_list.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._pinned_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._pinned_list.customContextMenuRequested.connect(self._on_pinned_list_context_menu)
        pinned_layout.addWidget(self._pinned_list)
        clear_pinned_btn = QPushButton(self.tr("Clear all from top"))
        clear_pinned_btn.clicked.connect(self._clear_pinned)
        pinned_layout.addWidget(clear_pinned_btn)
        layout.addWidget(pinned_group)

        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel(self.tr("Filter actions:")))
        self._filter_edit = QLineEdit(self)
        self._filter_edit.setPlaceholderText(self.tr("Search menu actions, categories, or action IDs..."))
        self._filter_edit.textChanged.connect(self._apply_filter)
        filter_row.addWidget(self._filter_edit, 1)
        layout.addLayout(filter_row)

        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel(self.tr("Quick profiles:")))
        btn_edit = QPushButton(self.tr("Editing-focused"))
        btn_edit.clicked.connect(lambda: self._apply_profile("editing"))
        preset_row.addWidget(btn_edit)
        btn_pipe = QPushButton(self.tr("Pipeline-focused"))
        btn_pipe.clicked.connect(lambda: self._apply_profile("pipeline"))
        preset_row.addWidget(btn_pipe)
        btn_layout = QPushButton(self.tr("Layout-focused"))
        btn_layout.clicked.connect(lambda: self._apply_profile("layout"))
        preset_row.addWidget(btn_layout)
        preset_row.addStretch(1)
        layout.addLayout(preset_row)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; }")
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setSpacing(12)

        self._filter_rows = []
        for cat_title, items in CONTEXT_MENU_ITEMS:
            group = QGroupBox(self.tr(cat_title))
            group_layout = QVBoxLayout(group)
            for key, label in items:
                row_widget = QWidget(group)
                row = QHBoxLayout(row_widget)
                row.setContentsMargins(0, 0, 0, 0)
                cb = QCheckBox(self.tr(label))
                cb.setToolTip(self.tr(f"Canvas context menu action: {label} ({key})"))
                cb.setChecked(pcfg.context_menu.get(key, True))
                self._checkboxes[key] = cb
                row.addWidget(cb, 1)
                pin_btn = QPushButton(self.tr("Pin to top"))
                pin_btn.setToolTip(self.tr("Show this action at the top of the canvas context menu."))
                pin_btn.setMinimumWidth(96)
                pin_btn.clicked.connect(lambda checked=False, k=key: self._pin_key(k))
                row.addWidget(pin_btn)
                group_layout.addWidget(row_widget)
                self._filter_rows.append((group, row_widget, cat_title, key, label))
            content_layout.addWidget(group)

        content_layout.addStretch(1)
        scroll.setWidget(content)
        layout.addWidget(scroll)

        macros_group = QGroupBox(self.tr("Run macros (JSON list)"))
        macros_layout = QVBoxLayout(macros_group)
        self._macros_edit = QPlainTextEdit(self)
        self._macros_edit.setPlaceholderText('[{"id":"detect_ocr_translate","label":"Macro: Detect+OCR+Translate","mode":1,"detect_first":true}]')
        try:
            import json
            self._macros_edit.setPlainText(json.dumps(getattr(pcfg, 'context_run_macros', []), ensure_ascii=False, indent=2))
        except Exception:
            self._macros_edit.setPlainText('[]')
        self._macros_edit.setMaximumHeight(140)
        macros_layout.addWidget(self._macros_edit)
        layout.addWidget(macros_group)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch(1)
        restore_btn = QPushButton(self.tr("Restore defaults"))
        restore_btn.clicked.connect(self._restore_defaults)
        ok_btn = QPushButton(self.tr("OK"))
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton(self.tr("Cancel"))
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(restore_btn)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        self._refresh_pinned_list()

    def _apply_filter(self, text: str = None):
        needle = (text if text is not None else self._filter_edit.text()).strip().lower()
        visible_by_group = {}
        for group, row_widget, cat_title, key, label in getattr(self, '_filter_rows', []):
            haystack = f"{cat_title} {key} {label}".lower()
            visible = not needle or needle in haystack
            row_widget.setVisible(visible)
            visible_by_group[group] = visible_by_group.get(group, False) or visible
        for group, visible in visible_by_group.items():
            group.setVisible(visible)

    def _apply_profile(self, profile: str):
        editing_keys = {
            'edit_copy','edit_paste','edit_copy_trans','edit_paste_trans','edit_delete','edit_select_all',
            'text_trim','text_upper','text_lower','format_apply','format_layout','format_fit_to_bubble','create_textbox'
        }
        pipeline_keys = {'run_detect_region','run_detect_page','run_translate','run_ocr','run_ocr_translate','run_ocr_translate_inpaint','run_inpaint'}
        layout_keys = {'format_apply','format_layout','format_fit_to_bubble','format_auto_fit','format_auto_fit_binary','format_balloon_shape','format_resize_to_fit_content','format_fit_to_mask_safe_box','format_center_in_bubble','format_writing_mode','format_recenter_text_in_box','format_angle','format_squeeze'}
        chosen = editing_keys if profile == 'editing' else pipeline_keys if profile == 'pipeline' else layout_keys
        for key, cb in self._checkboxes.items():
            cb.setChecked(key in chosen)

    def _refresh_pinned_list(self):
        self._pinned_list.clear()
        for key in self._pinned_keys:
            label = KEY_TO_LABEL.get(key, key)
            item = QListWidgetItem(self.tr(label))
            item.setData(Qt.ItemDataRole.UserRole, key)
            self._pinned_list.addItem(item)

    def _pin_key(self, key: str):
        if key not in self._pinned_keys:
            self._pinned_keys.append(key)
            self._refresh_pinned_list()

    def _clear_pinned(self):
        self._pinned_keys.clear()
        self._refresh_pinned_list()

    def _on_pinned_list_context_menu(self, pos):
        item = self._pinned_list.itemAt(pos)
        if item is None:
            return
        key = item.data(Qt.ItemDataRole.UserRole)
        if not key:
            return
        m = QMenu(self)
        remove_act = m.addAction(self.tr("Remove from top"))
        if m.exec(self._pinned_list.mapToGlobal(pos)) == remove_act:
            if key in self._pinned_keys:
                self._pinned_keys.remove(key)
            self._refresh_pinned_list()

    def _restore_defaults(self):
        for key, cb in self._checkboxes.items():
            cb.setChecked(CONTEXT_MENU_DEFAULT.get(key, True))
        self._pinned_keys = []
        self._refresh_pinned_list()

    def accept(self):
        if not hasattr(pcfg, 'context_menu'):
            pcfg.context_menu = {}
        for key, cb in self._checkboxes.items():
            pcfg.context_menu[key] = cb.isChecked()
        # Persist pinned order from list widget
        self._pinned_keys = []
        for i in range(self._pinned_list.count()):
            item = self._pinned_list.item(i)
            k = item.data(Qt.ItemDataRole.UserRole)
            if k and k not in self._pinned_keys:
                self._pinned_keys.append(k)
        pcfg.context_menu_pinned = self._pinned_keys
        try:
            pcfg.context_run_macros = parse_context_run_macros(self._macros_edit.toPlainText())
        except Exception as e:
            from qtpy.QtWidgets import QMessageBox
            QMessageBox.warning(self, self.tr('Invalid run macros JSON'), str(e))
            return
        save_config()
        super().accept()
