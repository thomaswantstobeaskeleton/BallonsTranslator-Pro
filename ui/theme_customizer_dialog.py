# -*- coding: utf-8 -*-
"""
Theme and UI customizer: accent color, app font, light/dark, simple vs advanced UI.
"""
from typing import Optional

from qtpy.QtCore import Qt, Signal, QTimer
from qtpy.QtGui import QColor, QFont, QFontDatabase
from qtpy.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QCheckBox,
    QComboBox,
    QSpinBox,
    QGroupBox,
    QWidget,
    QColorDialog,
    QApplication,
)

from utils.config import pcfg, save_config

try:
    from .custom_widget.hover_animation import install_button_animations
    _has_hover_anim = True
except Exception:
    _has_hover_anim = False
    install_button_animations = None


def _hex_from_qcolor(c: QColor) -> str:
    return c.name(QColor.NameFormat.HexRgb)


def _qcolor_from_hex(h: str) -> QColor:
    h = (h or "").strip()
    if not h.startswith("#"):
        h = "#" + h
    return QColor(h)


class ThemeCustomizerDialog(QDialog):
    """Dialog to customize theme (light/dark), accent color, app font, and UI style (simple vs advanced)."""

    theme_applied = Signal(str, int)  # font_family, font_size (so main window applies exactly what was selected)
    ui_mode_changed = Signal(str)
    startup_mode_changed = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Theme & UI Customizer"))
        self.setMinimumWidth(360)

        layout = QVBoxLayout(self)

        # --- Theme (Light / Dark) ---
        theme_grp = QGroupBox(self.tr("Theme"))
        theme_layout = QVBoxLayout(theme_grp)
        self.dark_check = QCheckBox(self.tr("Dark mode"))
        self.dark_check.setChecked(getattr(pcfg, "darkmode", False))
        self.dark_check.setToolTip(self.tr("Use dark theme (Eva Dark). Uncheck for light (Eva Light)."))
        theme_layout.addWidget(self.dark_check)
        layout.addWidget(theme_grp)

        # --- Accent color ---
        accent_grp = QGroupBox(self.tr("Accent color"))
        accent_layout = QHBoxLayout(accent_grp)
        accent_layout.addWidget(QLabel(self.tr("Used for buttons, focus, links (e.g. blue → purple):")))
        self.accent_btn = QPushButton(self.tr("Choose color..."))
        self._accent_hex = (getattr(pcfg, "accent_color_hex", None) or "").strip()
        if not self._accent_hex or not QColor(self._accent_hex).isValid():
            self._accent_hex = "#1E93E5"
        self._update_accent_button_color()
        self.accent_btn.clicked.connect(self._pick_accent_color)
        accent_layout.addWidget(self.accent_btn)
        layout.addWidget(accent_grp)

        # --- App font ---
        font_grp = QGroupBox(self.tr("App font"))
        font_layout = QHBoxLayout(font_grp)
        font_layout.addWidget(QLabel(self.tr("Font family:")))
        self.font_combo = QComboBox(self)
        self.font_combo.setEditable(True)
        for f in QFontDatabase.families():
            self.font_combo.addItem(f)
        app_font = getattr(pcfg, "app_font_family", None) or ""
        if app_font:
            idx = self.font_combo.findText(app_font)
            if idx >= 0:
                self.font_combo.setCurrentIndex(idx)
            else:
                self.font_combo.setCurrentText(app_font)
        else:
            self.font_combo.setCurrentIndex(0)
        font_layout.addWidget(self.font_combo)
        font_layout.addWidget(QLabel(self.tr("Size:")))
        self.font_size_spin = QSpinBox(self)
        self.font_size_spin.setRange(0, 72)
        self.font_size_spin.setSpecialValueText(self.tr("Default"))
        self.font_size_spin.setValue(getattr(pcfg, "app_font_size", 0) or 0)
        self.font_size_spin.setToolTip(self.tr("0 = use system default."))
        font_layout.addWidget(self.font_size_spin)
        layout.addWidget(font_grp)

        # --- UI mode ---
        ui_grp = QGroupBox(self.tr("UI mode"))
        ui_layout = QHBoxLayout(ui_grp)
        ui_layout.addWidget(QLabel(self.tr("Complexity:")))
        self.ui_mode_combo = QComboBox(self)
        self.ui_mode_combo.addItem(self.tr("Simple"), "simple")
        self.ui_mode_combo.addItem(self.tr("Advanced"), "advanced")
        self.ui_mode_combo.addItem(self.tr("Developer"), "developer")
        current_mode = str(getattr(pcfg, "ui_mode", "advanced") or "advanced")
        idx = max(0, self.ui_mode_combo.findData(current_mode))
        self.ui_mode_combo.setCurrentIndex(idx)
        self.ui_mode_combo.setToolTip(self.tr("Simple hides advanced and diagnostic-heavy actions. Advanced keeps full standard menus. Developer keeps all actions and diagnostics."))
        ui_layout.addWidget(self.ui_mode_combo)

        self.show_legacy_checker = QCheckBox(self.tr("Show legacy menu layout"))
        self.show_legacy_checker.setChecked(bool(getattr(pcfg, "show_legacy_menus", True)))
        self.show_legacy_checker.setToolTip(self.tr("Keep the current classic menu layout visible while the new workflow UI rolls out."))
        ui_layout.addWidget(self.show_legacy_checker)
        layout.addWidget(ui_grp)

        startup_grp = QGroupBox(self.tr("Startup"))
        startup_layout = QHBoxLayout(startup_grp)
        startup_layout.addWidget(QLabel(self.tr("Open on launch:")))
        self.startup_mode_combo = QComboBox(self)
        self.startup_mode_combo.addItem(self.tr("Home / Launcher"), "home")
        self.startup_mode_combo.addItem(self.tr("Editor"), "editor")
        self.startup_mode_combo.addItem(self.tr("Last used workflow"), "last_used")
        self.startup_mode_combo.addItem(self.tr("Settings"), "settings")
        self.startup_mode_combo.addItem(self.tr("Live translator"), "live")
        self.startup_mode_combo.addItem(self.tr("Raw downloader"), "downloader")
        self.startup_mode_combo.addItem(self.tr("Batch queue"), "batch")
        self.startup_mode_combo.addItem(self.tr("Models hub"), "models")
        self.startup_mode_combo.addItem(self.tr("Diagnostics"), "diagnostics")
        _sm = str(getattr(pcfg, "startup_mode", "last_used") or "last_used")
        _idx = self.startup_mode_combo.findData(_sm)
        self.startup_mode_combo.setCurrentIndex(_idx if _idx >= 0 else 0)
        self.startup_mode_combo.setToolTip(self.tr("Default startup target when no explicit project path is provided."))
        startup_layout.addWidget(self.startup_mode_combo)
        layout.addWidget(startup_grp)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)
        self.apply_btn = QPushButton(self.tr("Apply"))
        self.apply_btn.clicked.connect(self._apply)
        self.close_btn = QPushButton(self.tr("Close"))
        self.close_btn.clicked.connect(self.accept)
        if _has_hover_anim and install_button_animations:
            install_button_animations(self.apply_btn, normal_opacity=0.9, press_opacity=0.74)
            install_button_animations(self.close_btn, normal_opacity=0.9, press_opacity=0.74)
        btn_layout.addWidget(self.apply_btn)
        btn_layout.addWidget(self.close_btn)
        layout.addLayout(btn_layout)

    def _update_accent_button_color(self):
        c = _qcolor_from_hex(self._accent_hex)
        if c.isValid():
            self.accent_btn.setStyleSheet(
                f"background-color: {c.name(QColor.NameFormat.HexRgb)}; color: {'white' if c.lightness() < 128 else 'black'};"
            )

    def _pick_accent_color(self):
        c = QColorDialog.getColor(_qcolor_from_hex(self._accent_hex), self, self.tr("Accent color"))
        if c.isValid():
            self._accent_hex = _hex_from_qcolor(c)
            self._update_accent_button_color()

    def _apply(self):
        # Run apply twice so font combo has committed by the second run (workaround for Qt editable combo)
        self.font_combo.clearFocus()
        self.font_size_spin.clearFocus()
        QApplication.processEvents()
        self._do_apply()
        QTimer.singleShot(80, self._do_apply)

    def _do_apply(self):
        pcfg.darkmode = self.dark_check.isChecked()
        pcfg.accent_color_hex = self._accent_hex
        idx = self.font_combo.currentIndex()
        family = (self.font_combo.itemText(idx) if idx >= 0 else self.font_combo.currentText() or "").strip()
        size = max(0, self.font_size_spin.value())
        pcfg.app_font_family = family
        pcfg.app_font_size = size
        pcfg.ui_mode = str(self.ui_mode_combo.currentData() or "advanced")
        pcfg.show_legacy_menus = bool(self.show_legacy_checker.isChecked())
        pcfg.startup_mode = str(self.startup_mode_combo.currentData() or "last_used")
        save_config()
        self.theme_applied.emit(family, size)
        self.ui_mode_changed.emit(pcfg.ui_mode)
        self.startup_mode_changed.emit(pcfg.startup_mode)
