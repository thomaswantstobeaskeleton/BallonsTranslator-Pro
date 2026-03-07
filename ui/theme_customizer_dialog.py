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

        # --- UI style ---
        ui_grp = QGroupBox(self.tr("UI style"))
        ui_layout = QVBoxLayout(ui_grp)
        self.bubbly_check = QCheckBox(self.tr("Advanced UI (rounder corners, gradients)"))
        self.bubbly_check.setChecked(getattr(pcfg, "bubbly_ui", True))
        self.bubbly_check.setToolTip(self.tr("Uncheck for a simpler, flatter look."))
        ui_layout.addWidget(self.bubbly_check)
        layout.addWidget(ui_grp)

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
        pcfg.bubbly_ui = self.bubbly_check.isChecked()
        save_config()
        self.theme_applied.emit(family, size)
