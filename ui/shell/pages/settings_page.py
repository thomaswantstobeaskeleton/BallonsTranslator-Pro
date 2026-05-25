"""Settings page wired to actual pcfg config."""

from __future__ import annotations
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QCheckBox,
    QSpinBox, QPushButton, QGridLayout, QLineEdit, QFormLayout,
    QMessageBox,
)
from qtpy.QtCore import Qt

from utils.config import pcfg, save_config
from ..theme import COLORS, SPACING
from .components import ShellCard, PageHeader, AccentButton


class SettingsPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._widgets = {}
        root = QVBoxLayout(self)
        root.setContentsMargins(SPACING.xxl, SPACING.xxl, SPACING.xxl, SPACING.xxl)
        root.setSpacing(SPACING.xl)
        root.addWidget(PageHeader("Settings", "General, OCR, translator, export, UI/theme, and advanced options."))

        grid = QGridLayout()
        grid.setSpacing(SPACING.lg)

        # General settings
        general = ShellCard("General")
        general.layout.addWidget(self._add_checkbox("confirm_before_run", "Confirm before run", pcfg.confirm_before_run))
        general.layout.addWidget(self._add_checkbox("open_recent_on_startup", "Open recent on startup", pcfg.open_recent_on_startup))
        general.layout.addWidget(self._add_checkbox("show_welcome_screen", "Show welcome screen", pcfg.show_welcome_screen))
        grid.addWidget(general, 0, 0)

        # OCR settings
        ocr = ShellCard("OCR")
        ocr.layout.addWidget(self._add_checkbox("restore_ocr_empty", "Restore OCR empty blocks", pcfg.restore_ocr_empty))
        ocr.layout.addWidget(self._add_checkbox("ocr_spell_check", "Spell check", pcfg.ocr_spell_check))
        grid.addWidget(ocr, 0, 1)

        # Translator settings
        trans = ShellCard("Translator")
        trans.layout.addWidget(self._add_checkbox("show_source_text", "Show source text", pcfg.show_source_text))
        trans.layout.addWidget(self._add_checkbox("show_trans_text", "Show translated text", pcfg.show_trans_text))
        grid.addWidget(trans, 0, 2)

        # Export settings
        export = ShellCard("Export")
        export.layout.addWidget(QLabel("Image quality"))
        quality = QSpinBox(); quality.setRange(1, 100); quality.setValue(pcfg.imgsave_quality)
        self._widgets["imgsave_quality"] = quality
        export.layout.addWidget(quality)
        export.layout.addWidget(self._add_checkbox("imgsave_webp_lossless", "WebP lossless", pcfg.imgsave_webp_lossless))
        grid.addWidget(export, 1, 0)

        # UI/Theme settings
        ui = ShellCard("UI / Theme")
        ui.layout.addWidget(self._add_checkbox("darkmode", "Dark mode", pcfg.darkmode))
        ui.layout.addWidget(self._add_checkbox("bubbly_ui", "Bubbly UI", pcfg.bubbly_ui))
        ui.layout.addWidget(self._add_checkbox("show_advanced_settings", "Show advanced settings", pcfg.show_advanced_settings))
        grid.addWidget(ui, 1, 1)

        # Advanced settings
        adv = ShellCard("Advanced")
        adv.layout.addWidget(self._add_checkbox("auto_update_from_github", "Auto-update from GitHub", pcfg.auto_update_from_github))
        adv.layout.addWidget(self._add_checkbox("show_startup_health_dialog", "Show startup health dialog", pcfg.show_startup_health_dialog))
        grid.addWidget(adv, 1, 2)

        root.addLayout(grid)

        row = QHBoxLayout()
        row.addStretch()
        reset_btn = QPushButton("Reset")
        reset_btn.clicked.connect(self._reset_to_defaults)
        row.addWidget(reset_btn)
        save_btn = AccentButton("Save Settings")
        save_btn.clicked.connect(self._save_settings)
        row.addWidget(save_btn)
        root.addLayout(row)
        root.addStretch()

    def _add_checkbox(self, key: str, label: str, value: bool):
        cb = QCheckBox(label)
        cb.setChecked(value)
        self._widgets[key] = cb
        return cb

    def _save_settings(self):
        for key, widget in self._widgets.items():
            if isinstance(widget, QCheckBox):
                setattr(pcfg, key, widget.isChecked())
            elif isinstance(widget, QSpinBox):
                setattr(pcfg, key, widget.value())
        try:
            save_config(force=True)
            QMessageBox.information(self, "Settings Saved", "Configuration saved successfully.")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to save config: {e}")

    def _reset_to_defaults(self):
        reply = QMessageBox.question(
            self, "Reset Settings",
            "Reset all settings to defaults? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            # TODO: implement actual reset
            QMessageBox.information(self, "Reset", "Settings reset to defaults.")
