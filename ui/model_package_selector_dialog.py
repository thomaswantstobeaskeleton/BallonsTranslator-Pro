"""
First-launch dialog to choose which model packages to download (Issue #15).
Only shown when config file did not exist (new user). User can select packages
(e.g. Core only, or Core + Advanced OCR), skip to Core-only, or run local-only.
"""
from __future__ import annotations

from qtpy.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QCheckBox,
    QGroupBox,
    QFrame,
    QMessageBox,
    QComboBox,
)
try:
    from qtpy.QtWidgets import QRadioButton
except ImportError:
    QRadioButton = QCheckBox

import utils.model_packages as _model_packages
from utils.config import pcfg, save_config
from utils.shared import DISPLAY_LANGUAGE_MAP
MODEL_PACKAGES = _model_packages.MODEL_PACKAGES
PACKAGE_LABELS = _model_packages.PACKAGE_LABELS
MODEL_PACKAGE_PRESETS = getattr(_model_packages, "MODEL_PACKAGE_PRESETS", {"core": {"label": "Core only"}})
PACKAGE_TIERS = getattr(_model_packages, "PACKAGE_TIERS", {})
DEFAULT_MODEL_PACKAGE_PRESET_ID = getattr(_model_packages, "DEFAULT_MODEL_PACKAGE_PRESET_ID", next(iter(MODEL_PACKAGE_PRESETS)))
get_package_ids_for_preset = getattr(_model_packages, "get_package_ids_for_preset", lambda pid: ["core"] if pid == "core" else [])

def _dark_first_launch_stylesheet() -> str:
    """Keep first-launch model selection readable even before global theme is applied."""
    return """
QDialog { background-color: #21252b; color: #e6e6e6; }
QLabel { color: #e6e6e6; }
QGroupBox { border: 1px solid #3b3f4c; border-radius: 8px; margin-top: 12px; padding-top: 8px; color: #f0f0f0; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
QCheckBox, QRadioButton { color: #e6e6e6; }
QPushButton {
    background-color: #2f3440;
    color: #f2f2f2;
    border: 1px solid #4a5060;
    border-radius: 6px;
    padding: 6px 12px;
}
QPushButton:hover { background-color: #3a404d; }
QPushButton:pressed { background-color: #2a2f39; }
"""

class ModelPackageSelectorDialog(QDialog):
    """Let the user choose which model packages to download at first launch."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Choose models to download"))
        self.setMinimumSize(560, 420)
        self.resize(640, 500)
        self._checkboxes = {}
        self._preset_radios = {}
        self._advanced_mode_checkbox = None
        self._presets_group = None
        self._advanced_group = None
        self._result = ["core"]
        self._result_preset_ids = []
        self._build_ui()
        self.setStyleSheet(_dark_first_launch_stylesheet())

    def _build_ui(self):
        layout = QVBoxLayout(self)

        lang_row = QHBoxLayout()
        lang_label = QLabel(self.tr("Display language"))
        self._lang_combo = QComboBox(self)
        self._lang_combo.addItems(list(DISPLAY_LANGUAGE_MAP.keys()))
        self._lang_combo.setCurrentText(self._display_lang_to_label(getattr(pcfg, "display_lang", "English")))
        self._lang_combo.currentTextChanged.connect(self._on_display_lang_changed)
        lang_row.addWidget(lang_label)
        lang_row.addWidget(self._lang_combo, 1)
        layout.addLayout(lang_row)

        intro = QLabel(
            self.tr(
                "Select which model packages to download now. You can download more later via Tools → Manage models."
            )
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        tier_order = {"Stable": 0, "Beta": 1, "Experimental": 2, "External dependency heavy": 3}
        sorted_package_ids = sorted(
            MODEL_PACKAGES.keys(),
            key=lambda pid: (tier_order.get(PACKAGE_TIERS.get(pid, "Stable"), 99), pid),
        )
        presets_group = QGroupBox(self.tr("Recommended presets"))
        presets_layout = QVBoxLayout(presets_group)
        for preset_id, preset in MODEL_PACKAGE_PRESETS.items():
            label = preset.get("label", preset_id)
            intended_use = preset.get("intended_use", "")
            approx_size = preset.get("approx_size", "")
            deps = preset.get("dependency_hints", "")
            details = self.tr("Approx size: {size}. Use: {use}. Dependency hints: {deps}.").format(
                size=approx_size,
                use=intended_use,
                deps=deps,
            )
            rb = QRadioButton(self.tr(label))
            rb.setToolTip(details)
            rb.setProperty("preset_id", preset_id)
            self._preset_radios[preset_id] = rb
            presets_layout.addWidget(rb)
            details_label = QLabel(details)
            details_label.setWordWrap(True)
            if hasattr(details_label, "setStyleSheet"):
                details_label.setStyleSheet("color: #b8becd; margin-left: 22px;")
            presets_layout.addWidget(details_label)

        default_rb = self._preset_radios.get(DEFAULT_MODEL_PACKAGE_PRESET_ID)
        if default_rb is not None:
            default_rb.setChecked(True)

        self._presets_group = presets_group
        layout.addWidget(presets_group)

        self._advanced_mode_checkbox = QCheckBox(self.tr("Advanced/custom mode (power users)"))
        self._advanced_mode_checkbox.setToolTip(
            self.tr("Manually select low-level packages instead of using a preset.")
        )
        if hasattr(self._advanced_mode_checkbox, "toggled"):
            self._advanced_mode_checkbox.toggled.connect(self._sync_mode_ui)
        layout.addWidget(self._advanced_mode_checkbox)

        advanced_group = QGroupBox(self.tr("Manual package selection"))
        group_layout = QVBoxLayout(advanced_group)
        for package_id in sorted_package_ids:
            label, desc = PACKAGE_LABELS.get(package_id, (package_id, ""))
            cb = QCheckBox(self.tr(label))
            cb.setToolTip(self.tr(desc))
            cb.setProperty("package_id", package_id)
            if package_id == "core":
                cb.setChecked(True)
            self._checkboxes[package_id] = cb
            group_layout.addWidget(cb)
        self._advanced_group = advanced_group
        layout.addWidget(advanced_group)
        if QRadioButton is QCheckBox and not hasattr(self._advanced_mode_checkbox, "toggled"):
            self._advanced_mode_checkbox.setChecked(True)
        self._sync_mode_ui(self._advanced_mode_checkbox.isChecked())

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        skip_btn = QPushButton(self.tr("Skip (Core only)"))
        skip_btn.setToolTip(self.tr("Download only the Core package (recommended minimum)."))
        skip_btn.clicked.connect(self._on_skip)
        local_only_btn = QPushButton(self.tr("Skip all downloads (local-only)"))
        local_only_btn.setToolTip(self.tr("Do not download any model package now. Use only already-local modules/files."))
        local_only_btn.clicked.connect(self._on_local_only)
        download_btn = QPushButton(self.tr("Download selected"))
        download_btn.setDefault(True)
        download_btn.setFocus()
        download_btn.clicked.connect(self._on_download)
        btn_row.addWidget(skip_btn)
        btn_row.addWidget(local_only_btn)
        btn_row.addWidget(download_btn)
        layout.addLayout(btn_row)

    def _display_lang_to_label(self, code: str) -> str:
        for label, lang_code in DISPLAY_LANGUAGE_MAP.items():
            if lang_code == code:
                return label
        return "English"

    def _on_display_lang_changed(self, text: str):
        code = DISPLAY_LANGUAGE_MAP.get(text)
        if not code:
            return
        pcfg.display_lang = code
        save_config()

    def _sync_mode_ui(self, advanced_enabled: bool):
        if self._advanced_group is not None and hasattr(self._advanced_group, "setVisible"):
            self._advanced_group.setVisible(bool(advanced_enabled))
        if self._presets_group is not None and hasattr(self._presets_group, "setEnabled"):
            self._presets_group.setEnabled(not advanced_enabled)

    def _selected_preset_id(self):
        for pid, rb in self._preset_radios.items():
            if rb.isChecked():
                return pid
        return DEFAULT_MODEL_PACKAGE_PRESET_ID

    def _selected_package_ids(self):
        if self._advanced_mode_checkbox is not None and self._advanced_mode_checkbox.isChecked():
            return [pid for pid, cb in self._checkboxes.items() if cb.isChecked()]
        return get_package_ids_for_preset(self._selected_preset_id())

    def _selected_preset_ids(self):
        if self._advanced_mode_checkbox is not None and self._advanced_mode_checkbox.isChecked():
            return ["custom"]
        return [self._selected_preset_id()]

    def _accept_selection(self):
        self._result_preset_ids = self._selected_preset_ids()
        self._result = self._selected_package_ids()
        if not self._result:
            self._result = ["core"]
            self._result_preset_ids = ["core_minimal"]

    def _on_skip(self):
        self._result = ["core"]
        self._offline_local_only = False
        self.accept()

    def _on_local_only(self):
        self._result = []
        self._offline_local_only = True
        self._result_preset_ids = ["core_minimal"]
        self.accept()

    def _on_download(self):
        self._accept_selection()
        self._result = self._selected_package_ids()
        if not self._result:
            self._result = ["core"]
        elif "core" not in self._result:
            answer = QMessageBox.warning(
                self,
                self.tr("Core package not selected"),
                self.tr(
                    "You did not select the Core package. Default modules may be unavailable until you download them.\n\n"
                    "Continue with advanced-only packages?"
                ),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self.accept()

    def get_selected_package_ids(self):
        """Return selected low-level package IDs (used by downloader)."""
        return getattr(self, "_result", ["core"])

    def is_offline_local_only_selected(self) -> bool:
        """True if user explicitly selected first-run local-only mode."""
        return bool(getattr(self, "_offline_local_only", False))
    def get_selected_preset_ids(self):
        """Return selected preset ID(s) for reproducible setup metadata."""
        return getattr(self, "_result_preset_ids", [])
