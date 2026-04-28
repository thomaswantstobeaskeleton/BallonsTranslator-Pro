"""
First-launch dialog to choose which model packages to download (Issue #15).
Only shown when config file did not exist (new user). User can select packages
(e.g. Core only, or Core + Advanced OCR) then download; or skip to use Core only.
"""
from __future__ import annotations

from qtpy.QtCore import QCoreApplication
from qtpy.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QCheckBox,
    QGroupBox,
)

from utils.model_packages import MODEL_PACKAGES, PACKAGE_LABELS


class ModelPackageSelectorDialog(QDialog):
    """Let the user choose which model packages to download at first launch."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Choose models to download"))
        self.setMinimumSize(480, 340)
        self.resize(520, 400)
        self._checkboxes = {}
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        intro = QLabel(
            self.tr(
                "Select which model packages to download now. You can download more later via Tools → Manage models."
            )
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        group = QGroupBox(self.tr("Model packages"))
        group_layout = QVBoxLayout(group)
        for package_id in MODEL_PACKAGES:
            label, desc = PACKAGE_LABELS.get(package_id, (package_id, ""))
            cb = QCheckBox(QCoreApplication.translate("ModelPackageCatalog", label))
            cb.setToolTip(QCoreApplication.translate("ModelPackageCatalog", desc))
            cb.setProperty("package_id", package_id)
            if package_id == "core":
                cb.setChecked(True)
            self._checkboxes[package_id] = cb
            group_layout.addWidget(cb)
        layout.addWidget(group)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        skip_btn = QPushButton(self.tr("Skip (Core only)"))
        skip_btn.setToolTip(self.tr("Download only the Core package (recommended minimum)."))
        skip_btn.clicked.connect(self._on_skip)
        download_btn = QPushButton(self.tr("Download selected"))
        download_btn.setDefault(True)
        download_btn.setFocus()
        download_btn.clicked.connect(self._on_download)
        btn_row.addWidget(skip_btn)
        btn_row.addWidget(download_btn)
        layout.addLayout(btn_row)

    def _selected_package_ids(self):
        return [pid for pid, cb in self._checkboxes.items() if cb.isChecked()]

    def _on_skip(self):
        self._result = ["core"]
        self.accept()

    def _on_download(self):
        self._result = self._selected_package_ids()
        if not self._result:
            self._result = ["core"]
        self.accept()

    def get_selected_package_ids(self):
        """Return the selected package IDs (set after dialog is accepted)."""
        return getattr(self, "_result", ["core"])
