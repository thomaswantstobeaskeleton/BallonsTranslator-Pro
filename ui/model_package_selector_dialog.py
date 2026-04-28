"""
First-launch dialog to choose which model packages to download (Issue #15).
Only shown when config file did not exist (new user). User can select packages
(e.g. Core only, or Core + Advanced OCR), skip to Core-only, or run local-only.
"""
from __future__ import annotations

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QCheckBox,
    QScrollArea,
    QWidget,
    QGroupBox,
    QFrame,
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
            cb = QCheckBox(self.tr(label))
            cb.setToolTip(self.tr(desc))
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

    def _selected_package_ids(self):
        return [pid for pid, cb in self._checkboxes.items() if cb.isChecked()]

    def _on_skip(self):
        self._result = ["core"]
        self._offline_local_only = False
        self.accept()

    def _on_local_only(self):
        self._result = []
        self._offline_local_only = True
        self.accept()

    def _on_download(self):
        self._result = self._selected_package_ids()
        if not self._result:
            self._result = ["core"]
        self.accept()

    def get_selected_package_ids(self):
        """Return the selected package IDs (set after dialog is accepted)."""
        return getattr(self, "_result", ["core"])

    def is_offline_local_only_selected(self) -> bool:
        """True if user explicitly selected first-run local-only mode."""
        return bool(getattr(self, "_offline_local_only", False))
