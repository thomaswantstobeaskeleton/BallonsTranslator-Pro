"""
Export dialog: choose output folder and format for batch export (#126).
"""

import os.path as osp
from qtpy.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QComboBox,
    QFormLayout,
)
from qtpy.QtCore import Qt


EXPORT_FORMATS = [
    ('.png', 'PNG'),
    ('.jpg', 'JPEG'),
    ('.webp', 'WebP'),
    ('.jxl', 'JXL'),
]


class ExportFormatDialog(QDialog):
    """Dialog to choose output folder and image format for Export all pages as."""

    def __init__(self, parent=None, initial_dir: str = None):
        super().__init__(parent)
        self.setWindowTitle(self.tr('Export all pages as...'))
        self._out_dir = initial_dir or ''
        self._format_index = 0
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.folder_btn = QPushButton(self.tr('Choose folder...'))
        self.folder_label = QLabel(self._out_dir or self.tr('(none)'))
        self.folder_label.setMinimumWidth(280)
        self.folder_btn.clicked.connect(self._pick_folder)
        form.addRow(self.tr('Output folder:'), self.folder_label)
        row = QHBoxLayout()
        row.addWidget(self.folder_btn)
        form.addRow(row)

        self.format_combo = QComboBox()
        for ext, name in EXPORT_FORMATS:
            self.format_combo.addItem(f'{name} ({ext})', ext)
        form.addRow(self.tr('Format:'), self.format_combo)
        layout.addLayout(form)

        btns = QHBoxLayout()
        btns.addStretch()
        self.ok_btn = QPushButton(self.tr('Export'))
        self.ok_btn.setDefault(True)
        self.cancel_btn = QPushButton(self.tr('Cancel'))
        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)
        btns.addWidget(self.ok_btn)
        btns.addWidget(self.cancel_btn)
        layout.addLayout(btns)

    def _pick_folder(self):
        title = self.tr('Select output folder')
        out = QFileDialog.getExistingDirectory(self, title, self._out_dir or '')
        if out:
            self._out_dir = osp.normpath(out)
            self.folder_label.setText(self._out_dir)

    def get_folder(self) -> str:
        return self._out_dir

    def get_extension(self) -> str:
        return self.format_combo.currentData()
