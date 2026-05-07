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
    QCheckBox,
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

        self.also_pdf_check = QCheckBox(self.tr('Also create PDF from exported images'))
        self.also_pdf_check.setToolTip(self.tr('Build a single PDF file from the exported pages (requires img2pdf).'))
        self.also_pdf_check.setChecked(False)
        form.addRow('', self.also_pdf_check)

        self.export_as_zip_check = QCheckBox(self.tr('Export as archive (ZIP/CBZ)'))
        self.export_as_zip_check.setToolTip(self.tr('Export pages to a temporary folder, then pack into a ZIP or manga-reader CBZ archive.'))
        self.export_as_zip_check.setChecked(False)
        self.export_as_zip_check.toggled.connect(self._on_export_as_zip_toggled)
        form.addRow('', self.export_as_zip_check)

        from qtpy.QtWidgets import QWidget
        self.zip_row_container = QWidget()
        zip_row = QHBoxLayout(self.zip_row_container)
        zip_row.setContentsMargins(0, 0, 0, 0)
        self.archive_format_combo = QComboBox(self)
        self.archive_format_combo.addItem(self.tr('ZIP archive (.zip)'), 'zip')
        self.archive_format_combo.addItem(self.tr('Comic Book ZIP (.cbz)'), 'cbz')
        self.archive_format_combo.currentIndexChanged.connect(self._on_archive_format_changed)
        zip_row.addWidget(self.archive_format_combo)
        zip_row.addWidget(QLabel(self.tr('Archive path:')))
        self.zip_path_label = QLabel(self.tr('(none)'))
        self.zip_path_label.setMinimumWidth(280)
        self._zip_path = ''
        self.zip_path_btn = QPushButton(self.tr('Choose archive file...'))
        self.zip_path_btn.clicked.connect(self._pick_zip_path)
        zip_row.addWidget(self.zip_path_label)
        zip_row.addWidget(self.zip_path_btn)
        form.addRow('', self.zip_row_container)

        self.include_intermediate_check = QCheckBox(self.tr('Include clean/mask helper images'))
        self.include_intermediate_check.setToolTip(self.tr('Copy available inpainted clean pages and masks into clean/ and masks/ subfolders beside the rendered export.'))
        self.include_intermediate_check.setChecked(False)
        form.addRow('', self.include_intermediate_check)

        self.open_folder_after_export_check = QCheckBox(self.tr('Open output folder when done'))
        self.open_folder_after_export_check.setToolTip(self.tr('Use the persisted Settings default for this export.'))
        try:
            from utils.config import pcfg
            self.open_folder_after_export_check.setChecked(bool(getattr(pcfg, 'export_open_folder_after_batch', False)))
        except Exception:
            self.open_folder_after_export_check.setChecked(False)
        self.open_folder_after_export_check.toggled.connect(self._on_open_folder_after_export_toggled)
        form.addRow('', self.open_folder_after_export_check)

        self.clean_after_export_check = QCheckBox(self.tr('Clean cache after export'))
        self.clean_after_export_check.setToolTip(self.tr('Remove mask and inpainted caches for this project after export (saves disk space).'))
        self.clean_after_export_check.setChecked(False)
        form.addRow('', self.clean_after_export_check)

        self._on_export_as_zip_toggled(False)

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

    def _on_export_as_zip_toggled(self, checked: bool):
        self.zip_row_container.setVisible(checked)
        if not checked:
            self._zip_path = ''
            self.zip_path_label.setText(self.tr('(none)'))

    def _pick_zip_path(self):
        title = self.tr('Save archive as')
        path, _ = QFileDialog.getSaveFileName(
            self, title, self._zip_path or self._out_dir or '',
            self.tr('Archives (*.zip *.cbz)'),
        )
        if path:
            wanted = '.cbz' if self.get_archive_format() == 'cbz' else '.zip'
            if not path.lower().endswith(('.zip', '.cbz')):
                path += wanted
            self._zip_path = osp.normpath(path)
            self.zip_path_label.setText(self._zip_path)

    def get_export_as_zip(self) -> bool:
        return self.export_as_zip_check.isChecked()

    def get_zip_path(self) -> str:
        return (self._zip_path or '').strip()

    def get_archive_format(self) -> str:
        return str(self.archive_format_combo.currentData() or 'zip')

    def _on_archive_format_changed(self):
        if self._zip_path:
            base, _ext = osp.splitext(self._zip_path)
            self._zip_path = base + ('.cbz' if self.get_archive_format() == 'cbz' else '.zip')
            self.zip_path_label.setText(self._zip_path)

    def _on_open_folder_after_export_toggled(self, checked: bool):
        try:
            from utils.config import pcfg
            pcfg.export_open_folder_after_batch = bool(checked)
        except Exception:
            pass

    def get_include_intermediate(self) -> bool:
        return self.include_intermediate_check.isChecked()

    def get_open_folder_after_export(self) -> bool:
        return self.open_folder_after_export_check.isChecked()

    def get_clean_after_export(self) -> bool:
        return self.clean_after_export_check.isChecked()

    def get_folder(self) -> str:
        return self._out_dir

    def get_extension(self) -> str:
        return self.format_combo.currentData()

    def get_also_pdf(self) -> bool:
        return self.also_pdf_check.isChecked()
