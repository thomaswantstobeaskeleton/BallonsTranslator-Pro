"""
Manage models dialog: check status of all module files (downloaded / missing / hash mismatch)
and download selected models. Opened from Tools → Manage models...
"""
from __future__ import annotations

import os
from typing import List, Dict, Any

from qtpy.QtCore import Qt, Signal, QThread, QObject
from qtpy.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QCheckBox,
    QGroupBox,
    QProgressBar,
    QScrollArea,
    QWidget,
    QMessageBox,
    QHeaderView,
    QAbstractItemView,
    QGridLayout,
    QLineEdit,
    QComboBox,
)

from utils.model_manager import (
    get_all_downloadable_modules,
    check_all_models,
)


class CheckModelsWorker(QObject):
    """Run check_all_models() in a background thread."""
    finished = Signal(list)  # list of result dicts
    error = Signal(str)

    def __init__(self, include_import_check: bool = False):
        super().__init__()
        self.include_import_check = include_import_check

    def run(self):
        try:
            results = check_all_models(include_import_check=self.include_import_check)
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))


class DownloadModelsWorker(QObject):
    """Run download for selected module classes in a background thread."""
    progress = Signal(str)   # current module name
    finished = Signal(int, int, list)  # success_count, fail_count, errors
    error = Signal(str)

    def __init__(self, module_class_list: List):
        super().__init__()
        self.module_class_list = module_class_list

    def run(self):
        try:
            import utils.shared as shared
            from modules.prepare_local_files import download_and_check_module_files
            cwd = os.getcwd()
            try:
                if hasattr(shared, 'PROGRAM_PATH'):
                    os.chdir(shared.PROGRAM_PATH)
                success = 0
                failed = 0
                errors = []
                for i, module_class in enumerate(self.module_class_list):
                    if getattr(module_class, 'download_file_on_load', False) or getattr(module_class, 'download_file_list', None) is None:
                        continue
                    name = getattr(module_class, '__name__', str(module_class))
                    self.progress.emit(name)
                    try:
                        download_and_check_module_files([module_class])
                        success += 1
                    except Exception as e:
                        failed += 1
                        errors.append(f'{name}: {e}')
                self.finished.emit(success, failed, errors)
            finally:
                os.chdir(cwd)
        except Exception as e:
            self.error.emit(str(e))


class ModelManagerDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr('Manage models'))
        self.setMinimumSize(700, 500)
        self.resize(800, 560)
        self._check_thread = None
        self._check_worker = None
        self._download_thread = None
        self._download_worker = None
        self._module_infos = []
        self._check_results: List[Dict[str, Any]] = []
        self._check_boxes: List[QCheckBox] = []
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # --- Check section ---
        check_group = QGroupBox(self.tr('Check model files'))
        check_layout = QVBoxLayout(check_group)
        check_btn_layout = QHBoxLayout()
        self.checkBtn = QPushButton(self.tr('Check all models'))
        self.checkBtn.setToolTip(self.tr('Check which models have their files downloaded, missing, or with hash mismatch.'))
        self.checkBtn.clicked.connect(self._on_check_clicked)
        check_btn_layout.addWidget(self.checkBtn)
        self.checkCompatibilityCb = QCheckBox(self.tr('Check compatibility (slow)'))
        self.checkCompatibilityCb.setToolTip(self.tr('Try loading each module to detect missing dependencies or incompatible environment. Can be slow.'))
        check_btn_layout.addWidget(self.checkCompatibilityCb)
        check_btn_layout.addStretch()
        check_layout.addLayout(check_btn_layout)
        self.checkProgress = QProgressBar()
        self.checkProgress.setRange(0, 0)
        self.checkProgress.setVisible(False)
        check_layout.addWidget(self.checkProgress)
        filter_row = QHBoxLayout()
        self.searchEdit = QLineEdit()
        self.searchEdit.setPlaceholderText(self.tr('Search module key/name/description…'))
        self.searchEdit.textChanged.connect(self._apply_result_filters)
        filter_row.addWidget(self.searchEdit, 2)
        self.categoryFilter = QComboBox()
        self.categoryFilter.addItem(self.tr('All categories'), '')
        self.categoryFilter.currentIndexChanged.connect(self._apply_result_filters)
        filter_row.addWidget(self.categoryFilter, 1)
        self.statusFilter = QComboBox()
        self.statusFilter.addItem(self.tr('All statuses'), '')
        self.statusFilter.currentIndexChanged.connect(self._apply_result_filters)
        filter_row.addWidget(self.statusFilter, 1)
        check_layout.addLayout(filter_row)
        self.resultTable = QTableWidget()
        self.resultTable.setColumnCount(4)
        self.resultTable.setHorizontalHeaderLabels([
            self.tr('Category'), self.tr('Module'), self.tr('Status'), self.tr('Details')
        ])
        self.resultTable.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.resultTable.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.resultTable.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        check_layout.addWidget(self.resultTable)
        layout.addWidget(check_group)

        # --- Download section ---
        download_group = QGroupBox(self.tr('Download models'))
        download_layout = QVBoxLayout(download_group)
        desc = QLabel(self.tr('Select the models you want to download. Only modules with a predefined file list are listed.'))
        desc.setWordWrap(True)
        download_layout.addWidget(desc)
        btn_row = QHBoxLayout()
        select_all_btn = QPushButton(self.tr('Select all'))
        select_all_btn.clicked.connect(self._select_all_download)
        deselect_all_btn = QPushButton(self.tr('Deselect all'))
        deselect_all_btn.clicked.connect(self._deselect_all_download)
        self.downloadSelectedBtn = QPushButton(self.tr('Download selected'))
        self.downloadSelectedBtn.clicked.connect(self._on_download_clicked)
        btn_row.addWidget(select_all_btn)
        btn_row.addWidget(deselect_all_btn)
        btn_row.addWidget(self.downloadSelectedBtn)
        btn_row.addStretch()
        download_layout.addLayout(btn_row)
        self.downloadProgress = QProgressBar()
        self.downloadProgress.setVisible(False)
        download_layout.addWidget(self.downloadProgress)
        self.downloadStatusLabel = QLabel('')
        download_layout.addWidget(self.downloadStatusLabel)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(120)
        self.downloadCheckboxWidget = QWidget()
        self.downloadCheckboxLayout = QGridLayout(self.downloadCheckboxWidget)
        scroll.setWidget(self.downloadCheckboxWidget)
        download_layout.addWidget(scroll)
        layout.addWidget(download_group)

        close_btn = QPushButton(self.tr('Close'))
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

        self._refresh_download_list()

    def _refresh_download_list(self):
        self._module_infos = get_all_downloadable_modules()
        for cb in self._check_boxes:
            cb.deleteLater()
        self._check_boxes.clear()
        row, col = 0, 0
        max_cols = 3
        for mod in self._module_infos:
            if not mod['can_download']:
                continue
            cb = QCheckBox(mod['display_name'])
            meta = mod.get('manifest_meta') or {}
            tooltip_bits = []
            if meta.get('size_estimate'):
                tooltip_bits.append(self.tr('Size: {0}').format(meta['size_estimate']))
            deps = meta.get('required_deps') or []
            if deps:
                tooltip_bits.append(self.tr('Dependencies: {0}').format(', '.join(deps)))
            if meta.get('support_tier'):
                tooltip_bits.append(self.tr('Support tier: {0}').format(meta['support_tier']))
            if tooltip_bits:
                cb.setToolTip('\n'.join(tooltip_bits))
            cb.setToolTip(self._build_module_tooltip(mod))
            cb.setProperty('module_info', mod)
            self.downloadCheckboxLayout.addWidget(cb, row, col)
            self._check_boxes.append(cb)
            col += 1
            if col >= max_cols:
                col = 0
                row += 1

    def _select_all_download(self):
        for cb in self._check_boxes:
            cb.setChecked(True)

    def _deselect_all_download(self):
        for cb in self._check_boxes:
            cb.setChecked(False)

    def _on_check_clicked(self):
        self.checkBtn.setEnabled(False)
        self.checkProgress.setVisible(True)
        self.resultTable.setRowCount(0)
        include_import = self.checkCompatibilityCb.isChecked()
        self._check_thread = QThread()
        self._check_worker = CheckModelsWorker(include_import)
        self._check_worker.moveToThread(self._check_thread)
        self._check_thread.started.connect(self._check_worker.run)
        self._check_worker.finished.connect(self._on_check_finished)
        self._check_worker.error.connect(self._on_check_error)
        self._check_thread.start()

    def _on_check_finished(self, results: List[Dict[str, Any]]):
        if self._check_thread and self._check_thread.isRunning():
            self._check_thread.quit()
            self._check_thread.wait(2000)
        self.checkBtn.setEnabled(True)
        self.checkProgress.setVisible(False)
        status_text = {
            'ok': self.tr('Downloaded'),
            'missing': self.tr('Missing'),
            'hash_mismatch': self.tr('Hash mismatch'),
            'no_download_list': self.tr('No file list'),
            'download_on_load': self.tr('Download on load'),
        }
        self._check_results = results
        self._refresh_filter_options()
        self._apply_result_filters()

    def _on_check_error(self, msg: str):
        if self._check_thread and self._check_thread.isRunning():
            self._check_thread.quit()
            self._check_thread.wait(2000)
        self.checkBtn.setEnabled(True)
        self.checkProgress.setVisible(False)
        QMessageBox.warning(self, self.tr('Check models'), msg)

    def _on_download_clicked(self):
        selected = [cb.property('module_info') for cb in self._check_boxes if cb.isChecked()]
        if not selected:
            QMessageBox.information(self, self.tr('Download models'), self.tr('Select at least one model to download.'))
            return
        self.checkBtn.setEnabled(False)
        self.downloadSelectedBtn.setEnabled(False)
        self.downloadProgress.setVisible(True)
        self.downloadProgress.setRange(0, 0)
        self.downloadStatusLabel.setText(self.tr('Downloading…'))
        classes = [m['module_class'] for m in selected]
        self._download_thread = QThread()
        self._download_worker = DownloadModelsWorker(classes)
        self._download_worker.moveToThread(self._download_thread)
        self._download_thread.started.connect(self._download_worker.run)
        self._download_worker.finished.connect(self._on_download_finished)
        self._download_worker.error.connect(self._on_download_error)
        self._download_thread.start()

    def _on_download_finished(self, success: int, failed: int, errors: List[str]):
        if self._download_thread and self._download_thread.isRunning():
            self._download_thread.quit()
            self._download_thread.wait(5000)
        self.checkBtn.setEnabled(True)
        self.downloadSelectedBtn.setEnabled(True)
        self.downloadProgress.setVisible(False)
        msg = self.tr('Download finished: {0} succeeded, {1} failed.').format(success, failed)
        if errors:
            msg += '\n\n' + '\n'.join(errors[:15])
            if len(errors) > 15:
                msg += '\n…'
        self.downloadStatusLabel.setText(msg)
        if failed > 0:
            QMessageBox.warning(self, self.tr('Download models'), msg)
        else:
            QMessageBox.information(self, self.tr('Download models'), msg)

    def _on_download_error(self, msg: str):
        if self._download_thread and self._download_thread.isRunning():
            self._download_thread.quit()
            self._download_thread.wait(2000)
        self.checkBtn.setEnabled(True)
        self.downloadSelectedBtn.setEnabled(True)
        self.downloadProgress.setVisible(False)
        self.downloadStatusLabel.setText('')
        QMessageBox.warning(self, self.tr('Download models'), msg)

    def closeEvent(self, event):
        if self._check_thread and self._check_thread.isRunning():
            self._check_thread.quit()
            self._check_thread.wait(1000)
        if self._download_thread and self._download_thread.isRunning():
            self._download_thread.quit()
            self._download_thread.wait(1000)
        super().closeEvent(event)

    def _refresh_filter_options(self):
        current_category = self.categoryFilter.currentData()
        current_status = self.statusFilter.currentData()
        categories = sorted({r['module_info'].get('category_label', '') for r in self._check_results if r.get('module_info')})
        self.categoryFilter.blockSignals(True)
        self.categoryFilter.clear()
        self.categoryFilter.addItem(self.tr('All categories'), '')
        for c in categories:
            self.categoryFilter.addItem(c, c)
        idx = self.categoryFilter.findData(current_category)
        if idx >= 0:
            self.categoryFilter.setCurrentIndex(idx)
        self.categoryFilter.blockSignals(False)

        status_entries = [
            ('ok', self.tr('Downloaded')),
            ('missing', self.tr('Missing')),
            ('hash_mismatch', self.tr('Hash mismatch')),
            ('no_download_list', self.tr('No file list')),
            ('download_on_load', self.tr('Download on load')),
            ('incompatible', self.tr('Incompatible')),
            ('optional', self.tr('Optional')),
        ]
        self.statusFilter.blockSignals(True)
        self.statusFilter.clear()
        self.statusFilter.addItem(self.tr('All statuses'), '')
        for key, label in status_entries:
            self.statusFilter.addItem(label, key)
        idx = self.statusFilter.findData(current_status)
        if idx >= 0:
            self.statusFilter.setCurrentIndex(idx)
        self.statusFilter.blockSignals(False)

    def _apply_result_filters(self):
        text = (self.searchEdit.text() or '').strip().lower()
        category = self.categoryFilter.currentData() or ''
        status_filter = self.statusFilter.currentData() or ''
        status_text = {
            'ok': self.tr('Downloaded'),
            'missing': self.tr('Missing'),
            'hash_mismatch': self.tr('Hash mismatch'),
            'no_download_list': self.tr('No file list'),
            'download_on_load': self.tr('Download on load'),
        }
        filtered = []
        for r in self._check_results:
            mod = r.get('module_info', {})
            raw_status = r.get('status', '')
            effective_status = raw_status
            if raw_status == 'no_download_list' and r.get('details') and any(r.get('details', [])):
                effective_status = 'optional'
            if r.get('import_error'):
                effective_status = 'incompatible'
            if category and mod.get('category_label') != category:
                continue
            if status_filter and effective_status != status_filter:
                continue
            haystack = ' '.join([
                str(mod.get('module_key', '')),
                str(mod.get('human_name', '')),
                str(mod.get('display_name', '')),
                str(mod.get('description', '')),
                str(mod.get('category_label', '')),
            ]).lower()
            if text and text not in haystack:
                continue
            filtered.append(r)

        self.resultTable.setRowCount(len(filtered))
        for row, r in enumerate(filtered):
            mod = r['module_info']
            self.resultTable.setItem(row, 0, QTableWidgetItem(mod.get('category_label', '')))
            module_item = QTableWidgetItem(mod.get('display_name', mod.get('module_key', '')))
            module_item.setToolTip(self._build_module_tooltip(mod))
            self.resultTable.setItem(row, 1, module_item)
            st = r['status']
            status_str = status_text.get(st, st)
            if st == 'no_download_list' and r.get('details') and any(r['details']):
                status_str = self.tr('Optional')
            if r.get('import_error'):
                status_str = self.tr('Incompatible')
            self.resultTable.setItem(row, 2, QTableWidgetItem(status_str))
            details = '; '.join(r['details'][:5])
            if r.get('import_error'):
                details = r['import_error'][:200] + ('…' if len(r['import_error']) > 200 else '')
            elif len(r['details']) > 5:
                details += '…'
            extra = []
            size_text = self._format_size(mod.get('estimated_download_size_bytes'))
            if size_text:
                extra.append(self.tr('Estimated size: {0}').format(size_text))
            target_paths = mod.get('target_paths') or []
            if target_paths:
                extra.append(self.tr('Target: {0}').format(target_paths[0]))
            if extra:
                details = (details + ' | ' if details else '') + ' | '.join(extra)
            self.resultTable.setItem(row, 3, QTableWidgetItem(details))

    def _build_module_tooltip(self, mod: Dict[str, Any]) -> str:
        tip = [self.tr('Key: {0}').format(mod.get('module_key', ''))]
        desc = (mod.get('description') or '').strip()
        if desc:
            tip.append(desc)
        size_text = self._format_size(mod.get('estimated_download_size_bytes'))
        if size_text:
            tip.append(self.tr('Estimated size: {0}').format(size_text))
        target_paths = mod.get('target_paths') or []
        if target_paths:
            preview = ', '.join(target_paths[:3])
            if len(target_paths) > 3:
                preview += ', …'
            tip.append(self.tr('Target path(s): {0}').format(preview))
        return '\n'.join([t for t in tip if t])

    def _format_size(self, size_bytes: Any) -> str:
        if not isinstance(size_bytes, (int, float)) or size_bytes <= 0:
            return ''
        units = ['B', 'KB', 'MB', 'GB', 'TB']
        size = float(size_bytes)
        for unit in units:
            if size < 1024 or unit == units[-1]:
                return f'{size:.1f} {unit}' if unit != 'B' else f'{int(size)} B'
            size /= 1024.0
        return ''
