from __future__ import annotations

import json
import os.path as osp

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from utils.rendering_qa import build_project_rendering_qa, flatten_rendering_qa_rows, rendering_qa_to_markdown


class TypographyQAReportDialog(QDialog):
    """Preview/export/apply project typography QA without adding mainwindow bloat."""

    def __init__(self, project, current_page: str, config_obj, parent=None):
        super().__init__(parent)
        self.project = project
        self.current_page = current_page
        self.config_obj = config_obj
        self._report = None
        self.setWindowTitle(self.tr("Typography QA Report"))
        self.resize(980, 620)

        outer = QVBoxLayout(self)
        form = QFormLayout()
        outer.addLayout(form)

        self.scope_combo = QComboBox(self)
        self.scope_combo.addItems([self.tr("Current page"), self.tr("Whole project")])
        self.include_ok_check = QCheckBox(self.tr("Include text boxes without warnings"), self)
        self.apply_fixes_check = QCheckBox(self.tr("Apply checked conservative fixes after export"), self)
        self.apply_fixes_check.setToolTip(self.tr("Only checked rows in the preview are changed. Fixes include shrink/balance, strict vertical CJK breaks, punctuation normalization, RTL alignment, stroke padding, contrast stroke, and fallback chains."))
        self.select_all_check = QCheckBox(self.tr("Select all issue rows for fixing"), self)
        self.select_all_check.setChecked(True)
        self.select_all_check.stateChanged.connect(self._on_select_all_changed)
        self.issue_filter_combo = QComboBox(self)
        self.issue_filter_combo.addItem(self.tr("All warning types"), "")
        self.issue_filter_combo.setToolTip(self.tr("Filter the preview table to a warning type before selecting/applying fixes."))
        self.issue_filter_combo.currentIndexChanged.connect(self._apply_issue_filter)
        self.check_visible_btn = QPushButton(self.tr("Check visible warnings"), self)
        self.check_visible_btn.setToolTip(self.tr("Select only warning rows currently visible after filtering."))
        self.check_visible_btn.clicked.connect(self._check_visible_warning_rows)
        self.clear_checks_btn = QPushButton(self.tr("Clear checks"), self)
        self.clear_checks_btn.clicked.connect(self._clear_checked_rows)

        path_row = QHBoxLayout()
        self.path_edit = QLineEdit(self)
        self.path_edit.setPlaceholderText(self.tr("Choose JSON or Markdown report path..."))
        browse_btn = QPushButton(self.tr("Browse..."), self)
        browse_btn.clicked.connect(self._browse)
        path_row.addWidget(self.path_edit)
        path_row.addWidget(browse_btn)

        form.addRow(self.tr("Scope"), self.scope_combo)
        form.addRow("", self.include_ok_check)
        form.addRow("", self.apply_fixes_check)
        form.addRow("", self.select_all_check)
        filter_row = QHBoxLayout()
        filter_row.addWidget(self.issue_filter_combo, 1)
        filter_row.addWidget(self.check_visible_btn)
        filter_row.addWidget(self.clear_checks_btn)
        form.addRow(self.tr("Warning filter"), filter_row)
        form.addRow(self.tr("Report path"), path_row)

        refresh_row = QHBoxLayout()
        self.summary_label = QLabel(self)
        self.summary_label.setWordWrap(True)
        refresh_btn = QPushButton(self.tr("Refresh preview"), self)
        refresh_btn.clicked.connect(self.refresh_report)
        refresh_row.addWidget(self.summary_label, 1)
        refresh_row.addWidget(refresh_btn)
        outer.addLayout(refresh_row)

        self.table = QTableWidget(self)
        self.table.setColumnCount(11)
        self.table.setHorizontalHeaderLabels([
            self.tr("Fix"), self.tr("Page"), self.tr("#"), self.tr("Severity"), self.tr("Quality"), self.tr("Warnings"),
            self.tr("Suggestions"), self.tr("Mode"), self.tr("Fit"), self.tr("Break"), self.tr("Missing"),
        ])
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(True)
        outer.addWidget(self.table, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText(self.tr("Export / Apply"))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)
        self.scope_combo.currentIndexChanged.connect(self.refresh_report)
        self.include_ok_check.stateChanged.connect(self.refresh_report)
        self._rows = []
        self.refresh_report()

    def selected_pages(self):
        if self.scope_combo.currentIndex() == 0:
            return [self.current_page] if self.current_page else []
        return list((getattr(self.project, "pages", {}) or {}).keys())

    def report(self):
        if self._report is None:
            self.refresh_report()
        return self._report

    def should_apply_fixes(self) -> bool:
        return self.apply_fixes_check.isChecked()

    def selected_fixes(self):
        fixes = []
        for r, row in enumerate(getattr(self, "_rows", []) or []):
            item = self.table.item(r, 0)
            if item is None or item.checkState() != Qt.CheckState.Checked:
                continue
            row_data = item.data(Qt.ItemDataRole.UserRole) if hasattr(Qt, "ItemDataRole") else item.data(Qt.UserRole)
            if isinstance(row_data, dict):
                row = row_data
            fixes.append({
                "page": row.get("page", ""),
                "index": int(row.get("index", -1)),
                "actions": list(row.get("suggestions", []) or []),
            })
        return fixes

    def report_path(self) -> str:
        return self.path_edit.text().strip()

    def refresh_report(self):
        self._report = build_project_rendering_qa(
            self.project,
            pages=self.selected_pages(),
            include_ok=self.include_ok_check.isChecked(),
            config_obj=self.config_obj,
        )
        summary = self._report.get("summary", {}) or {}
        counts = summary.get("issue_counts", {}) or {}
        counts_txt = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())) if counts else self.tr("none")
        self.summary_label.setText(
            self.tr("Pages: {0}  Text boxes: {1}  Issue boxes: {2}  Counts: {3}").format(
                summary.get("pages", 0), summary.get("textboxes", 0), summary.get("issues", 0), counts_txt
            )
        )
        rows = flatten_rendering_qa_rows(self._report)
        self._rows = rows
        self._refresh_issue_filter_options(rows)
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            fix_item = QTableWidgetItem("")
            fix_item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            fix_item.setCheckState(Qt.CheckState.Checked if row.get("warnings") else Qt.CheckState.Unchecked)
            try:
                fix_item.setData(Qt.ItemDataRole.UserRole, row)
            except AttributeError:
                fix_item.setData(Qt.UserRole, row)
            self.table.setItem(r, 0, fix_item)
            values = [
                row.get("page", ""), row.get("index", ""), row.get("severity", ""), row.get("quality_score", ""),
                ", ".join(row.get("warnings", [])), ", ".join(row.get("suggestions", [])),
                row.get("writing_mode", ""), row.get("fit_mode", ""), row.get("line_break_strategy", ""),
                row.get("missing_glyphs", ""),
            ]
            for c, value in enumerate(values, start=1):
                item = QTableWidgetItem(str(value))
                if c == 2:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(r, c, item)
        self.table.resizeColumnsToContents()
        self.table.setSortingEnabled(True)
        self._apply_issue_filter()

    def _refresh_issue_filter_options(self, rows):
        current = self.issue_filter_combo.currentData() if hasattr(self, 'issue_filter_combo') else ''
        warnings = sorted({w for row in rows for w in (row.get("warnings", []) or [])})
        self.issue_filter_combo.blockSignals(True)
        self.issue_filter_combo.clear()
        self.issue_filter_combo.addItem(self.tr("All warning types"), "")
        for warning in warnings:
            self.issue_filter_combo.addItem(warning, warning)
        idx = self.issue_filter_combo.findData(current)
        self.issue_filter_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.issue_filter_combo.blockSignals(False)

    def _apply_issue_filter(self):
        selected = self.issue_filter_combo.currentData() if hasattr(self, 'issue_filter_combo') else ''
        for r, row in enumerate(getattr(self, '_rows', []) or []):
            visible = not selected or selected in (row.get("warnings", []) or [])
            self.table.setRowHidden(r, not visible)

    def _check_visible_warning_rows(self):
        for r, row in enumerate(getattr(self, '_rows', []) or []):
            item = self.table.item(r, 0)
            if item is not None:
                item.setCheckState(Qt.CheckState.Checked if (not self.table.isRowHidden(r) and row.get("warnings")) else Qt.CheckState.Unchecked)

    def _clear_checked_rows(self):
        for r in range(self.table.rowCount()):
            item = self.table.item(r, 0)
            if item is not None:
                item.setCheckState(Qt.CheckState.Unchecked)

    def _on_select_all_changed(self):
        state = Qt.CheckState.Checked if self.select_all_check.isChecked() else Qt.CheckState.Unchecked
        for r in range(self.table.rowCount()):
            item = self.table.item(r, 0)
            if item is not None:
                item.setCheckState(state)

    def write_report(self, path: str = "") -> str:
        path = path or self.report_path()
        if not path:
            return ""
        report = self.report()
        if path.lower().endswith(('.md', '.markdown')):
            with open(path, 'w', encoding='utf-8') as f:
                f.write(rendering_qa_to_markdown(report))
        else:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
        return path

    def _browse(self):
        base = getattr(self.project, 'directory', '') if self.project is not None else ''
        path, _ = QFileDialog.getSaveFileName(
            self,
            self.tr('Save typography QA report'),
            osp.join(base, 'typography_qa_report.json'),
            self.tr('JSON (*.json);;Markdown (*.md)'),
        )
        if path:
            self.path_edit.setText(path)
