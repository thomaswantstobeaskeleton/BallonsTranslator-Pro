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
        self.apply_fixes_check = QCheckBox(self.tr("Apply conservative fixes after export"), self)
        self.apply_fixes_check.setToolTip(self.tr("Shrinks overflow text, applies strict vertical CJK breaks, right-aligns RTL, increases stroke padding, and fills empty fallback chains."))

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
        self.table.setColumnCount(9)
        self.table.setHorizontalHeaderLabels([
            self.tr("Page"), self.tr("#"), self.tr("Severity"), self.tr("Warnings"),
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
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            values = [
                row.get("page", ""), row.get("index", ""), row.get("severity", ""),
                ", ".join(row.get("warnings", [])), ", ".join(row.get("suggestions", [])),
                row.get("writing_mode", ""), row.get("fit_mode", ""), row.get("line_break_strategy", ""),
                row.get("missing_glyphs", ""),
            ]
            for c, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if c == 1:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(r, c, item)
        self.table.resizeColumnsToContents()
        self.table.setSortingEnabled(True)

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
