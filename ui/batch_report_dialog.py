"""
Batch report dialog: show skipped pages and reasons from the last pipeline run.
Double-click a row to switch to that page.
"""
from __future__ import annotations

from qtpy.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView,
)
from qtpy.QtCore import Signal, Qt


class BatchReportDialog(QDialog):
    page_activated = Signal(str)  # page_key when user double-clicks a row

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Batch report"))
        self.setMinimumSize(500, 300)
        self.resize(600, 400)
        layout = QVBoxLayout(self)
        self.status_label = QLabel(self.tr("Status: —"))
        layout.addWidget(self.status_label)
        self.stats_label = QLabel(self.tr("Total: —  Skipped: —  Completed: —"))
        layout.addWidget(self.stats_label)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels([
            self.tr("Page"),
            self.tr("Reason"),
            self.tr("Suggested action"),
        ])
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.doubleClicked.connect(self._on_double_click)
        layout.addWidget(self.table)
        self._report = None

    def set_report(self, report: dict):
        """Set the report dict from batch_report.finalize_batch_report()."""
        self._report = report
        if not report:
            self.status_label.setText(self.tr("Status: No report."))
            self.stats_label.setText(self.tr("Total: —  Skipped: —  Completed: —"))
            self.table.setRowCount(0)
            return
        page_keys = report.get("page_keys") or []
        skipped = report.get("skipped") or {}
        cancelled = report.get("cancelled", False)
        finished_at = report.get("finished_at", "")
        total = len(page_keys)
        skip_count = len(skipped)
        completed = total - skip_count
        self.status_label.setText(
            self.tr("Status: {}  Finished: {}").format(
                self.tr("Cancelled") if cancelled else self.tr("Completed"),
                finished_at[:19] if finished_at else "—",
            )
        )
        self.stats_label.setText(
            self.tr("Total: {}  Skipped: {}  Completed: {}").format(total, skip_count, completed)
        )
        self.table.setRowCount(skip_count)
        for row, (page_key, info) in enumerate(skipped.items()):
            if not isinstance(info, dict):
                info = {"reason": str(info), "detail": "", "action": ""}
            self.table.setItem(row, 0, QTableWidgetItem(page_key))
            self.table.setItem(row, 1, QTableWidgetItem(info.get("detail") or info.get("reason") or ""))
            self.table.setItem(row, 2, QTableWidgetItem(info.get("action") or ""))
        self.table.resizeColumnsToContents()

    def _on_double_click(self, index):
        row = index.row()
        if row < 0 or not self.table.item(row, 0):
            return
        page_key = self.table.item(row, 0).text()
        if page_key:
            self.page_activated.emit(page_key)
            self.accept()
