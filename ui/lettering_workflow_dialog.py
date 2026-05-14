from __future__ import annotations

from typing import List, Sequence

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from utils.lettering_workflow import build_lettering_workflow_plan


class LetteringWorkflowDialog(QDialog):
    """Review and run a Koharu-style lettering workflow over pages."""

    def __init__(self, project, current_page: str, selected_pages: Sequence[str], config_obj, parent=None):
        super().__init__(parent)
        self.project = project
        self.current_page = current_page or ""
        self._selected_pages = [p for p in (selected_pages or []) if p]
        self.config_obj = config_obj
        self._plan = None
        self.setWindowTitle(self.tr("Lettering workflow"))
        self.resize(980, 640)

        outer = QVBoxLayout(self)
        intro = QLabel(
            self.tr(
                "Plan typography polish, smart fitting, layout review escalation, proof export, "
                "and final render from the renderer QA diagnostics."
            ),
            self,
        )
        intro.setWordWrap(True)
        outer.addWidget(intro)

        form = QFormLayout()
        outer.addLayout(form)
        self.scope_combo = QComboBox(self)
        self.scope_combo.addItem(self.tr("Current page"), "current")
        if self._selected_pages:
            self.scope_combo.addItem(self.tr("Selected pages ({0})").format(len(self._selected_pages)), "selected")
        self.scope_combo.addItem(self.tr("Whole project"), "project")
        self.scope_combo.currentIndexChanged.connect(self.refresh_plan)
        form.addRow(self.tr("Scope"), self.scope_combo)

        option_row = QHBoxLayout()
        self.apply_check = QCheckBox(self.tr("Apply conservative fixes"), self)
        self.apply_check.setToolTip(self.tr("Applies checked renderer-QA fixes such as polish typography, smart fit, padding, and fallback chains."))
        self.apply_check.setChecked(True)
        self.export_proof_check = QCheckBox(self.tr("Export proof pack for current page"), self)
        self.export_proof_check.setToolTip(self.tr("Proof packs include QA JSON/Markdown, SVG handoff, PSD-helper manifest/layers where available, and final composite when rendered."))
        self.render_check = QCheckBox(self.tr("Render current page after fixes"), self)
        self.render_check.setToolTip(self.tr("Refreshes the current page composite and writes a render manifest."))
        option_row.addWidget(self.apply_check)
        option_row.addWidget(self.export_proof_check)
        option_row.addWidget(self.render_check)
        form.addRow(self.tr("Actions"), option_row)

        refresh_row = QHBoxLayout()
        self.summary_label = QLabel(self)
        self.summary_label.setWordWrap(True)
        refresh_btn = QPushButton(self.tr("Refresh plan"), self)
        refresh_btn.clicked.connect(self.refresh_plan)
        refresh_row.addWidget(self.summary_label, 1)
        refresh_row.addWidget(refresh_btn)
        outer.addLayout(refresh_row)

        self.steps_table = QTableWidget(self)
        self.steps_table.setColumnCount(4)
        self.steps_table.setHorizontalHeaderLabels([self.tr("Step"), self.tr("Affected"), self.tr("Reason"), self.tr("Order")])
        self.steps_table.setAlternatingRowColors(True)
        outer.addWidget(self.steps_table, 1)

        self.focus_table = QTableWidget(self)
        self.focus_table.setColumnCount(7)
        self.focus_table.setHorizontalHeaderLabels([
            self.tr("Page"), self.tr("#"), self.tr("Quality"), self.tr("Warnings"),
            self.tr("Suggestions"), self.tr("Mode"), self.tr("Fit"),
        ])
        self.focus_table.setAlternatingRowColors(True)
        outer.addWidget(QLabel(self.tr("Highest-priority text boxes"), self))
        outer.addWidget(self.focus_table, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText(self.tr("Run workflow"))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)
        self.refresh_plan()

    def selected_pages(self) -> List[str]:
        scope = self.scope_combo.currentData()
        if scope == "project":
            return list((getattr(self.project, "pages", {}) or {}).keys())
        if scope == "selected" and self._selected_pages:
            return list(self._selected_pages)
        return [self.current_page] if self.current_page else []

    def should_apply(self) -> bool:
        return self.apply_check.isChecked()

    def should_export_proof(self) -> bool:
        return self.export_proof_check.isChecked()

    def should_render(self) -> bool:
        return self.render_check.isChecked()

    def plan(self):
        if self._plan is None:
            self.refresh_plan()
        return self._plan or {}

    def refresh_plan(self):
        pages = self.selected_pages()
        self._plan = build_lettering_workflow_plan(self.project, pages=pages, config_obj=self.config_obj, include_ok=False)
        summary = self._plan.get("summary", {}) or {}
        self.summary_label.setText(
            self.tr("{pages} page(s), {boxes} text box(es), {issues} issue text box(es). Core actions: {actions}").format(
                pages=summary.get("pages", len(pages)),
                boxes=summary.get("textboxes", 0),
                issues=summary.get("issues", 0),
                actions=", ".join(f"{k}={v}" for k, v in sorted((self._plan.get("core_action_summary", {}) or {}).items())) or self.tr("none"),
            )
        )
        self._populate_steps(self._plan.get("steps", []) or [])
        self._populate_focus(self._plan.get("focus_items", []) or [])

    def _populate_steps(self, steps):
        self.steps_table.setSortingEnabled(False)
        self.steps_table.setRowCount(len(steps))
        for r, step in enumerate(steps):
            values = [step.get("label", step.get("id", "")), step.get("affected_textboxes", 0), step.get("reason", ""), step.get("apply_order", 0)]
            for c, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if c in {1, 3}:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.steps_table.setItem(r, c, item)
        self.steps_table.resizeColumnsToContents()
        self.steps_table.setSortingEnabled(True)

    def _populate_focus(self, rows):
        self.focus_table.setSortingEnabled(False)
        self.focus_table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            values = [
                row.get("page", ""), row.get("index", -1), row.get("quality_score", ""),
                ", ".join(row.get("warnings", []) or []),
                ", ".join(row.get("suggestions", []) or []),
                row.get("writing_mode", ""), row.get("fit_mode", ""),
            ]
            for c, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if c in {1, 2}:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.focus_table.setItem(r, c, item)
        self.focus_table.resizeColumnsToContents()
        self.focus_table.setSortingEnabled(True)
