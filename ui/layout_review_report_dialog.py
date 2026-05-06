from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFrame,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
)


class LayoutReviewReportDialog(QDialog):
    """Human-in-the-loop report for layout review proposals."""

    def __init__(self, result, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Layout Review Agent Report"))
        self.setMinimumSize(620, 420)
        self._result = result
        self._action_items = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        summary = self._summarize(result)
        title = QLabel(
            self.tr("{issues} issue(s), {actions} proposed fix(es), average score {score:.0f}%").format(
                issues=summary["issues"], actions=summary["actions"], score=summary["score"] * 100.0
            ),
            self,
        )
        title.setObjectName("LayoutReviewReportTitle")
        layout.addWidget(title)

        hint = QLabel(
            self.tr("Review the deterministic changes below. Choose Apply Fixes to mutate text boxes through the undo stack, or Close to keep the page unchanged."),
            self,
        )
        hint.setWordWrap(True)
        hint.setObjectName("LayoutReviewReportHint")
        layout.addWidget(hint)

        self.list_widget = QListWidget(self)
        self.list_widget.setFrameShape(QFrame.NoFrame)
        self.list_widget.setAlternatingRowColors(True)
        self._populate(result)
        layout.addWidget(self.list_widget, 1)

        if self.list_widget.count() == 0:
            empty = QListWidgetItem(self.tr("No layout issues were detected for the selected scope."))
            empty.setFlags(empty.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            self.list_widget.addItem(empty)

        self.buttons = QDialogButtonBox(self)
        self.apply_button = self.buttons.addButton(self.tr("Apply Fixes"), QDialogButtonBox.ButtonRole.AcceptRole)
        self.close_button = self.buttons.addButton(QDialogButtonBox.StandardButton.Close)
        self.apply_button.setEnabled(summary["actions"] > 0)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

    def _summarize(self, result):
        blocks = list(getattr(result, "blocks", []) or [])
        issue_count = sum(len(getattr(b, "issues", []) or []) for b in blocks)
        action_count = sum(len(getattr(b, "actions", []) or []) for b in blocks)
        if blocks:
            score = sum(float(getattr(b, "score_after", 1.0) or 0.0) for b in blocks) / len(blocks)
        else:
            score = 1.0
        return {"issues": issue_count, "actions": action_count, "score": max(0.0, min(1.0, score))}

    def _populate(self, result):
        for block in getattr(result, "blocks", []) or []:
            issues = getattr(block, "issues", []) or []
            actions = getattr(block, "actions", []) or []
            if not issues and not actions:
                continue
            parts = [self.tr("Block #{0}").format(int(getattr(block, "block_index", -1)) + 1)]
            if issues:
                issue_text = "; ".join(
                    f"{getattr(issue, 'severity', 'info')}:{getattr(issue, 'code', 'issue')} — {getattr(issue, 'message', '')}"
                    for issue in issues
                )
                parts.append(issue_text)
            if actions:
                action_text = "; ".join(
                    f"{getattr(action, 'action', 'fix')} ({getattr(action, 'reason', '')})".strip()
                    for action in actions
                )
                parts.append(self.tr("Fixes: ") + action_text)
            item = QListWidgetItem("\n".join(parts))
            item.setToolTip("\n".join(parts))
            if actions:
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Checked)
                item.setData(Qt.ItemDataRole.UserRole, list(actions))
                self._action_items.append(item)
            else:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)
            self.list_widget.addItem(item)

    def selected_actions(self):
        """Return only actions checked by the user in the report."""
        selected = []
        for item in self._action_items:
            if item.checkState() == Qt.CheckState.Checked:
                selected.extend(item.data(Qt.ItemDataRole.UserRole) or [])
        return selected
