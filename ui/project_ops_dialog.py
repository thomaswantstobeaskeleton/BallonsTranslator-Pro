import json
from qtpy.QtCore import Qt, QPropertyAnimation, QEasingCurve
from qtpy.QtWidgets import QDialog, QVBoxLayout, QLabel, QPlainTextEdit, QHBoxLayout, QPushButton, QListWidget, QListWidgetItem, QGraphicsOpacityEffect

from utils.project_ops_protocol import ProjectOpSession, apply_ops, undo, redo


class ProjectOpsDialog(QDialog):
    def __init__(self, page: str, blocks: list, on_commit, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Project Ops Console"))
        self.resize(760, 520)
        self._page = page
        self._blocks = blocks
        self._on_commit = on_commit
        self._session = ProjectOpSession(pages={page: [{"translation": getattr(b, 'translation', '') or ''} for b in blocks]})

        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(self.tr("JSON Ops (supports UpdateText and Batch)")))
        self.editor = QPlainTextEdit(self)
        self.editor.setPlainText(json.dumps([
            {"op": "UpdateText", "page": page, "index": 0, "text": "Edited via ProjectOps"}
        ], indent=2))
        lay.addWidget(self.editor, 2)
        btns = QHBoxLayout()
        self.apply_btn = QPushButton(self.tr("Apply"), self)
        self.undo_btn = QPushButton(self.tr("Undo"), self)
        self.redo_btn = QPushButton(self.tr("Redo"), self)
        self.commit_btn = QPushButton(self.tr("Commit to page"), self)
        for b in (self.apply_btn, self.undo_btn, self.redo_btn, self.commit_btn):
            btns.addWidget(b)
        lay.addLayout(btns)
        self.log = QListWidget(self)
        lay.addWidget(self.log, 1)
        self.preview = QPlainTextEdit(self)
        self.preview.setReadOnly(True)
        lay.addWidget(self.preview, 1)
        self.apply_btn.clicked.connect(self.on_apply)
        self.undo_btn.clicked.connect(self.on_undo)
        self.redo_btn.clicked.connect(self.on_redo)
        self.commit_btn.clicked.connect(self.on_commit)

        self._opacity = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self._opacity)
        self._anim = QPropertyAnimation(self._opacity, b"opacity", self)
        self._anim.setDuration(220)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        self._anim.start()
        self._refresh_preview()

    def _refresh_preview(self):
        arr = self._session.pages.get(self._page, [])
        self.preview.setPlainText("\n".join(f"#{i+1}: {(it.get('translation','') or '')}" for i, it in enumerate(arr[:30])))

    def on_apply(self):
        try:
            ops = json.loads(self.editor.toPlainText() or "[]")
            if not isinstance(ops, list):
                raise ValueError("root must be a list")
            out = apply_ops(self._session, ops)
            self.log.addItem(QListWidgetItem(self.tr(f'Applied {out.get("count", 0)} op(s)')))
            self._refresh_preview()
        except Exception as e:
            self.log.addItem(QListWidgetItem(self.tr(f'Apply failed: {e}')))

    def on_undo(self):
        n = undo(self._session)
        self.log.addItem(QListWidgetItem(self.tr(f'Undo: {n} op(s)')))
        self._refresh_preview()

    def on_redo(self):
        n = redo(self._session)
        self.log.addItem(QListWidgetItem(self.tr(f'Redo: {n} op(s)')))
        self._refresh_preview()

    def on_commit(self):
        self._on_commit(self._session.pages.get(self._page, []))
        self.accept()
