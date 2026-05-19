from __future__ import annotations

from qtpy.QtWidgets import QDialog, QVBoxLayout, QLabel, QTableWidget, QTableWidgetItem, QPushButton, QHBoxLayout, QComboBox, QCheckBox

from utils.auto_format_qa import score_auto_format_candidates, summarize_auto_format_scores
from utils.renderer_diagnostics import collect_renderer_diagnostics


class AutoFormatQADialog(QDialog):
    def __init__(self, parent, block_items, apply_callback):
        super().__init__(parent)
        self._block_items = list(block_items or [])
        self._apply_callback = apply_callback
        self.setWindowTitle(self.tr('Auto-format QA (Before/After)'))
        self.resize(920, 520)

        lay = QVBoxLayout(self)
        self.hint = QLabel(self)
        self.hint.setWordWrap(True)
        lay.addWidget(self.hint)

        ctl = QHBoxLayout()
        self.profile = QComboBox(self)
        for p in ('balanced', 'comfortable', 'dense', 'caption', 'sfx'):
            self.profile.addItem(p, p)
        ctl.addWidget(QLabel(self.tr('Profile:'), self))
        ctl.addWidget(self.profile)
        self.refresh_btn = QPushButton(self.tr('Refresh Preview'), self)
        self.only_risky = QCheckBox(self.tr('Only risky/improvable'), self)
        self.apply_all_btn = QPushButton(self.tr('Apply to All Listed'), self)
        ctl.addWidget(self.refresh_btn)
        ctl.addWidget(self.only_risky)
        ctl.addWidget(self.apply_all_btn)
        ctl.addStretch(1)
        lay.addLayout(ctl)

        self.table = QTableWidget(self)
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels([
            self.tr('Idx'), self.tr('Text preview'), self.tr('Before score'), self.tr('After score'),
            self.tr('Δ score'), self.tr('Before overflow'), self.tr('After overflow'), self.tr('After actions')
        ])
        lay.addWidget(self.table)

        self.refresh_btn.clicked.connect(self.refresh)
        self.apply_all_btn.clicked.connect(self.apply_all)
        self.profile.currentIndexChanged.connect(self.refresh)
        self.only_risky.clicked.connect(self.refresh)

        self.refresh()

    def refresh(self):
        profile = str(self.profile.currentData() or 'balanced')
        rows = score_auto_format_candidates([getattr(x, 'blk', None) for x in self._block_items], profile=profile)
        if self.only_risky.isChecked():
            rows = [r for r in rows if bool(r.get('before_overflow')) or float(r.get('improvement', 0.0) or 0.0) > 0.04]
        summary = summarize_auto_format_scores(rows)
        self.table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            vals = [
                str(row.get('index', '')),
                str(row.get('text_preview', '')),
                f"{float(row.get('before_score', 0.0)):.3f}",
                f"{float(row.get('after_score', 0.0)):.3f}",
                f"{float(row.get('improvement', 0.0)):+.3f}",
                'Y' if row.get('before_overflow') else 'N',
                'Y' if row.get('after_overflow') else 'N',
                ', '.join(row.get('after_actions', []) or []),
            ]
            for c, v in enumerate(vals):
                self.table.setItem(r, c, QTableWidgetItem(v))
        self.table.resizeColumnsToContents()

        d = collect_renderer_diagnostics()
        self.setWindowTitle(self.tr('Auto-format QA (Before/After) - {0} blocks, improved {1}').format(summary.get('count', 0), summary.get('improved_count', 0)))
        if not d.get('advanced_backend_available', False):
            hints = ', '.join(d.get('install_hints', [])[:3])
            self.hint.setText(self.tr('Advanced shaping backend is unavailable on this environment. Auto-format preview still works with Qt fallback. Suggested installs: {0}').format(hints))
        else:
            self.hint.setText(self.tr('Advanced shaping backend available. Scores reflect current layout heuristics and can be applied directly.'))

    def apply_all(self):
        profile = str(self.profile.currentData() or 'balanced')
        indices = [int(getattr(item, 'idx', -1)) for item in self._block_items if int(getattr(item, 'idx', -1)) >= 0]
        if self._apply_callback is not None:
            self._apply_callback(indices, profile)
        self.accept()
