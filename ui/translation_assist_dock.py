from __future__ import annotations

from typing import Callable, Dict, Optional
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QLabel, QPlainTextEdit, QPushButton,
    QListWidget, QListWidgetItem, QHBoxLayout, QCheckBox, QSpinBox,
    QComboBox
)


class TranslationAssistDock(QDockWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr('Translation Assist (beta)'))
        self._refresh_cb: Optional[Callable[[Dict], None]] = None
        self._apply_cb: Optional[Callable[[str], None]] = None
        self._add_tm_cb: Optional[Callable[[str], None]] = None
        self._add_glossary_cb: Optional[Callable[[str], None]] = None
        self._apply_text_cb: Optional[Callable[[str], None]] = None
        root = QWidget(self)
        lay = QVBoxLayout(root)
        hint = QLabel(self.tr('Translation Assist never overwrites text automatically. Click Apply to update current block.'), root)
        hint.setWordWrap(True)
        lay.addWidget(hint)
        self.source_box = QPlainTextEdit(root)
        self.source_box.setReadOnly(True)
        self.source_box.setPlaceholderText(self.tr('Source text for selected block'))
        self.target_box = QPlainTextEdit(root)
        self.target_box.setReadOnly(True)
        self.target_box.setPlaceholderText(self.tr('Current target text for selected block'))
        self.candidates = QListWidget(root)
        self.candidates.setToolTip(self.tr('Candidates from MT/LLM/TM/Glossary/SFX.'))
        self.summary = QLabel(self.tr('Select a text block and click Refresh Assist.'), root)
        self.summary.setWordWrap(True)
        self.qa_box = QPlainTextEdit(root)
        self.qa_box.setReadOnly(True)
        self.qa_box.setPlaceholderText(self.tr('Translation QA warnings for current block'))
        self.edit_candidate_box = QPlainTextEdit(root)
        self.edit_candidate_box.setPlaceholderText(self.tr('Edit/merge selected candidate text before applying'))
        self.query_tm = QCheckBox(self.tr('TM'), root)
        self.query_glossary = QCheckBox(self.tr('Glossary'), root)
        self.query_sfx = QCheckBox(self.tr('SFX'), root)
        self.query_concordance = QCheckBox(self.tr('Concordance'), root)
        for cb in (self.query_tm, self.query_glossary, self.query_sfx):
            cb.setChecked(True)
        self.max_candidates = QSpinBox(root)
        self.max_candidates.setRange(1, 20)
        self.max_candidates.setValue(6)
        self.max_candidates.setToolTip(self.tr('Maximum number of assist candidates kept after de-duplication.'))
        self.prompt_profile = QComboBox(root)
        self.prompt_profile.setToolTip(self.tr('Prompt profile for assist candidate generation/QA context.'))
        self.provider_list = QListWidget(root)
        self.provider_list.setToolTip(self.tr('Select providers for assist/compare.'))
        self.provider_list.setMaximumHeight(160)
        for nm in ['TM', 'Glossary', 'Concordance', 'SFX', 'google', 'deepl', 'openai', 'ollama', 'lmstudio']:
            it = QListWidgetItem(str(nm))
            it.setFlags(it.flags() | it.flags().ItemIsUserCheckable)
            it.setCheckState(Qt.Unchecked)
            self.provider_list.addItem(it)
        self.compare_preset = QComboBox(root)
        self.compare_preset.addItems(["low_latency", "high_quality"])
        self.compare_preset.setToolTip(self.tr('Preset for multi-provider compare orchestration.'))
        self.compare_scope = QComboBox(root)
        self.compare_scope.addItems(["translator", "ocr", "detector", "inpainter"])
        self.compare_scope.setToolTip(self.tr('Compare scope: translation providers or pipeline module choices for current block/page context.'))
        scope_row = QHBoxLayout()
        scope_row.addWidget(QLabel(self.tr('Sources:'), root))
        scope_row.addWidget(self.query_tm)
        scope_row.addWidget(self.query_glossary)
        scope_row.addWidget(self.query_sfx)
        scope_row.addWidget(self.query_concordance)
        scope_row.addWidget(QLabel(self.tr('Max'), root))
        scope_row.addWidget(self.max_candidates)
        scope_row.addWidget(QLabel(self.tr('Profile'), root))
        scope_row.addWidget(self.prompt_profile)
        scope_row.addWidget(QLabel(self.tr('Compare'), root))
        scope_row.addWidget(self.compare_preset)
        scope_row.addWidget(QLabel(self.tr('Scope'), root))
        scope_row.addWidget(self.compare_scope)
        row = QHBoxLayout()
        self.refresh_btn = QPushButton(self.tr('Refresh Assist'), root)
        self.refresh_btn.setToolTip(self.tr('Query selected sources and rebuild candidate list for current text block.'))
        self.compare_btn = QPushButton(self.tr('Compare Providers'), root)
        self.compare_btn.setToolTip(self.tr('Run provider compare using selected providers/preset.'))
        self.apply_btn = QPushButton(self.tr('Apply Selected Candidate'), root)
        self.apply_btn.setToolTip(self.tr('Apply selected candidate to current block (undoable).'))
        self.apply_btn.setEnabled(False)
        self.add_tm_btn = QPushButton(self.tr('Add Selected to TM'), root)
        self.add_glossary_btn = QPushButton(self.tr('Add Selected to Glossary'), root)
        self.apply_edited_btn = QPushButton(self.tr('Apply Edited Text'), root)
        self.clear_cache_btn = QPushButton(self.tr('Clear Assist Cache'), root)
        self.fullscreen_btn = QPushButton(self.tr('Fullscreen'), root)
        self.add_tm_btn.setEnabled(False)
        self.add_glossary_btn.setEnabled(False)
        self.apply_edited_btn.setEnabled(False)
        row.addWidget(self.refresh_btn)
        row.addWidget(self.compare_btn)
        row.addWidget(self.apply_btn)
        row.addWidget(self.fullscreen_btn)
        lay.addWidget(self.source_box)
        lay.addWidget(self.target_box)
        lay.addWidget(self.candidates)
        lay.addWidget(self.qa_box)
        lay.addWidget(self.summary)
        lay.addLayout(scope_row)
        lay.addWidget(self.provider_list)
        lay.addLayout(row)
        row2 = QHBoxLayout()
        row2.addWidget(self.add_tm_btn)
        row2.addWidget(self.add_glossary_btn)
        row2.addWidget(self.apply_edited_btn)
        row2.addWidget(self.clear_cache_btn)
        lay.addWidget(self.edit_candidate_box)
        lay.addLayout(row2)
        self.setWidget(root)
        self.refresh_btn.clicked.connect(self._on_refresh)
        self.compare_btn.clicked.connect(self._on_compare)
        self.apply_btn.clicked.connect(self._on_apply)
        self.add_tm_btn.clicked.connect(self._on_add_tm)
        self.add_glossary_btn.clicked.connect(self._on_add_glossary)
        self.apply_edited_btn.clicked.connect(self._on_apply_edited)
        self.candidates.itemSelectionChanged.connect(self._on_candidate_selection_changed)
        self.clear_cache_btn.clicked.connect(self._on_clear_cache)
        self.fullscreen_btn.clicked.connect(self._toggle_fullscreen)

    def _toggle_fullscreen(self):
        if self.isFloating():
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()
        else:
            self.setFloating(True)
            self.showFullScreen()

    def set_preview(self, source: str, target: str):
        self.source_box.setPlainText(source or '')
        self.target_box.setPlainText(target or '')

    def set_callbacks(self, refresh_cb: Callable[[Dict], None], apply_cb: Callable[[str], None], add_tm_cb: Callable[[str], None], add_glossary_cb: Callable[[str], None], apply_text_cb: Callable[[str], None]):
        self._refresh_cb = refresh_cb
        self._apply_cb = apply_cb
        self._add_tm_cb = add_tm_cb
        self._add_glossary_cb = add_glossary_cb
        self._apply_text_cb = apply_text_cb

    def set_query_defaults(self, opts: Dict):
        self.query_tm.setChecked(bool(opts.get('auto_query_tm', True)))
        self.query_glossary.setChecked(bool(opts.get('auto_query_glossary', True)))
        self.query_sfx.setChecked(bool(opts.get('auto_query_sfx', True)))
        self.query_concordance.setChecked(bool(opts.get('auto_query_concordance', False)))
        self.max_candidates.setValue(int(opts.get('max_mt_candidates', 6) or 6))
        selected = set(str(x).strip() for x in list(opts.get('preferred_assist_providers', []) or []) if str(x).strip())
        for i in range(self.provider_list.count()):
            it = self.provider_list.item(i)
            it.setCheckState(Qt.Checked if it.text() in selected else Qt.Unchecked)
        self.prompt_profile.clear()
        for p in list(opts.get('prompt_profiles', []) or []):
            self.prompt_profile.addItem(str(p))
        dflt = str(opts.get('default_assist_prompt_profile', 'dialogue') or 'dialogue')
        idx = self.prompt_profile.findText(dflt)
        if idx >= 0:
            self.prompt_profile.setCurrentIndex(idx)

    def set_assist_payload(self, payload: Dict):
        self.set_preview(payload.get('source_text', ''), payload.get('current_target_text', ''))
        self.candidates.clear()
        for row in list(payload.get('candidates', []) or []):
            cid = str(row.get('candidate_id', '') or '').strip()
            provider = str(row.get('provider', '') or '').strip()
            text = str(row.get('text', '') or '').strip()
            tel = dict(row.get('telemetry', {}) or {})
            ms = int(tel.get('latency_ms', 0) or 0)
            src = str(tel.get('source', '') or '').strip()
            meta = f" ({ms}ms{', '+src if src else ''})" if (ms or src) else ""
            item = QListWidgetItem(f"[{provider}{meta}] {text}")
            item.setData(32, cid)
            item.setData(33, text)
            self.candidates.addItem(item)
        self.summary.setText(str(payload.get('summary', self.tr('Assist refreshed.'))))
        qa_lines = list(payload.get('qa_warnings', []) or [])
        self.qa_box.setPlainText("\n".join(str(x) for x in qa_lines) if qa_lines else "")
        self.apply_btn.setEnabled(self.candidates.count() > 0)

    def _on_refresh(self):
        if self._refresh_cb is not None:
            self._refresh_cb({
                'auto_query_tm': self.query_tm.isChecked(),
                'auto_query_glossary': self.query_glossary.isChecked(),
                'auto_query_sfx': self.query_sfx.isChecked(),
                'auto_query_concordance': self.query_concordance.isChecked(),
                'max_mt_candidates': int(self.max_candidates.value()),
                'preferred_assist_providers': self._selected_provider_list(),
                'default_assist_prompt_profile': str(self.prompt_profile.currentText() or 'dialogue'),
            })

    def _on_compare(self):
        if self._refresh_cb is not None:
            self._refresh_cb({
                'run_compare': True,
                'compare_preset': str(self.compare_preset.currentText() or 'low_latency'),
                'compare_scope': str(self.compare_scope.currentText() or 'translator'),
                'preferred_assist_providers': self._selected_provider_list(),
                'max_mt_candidates': int(self.max_candidates.value()),
            })

    def _on_apply(self):
        item = self.candidates.currentItem()
        if item is None:
            return
        cid = str(item.data(32) or '').strip()
        if cid and self._apply_cb is not None:
            self._apply_cb(cid)

    def _on_candidate_selection_changed(self):
        active = self.candidates.currentItem() is not None
        self.apply_btn.setEnabled(active)
        self.add_tm_btn.setEnabled(active)
        self.add_glossary_btn.setEnabled(active)
        self.apply_edited_btn.setEnabled(active)
        if active:
            self.edit_candidate_box.setPlainText(str(self.candidates.currentItem().data(33) or ''))

    def _on_add_tm(self):
        item = self.candidates.currentItem()
        if item is None:
            return
        cid = str(item.data(32) or '').strip()
        if cid and self._add_tm_cb is not None:
            self._add_tm_cb(cid)

    def _on_add_glossary(self):
        item = self.candidates.currentItem()
        if item is None:
            return
        cid = str(item.data(32) or '').strip()
        if cid and self._add_glossary_cb is not None:
            self._add_glossary_cb(cid)

    def _on_apply_edited(self):
        text = str(self.edit_candidate_box.toPlainText() or '').strip()
        if text and self._apply_text_cb is not None:
            self._apply_text_cb(text)

    def _on_clear_cache(self):
        if self._refresh_cb is not None:
            self._refresh_cb({'clear_assist_cache': True})

    def set_busy(self, busy: bool):
        self.refresh_btn.setEnabled(not busy)
        self.compare_btn.setEnabled(not busy)
        self.apply_btn.setEnabled(not busy and self.candidates.currentItem() is not None)
        self.summary.setText(self.tr('Loading...') if busy else self.tr('Ready.'))

    def refresh_provider_list(self, providers: list, preserve_checked: bool = True):
        checked = set()
        if preserve_checked:
            for i in range(self.provider_list.count()):
                it = self.provider_list.item(i)
                if it.checkState() == Qt.Checked:
                    checked.add(str(it.text()))
        self.provider_list.clear()
        for nm in providers:
            it = QListWidgetItem(str(nm))
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if str(nm) in checked else Qt.Unchecked)
            self.provider_list.addItem(it)

    def _selected_provider_list(self):
        out = []
        for i in range(self.provider_list.count()):
            it = self.provider_list.item(i)
            if it.checkState() == Qt.Checked:
                out.append(str(it.text()))
        return out
