from qtpy.QtCore import Qt, Signal, QPropertyAnimation, QEasingCurve
from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QListWidget, QListWidgetItem, QFrame, QProgressBar, QGraphicsOpacityEffect, QGridLayout, QComboBox


class PipelineInsightsWidget(QWidget):
    rerun_stage_requested = Signal(str)
    apply_regex_profile_requested = Signal()
    open_mask_diagnostics_requested = Signal()
    apply_project_ops_requested = Signal()
    open_ocr_crop_inspector_requested = Signal()
    open_reading_order_editor_requested = Signal()
    run_layout_review_requested = Signal()
    open_batch_style_requested = Signal()
    open_typography_qa_requested = Signal()
    apply_workflow_preset_requested = Signal(str)
    run_workflow_preset_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('PipelineInsightsWidget')
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        title = QLabel(self.tr('Pipeline Insights'))
        title.setObjectName('PipelineInsightsTitle')
        lay.addWidget(title)
        self.warning_badge = QLabel(self.tr('Warnings: 0'), self)
        self.warning_badge.setObjectName('PipelineInsightsWarningBadge')
        lay.addWidget(self.warning_badge)

        progress_card = QFrame(self)
        progress_card.setObjectName('PipelineCard')
        progress_lay = QVBoxLayout(progress_card)
        progress_lay.setContentsMargins(10, 10, 10, 10)
        progress_lay.addWidget(QLabel(self.tr('Pipeline Progress')))
        self.stage_progress = QProgressBar(self)
        self.stage_progress.setRange(0, 100)
        self.stage_progress.setValue(0)
        progress_lay.addWidget(self.stage_progress)
        self.job_label = QLabel(self.tr('Job: idle'), self)
        self.job_label.setObjectName('PipelineJobLabel')
        progress_lay.addWidget(self.job_label)
        lay.addWidget(progress_card)

        preset_card = QFrame(self)
        preset_card.setObjectName('PipelineCard')
        preset_lay = QVBoxLayout(preset_card)
        preset_lay.setContentsMargins(10, 10, 10, 10)
        preset_lay.addWidget(QLabel(self.tr('Workflow preset')))
        preset_row = QHBoxLayout()
        self.workflow_preset_combo = QComboBox(self)
        self.workflow_preset_combo.setToolTip(self.tr('Apply a staged manga workflow preset without opening the Pipeline menu.'))
        try:
            from utils.workflow_presets import list_workflow_presets
            for preset_id, preset in list_workflow_presets().items():
                self.workflow_preset_combo.addItem(self.tr(str(preset.get('label', preset_id))), preset_id)
        except Exception:
            for preset_id in ('full', 'detect_ocr', 'translate', 'inpaint', 'lettering_review'):
                self.workflow_preset_combo.addItem(preset_id, preset_id)
        preset_row.addWidget(self.workflow_preset_combo, 1)
        self.apply_workflow_preset_btn = QPushButton(self.tr('Apply'), self)
        self.apply_workflow_preset_btn.setToolTip(self.tr('Set pipeline stage toggles to this preset.'))
        self.apply_workflow_preset_btn.clicked.connect(self._emit_apply_workflow_preset)
        preset_row.addWidget(self.apply_workflow_preset_btn)
        self.run_workflow_preset_btn = QPushButton(self.tr('Apply + Run'), self)
        self.run_workflow_preset_btn.setToolTip(self.tr('Apply the preset and immediately run the pipeline.'))
        self.run_workflow_preset_btn.clicked.connect(self._emit_run_workflow_preset)
        preset_row.addWidget(self.run_workflow_preset_btn)
        preset_lay.addLayout(preset_row)
        lay.addWidget(preset_card)

        btn_row = QHBoxLayout()
        for stage in ('detect', 'ocr', 'translate', 'inpaint', 'render'):
            b = QPushButton(stage.capitalize(), self)
            b.clicked.connect(lambda _=False, s=stage: self.rerun_stage_requested.emit(s))
            btn_row.addWidget(b)
        lay.addLayout(btn_row)
        self.apply_profile_btn = QPushButton(self.tr('Apply Regex Profile'), self)
        self.apply_profile_btn.clicked.connect(self.apply_regex_profile_requested.emit)
        lay.addWidget(self.apply_profile_btn)
        self.mask_diag_btn = QPushButton(self.tr('Mask Diagnostics'), self)
        self.mask_diag_btn.clicked.connect(self.open_mask_diagnostics_requested.emit)
        lay.addWidget(self.mask_diag_btn)
        self.ops_btn = QPushButton(self.tr('Apply Project Ops'), self)
        self.ops_btn.clicked.connect(self.apply_project_ops_requested.emit)
        lay.addWidget(self.ops_btn)
        self.ocr_inspector_btn = QPushButton(self.tr('OCR Crop Inspector'), self)
        self.ocr_inspector_btn.clicked.connect(self.open_ocr_crop_inspector_requested.emit)
        lay.addWidget(self.ocr_inspector_btn)
        self.reading_order_btn = QPushButton(self.tr('Reading Order Editor'), self)
        self.reading_order_btn.clicked.connect(self.open_reading_order_editor_requested.emit)
        lay.addWidget(self.reading_order_btn)
        self.layout_review_btn = QPushButton(self.tr('Layout Review Agent'), self)
        self.layout_review_btn.setObjectName('PipelinePrimaryAction')
        self.layout_review_btn.clicked.connect(self.run_layout_review_requested.emit)
        lay.addWidget(self.layout_review_btn)
        self.batch_style_btn = QPushButton(self.tr('Batch Text Style Override'), self)
        self.batch_style_btn.clicked.connect(self.open_batch_style_requested.emit)
        lay.addWidget(self.batch_style_btn)
        self.typography_qa_btn = QPushButton(self.tr('Typography QA Report'), self)
        self.typography_qa_btn.clicked.connect(self.open_typography_qa_requested.emit)
        self.typography_qa_btn.setToolTip(self.tr('Export or apply project-wide rendering QA fixes for overflow, fallback fonts, RTL, and vertical CJK.'))
        lay.addWidget(self.typography_qa_btn)
        self.api_status_label = QLabel(self.tr('Automation API: off'), self)
        self.api_status_label.setObjectName('PipelineApiStatus')
        lay.addWidget(self.api_status_label)

        self.empty_state = QLabel(self.tr('No warnings yet. Run the pipeline or layout review to populate actionable checks.'), self)
        self.empty_state.setWordWrap(True)
        self.empty_state.setObjectName('PipelineEmptyState')
        lay.addWidget(self.empty_state)
        self.warning_list = QListWidget(self)
        self.warning_list.setFrameShape(QFrame.NoFrame)
        lay.addWidget(self.warning_list, 1)
        lay.addWidget(QLabel(self.tr('Event Timeline')))
        self.event_list = QListWidget(self)
        self.event_list.setFrameShape(QFrame.NoFrame)
        self.event_list.setMaximumHeight(120)
        lay.addWidget(self.event_list)
        self.provider_grid = QGridLayout()
        self.provider_grid.setHorizontalSpacing(8)
        self.provider_grid.setVerticalSpacing(4)
        self.provider_labels = {}
        providers = (
            ('detector', self.tr('Detector')),
            ('ocr', self.tr('OCR')),
            ('translator', self.tr('Translator')),
            ('inpainter', self.tr('Inpainter')),
        )
        for row, (key, label_text) in enumerate(providers):
            label = QLabel(f'● {label_text}: {self.tr("unknown")}', self)
            label.setObjectName('PipelineProviderStatus')
            self.provider_labels[key] = label
            self.provider_grid.addWidget(label, row, 0)
        lay.addLayout(self.provider_grid)
        lay.addWidget(QLabel(self.tr('Engine Registry')))
        self.engine_list = QListWidget(self)
        self.engine_list.setMaximumHeight(110)
        lay.addWidget(self.engine_list)

        self._opacity = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self._opacity)
        self._anim = QPropertyAnimation(self._opacity, b'opacity', self)
        self._anim.setDuration(320)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        self._anim.start()
        self._progress_pulse = QPropertyAnimation(self.stage_progress, b"windowOpacity", self)
        self._progress_pulse.setDuration(180)
        self._progress_pulse.setStartValue(0.75)
        self._progress_pulse.setEndValue(1.0)

    def _current_workflow_preset_id(self) -> str:
        return str(self.workflow_preset_combo.currentData() or self.workflow_preset_combo.currentText() or '')

    def _emit_apply_workflow_preset(self):
        self.apply_workflow_preset_requested.emit(self._current_workflow_preset_id())

    def _emit_run_workflow_preset(self):
        self.run_workflow_preset_requested.emit(self._current_workflow_preset_id())

    def set_pipeline_progress(self, value: int):
        self.stage_progress.setValue(max(0, min(100, int(value))))
        self._progress_pulse.stop()
        self._progress_pulse.start()

    def set_job_id(self, job_id: str):
        self.job_label.setText(self.tr(f'Job: {job_id}'))

    def set_provider_status(self, key: str, label: str, status: str):
        color = {
            'ready': '#2ecc71',
            'missing': '#f1c40f',
            'error': '#e74c3c',
            'unknown': '#95a5a6',
        }.get((status or 'unknown').lower(), '#95a5a6')
        text_status = (status or 'unknown').lower()
        widget = self.provider_labels.get(key)
        if widget is None:
            return
        widget.setText(f'● {label}: {text_status}')
        widget.setStyleSheet(f'color: {color};')

    def add_warning(self, code: str, message: str):
        item = QListWidgetItem(f'[{code}] {message}')
        item.setToolTip(message)
        self.warning_list.addItem(item)
        self.empty_state.setVisible(False)
        self.warning_badge.setText(self.tr(f'Warnings: {self.warning_list.count()}'))
        self._anim.stop()
        self._anim.setDuration(140)
        self._anim.start()

    def add_event(self, code: str, message: str):
        self.event_list.addItem(QListWidgetItem(f'[{code}] {message}'))
        while self.event_list.count() > 60:
            self.event_list.takeItem(0)

    def set_engine_registry(self, snapshot: dict):
        self.engine_list.clear()
        for key, info in (snapshot or {}).items():
            sel = (info or {}).get("selected", "")
            loaded = (info or {}).get("loaded", "")
            self.engine_list.addItem(QListWidgetItem(f'{key}: selected={sel or "-"} | loaded={loaded or "-"}'))

    def reset(self):
        self.warning_list.clear()
        self.empty_state.setVisible(True)
        self.warning_badge.setText(self.tr('Warnings: 0'))
        self.event_list.clear()
        self.engine_list.clear()
        self.stage_progress.setValue(0)
        self.set_job_id('idle')

    def set_api_status(self, enabled: bool, queue_depth: int = 0):
        self.api_status_label.setText(self.tr(f'Automation API: {"on" if enabled else "off"} | queue={max(0, int(queue_depth))}'))
