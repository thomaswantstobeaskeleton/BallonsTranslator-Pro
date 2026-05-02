from qtpy.QtCore import Qt, Signal, QPropertyAnimation, QEasingCurve
from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QListWidget, QListWidgetItem, QFrame, QProgressBar, QGraphicsOpacityEffect, QGridLayout


class PipelineInsightsWidget(QWidget):
    rerun_stage_requested = Signal(str)
    apply_regex_profile_requested = Signal()
    open_mask_diagnostics_requested = Signal()

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
        self.warning_badge.setText(self.tr(f'Warnings: {self.warning_list.count()}'))

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
        self.warning_badge.setText(self.tr('Warnings: 0'))
        self.event_list.clear()
        self.engine_list.clear()
        self.stage_progress.setValue(0)
        self.set_job_id('idle')
