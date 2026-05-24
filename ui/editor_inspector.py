from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .design_tokens import TOKENS, badge_style, inspector_section_style


@dataclass(frozen=True)
class InspectorTabSpec:
    key: str
    title: str
    description: str


DEFAULT_INSPECTOR_TABS: tuple[InspectorTabSpec, ...] = (
    InspectorTabSpec("text", "Text", "Source and translated text for the selected block."),
    InspectorTabSpec("style", "Style", "Font, size, stroke, fill, shadow, and presets."),
    InspectorTabSpec("layout", "Layout", "Bubble fitting, safe area, reading direction, and overflow checks."),
    InspectorTabSpec("assist", "Assist", "Translation Assist candidates, TM, glossary, SFX, and concordance."),
    InspectorTabSpec("ocr", "OCR", "OCR crop, confidence, engine, and rerun controls."),
    InspectorTabSpec("qa", "QA", "Translation, typography, clipping, and render warnings."),
    InspectorTabSpec("metadata", "Metadata", "Block IDs, geometry, model provenance, and export fields."),
)


class InspectorSection(QFrame):
    def __init__(self, title: str, status: str = "idle", parent=None):
        super().__init__(parent)
        self.setObjectName("InspectorSection")
        self.setStyleSheet(inspector_section_style())
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(TOKENS.spacing.md, TOKENS.spacing.md, TOKENS.spacing.md, TOKENS.spacing.md)
        self._layout.setSpacing(TOKENS.spacing.sm)
        header = QHBoxLayout()
        title_label = QLabel(title)
        title_label.setStyleSheet(f"font-size: {TOKENS.typography.body_large}px; font-weight: 700;")
        header.addWidget(title_label, 1)
        self.status_label = QLabel(status)
        self.status_label.setStyleSheet(badge_style(status))
        header.addWidget(self.status_label)
        self._layout.addLayout(header)

    def addWidget(self, widget):
        self._layout.addWidget(widget)

    def addLayout(self, layout):
        self._layout.addLayout(layout)

    def set_status(self, status: str):
        self.status_label.setText(status)
        self.status_label.setStyleSheet(badge_style(status))


class EditorInspector(QWidget):
    """Right-side inspector shell for the reworked editor workspace.

    The component is standalone so it can be introduced beside the legacy
    `TextPanel`/`DrawingPanel` stack first.  Later PRs should map each tab to
    the existing concrete controls instead of duplicating behavior.
    """

    request_translation_assist = Signal()
    request_ocr_rerun = Signal()
    request_layout_review = Signal()
    request_typography_qa = Signal()

    def __init__(self, tabs: Optional[Iterable[InspectorTabSpec]] = None, parent=None):
        super().__init__(parent)
        self.tabs: List[InspectorTabSpec] = list(tabs or DEFAULT_INSPECTOR_TABS)
        self._tab_widgets: Dict[str, QWidget] = {}
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(TOKENS.spacing.sm)

        title = QLabel(self.tr("Inspector"))
        title.setObjectName("EditorInspectorTitle")
        title.setStyleSheet(f"font-size: {TOKENS.typography.title}px; font-weight: 700;")
        root.addWidget(title)

        subtitle = QLabel(self.tr("Selected block tools, assist, OCR, layout, and QA live here."))
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet(f"font-size: {TOKENS.typography.caption}px; color: {TOKENS.colors.text_muted};")
        root.addWidget(subtitle)

        self.tab_widget = QTabWidget(self)
        root.addWidget(self.tab_widget, 1)

        for spec in self.tabs:
            page = self._build_tab_page(spec)
            self._tab_widgets[spec.key] = page
            self.tab_widget.addTab(page, spec.title)

    def _build_tab_page(self, spec: InspectorTabSpec) -> QWidget:
        page = QWidget(self)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(TOKENS.spacing.sm, TOKENS.spacing.sm, TOKENS.spacing.sm, TOKENS.spacing.sm)
        layout.setSpacing(TOKENS.spacing.md)

        desc = QLabel(spec.description)
        desc.setWordWrap(True)
        desc.setStyleSheet(f"font-size: {TOKENS.typography.caption}px; color: {TOKENS.colors.text_muted};")
        layout.addWidget(desc)

        if spec.key == "text":
            self.source_preview = QPlainTextEdit(page)
            self.source_preview.setReadOnly(True)
            self.source_preview.setPlaceholderText(self.tr("Source text"))
            self.target_preview = QPlainTextEdit(page)
            self.target_preview.setPlaceholderText(self.tr("Translated text"))
            layout.addWidget(self.source_preview)
            layout.addWidget(self.target_preview)
        elif spec.key == "assist":
            section = InspectorSection(self.tr("Translation Assist"), "CAT", page)
            button = QPushButton(self.tr("Open Translation Assist"), section)
            button.clicked.connect(self.request_translation_assist.emit)
            section.addWidget(button)
            layout.addWidget(section)
        elif spec.key == "ocr":
            section = InspectorSection(self.tr("OCR Crop Inspector"), "idle", page)
            button = QPushButton(self.tr("Rerun OCR for selected block"), section)
            button.clicked.connect(self.request_ocr_rerun.emit)
            section.addWidget(button)
            layout.addWidget(section)
        elif spec.key == "layout":
            section = InspectorSection(self.tr("Auto Layout"), "warning", page)
            button = QPushButton(self.tr("Review selected layout"), section)
            button.clicked.connect(self.request_layout_review.emit)
            section.addWidget(button)
            layout.addWidget(section)
        elif spec.key == "qa":
            section = InspectorSection(self.tr("Quality checks"), "warning", page)
            button = QPushButton(self.tr("Run typography QA"), section)
            button.clicked.connect(self.request_typography_qa.emit)
            section.addWidget(button)
            layout.addWidget(section)
        else:
            section = InspectorSection(spec.title, "idle", page)
            placeholder = QLabel(self.tr("This tab will be wired to existing controls in a later UI migration step."), section)
            placeholder.setWordWrap(True)
            section.addWidget(placeholder)
            layout.addWidget(section)

        layout.addStretch(1)
        return page

    def tab_keys(self) -> List[str]:
        return [t.key for t in self.tabs]

    def tab_widget_for_key(self, key: str) -> Optional[QWidget]:
        return self._tab_widgets.get(key)

    def set_text_preview(self, source: str, target: str):
        if hasattr(self, "source_preview"):
            self.source_preview.setPlainText(source or "")
        if hasattr(self, "target_preview"):
            self.target_preview.setPlainText(target or "")
