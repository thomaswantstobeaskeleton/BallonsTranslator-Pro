from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from .custom_widget.push_button import NoBorderPushBtn
from .design_tokens import TOKENS, WORKFLOW_ACCENTS, badge_style, card_style


@dataclass(frozen=True)
class WorkflowCardSpec:
    key: str
    title: str
    description: str
    primary_action: str
    badge: str = ""
    recommended_for: str = ""
    status: str = "idle"


DEFAULT_WORKFLOW_CARDS: tuple[WorkflowCardSpec, ...] = (
    WorkflowCardSpec(
        key="editor",
        title="Manga & Comic Editor",
        description="Open raws or an existing project for OCR, translation, inpaint, typesetting, QA, and export.",
        primary_action="Open project or folder",
        badge="Production",
        recommended_for="Best for scanlation and full chapter work.",
        status="running",
    ),
    WorkflowCardSpec(
        key="live",
        title="Live Screen Translator",
        description="Translate a selected screen or Chrome manhua reader region without creating a full project.",
        primary_action="Start live mode",
        badge="Reader",
        recommended_for="Best for reading in Chrome, games, or visual novels.",
        status="experimental",
    ),
    WorkflowCardSpec(
        key="quick_image",
        title="Image Quick Translation",
        description="Drop one or more images, run default OCR/translation/inpaint, preview, and export quickly.",
        primary_action="Translate images",
        badge="Fast",
        recommended_for="Best for one-off images and testing providers.",
        status="success",
    ),
    WorkflowCardSpec(
        key="downloader",
        title="Raw Downloader",
        description="Search sources, choose chapters, queue downloads, and import results into a project.",
        primary_action="Find raws",
        badge="Sources",
        recommended_for="Best before starting a chapter project.",
        status="warning",
    ),
    WorkflowCardSpec(
        key="batch",
        title="Batch Queue",
        description="Process multiple folders, archives, or child projects with progress, cancel, and resume.",
        primary_action="Open queue",
        badge="Automation",
        recommended_for="Best for many chapters or repeated jobs.",
        status="idle",
    ),
    WorkflowCardSpec(
        key="assist",
        title="Translation Assist / QA",
        description="Compare providers, TM matches, glossary hits, concordance, SFX suggestions, and QA warnings.",
        primary_action="Open assist",
        badge="CAT",
        recommended_for="Best for final polish and consistency.",
        status="success",
    ),
    WorkflowCardSpec(
        key="models",
        title="Models & Providers",
        description="Install models, test OCR/translation providers, check caches, and fix missing dependencies.",
        primary_action="Manage setup",
        badge="Setup",
        recommended_for="Best when a module fails or first-run setup is incomplete.",
        status="idle",
    ),
    WorkflowCardSpec(
        key="diagnostics",
        title="Diagnostics / Help",
        description="Run environment checks, inspect logs, export startup reports, and find docs.",
        primary_action="Open diagnostics",
        badge="Support",
        recommended_for="Best when something looks broken.",
        status="warning",
    ),
)


class WorkflowCard(QFrame):
    activated = Signal(str)

    def __init__(self, spec: WorkflowCardSpec, parent=None):
        super().__init__(parent)
        self.spec = spec
        self.setObjectName("WorkflowCard")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(160)
        accent = WORKFLOW_ACCENTS.get(spec.key, TOKENS.colors.border)
        self.setStyleSheet(card_style(accent))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(TOKENS.spacing.lg, TOKENS.spacing.lg, TOKENS.spacing.lg, TOKENS.spacing.lg)
        layout.setSpacing(TOKENS.spacing.sm)

        header = QHBoxLayout()
        title = QLabel(spec.title)
        title.setObjectName("WorkflowCardTitle")
        title.setStyleSheet(f"font-size: {TOKENS.typography.title}px; font-weight: 700; color: {TOKENS.colors.text};")
        title.setWordWrap(True)
        header.addWidget(title, 1)
        if spec.badge:
            badge = QLabel(spec.badge)
            badge.setObjectName("WorkflowCardBadge")
            badge.setStyleSheet(badge_style(spec.status))
            header.addWidget(badge, 0, Qt.AlignmentFlag.AlignTop)
        layout.addLayout(header)

        desc = QLabel(spec.description)
        desc.setObjectName("WorkflowCardDescription")
        desc.setWordWrap(True)
        desc.setStyleSheet(f"font-size: {TOKENS.typography.body}px; color: {TOKENS.colors.text_muted};")
        layout.addWidget(desc)

        if spec.recommended_for:
            rec = QLabel(spec.recommended_for)
            rec.setObjectName("WorkflowCardRecommended")
            rec.setWordWrap(True)
            rec.setStyleSheet(f"font-size: {TOKENS.typography.caption}px; color: {TOKENS.colors.text_muted};")
            layout.addWidget(rec)

        layout.addStretch(1)
        action = NoBorderPushBtn(spec.primary_action)
        action.setObjectName("WorkflowCardAction")
        action.setMinimumHeight(34)
        action.clicked.connect(lambda: self.activated.emit(self.spec.key))
        layout.addWidget(action)


class WorkflowHomeWidget(QWidget):
    """Workflow-card launcher for the reworked Home screen.

    This widget is intentionally standalone so it can be embedded into the
    existing WelcomeWidget first, then promoted to the final app shell later.
    """

    workflow_requested = Signal(str)

    def __init__(self, cards: Optional[Iterable[WorkflowCardSpec]] = None, parent=None):
        super().__init__(parent)
        self.cards: List[WorkflowCardSpec] = list(cards or DEFAULT_WORKFLOW_CARDS)
        self._card_widgets: Dict[str, WorkflowCard] = {}
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(TOKENS.spacing.md)

        header = QLabel(self.tr("Choose a workflow"))
        header.setObjectName("WorkflowHomeTitle")
        header.setStyleSheet(f"font-size: {TOKENS.typography.headline}px; font-weight: 700;")
        layout.addWidget(header)

        subtitle = QLabel(self.tr("Pick the mode that matches what you want to do. You can still reach every advanced command from menus or the command palette."))
        subtitle.setObjectName("WorkflowHomeSubtitle")
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet(f"font-size: {TOKENS.typography.body_large}px; color: {TOKENS.colors.text_muted};")
        layout.addWidget(subtitle)

        grid = QGridLayout()
        grid.setContentsMargins(0, TOKENS.spacing.sm, 0, 0)
        grid.setHorizontalSpacing(TOKENS.spacing.md)
        grid.setVerticalSpacing(TOKENS.spacing.md)
        layout.addLayout(grid)

        for idx, spec in enumerate(self.cards):
            card = WorkflowCard(spec, self)
            card.activated.connect(self.workflow_requested.emit)
            self._card_widgets[spec.key] = card
            row, col = divmod(idx, 2)
            grid.addWidget(card, row, col)

    def card_keys(self) -> List[str]:
        return [card.key for card in self.cards]

    def card_widget(self, key: str) -> Optional[WorkflowCard]:
        return self._card_widgets.get(key)
