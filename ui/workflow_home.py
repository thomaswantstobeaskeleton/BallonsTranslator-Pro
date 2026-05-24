from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from .custom_widget.push_button import NoBorderPushBtn
from .design_tokens import (
    TOKENS,
    WORKFLOW_ACCENTS,
    WORKFLOW_GRADIENTS,
    badge_style,
    card_hover_style,
    card_style,
    hero_panel_style,
    primary_button_style,
)


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
    WorkflowCardSpec("editor", "Manga & Comic Editor", "Open raws or an existing project for OCR, translation, inpaint, typesetting, QA, and export.", "Open project or folder", "Production", "Best for scanlation and full chapter work.", "running"),
    WorkflowCardSpec("live", "Live Screen Translator", "Translate a selected screen or Chrome manhua reader region without creating a full project.", "Start live mode", "Reader", "Best for reading in Chrome, games, or visual novels.", "experimental"),
    WorkflowCardSpec("quick_image", "Image Quick Translation", "Drop one or more images, run default OCR/translation/inpaint, preview, and export quickly.", "Translate images", "Fast", "Best for one-off images and testing providers.", "success"),
    WorkflowCardSpec("downloader", "Raw Downloader", "Search sources, choose chapters, queue downloads, and import results into a project.", "Find raws", "Sources", "Best before starting a chapter project.", "warning"),
    WorkflowCardSpec("batch", "Batch Queue", "Process multiple folders, archives, or child projects with progress, cancel, and resume.", "Open queue", "Automation", "Best for many chapters or repeated jobs.", "idle"),
    WorkflowCardSpec("assist", "Translation Assist / QA", "Compare providers, TM matches, glossary hits, concordance, SFX suggestions, and QA warnings.", "Open assist", "CAT", "Best for final polish and consistency.", "success"),
    WorkflowCardSpec("models", "Models & Providers", "Install models, test OCR/translation providers, check caches, and fix missing dependencies.", "Manage setup", "Setup", "Best when a module fails or first-run setup is incomplete.", "idle"),
    WorkflowCardSpec("diagnostics", "Diagnostics / Help", "Run environment checks, inspect logs, export startup reports, and find docs.", "Open diagnostics", "Support", "Best when something looks broken.", "warning"),
)


class WorkflowCard(QFrame):
    activated = Signal(str)

    def __init__(self, spec: WorkflowCardSpec, parent=None):
        super().__init__(parent)
        self.spec = spec
        self.setObjectName("WorkflowCard")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(178)
        accent = WORKFLOW_ACCENTS.get(spec.key, TOKENS.colors.border)
        gradient = WORKFLOW_GRADIENTS.get(spec.key, "transparent")
        self.setStyleSheet(
            "QFrame#WorkflowCard {"
            f"{card_style(accent)}"
            f"background: {TOKENS.colors.surface_glass};"
            "}"
            "QFrame#WorkflowCard:hover {"
            f"{card_hover_style(accent)}"
            "}"
            "QLabel#WorkflowCardAccent {"
            f"background: {gradient};"
            f"border-radius: {TOKENS.radius.pill}px;"
            "}"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(TOKENS.spacing.lg, TOKENS.spacing.lg, TOKENS.spacing.lg, TOKENS.spacing.lg)
        layout.setSpacing(TOKENS.spacing.sm)

        accent_bar = QLabel()
        accent_bar.setObjectName("WorkflowCardAccent")
        accent_bar.setFixedHeight(5)
        layout.addWidget(accent_bar)

        header = QHBoxLayout()
        title = QLabel(spec.title)
        title.setObjectName("WorkflowCardTitle")
        title.setStyleSheet(f"font-size: {TOKENS.typography.title}px; font-weight: 800; color: {TOKENS.colors.text};")
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
        desc.setStyleSheet(f"font-size: {TOKENS.typography.body}px; color: {TOKENS.colors.text_muted}; line-height: 150%;")
        layout.addWidget(desc)

        if spec.recommended_for:
            rec = QLabel(spec.recommended_for)
            rec.setObjectName("WorkflowCardRecommended")
            rec.setWordWrap(True)
            rec.setStyleSheet(f"font-size: {TOKENS.typography.caption}px; color: {TOKENS.colors.text_subtle};")
            layout.addWidget(rec)

        layout.addStretch(1)
        action = NoBorderPushBtn(spec.primary_action)
        action.setObjectName("WorkflowCardAction")
        action.setMinimumHeight(38)
        action.setStyleSheet(primary_button_style(spec.key))
        action.clicked.connect(lambda: self.activated.emit(self.spec.key))
        layout.addWidget(action)


class WorkflowHomeWidget(QWidget):
    """Workflow-card launcher for the reworked Home screen."""

    workflow_requested = Signal(str)

    def __init__(self, cards: Optional[Iterable[WorkflowCardSpec]] = None, parent=None):
        super().__init__(parent)
        self.cards: List[WorkflowCardSpec] = list(cards or DEFAULT_WORKFLOW_CARDS)
        self._card_widgets: Dict[str, WorkflowCard] = {}
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(TOKENS.spacing.lg)

        hero = QFrame(self)
        hero.setObjectName("WorkflowHomeHero")
        hero.setStyleSheet(hero_panel_style("home"))
        hero_layout = QVBoxLayout(hero)
        hero_layout.setContentsMargins(TOKENS.spacing.xl, TOKENS.spacing.xl, TOKENS.spacing.xl, TOKENS.spacing.xl)
        hero_layout.setSpacing(TOKENS.spacing.sm)

        header = QLabel(self.tr("Choose your workflow"))
        header.setObjectName("WorkflowHomeTitle")
        header.setStyleSheet(f"font-size: {TOKENS.typography.display}px; font-weight: 900; color: {TOKENS.colors.text};")
        hero_layout.addWidget(header)

        subtitle = QLabel(self.tr("Start with a simple card, then use menus or the command palette whenever you need advanced tools."))
        subtitle.setObjectName("WorkflowHomeSubtitle")
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet(f"font-size: {TOKENS.typography.body_large}px; color: {TOKENS.colors.text_muted};")
        hero_layout.addWidget(subtitle)
        layout.addWidget(hero)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(TOKENS.spacing.lg)
        grid.setVerticalSpacing(TOKENS.spacing.lg)
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
