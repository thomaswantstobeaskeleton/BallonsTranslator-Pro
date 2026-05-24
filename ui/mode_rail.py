from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QButtonGroup, QFrame, QLabel, QSizePolicy, QToolButton, QVBoxLayout, QWidget

from .design_tokens import TOKENS, WORKFLOW_ACCENTS, badge_style


@dataclass(frozen=True)
class ModeRailItemSpec:
    key: str
    label: str
    tooltip: str = ""
    badge: str = ""
    status: str = "idle"


DEFAULT_MODE_RAIL_ITEMS: tuple[ModeRailItemSpec, ...] = (
    ModeRailItemSpec("home", "Home", "Choose a workflow or continue recent work."),
    ModeRailItemSpec("editor", "Editor", "Manga/comic localization workspace."),
    ModeRailItemSpec("live", "Live", "Realtime screen or Chrome manhua translation."),
    ModeRailItemSpec("quick_image", "Quick", "Quick image translation."),
    ModeRailItemSpec("downloader", "Raws", "Raw/source downloader."),
    ModeRailItemSpec("batch", "Batch", "Batch queue and parent/child projects."),
    ModeRailItemSpec("assist", "Assist", "Translation Assist and QA.", badge="CAT", status="success"),
    ModeRailItemSpec("models", "Models", "Models and providers."),
    ModeRailItemSpec("settings", "Settings", "App and workflow settings."),
    ModeRailItemSpec("diagnostics", "Help", "Diagnostics, logs, docs, and support.", badge="!", status="warning"),
)


class ModeRailButton(QToolButton):
    def __init__(self, spec: ModeRailItemSpec, parent=None):
        super().__init__(parent)
        self.spec = spec
        self.setObjectName("ModeRailButton")
        self.setText(spec.label)
        self.setToolTip(spec.tooltip or spec.label)
        self.setCheckable(True)
        self.setAutoRaise(True)
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.setMinimumHeight(42)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        accent = WORKFLOW_ACCENTS.get(spec.key, TOKENS.colors.border)
        self.setStyleSheet(
            "QToolButton#ModeRailButton {"
            f"border: 1px solid transparent; border-radius: {TOKENS.radius.md}px;"
            f"padding: {TOKENS.spacing.sm}px; color: {TOKENS.colors.text_muted};"
            "text-align: left;"
            "}"
            "QToolButton#ModeRailButton:hover {"
            f"border-color: {accent}; color: {TOKENS.colors.text};"
            "}"
            "QToolButton#ModeRailButton:checked {"
            f"border-color: {accent}; color: {TOKENS.colors.text}; font-weight: 700;"
            f"background: {TOKENS.colors.surface_alt};"
            "}"
        )


class ModeRail(QWidget):
    """Reusable left navigation rail for the phased UI rework.

    It is intentionally independent from MainWindow.  The current left bar can
    keep working while this rail is tested in isolation, then later replace or
    sit beside it in the final app shell.
    """

    mode_requested = Signal(str)

    def __init__(self, items: Optional[Iterable[ModeRailItemSpec]] = None, parent=None):
        super().__init__(parent)
        self.items: List[ModeRailItemSpec] = list(items or DEFAULT_MODE_RAIL_ITEMS)
        self._buttons: dict[str, ModeRailButton] = {}
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._build_ui()

    def _build_ui(self):
        self.setObjectName("ModeRail")
        self.setMinimumWidth(96)
        self.setMaximumWidth(160)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(TOKENS.spacing.sm, TOKENS.spacing.md, TOKENS.spacing.sm, TOKENS.spacing.md)
        layout.setSpacing(TOKENS.spacing.sm)

        title = QLabel(self.tr("Modes"))
        title.setObjectName("ModeRailTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(f"font-size: {TOKENS.typography.caption}px; color: {TOKENS.colors.text_muted};")
        layout.addWidget(title)

        for spec in self.items:
            btn = ModeRailButton(spec, self)
            self._buttons[spec.key] = btn
            self._group.addButton(btn)
            btn.clicked.connect(lambda checked=False, key=spec.key: self.mode_requested.emit(key))
            layout.addWidget(btn)
            if spec.badge:
                badge = QLabel(spec.badge)
                badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
                badge.setStyleSheet(badge_style(spec.status))
                layout.addWidget(badge)

        layout.addStretch(1)

    def set_current_mode(self, key: str):
        btn = self._buttons.get(str(key or ""))
        if btn is not None:
            btn.setChecked(True)

    def current_mode(self) -> str:
        for key, btn in self._buttons.items():
            if btn.isChecked():
                return key
        return ""

    def mode_keys(self) -> List[str]:
        return [it.key for it in self.items]

    def button_for_mode(self, key: str) -> Optional[ModeRailButton]:
        return self._buttons.get(key)
