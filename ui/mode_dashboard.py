from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Optional

from qtpy.QtCore import Signal
from qtpy.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget

from .design_tokens import TOKENS, WORKFLOW_ACCENTS, badge_style, card_style, hero_panel_style, primary_button_style, secondary_button_style


@dataclass(frozen=True)
class DashboardActionSpec:
    key: str
    label: str
    description: str
    category: str = "Action"
    primary: bool = False
    status: str = "idle"


@dataclass(frozen=True)
class DashboardMetricSpec:
    key: str
    label: str
    value: str
    status: str = "idle"
    description: str = ""


@dataclass(frozen=True)
class ModeDashboardSpec:
    key: str
    title: str
    description: str
    primary_action: str = "Open"
    actions: tuple[DashboardActionSpec, ...] = tuple()
    metrics: tuple[DashboardMetricSpec, ...] = tuple()
    advanced_notes: tuple[str, ...] = tuple()


DEFAULT_MODE_DASHBOARDS: Dict[str, ModeDashboardSpec] = {
    "editor": ModeDashboardSpec(
        key="editor",
        title="Manga & Comic Editor",
        description="Production workspace for detection, OCR, translation, cleanup, typesetting, QA, and export.",
        primary_action="Open project or folder",
        actions=(
            DashboardActionSpec("run_pipeline", "Run full pipeline", "Detection -> OCR -> Translation -> Inpaint -> Render.", "Pipeline", True, "running"),
            DashboardActionSpec("selected_ocr", "OCR selected blocks", "Rerun OCR without touching the whole page.", "OCR"),
            DashboardActionSpec("layout_review", "Review layout", "Check overflow, clipping, safe areas, and typography.", "QA", False, "warning"),
            DashboardActionSpec("export_proof", "Export proof pack", "Create reviewer handoff with warnings and manifests.", "Export"),
        ),
        metrics=(
            DashboardMetricSpec("pages", "Pages", "Project", "idle", "Current project/page state."),
            DashboardMetricSpec("warnings", "Warnings", "0", "success", "Typography, mask, and export warnings."),
            DashboardMetricSpec("models", "Models", "Ready", "success", "Detector/OCR/inpaint/model health."),
        ),
        advanced_notes=("Keep manual text, style, mask, and layout controls visible.", "Do not hide professional scanlation tools behind a beginner-only flow."),
    ),
    "live": ModeDashboardSpec(
        key="live",
        title="Live Screen / Chrome Manhua Translator",
        description="Watch a selected screen/window/Chrome region, OCR changed text, translate, and show an overlay.",
        primary_action="Start live region picker",
        actions=(
            DashboardActionSpec("pick_region", "Select region", "Draw a screen region or choose a window.", "Capture", True, "running"),
            DashboardActionSpec("chrome_profile", "Chrome Manhua preset", "Follow Chrome and translate after scroll changes.", "Profile", False, "experimental"),
            DashboardActionSpec("overlay", "Overlay settings", "Opacity, click-through, font, padding, and history.", "Display"),
            DashboardActionSpec("privacy", "Privacy controls", "Confirm captures/text are not persisted by default.", "Safety", False, "success"),
        ),
        metrics=(
            DashboardMetricSpec("capture", "Capture", "Idle", "idle", "Region watcher status."),
            DashboardMetricSpec("ocr", "OCR", "Provider", "warning", "Realtime provider health."),
            DashboardMetricSpec("latency", "Latency", "-- ms", "idle", "OCR + translation timing."),
        ),
        advanced_notes=("SAM/refiner helpers should run only in high-quality/slower mode.", "Overlay must never contaminate OCR capture."),
    ),
    "assist": ModeDashboardSpec(
        key="assist",
        title="Translation Assist / QA",
        description="ImageTrans-style professional assist surface for MT candidates, TM, glossary, concordance, SFX, and QA.",
        primary_action="Open Translation Assist",
        actions=(
            DashboardActionSpec("compare", "Compare providers", "Generate candidate translations from selected providers.", "MT", True, "cat"),
            DashboardActionSpec("tm", "Translation memory", "Find fuzzy matches and add current source/target pair.", "TM"),
            DashboardActionSpec("glossary", "Glossary violations", "Check hard/soft termbase rules.", "QA", False, "warning"),
            DashboardActionSpec("sfx", "SFX dictionary", "Search sound effects and lettering notes.", "Manga"),
        ),
        metrics=(
            DashboardMetricSpec("candidates", "Candidates", "0", "idle", "Provider candidates for selected text."),
            DashboardMetricSpec("glossary", "Glossary", "Ready", "success", "Termbase status."),
            DashboardMetricSpec("qa", "QA", "Needs run", "warning", "Project/block QA status."),
        ),
        advanced_notes=("Never overwrite translation text unless the user applies a candidate.", "Keep TM, glossary, concordance, and SFX visible beside the selected block."),
    ),
    "downloader": ModeDashboardSpec(
        key="downloader",
        title="Raw Downloader",
        description="Source browser, search, chapter selection, queues, health checks, and import to project.",
        primary_action="Search sources",
        actions=(
            DashboardActionSpec("search", "Search manga/manhua", "Search configured source providers.", "Sources", True),
            DashboardActionSpec("queue", "Chapter queue", "Review downloads and output structure.", "Queue"),
            DashboardActionSpec("health", "Source health", "Check broken/experimental/disabled sources.", "Diagnostics", False, "warning"),
            DashboardActionSpec("import", "Import to project", "Create/open Pro project from downloaded raws.", "Project"),
        ),
        metrics=(
            DashboardMetricSpec("sources", "Sources", "Configured", "idle", "Registered providers."),
            DashboardMetricSpec("queue", "Queue", "0", "idle", "Pending chapters."),
            DashboardMetricSpec("rate", "Rate limits", "Safe", "success", "Downloader pacing status."),
        ),
        advanced_notes=("Experimental sources must be opt-in.", "Do not bypass DRM, paywalls, private APIs, or login restrictions."),
    ),
}


class MetricCard(QFrame):
    def __init__(self, spec: DashboardMetricSpec, parent=None):
        super().__init__(parent)
        self.spec = spec
        self.setObjectName("MetricCard")
        self.setStyleSheet(card_style())
        layout = QVBoxLayout(self)
        layout.setContentsMargins(TOKENS.spacing.md, TOKENS.spacing.md, TOKENS.spacing.md, TOKENS.spacing.md)
        layout.setSpacing(TOKENS.spacing.xs)
        row = QHBoxLayout()
        self.label = QLabel(spec.label, self)
        self.label.setStyleSheet(f"font-size: {TOKENS.typography.caption}px; color: {TOKENS.colors.text_muted};")
        row.addWidget(self.label, 1)
        self.status_label = QLabel(spec.status, self)
        row.addWidget(self.status_label)
        layout.addLayout(row)
        self.value_label = QLabel(spec.value, self)
        self.value_label.setStyleSheet(f"font-size: {TOKENS.typography.title}px; font-weight: 800; color: {TOKENS.colors.text};")
        layout.addWidget(self.value_label)
        self.description_label = QLabel("", self)
        self.description_label.setWordWrap(True)
        self.description_label.setStyleSheet(f"font-size: {TOKENS.typography.caption}px; color: {TOKENS.colors.text_subtle};")
        layout.addWidget(self.description_label)
        self.update_spec(spec)

    def update_spec(self, spec: DashboardMetricSpec):
        self.spec = spec
        self.label.setText(spec.label)
        self.status_label.setText(spec.status)
        self.status_label.setStyleSheet(badge_style(spec.status))
        self.value_label.setText(spec.value)
        self.description_label.setText(spec.description or "")
        self.description_label.setVisible(bool(spec.description))

    def update_value(self, value: str, status: Optional[str] = None, description: Optional[str] = None):
        self.update_spec(
            replace(
                self.spec,
                value=str(value),
                status=self.spec.status if status is None else str(status),
                description=self.spec.description if description is None else str(description),
            )
        )


class ActionCard(QFrame):
    activated = Signal(str)

    def __init__(self, spec: DashboardActionSpec, workflow: str, parent=None):
        super().__init__(parent)
        self.spec = spec
        self.setObjectName("ActionCard")
        accent = WORKFLOW_ACCENTS.get(workflow, TOKENS.colors.primary)
        self.setStyleSheet(card_style(accent if spec.primary else None))
        layout = QVBoxLayout(self)
        layout.setContentsMargins(TOKENS.spacing.md, TOKENS.spacing.md, TOKENS.spacing.md, TOKENS.spacing.md)
        layout.setSpacing(TOKENS.spacing.sm)
        row = QHBoxLayout()
        title = QLabel(spec.label, self)
        title.setStyleSheet(f"font-size: {TOKENS.typography.body_large}px; font-weight: 800; color: {TOKENS.colors.text};")
        row.addWidget(title, 1)
        badge = QLabel(spec.category, self)
        badge.setStyleSheet(badge_style(spec.status))
        row.addWidget(badge)
        layout.addLayout(row)
        desc = QLabel(spec.description, self)
        desc.setWordWrap(True)
        desc.setStyleSheet(f"font-size: {TOKENS.typography.caption}px; color: {TOKENS.colors.text_muted};")
        layout.addWidget(desc)
        btn = QPushButton(spec.label, self)
        btn.setStyleSheet(primary_button_style(workflow) if spec.primary else secondary_button_style())
        btn.clicked.connect(lambda: self.activated.emit(spec.key))
        layout.addWidget(btn)


class ModeDashboard(QWidget):
    """Dense, modern dashboard for each workflow mode.

    Modern does not mean simple: this page exposes professional actions, health
    metrics, and advanced notes in a polished layout instead of hiding them.
    """

    action_requested = Signal(str, str)

    def __init__(self, spec: ModeDashboardSpec, parent=None):
        super().__init__(parent)
        self.spec = spec
        self.metric_cards: Dict[str, MetricCard] = {}
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        root.addWidget(scroll)

        body = QWidget(scroll)
        scroll.setWidget(body)
        layout = QVBoxLayout(body)
        layout.setContentsMargins(TOKENS.spacing.xl, TOKENS.spacing.xl, TOKENS.spacing.xl, TOKENS.spacing.xl)
        layout.setSpacing(TOKENS.spacing.lg)

        hero = QFrame(body)
        hero.setObjectName("ModeDashboardHero")
        hero.setStyleSheet(hero_panel_style(self.spec.key))
        hero_layout = QVBoxLayout(hero)
        hero_layout.setContentsMargins(TOKENS.spacing.xl, TOKENS.spacing.xl, TOKENS.spacing.xl, TOKENS.spacing.xl)
        hero_layout.setSpacing(TOKENS.spacing.sm)
        title = QLabel(self.spec.title, hero)
        title.setStyleSheet(f"font-size: {TOKENS.typography.display}px; font-weight: 900; color: {TOKENS.colors.text};")
        hero_layout.addWidget(title)
        desc = QLabel(self.spec.description, hero)
        desc.setWordWrap(True)
        desc.setStyleSheet(f"font-size: {TOKENS.typography.body_large}px; color: {TOKENS.colors.text_muted};")
        hero_layout.addWidget(desc)
        primary = QPushButton(self.spec.primary_action, hero)
        primary.setStyleSheet(primary_button_style(self.spec.key))
        primary.clicked.connect(lambda: self.action_requested.emit(self.spec.key, "primary"))
        hero_layout.addWidget(primary)
        layout.addWidget(hero)

        metrics_grid = QGridLayout()
        metrics_grid.setHorizontalSpacing(TOKENS.spacing.md)
        metrics_grid.setVerticalSpacing(TOKENS.spacing.md)
        for idx, metric in enumerate(self.spec.metrics):
            card = MetricCard(metric, body)
            self.metric_cards[metric.key] = card
            metrics_grid.addWidget(card, 0, idx)
        layout.addLayout(metrics_grid)

        actions_title = QLabel(self.tr("Advanced actions"), body)
        actions_title.setStyleSheet(f"font-size: {TOKENS.typography.title}px; font-weight: 800; color: {TOKENS.colors.text};")
        layout.addWidget(actions_title)
        action_grid = QGridLayout()
        action_grid.setHorizontalSpacing(TOKENS.spacing.md)
        action_grid.setVerticalSpacing(TOKENS.spacing.md)
        for idx, action in enumerate(self.spec.actions):
            card = ActionCard(action, self.spec.key, body)
            card.activated.connect(lambda action_key, mode=self.spec.key: self.action_requested.emit(mode, action_key))
            row, col = divmod(idx, 2)
            action_grid.addWidget(card, row, col)
        layout.addLayout(action_grid)

        if self.spec.advanced_notes:
            notes = QFrame(body)
            notes.setObjectName("AdvancedNotes")
            notes.setStyleSheet(card_style(WORKFLOW_ACCENTS.get(self.spec.key)))
            notes_layout = QVBoxLayout(notes)
            notes_layout.setContentsMargins(TOKENS.spacing.lg, TOKENS.spacing.lg, TOKENS.spacing.lg, TOKENS.spacing.lg)
            notes_title = QLabel(self.tr("Professional workflow notes"), notes)
            notes_title.setStyleSheet(f"font-size: {TOKENS.typography.body_large}px; font-weight: 800;")
            notes_layout.addWidget(notes_title)
            for note in self.spec.advanced_notes:
                label = QLabel(f"- {note}", notes)
                label.setWordWrap(True)
                label.setStyleSheet(f"font-size: {TOKENS.typography.caption}px; color: {TOKENS.colors.text_muted};")
                notes_layout.addWidget(label)
            layout.addWidget(notes)
        layout.addStretch(1)

    def update_metric(self, key: str, value: str, status: Optional[str] = None, description: Optional[str] = None) -> bool:
        card = self.metric_cards.get(str(key or ""))
        if card is None:
            return False
        card.update_value(value, status=status, description=description)
        return True

    def update_metrics(self, metrics: Iterable[DashboardMetricSpec]):
        for metric in metrics:
            card = self.metric_cards.get(metric.key)
            if card is not None:
                card.update_spec(metric)

    def metric_snapshot(self) -> Dict[str, DashboardMetricSpec]:
        return {key: card.spec for key, card in self.metric_cards.items()}


def dashboard_for_mode(key: str, title: str, description: str) -> ModeDashboardSpec:
    return DEFAULT_MODE_DASHBOARDS.get(key) or ModeDashboardSpec(key=key, title=title, description=description, primary_action="Open")
