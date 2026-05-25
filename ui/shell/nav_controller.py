"""
Navigation controller – bridges the sidebar (QML or QWidget) to the
QStackedWidget that holds each section page.

Exposes a QObject with properties/signals that QML can bind to, and
Python pages can connect to.
"""

from __future__ import annotations
from typing import List, Optional

from qtpy.QtCore import QObject, Signal, Property, Slot, Qt


# Section IDs — order matches the sidebar top-to-bottom
SECTIONS = [
    "home",
    "editor",
    "live_translate",
    "quick_image",
    "downloader",
    "batch_queue",
    "assist_qa",
    "models_ai",
    "settings",
    "diagnostics",
]

# Display labels matching the mockup
SECTION_LABELS = {
    "home": "Home",
    "editor": "Editor",
    "live_translate": "Live Translate",
    "quick_image": "Quick Image",
    "downloader": "Downloader",
    "batch_queue": "Batch Queue",
    "assist_qa": "Assist / QA",
    "models_ai": "Models / AI",
    "settings": "Settings",
    "diagnostics": "Diagnostics",
}

# SVG icon names (relative to icons/ dir)
SECTION_ICONS = {
    "home": "home",
    "editor": "edit",
    "live_translate": "realtime",
    "quick_image": "image",
    "downloader": "download",
    "batch_queue": "queue",
    "assist_qa": "assist",
    "models_ai": "models",
    "settings": "settings",
    "diagnostics": "diagnostics",
}


class NavController(QObject):
    """Central navigation state.

    The sidebar sets ``currentSection``; the shell listens to
    ``sectionChanged`` and swaps the stacked widget page.
    """

    sectionChanged = Signal(str)  # emitted with new section id

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._current: str = "home"

    # ── current section property ──────────────────────────────
    def _get_current(self) -> str:
        return self._current

    def _set_current(self, section_id: str) -> None:
        if section_id == self._current:
            return
        if section_id not in SECTIONS:
            return
        self._current = section_id
        self.sectionChanged.emit(section_id)

    currentSection = Property(str, _get_current, _set_current, notify=sectionChanged)

    # ── QML-callable slots ────────────────────────────────────
    @Slot(str)
    def navigate(self, section_id: str) -> None:
        """Navigate to a section by id."""
        self._set_current(section_id)

    @Slot(result=list)
    def sectionList(self) -> List[str]:
        """Return ordered list of section ids."""
        return list(SECTIONS)

    @Slot(str, result=str)
    def sectionLabel(self, section_id: str) -> str:
        return SECTION_LABELS.get(section_id, section_id)

    @Slot(str, result=str)
    def sectionIcon(self, section_id: str) -> str:
        return SECTION_ICONS.get(section_id, "home")
