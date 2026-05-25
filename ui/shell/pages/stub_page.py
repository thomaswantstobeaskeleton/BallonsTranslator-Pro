"""
Generic stub page used as a placeholder for sections not yet implemented.
Shows section name and a "Coming soon" message.
"""

from __future__ import annotations
from typing import Optional

from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel
from qtpy.QtCore import Qt

from ..theme import COLORS, FONTS, SPACING
from ..nav_controller import SECTION_LABELS


class StubPage(QWidget):
    """Placeholder page for a section under construction."""

    def __init__(self, section_id: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._section_id = section_id
        label_text = SECTION_LABELS.get(section_id, section_id)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(SPACING.xxl, SPACING.xxl, SPACING.xxl, SPACING.xxl)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title = QLabel(label_text)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(f"""
            color: {COLORS.text_primary};
            font-size: {FONTS.size_h1}px;
            font-weight: 700;
            background: transparent;
        """)

        subtitle = QLabel("This section is under construction.")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet(f"""
            color: {COLORS.text_muted};
            font-size: {FONTS.size_lg}px;
            background: transparent;
        """)

        icon_label = QLabel("🚧")
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_label.setStyleSheet(f"font-size: 48px; background: transparent;")

        layout.addStretch(2)
        layout.addWidget(icon_label)
        layout.addSpacing(SPACING.lg)
        layout.addWidget(title)
        layout.addSpacing(SPACING.sm)
        layout.addWidget(subtitle)
        layout.addStretch(3)
