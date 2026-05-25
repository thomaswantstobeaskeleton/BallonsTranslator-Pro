"""Top action bar matching the mockup's main chrome."""

from __future__ import annotations
from qtpy.QtWidgets import QWidget, QHBoxLayout, QPushButton, QLabel, QFrame
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QFont

from .theme import COLORS, FONTS, SPACING, RADIUS


class TopActionBar(QWidget):
    new_project_clicked = Signal()
    open_project_clicked = Signal()
    save_clicked = Signal()
    import_clicked = Signal()
    export_clicked = Signal()
    ocr_clicked = Signal()
    translate_clicked = Signal()
    inpaint_clicked = Signal()
    typeset_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("TopActionBar")
        self.setFixedHeight(44)
        self.setStyleSheet(f"""
            #TopActionBar {{
                background-color: {COLORS.bg_deepest};
                border-bottom: 1px solid {COLORS.border_subtle};
            }}
            #TopActionBar QPushButton {{
                background-color: transparent;
                color: {COLORS.text_secondary};
                border: none;
                border-radius: {RADIUS.sm}px;
                padding: 4px 12px;
                font-weight: 600;
                font-size: {FONTS.size_sm}px;
            }}
            #TopActionBar QPushButton:hover {{
                color: {COLORS.text_primary};
                background-color: {COLORS.bg_surface};
            }}
            #TopActionBar QPushButton[accent="true"] {{
                background-color: {COLORS.accent};
                color: {COLORS.text_inverse};
                font-weight: 700;
            }}
            #TopActionBar QPushButton[accent="true"]:hover {{
                background-color: {COLORS.accent_hover};
            }}
            #TopActionBar QPushButton[nav="true"] {{
                color: {COLORS.accent};
            }}
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(SPACING.lg, 0, SPACING.lg, 0)
        layout.setSpacing(0)

        self._add_button(layout, "New Project", self.new_project_clicked)
        self._add_button(layout, "Open", self.open_project_clicked)
        self._add_button(layout, "Save", self.save_clicked)
        self._add_button(layout, "Import", self.import_clicked)
        self._add_button(layout, "Export", self.export_clicked)

        sep = QFrame()
        sep.setFixedWidth(1)
        sep.setFixedHeight(20)
        sep.setStyleSheet(f"background-color: {COLORS.border_subtle};")
        layout.addSpacing(SPACING.md)
        layout.addWidget(sep)
        layout.addSpacing(SPACING.md)

        self._add_button(layout, "OCR", self.ocr_clicked, accent=False)
        self._add_button(layout, "Translate", self.translate_clicked, accent=False)
        self._add_button(layout, "Inpaint", self.inpaint_clicked, accent=False)
        self._add_button(layout, "Typeset", self.typeset_clicked, accent=False)

        layout.addStretch()

        # Mode selector on the right
        self.mode_btn = QPushButton("Editor")
        self.mode_btn.setProperty("nav", True)
        layout.addWidget(self.mode_btn)

    def _add_button(self, layout, text, signal, accent=False):
        btn = QPushButton(text)
        if accent:
            btn.setProperty("accent", True)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.clicked.connect(signal.emit)
        layout.addWidget(btn)
        return btn
