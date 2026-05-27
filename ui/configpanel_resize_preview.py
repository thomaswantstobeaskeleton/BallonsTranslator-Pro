"""
Settings Resize Preview — live preview widget for font/size settings changes.

Shows a miniature canvas-like preview that updates in real time when the user
changes text rendering settings (font family, size, line spacing, etc.) in
the config panel.

Usage (inside ConfigPanel):
    preview = ConfigResizePreview(self)
    preview.update_preview(fontformat=pcfg.global_fontformat)
"""

from typing import Optional

from qtpy.QtCore import Qt, QSize
from qtpy.QtGui import QFont, QColor, QPixmap, QPainter, QTextDocument, QTextOption
from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout, QSizePolicy

from utils.fontformat import FontFormat
from utils.config import pcfg


SAMPLE_TEXT = (
    "The quick brown fox jumps over the lazy dog.\n"
    "快速棕色狐狸跳過懶惰的狗。\n"
    "あいうえおかきくけこさしすせそ"
)


class ConfigResizePreview(QWidget):
    """
    A small live-preview widget that renders sample text with the currently
    selected font format. Attach to ConfigPanel signals to update automatically.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._font_format: Optional[FontFormat] = None
        self._preview_size = QSize(320, 140)
        self._build_ui()
        self.update_preview()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(6)

        header = QLabel(self.tr("Preview"))
        header.setStyleSheet("font-weight: bold; font-size: 11px; color: #888;")
        layout.addWidget(header)

        self._canvas = QLabel()
        self._canvas.setMinimumSize(self._preview_size)
        self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._canvas.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self._canvas.setFrameShape(QLabel.Shape.StyledPanel)
        layout.addWidget(self._canvas)

    def set_font_format(self, fmt: FontFormat) -> None:
        """Update the preview font format and re-render."""
        self._font_format = fmt
        self.update_preview()

    def update_preview(self, fontformat: Optional[FontFormat] = None) -> None:
        """Render the sample text with the given (or current) font format."""
        if fontformat is not None:
            self._font_format = fontformat
        if self._font_format is None:
            self._font_format = pcfg.global_fontformat

        fmt = self._font_format
        pixmap = QPixmap(self._preview_size)
        pixmap.fill(Qt.GlobalColor.white)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        # Build font from FontFormat
        font = QFont(fmt.font_family or "Arial", max(1, int(fmt.size or 14)))
        font.setBold(fmt.bold)
        font.setItalic(fmt.italic)
        font.setUnderline(fmt.underline)
        font.setStrikeOut(fmt.strike_out)
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, fmt.letter_spacing or 0)
        painter.setFont(font)

        # Set text color
        color = QColor(fmt.frgb[0], fmt.frgb[1], fmt.frgb[2], fmt.frgb[3]) if hasattr(fmt, 'frgb') and fmt.frgb else QColor(0, 0, 0)
        painter.setPen(color)

        # Draw sample text with line spacing
        line_spacing = fmt.line_spacing if hasattr(fmt, 'line_spacing') and fmt.line_spacing else 1.2
        fm = painter.fontMetrics()
        line_height = int(fm.height() * line_spacing)
        x, y = 8, 8 + fm.ascent()
        for line in SAMPLE_TEXT.splitlines():
            painter.drawText(x, y, line)
            y += line_height
            if y > self._preview_size.height() - 8:
                break

        painter.end()
        self._canvas.setPixmap(pixmap)
