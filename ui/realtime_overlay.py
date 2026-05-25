from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np

from qtpy.QtCore import Qt, QPoint, QRect
from qtpy.QtGui import QPainter, QFont, QColor, QPen, QBrush, QFontMetrics
from qtpy.QtWidgets import QWidget

try:
    import win32gui
    import win32con
    _WIN32 = True
except Exception:
    _WIN32 = False


class TranslatedBlock:
    """One detected+translated region for the overlay to render."""
    def __init__(self, rect: Tuple[int, int, int, int], src_text: str = "", trans_text: str = ""):
        self.rect = rect
        self.src_text = src_text
        self.trans_text = trans_text


class RealtimeOverlayWidget(QWidget):
    """Always-on-top frameless overlay that renders translated text boxes over a screen region.

    - Click-through when `pinned=False` (mouse passes through to the app underneath).
    - Draggable/resizable when `pinned=True`.
    - Pure QPainter rendering for performance.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._blocks: List[TranslatedBlock] = []
        self._pinned = False
        self._dragging = False
        self._drag_offset = QPoint()
        self._font_size = 14
        self._bg_opacity = 180
        self._border_color = QColor(255, 255, 255, 80)
        self._text_color = QColor(255, 255, 255)
        self._bg_color = QColor(0, 0, 0, self._bg_opacity)

        flags = (
            Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        if _WIN32:
            self._enforce_topmost()

    def _enforce_topmost(self):
        """Extra HWND_TOPMOST enforcement on Windows."""
        try:
            hwnd = int(self.winId())
            win32gui.SetWindowPos(
                hwnd,
                win32con.HWND_TOPMOST,
                0, 0, 0, 0,
                win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_SHOWWINDOW
            )
        except Exception:
            pass

    def set_pinned(self, pinned: bool):
        """Toggle click-through vs interactive mode."""
        self._pinned = bool(pinned)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, not self._pinned)
        self.update()

    def set_blocks(self, blocks: List[TranslatedBlock]):
        self._blocks = list(blocks)
        self.update()

    def set_font_size(self, size: int):
        self._font_size = max(8, int(size))
        self.update()

    def show_at(self, x: int, y: int, w: int, h: int):
        self.setGeometry(x, y, w, h)
        self.show()
        self.raise_()
        self.activateWindow()
        if _WIN32:
            self._enforce_topmost()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        for blk in self._blocks:
            x, y, w, h = blk.rect
            # Clip to widget bounds
            if x >= self.width() or y >= self.height() or w <= 0 or h <= 0:
                continue
            x = max(0, x)
            y = max(0, y)
            w = min(w, self.width() - x)
            h = min(h, self.height() - y)

            # Background
            painter.setBrush(QBrush(self._bg_color))
            painter.setPen(QPen(self._border_color, 1))
            painter.drawRoundedRect(x, y, w, h, 6, 6)

            # Text
            if blk.trans_text:
                font = QFont("Microsoft YaHei", self._font_size)
                font.setBold(True)
                painter.setFont(font)
                painter.setPen(QPen(self._text_color))

                fm = QFontMetrics(font)
                padding = 6
                text_w = w - padding * 2
                text_h = h - padding * 2
                # Word-wrap into available width
                lines: List[str] = []
                current_line = ""
                for word in blk.trans_text.split():
                    test = (current_line + " " + word).strip() if current_line else word
                    if fm.horizontalAdvance(test) <= text_w:
                        current_line = test
                    else:
                        if current_line:
                            lines.append(current_line)
                        current_line = word
                if current_line:
                    lines.append(current_line)

                line_height = fm.height()
                total_text_h = len(lines) * line_height
                start_y = y + (h - total_text_h) // 2
                for i, line in enumerate(lines):
                    lx = x + (w - min(fm.horizontalAdvance(line), text_w)) // 2
                    ly = start_y + i * line_height
                    painter.drawText(lx, ly + fm.ascent(), line)

        # Drag handle when pinned
        if self._pinned:
            painter.setBrush(QBrush(QColor(255, 255, 255, 200)))
            painter.setPen(QPen(QColor(255, 255, 255, 200)))
            painter.drawEllipse(self.width() - 16, 4, 12, 12)

        painter.end()

    def mousePressEvent(self, event):
        if self._pinned and event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._drag_offset = event.pos()

    def mouseMoveEvent(self, event):
        if self._dragging:
            new_pos = self.mapToGlobal(event.pos() - self._drag_offset)
            self.move(new_pos)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False

    def hide(self):
        super().hide()

    def show(self):
        super().show()
        if _WIN32:
            self._enforce_topmost()
