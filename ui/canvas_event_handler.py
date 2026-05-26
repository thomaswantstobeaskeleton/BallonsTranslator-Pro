"""
Canvas Event Handler — composition-based input handling for Canvas.

Extracts monolithic mouse/keyboard event logic from ui.canvas.Canvas
into a delegating handler, following Comic Translate's composition pattern.

The Canvas keeps its event overrides but delegates core logic here,
so custom behaviors (e.g. webtoon pan-only mode) can swap handlers
without subclassing the entire scene.
"""

from typing import Optional, Callable

from qtpy.QtCore import Qt, QPointF, QSizeF, QRectF
from qtpy.QtWidgets import QGraphicsSceneMouseEvent

from .image_edit import ImageEditMode

# Minimum drag distance before a right-click in textblock_mode is considered a drag
MIN_DRAG_SIZE = 6
DEFAULT_TEXTBOX_WIDTH = 150
DEFAULT_TEXTBOX_HEIGHT = 60


class CanvasEventHandler:
    """
    Handles mouse input logic for Canvas.

    Usage (inside Canvas):
        self._handler = CanvasEventHandler(self)
        # In Canvas.mousePressEvent:
        if self._handler.on_mouse_press(event):
            return
        super().mousePressEvent(event)
    """

    def __init__(self, canvas) -> None:
        self.canvas = canvas
        self._right_click_origin: Optional[QPointF] = None

    # --- Predicates (mirror Canvas state) ---

    def _img_valid(self) -> bool:
        proj = getattr(self.canvas, "imgtrans_proj", None)
        return proj is not None and getattr(proj, "img_valid", False)

    def _textblock_mode(self) -> bool:
        return getattr(self.canvas, "textblock_mode", False)

    def _textEditMode(self) -> bool:
        # Canvas has a method textEditMode() — call it if present
        return bool(getattr(self.canvas, "textEditMode", lambda: False)())

    def _creating_normal_rect(self) -> bool:
        iem = getattr(self.canvas, "image_edit_mode", ImageEditMode.NONE)
        idx = getattr(self.canvas, "editor_index", 0)
        return iem == ImageEditMode.RectTool and idx == 0

    def _painting(self) -> bool:
        return getattr(self.canvas, "painting", False)

    def _scale_tool_mode(self) -> bool:
        return getattr(self.canvas, "scale_tool_mode", False)

    # --- Event responders ---

    def on_mouse_press(self, event: QGraphicsSceneMouseEvent) -> bool:
        """
        Handle mouse press logic. Returns True if the event was fully consumed
        (caller should return without calling super).
        """
        btn = event.button()
        scene_pos = event.scenePos()

        # Middle button: start pan (Canvas stores state, we just report)
        if btn == Qt.MouseButton.MiddleButton:
            self.canvas.mid_btn_pressed = True
            self.canvas.pan_initial_pos = event.screenPos()
            return True

        if btn != Qt.MouseButton.RightButton:
            self._right_click_origin = None

        if not self._img_valid():
            return False

        # Text block creation mode (right-click drag)
        if self._textblock_mode() and len(self.canvas.selectedItems()) == 0 and self._textEditMode():
            if btn == Qt.MouseButton.RightButton:
                self._right_click_origin = scene_pos
                event.accept()
                return True
            return False

        # Rectangle creation mode (left or right click)
        if self._creating_normal_rect():
            if btn in (Qt.MouseButton.RightButton, Qt.MouseButton.LeftButton):
                self.canvas.startCreateTextblock(scene_pos, hide_control=True)
                event.accept()
                return True
            return False

        # Left button: drawing or scale tool
        if btn == Qt.MouseButton.LeftButton:
            if self._scale_tool_mode():
                self.canvas.begin_scale_tool.emit(scene_pos)
                return True
            if self._painting() and getattr(self.canvas, "editor_index", 0) == 0:
                text_eraser = getattr(self.canvas, "image_edit_mode", ImageEditMode.NONE) == ImageEditMode.TextEraserTool
                pen = getattr(self.canvas, "painting_pen", None)
                if pen is not None:
                    layer = getattr(self.canvas, "inpaintLayer", None)
                    if layer is not None:
                        self.canvas.addStrokeImageItem(
                            layer.mapFromScene(scene_pos), pen, text_eraser=text_eraser
                        )
                return True

        # Right button: erasing or rubber-band selection
        if btn == Qt.MouseButton.RightButton:
            if self._painting() and getattr(self.canvas, "editor_index", 0) == 0:
                iem = getattr(self.canvas, "image_edit_mode", ImageEditMode.NONE)
                if iem != ImageEditMode.TextEraserTool:
                    erasing = iem == ImageEditMode.PenTool
                    pen = getattr(self.canvas, "erasing_pen", None)
                    layer = getattr(self.canvas, "inpaintLayer", None)
                    if pen is not None and layer is not None:
                        self.canvas.addStrokeImageItem(
                            layer.mapFromScene(scene_pos), pen, erasing
                        )
                    return True
            else:
                # Start rubber-band selection
                self.canvas.rubber_band_origin = scene_pos
                rb = getattr(self.canvas, "rubber_band", None)
                if rb is not None:
                    rb.setGeometry(QRectF(scene_pos, scene_pos).normalized())
                    rb.show()
                    rb.setZValue(1)
                event.accept()
                return True

        return False

    def on_mouse_move(self, event: QGraphicsSceneMouseEvent) -> bool:
        """
        Handle mouse move logic. Returns True if fully consumed.
        """
        scene_pos = event.scenePos()

        # Middle-button panning
        if getattr(self.canvas, "mid_btn_pressed", False):
            new_pos = event.screenPos()
            init_pos = getattr(self.canvas, "pan_initial_pos", None)
            if init_pos is not None:
                delta = new_pos - init_pos
                h = getattr(self.canvas.gv, "horizontalScrollBar", lambda: None)()
                v = getattr(self.canvas.gv, "verticalScrollBar", lambda: None)()
                if h is not None:
                    h.setValue(h.value() - delta.x())
                if v is not None:
                    v.setValue(v.value() - delta.y())
                self.canvas.pan_initial_pos = new_pos
            return True

        # Right-click drag in textblock_mode: start create if dragged enough
        origin = self._right_click_origin
        if origin is not None:
            dx = abs(scene_pos.x() - origin.x())
            dy = abs(scene_pos.y() - origin.y())
            if dx > MIN_DRAG_SIZE or dy > MIN_DRAG_SIZE:
                self.canvas.startCreateTextblock(origin, hide_control=True)
                self._right_click_origin = None
            return True

        # Creating text block: update shape control
        if getattr(self.canvas, "creating_textblock", False):
            ctrl = getattr(self.canvas, "txtblkShapeControl", None)
            if ctrl is not None:
                origin = getattr(self.canvas, "create_block_origin", None)
                if origin is not None:
                    ctrl.setRect(QRectF(origin, scene_pos).normalized())
            return True

        # Creating rect tool: update rubber band
        if self._creating_normal_rect():
            rb = getattr(self.canvas, "rubber_band", None)
            origin = getattr(self.canvas, "rubber_band_origin", None)
            if rb is not None and origin is not None:
                rb.setGeometry(QRectF(origin, scene_pos).normalized())
            return True

        # Drawing: update stroke position
        if self._painting() and getattr(self.canvas, "stroke_img_item", None) is not None:
            layer = getattr(self.canvas, "inpaintLayer", None)
            if layer is not None:
                self.canvas.stroke_img_item.setPos(layer.mapFromScene(scene_pos))
            return True

        return False

    def on_mouse_release(self, event: QGraphicsSceneMouseEvent) -> bool:
        """
        Handle mouse release logic. Returns True if fully consumed.
        """
        btn = event.button()
        scene_pos = event.scenePos()

        # Rubber-band finish + region detect
        if btn == Qt.MouseButton.RightButton:
            rb = getattr(self.canvas, "rubber_band", None)
            origin = getattr(self.canvas, "rubber_band_origin", None)
            if rb is not None and rb.isVisible() and origin is not None:
                r = rb.geometry().normalized()
                if r.width() >= 10 and r.height() >= 10:
                    self.canvas._last_rubber_band_rect = QRectF(r)
            self.canvas.hide_rubber_band()

            # Right-click without drag in textblock_mode → context menu
            if self._right_click_origin is not None:
                self.canvas._context_menu_scene_pos = scene_pos
                self.canvas.context_menu_requested.emit(event.screenPos(), False)
                self._right_click_origin = None
                return True

        # Middle button release
        if btn == Qt.MouseButton.MiddleButton:
            self.canvas.mid_btn_pressed = False
            return True

        # Finish text block creation
        textblk_created = False
        if getattr(self.canvas, "creating_textblock", False):
            tgt = 0 if btn == Qt.MouseButton.LeftButton else 1
            ctrl = getattr(self.canvas, "txtblkShapeControl", None)
            if ctrl is not None:
                rect = ctrl.rect()
                if btn == Qt.MouseButton.RightButton and rect.width() < MIN_DRAG_SIZE and rect.height() < MIN_DRAG_SIZE:
                    default_rect = QRectF(
                        getattr(self.canvas, "create_block_origin", QPointF(0, 0)),
                        QSizeF(DEFAULT_TEXTBOX_WIDTH, DEFAULT_TEXTBOX_HEIGHT)
                    )
                    self.canvas.end_create_textblock.emit(default_rect)
                    self.canvas.endCreateTextblock(btn=tgt)
                    textblk_created = True
                else:
                    textblk_created = self.canvas.endCreateTextblock(btn=tgt)

        # Right button: finish erasing or context menu
        if btn == Qt.MouseButton.RightButton:
            if getattr(self.canvas, "stroke_img_item", None) is not None:
                self.canvas.finish_erasing.emit(self.canvas.stroke_img_item)
            if self._textEditMode() and not textblk_created:
                self.canvas._context_menu_scene_pos = scene_pos
                self.canvas.context_menu_requested.emit(event.screenPos(), False)
            return True

        # Left button: finish painting or scale tool
        if btn == Qt.MouseButton.LeftButton:
            if getattr(self.canvas, "stroke_img_item", None) is not None:
                self.canvas.finish_painting.emit(self.canvas.stroke_img_item)
                return True
            if self._scale_tool_mode():
                self.canvas.end_scale_tool.emit()
                return True

        return False

    # --- Keyboard helpers ---

    def on_key_press(self, event) -> bool:
        """
        Handle key press logic. Returns True if consumed.
        (Called from Canvas.keyPressEvent before per-item handling.)
        """
        key = event.key()
        modifiers = event.modifiers()

        # Ctrl+A: select all text blocks
        if key == Qt.Key.Key_A and modifiers == Qt.KeyboardModifier.ControlModifier:
            self.canvas.select_all_signal.emit()
            return True

        # Delete / Backspace: delete selected blocks
        if key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.canvas.delete_textblks.emit(0)
            return True

        # Ctrl+C / Ctrl+V for text blocks
        if modifiers == Qt.KeyboardModifier.ControlModifier:
            if key == Qt.Key.Key_C:
                self.canvas.copy_textblks.emit()
                return True
            if key == Qt.Key.Key_V:
                paste_pos = getattr(self.canvas, "_context_menu_scene_pos", QPointF(0, 0))
                self.canvas.paste_textblks.emit(paste_pos)
                return True

        return False
