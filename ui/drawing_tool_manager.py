"""
Drawing Tool Manager — composition-based drawing tool state machine.

Replaces monolithic tool-switching in Canvas with composable tool objects
that each handle activation, cursor, mouse events, and deactivation.

Inspired by Comic Translate's tool abstraction and modern state-machine patterns.

Usage (inside Canvas):
    self._tool_manager = DrawingToolManager(self)
    self._tool_manager.set_tool(InpaintTool(self))
    # In mousePressEvent:
    if self._tool_manager.active_tool:
        handled = self._tool_manager.active_tool.on_mouse_press(event)
        if handled: return
"""

from typing import Optional, Dict, List
from qtpy.QtCore import Qt
from qtpy.QtGui import QCursor, QPen
from qtpy.QtWidgets import QGraphicsSceneMouseEvent, QGraphicsView

from .image_edit import ImageEditMode


class BaseDrawingTool:
    """Abstract base for a drawing tool."""

    tool_id: int = ImageEditMode.NONE
    tool_name: str = "none"
    cursor: QCursor = QCursor(Qt.CursorShape.ArrowCursor)

    def __init__(self, canvas) -> None:
        self.canvas = canvas

    def on_activate(self) -> None:
        """Called when this tool becomes active."""
        pass

    def on_deactivate(self) -> None:
        """Called when this tool is switched away from."""
        pass

    def on_mouse_press(self, event: QGraphicsSceneMouseEvent) -> bool:
        """Return True if the event was consumed."""
        return False

    def on_mouse_move(self, event: QGraphicsSceneMouseEvent) -> bool:
        """Return True if the event was consumed."""
        return False

    def on_mouse_release(self, event: QGraphicsSceneMouseEvent) -> bool:
        """Return True if the event was consumed."""
        return False

    def on_key_press(self, event) -> bool:
        """Return True if the event was consumed."""
        return False


class PenTool(BaseDrawingTool):
    """Pen / inpainting brush tool."""

    tool_id = ImageEditMode.PenTool
    tool_name = "pen"
    cursor = QCursor(Qt.CursorShape.CrossCursor)

    def on_mouse_press(self, event: QGraphicsSceneMouseEvent) -> bool:
        if event.button() == Qt.MouseButton.LeftButton:
            self.canvas.addStrokeImageItem(
                self.canvas.inpaintLayer.mapFromScene(event.scenePos()),
                self.canvas.painting_pen,
                text_eraser=False,
            )
            return True
        if event.button() == Qt.MouseButton.RightButton:
            erasing = self.canvas.image_edit_mode == ImageEditMode.PenTool
            self.canvas.addStrokeImageItem(
                self.canvas.inpaintLayer.mapFromScene(event.scenePos()),
                self.canvas.erasing_pen,
                erasing=erasing,
            )
            return True
        return False

    def on_mouse_move(self, event: QGraphicsSceneMouseEvent) -> bool:
        if self.canvas.stroke_img_item is not None and self.canvas.stroke_img_item.is_painting:
            pos = self.canvas.inpaintLayer.mapFromScene(event.scenePos())
            if self.canvas.erase_img_key is None:
                self.canvas.stroke_img_item.lineTo(pos)
            else:
                rect = self.canvas.stroke_img_item.lineTo(pos, update=False)
                if rect is not None:
                    self.canvas.drawingLayer.update(rect)
            return True
        return False

    def on_mouse_release(self, event: QGraphicsSceneMouseEvent) -> bool:
        if event.button() == Qt.MouseButton.LeftButton and self.canvas.stroke_img_item is not None:
            self.canvas.finish_painting.emit(self.canvas.stroke_img_item)
            return True
        if event.button() == Qt.MouseButton.RightButton and self.canvas.stroke_img_item is not None:
            self.canvas.finish_erasing.emit(self.canvas.stroke_img_item)
            return True
        return False


class InpaintTool(BaseDrawingTool):
    """Inpainting brush tool (same as pen but semantically distinct)."""

    tool_id = ImageEditMode.InpaintTool
    tool_name = "inpaint"
    cursor = QCursor(Qt.CursorShape.CrossCursor)

    def on_mouse_press(self, event: QGraphicsSceneMouseEvent) -> bool:
        if event.button() == Qt.MouseButton.LeftButton:
            self.canvas.addStrokeImageItem(
                self.canvas.inpaintLayer.mapFromScene(event.scenePos()),
                self.canvas.painting_pen,
                text_eraser=False,
            )
            return True
        return False

    def on_mouse_move(self, event: QGraphicsSceneMouseEvent) -> bool:
        if self.canvas.stroke_img_item is not None and self.canvas.stroke_img_item.is_painting:
            pos = self.canvas.inpaintLayer.mapFromScene(event.scenePos())
            self.canvas.stroke_img_item.lineTo(pos)
            return True
        return False

    def on_mouse_release(self, event: QGraphicsSceneMouseEvent) -> bool:
        if event.button() == Qt.MouseButton.LeftButton and self.canvas.stroke_img_item is not None:
            self.canvas.finish_painting.emit(self.canvas.stroke_img_item)
            return True
        return False


class TextEraserTool(BaseDrawingTool):
    """Text eraser tool: erases parts of text blocks (mask-based)."""

    tool_id = ImageEditMode.TextEraserTool
    tool_name = "text_eraser"
    cursor = QCursor(Qt.CursorShape.CrossCursor)

    def on_activate(self) -> None:
        self.canvas._text_eraser_selected_blocks = self.canvas.selected_text_items()

    def on_mouse_press(self, event: QGraphicsSceneMouseEvent) -> bool:
        if event.button() == Qt.MouseButton.LeftButton:
            self.canvas.addStrokeImageItem(
                self.canvas.inpaintLayer.mapFromScene(event.scenePos()),
                self.canvas.painting_pen,
                text_eraser=True,
            )
            return True
        return False

    def on_mouse_move(self, event: QGraphicsSceneMouseEvent) -> bool:
        if self.canvas.stroke_img_item is not None and self.canvas.stroke_img_item.is_painting:
            pos = self.canvas.inpaintLayer.mapFromScene(event.scenePos())
            self.canvas.stroke_img_item.lineTo(pos)
            return True
        return False

    def on_mouse_release(self, event: QGraphicsSceneMouseEvent) -> bool:
        if event.button() == Qt.MouseButton.LeftButton and self.canvas.stroke_img_item is not None:
            self.canvas.finish_painting.emit(self.canvas.stroke_img_item)
            return True
        return False


class RectTool(BaseDrawingTool):
    """Rectangle creation tool (for manual block drawing)."""

    tool_id = ImageEditMode.RectTool
    tool_name = "rect"
    cursor = QCursor(Qt.CursorShape.CrossCursor)

    def on_mouse_press(self, event: QGraphicsSceneMouseEvent) -> bool:
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            self.canvas.startCreateTextblock(event.scenePos(), hide_control=True)
            event.accept()
            return True
        return False

    def on_mouse_move(self, event: QGraphicsSceneMouseEvent) -> bool:
        if self.canvas.txtblkShapeControl.isVisible() and self.canvas.txtblkShapeControl.blk_item is None:
            self.canvas.txtblkShapeControl.setRect(
                __import__('qtpy.QtCore', fromlist=['QRectF']).QRectF(
                    self.canvas.create_block_origin, event.scenePos()
                ).normalized()
            )
            return True
        return False

    def on_mouse_release(self, event: QGraphicsSceneMouseEvent) -> bool:
        # Handled by canvas endCreateTextblock logic
        return False


class HandTool(BaseDrawingTool):
    """Hand / pan tool (default)."""

    tool_id = ImageEditMode.HandTool
    tool_name = "hand"
    cursor = QCursor(Qt.CursorShape.OpenHandCursor)

    def on_activate(self) -> None:
        gv = getattr(self.canvas, "gv", None)
        if gv is not None:
            gv.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

    def on_deactivate(self) -> None:
        pass


class ScaleTool(BaseDrawingTool):
    """Scale tool (Alt + drag). Activated dynamically by keyboard modifier."""

    tool_id = -1  # Dynamic, not set via image_edit_mode
    tool_name = "scale"
    cursor = QCursor(Qt.CursorShape.SizeAllCursor)

    def on_mouse_press(self, event: QGraphicsSceneMouseEvent) -> bool:
        if event.button() == Qt.MouseButton.LeftButton:
            self.canvas.begin_scale_tool.emit(event.scenePos())
            return True
        return False

    def on_mouse_move(self, event: QGraphicsSceneMouseEvent) -> bool:
        self.canvas.scale_tool.emit(event.scenePos())
        return True

    def on_mouse_release(self, event: QGraphicsSceneMouseEvent) -> bool:
        if event.button() == Qt.MouseButton.LeftButton:
            self.canvas.end_scale_tool.emit()
            return True
        return False


class DrawingToolManager:
    """
    Manages drawing tool registration, activation, and event delegation.

    Usage:
        mgr = DrawingToolManager(canvas)
        mgr.register_tool(PenTool(canvas))
        mgr.register_tool(InpaintTool(canvas))
        mgr.set_tool(ImageEditMode.PenTool)
    """

    def __init__(self, canvas) -> None:
        self.canvas = canvas
        self._tools: Dict[int, BaseDrawingTool] = {}
        self._active_tool: Optional[BaseDrawingTool] = None
        self._previous_mode: int = ImageEditMode.NONE

        # Register built-in tools
        self.register_tool(HandTool(canvas))
        self.register_tool(PenTool(canvas))
        self.register_tool(InpaintTool(canvas))
        self.register_tool(TextEraserTool(canvas))
        self.register_tool(RectTool(canvas))
        self.register_tool(ScaleTool(canvas))

    def register_tool(self, tool: BaseDrawingTool) -> None:
        self._tools[tool.tool_id] = tool

    @property
    def active_tool(self) -> Optional[BaseDrawingTool]:
        return self._active_tool

    def set_tool(self, tool_id: int) -> None:
        """Activate a tool by its ImageEditMode ID."""
        if self._active_tool is not None:
            self._active_tool.on_deactivate()
        self._previous_mode = getattr(self.canvas, "image_edit_mode", ImageEditMode.NONE)
        self.canvas.image_edit_mode = tool_id
        self._active_tool = self._tools.get(tool_id)
        if self._active_tool is not None:
            self._active_tool.on_activate()
            gv = getattr(self.canvas, "gv", None)
            if gv is not None:
                gv.setCursor(self._active_tool.cursor)

    def restore_previous_tool(self) -> None:
        """Restore the previous active tool (e.g. after a temporary mode)."""
        self.set_tool(self._previous_mode)

    def reset_to_hand(self) -> None:
        """Reset to hand/pan tool."""
        self.set_tool(ImageEditMode.HandTool)

    # --- Event delegation ---

    def on_mouse_press(self, event: QGraphicsSceneMouseEvent) -> bool:
        # Scale tool is a dynamic override (Alt modifier)
        if self.canvas.drawMode() and getattr(self.canvas, "gv", None) and \
           __import__('qtpy.QtWidgets', fromlist=['QApplication']).QApplication.keyboardModifiers() == Qt.KeyboardModifier.AltModifier:
            scale_tool = self._tools.get(ScaleTool.tool_id)
            if scale_tool is not None:
                return scale_tool.on_mouse_press(event)

        if self._active_tool is not None:
            return self._active_tool.on_mouse_press(event)
        return False

    def on_mouse_move(self, event: QGraphicsSceneMouseEvent) -> bool:
        if self._active_tool is not None:
            return self._active_tool.on_mouse_move(event)
        return False

    def on_mouse_release(self, event: QGraphicsSceneMouseEvent) -> bool:
        if self._active_tool is not None:
            return self._active_tool.on_mouse_release(event)
        return False

    def on_key_press(self, event) -> bool:
        if self._active_tool is not None:
            return self._active_tool.on_key_press(event)
        return False
