"""
Canvas Interaction Manager

Handles selection, rotation, and resizing of text and rectangle items.
Extracted from the monolithic Canvas/CanvasView event handling to follow
Comic Translate's composition pattern.

Key features:
  - Rotation-aware cursor mapping (8-directional resize cursors adjusted for item rotation)
  - Resize ring and rotation ring detection
  - clear_items_in_viewport() for webtoon mode optimization
  - Selection helpers (multi-select, select all, clear)
"""

import math
from typing import List, Optional, Tuple

from qtpy.QtCore import Qt, QPointF, QRectF
from qtpy.QtGui import QCursor
from qtpy.QtWidgets import QGraphicsItem, QGraphicsView, QGraphicsScene

from .textitem import TextBlkItem
from .cursor import rotateCursorList, resizeCursorList


# Resize handle indices: top-left, top, top-right, right, bottom-right, bottom, bottom-left, left
_RESIZE_HANDLES = 8

# Angle thresholds for cursor mapping (each sector is 45 degrees, centered on the handles)
_SECTOR_ANGLE = 360.0 / _RESIZE_HANDLES


def _normalize_angle(angle_deg: float) -> float:
    """Normalize angle to [0, 360)."""
    while angle_deg < 0:
        angle_deg += 360
    while angle_deg >= 360:
        angle_deg -= 360
    return angle_deg


def _get_rotated_cursor_index(base_idx: int, item_rotation: float) -> int:
    """
    Map a base resize cursor index to the correct one after item rotation.
    Comic Translate does this with 8 discrete steps based on rotation.
    """
    angle = _normalize_angle(item_rotation + _SECTOR_ANGLE / 2)
    steps = int(angle / _SECTOR_ANGLE) % _RESIZE_HANDLES
    return (base_idx + steps) % _RESIZE_HANDLES


def get_resize_cursor(angle_deg: float, handle_idx: int) -> QCursor:
    """
    Return the correct resize cursor for a given handle, accounting for item rotation.

    Args:
        angle_deg: Item rotation in degrees.
        handle_idx: Base handle index (0=top-left, 1=top, ... 7=left).

    Returns:
        QCursor with the appropriate resize shape.
    """
    rotated_idx = _get_rotated_cursor_index(handle_idx, angle_deg)
    return resizeCursorList[rotated_idx]


def get_rotation_cursor() -> QCursor:
    """Return the cursor used when hovering over the rotation ring."""
    # Use the Qt built-in open-hand cursor as a rotation indicator
    return QCursor(Qt.CursorShape.OpenHandCursor)


def _in_rotate_ring(
    pos: QPointF,
    item_rect: QRectF,
    ring_width: float = 20.0,
) -> bool:
    """
    Check if *pos* is inside the rotation ring around *item_rect*.

    The ring is a band just outside the bounding rect.
    """
    outer = item_rect.adjusted(-ring_width, -ring_width, ring_width, ring_width)
    return outer.contains(pos) and not item_rect.contains(pos)


def _in_resize_area(
    pos: QPointF,
    item_rect: QRectF,
    margin: float = 10.0,
) -> bool:
    """Check if *pos* is on the edge of *item_rect* (resize grab area)."""
    # Must be inside the margin-expanded rect
    expanded = item_rect.adjusted(-margin, -margin, margin, margin)
    if not expanded.contains(pos):
        return False
    # But outside the margin-shrunken rect (i.e., on the border)
    shrunken = item_rect.adjusted(margin, margin, -margin, -margin)
    # If shrunken is invalid (item too small), treat entire expanded area as resize zone
    if shrunken.width() <= 0 or shrunken.height() <= 0:
        return True
    return not shrunken.contains(pos)


def get_handle_at_position(
    pos: QPointF,
    item_rect: QRectF,
    rotation: float = 0.0,
    handle_size: float = 12.0,
) -> int:
    """
    Determine which resize handle (0-7) is under *pos*.

    Returns:
        Handle index (0=top-left, clockwise), or -1 if none.
    """
    center = item_rect.center()
    dx = pos.x() - center.x()
    dy = pos.y() - center.y()

    # Rotate point backwards by item rotation to get normalized coordinates
    rad = math.radians(-rotation)
    cos_r = math.cos(rad)
    sin_r = math.sin(rad)
    rx = dx * cos_r - dy * sin_r
    ry = dx * sin_r + dy * cos_r

    half_w = item_rect.width() / 2
    half_h = item_rect.height() / 2

    # Determine which quadrant/edge
    # Handle zones: corners are handle_size×handle_size squares at each corner
    # edges are strips between corners

    near_left = abs(rx + half_w) < handle_size
    near_right = abs(rx - half_w) < handle_size
    near_top = abs(ry + half_h) < handle_size
    near_bottom = abs(ry - half_h) < handle_size

    if near_top and near_left:
        return 0
    if near_top and near_right:
        return 2
    if near_bottom and near_right:
        return 4
    if near_bottom and near_left:
        return 6
    if near_top:
        return 1
    if near_right:
        return 3
    if near_bottom:
        return 5
    if near_left:
        return 7

    return -1


class InteractionManager:
    """
    Manages canvas interactions: selection, resize cursor mapping, rotation zones.

    This class does NOT own the items; it queries the scene/view and returns
    decisions that the caller (CanvasEventHandler or CanvasView) acts on.
    """

    def __init__(self, view: QGraphicsView) -> None:
        self.view = view
        self.scene: Optional[QGraphicsScene] = view.scene()
        self._rotation_ring_width = 20.0
        self._resize_margin = 10.0
        self._handle_size = 12.0

    # --- Selection helpers ---

    def selected_text_items(self, sort: bool = True) -> List[TextBlkItem]:
        """Return currently selected TextBlkItem instances."""
        if self.scene is None:
            return []
        items = [item for item in self.scene.selectedItems() if isinstance(item, TextBlkItem)]
        if sort:
            items.sort(key=lambda it: it.idx)
        return items

    def select_item(self, item: QGraphicsItem, extend: bool = False) -> None:
        """Select *item*, optionally extending the current selection (Ctrl-style)."""
        if not extend:
            if self.scene is not None:
                self.scene.clearSelection()
        item.setSelected(True)

    def deselect_all(self) -> None:
        """Clear the scene selection."""
        if self.scene is not None:
            self.scene.clearSelection()

    def select_all_text_items(self) -> None:
        """Select every TextBlkItem in the scene."""
        if self.scene is None:
            return
        for item in self.scene.items():
            if isinstance(item, TextBlkItem):
                item.setSelected(True)

    # --- Hit testing ---

    def hit_test_item(self, scene_pos: QPointF) -> Optional[QGraphicsItem]:
        """Return the topmost selectable item at *scene_pos*, or None."""
        if self.scene is None:
            return None
        for item in self.scene.items(scene_pos):
            if item.flags() & QGraphicsItem.GraphicsItemFlag.ItemIsSelectable:
                return item
        return None

    def is_on_rotate_ring(self, item: QGraphicsItem, scene_pos: QPointF) -> bool:
        """Check if *scene_pos* is on the rotation ring of *item*."""
        rect = item.boundingRect()
        local_pos = item.mapFromScene(scene_pos)
        return _in_rotate_ring(local_pos, rect, self._rotation_ring_width)

    def is_on_resize_area(self, item: QGraphicsItem, scene_pos: QPointF) -> bool:
        """Check if *scene_pos* is on a resize edge of *item*."""
        rect = item.boundingRect()
        local_pos = item.mapFromScene(scene_pos)
        return _in_resize_area(local_pos, rect, self._resize_margin)

    def get_resize_handle(self, item: QGraphicsItem, scene_pos: QPointF) -> int:
        """
        Get the resize handle index at *scene_pos* for *item*.
        Returns -1 if not on a handle.
        """
        rect = item.boundingRect()
        local_pos = item.mapFromScene(scene_pos)
        rotation = item.rotation() if hasattr(item, "rotation") else 0.0
        return get_handle_at_position(
            local_pos, rect, rotation, self._handle_size
        )

    def get_cursor_for_item(self, item: QGraphicsItem, scene_pos: QPointF) -> QCursor:
        """
        Determine the correct cursor when hovering over *item* at *scene_pos*.
        """
        if self.is_on_rotate_ring(item, scene_pos):
            return get_rotation_cursor()

        handle = self.get_resize_handle(item, scene_pos)
        if handle >= 0:
            rotation = item.rotation() if hasattr(item, "rotation") else 0.0
            return get_resize_cursor(rotation, handle)

        return QCursor(Qt.CursorShape.ArrowCursor)

    # --- Viewport culling ---

    def clear_items_in_viewport(self, item_type=TextBlkItem) -> int:
        """
        Deselect and hide items of *item_type* that are outside the visible viewport.
        Returns the number of items processed.

        Useful for webtoon mode where only a subset of pages should be visible.
        """
        if self.scene is None:
            return 0
        view_rect = self.view.mapToScene(self.view.viewport().rect()).boundingRect()
        count = 0
        for item in self.scene.items():
            if isinstance(item, item_type):
                item_rect = item.sceneBoundingRect()
                if not view_rect.intersects(item_rect):
                    item.setSelected(False)
                    item.setVisible(False)
                    count += 1
                else:
                    item.setVisible(True)
        return count

    def restore_all_items(self, item_type=TextBlkItem) -> int:
        """Make all hidden items of *item_type* visible again."""
        if self.scene is None:
            return 0
        count = 0
        for item in self.scene.items():
            if isinstance(item, item_type) and not item.isVisible():
                item.setVisible(True)
                count += 1
        return count
