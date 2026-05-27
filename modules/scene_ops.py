"""
Scene Ops — declarative, invertible scene mutations for robust undo/redo.

Instead of storing pre/post snapshots (which can be memory-heavy for large scenes),
SceneOp records a small, typed command that describes a change and knows how to
invert itself. This is inspired by CRDT/OT inverse operations and functional
reactivity patterns.

Each operation implements:
  apply(scene) -> None
  invert() -> SceneOp

The undo stack stores the inverse of each applied operation, making undo O(1)
instead of O(n) snapshot diffing.

Usage:
    op = MoveOp(item, old_pos=(0, 0), new_pos=(10, 20))
    op.apply(scene)
    undo_stack.push(op.invert())  # to undo later
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
from abc import ABC, abstractmethod

from qtpy.QtCore import QPointF, QRectF
from qtpy.QtWidgets import QGraphicsScene, QGraphicsItem

from utils.textblock import TextBlock
from utils.logger import logger as LOGGER


class SceneOp(ABC):
    """Base class for an invertible scene operation."""

    @abstractmethod
    def apply(self, scene: QGraphicsScene) -> None:
        """Apply this operation to the scene."""
        raise NotImplementedError

    @abstractmethod
    def invert(self) -> "SceneOp":
        """Return the inverse operation (to undo this one)."""
        raise NotImplementedError

    @abstractmethod
    def describe(self) -> str:
        """Human-readable description for undo menus."""
        raise NotImplementedError


# --- Concrete ops ---

@dataclass
class MoveOp(SceneOp):
    """Move a graphics item from old_pos to new_pos."""

    item_id: int  # QGraphicsItem.type() or a stable ID
    old_pos: Tuple[float, float]
    new_pos: Tuple[float, float]

    def apply(self, scene: QGraphicsScene) -> None:
        item = self._find_item(scene)
        if item is not None:
            item.setPos(QPointF(*self.new_pos))

    def invert(self) -> SceneOp:
        return MoveOp(self.item_id, self.new_pos, self.old_pos)

    def describe(self) -> str:
        return f"Move item {self.item_id}"

    def _find_item(self, scene: QGraphicsScene) -> Optional[QGraphicsItem]:
        for item in scene.items():
            if id(item) == self.item_id:
                return item
        return None


@dataclass
class ResizeOp(SceneOp):
    """Resize a graphics item (e.g. text block rect)."""

    item_id: int
    old_rect: Tuple[float, float, float, float]  # x, y, w, h
    new_rect: Tuple[float, float, float, float]

    def apply(self, scene: QGraphicsScene) -> None:
        item = self._find_item(scene)
        if item is not None and hasattr(item, "setRect"):
            x, y, w, h = self.new_rect
            item.setRect(QRectF(x, y, w, h))

    def invert(self) -> SceneOp:
        return ResizeOp(self.item_id, self.new_rect, self.old_rect)

    def describe(self) -> str:
        return f"Resize item {self.item_id}"

    def _find_item(self, scene: QGraphicsScene) -> Optional[QGraphicsItem]:
        for item in scene.items():
            if id(item) == self.item_id:
                return item
        return None


@dataclass
class AddItemOp(SceneOp):
    """Add an item to the scene. Inverse removes it."""

    item_data: Dict[str, Any]  # Serialized item state
    z_value: float = 0.0

    def apply(self, scene: QGraphicsScene) -> None:
        # Concrete subclasses override this
        pass

    def invert(self) -> SceneOp:
        return RemoveItemOp(self.item_data, self.z_value)

    def describe(self) -> str:
        return "Add item"


@dataclass
class RemoveItemOp(SceneOp):
    """Remove an item from the scene. Inverse re-adds it."""

    item_data: Dict[str, Any]
    z_value: float = 0.0

    def apply(self, scene: QGraphicsScene) -> None:
        pass  # Concrete subclasses override

    def invert(self) -> SceneOp:
        return AddItemOp(self.item_data, self.z_value)

    def describe(self) -> str:
        return "Remove item"


@dataclass
class TextEditOp(SceneOp):
    """Edit text content of a text block item."""

    item_id: int
    old_text: str
    new_text: str
    old_translation: str = ""
    new_translation: str = ""

    def apply(self, scene: QGraphicsScene) -> None:
        item = self._find_item(scene)
        if item is not None:
            if hasattr(item, "setPlainText"):
                item.setPlainText(self.new_text)
            if hasattr(item, "setTranslation"):
                item.setTranslation(self.new_translation)

    def invert(self) -> SceneOp:
        return TextEditOp(
            self.item_id, self.new_text, self.old_text,
            self.new_translation, self.old_translation,
        )

    def describe(self) -> str:
        return f"Edit text ({self.item_id})"

    def _find_item(self, scene: QGraphicsScene) -> Optional[QGraphicsItem]:
        for item in scene.items():
            if id(item) == self.item_id:
                return item
        return None


@dataclass
class FormatChangeOp(SceneOp):
    """Change font/format properties of a text block."""

    item_id: int
    old_format: Dict[str, Any]
    new_format: Dict[str, Any]

    def apply(self, scene: QGraphicsScene) -> None:
        item = self._find_item(scene)
        if item is not None and hasattr(item, "set_fontformat"):
            item.set_fontformat(self.new_format)

    def invert(self) -> SceneOp:
        return FormatChangeOp(self.item_id, self.new_format, self.old_format)

    def describe(self) -> str:
        return f"Format change ({self.item_id})"

    def _find_item(self, scene: QGraphicsScene) -> Optional[QGraphicsItem]:
        for item in scene.items():
            if id(item) == self.item_id:
                return item
        return None


@dataclass
class MultiOp(SceneOp):
    """Atomic batch of multiple operations (all-or-nothing)."""

    ops: List[SceneOp]

    def apply(self, scene: QGraphicsScene) -> None:
        for op in self.ops:
            op.apply(scene)

    def invert(self) -> SceneOp:
        # Reverse order and invert each sub-op
        return MultiOp([op.invert() for op in reversed(self.ops)])

    def describe(self) -> str:
        return f"Batch ({len(self.ops)} ops)"


# --- Scene Op History / Undo Stack ---

class SceneOpHistory:
    """
    Lightweight undo stack using invertible SceneOps.

    More memory-efficient than full state snapshots for large scenes,
    because each entry is a small typed command rather than a copy
    of all item states.
    """

    def __init__(self, max_depth: int = 200) -> None:
        self._undo: List[SceneOp] = []
        self._redo: List[SceneOp] = []
        self.max_depth = max_depth

    def push(self, op: SceneOp, scene: QGraphicsScene) -> None:
        """Apply an operation and push its inverse onto the undo stack."""
        op.apply(scene)
        self._undo.append(op.invert())
        self._redo.clear()
        if len(self._undo) > self.max_depth:
            self._undo.pop(0)

    def undo(self, scene: QGraphicsScene) -> bool:
        """Undo the last operation. Returns True if something was undone."""
        if not self._undo:
            return False
        inv = self._undo.pop()
        inv.apply(scene)
        self._redo.append(inv.invert())
        return True

    def redo(self, scene: QGraphicsScene) -> bool:
        """Redo the last undone operation. Returns True if something was redone."""
        if not self._redo:
            return False
        op = self._redo.pop()
        op.apply(scene)
        self._undo.append(op.invert())
        return True

    def clear(self) -> None:
        self._undo.clear()
        self._redo.clear()

    def can_undo(self) -> bool:
        return len(self._undo) > 0

    def can_redo(self) -> bool:
        return len(self._redo) > 0

    def undo_description(self) -> Optional[str]:
        if self._undo:
            return self._undo[-1].describe()
        return None

    def redo_description(self) -> Optional[str]:
        if self._redo:
            return self._redo[-1].describe()
        return None
