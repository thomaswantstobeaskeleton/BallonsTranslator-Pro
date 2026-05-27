"""
Webtoon Mode Manager — viewport-aware lazy loading for long vertical images.

Key features:
  - Viewport-culling: only keep QGraphicsPixmapItem visible when in viewport
  - Memory unloading: free far-away page pixmaps to cap RAM usage
  - Directional preloading: load next page before it scrolls into view
  - Scroll snapping (optional): snap to page boundaries after scroll stops

Inspired by Comic Translate's image_viewer.py lazy loading.
"""

import weakref
from typing import Dict, List, Optional, Tuple

from qtpy.QtCore import Qt, QObject, Signal, QRectF, QPointF, QTimer
from qtpy.QtGui import QPixmap
from qtpy.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem

from utils.logger import logger as LOGGER


# Viewport padding: load items within this many pixels of the viewport edge
DEFAULT_VIEWPORT_PADDING = 800
# Unload distance: free pixmaps for items further than this from viewport
DEFAULT_UNLOAD_DISTANCE = 2400
# Preload count: how many upcoming pages to preload ahead of scroll direction
DEFAULT_PRELOAD_COUNT = 2


class WebtoonPageItem(QGraphicsPixmapItem):
    """
    A graphics item representing one webtoon page (or a slice of a very long page).
    Can be in one of three states:
      - loaded: pixmap is set, item is visible
      - placeholder: empty rect, border outline, item visible but no pixmap
      - hidden: item not in scene (far from viewport)
    """
    def __init__(self, page_index: int, page_name: str, parent=None):
        super().__init__(parent)
        self.page_index = page_index
        self.page_name = page_name
        self._loaded = False
        self._pixmap_path: Optional[str] = None
        self.setAcceptDrops(False)

    def set_loaded_pixmap(self, pixmap: QPixmap):
        self.setPixmap(pixmap)
        self._loaded = True

    def unload_pixmap(self):
        if self._loaded:
            self.setPixmap(QPixmap())
            self._loaded = False

    def is_loaded(self) -> bool:
        return self._loaded


class WebtoonManager(QObject):
    """
    Manages lazy loading and memory culling for webtoon (long-scroll) mode.

    Attach to a Canvas scene + CustomGV view:
        manager = WebtoonManager(canvas, canvas.gv)
        manager.set_pages(page_names, img_array_loader)
        manager.enabled = True
    """

    page_visibility_changed = Signal(int, bool)  # page_index, is_visible
    page_load_requested = Signal(int)  # page_index (consumer loads pixmap and calls set_page_pixmap)
    page_unload_requested = Signal(int)  # page_index (consumer can drop from RAM cache)

    def __init__(
        self,
        scene: QGraphicsScene,
        view: QGraphicsView,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._scene_ref = weakref.ref(scene)
        self._view_ref = weakref.ref(view)
        self._pages: List[WebtoonPageItem] = []
        self._page_names: List[str] = []
        self._page_y_positions: List[float] = []
        self._enabled = False
        self._viewport_padding = DEFAULT_VIEWPORT_PADDING
        self._unload_distance = DEFAULT_UNLOAD_DISTANCE
        self._preload_count = DEFAULT_PRELOAD_COUNT
        self._current_focus_index = 0

        # Debounced viewport update timer
        self._update_timer = QTimer(self)
        self._update_timer.setSingleShot(True)
        self._update_timer.setInterval(80)  # ms
        self._update_timer.timeout.connect(self._on_update_viewport)

        # Scroll tracking
        self._scroll_direction = 0  # -1 up, 0 none, 1 down
        self._last_scroll_value = 0

    # --- Configuration ---

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value
        if value:
            self._install_scroll_tracking()
            self._on_update_viewport()
        else:
            self._uninstall_scroll_tracking()
            self._restore_all_pages()

    def set_viewport_padding(self, px: int) -> None:
        self._viewport_padding = px

    def set_unload_distance(self, px: int) -> None:
        self._unload_distance = px

    def set_preload_count(self, n: int) -> None:
        self._preload_count = n

    # --- Page lifecycle ---

    def set_pages(self, page_names: List[str], pixmaps: Optional[List[QPixmap]] = None) -> None:
        """
        Register pages for lazy loading.

        Args:
            page_names: Ordered list of page identifiers.
            pixmaps: Optional initial pixmaps; None = all start unloaded.
        """
        self.clear_pages()
        self._page_names = list(page_names)
        scene = self._scene_ref()
        if scene is None:
            return

        y_offset = 0.0
        for idx, name in enumerate(page_names):
            item = WebtoonPageItem(idx, name)
            item.setOffset(0, y_offset)
            if pixmaps and idx < len(pixmaps) and pixmaps[idx] is not None:
                item.set_loaded_pixmap(pixmaps[idx])
                y_offset += pixmaps[idx].height()
            else:
                # Placeholder height (will be updated when pixmap loads)
                y_offset += 1200  # default assumed height
            scene.addItem(item)
            self._pages.append(item)
            self._page_y_positions.append(item.offset().y())

        self._on_update_viewport()

    def set_page_pixmap(self, page_index: int, pixmap: QPixmap) -> None:
        """Call this when a page pixmap has been loaded externally."""
        if 0 <= page_index < len(self._pages):
            item = self._pages[page_index]
            item.set_loaded_pixmap(pixmap)
            # Update y positions for subsequent pages
            self._recompute_y_positions()

    def clear_pages(self) -> None:
        """Remove all managed page items from the scene."""
        scene = self._scene_ref()
        for item in self._pages:
            if scene is not None:
                scene.removeItem(item)
            item.unload_pixmap()
        self._pages.clear()
        self._page_names.clear()
        self._page_y_positions.clear()
        self._current_focus_index = 0

    # --- Viewport culling ---

    def _get_viewport_rect(self) -> QRectF:
        view = self._view_ref()
        if view is None:
            return QRectF()
        return view.mapToScene(view.viewport().rect()).boundingRect()

    def _on_update_viewport(self) -> None:
        if not self._enabled or not self._pages:
            return

        viewport = self._get_viewport_rect()
        padded = viewport.adjusted(
            0, -self._viewport_padding, 0, self._viewport_padding
        )
        unload_zone = viewport.adjusted(
            0, -self._unload_distance, 0, self._unload_distance
        )

        # Directional preload zone
        if self._scroll_direction > 0:
            # Scrolling down: preload below
            preload = viewport.adjusted(
                0, 0, 0, self._viewport_padding + (self._preload_count * 1200)
            )
        elif self._scroll_direction < 0:
            # Scrolling up: preload above
            preload = viewport.adjusted(
                0, -(self._viewport_padding + (self._preload_count * 1200)), 0, 0
            )
        else:
            preload = padded

        for idx, item in enumerate(self._pages):
            item_rect = item.sceneBoundingRect()
            in_viewport = padded.intersects(item_rect)
            in_preload = preload.intersects(item_rect)
            in_unload_zone = unload_zone.intersects(item_rect)

            if in_viewport:
                if not item.isVisible():
                    item.setVisible(True)
                    self.page_visibility_changed.emit(idx, True)
                if not item.is_loaded():
                    self.page_load_requested.emit(idx)
            elif in_preload:
                # Preload but keep hidden
                if not item.isVisible():
                    item.setVisible(False)
                if not item.is_loaded():
                    self.page_load_requested.emit(idx)
            else:
                # Far from viewport
                if item.isVisible():
                    item.setVisible(False)
                    self.page_visibility_changed.emit(idx, False)
                if item.is_loaded() and not in_unload_zone:
                    item.unload_pixmap()
                    self.page_unload_requested.emit(idx)

        # Update current focus
        self._update_focus_index(viewport)

    def _update_focus_index(self, viewport: QRectF) -> None:
        """Determine which page is currently in focus (center of viewport)."""
        center_y = viewport.center().y()
        best_idx = 0
        best_dist = float("inf")
        for idx, item in enumerate(self._pages):
            item_rect = item.sceneBoundingRect()
            dist = abs(item_rect.center().y() - center_y)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        if best_idx != self._current_focus_index:
            self._current_focus_index = best_idx

    def _recompute_y_positions(self) -> None:
        """Recompute Y offsets after a page loads/changes height."""
        y_offset = 0.0
        for idx, item in enumerate(self._pages):
            item.setOffset(0, y_offset)
            self._page_y_positions[idx] = y_offset
            bbox = item.sceneBoundingRect()
            y_offset += bbox.height()

    # --- Scroll tracking ---

    def _install_scroll_tracking(self) -> None:
        view = self._view_ref()
        if view is None:
            return
        try:
            vbar = view.verticalScrollBar()
            vbar.valueChanged.connect(self._on_scroll_changed)
        except Exception:
            pass

    def _uninstall_scroll_tracking(self) -> None:
        view = self._view_ref()
        if view is None:
            return
        try:
            vbar = view.verticalScrollBar()
            vbar.valueChanged.disconnect(self._on_scroll_changed)
        except Exception:
            pass

    def _on_scroll_changed(self, value: int) -> None:
        if value > self._last_scroll_value:
            self._scroll_direction = 1
        elif value < self._last_scroll_value:
            self._scroll_direction = -1
        else:
            self._scroll_direction = 0
        self._last_scroll_value = value
        self._update_timer.start()

    def _restore_all_pages(self) -> None:
        """When disabled, make all pages visible (revert to normal mode)."""
        for idx, item in enumerate(self._pages):
            item.setVisible(True)
            self.page_visibility_changed.emit(idx, True)

    # --- Queries ---

    def current_focus_index(self) -> int:
        return self._current_focus_index

    def current_focus_name(self) -> Optional[str]:
        if 0 <= self._current_focus_index < len(self._page_names):
            return self._page_names[self._current_focus_index]
        return None

    def page_count(self) -> int:
        return len(self._pages)

    def visible_page_indices(self) -> List[int]:
        result = []
        viewport = self._get_viewport_rect()
        padded = viewport.adjusted(0, -self._viewport_padding, 0, self._viewport_padding)
        for idx, item in enumerate(self._pages):
            if padded.intersects(item.sceneBoundingRect()):
                result.append(idx)
        return result
