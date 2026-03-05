import numpy as np
from typing import List, Union
import os

from qtpy.QtWidgets import QApplication, QSlider, QMenu, QGraphicsScene, QGraphicsSceneDragDropEvent , QGraphicsView, QGraphicsSceneDragDropEvent, QGraphicsRectItem, QGraphicsItem, QScrollBar, QGraphicsPixmapItem, QGraphicsSceneMouseEvent, QGraphicsSceneContextMenuEvent, QRubberBand
from qtpy.QtCore import Qt, QDateTime, QRectF, QPointF, QPoint, Signal, QSizeF, QEvent
from qtpy.QtGui import QKeySequence, QPixmap, QImage, QHideEvent, QKeyEvent, QWheelEvent, QResizeEvent, QPainter, QPen, QPainterPath, QCursor, QNativeGestureEvent

try:
    from qtpy.QtWidgets import QUndoStack, QUndoCommand
except:
    from qtpy.QtGui import QUndoStack, QUndoCommand

from .misc import ndarray2pixmap, QKEY, QNUMERIC_KEYS, ARROWKEY2DIRECTION
from .textitem import TextBlkItem, TextBlock
from .texteditshapecontrol import TextBlkShapeControl
from .textedit_commands import WarpItemCommand
from .custom_widget import ScrollBar, FadeLabel
from .image_edit import ImageEditMode, DrawingLayer, StrokeImgItem
from .page_search_widget import PageSearchWidget
from utils import shared as C
from utils.config import pcfg
from utils.config import context_menu_visible
from utils.proj_imgtrans import ProjImgTrans
from .context_menu_config_dialog import ContextMenuConfigDialog

CANVAS_SCALE_MAX = 10.0
CANVAS_SCALE_MIN = 0.01
CANVAS_SCALE_SPEED = 0.1

# Default size for "create text box" at cursor / right-click (no drag)
DEFAULT_TEXTBOX_WIDTH = 200
DEFAULT_TEXTBOX_HEIGHT = 50
MIN_DRAG_SIZE = 15  # below this, right-release is treated as "click" and creates default-size box

class MoveByKeyCommand(QUndoCommand):
    def __init__(self, blkitems: List[TextBlkItem], direction: QPointF, shape_ctrl: TextBlkShapeControl) -> None:
        super().__init__()
        self.blkitems = blkitems
        self.direction = direction
        self.ori_pos_list = []
        self.end_pos_list = []
        self.shape_ctrl = shape_ctrl
        for blk in blkitems:
            pos = blk.pos()
            self.ori_pos_list.append(pos)
            self.end_pos_list.append(pos + direction)

    def undo(self):
        for blk, pos in zip(self.blkitems, self.ori_pos_list):
            blk.setPos(pos)
            if blk.under_ctrl and self.shape_ctrl.blk_item == blk:
                self.shape_ctrl.updateBoundingRect()

    def redo(self):
        for blk, pos in zip(self.blkitems, self.end_pos_list):
            blk.setPos(pos)
            if blk.under_ctrl and self.shape_ctrl.blk_item == blk:
                self.shape_ctrl.updateBoundingRect()

    def mergeWith(self, other: QUndoCommand) -> bool:
        canmerge = self.blkitems == other.blkitems and self.direction == other.direction
        if canmerge:
            self.end_pos_list = other.end_pos_list
        return canmerge
    
    def id(self):
        return 1


class CustomGV(QGraphicsView):
    ctrl_pressed = False
    scale_up_signal = Signal()
    scale_down_signal = Signal()
    scale_with_value = Signal(float)
    view_resized = Signal()
    hide_canvas = Signal()
    ctrl_released = Signal()
    canvas: QGraphicsScene = None

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scrollbar_h = ScrollBar(Qt.Orientation.Horizontal, self, fadeout=True)
        self.scrollbar_v = ScrollBar(Qt.Orientation.Vertical, self, fadeout=True)

        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

    def wheelEvent(self, event : QWheelEvent) -> None:
        # qgraphicsview always scroll content according to wheelevent
        # which is not desired when scaling img
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            if event.angleDelta().y() > 0:
                self.scale_up_signal.emit()
            else:
                self.scale_down_signal.emit()
            return
        return super().wheelEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        if event.key() == QKEY.Key_Control:
            self.ctrl_pressed = False
            self.ctrl_released.emit()
        return super().keyReleaseEvent(event)

    def keyPressEvent(self, e: QKeyEvent) -> None:
        key = e.key()
        if key == QKEY.Key_Control:
            self.ctrl_pressed = True

        modifiers = e.modifiers()
        if modifiers == Qt.KeyboardModifier.ControlModifier:
            if key == QKEY.Key_V:
                # self.ctrlv_pressed.emit(e)
                if self.canvas.handle_ctrlv():
                    e.accept()
                    return
            if key == QKEY.Key_C:
                if self.canvas.handle_ctrlc():
                    e.accept()
                    return
                
        elif modifiers & Qt.KeyboardModifier.ControlModifier and modifiers & Qt.KeyboardModifier.ShiftModifier:
            if key == QKEY.Key_C:
                self.canvas.copy_src_signal.emit()
                e.accept()
                return
            elif key == QKEY.Key_V:
                self.canvas.paste_src_signal.emit()
                e.accept()
                return
            elif key == QKEY.Key_D:
                self.canvas.delete_textblks.emit(1)
                e.accept()
                return

        return super().keyPressEvent(e)

    def resizeEvent(self, event: QResizeEvent) -> None:
        self.view_resized.emit()
        return super().resizeEvent(event)

    def hideEvent(self, event: QHideEvent) -> None:
        self.hide_canvas.emit()
        return super().hideEvent(event)

    def event(self, e):
        if isinstance(e, QNativeGestureEvent):
            if e.gestureType() == Qt.NativeGestureType.ZoomNativeGesture:
                self.scale_with_value.emit(e.value() + 1)
                e.setAccepted(True)

        return super().event(e)
    
    def dragMoveEvent(self, e: QGraphicsSceneDragDropEvent):
        super().dragMoveEvent(e)
        if e.mimeData().hasUrls():
            # issue #908, https://stackoverflow.com/questions/4177720/accepting-drops-on-a-qgraphicsscene
            e.setAccepted(True)


class Canvas(QGraphicsScene):

    scalefactor_changed = Signal()
    end_create_textblock = Signal(QRectF)
    paste2selected_textitems = Signal()
    end_create_rect = Signal(QRectF, int)
    finish_painting = Signal(StrokeImgItem)
    finish_erasing = Signal(StrokeImgItem)
    delete_textblks = Signal(int)
    copy_textblks = Signal()
    paste_textblks = Signal(QPointF)
    copy_src_signal = Signal()
    paste_src_signal = Signal()
    copy_trans_signal = Signal()
    paste_trans_signal = Signal()
    clear_src_signal = Signal()
    clear_trans_signal = Signal()
    select_all_signal = Signal()
    spell_check_src_signal = Signal()
    spell_check_trans_signal = Signal()
    trim_whitespace_signal = Signal()
    to_uppercase_signal = Signal()
    to_lowercase_signal = Signal()
    toggle_strikethrough_signal = Signal()
    set_gradient_type_signal = Signal(int)  # 0 = Linear, 1 = Radial
    set_text_on_path_signal = Signal(int)  # 0 = None, 1 = Circular, 2 = Arc
    merge_selected_blocks_signal = Signal()
    split_selected_regions_signal = Signal()
    move_blocks_up_signal = Signal()
    move_blocks_down_signal = Signal()
    import_image_to_blk = Signal()  # PR #1070: import image as overlay for single empty block
    clear_overlay_signal = Signal()  # PR #1070: clear foreground image from selected block

    format_textblks = Signal()
    layout_textblks = Signal()
    auto_fit_font_signal = Signal()
    reset_angle = Signal()
    squeeze_blk = Signal()

    run_blktrans = Signal(int)
    run_detect_region = Signal(QRectF)

    begin_scale_tool = Signal(QPointF)
    scale_tool = Signal(QPointF)
    end_scale_tool = Signal()
    canvas_undostack_changed = Signal()
    
    imgtrans_proj: ProjImgTrans = None
    painting_pen = QPen()
    painting_shape = 0
    erasing_pen = QPen()
    image_edit_mode = ImageEditMode.NONE

    projstate_unsaved = False
    proj_savestate_changed = Signal(bool)
    textstack_changed = Signal()
    drop_open_folder = Signal(str)
    drop_open_files = Signal(list)
    context_menu_requested = Signal(QPoint, bool)
    incanvas_selection_changed = Signal()
    switch_text_item = Signal(int, QKeyEvent)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scale_factor = 1.
        self.text_transparency = 0
        self.textblock_mode = False
        self.creating_textblock = False
        self.create_block_origin: QPointF = None
        self.editing_textblkitem: TextBlkItem = None
        self.warp_edit_mode = False  # PR #1105: Free Transform (quad warp) editing

        self.gv = CustomGV(self)
        self.gv.scale_down_signal.connect(self.scaleDown)
        self.gv.scale_up_signal.connect(self.scaleUp)
        self.gv.scale_with_value.connect(self.scaleBy)
        self.gv.view_resized.connect(self.onViewResized)
        self.gv.hide_canvas.connect(self.on_hide_canvas)
        self.gv.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.gv.canvas = self
        self.gv.setAcceptDrops(True)
        self.gv.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.gv.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        self.context_menu_requested.connect(self.on_create_contextmenu)
        
        if not C.FLAG_QT6:
            # mitigate https://bugreports.qt.io/browse/QTBUG-93417
            # produce blurred result, saving imgs remain unaffected
            self.gv.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        self.search_widget = PageSearchWidget(self.gv)
        self.search_widget.hide()
        
        self.ctrl_relesed = self.gv.ctrl_released
        self.vscroll_bar = self.gv.verticalScrollBar()
        self.hscroll_bar = self.gv.horizontalScrollBar()
        # self.default_cursor = self.gv.cursor()
        self.rubber_band = self.addWidget(QRubberBand(QRubberBand.Shape.Rectangle))
        self.rubber_band.hide()
        self.rubber_band_origin = None
        self._last_rubber_band_rect = None  # for "Detect text in region" context menu

        self.draw_undo_stack = QUndoStack(self)
        self.text_undo_stack = QUndoStack(self)
        self.saved_drawundo_step = 0
        self.saved_textundo_step = 0

        self.scaleFactorLabel = FadeLabel(self.gv)
        self.scaleFactorLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scaleFactorLabel.setText('100%')
        self.scaleFactorLabel.gv = self.gv

        self.txtblkShapeControl = TextBlkShapeControl(self.gv)
        
        self.baseLayer = QGraphicsRectItem()
        pen = QPen()
        pen.setColor(Qt.GlobalColor.transparent)
        self.baseLayer.setPen(pen)

        self.inpaintLayer = QGraphicsPixmapItem()
        self.inpaintLayer.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.drawingLayer = DrawingLayer()
        self.drawingLayer.setTransformationMode(Qt.TransformationMode.FastTransformation)
        self.textLayer = QGraphicsPixmapItem()

        self.inpaintLayer.setAcceptDrops(True)
        self.drawingLayer.setAcceptDrops(True)
        self.textLayer.setAcceptDrops(True)
        self.baseLayer.setAcceptDrops(True)
        
        self.base_pixmap: QPixmap = None

        self.addItem(self.baseLayer)
        self.inpaintLayer.setParentItem(self.baseLayer)
        self.drawingLayer.setParentItem(self.baseLayer)
        self.textLayer.setParentItem(self.baseLayer)
        self.txtblkShapeControl.setParentItem(self.baseLayer)

        self.scalefactor_changed.connect(self.onScaleFactorChanged)
        self.selectionChanged.connect(self.on_selection_changed)     

        self.stroke_img_item: StrokeImgItem = None
        self.erase_img_key = None

        self.editor_index = 0 # 0: drawing 1: text editor
        self.mid_btn_pressed = False
        self.pan_initial_pos = QPoint(0, 0)

        self.saved_textundo_step = 0
        self.saved_drawundo_step = 0
        self.num_pushed_textstep = 0
        self.num_pushed_drawstep = 0

        self.clipboard_blks: List[TextBlock] = []

        self.drop_folder: str = None
        self.drop_files: List[str] = []
        self.block_selection_signal = False
        
        im_rect = QRectF(0, 0, C.SCREEN_W, C.SCREEN_H)
        self.baseLayer.setRect(im_rect)

        self.textlayer_trans_slider: QSlider = None
        self.originallayer_trans_slider: QSlider = None

    def on_switch_item(self, switch_delta: int, key_event: QKeyEvent = None):
        if self.textEditMode():
            self.switch_text_item.emit(switch_delta, key_event)

    def img_window_size(self):
        if self.imgtrans_proj.inpainted_valid:
            return self.inpaintLayer.pixmap().size()
        return self.baseLayer.rect().size().toSize()

    def dragEnterEvent(self, e: QGraphicsSceneDragDropEvent):
        
        self.drop_folder = None
        self.drop_files = []
        if e.mimeData().hasUrls():
            urls = e.mimeData().urls()
            image_files = []
            ufolder = None
            for url in urls:
                furl = url.toLocalFile()
                if os.path.isfile(furl) and os.path.splitext(furl)[1].lower() in {
                    '.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif', '.gif'
                }:
                    image_files.append(furl)
                elif os.path.isdir(furl) and not image_files:
                    ufolder = furl
            if image_files:
                e.acceptProposedAction()
                self.drop_files = image_files
            elif ufolder is not None:
                e.acceptProposedAction()
                self.drop_folder = ufolder

    def dropEvent(self, event) -> None:
        if self.drop_folder is not None:
            self.drop_open_folder.emit(self.drop_folder)
            self.drop_folder = None
        elif self.drop_files:
            self.drop_open_files.emit(self.drop_files)
            self.drop_files = []
        return super().dropEvent(event)

    def textEditMode(self) -> bool:
        return self.editor_index == 1

    def drawMode(self) -> bool:
        return self.editor_index == 0

    def scaleUp(self):
        self.scaleImage(1 + CANVAS_SCALE_SPEED)

    def scaleDown(self):
        self.scaleImage(1 - CANVAS_SCALE_SPEED)

    def scaleBy(self, value: float):
        self.scaleImage(value)

    def _set_scene_scale(self, scale: float):
        self.scale_factor = scale
        self.baseLayer.setScale(scale)
        self.setSceneRect(0, 0, self.baseLayer.sceneBoundingRect().width(), self.baseLayer.sceneBoundingRect().height())

    def render_result_img(self):

        self.inpaintLayer.hide()
        tlayer_opacity_before = self.textLayer.opacity()
        tlayer_visible = self.textLayer.isVisible()
        if tlayer_opacity_before != 1:
            self.textLayer.setOpacity(1)
        if not tlayer_visible:
            self.textLayer.show()
        scale_before = self.scale_factor
        if scale_before != 1:
            hb_pos = self.hscroll_bar.value()
            vb_pos = self.vscroll_bar.value()
            self._set_scene_scale(1)

        self.clearSelection()
        if self.textEditMode() and self.txtblkShapeControl.blk_item is not None:
            blk_item = self.txtblkShapeControl.blk_item
            if blk_item.is_editting():
                blk_item.endEdit(keep_focus=False)
            if blk_item.isSelected():
                blk_item.setSelected(False)

        # Hide OSB blocks that fell back to restore_original_region so we don't draw text on top.
        hidden_osb_items = []
        for item in self.items():
            if isinstance(item, TextBlkItem) and getattr(getattr(item, "blk", None), "restore_original_region", False):
                if item.isVisible():
                    item.hide()
                    hidden_osb_items.append(item)

        # Optional supersampling: render at Nx then downscale (smoother edges / better small text)
        ss = 1
        try:
            ss = int(getattr(pcfg, "supersampling_factor", 1) or 1)
        except Exception:
            ss = 1
        ss = max(1, min(4, ss))

        # OSB fallback: restore original image region for blocks that failed layout (Section 19).
        base_arr = np.copy(self.imgtrans_proj.inpainted_array)
        current_img = getattr(self.imgtrans_proj, "current_img", None)
        blk_list = self.imgtrans_proj.pages.get(current_img, []) if current_img else []
        img_array = self.imgtrans_proj.img_array
        im_h, im_w = base_arr.shape[:2]
        for blk in blk_list:
            if not getattr(blk, "restore_original_region", False):
                continue
            xyxy = getattr(blk, "xyxy", None)
            if not xyxy or len(xyxy) != 4:
                continue
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
            x1, x2 = max(0, min(x1, im_w)), max(0, min(x2, im_w))
            y1, y2 = max(0, min(y1, im_h)), max(0, min(y2, im_h))
            if x2 > x1 and y2 > y1 and img_array is not None and img_array.shape[:2] == base_arr.shape[:2]:
                base_arr[y1:y2, x1:x2] = img_array[y1:y2, x1:x2]

        base_qimg = ndarray2pixmap(base_arr, return_qimg=True)
        canvas_sz = self.img_window_size()
        if ss <= 1:
            result = base_qimg
            painter = QPainter(result)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            rect = QRectF(0, 0, canvas_sz.width(), canvas_sz.height())
            self.render(painter, rect, rect)   #  produce blurred result if target/source rect not specified #320
            painter.end()
        else:
            # Render scene into a bigger QImage, then downscale to base size.
            w = int(canvas_sz.width())
            h = int(canvas_sz.height())
            big = QImage(w * ss, h * ss, QImage.Format.Format_ARGB32)
            big.fill(Qt.GlobalColor.transparent)
            painter = QPainter(big)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            painter.scale(ss, ss)
            rect = QRectF(0, 0, w, h)
            self.render(painter, rect, rect)
            painter.end()

            # Composite big render over base image (scaled)
            result = QImage(base_qimg)  # copy
            p2 = QPainter(result)
            p2.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            p2.drawImage(QRectF(0, 0, w, h), big, QRectF(0, 0, w * ss, h * ss))
            p2.end()
        
        if tlayer_opacity_before != 1:
            self.textLayer.setOpacity(tlayer_opacity_before)
        if not tlayer_visible:
            self.textLayer.hide()
        if scale_before != 1:
            self._set_scene_scale(scale_before)
            if self.hscroll_bar.value() != hb_pos:
                self.hscroll_bar.setValue(hb_pos)
            if self.vscroll_bar.value() != vb_pos:
                self.vscroll_bar.setValue(vb_pos)
        self.inpaintLayer.show()
        for item in hidden_osb_items:
            item.show()

        return result
    
    def updateLayers(self):
        
        if not self.imgtrans_proj.img_valid:
            return
        
        view_mode = getattr(pcfg, "canvas_view_mode", "normal")
        inpainted_as_base = self.imgtrans_proj.inpainted_valid
        
        if inpainted_as_base:
            self.base_pixmap = ndarray2pixmap(self.imgtrans_proj.inpainted_array)
        else:
            self.base_pixmap = ndarray2pixmap(self.imgtrans_proj.img_array)

        if view_mode == "original":
            pixmap = ndarray2pixmap(self.imgtrans_proj.img_array)
            self.inpaintLayer.setPixmap(pixmap)
            return

        if view_mode == "translated" and inpainted_as_base:
            pixmap = self.base_pixmap.copy()
            self.inpaintLayer.setPixmap(pixmap)
            return

        pixmap = self.base_pixmap.copy()
        painter = QPainter(pixmap)
        origin = QPoint(0, 0)

        if view_mode == "debug":
            # Debug: show base image + mask overlay (boxes/masks) for QA
            if self.imgtrans_proj.mask_valid:
                painter.setOpacity(0.5)
                painter.drawPixmap(origin, ndarray2pixmap(self.imgtrans_proj.mask_array))
            painter.end()
            self.inpaintLayer.setPixmap(pixmap)
            return

        # normal
        if self.imgtrans_proj.img_valid and pcfg.original_transparency > 0:
            painter.setOpacity(pcfg.original_transparency)
            if inpainted_as_base:
                painter.drawPixmap(origin, ndarray2pixmap(self.imgtrans_proj.img_array))
            else:
                painter.drawPixmap(origin, pixmap)

        if self.imgtrans_proj.mask_valid and pcfg.mask_transparency > 0 and not self.textEditMode():
            painter.setOpacity(pcfg.mask_transparency)
            painter.drawPixmap(origin, ndarray2pixmap(self.imgtrans_proj.mask_array))

        painter.end()
        self.inpaintLayer.setPixmap(pixmap)

    def setMaskTransparency(self, transparency: float):
        pcfg.mask_transparency = transparency
        self.updateLayers()

    def setOriginalTransparency(self, transparency: float):
        pcfg.original_transparency = transparency
        self.updateLayers()

    def setTextLayerTransparency(self, transparency: float):
        self.textLayer.setOpacity(transparency)
        self.text_transparency = transparency

    def adjustScrollBar(self, scrollBar: QScrollBar, factor: float):
        scrollBar.setValue(int(factor * scrollBar.value() + ((factor - 1) * scrollBar.pageStep() / 2)))

    def scaleImage(self, factor: float):
        if not self.gv.isVisible() or not self.imgtrans_proj.img_valid:
            return
        s_f = self.scale_factor * factor
        s_f = np.clip(s_f, CANVAS_SCALE_MIN, CANVAS_SCALE_MAX)

        scale_changed = self.scale_factor != s_f
        self.scale_factor = s_f
        self.baseLayer.setScale(self.scale_factor)
        self.txtblkShapeControl.updateScale(self.scale_factor)

        if scale_changed:
            self.adjustScrollBar(self.gv.horizontalScrollBar(), factor)
            self.adjustScrollBar(self.gv.verticalScrollBar(), factor)
            self.scalefactor_changed.emit()
        self.setSceneRect(0, 0, self.baseLayer.sceneBoundingRect().width(), self.baseLayer.sceneBoundingRect().height())

    def fitToWidth(self):
        """Section 10: Zoom canvas so image width fits the view width."""
        if not self.gv.isVisible() or not self.imgtrans_proj.img_valid or self.base_pixmap is None:
            return
        view_w = max(1, self.gv.viewport().width())
        img_w = self.base_pixmap.width()
        if img_w <= 0:
            return
        target_scale = view_w / img_w
        target_scale = np.clip(target_scale, CANVAS_SCALE_MIN, CANVAS_SCALE_MAX)
        factor = target_scale / self.scale_factor
        self.scaleImage(factor)

    def onViewResized(self):
        gv_w, gv_h = self.gv.geometry().width(), self.gv.geometry().height()

        x = gv_w - self.scaleFactorLabel.width()
        y = gv_h - self.scaleFactorLabel.height()
        pos_new = (QPointF(x, y) / 2).toPoint()
        if self.scaleFactorLabel.pos() != pos_new:
            self.scaleFactorLabel.move(pos_new)
        
        x = gv_w - self.search_widget.width()
        pos = self.search_widget.pos()
        pos.setX(x-30)
        self.search_widget.move(pos)
        
    def onScaleFactorChanged(self):
        self.scaleFactorLabel.setText(f'{self.scale_factor*100:2.0f}%')
        self.scaleFactorLabel.raise_()
        self.scaleFactorLabel.startFadeAnimation()

    def on_selection_changed(self):
        if self.txtblkShapeControl.isVisible():
            blk_item = self.txtblkShapeControl.blk_item
            if blk_item is not None and blk_item.isEditing():
                blk_item.endEdit()
        if self.hasFocus() and not self.block_selection_signal:
            self.incanvas_selection_changed.emit()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()

        modifiers = event.modifiers()
        if (modifiers == Qt.KeyboardModifier.AltModifier) and \
            not key == QKEY.Key_Alt and \
                self.editing_textblkitem is None:
            if key in {QKEY.Key_W, QKEY.Key_A, QKEY.Key_Left, QKEY.Key_Up}:
                self.on_switch_item(-1, event)
                return
            elif key in {QKEY.Key_S, QKEY.Key_D, QKEY.Key_Right, QKEY.Key_Down}:
                self.on_switch_item(1, event)
                return

        if self.editing_textblkitem is not None:
            return super().keyPressEvent(event)
        elif key in ARROWKEY2DIRECTION:
            sel_blkitems = self.selected_text_items()
            if len(sel_blkitems) > 0:
                direction = ARROWKEY2DIRECTION[key]
                cmd = MoveByKeyCommand(sel_blkitems, direction, self.txtblkShapeControl)
                self.push_undo_command(cmd)
                event.setAccepted(True)
                return
        elif key in QNUMERIC_KEYS:
            value = QNUMERIC_KEYS[key]
            self.set_active_layer_transparency(value * 10)
        return super().keyPressEvent(event)
    
    def set_active_layer_transparency(self, value: int):
        if self.textEditMode():
            opacity = self.textLayer.opacity() * 100
            if value == 0 and opacity == 0:
                value = 100
            self.textlayer_trans_slider.setValue(value)
            self.originallayer_trans_slider.setValue(100 - value)
            self.updateLayers()

    def addStrokeImageItem(self, pos: QPointF, pen: QPen, erasing: bool = False, text_eraser: bool = False):
        if self.stroke_img_item is not None:
            self.stroke_img_item.startNewPoint(pos)
        else:
            if text_eraser:
                self._text_eraser_selected_blocks = self.selected_text_items()
            self.stroke_img_item = StrokeImgItem(pen, pos, self.img_window_size(), shape=self.painting_shape)
            if text_eraser:
                self.stroke_img_item.setParentItem(self.textLayer)
            elif not erasing:
                self.stroke_img_item.setParentItem(self.baseLayer)
            else:
                self.erase_img_key = str(QDateTime.currentMSecsSinceEpoch())
                compose_mode = QPainter.CompositionMode.CompositionMode_DestinationOut
                self.drawingLayer.addQImage(0, 0, self.stroke_img_item._img, compose_mode, self.erase_img_key)

    def startCreateTextblock(self, pos: QPointF, hide_control: bool = False):
        pos = pos / self.scale_factor
        self.creating_textblock = True
        self.create_block_origin = pos
        self.gv.setCursor(Qt.CursorShape.CrossCursor)
        self.txtblkShapeControl.setBlkItem(None)
        self.txtblkShapeControl.setPos(0, 0)
        self.txtblkShapeControl.setRotation(0)
        self.txtblkShapeControl.setRect(QRectF(pos, QSizeF(1, 1)))
        if hide_control:
            self.txtblkShapeControl.hideControls()
        self.txtblkShapeControl.show()

    def endCreateTextblock(self, btn=0):
        self.creating_textblock = False
        self.gv.setCursor(Qt.CursorShape.ArrowCursor)
        self.txtblkShapeControl.hide()
        textblk_created = False
        rect = self.txtblkShapeControl.rect()
        if self.creating_normal_rect:
            self.end_create_rect.emit(rect, btn)
            self.txtblkShapeControl.showControls()
        else:
            if rect.width() > 1 and rect.height() > 1:
                self.end_create_textblock.emit(rect)
                textblk_created = True
        return textblk_created

    def cancel_rect_selection(self) -> bool:
        """Cancel in-progress rect or hide rect tool selection (#126). Returns True if something was cancelled."""
        if self.creating_textblock and self.creating_normal_rect:
            self.creating_textblock = False
            self.gv.setCursor(Qt.CursorShape.ArrowCursor)
            self.txtblkShapeControl.hide()
            return True
        if self.image_edit_mode == ImageEditMode.RectTool and self.txtblkShapeControl.isVisible() and self.txtblkShapeControl.blk_item is None:
            self.txtblkShapeControl.hide()
            return True
        return False

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self.mid_btn_pressed:
            new_pos = event.screenPos()
            delta_pos = new_pos - self.pan_initial_pos
            self.pan_initial_pos = new_pos
            self.hscroll_bar.setValue(int(self.hscroll_bar.value() - delta_pos.x()))
            self.vscroll_bar.setValue(int(self.vscroll_bar.value() - delta_pos.y()))
            
        elif self.creating_textblock:
            self.txtblkShapeControl.setRect(QRectF(self.create_block_origin, event.scenePos() / self.scale_factor).normalized())
        
        elif self.stroke_img_item is not None:
            if self.stroke_img_item.is_painting:
                pos = self.inpaintLayer.mapFromScene(event.scenePos())
                if self.erase_img_key is None:
                    # painting
                    self.stroke_img_item.lineTo(pos)
                else:
                    rect = self.stroke_img_item.lineTo(pos, update=False)
                    if rect is not None:
                        self.drawingLayer.update(rect)
        
        elif self.scale_tool_mode:
            self.scale_tool.emit(event.scenePos())
        
        elif self.rubber_band.isVisible() and self.rubber_band_origin is not None:
            self.rubber_band.setGeometry(QRectF(self.rubber_band_origin, event.scenePos()).normalized())
            sel_path = QPainterPath(self.rubber_band_origin)
            sel_path.addRect(self.rubber_band.geometry())
            if C.FLAG_QT6:
                self.setSelectionArea(sel_path, deviceTransform=self.gv.viewportTransform())
            else:
                self.setSelectionArea(sel_path, Qt.ItemSelectionMode.IntersectsItemBoundingRect, self.gv.viewportTransform())
        
        return super().mouseMoveEvent(event)
    
    @property
    def scale_tool_mode(self):
        return self.drawMode() and self.gv.isVisible() and QApplication.keyboardModifiers() == Qt.KeyboardModifier.AltModifier

    def clearToolStates(self):
        self.end_scale_tool.emit()

    def selected_text_items(self, sort: bool = True) -> List[TextBlkItem]:
        sel_textitems = []
        selitems = self.selectedItems()
        for sel in selitems:
            if isinstance(sel, TextBlkItem):
                sel_textitems.append(sel)
        if sort:
            sel_textitems.sort(key = lambda x : x.idx)
        return sel_textitems

    def handle_ctrlv(self) -> bool:
        if not self.textEditMode():
            return False        
        if self.editing_textblkitem is not None and self.editing_textblkitem.isEditing():
            return False
        self.on_paste()
        return True

    def handle_ctrlc(self):
        if not self.textEditMode():
            return False        
        if self.editing_textblkitem is not None and self.editing_textblkitem.isEditing():
            return False
        self.on_copy()
        return True

    def scene_cursor_pos(self):
        origin = self.gv.mapFromGlobal(QCursor.pos())
        return self.gv.mapToScene(origin)

    def create_textbox_at_cursor(self) -> bool:
        """Create a default-size text box at current cursor position. Returns True if created."""
        if not self.textEditMode() or not self.imgtrans_proj.img_valid:
            return False
        sp = self.scene_cursor_pos()
        p = self.baseLayer.mapFromScene(sp)
        br = self.baseLayer.rect()
        x = max(0, min(p.x(), br.right() - DEFAULT_TEXTBOX_WIDTH))
        y = max(0, min(p.y(), br.bottom() - DEFAULT_TEXTBOX_HEIGHT))
        rect = QRectF(QPointF(x, y), QSizeF(DEFAULT_TEXTBOX_WIDTH, DEFAULT_TEXTBOX_HEIGHT))
        self.end_create_textblock.emit(rect)
        return True

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        btn = event.button()
        if btn == Qt.MouseButton.MiddleButton:
            self.mid_btn_pressed = True
            self.pan_initial_pos = event.screenPos()
            return
        
        if self.imgtrans_proj.img_valid:
            if self.textblock_mode and len(self.selectedItems()) == 0 and self.textEditMode():
                if btn == Qt.MouseButton.RightButton:
                    return self.startCreateTextblock(event.scenePos())
            elif self.creating_normal_rect:
                if btn == Qt.MouseButton.RightButton or btn == Qt.MouseButton.LeftButton:
                    return self.startCreateTextblock(event.scenePos(), hide_control=True)

            elif btn == Qt.MouseButton.LeftButton:
                # user is drawing using the pen/inpainting tool
                if self.scale_tool_mode:
                    self.begin_scale_tool.emit(event.scenePos())
                elif self.painting:
                    text_eraser = self.image_edit_mode == ImageEditMode.TextEraserTool
                    self.addStrokeImageItem(self.inpaintLayer.mapFromScene(event.scenePos()), self.painting_pen, text_eraser=text_eraser)

            elif btn == Qt.MouseButton.RightButton:
                if self.painting and self.image_edit_mode != ImageEditMode.TextEraserTool:
                    erasing = self.image_edit_mode == ImageEditMode.PenTool
                    self.addStrokeImageItem(self.inpaintLayer.mapFromScene(event.scenePos()), self.erasing_pen, erasing)
                elif not self.painting:
                    # rubber band selection
                    self.rubber_band_origin = event.scenePos()
                    self.rubber_band.setGeometry(QRectF(self.rubber_band_origin, self.rubber_band_origin).normalized())
                    self.rubber_band.show()
                    self.rubber_band.setZValue(1)

        return super().mousePressEvent(event)

    @property
    def creating_normal_rect(self):
        return self.image_edit_mode == ImageEditMode.RectTool and self.editor_index == 0

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        btn = event.button()
        if btn == Qt.MouseButton.RightButton and self.rubber_band.isVisible() and self.rubber_band_origin is not None:
            r = self.rubber_band.geometry().normalized()
            if r.width() >= 10 and r.height() >= 10:
                self._last_rubber_band_rect = QRectF(r)
        self.hide_rubber_band()

        Qt.MouseButton.LeftButton
        if btn == Qt.MouseButton.MiddleButton:
            self.mid_btn_pressed = False
        textblk_created = False
        if self.creating_textblock:
            tgt = 0 if btn == Qt.MouseButton.LeftButton else 1
            rect = self.txtblkShapeControl.rect()
            if btn == Qt.MouseButton.RightButton and rect.width() < MIN_DRAG_SIZE and rect.height() < MIN_DRAG_SIZE:
                default_rect = QRectF(self.create_block_origin, QSizeF(DEFAULT_TEXTBOX_WIDTH, DEFAULT_TEXTBOX_HEIGHT))
                self.end_create_textblock.emit(default_rect)
                self.endCreateTextblock(btn=tgt)  # clear cursor and shape control (won't emit again)
                textblk_created = True
            else:
                textblk_created = self.endCreateTextblock(btn=tgt)
        if btn == Qt.MouseButton.RightButton:
            if self.stroke_img_item is not None:
                self.finish_erasing.emit(self.stroke_img_item)
            if self.textEditMode() and not textblk_created:
                self._context_menu_scene_pos = event.scenePos()
                self.context_menu_requested.emit(event.screenPos(), False)
        if btn == Qt.MouseButton.LeftButton:
            if self.stroke_img_item is not None:
                self.finish_painting.emit(self.stroke_img_item)
            elif self.scale_tool_mode:
                self.end_scale_tool.emit()
        return super().mouseReleaseEvent(event)

    def updateCanvas(self):
        self.editing_textblkitem = None
        self.stroke_img_item = None
        self.erase_img_key = None
        self.txtblkShapeControl.setBlkItem(None)
        self.mid_btn_pressed = False
        self.search_widget.reInitialize()

        self.clearSelection()
        self.setProjSaveState(False)
        self.updateLayers()

        if self.base_pixmap is not None:
            pixmap = self.base_pixmap.copy()
            pixmap.fill(Qt.GlobalColor.transparent)
            self.textLayer.setPixmap(pixmap)

            im_rect = pixmap.rect()
            self.baseLayer.setRect(QRectF(im_rect))
            if im_rect != self.sceneRect():
                self.setSceneRect(0, 0, im_rect.width(), im_rect.height())
            self.scaleImage(1)

        self.setDrawingLayer()


    def setDrawingLayer(self, img: Union[QPixmap, np.ndarray] = None):
        
        self.drawingLayer.clearAllDrawings()

        if not self.imgtrans_proj.img_valid:
            return
        if img is None:
            drawing_map = self.inpaintLayer.pixmap().copy()
            drawing_map.fill(Qt.GlobalColor.transparent)
        elif not isinstance(img, QPixmap):
            drawing_map = ndarray2pixmap(img)
        else:
            drawing_map = img
        self.drawingLayer.setPixmap(drawing_map)

    def setPaintMode(self, painting: bool):
        if painting:
            self.editing_textblkitem = None
            self.textblock_mode = False
        else:
            # self.gv.setCursor(self.default_cursor)
            self.gv.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.image_edit_mode = ImageEditMode.NONE

    @property
    def painting(self):
        return self.image_edit_mode == ImageEditMode.PenTool or self.image_edit_mode == ImageEditMode.InpaintTool or self.image_edit_mode == ImageEditMode.TextEraserTool

    def setMaskTransparencyBySlider(self, slider_value: int):
        self.setMaskTransparency(slider_value / 100)

    def setOriginalTransparencyBySlider(self, slider_value: int):
        self.setOriginalTransparency(slider_value / 100)

    def setTextLayerTransparencyBySlider(self, slider_value: int):
        self.setTextLayerTransparency(slider_value / 100)

    def setTextBlockMode(self, mode: bool):
        self.textblock_mode = mode

    def on_create_contextmenu(self, pos: QPoint, is_textpanel: bool):
        if self.textEditMode() and not self.creating_textblock:
            menu = QMenu(self.gv)
            menu.setStyleSheet("QMenu::item:disabled { opacity: 0.5; }")
            sel = self.selected_text_items()
            n_sel = len(sel)
            n_total = len(self.imgtrans_proj.pages[self.imgtrans_proj.current_img]) if (self.imgtrans_proj and self.imgtrans_proj.current_img and self.imgtrans_proj.current_img in self.imgtrans_proj.pages) else 0
            saved_rect = getattr(self, '_last_rubber_band_rect', None)

            # --- Edit ---
            copy_act = paste_act = copy_trans_act = paste_trans_act = None
            copy_src_act = paste_src_act = delete_act = delete_recover_act = None
            clear_src_act = clear_trans_act = select_all_act = None
            edit_menu = None
            if context_menu_visible('edit_copy'):
                if edit_menu is None: edit_menu = menu.addMenu(self.tr("Edit"))
                copy_act = edit_menu.addAction(self.tr("Copy"))
                copy_act.setShortcut(QKeySequence.StandardKey.Copy)
            if context_menu_visible('edit_paste'):
                if edit_menu is None: edit_menu = menu.addMenu(self.tr("Edit"))
                paste_act = edit_menu.addAction(self.tr("Paste"))
                paste_act.setShortcut(QKeySequence.StandardKey.Paste)
            if context_menu_visible('edit_copy_trans'):
                if edit_menu is None: edit_menu = menu.addMenu(self.tr("Edit"))
                copy_trans_act = edit_menu.addAction(self.tr("Copy translation"))
            if context_menu_visible('edit_paste_trans'):
                if edit_menu is None: edit_menu = menu.addMenu(self.tr("Edit"))
                paste_trans_act = edit_menu.addAction(self.tr("Paste translation"))
            if context_menu_visible('edit_copy_src'):
                if edit_menu is None: edit_menu = menu.addMenu(self.tr("Edit"))
                copy_src_act = edit_menu.addAction(self.tr("Copy source text"))
                copy_src_act.setShortcut(QKeySequence("Ctrl+Shift+C"))
            if context_menu_visible('edit_paste_src'):
                if edit_menu is None: edit_menu = menu.addMenu(self.tr("Edit"))
                paste_src_act = edit_menu.addAction(self.tr("Paste source text"))
                paste_src_act.setShortcut(QKeySequence("Ctrl+Shift+V"))
            if context_menu_visible('edit_delete'):
                if edit_menu is None: edit_menu = menu.addMenu(self.tr("Edit"))
                delete_act = edit_menu.addAction(self.tr("Delete"))
                delete_act.setShortcut(QKeySequence("Ctrl+D"))
            if context_menu_visible('edit_delete_recover'):
                if edit_menu is None: edit_menu = menu.addMenu(self.tr("Edit"))
                delete_recover_act = edit_menu.addAction(self.tr("Delete and Recover removed text"))
                delete_recover_act.setShortcut(QKeySequence("Ctrl+Shift+D"))
            if context_menu_visible('edit_clear_src'):
                if edit_menu is None: edit_menu = menu.addMenu(self.tr("Edit"))
                clear_src_act = edit_menu.addAction(self.tr("Clear source text"))
            if context_menu_visible('edit_clear_trans'):
                if edit_menu is None: edit_menu = menu.addMenu(self.tr("Edit"))
                clear_trans_act = edit_menu.addAction(self.tr("Clear translation"))
            if context_menu_visible('edit_select_all'):
                if edit_menu is None: edit_menu = menu.addMenu(self.tr("Edit"))
                select_all_act = edit_menu.addAction(self.tr("Select all"))
                select_all_act.setShortcut(QKeySequence.StandardKey.SelectAll)

            # --- Text (selection) ---
            spell_check_src_act = spell_check_trans_act = trim_whitespace_act = None
            to_uppercase_act = to_lowercase_act = toggle_strikethrough_act = None
            gradient_linear_act = gradient_radial_act = None
            text_on_path_none_act = text_on_path_circular_act = text_on_path_arc_act = None
            text_menu = None
            if n_sel >= 1:
                if context_menu_visible('text_spell_src'):
                    if text_menu is None: text_menu = menu.addMenu(self.tr("Text"))
                    spell_check_src_act = text_menu.addAction(self.tr("Spell check source text"))
                if context_menu_visible('text_spell_trans'):
                    if text_menu is None: text_menu = menu.addMenu(self.tr("Text"))
                    spell_check_trans_act = text_menu.addAction(self.tr("Spell check translation"))
                if context_menu_visible('text_trim'):
                    if text_menu is None: text_menu = menu.addMenu(self.tr("Text"))
                    trim_whitespace_act = text_menu.addAction(self.tr("Trim whitespace"))
                if context_menu_visible('text_upper'):
                    if text_menu is None: text_menu = menu.addMenu(self.tr("Text"))
                    to_uppercase_act = text_menu.addAction(self.tr("To uppercase"))
                if context_menu_visible('text_lower'):
                    if text_menu is None: text_menu = menu.addMenu(self.tr("Text"))
                    to_lowercase_act = text_menu.addAction(self.tr("To lowercase"))
                if context_menu_visible('text_strikethrough'):
                    if text_menu is None: text_menu = menu.addMenu(self.tr("Text"))
                    toggle_strikethrough_act = text_menu.addAction(self.tr("Toggle strikethrough"))
                if context_menu_visible('text_gradient'):
                    if text_menu is None: text_menu = menu.addMenu(self.tr("Text"))
                    gradient_sub = text_menu.addMenu(self.tr("Gradient type"))
                    gradient_linear_act = gradient_sub.addAction(self.tr("Linear"))
                    gradient_radial_act = gradient_sub.addAction(self.tr("Radial"))
                if context_menu_visible('text_on_path'):
                    if text_menu is None: text_menu = menu.addMenu(self.tr("Text"))
                    path_sub = text_menu.addMenu(self.tr("Text on path"))
                    text_on_path_none_act = path_sub.addAction(self.tr("None"))
                    text_on_path_circular_act = path_sub.addAction(self.tr("Circular"))
                    text_on_path_arc_act = path_sub.addAction(self.tr("Arc"))

            # --- Block (selection) ---
            merge_blocks_act = split_regions_act = move_up_act = move_down_act = None
            create_textbox_act = None
            block_menu = None
            if context_menu_visible('create_textbox'):
                if block_menu is None: block_menu = menu.addMenu(self.tr("Block"))
                create_textbox_act = block_menu.addAction(self.tr("Create text box"))
                create_textbox_act.setToolTip(self.tr("Add a new text box at the right-click position (default size). Or right-click and drag to set size."))
            if n_sel >= 1:
                if context_menu_visible('block_merge'):
                    if block_menu is None: block_menu = menu.addMenu(self.tr("Block"))
                    merge_blocks_act = block_menu.addAction(self.tr("Merge selected blocks"))
                    merge_blocks_act.setEnabled(n_sel >= 2)
                if context_menu_visible('block_split'):
                    if block_menu is None: block_menu = menu.addMenu(self.tr("Block"))
                    split_regions_act = block_menu.addAction(self.tr("Split selected region(s)"))
                if context_menu_visible('block_move_up'):
                    if block_menu is None: block_menu = menu.addMenu(self.tr("Block"))
                    move_up_act = block_menu.addAction(self.tr("Move block(s) up"))
                    move_up_act.setEnabled(n_sel == 1 and sel[0].idx > 0)
                if context_menu_visible('block_move_down'):
                    if block_menu is None: block_menu = menu.addMenu(self.tr("Block"))
                    move_down_act = block_menu.addAction(self.tr("Move block(s) down"))
                    move_down_act.setEnabled(n_sel == 1 and n_total > 0 and sel[0].idx < n_total - 1)

            # --- Image / Overlay (selection) ---
            import_image_act = clear_overlay_act = None
            overlay_menu = None
            if n_sel >= 1:
                if context_menu_visible('overlay_import'):
                    if overlay_menu is None: overlay_menu = menu.addMenu(self.tr("Image / Overlay"))
                    import_image_act = overlay_menu.addAction(self.tr("Import Image"))
                    import_image_act.setEnabled(n_sel == 1 and sel[0].document().isEmpty())
                if context_menu_visible('overlay_clear'):
                    if overlay_menu is None: overlay_menu = menu.addMenu(self.tr("Image / Overlay"))
                    clear_overlay_act = overlay_menu.addAction(self.tr("Clear overlay image"))
                    clear_overlay_act.setEnabled(bool(n_sel == 1 and getattr(getattr(sel[0], 'blk', None), 'foreground_image_path', None)))

            # --- Transform (selection) ---
            free_transform_act = reset_warp_act = None
            warp_arc_up_act = warp_arc_down_act = warp_arch_act = warp_flag_act = None
            transform_menu = None
            if n_sel >= 1:
                if context_menu_visible('transform_free'):
                    if transform_menu is None: transform_menu = menu.addMenu(self.tr("Transform"))
                    free_transform_act = transform_menu.addAction(self.tr("Free Transform"))
                    free_transform_act.setCheckable(True)
                    free_transform_act.setChecked(getattr(self, 'warp_edit_mode', False))
                if context_menu_visible('transform_reset_warp'):
                    if transform_menu is None: transform_menu = menu.addMenu(self.tr("Transform"))
                    reset_warp_act = transform_menu.addAction(self.tr("Reset warp"))
                    reset_warp_act.setEnabled(bool(n_sel == 1 and getattr(getattr(sel[0], 'blk', None), 'warp_mode', None) in ('quad', 'mesh')))
                if context_menu_visible('transform_warp_preset'):
                    if transform_menu is None: transform_menu = menu.addMenu(self.tr("Transform"))
                    warp_preset_sub = transform_menu.addMenu(self.tr("Warp preset"))
                    warp_arc_up_act = warp_preset_sub.addAction(self.tr("Arc Up"))
                    warp_arc_down_act = warp_preset_sub.addAction(self.tr("Arc Down"))
                    warp_arch_act = warp_preset_sub.addAction(self.tr("Arch"))
                    warp_flag_act = warp_preset_sub.addAction(self.tr("Flag"))

            # --- Order (selection) ---
            bring_front_act = send_back_act = None
            order_menu = None
            if n_sel >= 1:
                if context_menu_visible('order_bring_front'):
                    if order_menu is None: order_menu = menu.addMenu(self.tr("Order"))
                    bring_front_act = order_menu.addAction(self.tr("Bring to front"))
                if context_menu_visible('order_send_back'):
                    if order_menu is None: order_menu = menu.addMenu(self.tr("Order"))
                    send_back_act = order_menu.addAction(self.tr("Send to back"))

            menu.addSeparator()

            # --- Format ---
            format_act = layout_act = auto_fit_act = angle_act = squeeze_act = None
            format_menu = None
            if context_menu_visible('format_apply'):
                if format_menu is None: format_menu = menu.addMenu(self.tr("Format"))
                format_act = format_menu.addAction(self.tr("Apply font formatting"))
            if context_menu_visible('format_layout'):
                if format_menu is None: format_menu = menu.addMenu(self.tr("Format"))
                layout_act = format_menu.addAction(self.tr("Auto layout"))
            if context_menu_visible('format_auto_fit') and n_sel >= 1:
                if format_menu is None: format_menu = menu.addMenu(self.tr("Format"))
                auto_fit_act = format_menu.addAction(self.tr("Auto fit font size to box"))
                if auto_fit_act is not None:
                    auto_fit_act.setToolTip(self.tr("Scale font size so text fits the selected text box(es). Use after changing font."))
            if context_menu_visible('format_angle'):
                if format_menu is None: format_menu = menu.addMenu(self.tr("Format"))
                angle_act = format_menu.addAction(self.tr("Reset Angle"))
            if context_menu_visible('format_squeeze'):
                if format_menu is None: format_menu = menu.addMenu(self.tr("Format"))
                squeeze_act = format_menu.addAction(self.tr("Squeeze"))

            menu.addSeparator()

            # --- Detect & Run ---
            detect_region_act = detect_page_act = translate_act = ocr_act = None
            ocr_translate_act = ocr_translate_inpaint_act = inpaint_act = None
            run_menu = None
            if context_menu_visible('run_detect_region') and saved_rect is not None:
                if run_menu is None: run_menu = menu.addMenu(self.tr("Detect & Run"))
                detect_region_act = run_menu.addAction(self.tr("Detect text in region"))
            if context_menu_visible('run_detect_page'):
                if run_menu is None: run_menu = menu.addMenu(self.tr("Detect & Run"))
                detect_page_act = run_menu.addAction(self.tr("Detect text on page"))
            if context_menu_visible('run_translate'):
                if run_menu is None: run_menu = menu.addMenu(self.tr("Detect & Run"))
                translate_act = run_menu.addAction(self.tr("Translate"))
            if context_menu_visible('run_ocr'):
                if run_menu is None: run_menu = menu.addMenu(self.tr("Detect & Run"))
                ocr_act = run_menu.addAction(self.tr("OCR"))
            if context_menu_visible('run_ocr_translate'):
                if run_menu is None: run_menu = menu.addMenu(self.tr("Detect & Run"))
                ocr_translate_act = run_menu.addAction(self.tr("OCR and translate"))
            if context_menu_visible('run_ocr_translate_inpaint'):
                if run_menu is None: run_menu = menu.addMenu(self.tr("Detect & Run"))
                ocr_translate_inpaint_act = run_menu.addAction(self.tr("OCR, translate and inpaint"))
            if context_menu_visible('run_inpaint'):
                if run_menu is None: run_menu = menu.addMenu(self.tr("Detect & Run"))
                inpaint_act = run_menu.addAction(self.tr("Inpaint"))

            menu.addSeparator()
            configure_menu_act = menu.addAction(self.tr("Configure menu..."))

            rst = menu.exec(pos)
            self._last_rubber_band_rect = None

            if rst == configure_menu_act:
                dlg = ContextMenuConfigDialog(self.gv)
                dlg.exec()
                return

            if rst == create_textbox_act:
                sp = getattr(self, '_context_menu_scene_pos', None)
                if sp is not None and self.imgtrans_proj.img_valid:
                    br = self.baseLayer.rect()
                    if saved_rect is not None and saved_rect.width() >= MIN_DRAG_SIZE and saved_rect.height() >= MIN_DRAG_SIZE:
                        # Use the right-drag rectangle size and position (scene coords -> baseLayer coords)
                        tl = self.baseLayer.mapFromScene(saved_rect.topLeft())
                        br_pt = self.baseLayer.mapFromScene(saved_rect.bottomRight())
                        rect = QRectF(tl, br_pt).normalized().intersected(br)
                        if rect.isEmpty():
                            p = self.baseLayer.mapFromScene(sp)
                            rect = QRectF(QPointF(max(0, min(p.x(), br.right() - DEFAULT_TEXTBOX_WIDTH)), max(0, min(p.y(), br.bottom() - DEFAULT_TEXTBOX_HEIGHT))), QSizeF(DEFAULT_TEXTBOX_WIDTH, DEFAULT_TEXTBOX_HEIGHT))
                    else:
                        p = self.baseLayer.mapFromScene(sp)
                        x = max(0, min(p.x(), br.right() - DEFAULT_TEXTBOX_WIDTH))
                        y = max(0, min(p.y(), br.bottom() - DEFAULT_TEXTBOX_HEIGHT))
                        rect = QRectF(QPointF(x, y), QSizeF(DEFAULT_TEXTBOX_WIDTH, DEFAULT_TEXTBOX_HEIGHT))
                    self.end_create_textblock.emit(rect)
                return

            if rst == delete_act:
                self.delete_textblks.emit(0)
            elif rst == delete_recover_act:
                self.delete_textblks.emit(1)
            elif rst == copy_act:
                self.on_copy()
            elif rst == paste_act:
                self.on_paste()
            elif rst == copy_trans_act:
                self.copy_trans_signal.emit()
            elif rst == paste_trans_act:
                self.paste_trans_signal.emit()
            elif rst == copy_src_act:
                self.copy_src_signal.emit()
            elif rst == paste_src_act:
                self.paste_src_signal.emit()
            elif rst == clear_src_act:
                self.clear_src_signal.emit()
            elif rst == clear_trans_act:
                self.clear_trans_signal.emit()
            elif rst == select_all_act:
                self.select_all_signal.emit()
            elif spell_check_src_act is not None and rst == spell_check_src_act:
                self.spell_check_src_signal.emit()
            elif spell_check_trans_act is not None and rst == spell_check_trans_act:
                self.spell_check_trans_signal.emit()
            elif trim_whitespace_act is not None and rst == trim_whitespace_act:
                self.trim_whitespace_signal.emit()
            elif to_uppercase_act is not None and rst == to_uppercase_act:
                self.to_uppercase_signal.emit()
            elif to_lowercase_act is not None and rst == to_lowercase_act:
                self.to_lowercase_signal.emit()
            elif toggle_strikethrough_act is not None and rst == toggle_strikethrough_act:
                self.toggle_strikethrough_signal.emit()
            elif gradient_linear_act is not None and rst == gradient_linear_act:
                self.set_gradient_type_signal.emit(0)
            elif gradient_radial_act is not None and rst == gradient_radial_act:
                self.set_gradient_type_signal.emit(1)
            elif text_on_path_none_act is not None and rst == text_on_path_none_act:
                self.set_text_on_path_signal.emit(0)
            elif text_on_path_circular_act is not None and rst == text_on_path_circular_act:
                self.set_text_on_path_signal.emit(1)
            elif text_on_path_arc_act is not None and rst == text_on_path_arc_act:
                self.set_text_on_path_signal.emit(2)
            elif merge_blocks_act is not None and rst == merge_blocks_act:
                self.merge_selected_blocks_signal.emit()
            elif split_regions_act is not None and rst == split_regions_act:
                self.split_selected_regions_signal.emit()
            elif move_up_act is not None and rst == move_up_act:
                self.move_blocks_up_signal.emit()
            elif move_down_act is not None and rst == move_down_act:
                self.move_blocks_down_signal.emit()
            elif import_image_act is not None and rst == import_image_act:
                self.import_image_to_blk.emit()
            elif clear_overlay_act is not None and rst == clear_overlay_act:
                self.clear_overlay_signal.emit()
            elif free_transform_act is not None and rst == free_transform_act:
                self.warp_edit_mode = not self.warp_edit_mode
                self.txtblkShapeControl.setWarpEditing(self.warp_edit_mode)
            elif reset_warp_act is not None and rst == reset_warp_act and n_sel == 1:
                item = sel[0]
                blk = getattr(item, 'blk', None)
                if blk is not None and getattr(blk, 'warp_mode', None) in ('quad', 'mesh'):
                    before = {'warp_mode': blk.warp_mode, 'warp_quad': [list(p) for p in blk.warp_quad]}
                    after = {'warp_mode': 'none', 'warp_quad': [[0, 0], [1, 0], [1, 1], [0, 1]]}
                    self.push_undo_command(WarpItemCommand(item, before, after, self.txtblkShapeControl))
                    if self.txtblkShapeControl.blk_item == item:
                        self.txtblkShapeControl.warp_editing = False
                        self.txtblkShapeControl.updateControlBlocks()
                    self.setProjSaveState(False)
            elif rst == warp_arc_up_act or rst == warp_arc_down_act or rst == warp_arch_act or rst == warp_flag_act:
                if n_sel == 1:
                    item = sel[0]
                    blk = getattr(item, 'blk', None)
                    if blk is not None:
                        presets = {
                            warp_arc_up_act: [[0.05, 0], [0.95, 0], [1, 1], [0, 1]],
                            warp_arc_down_act: [[0, 0.08], [1, 0.08], [1, 1], [0, 1]],
                            warp_arch_act: [[0, 0.1], [1, 0.1], [1, 1], [0, 1]],
                            warp_flag_act: [[0.02, 0], [0.98, 0.02], [0.98, 0.98], [0.02, 1]],
                        }
                        quad = presets.get(rst)
                        if quad is not None:
                            before = {'warp_mode': getattr(blk, 'warp_mode', 'none'), 'warp_quad': [list(p) for p in getattr(blk, 'warp_quad', [[0,0],[1,0],[1,1],[0,1]])]}
                            blk.warp_mode = 'quad'
                            blk.warp_quad = [list(p) for p in quad]
                            after = {'warp_mode': 'quad', 'warp_quad': blk.warp_quad}
                            self.push_undo_command(WarpItemCommand(item, before, after, self.txtblkShapeControl))
                            item.update()
                            if self.txtblkShapeControl.blk_item == item:
                                self.txtblkShapeControl.setWarpEditing(True)
                                self.txtblkShapeControl.updateControlBlocks()
                            self.setProjSaveState(False)
            elif bring_front_act is not None and rst == bring_front_act:
                items = self.selected_text_items()
                if items:
                    parent = items[0].parentItem()
                    if parent is not None:
                        children = parent.childItems()
                        text_items = [c for c in children if isinstance(c, TextBlkItem)]
                        max_z = max((c.zValue() for c in text_items), default=0)
                        for it in items:
                            it.setZValue(max_z + 1)
            elif send_back_act is not None and rst == send_back_act:
                items = self.selected_text_items()
                if items:
                    parent = items[0].parentItem()
                    if parent is not None:
                        children = parent.childItems()
                        text_items = [c for c in children if isinstance(c, TextBlkItem)]
                        min_z = min((c.zValue() for c in text_items), default=0)
                        for it in items:
                            it.setZValue(min_z - 1)
            elif rst == format_act:
                self.format_textblks.emit()
            elif rst == layout_act:
                self.layout_textblks.emit()
            elif rst == auto_fit_act:
                self.auto_fit_font_signal.emit()
            elif rst == angle_act:
                self.reset_angle.emit()
            elif rst == squeeze_act:
                self.squeeze_blk.emit()
            elif rst == detect_region_act:
                if saved_rect is not None:
                    self.run_detect_region.emit(saved_rect)
            elif rst == detect_page_act:
                if self.imgtrans_proj is not None and self.imgtrans_proj.img_valid and self.sceneRect().isValid():
                    self.run_detect_region.emit(self.sceneRect())
            elif rst == translate_act:
                self.run_blktrans.emit(-1)
            elif rst == ocr_act:
                self.run_blktrans.emit(0)
            elif rst == ocr_translate_act:
                self.run_blktrans.emit(1)
            elif rst == ocr_translate_inpaint_act:
                self.run_blktrans.emit(2)
            elif rst == inpaint_act:
                self.run_blktrans.emit(3)

    @property
    def have_selected_blkitem(self):
        return len(self.selected_text_items()) > 0

    def on_paste(self, p: QPointF = None):
        if self.textEditMode():
            if p is None:
                p = self.scene_cursor_pos()
            if self.have_selected_blkitem:
                self.paste2selected_textitems.emit()
            else:
                self.paste_textblks.emit(p)

    def on_copy(self):
        if self.textEditMode():
            if self.have_selected_blkitem:
                self.copy_textblks.emit()

    def hide_rubber_band(self):
        if self.rubber_band.isVisible():
            self.rubber_band.hide()
            self.rubber_band_origin = None
    
    def on_hide_canvas(self):
        self.clear_states()

    def on_activation_changed(self):
        self.clear_states()
        for textitem in self.selected_text_items():
            if textitem.isEditing():
                self.editing_textblkitem = textitem

    def clear_states(self):
        self.creating_textblock = False
        self.create_block_origin = None
        self.editing_textblkitem = None
        self.gv.ctrl_pressed = False
        if self.stroke_img_item is not None:
            self.removeItem(self.stroke_img_item)

    def setProjSaveState(self, un_saved: bool):
        if un_saved == self.projstate_unsaved:
            return
        else:
            self.projstate_unsaved = un_saved
            self.proj_savestate_changed.emit(un_saved)

    def removeItem(self, item: QGraphicsItem) -> None:
        self.block_selection_signal = True
        super().removeItem(item)
        if isinstance(item, StrokeImgItem):
            item.setParentItem(None)
            self.stroke_img_item = None
            self.erase_img_key = None
        self.block_selection_signal = False

    def get_active_undostack(self) -> QUndoStack:
        if self.textEditMode():
            return self.text_undo_stack
        elif self.drawMode():
            return self.draw_undo_stack
        return None

    def push_undo_command(self, command: QUndoCommand, update_pushed_step=True):
        if self.textEditMode():
            self.push_text_command(command, update_pushed_step)
        elif self.drawMode():
            self.push_draw_command(command, update_pushed_step)
        else:
            return

    def push_draw_command(self, command: QUndoCommand, update_pushed_step=True):
        if command is not None:
            self.draw_undo_stack.push(command)
        if update_pushed_step:
            self.num_pushed_drawstep += 1
            self.on_drawstack_changed()

    def push_text_command(self, command: QUndoCommand, update_pushed_step=True):
        if command is not None:
            self.text_undo_stack.push(command)
        if update_pushed_step:
            self.num_pushed_textstep += 1
            self.on_textstack_changed()

    def on_drawstack_changed(self):
        if self.num_pushed_drawstep != self.saved_drawundo_step or self.num_pushed_textstep != self.saved_textundo_step:
            self.setProjSaveState(True)
        else:
            self.setProjSaveState(False)

    def on_textstack_changed(self):
        if self.num_pushed_textstep != self.saved_textundo_step or self.num_pushed_drawstep != self.saved_drawundo_step:
            self.setProjSaveState(True)
        else:
            self.setProjSaveState(False)
        self.textstack_changed.emit()

    def redo_textedit(self):
        self.num_pushed_textstep += 1
        self.text_undo_stack.redo()

    def undo_textedit(self):
        if self.num_pushed_textstep > 0:
            self.num_pushed_textstep -= 1
        self.text_undo_stack.undo()

    def redo(self):
        if self.textEditMode():
            undo_stack = self.text_undo_stack
            self.num_pushed_textstep += 1
            self.on_textstack_changed()
        elif self.drawMode():
            undo_stack = self.draw_undo_stack
            self.num_pushed_drawstep += 1
            self.on_drawstack_changed()
        else:
            return
        if undo_stack is not None:
            undo_stack.redo()
            if undo_stack == self.text_undo_stack:
                self.txtblkShapeControl.updateBoundingRect()

    def undo(self):
        if self.textEditMode():
            undo_stack = self.text_undo_stack
            if self.num_pushed_textstep > 0:
                self.num_pushed_textstep -= 1
            self.on_textstack_changed()
        elif self.drawMode():
            undo_stack = self.draw_undo_stack
            if self.num_pushed_drawstep > 0:
                self.num_pushed_drawstep -= 1
            self.on_drawstack_changed()
        else:
            return
        if undo_stack is not None:
            undo_stack.undo()
            if undo_stack == self.text_undo_stack:
                self.txtblkShapeControl.updateBoundingRect()

    def clear_undostack(self, update_saved_step=False):
        if update_saved_step:
            self.saved_drawundo_step = 0
            self.saved_textundo_step = 0
            self.num_pushed_textstep = 0
            self.num_pushed_drawstep = 0
        self.draw_undo_stack.clear()
        self.text_undo_stack.clear()

    def clear_text_stack(self):
        self.num_pushed_textstep = 0
        self.text_undo_stack.clear()

    def clear_draw_stack(self):
        self.num_pushed_drawstep = 0
        self.draw_undo_stack.clear()

    def update_saved_undostep(self):
        self.saved_drawundo_step = self.num_pushed_drawstep
        self.saved_textundo_step = self.num_pushed_textstep

    def text_change_unsaved(self) -> bool:
        return self.saved_textundo_step != self.num_pushed_textstep

    def draw_change_unsaved(self) -> bool:
        return self.saved_drawundo_step != self.num_pushed_drawstep

    def prepareClose(self):
        self.blockSignals(True)
        self.text_undo_stack.blockSignals(True)
        self.draw_undo_stack.blockSignals(True)

