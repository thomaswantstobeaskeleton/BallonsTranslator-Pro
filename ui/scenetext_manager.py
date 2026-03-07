
from typing import List, Union, Tuple
import numpy as np
import copy

from qtpy.QtWidgets import QApplication, QWidget, QGraphicsItem
from qtpy.QtCore import QObject, QRectF, Qt, Signal, QPointF, QPoint
from qtpy.QtGui import QKeyEvent, QTextCursor, QFontMetricsF, QFont, QTextCharFormat, QClipboard
try:
    from qtpy.QtWidgets import QUndoCommand
except:
    from qtpy.QtGui import QUndoCommand

from .textitem import TextBlkItem, TextBlock
from .canvas import Canvas
from .textedit_area import TransTextEdit, SourceTextEdit, TransPairWidget, SelectTextMiniMenu, TextEditListScrollArea, QVBoxLayout, Widget
from utils.fontformat import FontFormat
from .textedit_commands import propagate_user_edit, TextEditCommand, ReshapeItemCommand, MoveBlkItemsCommand, AutoLayoutCommand, ApplyFontformatCommand, RotateItemCommand, WarpItemCommand, TextItemEditCommand, TextEditCommand, PageReplaceOneCommand, PageReplaceAllCommand, MultiPasteCommand, ResetAngleCommand, SqueezeCommand
from .text_panel import FontFormatPanel
from utils.config import pcfg
from utils import shared
from utils.imgproc_utils import extract_ballon_region, rotate_polygons, get_block_mask
from utils.text_processing import seg_text, is_cjk
from utils.text_layout import layout_text
from utils.line_breaking import find_optimal_breaks_dp, split_long_token_with_hyphenation
from modules.textdetector.outside_text_processor import OSB_LABELS

# Layout tuning: keep text inside bubbles and avoid overflow/tiny text (see layout_textblk)
LAYOUT_FIT_RATIO = 0.80  # fit text to this fraction of region (strict = no overflow)
LAYOUT_FIT_RATIO_MIN = 0.50  # minimum scale when fitting (avoid illegible text; was 0.35)
LAYOUT_MIN_FONT_PT = 10.0  # minimum font size in points (was 7; raise so text stays readable)
LAYOUT_HEIGHT_PADDING = 1.06  # multiplier for block height (bottom margin)
LAYOUT_WIDTH_STROKE_FACTOR = 1.28  # width = max line width * this (stroke/outline clearance)
LAYOUT_SCALE_UP_MAX = 1.4  # allow scaling font up to this when bubble is large and text is short
LAYOUT_SCALE_UP_FILL = 0.78  # target fill ratio when scaling up in large bubbles


class CreateItemCommand(QUndoCommand):
    def __init__(self, blk_item: TextBlkItem, ctrl, parent=None):
        super().__init__(parent)
        self.blk_item = blk_item
        self.ctrl: SceneTextManager = ctrl
        self.op_count = -1
        self.ctrl.addTextBlock(self.blk_item)
        self.pairw = self.ctrl.pairwidget_list[self.blk_item.idx]
        self.ctrl.txtblkShapeControl.setBlkItem(self.blk_item)

    def redo(self):
        if self.op_count < 0:
            self.op_count += 1
            self.blk_item.setSelected(True)
            return
        self.ctrl.recoverTextblkItemList([self.blk_item], [self.pairw])

    def undo(self):
        self.ctrl.deleteTextblkItemList([self.blk_item], [self.pairw])


class EmptyCommand(QUndoCommand):
    def __init__(self, parent=None):
        super().__init__(parent=parent)


class DeleteBlkItemsCommand(QUndoCommand):
    def __init__(self, blk_list: List[TextBlkItem], mode: int, ctrl, parent=None):
        super().__init__(parent)
        self.op_counter = 0
        self.blk_list = []
        self.pwidget_list: List[TransPairWidget] = []
        self.ctrl: SceneTextManager = ctrl
        self.sw = self.ctrl.canvas.search_widget
        self.canvas: Canvas = ctrl.canvas
        self.mode = mode

        self.undo_img_list = []
        self.redo_img_list = []
        self.inpaint_rect_lst = []
        self.mask_pnts = []
        img_array = self.canvas.imgtrans_proj.inpainted_array
        mask_array = self.canvas.imgtrans_proj.mask_array
        original_array = self.canvas.imgtrans_proj.img_array

        self.search_rstedit_list: List[SourceTextEdit] = []
        self.search_counter_list = []
        self.highlighter_list = []
        self.old_counter_sum = self.sw.counter_sum
        self.sw_changed = False

        blk_list.sort(key=lambda blk: blk.idx)
        
        for blkitem in blk_list:
            if not isinstance(blkitem, TextBlkItem):
                continue
            self.blk_list.append(blkitem)
            pw: TransPairWidget = ctrl.pairwidget_list[blkitem.idx]
            self.pwidget_list.append(pw)

            if mode == 1:
                is_empty = False
                msk, xyxy = get_block_mask(blkitem.absBoundingRect(), mask_array, blkitem.rotation())
                if msk is None:
                    is_empty = True
                if is_empty:
                    self.undo_img_list.append(None)
                    self.redo_img_list.append(None)
                    self.inpaint_rect_lst.append(None)
                    self.mask_pnts.append(None)
                else:
                    x1, y1, x2, y2 = xyxy
                    self.mask_pnts.append(np.where(msk))
                    self.undo_img_list.append(np.copy(img_array[y1: y2, x1: x2]))
                    self.redo_img_list.append(np.copy(original_array[y1: y2, x1: x2]))
                    self.inpaint_rect_lst.append([x1, y1, x2, y2])

            rst_idx = self.sw.get_result_edit_index(pw.e_trans)
            if rst_idx != -1:
                self.sw_changed = True
                highlighter = self.sw.highlighter_list.pop(rst_idx)
                counter = self.sw.search_counter_list.pop(rst_idx)
                self.sw.counter_sum -= counter
                if self.sw.current_edit == pw.e_trans:
                    highlighter.set_current_span(-1, -1)
                self.search_rstedit_list.append(self.sw.search_rstedit_list.pop(rst_idx))
                self.search_counter_list.append(counter)
                self.highlighter_list.append(highlighter)

            rst_idx = self.sw.get_result_edit_index(pw.e_source)
            if rst_idx != -1:
                self.sw_changed = True
                highlighter = self.sw.highlighter_list.pop(rst_idx)
                counter = self.sw.search_counter_list.pop(rst_idx)
                self.sw.counter_sum -= counter
                if self.sw.current_edit == pw.e_trans:
                    highlighter.set_current_span(-1, -1)
                self.search_rstedit_list.append(self.sw.search_rstedit_list.pop(rst_idx))
                self.search_counter_list.append(counter)
                self.highlighter_list.append(highlighter)

        self.new_counter_sum = self.sw.counter_sum
        if self.sw_changed:
            if self.sw.counter_sum > 0:
                idx = self.sw.get_result_edit_index(self.sw.current_edit)
                if self.sw.current_cursor is not None and idx != -1:
                    self.sw.result_pos = self.sw.highlighter_list[idx].matched_map[self.sw.current_cursor.position()]
                    if idx > 0:
                        self.sw.result_pos += sum(self.sw.search_counter_list[: idx])
                    self.sw.updateCounterText()
                else:
                    self.sw.setCurrentEditor(self.sw.search_rstedit_list[0])
            else:
                self.sw.setCurrentEditor(None)

        # Sync project pages: remove deleted blocks so save/run/pipeline stay consistent (Issue 9).
        self.pages_restore: List[Tuple[int, object]] = []
        page_name = self.canvas.imgtrans_proj.current_img
        if page_name and page_name in self.canvas.imgtrans_proj.pages:
            for blkitem in reversed(self.blk_list):
                idx = blkitem.idx
                removed = self.canvas.imgtrans_proj.pages[page_name].pop(idx)
                self.pages_restore.append((idx, removed))
            self.pages_restore.sort(key=lambda x: x[0])

        self.ctrl.deleteTextblkItemList(self.blk_list, self.pwidget_list)
        self.canvas.updateLayers()

    def redo(self):

        if self.mode == 1:
            self.canvas.saved_drawundo_step -= 1
            img_array = self.canvas.imgtrans_proj.inpainted_array
            mask_array = self.canvas.imgtrans_proj.mask_array
            for mskpnt, inpaint_rect, redo_img in zip(self.mask_pnts, self.inpaint_rect_lst, self.redo_img_list):
                if mskpnt == None:
                    continue
                x1, y1, x2, y2 = inpaint_rect
                img_array[y1: y2, x1: x2][mskpnt] = redo_img[mskpnt]
                mask_array[y1: y2, x1: x2][mskpnt] = 0
            self.canvas.updateLayers()

        if self.op_counter == 0:
            self.op_counter += 1
            return

        # Sync project pages again when re-doing delete (blocks were restored on undo).
        if getattr(self, 'pages_restore', None):
            page_name = self.canvas.imgtrans_proj.current_img
            if page_name and page_name in self.canvas.imgtrans_proj.pages:
                for idx, _ in sorted(self.pages_restore, key=lambda x: -x[0]):
                    self.canvas.imgtrans_proj.pages[page_name].pop(idx)

        self.ctrl.deleteTextblkItemList(self.blk_list, self.pwidget_list)
        self.canvas.updateLayers()
        if self.sw_changed:
            self.sw.counter_sum = self.new_counter_sum
            cursor_removed = False
            for edit in self.search_rstedit_list:
                idx = self.sw.get_result_edit_index(edit)
                if idx != -1:
                    self.sw.search_rstedit_list.pop(idx)
                    self.sw.search_counter_list.pop(idx)
                    self.sw.highlighter_list.pop(idx)
                if edit == self.sw.current_edit:
                    cursor_removed = True
            if cursor_removed:
                if self.sw.counter_sum > 0:
                    self.sw.setCurrentEditor(self.sw.search_rstedit_list[0])
                else:
                    self.sw.setCurrentEditor(None)

    def undo(self):

        if self.mode == 1:
            self.canvas.saved_drawundo_step += 1
            img_array = self.canvas.imgtrans_proj.inpainted_array
            mask_array = self.canvas.imgtrans_proj.mask_array
            for mskpnt, inpaint_rect, undo_img in zip(self.mask_pnts, self.inpaint_rect_lst, self.undo_img_list):
                if mskpnt == None:
                    continue
                x1, y1, x2, y2 = inpaint_rect
                img_array[y1: y2, x1: x2][mskpnt] = undo_img[mskpnt]
                mask_array[y1: y2, x1: x2][mskpnt] = 255
            self.canvas.updateLayers()

        # Restore blocks to project pages so delete/recover stays in sync (Issue 9).
        if getattr(self, 'pages_restore', None):
            page_name = self.canvas.imgtrans_proj.current_img
            if page_name and page_name in self.canvas.imgtrans_proj.pages:
                for idx, blk in self.pages_restore:
                    self.canvas.imgtrans_proj.pages[page_name].insert(idx, blk)

        self.ctrl.recoverTextblkItemList(self.blk_list, self.pwidget_list)
        self.canvas.updateLayers()
        if self.sw_changed:
            self.sw.counter_sum = self.old_counter_sum
            self.sw.search_rstedit_list += self.search_rstedit_list
            self.sw.search_counter_list += self.search_counter_list
            self.sw.highlighter_list += self.highlighter_list
            self.sw.updateCounterText()


class PasteBlkItemsCommand(QUndoCommand):
    def __init__(self, blk_list: List[TextBlkItem], pwidget_list: List[TransPairWidget], ctrl, parent=None):
        super().__init__(parent)
        self.op_counter = 0
        self.blk_list = blk_list
        self.ctrl:SceneTextManager = ctrl
        blk_list.sort(key=lambda blk: blk.idx)

        self.ctrl.canvas.block_selection_signal = True
        for blkitem in blk_list:
            blkitem.setSelected(True)
        self.ctrl.on_incanvas_selection_changed()
        self.ctrl.canvas.block_selection_signal = False
        self.pwidget_list = pwidget_list
        

    def redo(self):
        if self.op_counter == 0:
            self.op_counter += 1
            return
        self.ctrl.recoverTextblkItemList(self.blk_list, self.pwidget_list)

    def undo(self):
        self.ctrl.deleteTextblkItemList(self.blk_list, self.pwidget_list)


class PasteSrcItemsCommand(QUndoCommand):
    def __init__(self, src_list: List[SourceTextEdit], paste_list: List[str]):
        super().__init__()
        self.src_list = src_list
        self.paste_list = paste_list
        self.ori_text_list = [src.toPlainText() for src in src_list]

    def redo(self):
        for src, text in zip(self.src_list, self.paste_list):
            src.setPlainText(text)

    def undo(self):
        for src, text in zip(self.src_list, self.ori_text_list):
            src.setPlainText(text)


class RearrangeBlksCommand(QUndoCommand):

    def __init__(self, rmap: Tuple, ctrl, parent=None):
        super().__init__(parent)
        self.ctrl: SceneTextManager = ctrl
        self.src_ids, self.tgt_ids = rmap[0], rmap[1]

        self.nr = len(self.src_ids)
        self.src2tgt = {}
        self.tgt2src = {}
        for s, t in zip(self.src_ids, self.tgt_ids):
            self.src2tgt[s] = t
            self.tgt2src[t] = s
        self.visible_ = None
        self.redo_visible_idx = self.undo_visible_idx = None
        if len(rmap) > 2:
            self.redo_visible_idx, self.undo_visible_idx = rmap[2]

    def redo(self):
        self.rearange_blk_ids(self.src_ids, self.tgt_ids, self.redo_visible_idx)

    def undo(self):
        self.rearange_blk_ids(self.tgt_ids, self.src_ids, self.undo_visible_idx)

    def rearange_blk_ids(self, src_ids, tgt_ids, visible_idx = None):
        src_ids = np.array(src_ids)
        tgt_ids = np.array(tgt_ids)
        src_order_ids = np.argsort(src_ids)[::-1]

        src_ids = src_ids[src_order_ids]
        tgt_ids = tgt_ids[src_order_ids]
        
        blks: List[TextBlkItem] = []
        pws: List[TransPairWidget] = []
        for pos, pos_tgt in zip(src_ids, tgt_ids):
            pw = self.ctrl.pairwidget_list.pop(pos)
            if visible_idx == pos_tgt:
                pw.hide()
            blk = self.ctrl.textblk_item_list.pop(pos)
            pws.append(pw)
            blks.append(blk)

        tgt_order_ids = np.argsort(tgt_ids)
        for ii in tgt_order_ids:
            pos = tgt_ids[ii]
            self.ctrl.textblk_item_list.insert(pos, blks[ii])
            
            self.ctrl.textEditList.insertPairWidget(pws[ii], pos)
            self.ctrl.pairwidget_list.insert(pos, pws[ii])

        self.ctrl.updateTextBlkItemIdx(set(tgt_ids))
        if visible_idx is not None:
            pw_ct = self.ctrl.pairwidget_list[visible_idx]
            pw_ct.show()
            self.ctrl.textEditList.ensureWidgetVisible(pw_ct, yMargin=pw.height())


class TextPanel(Widget):
    def __init__(self, app: QApplication, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        layout = QVBoxLayout(self)
        self.textEditList = TextEditListScrollArea(self)
        self.formatpanel = FontFormatPanel(app, self)
        layout.addWidget(self.formatpanel)
        layout.addWidget(self.textEditList)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(7)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)


class SceneTextManager(QObject):
    new_textblk = Signal(int)
    def __init__(self, 
                 app: QApplication,
                 mainwindow: QWidget,
                 canvas: Canvas, 
                 textpanel: TextPanel, 
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.app = app     
        self.mainwindow = mainwindow
        self.canvas = canvas
        canvas.switch_text_item.connect(self.on_switch_textitem)
        self.selectext_minimenu: SelectTextMiniMenu = None
        self.canvas.scalefactor_changed.connect(self.adjustSceneTextRect)
        self.canvas.end_create_textblock.connect(self.onEndCreateTextBlock)
        self.canvas.paste2selected_textitems.connect(self.on_paste2selected_textitems)
        self.canvas.delete_textblks.connect(self.onDeleteBlkItems)
        self.canvas.copy_textblks.connect(self.onCopyBlkItems)
        self.canvas.paste_textblks.connect(self.onPasteBlkItems)
        self.canvas.format_textblks.connect(self.onFormatTextblks)
        self.canvas.layout_textblks.connect(self.onAutoLayoutTextblks)
        self.canvas.auto_fit_font_signal.connect(self.onAutoFitFontToBox)
        self.canvas.reset_angle.connect(self.onResetAngle)
        self.canvas.squeeze_blk.connect(self.onSqueezeBlk)
        self.canvas.incanvas_selection_changed.connect(self.on_incanvas_selection_changed)
        self.txtblkShapeControl = canvas.txtblkShapeControl
        self.textpanel = textpanel
        self.textEditList = textpanel.textEditList
        self.textEditList.focus_out.connect(self.on_textedit_list_focusout)
        self.textEditList.textpanel_contextmenu_requested.connect(canvas.on_create_contextmenu)
        self.textEditList.selection_changed.connect(self.on_transwidget_selection_changed)
        self.textEditList.rearrange_blks.connect(self.on_rearrange_blks)
        self.formatpanel = textpanel.formatpanel
        self.formatpanel.textstyle_panel.apply_fontfmt.connect(self.onFormatTextblks)
        self.formatpanel.apply_global_to_all_blocks_requested.connect(self.apply_global_format_to_all_blocks)

        self.imgtrans_proj = self.canvas.imgtrans_proj
        self.textblk_item_list: List[TextBlkItem] = []
        self.pairwidget_list: List[TransPairWidget] = self.textEditList.pairwidget_list

        self.auto_textlayout_flag = False
        self.hovering_transwidget : TransTextEdit = None

        self.prev_blkitem: TextBlkItem = None

    def on_switch_textitem(self, switch_delta: int, key_event: QKeyEvent = None, current_editing_widget: Union[SourceTextEdit, TransTextEdit] = None):
        n_blk = len(self.textblk_item_list)
        if n_blk < 1:
            return
        
        editing_blk = None
        if current_editing_widget is None:
            editing_blk = self.editingTextItem()
            if editing_blk is not None:
                tgt_idx = editing_blk.idx + switch_delta
            else:
                sel_blks = self.canvas.selected_text_items(sort=False)
                if len(sel_blks) == 0:
                    return
                sel_blk = sel_blks[0]
                tgt_idx = sel_blk.idx + switch_delta
        else:
            tgt_idx = current_editing_widget.idx + switch_delta

        if tgt_idx < 0:
            tgt_idx += n_blk
        elif tgt_idx >= n_blk:
            tgt_idx -= n_blk
        blk = self.textblk_item_list[tgt_idx]

        if current_editing_widget is None:
            if editing_blk is None:
                self.canvas.block_selection_signal = True
                self.canvas.clearSelection()
                blk.setSelected(True)
                self.canvas.block_selection_signal = False
                self.canvas.gv.ensureVisible(blk)
                self.txtblkShapeControl.setBlkItem(blk)
                edit = self.pairwidget_list[tgt_idx].e_trans
                self.changeHoveringWidget(edit)
                self.textEditList.set_selected_list([blk.idx])
            else:
                editing_blk.endEdit()
                editing_blk.setSelected(False)
                self.txtblkShapeControl.setBlkItem(blk)
                blk.setSelected(True)
                blk.startEdit()
                self.canvas.gv.ensureVisible(blk)
        else:
            self.textblk_item_list[current_editing_widget.idx].setSelected(False)
            current_pw = self.pairwidget_list[tgt_idx]
            is_trans = isinstance(current_editing_widget, TransTextEdit)
            if is_trans:
                w = current_pw.e_trans
            else:
                w = current_pw.e_source

            self.changeHoveringWidget(w)
            w.setFocus()

        if key_event is not None:
            key_event.accept()

    def setTextEditMode(self, edit: bool = False):
        if edit:
            self.textpanel.show()
            self.canvas.textLayer.show()
        else:
            self.txtblkShapeControl.setBlkItem(None)
            self.textpanel.hide()
            self.textpanel.formatpanel.set_textblk_item()
            self.canvas.textLayer.hide()

    def adjustSceneTextRect(self):
        self.txtblkShapeControl.updateBoundingRect()

    def clearSceneTextitems(self):
        self.hovering_transwidget = None
        self.txtblkShapeControl.setBlkItem(None)
        for blkitem in self.textblk_item_list:
            self.canvas.removeItem(blkitem)
        self.textblk_item_list.clear()
        self.textEditList.clearAllSelected()
        for textwidget in self.pairwidget_list:
            self.textEditList.removeWidget(textwidget)
        self.pairwidget_list.clear()

    def updateSceneTextitems(self):
        self.hovering_transwidget = None
        self.txtblkShapeControl.setBlkItem(None)
        self.clearSceneTextitems()
        for textblock in self.imgtrans_proj.current_block_list():
            if textblock.font_family is None or textblock.font_family.strip() == '':
                textblock.font_family = self.formatpanel.familybox.currentText()
            blk_item = self.addTextBlock(textblock)
        if self.auto_textlayout_flag:
            self.updateTextBlkList()

    def addTextBlock(self, blk: Union[TextBlock, TextBlkItem] = None) -> TextBlkItem:
        if isinstance(blk, TextBlkItem):
            blk_item = blk
            blk_item.idx = len(self.textblk_item_list)
        else:
            # Apply global stroke default to new blocks when they have no stroke (#1143)
            if blk.fontformat.stroke_width == 0 and getattr(pcfg.global_fontformat, 'stroke_width', 0) > 0:
                blk.fontformat.stroke_width = pcfg.global_fontformat.stroke_width
                blk.fontformat.srgb = list(getattr(pcfg.global_fontformat, 'srgb', [0, 0, 0]))
            translation = ''
            if self.auto_textlayout_flag and not blk.vertical:
                translation = blk.translation
                blk.translation = ''
            blk_item = TextBlkItem(blk, len(self.textblk_item_list), show_rect=self.canvas.textblock_mode)
            if translation:
                blk.translation = translation
                rst = self.layout_textblk(blk_item, text=translation)
                if rst is None:
                    blk_item.setPlainText(translation)
        self.addTextBlkItem(blk_item)

        pair_widget = TransPairWidget(blk, len(self.pairwidget_list), pcfg.fold_textarea)
        self.pairwidget_list.append(pair_widget)
        self.textEditList.addPairWidget(pair_widget)
        pair_widget.e_source.setPlainText(blk_item.blk.get_text())
        pair_widget.e_source.focus_in.connect(self.on_transwidget_focus_in)
        pair_widget.e_source.ensure_scene_visible.connect(self.on_ensure_textitem_svisible)
        pair_widget.e_source.push_undo_stack.connect(self.on_push_edit_stack)
        pair_widget.e_source.redo_signal.connect(self.on_textedit_redo)
        pair_widget.e_source.undo_signal.connect(self.on_textedit_undo)
        pair_widget.e_source.show_select_menu.connect(self.on_show_select_menu)
        pair_widget.e_source.focus_out.connect(self.on_pairw_focusout)
        pair_widget.e_source.select_all_blocks_requested.connect(lambda: self.set_blkitems_selection(True))
        pair_widget.e_trans.select_all_blocks_requested.connect(lambda: self.set_blkitems_selection(True))

        pair_widget.e_trans.setPlainText(blk_item.toPlainText())
        pair_widget.e_trans.focus_in.connect(self.on_transwidget_focus_in)
        pair_widget.e_trans.propagate_user_edited.connect(self.on_propagate_transwidget_edit)
        pair_widget.e_trans.ensure_scene_visible.connect(self.on_ensure_textitem_svisible)
        pair_widget.e_trans.push_undo_stack.connect(self.on_push_edit_stack)
        pair_widget.e_trans.redo_signal.connect(self.on_textedit_redo)
        pair_widget.e_trans.undo_signal.connect(self.on_textedit_undo)
        pair_widget.e_trans.show_select_menu.connect(self.on_show_select_menu)
        pair_widget.e_trans.focus_out.connect(self.on_pairw_focusout)
        pair_widget.drag_move.connect(self.textEditList.handle_drag_pos)
        pair_widget.pw_drop.connect(self.textEditList.on_pw_dropped)
        pair_widget.idx_edited.connect(self.textEditList.on_idx_edited)

        self.new_textblk.emit(blk_item.idx)
        return blk_item

    def insertTextBlocksAt(self, idx: int, blk_list: List[TextBlock]) -> List[TextBlkItem]:
        """Insert new text blocks at index idx (used e.g. after splitting a region). Caller must update imgtrans_proj.pages[page_name] before calling."""
        if idx < 0 or idx > len(self.textblk_item_list) or not blk_list:
            return []
        new_items = []
        new_widgets = []
        for k, blk in enumerate(blk_list):
            blk_item = TextBlkItem(blk, idx + k, show_rect=self.canvas.textblock_mode)
            pair_widget = TransPairWidget(blk, idx + k, pcfg.fold_textarea)
            pair_widget.e_source.setPlainText(blk_item.blk.get_text())
            pair_widget.e_trans.setPlainText(blk_item.toPlainText())
            pair_widget.e_source.focus_in.connect(self.on_transwidget_focus_in)
            pair_widget.e_source.ensure_scene_visible.connect(self.on_ensure_textitem_svisible)
            pair_widget.e_source.push_undo_stack.connect(self.on_push_edit_stack)
            pair_widget.e_source.redo_signal.connect(self.on_textedit_redo)
            pair_widget.e_source.undo_signal.connect(self.on_textedit_undo)
            pair_widget.e_source.show_select_menu.connect(self.on_show_select_menu)
            pair_widget.e_source.focus_out.connect(self.on_pairw_focusout)
            pair_widget.e_source.select_all_blocks_requested.connect(lambda: self.set_blkitems_selection(True))
            pair_widget.e_trans.select_all_blocks_requested.connect(lambda: self.set_blkitems_selection(True))
            pair_widget.e_trans.focus_in.connect(self.on_transwidget_focus_in)
            pair_widget.e_trans.propagate_user_edited.connect(self.on_propagate_transwidget_edit)
            pair_widget.e_trans.ensure_scene_visible.connect(self.on_ensure_textitem_svisible)
            pair_widget.e_trans.push_undo_stack.connect(self.on_push_edit_stack)
            pair_widget.e_trans.redo_signal.connect(self.on_textedit_redo)
            pair_widget.e_trans.undo_signal.connect(self.on_textedit_undo)
            pair_widget.e_trans.show_select_menu.connect(self.on_show_select_menu)
            pair_widget.e_trans.focus_out.connect(self.on_pairw_focusout)
            pair_widget.drag_move.connect(self.textEditList.handle_drag_pos)
            pair_widget.pw_drop.connect(self.textEditList.on_pw_dropped)
            pair_widget.idx_edited.connect(self.textEditList.on_idx_edited)
            new_items.append(blk_item)
            new_widgets.append(pair_widget)
        for k in range(len(blk_list)):
            self.textblk_item_list.insert(idx + k, new_items[k])
            self.pairwidget_list.insert(idx + k, new_widgets[k])
            self.textEditList.insertPairWidget(new_widgets[k], idx + k)
        for blk_item in new_items:
            blk_item.setParentItem(self.canvas.textLayer)
            blk_item.begin_edit.connect(self.onTextBlkItemBeginEdit)
            blk_item.end_edit.connect(self.onTextBlkItemEndEdit)
            blk_item.hover_enter.connect(self.onTextBlkItemHoverEnter)
            blk_item.leftbutton_pressed.connect(self.onLeftbuttonPressed)
            blk_item.moving.connect(self.onTextBlkItemMoving)
            blk_item.moved.connect(self.onTextBlkItemMoved)
            blk_item.reshaped.connect(self.onTextBlkItemReshaped)
            blk_item.rotated.connect(self.onTextBlkItemRotated)
            blk_item.warped.connect(self.onTextBlkItemWarped)
            blk_item.push_undo_stack.connect(self.on_push_textitem_undostack)
            blk_item.undo_signal.connect(self.on_textedit_undo)
            blk_item.redo_signal.connect(self.on_textedit_redo)
            blk_item.propagate_user_edited.connect(self.on_propagate_textitem_edit)
            blk_item.doc_size_changed.connect(self.onTextBlkItemSizeChanged)
            blk_item.pasted.connect(self.onBlkitemPaste)
        self.updateTextBlkItemIdx()
        for k in range(len(blk_list)):
            self.new_textblk.emit(idx + k)
        return new_items

    def addTextBlkItem(self, textblk_item: TextBlkItem) -> TextBlkItem:
        self.textblk_item_list.append(textblk_item)
        textblk_item.setParentItem(self.canvas.textLayer)
        textblk_item.begin_edit.connect(self.onTextBlkItemBeginEdit)
        textblk_item.end_edit.connect(self.onTextBlkItemEndEdit)
        textblk_item.hover_enter.connect(self.onTextBlkItemHoverEnter)
        textblk_item.leftbutton_pressed.connect(self.onLeftbuttonPressed)
        textblk_item.moving.connect(self.onTextBlkItemMoving)
        textblk_item.moved.connect(self.onTextBlkItemMoved)
        textblk_item.reshaped.connect(self.onTextBlkItemReshaped)
        textblk_item.rotated.connect(self.onTextBlkItemRotated)
        textblk_item.warped.connect(self.onTextBlkItemWarped)
        textblk_item.push_undo_stack.connect(self.on_push_textitem_undostack)
        textblk_item.undo_signal.connect(self.on_textedit_undo)
        textblk_item.redo_signal.connect(self.on_textedit_redo)
        textblk_item.propagate_user_edited.connect(self.on_propagate_textitem_edit)
        textblk_item.doc_size_changed.connect(self.onTextBlkItemSizeChanged)
        textblk_item.pasted.connect(self.onBlkitemPaste)
        return textblk_item

    def deleteTextblkItemList(self, blkitem_list: List[TextBlkItem], p_widget_list: List[TransPairWidget]):
        selection_changed = False
        for blkitem, p_widget in zip(blkitem_list, p_widget_list):
            if blkitem.isSelected():
                selection_changed = True
            self.canvas.removeItem(blkitem) # removeItem itself will block incanvas_selection_changed
            self.textblk_item_list.remove(blkitem)
            self.pairwidget_list.remove(p_widget)
            self.textEditList.removeWidget(p_widget)
        self.updateTextBlkItemIdx()
        self.txtblkShapeControl.setBlkItem(None)
        if selection_changed:
            # it must be called after updateTextBlkItemIdx if blk.idx changed
            self.on_incanvas_selection_changed()

    def swap_block_positions(self, i: int, j: int):
        """Swap two text blocks (and their pair widgets) at indices i and j. Updates project page list and scene."""
        if i == j or i < 0 or j < 0 or i >= len(self.textblk_item_list) or j >= len(self.textblk_item_list):
            return
        page_name = self.canvas.imgtrans_proj.current_img
        if not page_name or page_name not in self.canvas.imgtrans_proj.pages:
            return
        pages = self.canvas.imgtrans_proj.pages[page_name]
        if len(pages) != len(self.textblk_item_list):
            return
        pages[i], pages[j] = pages[j], pages[i]
        self.textblk_item_list[i], self.textblk_item_list[j] = self.textblk_item_list[j], self.textblk_item_list[i]
        self.pairwidget_list[i], self.pairwidget_list[j] = self.pairwidget_list[j], self.pairwidget_list[i]
        self.textblk_item_list[i].idx = i
        self.textblk_item_list[j].idx = j
        self.pairwidget_list[i].idx = i
        self.pairwidget_list[j].idx = j
        self.textEditList.removeWidget(self.pairwidget_list[i])
        self.textEditList.removeWidget(self.pairwidget_list[j])
        self.textEditList.insertPairWidget(self.pairwidget_list[i], i)
        self.textEditList.insertPairWidget(self.pairwidget_list[j], j)

    def recoverTextblkItemList(self, blkitem_list: List[TextBlkItem], p_widget_list: List[TransPairWidget]):
        self.canvas.block_selection_signal = True
        for blkitem, p_widget in zip(blkitem_list, p_widget_list):
            self.textblk_item_list.insert(blkitem.idx, blkitem)
            blkitem.setParentItem(self.canvas.textLayer)
            self.pairwidget_list.insert(p_widget.idx, p_widget)
            self.textEditList.insertPairWidget(p_widget, p_widget.idx)
            if self.txtblkShapeControl.blk_item is not None and blkitem.isSelected():
                blkitem.setSelected(False)
        self.updateTextBlkItemIdx()
        self.on_incanvas_selection_changed()
        self.canvas.block_selection_signal = False
        
    def onTextBlkItemSizeChanged(self, idx: int):
        blk_item = self.textblk_item_list[idx]
        if not self.txtblkShapeControl.reshaping:
            if self.txtblkShapeControl.blk_item == blk_item:
                self.txtblkShapeControl.updateBoundingRect()

    @property
    def app_clipborad(self) -> QClipboard:
        return self.app.clipboard()

    def onBlkitemPaste(self, idx: int):
        blk_item = self.textblk_item_list[idx]
        text = self.app_clipborad.text()
        cursor = blk_item.textCursor()
        cursor.insertText(text)

    def onTextBlkItemBeginEdit(self, blk_id: int):
        blk_item = self.textblk_item_list[blk_id]
        self.txtblkShapeControl.setBlkItem(blk_item)
        self.canvas.editing_textblkitem = blk_item
        self.formatpanel.set_textblk_item(blk_item)
        self.txtblkShapeControl.startEditing()
        e_trans = self.pairwidget_list[blk_item.idx].e_trans
        self.changeHoveringWidget(e_trans)

    def changeHoveringWidget(self, edit: SourceTextEdit):
        if self.hovering_transwidget is not None and self.hovering_transwidget != edit:
            self.hovering_transwidget.setHoverEffect(False)
        self.hovering_transwidget = edit
        if edit is not None:
            pw = self.pairwidget_list[edit.idx]
            h = pw.height()
            if shared.USE_PYSIDE6:
                self.textEditList.ensureWidgetVisible(pw, ymargin=h)
            else:
                self.textEditList.ensureWidgetVisible(pw, yMargin=h)
            edit.setHoverEffect(True)

    def onLeftbuttonPressed(self, blk_id: int):
        blk_item = self.textblk_item_list[blk_id]
        self.txtblkShapeControl.setBlkItem(blk_item)
        selections: List[TextBlkItem] = self.canvas.selectedItems()
        if len(selections) > 1:
            for item in selections:
                item.oldPos = item.pos()
        self.changeHoveringWidget(self.pairwidget_list[blk_id].e_trans)
        # Select the corresponding pair widget on the right bar (no focus in source/translation)
        self.textEditList.set_selected_list([blk_id])

    def onTextBlkItemEndEdit(self, blk_id: int):
        self.canvas.editing_textblkitem = None
        self.textblk_item_list[blk_id].setSelected(True)
        self.txtblkShapeControl.endEditing()

    def editingTextItem(self) -> TextBlkItem:
        if self.txtblkShapeControl.isVisible() and self.canvas.editing_textblkitem is not None:
            return self.canvas.editing_textblkitem
        return None

    def savePrevBlkItem(self, blkitem: TextBlkItem):
        self.prev_blkitem = blkitem
        self.prev_textCursor = QTextCursor(self.prev_blkitem.textCursor())

    def is_editting(self):
        blk_item = self.txtblkShapeControl.blk_item
        return blk_item is not None and blk_item.is_editting()

    def onTextBlkItemHoverEnter(self, blk_id: int):
        if self.is_editting():
            return
        blk_item = self.textblk_item_list[blk_id]
        if not blk_item.hasFocus():
            self.txtblkShapeControl.setBlkItem(blk_item)

    def onTextBlkItemMoving(self, item: TextBlkItem):
        self.txtblkShapeControl.updateBoundingRect()

    def onTextBlkItemMoved(self):
        selected_blks = self.canvas.selected_text_items()
        if len(selected_blks) > 0:
            self.canvas.push_undo_command(MoveBlkItemsCommand(selected_blks, self.txtblkShapeControl))
        
    def onTextBlkItemReshaped(self, item: TextBlkItem):
        self.canvas.push_undo_command(ReshapeItemCommand(item))

    def onTextBlkItemRotated(self, new_angle: float):
        blk_item = self.txtblkShapeControl.blk_item
        if blk_item:
            self.canvas.push_undo_command(RotateItemCommand(blk_item, new_angle, self.txtblkShapeControl))

    def onTextBlkItemWarped(self, before: dict, after: dict):
        """PR #1105: Push undo command for quad warp change."""
        item = self.sender()
        if isinstance(item, TextBlkItem):
            self.canvas.push_undo_command(WarpItemCommand(item, before, after, self.txtblkShapeControl))

    def onDeleteBlkItems(self, mode: int):
        selected_blks = self.canvas.selected_text_items()
        if len(selected_blks) == 0 and self.txtblkShapeControl.blk_item is not None:
            selected_blks.append(self.txtblkShapeControl.blk_item)
        if len(selected_blks) > 0:
            self.canvas.push_undo_command(DeleteBlkItemsCommand(selected_blks, mode, self))

    def onCopyBlkItems(self):
        selected_blks = self.canvas.selected_text_items()
        if len(selected_blks) == 0 and self.txtblkShapeControl.blk_item is not None:
            selected_blks.append(self.txtblkShapeControl.blk_item)

        if len(selected_blks) == 0:            
            return

        self.canvas.clipboard_blks.clear()
        if self.canvas.text_change_unsaved():
            self.updateTextBlkList()

        pos = selected_blks[0].blk.bounding_rect()
        pos_x = int(pos[0] + pos[2] / 2)
        pos_y = int(pos[1] + pos[3] / 2)

        textlist = []
        for blkitem in selected_blks:
            blk = copy.deepcopy(blkitem.blk)
            blk.adjust_pos(-pos_x, -pos_y)
            self.canvas.clipboard_blks.append(blk)
            textlist.append(blkitem.toPlainText().strip())
        textlist = '\n'.join(textlist)
        self.app_clipborad.setText(textlist, QClipboard.Mode.Clipboard)


    def onPasteBlkItems(self, pos: QPointF):
        if pos is None:
            pos_x, pos_y = 0, 0
        else:
            pos_x, pos_y = pos.x(), pos.y()
            pos_x = int(pos_x / self.canvas.scale_factor)
            pos_y = int(pos_y / self.canvas.scale_factor)
        blkitem_list, pair_widget_list = [], []
        for blk in self.canvas.clipboard_blks:
            blk = copy.deepcopy(blk)
            blk.adjust_pos(pos_x, pos_y)
            blkitem = self.addTextBlock(blk)
            pairw = self.pairwidget_list[-1]
            blkitem_list.append(blkitem)
            pair_widget_list.append(pairw)
        if len(blkitem_list) > 0:
            self.canvas.clearSelection()
            self.canvas.push_undo_command(PasteBlkItemsCommand(blkitem_list, pair_widget_list, self))
            if len(blkitem_list) == 1:
                self.formatpanel.set_textblk_item(blkitem_list[0])
            else:
                self.formatpanel.set_textblk_item(multi_select=True)

    def onFormatTextblks(self, fmt: FontFormat = None):
        if fmt is None:
            fmt = self.formatpanel.global_format
        self.apply_fontformat(fmt)

    def onAutoLayoutTextblks(self):
        selected_blks = self.canvas.selected_text_items()
        old_html_lst, old_rect_lst, trans_widget_lst = [], [], []
        selected_blks = [blk for blk in selected_blks if not blk.fontformat.vertical]
        if len(selected_blks) > 0:
            for blkitem in selected_blks:
                old_html_lst.append(blkitem.toHtml())
                old_rect_lst.append(blkitem.absBoundingRect(qrect=True))
                trans_widget_lst.append(self.pairwidget_list[blkitem.idx].e_trans)
                self.layout_textblk(blkitem)

            self.canvas.push_undo_command(AutoLayoutCommand(selected_blks, old_rect_lst, old_html_lst, trans_widget_lst))

    def onAutoFitFontToBox(self):
        """Re-run auto fit font size for selected blocks (e.g. after changing font). Forces fit-to-box for selection."""
        from utils.config import pcfg
        selected_blks = self.canvas.selected_text_items()
        selected_blks = [blk for blk in selected_blks if not blk.fontformat.vertical]
        if not selected_blks:
            return
        old_fntsize = getattr(pcfg, 'let_fntsize_flag', 0)
        old_autolayout = getattr(pcfg, 'let_autolayout_flag', True)
        old_auto_flag = self.auto_textlayout_flag
        try:
            pcfg.let_fntsize_flag = 0
            pcfg.let_autolayout_flag = True
            self.auto_textlayout_flag = True
            old_html_lst, old_rect_lst, trans_widget_lst = [], [], []
            for blkitem in selected_blks:
                old_html_lst.append(blkitem.toHtml())
                old_rect_lst.append(blkitem.absBoundingRect(qrect=True))
                trans_widget_lst.append(self.pairwidget_list[blkitem.idx].e_trans)
                self.layout_textblk(blkitem)
            self.canvas.push_undo_command(AutoLayoutCommand(selected_blks, old_rect_lst, old_html_lst, trans_widget_lst))
        finally:
            pcfg.let_fntsize_flag = old_fntsize
            pcfg.let_autolayout_flag = old_autolayout
            self.auto_textlayout_flag = old_auto_flag

    def onResetAngle(self):
        selected_blks = self.canvas.selected_text_items()
        if len(selected_blks) > 0:
            self.canvas.push_undo_command(ResetAngleCommand(selected_blks, self.txtblkShapeControl))

    def onSqueezeBlk(self):
        selected_blks = self.canvas.selected_text_items()
        if len(selected_blks) > 0:
            self.canvas.push_undo_command(SqueezeCommand(selected_blks, self.txtblkShapeControl))

    def on_incanvas_selection_changed(self):
        if self.canvas.textEditMode():
            textitems = self.canvas.selected_text_items()
            self.textEditList.set_selected_list([t.idx for t in textitems])
            if len(textitems) == 1:
                self.formatpanel.set_textblk_item(textitems[-1])
            else:
                self.formatpanel.set_textblk_item(multi_select=bool(textitems))

    def layout_textblk(self, blkitem: TextBlkItem, text: str = None, mask: np.ndarray = None, bounding_rect: List = None, region_rect: List = None):
        
        '''
        auto text layout, vertical writing is not supported yet.
        '''
        is_osb = (getattr(blkitem.blk, "label", None) or "").strip().lower() in OSB_LABELS
        if is_osb:
            setattr(blkitem.blk, "restore_original_region", False)

        img = self.imgtrans_proj.img_array
        if img is None:
            return
        im_h, im_w = img.shape[:2]
        img_area = im_h * im_w

        src_is_cjk = is_cjk(pcfg.module.translate_source)
        tgt_is_cjk = is_cjk(pcfg.module.translate_target)

        # disable for vertical writing
        if blkitem.blk.vertical:
            return
        
        old_br = blkitem.absBoundingRect(qrect=True)
        old_br = [old_br.x(), old_br.y(), old_br.width(), old_br.height()]
        if old_br[2] < 1:
            return

        blk_font = blkitem.font()
        fmt = blkitem.get_fontformat()
        blk_font.setLetterSpacing(QFont.SpacingType.PercentageSpacing, fmt.letter_spacing * 100)
        text_size_func = lambda text: get_text_size(QFontMetricsF(blk_font), text)

        restore_charfmts = False
        if text is None:
            text = blkitem.toPlainText()
            restore_charfmts = True

        if not text.strip():
            return

        original_block_area = None  # used later to cap resize for huge blocks
        if mask is None:
            bounding_rect = blkitem.absBoundingRect(max_h=im_h, max_w=im_w)
            if bounding_rect[2] <= 0 or bounding_rect[3] <= 0:
                blkitem.setPlainText(text)
                if len(self.pairwidget_list) > blkitem.idx:
                    self.pairwidget_list[blkitem.idx].e_trans.setPlainText(text)
                return
            # Constrain only clearly abnormal blocks (>50% image) so normal bubbles keep good size
            block_area = bounding_rect[2] * bounding_rect[3]
            original_block_area = block_area
            if img_area > 0 and block_area > 0.5 * img_area:
                max_w = max(80, int(0.5 * im_w))
                max_h = max(60, int(0.5 * im_h))
                cx = bounding_rect[0] + bounding_rect[2] / 2
                cy = bounding_rect[1] + bounding_rect[3] / 2
                bounding_rect = [
                    int(cx - max_w / 2),
                    int(cy - max_h / 2),
                    max_w,
                    max_h,
                ]
                bounding_rect[0] = max(0, min(bounding_rect[0], im_w - max_w))
                bounding_rect[1] = max(0, min(bounding_rect[1], im_h - max_h))
            if tgt_is_cjk:
                max_enlarge_ratio = 2.5
            else:
                max_enlarge_ratio = 3
            enlarge_ratio = min(max(bounding_rect[2] / bounding_rect[3], bounding_rect[3] / bounding_rect[2]) * 1.5, max_enlarge_ratio)
            mask, ballon_area, mask_xyxy, region_rect = extract_ballon_region(img, bounding_rect, enlarge_ratio=enlarge_ratio, cal_region_rect=True)
            # Shrink text region strictly so text stays well inside bubble (no overflow/clipping)
            if region_rect is not None and len(region_rect) >= 4:
                rw, rh = region_rect[2], region_rect[3]
                if rw > 0 and rh > 0:
                    ar = min(rw, rh) / max(rw, rh)
                    if ar >= 0.5:  # round or oval: strict inset so text stays inside curve
                        inset = 0.72  # 28% shrink
                    else:  # elongated: margin so text doesn't touch edges
                        inset = 0.85  # 15% shrink
                    new_w, new_h = rw * inset, rh * inset
                    region_rect = [
                        region_rect[0] + (rw - new_w) / 2,
                        region_rect[1] + (rh - new_h) / 2,
                        new_w,
                        new_h,
                    ]
        else:
            mask_xyxy = [bounding_rect[0], bounding_rect[1], bounding_rect[0]+bounding_rect[2], bounding_rect[1]+bounding_rect[3]]
        
        words, delimiter = seg_text(text, pcfg.module.translate_target)
        if len(words) < 1:
            return

        wl_list = get_words_length_list(QFontMetricsF(blk_font), words)
        text_w, text_h = text_size_func(text)
        text_area = text_w * text_h
        if tgt_is_cjk:
            line_height = int(round(fmt.line_spacing * text_size_func('X木')[1]))
        else:
            line_height = int(round(fmt.line_spacing * text_size_func('X')[1]))
        delimiter_len = text_size_func(delimiter)[0]
 
        ref_src_lines = False
        if not blkitem.blk.src_is_vertical:
            ref_src_lines = blkitem.blk.line_coord_valid(old_br)

        adaptive_fntsize = False
        resize_ratio = 1
        force_fit = getattr(fmt, 'auto_fit_font_size', False)
        if (self.auto_textlayout_flag and pcfg.let_fntsize_flag == 0 and pcfg.let_autolayout_flag) or force_fit:
            if blkitem.blk.src_is_vertical and blkitem.blk.vertical != blkitem.blk.src_is_vertical:
                adaptive_fntsize = True
                area_ratio = ballon_area / text_area
                ballon_area_thresh = 1.7
                downscale_constraint = 0.6
                resize_ratio = np.clip(min(area_ratio / ballon_area_thresh, region_rect [2] / max(wl_list)), downscale_constraint, 1.0)

            else:
                if not src_is_cjk:
                    # Stricter scale-down so text stays inside bubble with margin (avoid overflow)
                    resize_ratio_ballon = max(ballon_area / 1.70 / text_area, 0.4)
                    if ref_src_lines:
                        _, src_width = blkitem.blk.normalizd_width_list(normalize=False)
                        resize_ratio_src = src_width / (sum(wl_list) + max((len(wl_list) - 1 - len(blkitem.blk.lines_array())), 0) * delimiter_len)
                        resize_ratio = min(resize_ratio_ballon, resize_ratio_src)
                    else:
                        resize_ratio = resize_ratio_ballon
                elif not blkitem.blk.src_is_vertical and ref_src_lines:
                    _, src_width = blkitem.blk.normalizd_width_list(normalize=False)
                    resize_ratio_src = src_width / (sum(wl_list) + max((len(wl_list) - 1 - len(blkitem.blk.lines_array())), 0) * delimiter_len)
                    resize_ratio = max(resize_ratio_src * 1.5, 0.4)
                # Minimum scale so text fits but isn't too small; only cap for truly huge blocks
                resize_ratio = min(max(resize_ratio, 0.5), 1.0)
                area_for_cap = (original_block_area if original_block_area is not None else bounding_rect[2] * bounding_rect[3])
                if img_area > 0 and area_for_cap > 0.5 * img_area:
                    resize_ratio = min(resize_ratio, 0.5)
                # When region is much larger than text: scale UP so short text in big bubbles is readable (was: cap only)
                if region_rect is not None and len(region_rect) >= 4 and text_area > 0:
                    region_area = region_rect[2] * region_rect[3]
                    if region_area > 2.5 * text_area:
                        scale_up = np.sqrt(LAYOUT_SCALE_UP_FILL * region_area / text_area)
                        scale_up_max = LAYOUT_SCALE_UP_MAX * (1.15 if getattr(pcfg.module, "optimize_line_breaks", False) else 1.0)
                        scale_up = min(scale_up, scale_up_max)
                        resize_ratio = max(resize_ratio, min(scale_up, scale_up_max))

        if resize_ratio != 1:
            new_font_size = max(1.0, blk_font.pointSizeF() * resize_ratio)
            new_font_size = max(new_font_size, LAYOUT_MIN_FONT_PT)
            blk_font.setPointSizeF(new_font_size)
            wl_list = (np.array(wl_list, np.float64) * resize_ratio).astype(np.int32).tolist()
            line_height = int(line_height * resize_ratio)
            text_w = int(text_w * resize_ratio)
            delimiter_len = int(delimiter_len * resize_ratio)

        max_central_width = np.inf
        if fmt.alignment == 1:
            if len(blkitem.blk) > 0:
                centroid = blkitem.blk.center().astype(np.int64).tolist()
                centroid[0] -= mask_xyxy[0]
                centroid[1] -= mask_xyxy[1]
            else:
                centroid = [bounding_rect[2] // 2, bounding_rect[3] // 2]
        else:
            max_central_width = np.inf
            centroid = [0, 0]
            abs_centroid = [bounding_rect[0], bounding_rect[1]]
            if len(blkitem.blk) > 0:
                blkitem.blk.lines[0]
                abs_centroid = blkitem.blk.lines[0][0]
                centroid[0] = int(abs_centroid[0] - mask_xyxy[0])
                centroid[1] = int(abs_centroid[1] - mask_xyxy[1])

        # Optional: DP optimal breaks + hyphenation for non-CJK text (improves ugly wraps)
        forced_lines = None
        forced_wl_lines = None
        try:
            if (
                not tgt_is_cjk
                and getattr(pcfg.module, "layout_optimal_breaks", True)
                and region_rect is not None
                and len(region_rect) >= 4
            ):
                maxw_px = int(max(32, min(region_rect[2], bounding_rect[2]) * 0.95))

                def measure_token(s: str) -> int:
                    return int(ffmt.horizontalAdvance(s))

                expanded_words = []
                expanded_wl = []
                for w0 in words:
                    parts = split_long_token_with_hyphenation(
                        w0,
                        measure=measure_token,
                        max_width=maxw_px,
                        hyphenate=bool(getattr(pcfg.module, "layout_hyphenation", True)),
                    )
                    for p, pw in parts:
                        expanded_words.append(p)
                        expanded_wl.append(int(pw))
                if expanded_words:
                    words = expanded_words
                    wl_list = expanded_wl
                ends = find_optimal_breaks_dp(wl_list, max_width=maxw_px, delimiter_width=int(delimiter_len))
                forced_lines = []
                forced_wl_lines = []
                start = 0
                for e in ends:
                    forced_lines.append(words[start:e])
                    forced_wl_lines.append(wl_list[start:e])
                    start = e
        except Exception:
            forced_lines = None
            forced_wl_lines = None

        try:
            new_text, xywh, start_from_top, adjust_xy = layout_text(
            blkitem.blk,
            mask, 
            mask_xyxy, 
            centroid, 
            words, 
            wl_list, 
            delimiter, 
            delimiter_len, 
            line_height, 
            0, 
            max_central_width,
            src_is_cjk=src_is_cjk,
            tgt_is_cjk=tgt_is_cjk,
            ref_src_lines=ref_src_lines,
            forced_lines=forced_lines,
            forced_wl_lines=forced_wl_lines,
            collision_check=bool(getattr(pcfg.module, "layout_collision_check", True)),
            collision_min_mask_ratio=float(getattr(pcfg.module, "layout_collision_min_mask_ratio", 0.85) or 0.85),
            collision_max_retries=int(getattr(pcfg.module, "layout_collision_max_retries", 3) or 3),
        )
            if not (new_text and new_text.strip()):
                return

            # font size post adjustment: force text to stay inside bubble/region
            post_resize_ratio = 1
            if adaptive_fntsize:
                downscale_constraint = 0.5
                w = xywh[2]
                post_resize_ratio = np.clip(max(region_rect[2] / w, downscale_constraint), 0, 1)
                resize_ratio *= post_resize_ratio
            elif region_rect is not None and len(region_rect) >= 4:
                # If laid-out text overflows region, scale down so it fits inside bubble with margin
                rw, rh = region_rect[2], region_rect[3]
                w, h = xywh[2], xywh[3]
                if rw > 0 and rh > 0 and (w > rw or h > rh):
                    fit_ratio = min(rw / w, rh / h) * LAYOUT_FIT_RATIO
                    fit_ratio = max(fit_ratio, LAYOUT_FIT_RATIO_MIN)
                    post_resize_ratio = fit_ratio
                    resize_ratio *= post_resize_ratio

            if post_resize_ratio != 1:
                cx, cy = xywh[0] + xywh[2] / 2, xywh[1] + xywh[3] / 2
                w, h = xywh[2] * post_resize_ratio, xywh[3] * post_resize_ratio
                xywh = [int(cx - w / 2), int(cy - h / 2), int(w), int(h)]

            if resize_ratio != 1:
                new_font_size = max(1.0, blkitem.font().pointSizeF() * resize_ratio)
                if new_font_size < LAYOUT_MIN_FONT_PT:
                    new_font_size = LAYOUT_MIN_FONT_PT
                    resize_ratio = new_font_size / max(1.0, blkitem.font().pointSizeF())
                blkitem.textCursor().clearSelection()
                blkitem.setFontSize(new_font_size)
                blk_font.setPointSizeF(new_font_size)

            # Center text block inside bubble/region (manga-translator-ui style)
            if getattr(pcfg.module, "center_text_in_bubble", False) and region_rect is not None and len(region_rect) >= 4:
                rx, ry, rw, rh = region_rect[0], region_rect[1], region_rect[2], region_rect[3]
                tw, th = xywh[2], xywh[3]
                cx = rx + rw / 2
                cy = ry + rh / 2
                # xywh in layout_text is mask-local; center position in image coords for setRect below
                xywh[0] = int(cx - tw / 2)
                xywh[1] = int(cy - th / 2)

            if restore_charfmts:
                char_fmts = blkitem.get_char_fmts()        
        
            ffmt = QFontMetricsF(blk_font)
            maxw = max([ffmt.horizontalAdvance(t) for t in new_text.split('\n')])
            # Height: ensure minimum (single-line/descenders) and add bottom padding so last line doesn't sit on edge
            layout_h = max(int(xywh[3]), line_height)
            layout_h = int(layout_h * LAYOUT_HEIGHT_PADDING)
            blkitem.set_size(maxw * LAYOUT_WIDTH_STROKE_FACTOR, layout_h, set_layout_maxsize=True)
            blkitem.setPlainText(new_text)
            blkitem._ensure_transparent_document_background()
            if getattr(pcfg.module, "center_text_in_bubble", False) and region_rect is not None and len(region_rect) >= 4:
                br = blkitem.absBoundingRect(qrect=True)
                blkitem.setRect([
                    xywh[0],
                    xywh[1],
                    br.width(),
                    br.height(),
                ], repaint=True)
            if len(self.pairwidget_list) > blkitem.idx:
                self.pairwidget_list[blkitem.idx].e_trans.setPlainText(new_text)
            if restore_charfmts:
                self.restore_charfmts(blkitem, text, new_text, char_fmts)
            blkitem.squeezeBoundingRect()
            # Clamp block to image bounds so text never draws outside the panel
            abr = blkitem.absBoundingRect(qrect=True)
            x, y, w, h = abr.x(), abr.y(), abr.width(), abr.height()
            if w > 0 and h > 0:
                x2 = max(0, min(x, im_w - w))
                y2 = max(0, min(y, im_h - h))
                if x2 != x or y2 != y:
                    blkitem.setRect([x2, y2, w, h], repaint=True)
            return True
        except Exception:
            if is_osb:
                if getattr(pcfg.module, "osb_layout_fallbacks_enabled", True):
                    try:
                        self._layout_osb_vertical_fallback(blkitem, text, im_w, im_h, blk_font)
                        return
                    except Exception:
                        setattr(blkitem.blk, "restore_original_region", True)
                else:
                    setattr(blkitem.blk, "restore_original_region", True)
            else:
                raise
    
    def _layout_osb_vertical_fallback(self, blkitem: TextBlkItem, text: str, im_w: int, im_h: int, blk_font: QFont) -> None:
        """OSB last-chance fallback: stack lines vertically. Used when normal layout fails."""
        blk = blkitem.blk
        xyxy = getattr(blk, "xyxy", None)
        if not xyxy or len(xyxy) != 4:
            raise ValueError("Block has no valid xyxy")
        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
        x1, x2 = max(0, min(x1, im_w)), max(0, min(x2, im_w))
        y1, y2 = max(0, min(y1, im_h)), max(0, min(y2, im_h))
        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)
        lines = [s.strip() for s in (text or "").split("\n") if s.strip()]
        if not lines:
            return
        ffmt = QFontMetricsF(blk_font)
        line_height = int(round(ffmt.height() * 1.2))
        max_line_w = max(ffmt.horizontalAdvance(ln) for ln in lines)
        total_h = len(lines) * line_height
        scale = 1.0
        if total_h > box_h or max_line_w > box_w:
            scale = min(box_h / max(total_h, 1), box_w / max(max_line_w, 1), 1.0)
            scale = max(scale, 0.4)
        if scale != 1.0:
            new_pt = max(LAYOUT_MIN_FONT_PT, blk_font.pointSizeF() * scale)
            blk_font.setPointSizeF(new_pt)
            ffmt = QFontMetricsF(blk_font)
            line_height = int(round(ffmt.height() * 1.2))
            max_line_w = max(ffmt.horizontalAdvance(ln) for ln in lines)
            total_h = len(lines) * line_height
        new_text = "\n".join(lines)
        layout_w = int(max_line_w * LAYOUT_WIDTH_STROKE_FACTOR)
        layout_h = int(total_h * LAYOUT_HEIGHT_PADDING)
        blkitem.set_size(layout_w, layout_h, set_layout_maxsize=True)
        blkitem.setPlainText(new_text)
        blkitem._ensure_transparent_document_background()
        if len(self.pairwidget_list) > blkitem.idx:
            self.pairwidget_list[blkitem.idx].e_trans.setPlainText(new_text)
        blkitem.squeezeBoundingRect()
    
    def restore_charfmts(self, blkitem: TextBlkItem, text: str, new_text: str, char_fmts: List[QTextCharFormat]):
        cursor = blkitem.textCursor()
        cpos = 0
        num_text = len(new_text)
        num_fmt = len(char_fmts)
        blkitem.layout.relayout_on_changed = False
        blkitem.repaint_on_changed = False
        if num_text >= num_fmt:
            for fmt_i in range(num_fmt):
                fmt = char_fmts[fmt_i]
                ori_char = text[fmt_i].strip()
                if ori_char == '':
                    continue
                else:
                    if cursor.atEnd():   
                        break
                    matched = False
                    while cpos < num_text:
                        if new_text[cpos] == ori_char:
                            matched = True
                            break
                        cpos += 1
                    if matched:
                        cursor.clearSelection()
                        cursor.setPosition(cpos)
                        cursor.setPosition(cpos+1, QTextCursor.MoveMode.KeepAnchor)
                        cursor.setCharFormat(fmt)
                        cursor.setBlockCharFormat(fmt)
                        cpos += 1
        blkitem.repaint_on_changed = True
        blkitem.layout.relayout_on_changed = True
        blkitem.layout.reLayout()
        blkitem.repaint_background()

    def onEndCreateTextBlock(self, rect: QRectF):
        xyxy = np.array([rect.x(), rect.y(), rect.right(), rect.bottom()])        
        xyxy = np.round(xyxy).astype(np.int32)
        block = TextBlock(xyxy)
        xywh = np.copy(xyxy)
        xywh[[2, 3]] -= xywh[[0, 1]]
        block.set_lines_by_xywh(xywh)
        block.src_is_vertical = self.formatpanel.global_format.vertical
        blk_item = TextBlkItem(block, len(self.textblk_item_list), set_format=False, show_rect=True)
        blk_item.set_fontformat(self.formatpanel.global_format)
        self.canvas.push_undo_command(CreateItemCommand(blk_item, self))

    def on_paste2selected_textitems(self):
        blkitems = self.canvas.selected_text_items()
        text = self.app_clipborad.text()

        num_blk = len(blkitems)
        if num_blk < 1:
            return
        
        if num_blk > 1:
            text_list = text.rstrip().split('\n')
            num_text = len(text_list)
            if num_text > 1:
                if num_text > num_blk:
                    text_list = text_list[:num_blk]
                elif num_text < num_blk:
                    text_list = text_list + [text_list[-1]] * (num_blk - num_text)
                text = text_list
        
        etrans = [self.pairwidget_list[blkitem.idx].e_trans for blkitem in blkitems]
        self.canvas.push_undo_command(MultiPasteCommand(text, blkitems, etrans))

    def onRotateTextBlkItem(self, item: TextBlock):
        self.canvas.push_undo_command(RotateItemCommand(item))
    
    def on_transwidget_focus_in(self, idx: int):
        if self.is_editting():
            textitm = self.editingTextItem()
            textitm.endEdit()
            self.pairwidget_list[textitm.idx].e_trans.setHoverEffect(False)
            self.textEditList.clearAllSelected()

        if idx < len(self.textblk_item_list):
            blk_item = self.textblk_item_list[idx]
            sender = self.sender()
            if isinstance(sender, TransTextEdit):
                blk_item.setCacheMode(QGraphicsItem.CacheMode.NoCache)
            self.canvas.gv.ensureVisible(blk_item)
            self.txtblkShapeControl.setBlkItem(blk_item)

    def on_textedit_redo(self):
        self.canvas.redo_textedit()

    def on_textedit_undo(self):
        self.canvas.undo_textedit()

    def on_show_select_menu(self, pos: QPoint, selected_text: str):
        if pcfg.textselect_mini_menu:
            if not selected_text:
                if self.selectext_minimenu.isVisible():
                    self.selectext_minimenu.hide()
            else:
                self.selectext_minimenu.show()
                self.selectext_minimenu.move(self.mainwindow.mapFromGlobal(pos))
                self.selectext_minimenu.selected_text = selected_text

    def on_block_current_editor(self, block: bool):
        w: SourceTextEdit = self.app.focusWidget()
        if isinstance(w, SourceTextEdit) or isinstance(w, TextBlkItem):
            w.block_all_input = block

    def on_pairw_focusout(self, idx: int):
        if self.selectext_minimenu.isVisible():
            self.selectext_minimenu.hide()
        sender = self.sender()
        if isinstance(sender, TransTextEdit) and idx < len(self.textblk_item_list):
            blk_item = self.textblk_item_list[idx]
            blk_item.setCacheMode(QGraphicsItem.CacheMode.DeviceCoordinateCache)

    def on_push_textitem_undostack(self, num_steps: int, is_formatting: bool):
        blkitem: TextBlkItem = self.sender()
        e_trans = self.pairwidget_list[blkitem.idx].e_trans if not is_formatting else None
        self.canvas.push_undo_command(TextItemEditCommand(blkitem, e_trans, num_steps, self.textpanel.formatpanel), update_pushed_step=is_formatting)

    def on_push_edit_stack(self, num_steps: int):
        edit: Union[TransTextEdit, SourceTextEdit] = self.sender()
        is_trans = type(edit) == TransTextEdit
        blkitem = self.textblk_item_list[edit.idx] if is_trans else None
        self.canvas.push_undo_command(TextEditCommand(edit, num_steps, blkitem), update_pushed_step=not is_trans)

    def on_propagate_textitem_edit(self, pos: int, added_text: str, joint_previous: bool):
        blk_item: TextBlkItem = self.sender()
        edit = self.pairwidget_list[blk_item.idx].e_trans
        propagate_user_edit(blk_item, edit, pos, added_text, joint_previous)
        self.canvas.push_text_command(command=None, update_pushed_step=True)

    def on_propagate_transwidget_edit(self, pos: int, added_text: str, joint_previous: bool):
        edit: TransTextEdit = self.sender()
        blk_item = self.textblk_item_list[edit.idx]
        if blk_item.isEditing():
            blk_item.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        propagate_user_edit(edit, blk_item, pos, added_text, joint_previous)
        self.canvas.push_text_command(command=None, update_pushed_step=True)

    def apply_fontformat(self, fontformat: FontFormat):
        selected_blks = self.canvas.selected_text_items()
        trans_widget_list = []
        for blk in selected_blks:
            trans_widget_list.append(self.pairwidget_list[blk.idx].e_trans)
        if len(selected_blks) > 0:
            self.canvas.push_undo_command(ApplyFontformatCommand(selected_blks, trans_widget_list, fontformat))
            if self.formatpanel.global_mode():
                if id(self.formatpanel.active_text_style_format()) != id(fontformat):
                    self.formatpanel.deactivate_style_label()
                self.formatpanel.on_active_textstyle_label_changed()
            else:
                self.formatpanel.set_active_format(fontformat)

    def apply_global_format_to_all_blocks(self):
        """Apply current global font format to every text block on this page."""
        if len(self.textblk_item_list) == 0:
            return
        all_blks = list(self.textblk_item_list)
        trans_widget_list = [self.pairwidget_list[blk.idx].e_trans for blk in all_blks]
        self.canvas.push_undo_command(ApplyFontformatCommand(all_blks, trans_widget_list, self.formatpanel.global_format))
        self.formatpanel.on_active_textstyle_label_changed()

    def on_transwidget_selection_changed(self):
        selitems = self.canvas.selected_text_items()
        selset = {pw.idx: pw for pw in self.textEditList.checked_list}
        self.canvas.block_selection_signal = True
        for blkitem in selitems:
            if blkitem.idx not in selset:
                blkitem.setSelected(False)
            else:
                selset.pop(blkitem.idx)
        for idx in selset:
            self.textblk_item_list[idx].setSelected(True)
        self.canvas.block_selection_signal = False

    def on_textedit_list_focusout(self):
        fw = self.app.focusWidget()
        focusing_edit = isinstance(fw, (SourceTextEdit, TransTextEdit))
        if fw == self.canvas.gv or focusing_edit:
            self.textEditList.clearDrag()
        if focusing_edit:
            self.textEditList.clearAllSelected()

    def on_rearrange_blks(self, mv_map: Tuple[np.ndarray]):
        self.canvas.push_undo_command(RearrangeBlksCommand(mv_map, self))

    def updateTextBlkItemIdx(self, sel_ids: set = None):
        for ii, blk_item in enumerate(self.textblk_item_list):
            if sel_ids is not None and ii not in sel_ids:
                continue
            blk_item.idx = ii
            self.pairwidget_list[ii].updateIndex(ii)
        cl = self.textEditList.checked_list
        if len(cl) != 0:
            cl.sort(key=lambda x: x.idx)

    def updateTextBlkList(self):
        cbl = self.imgtrans_proj.current_block_list()
        if cbl is None:
            return
        cbl.clear()
        for blk_item, trans_pair in zip(self.textblk_item_list, self.pairwidget_list):
            if not blk_item.document().isEmpty():
                blk_item.blk.rich_text = blk_item.toHtml()
                blk_item.blk.translation = blk_item.toPlainText()
            else:
                blk_item.blk.rich_text = ''
                blk_item.blk.translation = ''
            blk_item.blk.text = [trans_pair.e_source.toPlainText()]
            blk_item.blk._bounding_rect = blk_item.absBoundingRect()
            blk_item.updateBlkFormat()
            cbl.append(blk_item.blk)

    def updateTranslation(self):
        for blk_item, transwidget in zip(self.textblk_item_list, self.pairwidget_list):
            transwidget.e_trans.setPlainText(blk_item.blk.translation)
            blk_item.setPlainText(blk_item.blk.translation)
        self.canvas.clear_text_stack()

    def showTextblkItemRect(self, draw_rect: bool):
        for blk_item in self.textblk_item_list:
            blk_item.draw_rect = draw_rect
            blk_item.update()

    def set_blkitems_selection(self, selected: bool, blk_items: List[TextBlkItem] = None):
        self.canvas.block_selection_signal = True
        if blk_items is None:
            blk_items = self.textblk_item_list
        for blk_item in blk_items:
            blk_item.setSelected(selected)
        self.canvas.block_selection_signal = False
        self.on_incanvas_selection_changed()

    def on_ensure_textitem_svisible(self):
        edit: Union[TransTextEdit, SourceTextEdit] = self.sender()
        self.changeHoveringWidget(edit)
        self.canvas.gv.ensureVisible(self.textblk_item_list[edit.idx])
        self.txtblkShapeControl.setBlkItem(self.textblk_item_list[edit.idx])

    def on_page_replace_one(self):
        self.canvas.push_undo_command(PageReplaceOneCommand(self.canvas.search_widget))

    def on_page_replace_all(self):
        self.canvas.push_undo_command(PageReplaceAllCommand(self.canvas.search_widget))

def get_text_size(fm: QFontMetricsF, text: str) -> Tuple[int, int]:
    brt = fm.tightBoundingRect(text)
    br = fm.boundingRect(text)
    return int(np.ceil(fm.horizontalAdvance(text))), int(np.ceil(brt.height()))
    
def get_words_length_list(fm: QFontMetricsF, words: List[str]) -> List[int]:
    return [int(np.ceil(fm.horizontalAdvance(word))) for word in words]

