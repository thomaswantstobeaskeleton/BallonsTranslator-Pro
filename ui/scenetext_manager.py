
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
from utils.fontformat import FontFormat, pt2px
from .textedit_commands import propagate_user_edit, TextEditCommand, ReshapeItemCommand, MoveBlkItemsCommand, AutoLayoutCommand, ApplyFontformatCommand, RotateItemCommand, WarpItemCommand, TextItemEditCommand, TextEditCommand, PageReplaceOneCommand, PageReplaceAllCommand, MultiPasteCommand, ResetAngleCommand, SqueezeCommand
from .text_panel import FontFormatPanel
from utils.config import pcfg
from utils import shared
from utils.imgproc_utils import extract_ballon_region, rotate_polygons, get_block_mask, classify_bubble_shape_from_mask
from utils.bubble_shape_model import get_bubble_shape_from_model as _bubble_shape_from_model_impl
from utils.text_processing import seg_text, is_cjk
from utils.text_layout import layout_text
from utils.line_breaking import split_long_token_with_hyphenation
from modules.textdetector.outside_text_processor import OSB_LABELS

# Layout tuning: keep text inside bubbles and avoid overflow/tiny text (see layout_textblk)
# Defaults favor more lines and smaller base size so text fits without overflowing wide boxes.
LAYOUT_FIT_RATIO = 0.50  # fit text to this fraction of region (use more of bubble width/height)
LAYOUT_FIT_RATIO_MIN = 0.15  # minimum scale when fitting (allow smaller text when bubble is crowded)
LAYOUT_MIN_FONT_PT = 10.0  # fallback minimum font size; config layout_font_size_min/max override (2.1).
LAYOUT_HEIGHT_PADDING = 1.12  # multiplier for block height (bottom margin; prevent last-line cropping)
LAYOUT_WIDTH_STROKE_FACTOR = 1.18  # width = max line width * this (stroke/outline clearance; lower = box follows content more)
LAYOUT_SCALE_UP_MAX = 1  # cap scale-up when bubble is large and text is short
LAYOUT_SCALE_UP_FILL = 0.68  # target fill ratio when scaling up in large bubbles
# Scale down final font so text is smaller and fits in more lines (narrower effective width).
LAYOUT_TEXT_SCALE = 0.68  # multiply final font size by this (smaller base = more lines in wide boxes)
LAYOUT_TEXT_SCALE_LONG = 0.58  # stronger scale-down for long paragraphs
LAYOUT_TEXT_SCALE_LONG_MIN_LINES = 3  # use LAYOUT_TEXT_SCALE_LONG when at least this many lines
LAYOUT_TEXT_SCALE_LONG_MIN_CHARS = 80  # or when at least this many characters
LAYOUT_MAX_LINE_WIDTH_FRAC = 0.92  # max line width = bubble width * this (higher = fuller lines, text fills box more)


def _aspect_ratio_shape(rw: int, rh: int) -> str:
    """Aspect-ratio heuristic: no mask/model."""
    if rw <= 0 or rh <= 0:
        return "round"
    ar = min(rw, rh) / max(rw, rh)
    return "round" if ar >= 0.5 else ("narrow" if rh > rw * 1.4 else "elongated")


def _resolve_auto_balloon_shape(
    method: str,
    mask: np.ndarray,
    mask_xyxy: list,
    img: np.ndarray,
    rw: int,
    rh: int,
) -> str:
    """
    Resolve balloon shape when layout_balloon_shape is 'auto'.
    method: one of aspect_ratio, contour, model, model_contour, model_ratio, contour_ratio, model_contour_ratio.
    Returns first successful shape from the chosen chain.
    """
    method = (method or "contour_ratio").strip().lower()
    model_id = (getattr(pcfg.module, "layout_balloon_shape_model_id", None) or "").strip()

    def try_model() -> str:
        if not model_id:
            return None
        return _bubble_shape_from_model_impl(mask, mask_xyxy, img, model_id)

    def try_contour() -> str:
        return classify_bubble_shape_from_mask(mask) if mask is not None and mask.size > 0 else None

    def try_ratio() -> str:
        return _aspect_ratio_shape(rw, rh)

    steps = []
    if method == "aspect_ratio":
        steps = ["ratio"]
    elif method == "contour":
        steps = ["contour"]
    elif method == "model":
        steps = ["model"]
    elif method == "model_contour":
        steps = ["model", "contour"]
    elif method == "model_ratio":
        steps = ["model", "ratio"]
    elif method == "contour_ratio":
        steps = ["contour", "ratio"]
    else:
        steps = ["model", "contour", "ratio"]

    for step in steps:
        if step == "model":
            s = try_model()
        elif step == "contour":
            s = try_contour()
        else:
            s = try_ratio()
        if s:
            return s
    return try_ratio()


def _word_count(line: str) -> int:
    """Number of non-empty tokens after split (strip each so 'the ' or ' the ' counts as 1)."""
    return len([w for w in line.split() if w.strip()])


def _merge_stub_lines_in_text(text: str) -> str:
    """Merge short lines so 'the', 'and', 'Realm.' never sit alone. First all 1-word lines, then 2-5 word stubs."""
    if not text or not text.strip():
        return text
    prev = ""
    while prev != text:
        prev = text
        lines = [ln.strip() for ln in text.splitlines() if ln is not None and ln.strip()]
        if len(lines) <= 1:
            return text
        # Eliminate every 0- or 1-word line (repeat until none left)
        while True:
            merged = []
            i = 0
            any_one = False
            while i < len(lines):
                line = lines[i]
                nw = _word_count(line)
                if nw <= 1 and i + 1 < len(lines):
                    merged.append((line + " " + lines[i + 1]).strip())
                    i += 2
                    any_one = True
                    continue
                if nw <= 1 and i == len(lines) - 1 and merged:
                    merged[-1] = (merged[-1] + " " + line).strip()
                    i += 1
                    any_one = True
                    continue
                merged.append(line)
                i += 1
            if not any_one:
                break
            lines = merged
            if len(lines) <= 1:
                break
        # Merge 2-5 word stubs
        merged = []
        i = 0
        while i < len(lines):
            line = lines[i]
            nw = _word_count(line)
            if nw <= 5 and i + 1 < len(lines):
                merged.append((line + " " + lines[i + 1]).strip())
                i += 2
                continue
            if nw <= 5 and i == len(lines) - 1 and merged:
                merged[-1] = (merged[-1] + " " + line).strip()
                i += 1
                continue
            merged.append(line)
            i += 1
        text = "\n".join(merged)
    # Final pass: merge any remaining 0-4 word line until none left (catches 'stage.', 'in', 'the', etc.)
    while True:
        lines = [ln.strip() for ln in text.splitlines() if ln is not None and ln.strip()]
        if len(lines) <= 1:
            break
        out = []
        i = 0
        any_stub = False
        while i < len(lines):
            line = lines[i]
            nw = _word_count(line)
            if nw <= 4 and i + 1 < len(lines):
                out.append((line + " " + lines[i + 1]).strip())
                i += 2
                any_stub = True
                continue
            if nw <= 4 and i == len(lines) - 1 and out:
                out[-1] = (out[-1] + " " + line).strip()
                i += 1
                any_stub = True
                continue
            out.append(line)
            i += 1
        if not any_stub:
            break
        text = "\n".join(out)
    # Last resort: merge any remaining 0-1 word line until none (ensures 'the' after 'years of' never stays alone)
    while True:
        lines = [ln.strip() for ln in text.splitlines() if ln is not None and ln.strip()]
        if len(lines) <= 1:
            break
        out = []
        i = 0
        any_one = False
        while i < len(lines):
            line = lines[i]
            nw = _word_count(line)
            if nw <= 1 and i + 1 < len(lines):
                out.append((line + " " + lines[i + 1]).strip())
                i += 2
                any_one = True
                continue
            if nw <= 1 and i == len(lines) - 1 and out:
                out[-1] = (out[-1] + " " + line).strip()
                i += 1
                any_one = True
                continue
            out.append(line)
            i += 1
        if not any_one:
            break
        text = "\n".join(out)
    return text


class CreateItemCommand(QUndoCommand):
    def __init__(self, blk_item: TextBlkItem, ctrl, page_name: str = None, mask_backup: tuple = None, parent=None):
        super().__init__(parent)
        self.blk_item = blk_item
        self.ctrl: SceneTextManager = ctrl
        self.op_count = -1
        self.page_name = page_name
        self.mask_backup = mask_backup  # (x1, y1, x2, y2, region_array) to restore mask on undo
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
        if self.page_name and self.ctrl.canvas.imgtrans_proj.current_img == self.page_name:
            pages = self.ctrl.canvas.imgtrans_proj.pages.get(self.page_name, [])
            blk = self.blk_item.blk
            for i in range(len(pages)):
                if pages[i] is blk:
                    pages.pop(i)
                    break
            if self.mask_backup is not None:
                x1, y1, x2, y2, region = self.mask_backup
                mask = self.ctrl.canvas.imgtrans_proj.load_mask_by_imgname(self.page_name)
                if mask is not None and 0 <= y1 < y2 <= mask.shape[0] and 0 <= x1 < x2 <= mask.shape[1]:
                    mask[y1:y2, x1:x2] = region
                    self.ctrl.canvas.imgtrans_proj.save_mask(self.page_name, mask)
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
        self._moving_blur_item: TextBlkItem = None  # reserved for optional drag effect cleanup

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
        # Apply auto layout when loading any page so layout persists after reopen (config-driven)
        self.auto_textlayout_flag = (
            pcfg.let_autolayout_flag
            and (pcfg.let_fntsize_flag == 0)
            and (pcfg.module.enable_detect or pcfg.module.enable_ocr or pcfg.module.enable_translate)
        )
        for textblock in self.imgtrans_proj.current_block_list():
            if textblock.font_family is None or textblock.font_family.strip() == '':
                textblock.font_family = self.formatpanel.familybox.currentText()
            blk_item = self.addTextBlock(textblock)
        if self.auto_textlayout_flag:
            self.updateTextBlkList()
            # Auto layout on page (re)load changes text geometry; mark project as dirty
            # so page switches / project close will persist the improved layout.
            self.canvas.setProjSaveState(True)

    def addTextBlock(self, blk: Union[TextBlock, TextBlkItem] = None) -> TextBlkItem:
        if isinstance(blk, TextBlkItem):
            blk_item = blk
            blk_item.idx = len(self.textblk_item_list)
        else:
            # Apply global stroke default to new blocks when they have no stroke (#1143)
            if blk.fontformat.stroke_width == 0 and getattr(pcfg.global_fontformat, 'stroke_width', 0) > 0:
                blk.fontformat.stroke_width = pcfg.global_fontformat.stroke_width
                blk.fontformat.srgb = list(getattr(pcfg.global_fontformat, 'srgb', [0, 0, 0]))
            # Apply global text box corner radius default (shape) when present: 0 = rectangle, >0 = rounded / circle-like.
            gb_radius = float(getattr(pcfg.global_fontformat, "text_box_corner_radius", 0.0) or 0.0)
            if gb_radius > 0 and getattr(blk.fontformat, "text_box_corner_radius", 0.0) == 0.0:
                blk.fontformat.text_box_corner_radius = gb_radius
            translation = ''
            if self.auto_textlayout_flag and not blk.vertical:
                translation = blk.translation
                blk.translation = ''
            blk_item = TextBlkItem(blk, len(self.textblk_item_list), show_rect=self.canvas.textblock_mode)
            if translation:
                blk.translation = translation
                # First pass: initial auto layout (stub/short-line penalties apply)
                rst = self.layout_textblk(blk_item, text=translation)
                if rst is None:
                    blk_item.setPlainText(_merge_stub_lines_in_text(translation))
                else:
                    # Second pass: re-layout using updated geometry; same scoring so penalties apply again
                    self.layout_textblk(blk_item)
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
        # Skip QGraphicsBlurEffect during drag to avoid "A paint device can only be painted by one painter at a time" / Painter not active spam
        self.txtblkShapeControl.updateBoundingRect()

    def onTextBlkItemMoved(self):
        if self._moving_blur_item is not None:
            self._moving_blur_item.setGraphicsEffect(None)
            self._moving_blur_item = None
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

    def run_auto_layout_on_current_page_once(self):
        """Run auto layout on all blocks of the current page (e.g. after opening project so first page is formatted)."""
        from utils.config import pcfg
        all_blks = [b for b in self.textblk_item_list if not getattr(b.fontformat, 'vertical', False)]
        if not all_blks:
            return
        old_autolayout = getattr(pcfg, 'let_autolayout_flag', True)
        old_flag = self.auto_textlayout_flag
        try:
            pcfg.let_autolayout_flag = True
            self.auto_textlayout_flag = True
            for blkitem in all_blks:
                self.layout_textblk(blkitem)
                self.layout_textblk(blkitem)
            for blkitem in all_blks:
                if blkitem.idx < len(self.pairwidget_list):
                    merged = _merge_stub_lines_in_text(blkitem.toPlainText())
                    blkitem.setPlainText(merged)
                    self.pairwidget_list[blkitem.idx].e_trans.setPlainText(merged)
            self.canvas.setProjSaveState(True)
        finally:
            pcfg.let_autolayout_flag = old_autolayout
            self.auto_textlayout_flag = old_flag

    def onAutoLayoutTextblks(self):
        selected_blks = self.canvas.selected_text_items()
        old_html_lst, old_rect_lst, trans_widget_lst = [], [], []
        selected_blks = [blk for blk in selected_blks if not blk.fontformat.vertical]
        if len(selected_blks) > 0:
            for blkitem in selected_blks:
                old_html_lst.append(blkitem.toHtml())
                old_rect_lst.append(blkitem.absBoundingRect(qrect=True))
                trans_widget_lst.append(self.pairwidget_list[blkitem.idx].e_trans)
                # Run auto layout twice: second pass uses the new box geometry to refine line breaks.
                self.layout_textblk(blkitem)
                self.layout_textblk(blkitem)

            self.canvas.push_undo_command(AutoLayoutCommand(selected_blks, old_rect_lst, old_html_lst, trans_widget_lst))

    def onAutoFitFontToBox(self):
        """Run layout (stub-aware line breaks) then scale font so text fits the selected box(es). Avoids Qt reflow introducing single-word lines."""
        selected_blks = self.canvas.selected_text_items()
        selected_blks = [blk for blk in selected_blks if not blk.fontformat.vertical]
        if not selected_blks:
            return
        old_html_lst, old_rect_lst, trans_widget_lst = [], [], []
        for blkitem in selected_blks:
            old_html_lst.append(blkitem.toHtml())
            old_rect_lst.append(blkitem.absBoundingRect(qrect=True))
            trans_widget_lst.append(self.pairwidget_list[blkitem.idx].e_trans)
            # Establish line breaks with our layout (stub penalties) before scaling, so scaling doesn't reflow into stubs.
            self.layout_textblk(blkitem)
            self.layout_textblk(blkitem)
            self._scale_font_to_fit_box(blkitem)
        self.canvas.push_undo_command(AutoLayoutCommand(selected_blks, old_rect_lst, old_html_lst, trans_widget_lst))

    def _scale_font_to_fit_box(self, blkitem: TextBlkItem):
        """Scale this block's font so its text fits inside the current box (no layout/box change)."""
        br = blkitem.absBoundingRect(qrect=True)
        box_w = br.width()
        box_h = br.height()
        if box_w < 1 or box_h < 1:
            return
        doc = blkitem.document()
        if doc is None:
            return
        doc_w = doc.size().width()
        doc_h = doc.size().height()
        if doc_w < 1 and doc_h < 1:
            return
        if doc_w < 1:
            doc_w = 1
        if doc_h < 1:
            doc_h = 1
        scale = min(box_w / doc_w, box_h / doc_h)
        scale = max(0.25, min(scale, 2.0))
        if abs(scale - 1.0) < 0.01:
            return
        blkitem.setRelFontSize(scale, repaint_background=True)
        if blkitem.blk is not None and hasattr(blkitem.blk, 'fontformat'):
            try:
                new_pt = blkitem.font().pointSizeF()
                if new_pt > 0:
                    blkitem.blk.fontformat.font_size = pt2px(new_pt)
            except Exception:
                pass
        self.canvas.setProjSaveState(True)

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
        Auto text layout (vertical writing not supported).
        Layout tuning is read from pcfg.module at runtime: layout_optimal_breaks, layout_hyphenation,
        layout_short_line_penalty, layout_height_overflow_penalty, optimize_line_breaks,
        layout_constrain_to_bubble, layout_collision_*. Font scaling runs when Auto layout is on
        and Font size = "decide by program" (or block has auto_fit_font_size).
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
        if blk_font.pointSizeF() <= 0 or blk_font.pointSize() <= 0:
            blk_font.setPointSizeF(10.0)
            blk_font.setPointSize(10)
        # Enforce a minimum readable font size whenever auto layout / auto-fit is driving font changes.
        # This prevents tiny (e.g. 3–4 pt) text for short lines like "That is also good.".
        if (
            ((self.auto_textlayout_flag and pcfg.let_fntsize_flag == 0 and pcfg.let_autolayout_flag)
             or getattr(blkitem.fontformat, 'auto_fit_font_size', False))
            and blk_font.pointSizeF() < LAYOUT_MIN_FONT_PT
        ):
            blk_font.setPointSizeF(LAYOUT_MIN_FONT_PT)
            blk_font.setPointSize(int(max(LAYOUT_MIN_FONT_PT, blk_font.pointSize() or 0)))
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
                text = _merge_stub_lines_in_text(text)
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
            # Diamond-Text style: shape-aware insets (round vs elongated vs narrow). Shrink text region so text stays inside bubble.
            if region_rect is not None and len(region_rect) >= 4:
                rw, rh = region_rect[2], region_rect[3]
                if rw > 0 and rh > 0:
                    shape_cfg = getattr(pcfg.module, "layout_balloon_shape", "auto") or "auto"
                    if shape_cfg == "auto":
                        balloon_shape = _resolve_auto_balloon_shape(
                            getattr(pcfg.module, "layout_balloon_shape_auto_method", "contour_ratio"),
                            mask, mask_xyxy, img, rw, rh,
                        )
                    else:
                        balloon_shape = shape_cfg
                    # Diamond-Text shapes: insets so text stays inside bubble (round, elongated, narrow, diamond, square, bevel, pentagon, point)
                    if balloon_shape == "round":
                        inset = 0.96  # 4% margin
                    elif balloon_shape == "narrow" or balloon_shape == "point":
                        inset = 0.97  # 3% for tall/narrow or pointed tail
                    elif balloon_shape == "diamond":
                        inset = 0.95  # 5% for pointy corners
                    elif balloon_shape in ("square", "bevel", "pentagon"):
                        inset = 0.96  # 4%
                    else:
                        inset = 0.98  # 2% for elongated
                    new_w, new_h = rw * inset, rh * inset
                    region_rect = [
                        region_rect[0] + (rw - new_w) / 2,
                        region_rect[1] + (rh - new_h) / 2,
                        new_w,
                        new_h,
                    ]
                else:
                    balloon_shape = "auto"
            else:
                balloon_shape = getattr(pcfg.module, "layout_balloon_shape", "auto") or "auto"
        else:
            mask_xyxy = [bounding_rect[0], bounding_rect[1], bounding_rect[0]+bounding_rect[2], bounding_rect[1]+bounding_rect[3]]
            balloon_shape = "auto"

        words, delimiter = seg_text(text, pcfg.module.translate_target)
        if len(words) < 1:
            return

        # Special-case very short English bubbles so they stay on a single line and expand horizontally.
        # Examples: "Mom", "Sing", "remind", "Short message", "TSK,".
        if not tgt_is_cjk:
            clean_chars = ''.join(ch for ch in text if ch.isalnum())
            if 1 <= len(clean_chars) <= 16 and len(words) <= 4:
                single = text.replace('\n', ' ').strip()
                if single:
                    fshort = QFontMetricsF(blk_font)
                    base_w = fshort.horizontalAdvance(single)
                    base_h = int(round(fmt.line_spacing * fshort.height()))
                    if base_w > 0 and base_h > 0:
                        # Prefer to fit inside bubble horizontally/vertically if region is known.
                        if region_rect is not None and len(region_rect) >= 4:
                            avail_w = max(8, region_rect[2] * 0.9)
                            avail_h = max(8, region_rect[3] * 0.9)
                        else:
                            avail_w = max(8, bounding_rect[2] * 0.9)
                            avail_h = max(8, bounding_rect[3] * 0.9)
                        scale = min(avail_w / base_w, avail_h / base_h, 1.0)
                        # Allow fairly small text for tiny bubbles, but not below minimum readable size.
                        scale = max(scale, 0.25)
                        new_size = max(LAYOUT_MIN_FONT_PT, blk_font.pointSizeF() * scale)
                        if LAYOUT_TEXT_SCALE != 1.0:
                            new_size = max(LAYOUT_MIN_FONT_PT, new_size * LAYOUT_TEXT_SCALE)
                        blkitem.setFontSize(new_size)
                        blk_font.setPointSizeF(new_size)
                        fshort = QFontMetricsF(blk_font)
                        base_w = fshort.horizontalAdvance(single)
                        base_h = int(round(fmt.line_spacing * fshort.height()))
                        layout_w = base_w * LAYOUT_WIDTH_STROKE_FACTOR
                        layout_h = int(base_h * LAYOUT_HEIGHT_PADDING)
                        blkitem.set_size(layout_w, layout_h, set_layout_maxsize=True)
                        blkitem.setPlainText(single)
                        blkitem._ensure_transparent_document_background()
                        # Let the existing clamp logic handle bubble/image bounds and syncing widgets.
                        if len(self.pairwidget_list) > blkitem.idx:
                            self.pairwidget_list[blkitem.idx].e_trans.setPlainText(single)
                        return True

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
        # 2.1 Font scaling: configurable min/max and fit-to-bubble (used when clamping font size below).
        layout_font_fit_bubble = bool(getattr(pcfg.module, "layout_font_fit_bubble", True))
        layout_min_pt = max(LAYOUT_MIN_FONT_PT, float(getattr(pcfg.module, "layout_font_size_min", 8.0)))
        layout_max_pt = max(layout_min_pt, float(getattr(pcfg.module, "layout_font_size_max", 72.0)))
        # When Auto layout is on (and font size = "decide by program"): scale font to fit balloon area and avoid overflow.
        # When off: we still wrap to balloon and may shrink if overflow (post_resize_ratio), but no initial balloon-based scaling.
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
                # Minimum scale so text fits but isn't too small; allow smaller text
                # for genuinely small bubbles, short translations, and very long sentences.
                base_min = 0.55
                char_count = sum(len(w) for w in words if w.strip())
                if region_rect is not None and len(region_rect) >= 4:
                    region_area = region_rect[2] * region_rect[3]
                    # Small bubble (by area) → permit smaller font
                    if img_area > 0 and region_area < 0.025 * img_area:
                        base_min = 0.35
                    # Very long sentence: allow even smaller font so it stays inside bubbles.
                    if len(words) >= 18 or char_count >= 80:
                        base_min = min(base_min, 0.30)
                    # Extremely long sentence: permit more shrink so it doesn't overflow tall/narrow bubbles.
                    if char_count >= 120:
                        base_min = min(base_min, 0.25)
                resize_ratio = min(max(resize_ratio, base_min), 1.0)
                area_for_cap = (original_block_area if original_block_area is not None else bounding_rect[2] * bounding_rect[3])
                if img_area > 0 and area_for_cap > 0.5 * img_area:
                    resize_ratio = min(resize_ratio, 0.5)
                # When region is much larger than text: scale UP so short text in big bubbles is readable.
                # Skip scale-up for long text so our scale-down (LAYOUT_TEXT_SCALE_LONG) can actually shrink it.
                is_long_text = (len(words) >= 12 or char_count >= LAYOUT_TEXT_SCALE_LONG_MIN_CHARS)
                if not is_long_text and region_rect is not None and len(region_rect) >= 4 and text_area > 0:
                    region_area = region_rect[2] * region_rect[3]
                    if region_area > 2.5 * text_area:
                        scale_up = np.sqrt(LAYOUT_SCALE_UP_FILL * region_area / text_area)
                        scale_up_max = LAYOUT_SCALE_UP_MAX * (1.15 if getattr(pcfg.module, "optimize_line_breaks", False) else 1.0)
                        scale_up = min(scale_up, scale_up_max)
                        resize_ratio = max(resize_ratio, min(scale_up, scale_up_max))

        if resize_ratio != 1:
            new_font_size = max(1.0, blk_font.pointSizeF() * resize_ratio)
            new_font_size = np.clip(new_font_size, layout_min_pt, layout_max_pt)
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
        # Limit line width so layout uses more, shorter lines (smaller frac = narrower lines).
        if region_rect is not None and len(region_rect) >= 4 and region_rect[2] > 0:
            max_central_width = float(region_rect[2] * LAYOUT_MAX_LINE_WIDTH_FRAC)

        # Optional: hyphenation for non-CJK text (improves ugly wraps for long tokens).
        # We no longer pre-force line breaks here; instead we let layout_text() score candidates.
        # IMPORTANT: do NOT hyphenate very short texts (1–2 words), so they can stay on a single line.
        ffmt = QFontMetricsF(blk_font)
        try:
            if (
                not tgt_is_cjk
                and getattr(pcfg.module, "layout_optimal_breaks", True)
                and region_rect is not None
                and len(region_rect) >= 4
            ):
                # Only hyphenate when the overall text is reasonably long; short bubbles should stay as simple lines.
                char_count = sum(len(w) for w in words if w.strip())
                if len(words) > 2 and char_count > 8:
                    maxw_px = int(max(32, region_rect[2] * 1.05))

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
        except Exception:
            pass

        # True binary search for font size: find largest size in [min_pt, max_pt] such that layout fits in bubble.
        layout_font_binary_search = bool(getattr(pcfg.module, "layout_font_binary_search", False))
        if (
            layout_font_binary_search
            and layout_font_fit_bubble
            and region_rect is not None
            and len(region_rect) >= 4
            and region_rect[2] > 0
            and region_rect[3] > 0
            and not tgt_is_cjk
        ):
            rw, rh = region_rect[2], region_rect[3]
            low_pt = layout_min_pt
            high_pt = layout_max_pt
            best_pt = low_pt
            for _ in range(12):
                mid_pt = (low_pt + high_pt) * 0.5
                blk_font.setPointSizeF(mid_pt)
                ffmt_bs = QFontMetricsF(blk_font)
                wl_list_bs = [int(ffmt_bs.horizontalAdvance(w)) for w in words]
                line_height_bs = int(round(fmt.line_spacing * ffmt_bs.height()))
                delimiter_len_bs = int(ffmt_bs.horizontalAdvance(delimiter)) or 1
                words_cpy = list(words)
                wl_cpy = list(wl_list_bs)
                try:
                    txt_bs, xywh_bs, _, _ = layout_text(
                        blkitem.blk,
                        mask,
                        mask_xyxy,
                        centroid,
                        words_cpy,
                        wl_cpy,
                        delimiter,
                        delimiter_len_bs,
                        line_height_bs,
                        0,
                        max_central_width,
                        src_is_cjk=src_is_cjk,
                        tgt_is_cjk=tgt_is_cjk,
                        ref_src_lines=ref_src_lines,
                        forced_lines=None,
                        forced_wl_lines=None,
                        collision_check=bool(getattr(pcfg.module, "layout_collision_check", True)),
                        collision_min_mask_ratio=float(getattr(pcfg.module, "layout_collision_min_mask_ratio", 0.85) or 0.85),
                        collision_max_retries=int(getattr(pcfg.module, "layout_collision_max_retries", 3) or 3),
                        target_box_height=int(rh) if rh > 0 else None,
                        optimize_for_fewer_lines=bool(getattr(pcfg.module, "optimize_line_breaks", False)),
                        balloon_shape=balloon_shape,
                    )
                except Exception:
                    high_pt = mid_pt
                    continue
                if not (txt_bs and txt_bs.strip()):
                    high_pt = mid_pt
                    continue
                # Fit with 2% margin
                if xywh_bs[2] <= rw * 1.02 and xywh_bs[3] <= rh * 1.02:
                    best_pt = mid_pt
                    low_pt = mid_pt
                else:
                    high_pt = mid_pt
            blk_font.setPointSizeF(best_pt)
            ffmt = QFontMetricsF(blk_font)
            wl_list = [int(ffmt.horizontalAdvance(w)) for w in words]
            line_height = int(round(fmt.line_spacing * ffmt.height()))
            delimiter_len = int(ffmt.horizontalAdvance(delimiter)) or 1

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
            forced_lines=None,
            forced_wl_lines=None,
            collision_check=bool(getattr(pcfg.module, "layout_collision_check", True)),
            collision_min_mask_ratio=float(getattr(pcfg.module, "layout_collision_min_mask_ratio", 0.85) or 0.85),
            collision_max_retries=int(getattr(pcfg.module, "layout_collision_max_retries", 3) or 3),
            target_box_height=int(region_rect[3]) if region_rect is not None and len(region_rect) >= 4 and region_rect[3] > 0 else None,
            optimize_for_fewer_lines=bool(getattr(pcfg.module, "optimize_line_breaks", False)),
            balloon_shape=balloon_shape,
        )
            if not (new_text and new_text.strip()):
                return
            # Safety net: merge any 1-2 word lines in the result (in case layout layer didn't merge)
            new_text = _merge_stub_lines_in_text(new_text)

            # Anti-cropping: if content would overflow bubble height when constrained, scale font down
            constrain_to_bubble = bool(getattr(pcfg.module, "layout_constrain_to_bubble", True)) and region_rect is not None and len(region_rect) >= 4 and region_rect[2] > 0 and region_rect[3] > 0
            # Additional safety: if the bubble is very small, shrink font more aggressively so text fits vertically.
            if region_rect is not None and len(region_rect) >= 4 and region_rect[3] > 0:
                n_lines = len(new_text.split('\n'))
                needed_h = n_lines * line_height * LAYOUT_HEIGHT_PADDING
                if needed_h > region_rect[3] and needed_h > 0:
                    scale_down = region_rect[3] / needed_h
                    # Allow more aggressive shrink for tiny bubbles, but keep a lower bound so text is still readable.
                    scale_down = max(scale_down, 0.35)
                    resize_ratio *= scale_down

            # 2.1 Font size post adjustment: scale to fit bubble when layout_font_fit_bubble; then clamp to min/max.
            post_resize_ratio = 1
            if layout_font_fit_bubble:
                if adaptive_fntsize:
                    downscale_constraint = 0.5
                    w = xywh[2]
                    post_resize_ratio = np.clip(max(region_rect[2] / w, downscale_constraint), 0, 1)
                    resize_ratio *= post_resize_ratio
                elif region_rect is not None and len(region_rect) >= 4:
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
                new_font_size = np.clip(new_font_size, layout_min_pt, layout_max_pt)
                resize_ratio = new_font_size / max(1.0, blkitem.font().pointSizeF())
                blkitem.textCursor().clearSelection()
                blkitem.setFontSize(new_font_size)
                blk_font.setPointSizeF(new_font_size)
                ffmt = QFontMetricsF(blk_font)
            # Font size we used for layout (for scale-down below); after setPlainText item font can be stale
            layout_font_pt = blk_font.pointSizeF()

            if restore_charfmts:
                char_fmts = blkitem.get_char_fmts()
            new_text = _merge_stub_lines_in_text(new_text)  # final pass so displayed text has no single-word lines
            maxw = max([ffmt.horizontalAdvance(t) for t in new_text.split('\n')]) if new_text.strip() else 0
            # Height: ensure minimum (single-line/descenders) and add bottom padding so last line doesn't sit on edge
            layout_h = max(int(xywh[3]), line_height)
            layout_h = int(layout_h * LAYOUT_HEIGHT_PADDING)
            layout_w = maxw * LAYOUT_WIDTH_STROKE_FACTOR
            # Constrain text box to detected bubble: use bubble region as exact box (no size change, keep centered)
            if constrain_to_bubble:
                # Use bubble region size and position so box stays fixed and centered in bubble
                # region_rect from extract_ballon_region is in mask/crop coords; convert position to image coords
                layout_w = region_rect[2]
                layout_h = region_rect[3]
            elif region_rect is not None and len(region_rect) >= 4 and region_rect[2] > 0 and region_rect[3] > 0:
                # Even when not fully constraining: cap height to bubble so box never extends past, and don't squeeze narrower than bubble
                layout_h = min(layout_h, region_rect[3])
                layout_w = max(layout_w, region_rect[2])
            # So the bubble text box shows the same line breaks as the right panel: scale font so each of our
            # lines fits in the box width (2.1: clamp to layout_font_size_min/max). Always do this to avoid Qt reflow.
            if layout_w > 0 and maxw > layout_w:
                fit_scale = layout_w / maxw
                fit_scale = min(fit_scale, 1.0)
                fit_pt = np.clip(blk_font.pointSizeF() * fit_scale, layout_min_pt, layout_max_pt)
                blk_font.setPointSizeF(fit_pt)
                blkitem.setFontSize(fit_pt)
                layout_font_pt = fit_pt
                ffmt = QFontMetricsF(blk_font)
                if blkitem.blk is not None:
                    blkitem.blk.fontformat.font_size = pt2px(fit_pt)
            blkitem.set_size(layout_w, layout_h, set_layout_maxsize=True)
            new_text = _merge_stub_lines_in_text(new_text)  # merge again right before set so 'stage.', 'in', 'the' never appear alone
            blkitem.setPlainText(new_text)
            if blkitem.blk is not None:
                blkitem.blk.translation = new_text  # keep block in sync so updateTranslation etc. don't overwrite with old text
            # Scale down only the text inside the box (same layout, same box size).
            # Use layout_font_pt (font size we used for layout). Applied before restore_charfmts;
            # restore_charfmts overwrites per-char formats with pre-scale size, so we re-apply after.
            scale = 1.0
            final_pt = 0.0
            if layout_font_pt > 0:
                n_lines = len(new_text.split('\n'))
                char_count = len(new_text.replace('\n', ''))
                is_long = (n_lines >= LAYOUT_TEXT_SCALE_LONG_MIN_LINES or
                           char_count >= LAYOUT_TEXT_SCALE_LONG_MIN_CHARS)
                scale = LAYOUT_TEXT_SCALE_LONG if is_long else LAYOUT_TEXT_SCALE
                if scale != 1.0:
                    final_pt = np.clip(layout_font_pt * scale, layout_min_pt, layout_max_pt)
                    blkitem.setFontSize(final_pt)
                    doc = blkitem.document()
                    if doc is not None:
                        default_font = doc.defaultFont()
                        if default_font.pointSizeF() != final_pt:
                            default_font.setPointSizeF(final_pt)
                            doc.setDefaultFont(default_font)
                    # Persist scaled size to block so save/load and second pass use it
                    if blkitem.blk is not None:
                        blkitem.blk.fontformat.font_size = pt2px(final_pt)
            if len(self.pairwidget_list) > blkitem.idx:
                self.pairwidget_list[blkitem.idx].e_trans.setPlainText(new_text)
            if restore_charfmts:
                self.restore_charfmts(blkitem, text, new_text, char_fmts)
            # Re-apply scale-down after restore_charfmts; it overwrites char formats with pre-scale size.
            if final_pt > 0:
                blkitem.setFontSize(final_pt)
                doc = blkitem.document()
                if doc is not None:
                    default_font = doc.defaultFont()
                    if default_font.pointSizeF() != final_pt:
                        default_font.setPointSizeF(final_pt)
                        doc.setDefaultFont(default_font)
                # Persist again so block and item stay in sync after restore_charfmts
                if blkitem.blk is not None:
                    blkitem.blk.fontformat.font_size = pt2px(final_pt)
                # Ensure layout/repaint uses the new size
                blkitem.layout.reLayoutEverything()
                blkitem.repaint_background()
            blkitem._ensure_transparent_document_background()
            rw, rh = layout_w, layout_h  # keep for final clamp when constrain is on
            if constrain_to_bubble:
                # Set box to exact bubble region in image coords (mask_xyxy = crop top-left in image)
                im_x = mask_xyxy[0] + region_rect[0]
                im_y = mask_xyxy[1] + region_rect[1]
                rw, rh = region_rect[2], region_rect[3]
                # Clamp to image bounds so box never goes outside the image (handles bad region_rect or upscale mismatch)
                im_x = max(0, min(im_x, im_w - 1))
                im_y = max(0, min(im_y, im_h - 1))
                rw = max(1, min(rw, im_w - im_x))
                rh = max(1, min(rh, im_h - im_y))
                blkitem.setRect([im_x, im_y, rw, rh], padding=False, repaint=True)
                if blkitem.blk is not None:
                    blkitem.blk._bounding_rect = blkitem.absBoundingRect()
            elif region_rect is None or len(region_rect) < 4:
                blkitem.squeezeBoundingRect()
            # Clamp block to image bounds so text never draws outside the panel
            abr = blkitem.absBoundingRect(qrect=True)
            x, y, w, h = abr.x(), abr.y(), abr.width(), abr.height()
            if constrain_to_bubble:
                # Keep bubble size; only move position if needed so we don't resize the box
                w, h = rw, rh
            if w > 0 and h > 0:
                x2 = max(0, min(x, im_w - w))
                y2 = max(0, min(y, im_h - h))
                if x2 != x or y2 != y:
                    blkitem.setRect([x2, y2, w, h], repaint=True)
                # For non-constrained layout with a known bubble region, gently clamp the box horizontally
                # (and vertically) so it stays mostly inside the bubble instead of drifting far away.
                if (not constrain_to_bubble
                        and region_rect is not None and len(region_rect) >= 4
                        and region_rect[2] > 0 and region_rect[3] > 0
                        and not is_osb):
                    bubble_x = mask_xyxy[0] + region_rect[0]
                    bubble_y = mask_xyxy[1] + region_rect[1]
                    bubble_w = region_rect[2]
                    bubble_h = region_rect[3]
                    # Re-read after potential image clamp
                    abr2 = blkitem.absBoundingRect(qrect=True)
                    bx, by, bw, bh = abr2.x(), abr2.y(), abr2.width(), abr2.height()
                    # Only adjust if the box isn't dramatically larger than the bubble
                    if bw > 0 and bh > 0 and bw <= bubble_w * 1.05 and bh <= bubble_h * 1.05:
                        min_x = bubble_x
                        max_x = bubble_x + bubble_w - bw
                        min_y = bubble_y
                        max_y = bubble_y + bubble_h - bh
                        new_x = bx
                        new_y = by
                        if max_x >= min_x:
                            new_x = min(max(bx, min_x), max_x)
                        if max_y >= min_y:
                            new_y = min(max(by, min_y), max_y)
                        if abs(new_x - bx) > 0.5 or abs(new_y - by) > 0.5:
                            blkitem.setRect([new_x, new_y, bw, bh], repaint=True)
                if constrain_to_bubble and blkitem.blk is not None:
                    blkitem.blk._bounding_rect = blkitem.absBoundingRect()
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

        page_name = self.canvas.imgtrans_proj.current_img
        mask_backup = None
        if page_name:
            self.canvas.imgtrans_proj.pages.setdefault(page_name, []).append(block)
            if self.canvas.imgtrans_proj.img_valid and self.canvas.imgtrans_proj.img_array is not None:
                img = self.canvas.imgtrans_proj.img_array
                im_h, im_w = img.shape[:2]
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                x1, x2 = max(0, min(x1, im_w)), max(0, min(x2, im_w))
                y1, y2 = max(0, min(y1, im_h)), max(0, min(y2, im_h))
                if x2 > x1 and y2 > y1:
                    mask = self.canvas.imgtrans_proj.load_mask_by_imgname(page_name)
                    if mask is None or mask.shape[0] != im_h or mask.shape[1] != im_w:
                        mask = np.full((im_h, im_w), 255, dtype=np.uint8)
                    region_backup = mask[y1:y2, x1:x2].copy()
                    mask[y1:y2, x1:x2] = 0
                    self.canvas.imgtrans_proj.save_mask(page_name, mask)
                    mask_backup = (x1, y1, x2, y2, region_backup)

        self.canvas.push_undo_command(CreateItemCommand(blk_item, self, page_name, mask_backup))

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
            def layout_after_apply(items):
                for blk in items:
                    if not getattr(blk.fontformat, 'vertical', False):
                        self.layout_textblk(blk)
                        self.layout_textblk(blk)
                    if blk.idx < len(self.pairwidget_list):
                        self.pairwidget_list[blk.idx].e_trans.setPlainText(_merge_stub_lines_in_text(blk.toPlainText()))
            layout_after = layout_after_apply if getattr(pcfg, 'let_autolayout_flag', True) else None
            self.canvas.push_undo_command(ApplyFontformatCommand(selected_blks, trans_widget_list, fontformat, layout_after=layout_after))
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

        def layout_after_apply(items):
            for blk in items:
                if not getattr(blk.fontformat, 'vertical', False):
                    self.layout_textblk(blk)
                    self.layout_textblk(blk)
                if blk.idx < len(self.pairwidget_list):
                    self.pairwidget_list[blk.idx].e_trans.setPlainText(_merge_stub_lines_in_text(blk.toPlainText()))
        layout_after = layout_after_apply if getattr(pcfg, 'let_autolayout_flag', True) else None
        self.canvas.push_undo_command(ApplyFontformatCommand(all_blks, trans_widget_list, self.formatpanel.global_format, layout_after=layout_after))
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
                blk_item.blk.translation = _merge_stub_lines_in_text(blk_item.toPlainText())
            else:
                blk_item.blk.rich_text = ''
                blk_item.blk.translation = ''
            blk_item.blk.text = [trans_pair.e_source.toPlainText()]
            blk_item.blk._bounding_rect = blk_item.absBoundingRect()
            blk_item.updateBlkFormat()
            cbl.append(blk_item.blk)

    def updateTranslation(self):
        for blk_item, transwidget in zip(self.textblk_item_list, self.pairwidget_list):
            raw = blk_item.blk.translation or ""
            merged = _merge_stub_lines_in_text(raw)  # avoid single-word lines when syncing from block
            if blk_item.blk is not None:
                blk_item.blk.translation = merged
            transwidget.e_trans.setPlainText(merged)
            blk_item.setPlainText(merged)
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

