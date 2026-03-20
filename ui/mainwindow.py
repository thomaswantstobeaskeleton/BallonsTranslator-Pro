import os.path as osp
import os, re, traceback, sys
import shutil
import copy
import numpy as np
from typing import List, Union
from pathlib import Path
import subprocess
from functools import partial
import time
import cv2

from tqdm import tqdm
from qtpy.QtWidgets import QAction, QFileDialog, QMenu, QHBoxLayout, QVBoxLayout, QApplication, QStackedWidget, QSplitter, QListWidget, QShortcut, QListWidgetItem, QMessageBox, QTextEdit, QPlainTextEdit, QDialog, QProgressBar, QLabel, QWidget
from qtpy.QtCore import Qt, QPoint, QSize, QEvent, Signal, QTimer
from qtpy.QtGui import QContextMenuEvent, QTextCursor, QGuiApplication, QIcon, QCloseEvent, QKeySequence, QKeyEvent, QPainter, QClipboard, QImage, QShowEvent, QCursor, QPixmap

from utils.logger import logger as LOGGER
from utils.text_processing import is_cjk, full_len, half_len
from utils.text_layout import _merge_stub_lines_in_string
from utils.textblock import TextBlock, TextAlignment, examine_textblk
from utils.split_text_region import split_textblock
from utils import shared
from utils.message import create_error_dialog, create_info_dialog
from modules.translators.trans_chatgpt import GPTTranslator
from modules.translators.base import lang_display_to_key
from modules import GET_VALID_TEXTDETECTORS, GET_VALID_INPAINTERS, GET_VALID_TRANSLATORS, GET_VALID_OCR
from .misc import parse_stylesheet, set_html_family, QKEY, pixmap2ndarray
from .custom_widget.animated_stack import AnimatedStackWidget
from utils.config import ProgramConfig, pcfg, save_config, text_styles, save_text_styles, load_textstyle_from, FontFormat
from utils.shortcuts import get_shortcut
from utils.proj_imgtrans import ProjImgTrans
from utils.zip_batch import ZipBatchManager
from .canvas import Canvas
from .configpanel import ConfigPanel
from .translation_context_dialog import TranslationContextDialog
from .module_manager import ModuleManager
from .textedit_area import SourceTextEdit, SelectTextMiniMenu, TransTextEdit
from .drawingpanel import DrawingPanel
from .scenetext_manager import SceneTextManager, TextPanel, PasteSrcItemsCommand
from .mainwindowbars import TitleBar, LeftBar, BottomBar
from .io_thread import ImgSaveThread, ImportDocThread, ExportDocThread, GitUpdateThread, ModelDownloadThread
from .custom_widget import Widget, ViewWidget
from .global_search_widget import GlobalSearchWidget
from .textedit_commands import GlobalRepalceAllCommand
from .framelesswindow import FramelessWindow, FramelessMoveResize
from .drawing_commands import RunBlkTransCommand
from .keywordsubwidget import KeywordSubWidget
from . import shared_widget as SW
from .custom_widget import MessageBox, FrameLessMessageBox, ImgtransProgressMessageBox
from .model_manager_dialog import ModelManagerDialog
from .shortcuts_dialog import ShortcutsDialog
from .context_menu_config_dialog import ContextMenuConfigDialog
from .batch_queue_dialog import BatchQueueDialog
from .export_dialog import ExportFormatDialog
from .spellcheck_panel import SpellCheckPanel
from .image_edit import ImageEditMode
from utils.image_colorization import apply_colorization


class PageListView(QListWidget):

    reveal_file = Signal()
    remove_images = Signal(list)
    translate_images = Signal(list)
    run_ocr_images = Signal(list)
    run_translate_images = Signal(list)
    run_inpaint_images = Signal(list)
    run_detect_images = Signal(list)
    toggle_ignore_requested = Signal(list, bool)  # (pagenames, ignored)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setIconSize(QSize(shared.PAGELIST_THUMBNAIL_SIZE, shared.PAGELIST_THUMBNAIL_SIZE))
        self.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_A and event.modifiers() in (Qt.KeyboardModifier.ControlModifier, Qt.KeyboardModifier.MetaModifier):
            self.selectAll()
            event.accept()
            return
        super().keyPressEvent(event)

    def contextMenuEvent(self, e: QContextMenuEvent):
        menu = QMenu()
        reveal_act = menu.addAction(self.tr('Reveal in File Explorer'))
        selected_items = self.selectedItems()
        n_total = self.count()
        select_all_pages_act = None
        if n_total > 0:
            menu.addSeparator()
            select_all_pages_act = menu.addAction(self.tr('Select all pages'))
        if selected_items:
            menu.addSeparator()
            translate_act = menu.addAction(self.tr('Translate Selected Images'))
            run_detect_act = menu.addAction(self.tr('Run detection on selected pages'))
            run_ocr_act = menu.addAction(self.tr('Run OCR on selected pages'))
            run_translate_act = menu.addAction(self.tr('Run translation on selected pages'))
            run_inpaint_act = menu.addAction(self.tr('Run inpainting on selected pages'))
            menu.addSeparator()
            ignore_act = menu.addAction(self.tr('Ignore in run'))
            ignore_act.setToolTip(self.tr('Skip these pages in full run and batch (when "Skip ignored pages" is on).'))
            include_act = menu.addAction(self.tr('Include in run'))
            include_act.setToolTip(self.tr('Do not skip these pages in full run and batch.'))
            menu.addSeparator()
            remove_act = menu.addAction(self.tr('Remove from Project'))
        rst = menu.exec_(e.globalPos())

        if rst == reveal_act:
            self.reveal_file.emit()
        elif rst == select_all_pages_act:
            self.selectAll()
        elif selected_items and rst == translate_act:
            self.translate_images.emit([item.text() for item in selected_items])
        elif selected_items and rst == run_detect_act:
            self.run_detect_images.emit([item.text() for item in selected_items])
        elif selected_items and rst == run_ocr_act:
            self.run_ocr_images.emit([item.text() for item in selected_items])
        elif selected_items and rst == run_translate_act:
            self.run_translate_images.emit([item.text() for item in selected_items])
        elif selected_items and rst == run_inpaint_act:
            self.run_inpaint_images.emit([item.text() for item in selected_items])
        elif selected_items and rst == ignore_act:
            self.toggle_ignore_requested.emit([item.text() for item in selected_items], True)
        elif selected_items and rst == include_act:
            self.toggle_ignore_requested.emit([item.text() for item in selected_items], False)
        elif selected_items and rst == remove_act:
            self.remove_images.emit([item.text() for item in selected_items])

        return super().contextMenuEvent(e)


class ModelDownloadProgressDialog(QDialog):
    """Modal dialog shown while retrying model downloads (Tools → Retry model downloads)."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(parent.tr('Downloading model packages') if parent else 'Downloading model packages')
        layout = QVBoxLayout(self)
        self._label = QLabel(parent.tr('Downloading model packages... This may take several minutes.') if parent else 'Downloading model packages... This may take several minutes.')
        layout.addWidget(self._label)
        self._bar = QProgressBar(self)
        self._bar.setRange(0, 0)  # indeterminate
        layout.addWidget(self._bar)
        self.setMinimumWidth(360)
        self._thread = None

    def set_thread(self, thread):
        self._thread = thread

    def closeEvent(self, event: QCloseEvent):
        if self._thread is not None and self._thread.isRunning():
            event.ignore()
            return
        super().closeEvent(event)


mainwindow_cls = Widget if (shared.HEADLESS or shared.HEADLESS_CONTINUOUS) else FramelessWindow
class MainWindow(mainwindow_cls):

    imgtrans_proj: ProjImgTrans = ProjImgTrans()
    save_on_page_changed = True
    opening_dir = False
    page_changing = False
    postprocess_mt_toggle = True

    translator = None

    restart_signal = Signal()
    create_errdialog = Signal(str, str, str)
    create_infodialog = Signal(dict)
    batch_queue_empty = Signal()
    batch_queue_cancelled = Signal()
    batch_queue_item_started = Signal(str)

    def __init__(self, app: QApplication, config: ProgramConfig, open_dir='', **exec_args) -> None:
        super().__init__()
        self._batch_cancelled = False
        self._batch_queue_dialog = None
        self._initial_model_download_done = False
        self._zip_batch = ZipBatchManager()
        self._current_batch_dir = None
        self._last_batch_report = None

        shared.create_errdialog_in_mainthread = self.create_errdialog.emit
        self.create_errdialog.connect(self.on_create_errdialog)
        shared.create_infodialog_in_mainthread = self.create_infodialog.emit
        self.create_infodialog.connect(self.on_create_infodialog)
        shared.register_view_widget = self.register_view_widget

        self.app = app
        self.backup_blkstyles = []
        self._run_imgtrans_wo_textstyle_update = False
        self._detection_only_restore = None
        self._idle_unload_timer = QTimer(self)
        self._idle_unload_timer.setSingleShot(True)
        self._idle_unload_timer.timeout.connect(self._on_idle_unload_timeout)

        self.setupThread()
        self.setupUi()
        self.setupConfig()
        self.setupShortcuts()
        self.setupRegisterWidget()
        # self.showMaximized()
        FramelessMoveResize.toggleMaxState(self)
        self.setAcceptDrops(True)

        if open_dir != '' and osp.exists(open_dir):
            if osp.isfile(open_dir) and open_dir.lower().endswith('.json'):
                self.openJsonProj(open_dir)
            else:
                self.OpenProj(open_dir)
        elif pcfg.open_recent_on_startup:
            # Auto-open most recent project when user has this preference.
            if len(self.leftBar.recent_proj_list) > 0:
                proj_dir = self.leftBar.recent_proj_list[0]
                if osp.exists(proj_dir):
                    self.OpenProj(proj_dir)

        if not (shared.HEADLESS or shared.HEADLESS_CONTINUOUS):
            # When no project is loaded, always show welcome screen so user can open or create one.
            if self.imgtrans_proj.is_empty or not self.imgtrans_proj.directory:
                self._show_welcome_screen()

        if shared.HEADLESS or shared.HEADLESS_CONTINUOUS:
            self.run_batch(**exec_args)
        elif exec_args.get('exec_dirs') and str(exec_args.get('exec_dirs')).strip():
            self.run_batch(**exec_args)

        if shared.ON_MACOS:
            # https://bugreports.qt.io/browse/QTBUG-133215
            self.hideSystemTitleBar()
            self.showMaximized()

        if not (shared.HEADLESS or shared.HEADLESS_CONTINUOUS) and getattr(pcfg, 'auto_update_from_github', False):
            QTimer.singleShot(1500, self.on_update_from_github)

    def on_create_errdialog(self, error_msg: str, detail_traceback: str = '', exception_type: str = ''):
        try:
            if exception_type != '':
                shared.showed_exception.add(exception_type)
            err = QMessageBox()
            err.setText(error_msg)
            err.setDetailedText(detail_traceback)
            err.exec()
            if exception_type != '':
                shared.showed_exception.remove(exception_type)
        except Exception:
            if exception_type in shared.showed_exception:
                shared.showed_exception.remove(exception_type)
            LOGGER.error('Failed to create error dialog')
            LOGGER.error(traceback.format_exc())

    def on_create_infodialog(self, info_dict: dict):
        # Normalize: callers may pass create_info_dialog({'title': '...', 'text': '...'}) so info_msg is a dict
        info_msg = info_dict.get('info_msg')
        if isinstance(info_msg, dict) and 'text' in info_msg:
            title = info_msg.get('title') or ''
            info_dict = {**info_dict, 'info_msg': info_msg['text']}
        else:
            title = None
        dialog = MessageBox(**info_dict)
        if title:
            dialog.setWindowTitle(title)
        dialog.show()   # exec_ will block main thread

    def register_view_widget(self, widget: ViewWidget):
        assert widget.config_name not in shared.config_name_to_view_widget
        d = {'widget': widget}
        shared.config_name_to_view_widget[widget.config_name] = d

    def setStyleSheet(self, styleSheet: str) -> None:
        self.imgtrans_progress_msgbox.setStyleSheet(styleSheet)
        self.export_doc_thread.progress_bar.setStyleSheet(styleSheet)
        self.import_doc_thread.progress_bar.setStyleSheet(styleSheet)
        return super().setStyleSheet(styleSheet)

    def setupThread(self):
        self.imsave_thread = ImgSaveThread()
        self.export_doc_thread = ExportDocThread()
        self.export_doc_thread.fin_io.connect(self.on_fin_export_doc)
        self.import_doc_thread = ImportDocThread(self)
        self.import_doc_thread.fin_io.connect(self.on_fin_import_doc)

    def resetStyleSheet(self, reverse_icon: bool = False):
        theme = 'eva-dark' if pcfg.darkmode else 'eva-light'
        self.setStyleSheet(parse_stylesheet(theme, reverse_icon))
        self._apply_custom_cursor()
        self._apply_app_font()

    def _apply_custom_cursor(self):
        if getattr(pcfg, 'use_custom_cursor', False) and getattr(pcfg, 'custom_cursor_path', ''):
            path = pcfg.custom_cursor_path
            if osp.isfile(path):
                try:
                    pix = QPixmap(path)
                    if not pix.isNull():
                        self.setCursor(QCursor(pix))
                        return
                except Exception:
                    pass
        self.unsetCursor()

    def _apply_app_font(self):
        """Apply app-wide font from theme customizer (View → Theme & UI). Never set point size <= 0."""
        family = getattr(pcfg, 'app_font_family', None) or ''
        size = max(0, getattr(pcfg, 'app_font_size', 0) or 0)
        app = QApplication.instance()
        if not app:
            return
        f = app.font()
        if f.pointSizeF() <= 0:
            f.setPointSizeF(10.0)
        if f.pointSize() <= 0:
            f.setPointSize(10)
        if family:
            f.setFamily(family)
        if size > 0 and size <= 72:
            f.setPointSize(size)
            f.setPointSizeF(float(size))
        if f.pointSizeF() <= 0:
            f.setPointSizeF(10.0)
        if f.pointSize() <= 0:
            f.setPointSize(10)
        app.setFont(f)
        # So main window and children inherit; widgets with stylesheet font-size may keep that size
        self.setFont(f)

    def setupUi(self):
        screen_size = QGuiApplication.primaryScreen().geometry().size()
        self.setMinimumWidth(screen_size.width() // 2)
        self.configPanel = ConfigPanel(self)
        self.configPanel.trans_config_panel.show_pre_MT_keyword_window.connect(self.show_pre_MT_keyword_window)
        self.configPanel.trans_config_panel.show_MT_keyword_window.connect(self.show_MT_keyword_window)
        self.configPanel.trans_config_panel.show_OCR_keyword_window.connect(self.show_OCR_keyword_window)
        self.configPanel.trans_config_panel.show_translation_context_requested.connect(self.show_translation_context_dialog)

        self.leftBar = LeftBar(self)
        self.leftBar.showPageListLabel.clicked.connect(self.pageLabelStateChanged)
        self.leftBar.imgTransChecked.connect(self.setupImgTransUI)
        self.leftBar.configChecked.connect(self.setupConfigUI)
        self.leftBar.globalSearchChecker.clicked.connect(self.on_set_gsearch_widget)
        self.leftBar.open_dir.connect(self.OpenProj)
        self.leftBar.open_json_proj.connect(self.openJsonProj)
        self.leftBar.save_proj.connect(self.manual_save)
        self.leftBar.run_clicked.connect(self.run_imgtrans)
        self.leftBar.close_project_requested.connect(self.close_project_and_show_welcome)
        self.leftBar.export_doc.connect(self.on_export_doc)
        self.leftBar.export_current_page_as.connect(self.on_export_current_page_as)
        self.leftBar.import_doc.connect(self.on_import_doc)
        self.leftBar.export_src_txt.connect(lambda : self.on_export_txt(dump_target='source'))
        self.leftBar.export_trans_txt.connect(lambda : self.on_export_txt(dump_target='translation'))
        self.leftBar.export_src_md.connect(lambda : self.on_export_txt(dump_target='source', suffix='.md'))
        self.leftBar.export_trans_md.connect(lambda : self.on_export_txt(dump_target='translation', suffix='.md'))
        self.leftBar.import_trans_txt.connect(self.on_import_trans_txt)

        self.pageList = PageListView()
        self.pageList.reveal_file.connect(self.on_reveal_file)
        self.pageList.remove_images.connect(self.on_remove_images)
        self.pageList.translate_images.connect(self.on_translate_images)
        self.pageList.toggle_ignore_requested.connect(self.on_toggle_page_ignored)
        self.pageList.run_ocr_images.connect(self.on_run_ocr_images)
        self.pageList.run_translate_images.connect(self.on_run_translate_images)
        self.pageList.run_inpaint_images.connect(self.on_run_inpaint_images)
        self.pageList.run_detect_images.connect(self.on_run_detect_images)
        self.pageList.setHidden(True)
        self.pageList.currentItemChanged.connect(self.pageListCurrentItemChanged)

        self.leftStackWidget = QStackedWidget(self)
        self.leftStackWidget.addWidget(self.pageList)
        self.leftStackWidget.setMinimumWidth(shared.PAGE_LIST_PANE_DEFAULT_WIDTH)
        self.leftStackWidget.setMaximumWidth(400)

        self.global_search_widget = GlobalSearchWidget(self.leftStackWidget)
        self.global_search_widget.req_update_pagetext.connect(self.on_req_update_pagetext)
        self.global_search_widget.req_move_page.connect(self.on_req_move_page)
        self.imsave_thread.img_writed.connect(self.global_search_widget.on_img_writed)
        self.global_search_widget.search_tree.result_item_clicked.connect(self.on_search_result_item_clicked)
        self.leftStackWidget.addWidget(self.global_search_widget)
        
        self.centralStackWidget = AnimatedStackWidget(self, duration_ms=220)

        from .welcome_widget import WelcomeWidget
        self.welcomeWidget = WelcomeWidget(self)
        self.welcomeWidget.open_folder_requested.connect(self._welcome_open_folder)
        self.welcomeWidget.open_project_requested.connect(self.OpenProj)
        self.welcomeWidget.open_images_requested.connect(self.leftBar.onOpenImages)
        self.welcomeWidget.open_acbf_requested.connect(self.leftBar.onOpenACBFCBZ)
        self.welcomeWidget.recent_project_clicked.connect(self._welcome_recent_clicked)

        self.titleBar = TitleBar(self)
        self.titleBar.closebtn_clicked.connect(self.on_closebtn_clicked)
        self.titleBar.display_lang_changed.connect(self.on_display_lang_changed)
        self.titleBar.setLeftBar(self.leftBar)
        self.bottomBar = BottomBar(self)
        self.bottomBar.textedit_checkchanged.connect(self.setTextEditMode)
        self.bottomBar.paintmode_checkchanged.connect(self.setPaintMode)
        self.bottomBar.textblock_checkchanged.connect(self.setTextBlockMode)
        self.bottomBar.spellcheck_checkchanged.connect(self.on_spellcheck_panel_toggled)

        mainHLayout = QHBoxLayout()
        mainHLayout.addWidget(self.leftBar)
        mainHLayout.setSpacing(1)
        mainHLayout.addWidget(self.centralStackWidget)
        mainHLayout.setContentsMargins(0, 0, 0, 0)

        # set up canvas
        SW.canvas = self.canvas = Canvas()
        self.canvas.imgtrans_proj = self.imgtrans_proj
        self.canvas.gv.hide_canvas.connect(self.onHideCanvas)
        self.canvas.proj_savestate_changed.connect(self.on_savestate_changed)
        self.canvas.textstack_changed.connect(self.on_textstack_changed)
        self.canvas.run_blktrans.connect(self.on_run_blktrans)
        self.canvas.drop_open_folder.connect(self.dropOpenDir)
        self.canvas.drop_open_files.connect(self.dropOpenFiles)
        self.canvas.originallayer_trans_slider = self.bottomBar.originalSlider
        self.canvas.textlayer_trans_slider = self.bottomBar.textlayerSlider
        self.canvas.copy_src_signal.connect(self.on_copy_src)
        self.canvas.paste_src_signal.connect(self.on_paste_src)
        self.canvas.copy_trans_signal.connect(self.on_copy_trans)
        self.canvas.paste_trans_signal.connect(self.on_paste_trans)
        self.canvas.clear_src_signal.connect(self.on_clear_src)
        self.canvas.clear_trans_signal.connect(self.on_clear_trans)
        self.canvas.select_all_signal.connect(self.on_select_all_canvas)
        self.canvas.spell_check_src_signal.connect(self.on_spell_check_src)
        self.canvas.spell_check_trans_signal.connect(self.on_spell_check_trans)
        self.canvas.trim_whitespace_signal.connect(self.on_trim_whitespace)
        self.canvas.to_uppercase_signal.connect(self.on_to_uppercase)
        self.canvas.to_lowercase_signal.connect(self.on_to_lowercase)
        self.canvas.toggle_strikethrough_signal.connect(self.on_toggle_strikethrough)
        self.canvas.set_gradient_type_signal.connect(self.on_set_gradient_type)
        self.canvas.set_text_on_path_signal.connect(self.on_set_text_on_path)
        self.canvas.run_detect_region.connect(self.on_run_detect_region)
        self.canvas.download_page_requested.connect(self.on_download_page_requested)
        self.canvas.merge_selected_blocks_signal.connect(self.on_merge_selected_blocks)
        self.canvas.split_selected_regions_signal.connect(self.on_split_selected_regions)
        self.canvas.move_blocks_up_signal.connect(self.on_move_blocks_up)
        self.canvas.move_blocks_down_signal.connect(self.on_move_blocks_down)
        self.canvas.import_image_to_blk.connect(self.on_import_image_to_blk)
        self.canvas.clear_overlay_signal.connect(self.on_clear_overlay)

        self.app.installEventFilter(self)
        self.canvas.gv.installEventFilter(self)
        self.canvas.gv.viewport().installEventFilter(self)

        self.bottomBar.originalSlider.valueChanged.connect(self.canvas.setOriginalTransparencyBySlider)
        self.bottomBar.textlayerSlider.valueChanged.connect(self.canvas.setTextLayerTransparencyBySlider)
        
        self.drawingPanel = DrawingPanel(self.canvas, self.configPanel.inpaint_config_panel)
        self.textPanel = TextPanel(self.app)
        self.textPanel.formatpanel.foldTextBtn.checkStateChanged.connect(self.fold_textarea)
        self.textPanel.formatpanel.sourceBtn.checkStateChanged.connect(self.show_source_text)
        self.textPanel.formatpanel.transBtn.checkStateChanged.connect(self.show_trans_text)
        self.textPanel.formatpanel.textstyle_panel.export_style.connect(self.export_tstyles)
        self.textPanel.formatpanel.textstyle_panel.import_style.connect(self.import_tstyles)
        self.textPanel.formatpanel.set_default_format_requested.connect(self.on_set_default_format_requested)

        self.ocrSubWidget = KeywordSubWidget(self.tr("Keyword substitution for source text"))
        self.ocrSubWidget.setParent(self)
        self.ocrSubWidget.setWindowFlags(Qt.WindowType.Window)
        self.ocrSubWidget.hide()
        self.mtPreSubWidget = KeywordSubWidget(self.tr("Keyword substitution for machine translation source text"))
        self.mtPreSubWidget.setParent(self)
        self.mtPreSubWidget.setWindowFlags(Qt.WindowType.Window)
        self.mtPreSubWidget.hide()
        self.mtSubWidget = KeywordSubWidget(self.tr("Keyword substitution for machine translation"))
        self.mtSubWidget.setParent(self)
        self.mtSubWidget.setWindowFlags(Qt.WindowType.Window)
        self.mtSubWidget.hide()

        SW.st_manager = self.st_manager = SceneTextManager(self.app, self, self.canvas, self.textPanel)
        self.st_manager.new_textblk.connect(self.canvas.search_widget.on_new_textblk)
        self.canvas.search_widget.pairwidget_list = self.st_manager.pairwidget_list
        self.canvas.search_widget.textblk_item_list = self.st_manager.textblk_item_list
        self.canvas.search_widget.replace_one.connect(self.st_manager.on_page_replace_one)
        self.canvas.search_widget.replace_all.connect(self.st_manager.on_page_replace_all)

        # comic trans pannel
        self.rightComicTransStackPanel = AnimatedStackWidget(self, duration_ms=180)
        self.rightComicTransStackPanel.addWidget(self.drawingPanel)
        self.rightComicTransStackPanel.addWidget(self.textPanel)
        self.spellCheckPanel = SpellCheckPanel(self)
        self.spellCheckPanel.set_get_blocks(self._spellcheck_get_blocks)
        self.spellCheckPanel.set_apply_replacement(self._spellcheck_apply_replacement)
        self.spellCheckPanel.close_requested.connect(self._on_spellcheck_panel_close)
        self.spellCheckPanel.focus_block_requested.connect(self._on_spellcheck_focus_block)
        self.rightComicTransStackPanel.addWidget(self.spellCheckPanel)
        self.rightComicTransStackPanel.currentChanged.connect(self.on_transpanel_changed)

        self.comicTransSplitter = QSplitter(Qt.Orientation.Horizontal)
        self.comicTransSplitter.addWidget(self.leftStackWidget)
        self.comicTransSplitter.addWidget(self.canvas.gv)
        self.comicTransSplitter.addWidget(self.rightComicTransStackPanel)

        self.centralStackWidget.addWidget(self.welcomeWidget)
        self.centralStackWidget.addWidget(self.comicTransSplitter)
        self.centralStackWidget.addWidget(self.configPanel)

        self.selectext_minimenu = self.st_manager.selectext_minimenu = SelectTextMiniMenu(self.app, self)
        self.selectext_minimenu.block_current_editor.connect(self.st_manager.on_block_current_editor)
        self.selectext_minimenu.hide()

        mainVBoxLayout = QVBoxLayout(self)
        mainVBoxLayout.addWidget(self.titleBar)
        mainVBoxLayout.addLayout(mainHLayout)
        mainVBoxLayout.addWidget(self.bottomBar)
        margin = mainVBoxLayout.contentsMargins()
        self.main_margin = margin
        mainVBoxLayout.setContentsMargins(0, 0, 0, 0)
        mainVBoxLayout.setSpacing(0)

        self.mainvlayout = mainVBoxLayout
        self.comicTransSplitter.setStretchFactor(0, 1)
        self.comicTransSplitter.setStretchFactor(1, 10)
        self.comicTransSplitter.setStretchFactor(2, 1)
        # Allow user to resize the right panel by dragging the splitter (wider or narrower)
        self.rightComicTransStackPanel.setMinimumWidth(280)
        self.rightComicTransStackPanel.setMaximumWidth(860)
        self._comic_trans_splitter_initialized = False
        self.imgtrans_progress_msgbox = ImgtransProgressMessageBox()
        self.resetStyleSheet()

    def on_finish_setdetector(self):
        """Defer UI update to avoid QPainter conflict when signal fires during paint (e.g. model load)."""
        module_manager = self.module_manager
        if module_manager.textdetector is not None:
            name = module_manager.textdetector.name
            def _update():
                pcfg.module.textdetector = name
                self.configPanel.detect_config_panel.setDetector(name)
                self.bottomBar.textdet_selector.setSelectedValue(name)
                LOGGER.info('Text detector set to {}'.format(name))
            QTimer.singleShot(0, _update)

    def on_finish_setocr(self):
        module_manager = self.module_manager
        if module_manager.ocr is not None:
            name = module_manager.ocr.name
            def _update():
                pcfg.module.ocr = name
                self.configPanel.ocr_config_panel.setOCR(name)
                self.bottomBar.ocr_selector.setSelectedValue(name)
                LOGGER.info('OCR set to {}'.format(name))
            QTimer.singleShot(0, _update)

    def on_finish_setinpainter(self):
        module_manager = self.module_manager
        if module_manager.inpainter is not None:
            name = module_manager.inpainter.name
            def _update():
                pcfg.module.inpainter = name
                self.configPanel.inpaint_config_panel.setInpainter(name)
                self.bottomBar.inpaint_selector.setSelectedValue(name)
                LOGGER.info('Inpainter set to {}'.format(name))
            QTimer.singleShot(0, _update)

    def on_finish_settranslator(self):
        module_manager = self.module_manager
        translator = module_manager.translator
        if translator is not None:
            name = translator.name
            def _update():
                pcfg.module.translator = name
                self.bottomBar.trans_selector.finishSetTranslator(translator)
                self.configPanel.trans_config_panel.finishSetTranslator(translator)
                LOGGER.info('Translator set to {}'.format(name))
            QTimer.singleShot(0, _update)
        else:
            LOGGER.error('invalid translator')
        
    def on_enable_module(self, idx, checked):
        if idx == 0:
            pcfg.module.enable_detect = checked
            self.bottomBar.textdet_selector.setVisible(checked)
        elif idx == 1:
            pcfg.module.enable_ocr = checked
            self.bottomBar.ocr_selector.setVisible(checked)
        elif idx == 2:
            pcfg.module.enable_translate = checked
            self.bottomBar.trans_selector.setVisible(checked)
        elif idx == 3:
            pcfg.module.enable_inpaint = checked
            self.bottomBar.inpaint_selector.setVisible(checked)
        pcfg.module.update_finish_code()

    def setupConfig(self):

        self.bottomBar.originalSlider.setValue(int(pcfg.original_transparency * 100))
        self.bottomBar.trans_selector.selector.addItems(GET_VALID_TRANSLATORS())
        self.bottomBar.ocr_selector.selector.addItems(GET_VALID_OCR())
        self.bottomBar.textdet_selector.selector.addItems(GET_VALID_TEXTDETECTORS())
        self.bottomBar.textdet_selector.selector.currentTextChanged.connect(self.on_textdet_changed)
        self.bottomBar.inpaint_selector.selector.addItems(GET_VALID_INPAINTERS())
        self.bottomBar.inpaint_selector.selector.currentTextChanged.connect(self.on_inpaint_changed)
        self.bottomBar.trans_selector.cfg_clicked.connect(self.to_trans_config)
        self.bottomBar.trans_selector.selector.currentTextChanged.connect(self.on_trans_changed)
        self.bottomBar.trans_selector.tgt_selector.currentTextChanged.connect(self.on_trans_tgt_changed)
        self.bottomBar.trans_selector.src_selector.currentTextChanged.connect(self.on_trans_src_changed)
        self.bottomBar.textdet_selector.cfg_clicked.connect(self.to_detect_config)
        self.bottomBar.inpaint_selector.cfg_clicked.connect(self.to_inpaint_config)
        self.bottomBar.ocr_selector.cfg_clicked.connect(self.to_ocr_config)
        self.bottomBar.ocr_selector.selector.currentTextChanged.connect(self.on_ocr_changed)
        self.bottomBar.textdet_selector.setVisible(pcfg.module.enable_detect)
        self.bottomBar.ocr_selector.setVisible(pcfg.module.enable_ocr)
        self.bottomBar.trans_selector.setVisible(pcfg.module.enable_translate)
        self.bottomBar.inpaint_selector.setVisible(pcfg.module.enable_inpaint)

        self.configPanel.trans_config_panel.target_combobox.currentTextChanged.connect(self.on_trans_tgt_changed)
        self.configPanel.trans_config_panel.source_combobox.currentTextChanged.connect(self.on_trans_src_changed)

        self.drawingPanel.maskTransperancySlider.setValue(int(pcfg.mask_transparency * 100))
        self.leftBar.initRecentProjMenu(pcfg.recent_proj_list)
        self.leftBar.showPageListLabel.setChecked(pcfg.show_page_list)
        self.updatePageList()
        self.leftBar.save_config.connect(self.save_config)
        self.leftBar.imgTransChecker.setChecked(True)
        self.st_manager.formatpanel.global_format = pcfg.global_fontformat
        self.st_manager.formatpanel.set_active_format(pcfg.global_fontformat)
        
        self.rightComicTransStackPanel.setHidden(True)
        self.st_manager.setTextEditMode(False)
        self.st_manager.formatpanel.foldTextBtn.setChecked(pcfg.fold_textarea)
        self.st_manager.formatpanel.transBtn.setCheckState(pcfg.show_trans_text)
        self.st_manager.formatpanel.sourceBtn.setCheckState(pcfg.show_source_text)
        self.fold_textarea(pcfg.fold_textarea)
        self.show_trans_text(pcfg.show_trans_text)
        self.show_source_text(pcfg.show_source_text)

        self.module_manager = module_manager = ModuleManager(self.imgtrans_proj)
        module_manager.finish_translate_page.connect(self.finishTranslatePage)
        module_manager.imgtrans_pipeline_finished.connect(self.on_imgtrans_pipeline_finished)
        module_manager.page_trans_finished.connect(self.on_pagtrans_finished)
        module_manager.setupThread(self.configPanel, self.imgtrans_progress_msgbox, self.ocr_postprocess, self.translate_preprocess, self.translate_postprocess)
        module_manager.progress_msgbox.showed.connect(self.on_imgtrans_progressbox_showed)
        module_manager.blktrans_pipeline_finished.connect(self.on_blktrans_finished)
        module_manager.detect_region_finished.connect(self.on_detect_region_finished)
        module_manager.imgtrans_thread.post_process_mask = self.drawingPanel.rectPanel.post_process_mask
        module_manager.inpaint_thread.finish_set_module.connect(self.on_finish_setinpainter)
        module_manager.translate_thread.finish_set_module.connect(self.on_finish_settranslator)
        module_manager.textdetect_thread.finish_set_module.connect(self.on_finish_setdetector)
        module_manager.ocr_thread.finish_set_module.connect(self.on_finish_setocr)
        module_manager.setTextDetector()
        module_manager.setOCR()
        module_manager.setTranslator()
        module_manager.setInpainter()

        self.titleBar.darkModeAction.setChecked(pcfg.darkmode)
        self.titleBar.themeLightAction.setChecked(not pcfg.darkmode)
        self.titleBar.themeDarkAction.setChecked(pcfg.darkmode)
        self.titleBar.bubblyUIAction.setChecked(getattr(pcfg, 'bubbly_ui', True))

        self.drawingPanel.set_config(pcfg.drawpanel)
        self.drawingPanel.initDLModule(module_manager)

        self.global_search_widget.imgtrans_proj = self.imgtrans_proj
        self.global_search_widget.setupReplaceThread(self.st_manager.pairwidget_list, self.st_manager.textblk_item_list)
        self.global_search_widget.replace_thread.finished.connect(self.on_global_replace_finished)

        self.configPanel.setupConfig()
        self.configPanel.save_config.connect(self.save_config)
        self.configPanel.reload_textstyle.connect(self.load_textstyle_from_proj_dir)
        self.configPanel.show_only_custom_font.connect(self.on_show_only_custom_font)
        self.configPanel.darkmode_changed.connect(self.on_config_darkmode_changed)
        self.configPanel.bubbly_ui_changed.connect(self.on_config_bubbly_ui_changed)
        self.configPanel.custom_cursor_changed.connect(self._on_config_custom_cursor_changed)
        self.configPanel.display_lang_changed.connect(self.on_display_lang_changed)
        self.configPanel.dev_mode_changed.connect(self.on_dev_mode_changed)
        self.configPanel.manual_mode_changed.connect(self._update_run_button_tooltip)
        if pcfg.let_show_only_custom_fonts_flag:
            self.on_show_only_custom_font(True)

        textblock_mode = pcfg.imgtrans_textblock
        if pcfg.imgtrans_textedit:
            if textblock_mode:
                self.bottomBar.textblockChecker.setChecked(True)
            self.bottomBar.texteditChecker.click()
        elif pcfg.imgtrans_paintmode:
            self.bottomBar.paintChecker.click()

        self.textPanel.formatpanel.textstyle_panel.initStyles(text_styles)
        self.textPanel.formatpanel.textstyle_panel.set_active_style_by_name(
            getattr(pcfg, 'default_text_style_name', '') or ''
        )

        self.canvas.search_widget.whole_word_toggle.setChecked(pcfg.fsearch_whole_word)
        self.canvas.search_widget.case_sensitive_toggle.setChecked(pcfg.fsearch_case)
        self.canvas.search_widget.regex_toggle.setChecked(pcfg.fsearch_regex)
        self.canvas.search_widget.range_combobox.setCurrentIndex(pcfg.fsearch_range)
        self.global_search_widget.whole_word_toggle.setChecked(pcfg.gsearch_whole_word)
        self.global_search_widget.case_sensitive_toggle.setChecked(pcfg.gsearch_case)
        self.global_search_widget.regex_toggle.setChecked(pcfg.gsearch_regex)
        self.global_search_widget.range_combobox.setCurrentIndex(pcfg.gsearch_range)

        if self.rightComicTransStackPanel.isHidden():
            self.setPaintMode()

        try:
            self.ocrSubWidget.loadCfgSublist(pcfg.ocr_sublist)
        except Exception as e:
            LOGGER.error(traceback.format_exc())
            pcfg.ocr_sublist = []
            self.ocrSubWidget.loadCfgSublist(pcfg.ocr_sublist)

        try:
            self.mtPreSubWidget.loadCfgSublist(pcfg.pre_mt_sublist)
        except Exception as e:
            LOGGER.error(traceback.format_exc())
            pcfg.pre_mt_sublist = []
            self.mtPreSubWidget.loadCfgSublist(pcfg.pre_mt_sublist)

        try:
            self.mtSubWidget.loadCfgSublist(pcfg.mt_sublist)
        except Exception as e:
            LOGGER.error(traceback.format_exc())
            pcfg.mt_sublist = []
            self.mtSubWidget.loadCfgSublist(pcfg.mt_sublist)

        self._update_run_button_tooltip()

    def _update_run_button_tooltip(self):
        base = self.tr('Run pipeline (same as Pipeline → Run).')
        if getattr(pcfg, 'manual_mode', False):
            base += ' ' + self.tr('Manual mode: runs current page only.')
        self.leftBar.runBtn.setToolTip(base)
        self.titleBar.runToolBtn.setToolTip(base)
        self.bottomBar.runBtn.setToolTip(base)

    def setupImgTransUI(self):
        self.centralStackWidget.setCurrentIndex(1)
        self.bottomBar.setPipelineVisible(True)
        if self.leftBar.needleftStackWidget():
            self.leftStackWidget.show()
            if self.leftBar.showPageListLabel.isChecked():
                self.leftStackWidget.setCurrentWidget(self.pageList)
                self.pageList.setHidden(False)
            else:
                self.leftStackWidget.setCurrentWidget(self.global_search_widget)
        else:
            self.leftStackWidget.hide()

    def setupConfigUI(self):
        self.centralStackWidget.setCurrentIndex(2)
        self.bottomBar.setPipelineVisible(False)

    def set_display_lang(self, lang: str):
        self.retranslateUI()

    def OpenProj(self, proj_path: str):
        if osp.isdir(proj_path):
            self.openDir(proj_path)
        else:
            self.openJsonProj(proj_path)
        
        if pcfg.let_textstyle_indep_flag and not (shared.HEADLESS or shared.HEADLESS_CONTINUOUS):
            self.load_textstyle_from_proj_dir(from_proj=True)

    def _welcome_open_folder(self):
        self.leftBar.onOpenFolder()

    def _welcome_recent_clicked(self, path: str):
        if path and osp.exists(path):
            self.OpenProj(path)

    def _show_welcome_screen(self):
        """Switch to welcome screen and refresh recent list."""
        if hasattr(self, 'welcomeWidget'):
            self.welcomeWidget.set_recent_projects(getattr(self.leftBar, 'recent_proj_list', []) or [])
        self.centralStackWidget.setCurrentIndex(0)

    def _show_main_content(self):
        """Switch from welcome to main translation view."""
        self.centralStackWidget.setCurrentIndex(1)
        # Default to text editor panel (index 1) on every project open / app start
        if not self.rightComicTransStackPanel.isHidden():
            self.rightComicTransStackPanel.setCurrentIndex(1)

    def load_textstyle_from_proj_dir(self, from_proj=False):
        if from_proj:
            text_style_path = osp.join(self.imgtrans_proj.directory, 'textstyles.json')
        else:
            text_style_path = 'config/textstyles/default.json'
        if osp.exists(text_style_path):
            load_textstyle_from(text_style_path)
            self.textPanel.formatpanel.textstyle_panel.setStyles(text_styles)
            self.textPanel.formatpanel.textstyle_panel.set_active_style_by_name(
                getattr(pcfg, 'default_text_style_name', '') or ''
            )
        else:
            pcfg.text_styles_path = text_style_path
            save_text_styles()

    def on_show_only_custom_font(self, only_custom: bool):
        if only_custom:
            font_list = shared.CUSTOM_FONTS
        else:
            font_list = shared.FONT_FAMILIES
        self.textPanel.formatpanel.familybox.update_font_list(font_list)

    def openDir(self, directory: str):
        try:
            self.opening_dir = True
            # 在加载项目前检查并生成TIF文件的预览图
            self.generate_tif_thumbnails(directory)
            # 重新加载项目，此时应该只加载预览图
            self.imgtrans_proj.load(directory)
            self.st_manager.clearSceneTextitems()
            self.titleBar.setTitleContent(osp.basename(directory))
            self.updatePageList()
            self._ensure_first_page_loaded_and_autolayout()
            self.opening_dir = False
            self._show_main_content()
        except Exception as e:
            self.opening_dir = False
            create_error_dialog(e, self.tr('Failed to load project ') + directory)
            return

    def generate_tif_thumbnails(self, directory: str):
        """
        为目录中的TIF文件生成预览图，并确保只加载预览图
        """
        try:
            from utils.io_utils import create_thumbnail, find_tif_files
            # 查找目录中的所有TIF文件
            tif_files = find_tif_files(directory)
            
            # 为每个TIF文件生成预览图
            for tif_file in tif_files:
                tif_path = osp.join(directory, tif_file)
                # 检查是否已经存在对应的预览图
                base_path = Path(tif_path)
                thumb_path = base_path.parent / f"{base_path.stem}_thumb.jpg"
                
                # 如果预览图不存在，则生成预览图
                if not osp.exists(thumb_path):
                    create_thumbnail(tif_path, max_width=1000)
                    
        except Exception as e:
            LOGGER.error(f"Failed to generate TIF thumbnails: {e}")
        
    def dropOpenDir(self, directory: str):
        if isinstance(directory, str) and osp.exists(directory):
            self.leftBar.updateRecentProjList(directory)
            self.OpenProj(directory)

    def dropOpenFiles(self, file_paths: list):
        """Open a project from a list of image file paths (e.g. drag-drop or Select Files). Uses first file's directory as project dir and only those files as pages."""
        if not file_paths:
            return
        from utils.logger import logger as LOGGER
        from utils import create_error_dialog
        try:
            if self.imgtrans_proj.directory is not None and self.canvas.projstate_unsaved:
                self.saveCurrentPage()
            first_path = osp.normpath(file_paths[0])
            target_dir = osp.dirname(first_path)
            image_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif', '.gif'}
            copied_files = []
            for p in file_paths:
                p = osp.normpath(p)
                if not osp.isfile(p) or osp.splitext(p)[1].lower() not in image_ext:
                    continue
                if osp.dirname(p) != target_dir:
                    continue
                copied_files.append(osp.basename(p))
            if not copied_files:
                return
            self.leftBar.updateRecentProjList(target_dir)
            self.imgtrans_proj.directory = target_dir
            self.imgtrans_proj.proj_path = osp.join(target_dir, self.imgtrans_proj.proj_name() + '.json')
            if not osp.exists(self.imgtrans_proj.inpainted_dir()):
                os.makedirs(self.imgtrans_proj.inpainted_dir())
            if not osp.exists(self.imgtrans_proj.mask_dir()):
                os.makedirs(self.imgtrans_proj.mask_dir())
            self.imgtrans_proj.pages = {}
            self.imgtrans_proj._pagename2idx = {}
            self.imgtrans_proj._idx2pagename = {}
            for ii, name in enumerate(copied_files):
                self.imgtrans_proj.pages[name] = []
                self.imgtrans_proj._pagename2idx[name] = ii
                self.imgtrans_proj._idx2pagename[ii] = name
            if copied_files:
                self.imgtrans_proj.set_current_img(copied_files[0])
            self.imgtrans_proj.save()
            self.st_manager.clearSceneTextitems()
            self.titleBar.setTitleContent(page_name=osp.basename(target_dir))
            self.updatePageList()
            self.canvas.clear_undostack(update_saved_step=True)
            self.canvas.drop_files = []
            self.canvas.drop_folder = None
        except Exception as e:
            LOGGER.error("Error importing images: %s", str(e))
            create_error_dialog(e, self.tr('Failed to import images'))

    def openJsonProj(self, json_path: str):
        try:
            self.opening_dir = True
            self.imgtrans_proj.load_from_json(json_path)
            self.st_manager.clearSceneTextitems()
            self.leftBar.updateRecentProjList(self.imgtrans_proj.proj_path)
            self.updatePageList()
            self.titleBar.setTitleContent(osp.basename(self.imgtrans_proj.proj_path))
            self._ensure_first_page_loaded_and_autolayout()
            self.opening_dir = False
            self._show_main_content()
        except Exception as e:
            self.opening_dir = False
            create_error_dialog(e, self.tr('Failed to load project from') + json_path)
        
    def updatePageList(self):
        if self.pageList.count() != 0:
            self.pageList.clear()
        if len(self.imgtrans_proj.pages) >= shared.PAGELIST_THUMBNAIL_MAXNUM:
            item_func = lambda imgname: QListWidgetItem(imgname)
        else:
            item_func = lambda imgname:\
                QListWidgetItem(QIcon(osp.join(self.imgtrans_proj.directory, imgname)), imgname)
        for imgname in self.imgtrans_proj.pages:
            lstitem =  item_func(imgname)
            self.pageList.addItem(lstitem)
            if imgname == self.imgtrans_proj.current_img:
                self.pageList.setCurrentItem(lstitem)
        self._update_page_list_ignored_style()

    def _ensure_first_page_loaded_and_autolayout(self):
        """After opening a project, ensure the first page is displayed. Use saved layout (no re-run of auto layout)."""
        if not self.imgtrans_proj.current_img:
            return
        self.canvas.updateCanvas()
        self.st_manager.updateSceneTextitems()

    def pageLabelStateChanged(self):
        setup = self.leftBar.showPageListLabel.isChecked()
        if setup:
            if self.leftStackWidget.isHidden():
                self.leftStackWidget.show()
            if self.leftBar.globalSearchChecker.isChecked():
                self.leftBar.globalSearchChecker.setChecked(False)
            self.leftStackWidget.setCurrentWidget(self.pageList)
            self.pageList.setHidden(False)
        else:
            self.leftStackWidget.hide()
        pcfg.show_page_list = setup
        save_config()

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        if not getattr(self, '_comic_trans_splitter_initialized', False):
            self._comic_trans_splitter_initialized = True
            total = self.comicTransSplitter.width()
            if total > 400:
                right_default = 282
                # Default left pane to the same width it locks at when dragging (PAGE_LIST_PANE_DEFAULT_WIDTH).
                left = shared.PAGE_LIST_PANE_DEFAULT_WIDTH
                center = max(100, total - left - right_default)
                self.comicTransSplitter.setSizes([left, center, right_default])

        # Deferred initial model download (launch.py sets DEFER_INITIAL_MODEL_DOWNLOAD so app opens first)
        if getattr(shared, 'DEFER_INITIAL_MODEL_DOWNLOAD', False) and not self._initial_model_download_done:
            shared.DEFER_INITIAL_MODEL_DOWNLOAD = False
            self._initial_model_download_done = True
            self._run_deferred_model_download()

    def closeEvent(self, event: QCloseEvent) -> None:
        reply = QMessageBox.question(
            self,
            self.tr('Confirm exit'),
            self.tr('Are you sure you want to quit?'),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.No:
            event.ignore()
            return
        if not self.imgtrans_proj.is_empty:
            self.conditional_save(keep_exist_as_backup=True)
            # Always sync current page scene to blocks and save so text boxes (e.g. after auto layout) are persisted
            if self.imgtrans_proj.current_img and self.st_manager.textblk_item_list:
                self.st_manager.updateTextBlkList()
                try:
                    self.imgtrans_proj.save(keep_exist_as_backup=True)
                except Exception:
                    pass
        while True:
            if not self.imsave_thread.isRunning():
                break
            time.sleep(0.1)
        self.st_manager.hovering_transwidget = None
        self.st_manager.blockSignals(True)
        self.canvas.prepareClose()
        self.save_config()
        return super().closeEvent(event)

    def changeEvent(self, event: QEvent):
        if event.type() == QEvent.Type.WindowStateChange:
            if self.windowState() & Qt.WindowState.WindowMaximized:
                if not shared.ON_MACOS:
                    self.titleBar.maxBtn.setChecked(True)
        elif event.type() == QEvent.Type.ActivationChange:
            self.canvas.on_activation_changed()

        super().changeEvent(event)
    
    def retranslateUI(self):
        # according to https://stackoverflow.com/questions/27635068/how-to-retranslate-dynamically-created-widgets
        # we got to do it manually ... I'd rather restart the program
        msg = QMessageBox()
        msg.setText(self.tr('Restart to apply changes? \n'))
        msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        ret = msg.exec_()
        if ret == QMessageBox.StandardButton.Yes:
            self.restart_signal.emit()

    def save_config(self):
        # Flush Region merge tool dialog state if it's open so it gets persisted
        if getattr(self, 'merge_dialog', None) is not None and self.merge_dialog.isVisible():
            self.merge_dialog.save_settings_to_pcfg()
        save_config()

    def on_set_default_format_requested(self):
        """Save current global font format to config so it is used as default for new projects/sessions."""
        # Persist which preset is selected so it shows as active (blue) on next launch
        panel = self.st_manager.formatpanel.textstyle_panel
        if panel.active_text_style_label is not None:
            pcfg.default_text_style_name = getattr(
                panel.active_text_style_label.fontfmt, '_style_name', ''
            ) or ''
        else:
            pcfg.default_text_style_name = ''
        save_config()

    def onHideCanvas(self):
        self.canvas.clearToolStates()

    def conditional_save(self, keep_exist_as_backup=False):
        if self.canvas.projstate_unsaved and not self.opening_dir:
            update_scene_text = save_proj = self.canvas.text_change_unsaved()
            save_rst_only = not self.canvas.draw_change_unsaved()
            if not save_rst_only:
                save_proj = True
            
            self.saveCurrentPage(update_scene_text, save_proj, restore_interface=True, save_rst_only=save_rst_only, keep_exist_as_backup=keep_exist_as_backup)

    def pageListCurrentItemChanged(self):
        item = self.pageList.currentItem()
        self.page_changing = True
        if item is not None:
            if self.save_on_page_changed:
                self.conditional_save()
            self.imgtrans_proj.set_current_img(item.text())
            self.canvas.clear_undostack(update_saved_step=True)
            self.canvas.updateCanvas()
            self.st_manager.updateSceneTextitems()
            self.titleBar.setTitleContent(page_name=self.imgtrans_proj.current_img)
            self.module_manager.handle_page_changed()
            self.drawingPanel.handle_page_changed()
            
        self.page_changing = False

    def setupShortcuts(self):
        self.titleBar.nextpage_trigger.connect(self.shortcutNext) 
        self.titleBar.prevpage_trigger.connect(self.shortcutBefore)
        self.titleBar.textedit_trigger.connect(self.shortcutTextedit)
        self.titleBar.drawboard_trigger.connect(self.shortcutDrawboard)
        self.titleBar.spellcheck_panel_trigger.connect(self.shortcutSpellCheckPanel)
        self.titleBar.redo_trigger.connect(self.on_redo)
        self.titleBar.undo_trigger.connect(self.on_undo)
        self.titleBar.page_search_trigger.connect(self.on_page_search)
        self.titleBar.global_search_trigger.connect(self.on_global_search)
        self.titleBar.replacePreMTkeyword_trigger.connect(self.show_pre_MT_keyword_window)
        self.titleBar.replaceMTkeyword_trigger.connect(self.show_MT_keyword_window)
        self.titleBar.replaceOCRkeyword_trigger.connect(self.show_OCR_keyword_window)
        self.titleBar.translation_context_trigger.connect(self.show_translation_context_dialog)
        self.titleBar.run_trigger.connect(self.run_imgtrans)
        self.bottomBar.run_clicked.connect(self.run_imgtrans)
        self.titleBar.run_woupdate_textstyle_trigger.connect(self.run_imgtrans_wo_textstyle_update)
        self.titleBar.translate_page_trigger.connect(self.on_transpagebtn_pressed)
        self.titleBar.enable_module.connect(self.on_enable_module)
        self.titleBar.importtstyle_trigger.connect(self.import_tstyles)
        self.titleBar.exporttstyle_trigger.connect(self.export_tstyles)
        self.titleBar.darkmode_trigger.connect(self.on_darkmode_triggered)
        self.titleBar.theme_light_trigger.connect(self._on_theme_light_triggered)
        self.titleBar.theme_dark_trigger.connect(self._on_theme_dark_triggered)
        self.titleBar.bubbly_ui_trigger.connect(self._on_bubbly_ui_triggered)
        self.titleBar.merge_tool_trigger.connect(self.on_open_merge_tool)
        self.titleBar.re_run_detection_only_trigger.connect(self.on_re_run_detection_only)
        self.titleBar.re_run_ocr_only_trigger.connect(self.on_re_run_ocr_only)
        self.titleBar.batch_export_trigger.connect(self.on_batch_export)
        self.titleBar.batch_export_as_trigger.connect(self.on_batch_export_as)
        self.titleBar.export_lptxt_trigger.connect(self.on_export_lptxt)
        self.titleBar.validate_project_trigger.connect(self.on_validate_project)
        self.titleBar.show_batch_report_trigger.connect(self._show_batch_report_dialog)
        self.titleBar.manga_source_trigger.connect(self.on_open_manga_source)
        self.titleBar.batch_queue_trigger.connect(self.on_open_batch_queue)
        self.titleBar.manage_models_trigger.connect(self.on_open_manage_models)
        self.titleBar.retry_models_trigger.connect(self.on_retry_model_downloads)
        self.titleBar.release_model_caches_trigger.connect(self.on_release_model_caches)
        self.titleBar.clear_pipeline_caches_trigger.connect(self.on_clear_pipeline_caches)
        self.titleBar.run_preset_full_trigger.connect(self.on_run_preset_full)
        self.titleBar.run_preset_detect_ocr_trigger.connect(self.on_run_preset_detect_ocr)
        self.titleBar.run_preset_translate_trigger.connect(self.on_run_preset_translate)
        self.titleBar.run_preset_inpaint_trigger.connect(self.on_run_preset_inpaint)
        self.titleBar.video_translator_trigger.connect(self.on_video_translator)
        self.titleBar.subtitle_file_translator_trigger.connect(self.on_subtitle_file_translator)
        self.titleBar.video_subtitle_editor_trigger.connect(self.on_video_subtitle_editor)
        self.titleBar.keyboard_shortcuts_trigger.connect(self.open_shortcuts_dialog)
        self.titleBar.theme_customizer_trigger.connect(self.open_theme_customizer)
        self.titleBar.context_menu_options_trigger.connect(self.shortcutContextMenuOptions)
        self.titleBar.help_doc_trigger.connect(self.on_help_documentation)
        self.titleBar.help_about_trigger.connect(self.on_help_about)
        self.titleBar.help_update_from_github_trigger.connect(self.on_update_from_github)

        sc = getattr(pcfg, "shortcuts", None) or {}
        self._shortcuts_list = []  # (action_id, default_key, QShortcut)

        def _mk_shortcut(action_id: str, default_key: str, slot):
            key = get_shortcut(action_id, sc) or default_key
            q = QShortcut(QKeySequence.fromString(key) if key else QKeySequence(), self)
            q.activated.connect(slot)
            self._shortcuts_list.append((action_id, default_key, q))
            return q

        _mk_shortcut("go.prev_page_alt", "A", self.shortcutBefore)
        _mk_shortcut("go.prev_page", "PgUp", self.shortcutBefore)
        _mk_shortcut("go.next_page_alt", "D", self.shortcutNext)
        _mk_shortcut("go.next_page", "PgDown", self.shortcutNext)
        _mk_shortcut("canvas.textblock_mode", "W", self.shortcutTextblock)
        _mk_shortcut("canvas.zoom_in", "Ctrl++", self.canvas.gv.scale_up_signal.emit)
        _mk_shortcut("canvas.zoom_out", "Ctrl+-", self.canvas.gv.scale_down_signal.emit)
        _mk_shortcut("canvas.delete", "Ctrl+D", self.shortcutCtrlD)
        _mk_shortcut("canvas.space", "Space", self.shortcutSpace)
        sel_all_shortcut = _mk_shortcut("canvas.select_all", "Ctrl+A", self.shortcutSelectAll)
        sel_all_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        _mk_shortcut("canvas.escape", "Escape", self.shortcutEscape)
        _mk_shortcut("edit.omni_search", "Ctrl+P", self.shortcutOmniSearch)
        # Widget-level Ctrl+A when canvas view has focus: select all text blocks on canvas
        self._canvas_select_all_shortcut = QShortcut(QKeySequence(QKeySequence.StandardKey.SelectAll), self.canvas.gv)
        self._canvas_select_all_shortcut.setContext(Qt.ShortcutContext.WidgetShortcut)
        self._canvas_select_all_shortcut.activated.connect(self.on_select_all_canvas)
        _mk_shortcut("format.bold", "Ctrl+B", self.shortcutBold)
        _mk_shortcut("format.italic", "Ctrl+I", self.shortcutItalic)
        _mk_shortcut("format.underline", "Ctrl+U", self.shortcutUnderline)
        _mk_shortcut("format.font_size_up", "Ctrl+Alt+Up", self.shortcutFontSizeUp)
        _mk_shortcut("format.font_size_down", "Ctrl+Alt+Down", self.shortcutFontSizeDown)
        _mk_shortcut("canvas.delete_line", "Delete", self.shortcutDelete)
        _mk_shortcut("canvas.create_textbox", "Ctrl+Shift+N", self.shortcutCreateTextbox)
        _mk_shortcut("view.context_menu_options", "Ctrl+Shift+O", self.shortcutContextMenuOptions)
        _mk_shortcut("format.apply", "", self.shortcutFormatApply)
        _mk_shortcut("format.layout", "", self.shortcutFormatLayout)
        _mk_shortcut("format.fit_to_bubble", "", self.shortcutFitToBubble)
        _mk_shortcut("format.auto_fit", "", self.shortcutFormatAutoFit)
        _mk_shortcut("format.auto_fit_binary", "", self.shortcutFormatAutoFitBinary)
        _mk_shortcut("format.balloon_shape_auto", "", self.shortcutBalloonShapeAuto)
        _mk_shortcut("format.resize_to_fit_content", "", self.shortcutResizeToFitContent)

        drawpanel_shortcuts = [
            ("draw.hand", "H", "hand"),
            ("draw.inpaint", "J", "inpaint"),
            ("draw.pen", "B", "pen"),
            ("draw.rect", "R", "rect"),
        ]
        for action_id, default_key, tool_name in drawpanel_shortcuts:
            key = get_shortcut(action_id, sc) or default_key
            shortcut = QShortcut(QKeySequence.fromString(key) if key else QKeySequence(), self)
            shortcut.activated.connect(partial(self.drawingPanel.shortcutSetCurrentToolByName, tool_name))
            self.drawingPanel.setShortcutTip(tool_name, key or default_key)
            self._shortcuts_list.append((action_id, default_key, shortcut))
        self._draw_shortcut_tools = drawpanel_shortcuts

    def shortcutOmniSearch(self):
        """Focus the top-bar omni search box (menus/settings/canvas)."""
        try:
            box = getattr(self.titleBar, "omniSearch", None)
            if box is None:
                return
            # omniSearch is a QLineEdit
            box.setFocus(Qt.FocusReason.ShortcutFocusReason)
            try:
                box.selectAll()
            except Exception:
                pass
        except Exception:
            return

    def apply_shortcuts(self, shortcuts_dict=None):
        """Apply keyboard shortcuts from config (action_id -> key string)."""
        sc = shortcuts_dict if shortcuts_dict is not None else getattr(pcfg, "shortcuts", None) or {}
        for action_id, default_key, qshortcut in self._shortcuts_list:
            key = sc.get(action_id) or default_key
            qshortcut.setKey(QKeySequence.fromString(key) if key else QKeySequence())
        for action_id, default_key, tool_name in self._draw_shortcut_tools:
            key = sc.get(action_id) or default_key
            self.drawingPanel.setShortcutTip(tool_name, key or default_key)
        self.leftBar.apply_shortcuts(sc)
        self.titleBar.apply_shortcuts(sc)

    def open_shortcuts_dialog(self):
        dlg = ShortcutsDialog(self)
        dlg.shortcuts_changed.connect(self.apply_shortcuts)
        dlg.show()

    def open_theme_customizer(self):
        from .theme_customizer_dialog import ThemeCustomizerDialog
        dlg = ThemeCustomizerDialog(self)
        dlg.theme_applied.connect(self._on_theme_customizer_applied)
        dlg.show()

    def _on_theme_customizer_applied(self, font_family: str, font_size: int):
        self.titleBar.darkModeAction.setChecked(pcfg.darkmode)
        self.titleBar.themeLightAction.setChecked(not pcfg.darkmode)
        self.titleBar.themeDarkAction.setChecked(pcfg.darkmode)
        self.titleBar.bubblyUIAction.setChecked(getattr(pcfg, 'bubbly_ui', True))
        self.resetStyleSheet()
        # Font is applied in resetStyleSheet() -> _apply_app_font() (pcfg already saved by dialog)

    def on_help_documentation(self):
        """Open project README (#126 Help menu)."""
        root = osp.dirname(osp.dirname(osp.abspath(__file__)))
        readme = osp.join(root, 'README.md')
        if osp.isfile(readme):
            if sys.platform == 'win32':
                os.startfile(readme)
            elif sys.platform == 'darwin':
                subprocess.run(['open', readme], check=False)
            else:
                subprocess.run(['xdg-open', readme], check=False)
        else:
            create_info_dialog({'title': self.tr('Documentation'), 'text': self.tr('README.md not found.')})

    def on_help_about(self):
        """Show About dialog (#126 Help menu)."""
        from launch import VERSION
        QMessageBox.about(
            self,
            self.tr('About'),
            self.tr('BallonsTranslatorPro — community fork') + f' {VERSION}\n\n' + self.tr('Deep learning–assisted comic/manga translation.')
        )

    # --- Omni search jump helpers (TitleBar search box) ---
    def jump_to_config_item(self, idx0: int, idx1: int) -> None:
        """Open Config UI (if needed) and focus the requested config sub-block."""
        try:
            if hasattr(self, "leftBar") and hasattr(self.leftBar, "configChecker"):
                if not self.leftBar.configChecker.isChecked():
                    self.leftBar.configChecker.setChecked(True)
                    try:
                        self.setupConfigUI()
                    except Exception:
                        pass
            cp = getattr(self, "configPanel", None)
            if cp is None:
                return
            try:
                cp.configTable.setCurrentItem(idx0, idx1)
            except Exception:
                pass
            try:
                cp.onTableItemPressed(idx0, idx1)
            except Exception:
                # Fallback: emit via table signal path
                try:
                    cp.configTable.tableitem_pressed.emit(idx0, idx1)
                except Exception:
                    pass
        except Exception:
            return

    def jump_to_canvas_block(self, block_idx: int) -> None:
        """Ensure canvas/text UI is visible and select a text block by index."""
        try:
            if hasattr(self, "leftBar") and hasattr(self.leftBar, "imgTransChecker"):
                if not self.leftBar.imgTransChecker.isChecked():
                    self.leftBar.imgTransChecker.setChecked(True)
                    try:
                        self.setupImgTransUI()
                    except Exception:
                        pass
            # Switch to main canvas view (index 0 = imgtrans)
            try:
                if hasattr(self, "centralStackWidget") and self.centralStackWidget.currentIndex() != 0:
                    self.centralStackWidget.setCurrentIndex(0)
            except Exception:
                pass
            if not getattr(self, "st_manager", None) or not getattr(self.st_manager, "textblk_item_list", None):
                return
            if block_idx < 0 or block_idx >= len(self.st_manager.textblk_item_list):
                return
            self.rightComicTransStackPanel.setCurrentIndex(1)
            self.canvas.block_selection_signal = True
            self.canvas.clearSelection()
            blk_item = self.st_manager.textblk_item_list[block_idx]
            blk_item.setSelected(True)
            self.canvas.block_selection_signal = False
            try:
                self.st_manager.textEditList.set_selected_list([block_idx])
            except Exception:
                pass
        except Exception:
            return

    def jump_to_text_panel(self) -> None:
        """Switch right-side stack to the text/format panel and ensure main canvas view is visible."""
        try:
            if hasattr(self, "leftBar") and hasattr(self.leftBar, "imgTransChecker"):
                if not self.leftBar.imgTransChecker.isChecked():
                    self.leftBar.imgTransChecker.setChecked(True)
                    try:
                        self.setupImgTransUI()
                    except Exception:
                        pass
            try:
                if hasattr(self, "centralStackWidget") and self.centralStackWidget.currentIndex() != 0:
                    self.centralStackWidget.setCurrentIndex(0)
            except Exception:
                pass
            # Right panel: index 1 is the text/format panel in rightComicTransStackPanel
            try:
                if hasattr(self, "rightComicTransStackPanel"):
                    self.rightComicTransStackPanel.setCurrentIndex(1)
            except Exception:
                pass
        except Exception:
            return

    def on_update_from_github(self):
        """Pull latest changes from GitHub (Help menu). Only updates tracked files; config and local files are preserved."""
        from .custom_widget import ProgressMessageBox
        progress = ProgressMessageBox(self.tr('Update from GitHub'))
        progress.setTaskName(self.tr('Checking for updates...'))
        progress.updateTaskProgress(0)
        progress.show()
        thread = GitUpdateThread(shared.PROGRAM_PATH)

        def on_finished(success: bool, message: str):
            progress.hide()
            progress.deleteLater()
            if hasattr(self, '_git_update_threads') and thread in self._git_update_threads:
                self._git_update_threads.remove(thread)
            if success:
                create_info_dialog({'title': self.tr('Update from GitHub'), 'text': message})
            else:
                create_error_dialog(RuntimeError(message), self.tr('Update from GitHub'), 'GitUpdateThread')

        thread.finished_with_result.connect(on_finished)
        thread.start()
        if not hasattr(self, '_git_update_threads'):
            self._git_update_threads = []
        self._git_update_threads.append(thread)

    def _spellcheck_get_blocks(self):
        """Return [(block_idx, text_lines, trans_lines), ...] for current page (PR #974).
        Uses the scene's block list so the panel sees the same text blocks as the canvas/text panel.
        """
        # Prefer blocks from scene manager (current page as displayed); fallback to project
        if getattr(self, 'st_manager', None) and getattr(self.st_manager, 'textblk_item_list', None):
            result = []
            for blk_item in self.st_manager.textblk_item_list:
                blk = getattr(blk_item, 'blk', None)
                if blk is None:
                    continue
                text_lines = getattr(blk, 'text', None)
                if text_lines is None:
                    text_lines = []
                elif isinstance(text_lines, str):
                    text_lines = text_lines.split('\n') if text_lines else []
                trans = getattr(blk, 'translation', None) or ''
                trans_lines = trans.split('\n') if isinstance(trans, str) else []
                result.append((blk_item.idx, text_lines, trans_lines))
            return result
        if not getattr(self, 'imgtrans_proj', None) or getattr(self.imgtrans_proj, 'is_empty', True):
            return []
        page = self.imgtrans_proj.pages.get(self.imgtrans_proj.current_img, [])
        return [
            (i, self._spellcheck_normalize_lines(getattr(blk, 'text', None)), (getattr(blk, 'translation', None) or '').split('\n'))
            for i, blk in enumerate(page)
        ]

    def _spellcheck_normalize_lines(self, text_attr):
        """Return a list of strings from block text (list or string)."""
        if text_attr is None:
            return []
        if isinstance(text_attr, str):
            return text_attr.split('\n') if text_attr else []
        return [line if isinstance(line, str) else str(line) for line in text_attr]

    def _spellcheck_apply_replacement(self, block_idx: int, line_idx: int, new_line: str, is_translation: bool):
        """Apply spell check replacement to block (PR #974)."""
        page = self.imgtrans_proj.pages.get(self.imgtrans_proj.current_img, [])
        if block_idx < 0 or block_idx >= len(page):
            return
        blk = page[block_idx]
        if is_translation:
            lines = (blk.translation or '').split('\n')
            if line_idx < len(lines):
                lines[line_idx] = new_line
                blk.translation = _merge_stub_lines_in_string('\n'.join(lines))
                if block_idx < len(self.st_manager.pairwidget_list):
                    self.st_manager.pairwidget_list[block_idx].e_trans.setPlainText(blk.translation)
        else:
            if hasattr(blk, 'text') and blk.text is not None and line_idx < len(blk.text):
                blk.text[line_idx] = new_line
                if block_idx < len(self.st_manager.pairwidget_list):
                    self.st_manager.pairwidget_list[block_idx].e_source.setPlainText(blk.get_text())
        self.canvas.setProjSaveState(False)

    def shortcutNext(self):
        sender: QShortcut = self.sender()
        if isinstance(sender, QShortcut):
            if sender.key() == QKEY.Key_D:
                if self.canvas.editing_textblkitem is not None:
                    return
        if self.centralStackWidget.currentIndex() == 0:
            focus_widget = self.app.focusWidget()
            if self.st_manager.is_editting():
                self.st_manager.on_switch_textitem(1)
            elif isinstance(focus_widget, (SourceTextEdit, TransTextEdit)):
                self.st_manager.on_switch_textitem(1, current_editing_widget=focus_widget)
            else:
                index = self.pageList.currentIndex()
                page_count = self.pageList.count()
                if index.isValid():
                    row = index.row()
                    row = (row + 1) % page_count
                    self.pageList.setCurrentRow(row)

    def shortcutBefore(self):
        sender: QShortcut = self.sender()
        if isinstance(sender, QShortcut):
            if sender.key() == QKEY.Key_A:
                if self.canvas.editing_textblkitem is not None:
                    return
        if self.centralStackWidget.currentIndex() == 0:
            focus_widget = self.app.focusWidget()
            if self.st_manager.is_editting():
                self.st_manager.on_switch_textitem(-1)
            elif isinstance(focus_widget, (SourceTextEdit, TransTextEdit)):
                self.st_manager.on_switch_textitem(-1, current_editing_widget=focus_widget)
            else:
                index = self.pageList.currentIndex()
                page_count = self.pageList.count()
                if index.isValid():
                    row = index.row()
                    row = (row - 1 + page_count) % page_count
                    self.pageList.setCurrentRow(row)

    def shortcutTextedit(self):
        if self.centralStackWidget.currentIndex() == 0:
            self.bottomBar.texteditChecker.click()

    def shortcutTextblock(self):
        if self.centralStackWidget.currentIndex() == 0:
            if self.bottomBar.texteditChecker.isChecked():
                self.bottomBar.textblockChecker.click()

    def shortcutDrawboard(self):
        if self.centralStackWidget.currentIndex() == 0:
            self.bottomBar.paintChecker.click()

    def shortcutSpellCheckPanel(self):
        """Show Spell check panel (PR #974)."""
        if self.centralStackWidget.currentIndex() != 0:
            return
        if self.rightComicTransStackPanel.isHidden():
            self.rightComicTransStackPanel.show()
        self.rightComicTransStackPanel.setCurrentIndex(2)
        self.bottomBar.spellCheckChecker.setChecked(True)

    def _on_spellcheck_panel_close(self):
        """Switch right panel from spell check back to text panel."""
        self.rightComicTransStackPanel.setCurrentIndex(1)
        self.bottomBar.spellCheckChecker.blockSignals(True)
        self.bottomBar.spellCheckChecker.setChecked(False)
        self.bottomBar.spellCheckChecker.blockSignals(False)

    def on_spellcheck_panel_toggled(self):
        """Show or hide Spell check panel from bottom bar toggle."""
        if self.bottomBar.spellCheckChecker.isChecked():
            if self.rightComicTransStackPanel.isHidden():
                self.rightComicTransStackPanel.show()
            self.rightComicTransStackPanel.setCurrentIndex(2)
        else:
            self.rightComicTransStackPanel.setCurrentIndex(1)

    def _on_spellcheck_focus_block(self, block_idx: int, line_idx: int, is_translation: bool):
        """After spell check replace: switch to text panel and select the block so the change is visible."""
        self.rightComicTransStackPanel.setCurrentIndex(1)
        if not self.st_manager.textblk_item_list or block_idx < 0 or block_idx >= len(self.st_manager.textblk_item_list):
            return
        self.canvas.block_selection_signal = True
        self.canvas.clearSelection()
        blk_item = self.st_manager.textblk_item_list[block_idx]
        blk_item.setSelected(True)
        self.canvas.block_selection_signal = False
        self.st_manager.textEditList.set_selected_list([block_idx])
        self.st_manager.formatpanel.set_textblk_item(blk_item)
        self.canvas.gv.ensureVisible(blk_item)
        if block_idx < len(self.st_manager.pairwidget_list):
            edit = self.st_manager.pairwidget_list[block_idx].e_trans if is_translation else self.st_manager.pairwidget_list[block_idx].e_source
            edit.setFocus()
            edit.blockSignals(True)
            lines = edit.toPlainText().split('\n')
            if line_idx < len(lines):
                from qtpy.QtGui import QTextCursor
                cursor = edit.textCursor()
                pos = 0
                for i, line in enumerate(lines):
                    if i == line_idx:
                        cursor.setPosition(pos)
                        cursor.movePosition(QTextCursor.MoveOperation.EndOfLine, QTextCursor.MoveMode.KeepAnchor)
                        edit.setTextCursor(cursor)
                        break
                    pos += len(line) + 1
            edit.blockSignals(False)

    def shortcutCtrlD(self):
        if self.centralStackWidget.currentIndex() == 0:
            if self.drawingPanel.isVisible():
                if self.drawingPanel.currentTool == self.drawingPanel.rectTool:
                    self.drawingPanel.rectPanel.delete_btn.click()
            elif self.canvas.textEditMode():
                self.canvas.delete_textblks.emit(0)

    def shortcutSelectAll(self):
        """Ctrl+A: select all pages when page list has focus, else select all text blocks on canvas."""
        focus = QApplication.focusWidget()
        if focus is self.pageList or (focus and self.pageList.isAncestorOf(focus)):
            self.pageList.selectAll()
            return
        if self.centralStackWidget.currentIndex() == 1:
            self.on_select_all_canvas()

    def eventFilter(self, obj, event):
        """Intercept Ctrl+A when canvas (or its viewport) has focus; select all text blocks on canvas. Skip when editing a text block so Ctrl+A selects all text in that block."""
        if event.type() == QEvent.Type.KeyPress:
            e = event
            if e.key() == Qt.Key.Key_A and e.modifiers() in (Qt.KeyboardModifier.ControlModifier, Qt.KeyboardModifier.MetaModifier):
                if self.centralStackWidget.currentIndex() == 1:
                    if self.canvas.editing_textblkitem is not None:
                        return super().eventFilter(obj, event)
                    if obj is self.canvas.gv or obj is self.canvas.gv.viewport():
                        self.on_select_all_canvas()
                        return True
                    focus = QApplication.focusWidget()
                    on_canvas = (
                        focus is self.canvas.gv or (focus and isinstance(focus, QWidget) and self.canvas.gv.isAncestorOf(focus))
                    )
                    if on_canvas:
                        self.on_select_all_canvas()
                        return True
        return super().eventFilter(obj, event)

    def shortcutSpace(self):
        if self.centralStackWidget.currentIndex() == 0:
            if self.drawingPanel.isVisible():
                if self.drawingPanel.currentTool == self.drawingPanel.rectTool:
                    self.drawingPanel.rectPanel.inpaint_btn.click()

    def shortcutBold(self):
        if self.textPanel.formatpanel.isVisible():
            self.textPanel.formatpanel.formatBtnGroup.boldBtn.click()

    def shortcutDelete(self):
        if self.canvas.gv.isVisible():
            self.canvas.delete_textblks.emit(1)

    def shortcutItalic(self):
        if self.textPanel.formatpanel.isVisible():
            self.textPanel.formatpanel.formatBtnGroup.italicBtn.click()

    def shortcutUnderline(self):
        if self.textPanel.formatpanel.isVisible():
            self.textPanel.formatpanel.formatBtnGroup.underlineBtn.click()

    def shortcutFontSizeUp(self):
        """Increase font size for selected text blocks."""
        if not self.textPanel.formatpanel.isVisible():
            return
        blks = self.canvas.selected_text_items()
        if not blks:
            return
        gf = self.textPanel.formatpanel.global_format
        from ui import fontformat_commands as FC
        FC.ffmt_change_rel_font_size('rel_font_size', 1.25, gf, True, blks, clip_size=True)
        self.st_manager.updateSceneTextitems()
        self.canvas.setProjSaveState(False)

    def shortcutFontSizeDown(self):
        """Decrease font size for selected text blocks."""
        if not self.textPanel.formatpanel.isVisible():
            return
        blks = self.canvas.selected_text_items()
        if not blks:
            return
        gf = self.textPanel.formatpanel.global_format
        from ui import fontformat_commands as FC
        FC.ffmt_change_rel_font_size('rel_font_size', 0.75, gf, True, blks, clip_size=True)
        self.st_manager.updateSceneTextitems()
        self.canvas.setProjSaveState(False)

    def shortcutCreateTextbox(self):
        """Create a default-size text box at cursor (Canvas)."""
        if self.centralStackWidget.currentIndex() == 0 and self.canvas.gv.isVisible():
            self.canvas.create_textbox_at_cursor()

    def shortcutContextMenuOptions(self):
        """Open Context menu options dialog (View)."""
        dlg = ContextMenuConfigDialog(self)
        dlg.exec()

    def shortcutFormatApply(self):
        """Apply font formatting to selection (Format shortcut)."""
        if self.centralStackWidget.currentIndex() == 0 and self.canvas.gv.isVisible():
            self.canvas.format_textblks.emit()

    def shortcutFormatLayout(self):
        """Auto layout selected text blocks (Format shortcut)."""
        if self.centralStackWidget.currentIndex() == 0 and self.canvas.gv.isVisible() and self.canvas.selected_text_items():
            self.canvas.layout_textblks.emit()

    def shortcutFitToBubble(self):
        """Fit to bubble for selection (Format shortcut)."""
        if self.centralStackWidget.currentIndex() == 0 and self.canvas.gv.isVisible() and self.canvas.selected_text_items():
            self.canvas.layout_textblks.emit()

    def shortcutFormatAutoFit(self):
        """Auto fit font size to box (Format shortcut)."""
        if self.centralStackWidget.currentIndex() == 0 and self.canvas.gv.isVisible() and self.canvas.selected_text_items():
            self.canvas.auto_fit_font_signal.emit()

    def shortcutFormatAutoFitBinary(self):
        """Auto fit font size binary search (Format shortcut)."""
        if self.centralStackWidget.currentIndex() == 0 and self.canvas.gv.isVisible() and self.canvas.selected_text_items():
            self.canvas.auto_fit_binary_signal.emit()

    def shortcutBalloonShapeAuto(self):
        """Set balloon shape to Auto (Format shortcut)."""
        if self.centralStackWidget.currentIndex() == 0 and self.canvas.gv.isVisible() and self.canvas.selected_text_items():
            self.canvas.set_balloon_shape_signal.emit("auto")

    def shortcutResizeToFitContent(self):
        """Resize selected text box(es) to fit content (Format shortcut)."""
        if self.centralStackWidget.currentIndex() == 0 and self.canvas.gv.isVisible() and self.canvas.selected_text_items():
            self.canvas.resize_to_fit_content_signal.emit()

    def on_redo(self):
        self.canvas.redo()

    def on_undo(self):
        self.canvas.undo()

    def on_page_search(self):
        if self.canvas.gv.isVisible():
            fo = self.app.focusObject()
            sel_text = ''
            tgt_edit = None
            blkitem = self.canvas.editing_textblkitem
            if fo == self.canvas.gv and blkitem is not None:
                sel_text = blkitem.textCursor().selectedText()
                tgt_edit = self.st_manager.pairwidget_list[blkitem.idx].e_trans
            elif isinstance(fo, QTextEdit) or isinstance(fo, QPlainTextEdit):
                sel_text = fo.textCursor().selectedText()
                if isinstance(fo, SourceTextEdit):
                    tgt_edit = fo
            se = self.canvas.search_widget.search_editor
            se.setFocus()
            if sel_text != '':
                se.setPlainText(sel_text)
                cursor = se.textCursor()
                cursor.select(QTextCursor.SelectionType.Document)
                se.setTextCursor(cursor)

            if self.canvas.search_widget.isHidden():
                self.canvas.search_widget.show()
            self.canvas.search_widget.setCurrentEditor(tgt_edit)

    def on_global_search(self):
        if self.canvas.gv.isVisible():
            if not self.leftBar.globalSearchChecker.isChecked():
                self.leftBar.globalSearchChecker.click()
            fo = self.app.focusObject()
            sel_text = ''
            blkitem = self.canvas.editing_textblkitem
            if fo == self.canvas.gv and blkitem is not None:
                sel_text = blkitem.textCursor().selectedText()
            elif isinstance(fo, QTextEdit) or isinstance(fo, QPlainTextEdit):
                sel_text = fo.textCursor().selectedText()
            se = self.global_search_widget.search_editor
            se.setFocus()
            if sel_text != '':
                se.setPlainText(sel_text)
                cursor = se.textCursor()
                cursor.select(QTextCursor.SelectionType.Document)
                se.setTextCursor(cursor)
                
                self.global_search_widget.commit_search()

    def show_pre_MT_keyword_window(self):
        self.mtPreSubWidget.show()

    def show_MT_keyword_window(self):
        self.mtSubWidget.show()

    def show_translation_context_dialog(self):
        """Open dialog to edit project series context path and glossary."""
        proj = self.imgtrans_proj
        if proj is None or getattr(proj, "directory", None) is None:
            create_info_dialog(
                self.tr("Open a project first (File → Open Folder / Open Project) to set project translation context."),
                self.tr("Translation context"),
                "TranslationContext",
            )
            return
        dlg = TranslationContextDialog(proj, self)
        dlg.exec()


    def show_OCR_keyword_window(self):
        self.ocrSubWidget.show()

    def on_open_merge_tool(self):
        """Open the region merge tool dialog."""
        if not hasattr(self, 'merge_dialog') or self.merge_dialog is None:
            from .merge_dialog import MergeDialog
            from qtpy.QtCore import QThread
            from qtpy.QtWidgets import QProgressDialog
            from utils import merger
            
            self.merge_dialog = MergeDialog(self)
            self.merge_dialog.run_current_clicked.connect(lambda: self.run_merge_task(on_current=True))
            self.merge_dialog.run_all_clicked.connect(lambda: self.run_merge_task(on_current=False))
        
        if self.merge_dialog.isVisible():
            self.merge_dialog.raise_()
            self.merge_dialog.activateWindow()
        else:
            self.merge_dialog.show()

    def on_open_manga_source(self):
        """Open the Manga / Comic source dialog (search, chapters, download)."""
        if not hasattr(self, 'manga_source_dialog') or self.manga_source_dialog is None:
            from .manga_source_dialog import MangaSourceDialog
            self.manga_source_dialog = MangaSourceDialog(self)
            self.manga_source_dialog.open_folder_requested.connect(self.OpenProj)
        if self.manga_source_dialog.isVisible():
            self.manga_source_dialog.raise_()
            self.manga_source_dialog.activateWindow()
        else:
            self.manga_source_dialog.show()

    def on_open_batch_queue(self):
        """Open the Batch queue dialog (multiple folders, Pause/Cancel)."""
        if self._batch_queue_dialog is None:
            self._batch_queue_dialog = BatchQueueDialog(self)
            self._batch_queue_dialog.start_queue_requested.connect(self._on_batch_start)
            self._batch_queue_dialog.pause_requested.connect(self.module_manager.requestPausePipeline)
            self._batch_queue_dialog.resume_requested.connect(self.module_manager.requestResumePipeline)
            self._batch_queue_dialog.cancel_requested.connect(self._on_batch_cancel)
            self.batch_queue_empty.connect(self._batch_queue_dialog.set_queue_empty)
            self.batch_queue_cancelled.connect(self._batch_queue_dialog.set_queue_cancelled)
            self.batch_queue_item_started.connect(self._on_batch_item_started)
        if self._batch_queue_dialog.isVisible():
            self._batch_queue_dialog.raise_()
            self._batch_queue_dialog.activateWindow()
        else:
            self._batch_queue_dialog.show()

    def _on_batch_start(self, paths: list, skip_ignored_pages: bool = True):
        self.run_batch(paths, skip_ignored_pages=skip_ignored_pages)

    def _on_batch_cancel(self):
        self._batch_cancelled = True
        self.exec_dirs = []
        # Cleanup any extracted ZIP temp dirs
        try:
            self._zip_batch.cleanup_all()
        except Exception:
            pass
        self.module_manager.stopImgtransPipeline()

    def _on_batch_item_started(self, path: str):
        if self._batch_queue_dialog is not None:
            self._batch_queue_dialog.set_running_state(True, paused=False, current_path=path)
            if self._batch_queue_dialog.list_widget.count() > 0:
                self._batch_queue_dialog.list_widget.takeItem(0)

    def on_open_manage_models(self):
        """Open the Manage models dialog (check downloaded models, download selected)."""
        dlg = ModelManagerDialog(self)
        dlg.exec()

    def _reset_modules_to_core_defaults(self):
        """Set detector, OCR, inpainter, and translator to core defaults after models are downloaded."""
        pcfg.module.textdetector = 'ctd'
        pcfg.module.ocr = 'manga_ocr'
        pcfg.module.inpainter = 'aot'
        pcfg.module.translator = 'google'
        save_config()
        self.module_manager.setTextDetector('ctd')
        self.module_manager.setOCR('manga_ocr')
        self.module_manager.setInpainter('aot')
        self.module_manager.setTranslator('google')

    def _run_deferred_model_download(self):
        """Run initial model download after window is shown (set by launch.py so app opens first)."""
        package_ids = getattr(pcfg, 'model_packages_enabled', None)
        if package_ids is not None and len(package_ids) == 0:
            return
        dlg = ModelDownloadProgressDialog(self)
        thread = ModelDownloadThread(self)
        dlg.set_thread(thread)

        def on_finished(success: bool, message: str):
            dlg.accept()
            if success:
                # Do not reset modules or show popup on normal launch; only Tools → Retry does that.
                pass
            else:
                tip = self.tr('If the connectivity check is slow or fails, set DISABLE_MODEL_SOURCE_CHECK=True and retry. See docs/TROUBLESHOOTING.md.')
                create_error_dialog(
                    Exception(message),
                    self.tr('Model download failed. You can retry from Tools → Retry model downloads.') + '\n\n' + tip,
                    'ModelDownloadThread'
                )

        thread.finished_with_result.connect(on_finished)
        thread.start()
        dlg.exec()

    def on_retry_model_downloads(self):
        """Retry downloading model packages (Tools → Retry model downloads). Useful when first-run download failed."""
        package_ids = getattr(pcfg, 'model_packages_enabled', None)
        if package_ids is not None and len(package_ids) == 0:
            create_info_dialog({
                'title': self.tr('No model packages selected.'),
                'text': self.tr('Use Tools → Manage models to download individual models, or restart the app to choose packages again.'),
            })
            return
        dlg = ModelDownloadProgressDialog(self)
        thread = ModelDownloadThread(self)
        dlg.set_thread(thread)

        def on_finished(success: bool, message: str):
            dlg.accept()
            if success:
                self._reset_modules_to_core_defaults()
                create_info_dialog({
                    'title': self.tr('Download complete.'),
                    'text': self.tr('Model packages have been downloaded. You can use the pipeline now.'),
                })
            else:
                tip = self.tr('If the connectivity check is slow or fails, set DISABLE_MODEL_SOURCE_CHECK=True and retry. See docs/TROUBLESHOOTING.md.')
                create_error_dialog(
                    Exception(message),
                    self.tr('Model download failed. You can retry from Tools → Retry model downloads.') + '\n\n' + tip,
                    'ModelDownloadThread'
                )

        thread.finished_with_result.connect(on_finished)
        thread.start()
        dlg.exec()

    def on_release_model_caches(self):
        """Unload all models and clear pipeline cache to free GPU/RAM (Tools → Release model caches)."""
        import gc
        if not (shared.HEADLESS or shared.HEADLESS_CONTINUOUS):
            self.module_manager.unload_all_models()
            try:
                from utils.pipeline_cache import get_pipeline_cache
                cache = get_pipeline_cache(True)
                if cache is not None:
                    cache.clear()
            except Exception:
                pass
            gc.collect()
            self._reset_idle_unload_timer()

    def on_clear_pipeline_caches(self):
        """Clear in-session OCR/translation caches (Tools → Clear OCR and translation caches)."""
        try:
            from utils.pipeline_cache import get_pipeline_cache
            cache = get_pipeline_cache(True)
            if cache is not None:
                cache.clear()
        except Exception:
            pass
        try:
            from utils.pipeline_cache_manager import clear_block_level_caches
            clear_block_level_caches()
        except Exception:
            pass

    def run_merge_task(self, on_current=False):
        """Run the region merge task."""
        from utils import merger
        from qtpy.QtWidgets import QMessageBox
        
        if self.imgtrans_proj.is_empty:
            QMessageBox.warning(self, "Warning", "Please open a project first.")
            return
        
        config = self.merge_dialog.get_config()
        
        if on_current:
            # Run on current page — operate in memory, no file I/O
            from utils.textblock import TextBlock
            
            current_img = self.imgtrans_proj.current_img
            if not current_img:
                QMessageBox.warning(self, "Warning", "No current page.")
                return

            # Get text blocks for current page from memory
            if current_img not in self.imgtrans_proj.pages:
                QMessageBox.warning(self, "Warning", "Current page data not found.")
                return

            textblocks = self.imgtrans_proj.pages[current_img]
            if not textblocks:
                QMessageBox.warning(self, "Info", "Current page has no text blocks.")
                return
            
            # Convert to dict format (merger expects dicts)
            initial_shapes = [blk.to_dict() for blk in textblocks]
            
            initial_count = len(initial_shapes)
            mode = config.get("MERGE_MODE", "NONE")
            total_merged = 0
            
            # Perform merge in memory
            if mode == "VERTICAL":
                final_shapes, count = merger.perform_merge(initial_shapes, "VERTICAL", config)
                total_merged += count
            elif mode == "HORIZONTAL":
                final_shapes, count = merger.perform_merge(initial_shapes, "HORIZONTAL", config)
                total_merged += count
            elif mode == "VERTICAL_THEN_HORIZONTAL":
                temp, count1 = merger.perform_merge(initial_shapes, "VERTICAL", config)
                final_shapes, count2 = merger.perform_merge(temp, "HORIZONTAL", config)
                total_merged += (count1 + count2)
            elif mode == "HORIZONTAL_THEN_VERTICAL":
                temp, count1 = merger.perform_merge(initial_shapes, "HORIZONTAL", config)
                final_shapes, count2 = merger.perform_merge(temp, "VERTICAL", config)
                total_merged += (count1 + count2)
            else:
                final_shapes = initial_shapes
            
            if total_merged > 0:
                # Convert dicts back to TextBlock and update memory
                self.imgtrans_proj.pages[current_img] = [TextBlock(**blk_dict) for blk_dict in final_shapes]
                self.canvas.updateCanvas()
                self.st_manager.updateSceneTextitems()
                final_count = len(final_shapes)
                QMessageBox.information(self, "Success", f"Merge done: {initial_count} -> {final_count} blocks (reduced by {initial_count - final_count})")
            else:
                raw_labels = set(s.get('label') for s in initial_shapes)
                label_list = sorted({str(lbl) for lbl in raw_labels if lbl})
                label_str = ", ".join(label_list) if label_list else "none"
                detail_msg = (
                    f"No blocks were merged.\n"
                    f"Total: {initial_count} text blocks.\n"
                    f"Label types: {label_str}\n\n"
                )
                detail_msg += "Suggestions:\n"
                detail_msg += "1. Try increasing max gap (e.g. 100–200)\n"
                detail_msg += "2. Lower min overlap ratio (e.g. 50–70%)\n"
                detail_msg += "3. Uncheck 'Enable exclude-from-merge labels'\n"
                detail_msg += "4. Check if labels are in the blacklist"
                QMessageBox.warning(self, "Info", detail_msg)
        else:
            # Run on all pages
            img_list = list(self.imgtrans_proj.pages.keys())
            if not img_list:
                QMessageBox.warning(self, "Warning", "Project has no images.")
                return

            json_path = self.imgtrans_proj.proj_path
            if not json_path or not osp.exists(json_path):
                QMessageBox.warning(self, "Warning", f"Project JSON not found: {json_path}")
                return

            self.run_merge_all_async(json_path, img_list, config)
    
    def run_merge_all_async(self, json_path, img_list, config):
        """Run merge on all pages in a background thread."""
        from .io_thread import MergeThread
        
        # Create merge thread if needed
        if not hasattr(self, 'merge_thread'):
            self.merge_thread = MergeThread()
            self.merge_thread.progress_changed.connect(self.on_merge_progress)
            self.merge_thread.merge_finished.connect(self.on_merge_finished)
            self.merge_thread.progress_bar.stop_clicked.connect(self.on_merge_stop)
        
        # Start merge
        if self.merge_thread.runMerge(json_path, img_list, config):
            self.merge_thread.progress_bar.zero_progress()
            self.merge_thread.progress_bar.show()

    def on_merge_progress(self, current, total):
        """Update merge progress."""
        progress = int(current / total * 100)
        self.merge_thread.progress_bar.updateTaskProgress(progress, f' {current}/{total}')
    
    def on_merge_stop(self):
        """Stop merge."""
        if hasattr(self, 'merge_thread'):
            self.merge_thread.requestStop()
            self.merge_thread.progress_bar.hide()
    
    def on_merge_finished(self, success_count, fail_count):
        """Merge finished."""
        self.merge_thread.progress_bar.hide()
        
        # Reload project
        try:
            json_path = self.imgtrans_proj.proj_path
            current_img = self.imgtrans_proj.current_img
            self.imgtrans_proj.load_from_json(json_path)
            if current_img and current_img in self.imgtrans_proj.pages:
                self.imgtrans_proj.set_current_img(current_img)
                self.canvas.updateCanvas()
                self.st_manager.updateSceneTextitems()
        except:
            pass
        
        # Show result
        total = success_count + fail_count
        QMessageBox.information(self, "Done", f"Region merge finished\nSuccess: {success_count}/{total}\nFailed: {fail_count}/{total}")

    def on_re_run_detection_only(self):
        """Run pipeline with only detection enabled (OCR, translate, inpaint disabled)."""
        if self.imgtrans_proj.is_empty:
            return
        self._detection_only_restore = (
            pcfg.module.enable_detect,
            pcfg.module.enable_ocr,
            pcfg.module.enable_translate,
            pcfg.module.enable_inpaint,
        )
        pcfg.module.enable_detect = True
        pcfg.module.enable_ocr = False
        pcfg.module.enable_translate = False
        pcfg.module.enable_inpaint = False
        for idx, sa in enumerate(self.titleBar.stageActions):
            sa.setChecked(pcfg.module.stage_enabled(idx))
        self.on_run_imgtrans()

    def on_re_run_ocr_only(self):
        """Run pipeline with only OCR enabled (re-recognize text, keep detection boxes)."""
        if self.imgtrans_proj.is_empty:
            return
        self._detection_only_restore = (
            pcfg.module.enable_detect,
            pcfg.module.enable_ocr,
            pcfg.module.enable_translate,
            pcfg.module.enable_inpaint,
        )
        pcfg.module.enable_detect = False
        pcfg.module.enable_ocr = True
        pcfg.module.enable_translate = False
        pcfg.module.enable_inpaint = False
        for idx, sa in enumerate(self.titleBar.stageActions):
            sa.setChecked(pcfg.module.stage_enabled(idx))
        self.on_run_imgtrans()

    def _on_idle_unload_timeout(self):
        if getattr(pcfg, 'unload_after_idle_minutes', 0) <= 0:
            return
        self.module_manager.unload_all_models()
        LOGGER.info('Models unloaded after idle timeout.')

    def _reset_idle_unload_timer(self):
        self._idle_unload_timer.stop()
        mins = getattr(pcfg, 'unload_after_idle_minutes', 0)
        if mins > 0:
            self._idle_unload_timer.start(mins * 60 * 1000)

    def on_run_preset_full(self):
        pcfg.module.enable_detect = True
        pcfg.module.enable_ocr = True
        pcfg.module.enable_translate = True
        pcfg.module.enable_inpaint = True
        for idx, sa in enumerate(self.titleBar.stageActions):
            sa.setChecked(True)
        pcfg.module.run_preset_name = 'Full'

    def on_run_preset_detect_ocr(self):
        pcfg.module.enable_detect = True
        pcfg.module.enable_ocr = True
        pcfg.module.enable_translate = False
        pcfg.module.enable_inpaint = False
        for idx, sa in enumerate(self.titleBar.stageActions):
            sa.setChecked(pcfg.module.stage_enabled(idx))
        pcfg.module.run_preset_name = 'Detect+OCR'

    def on_run_preset_translate(self):
        pcfg.module.enable_detect = False
        pcfg.module.enable_ocr = False
        pcfg.module.enable_translate = True
        pcfg.module.enable_inpaint = False
        for idx, sa in enumerate(self.titleBar.stageActions):
            sa.setChecked(pcfg.module.stage_enabled(idx))
        pcfg.module.run_preset_name = 'Translate'

    def on_run_preset_inpaint(self):
        pcfg.module.enable_detect = False
        pcfg.module.enable_ocr = False
        pcfg.module.enable_translate = False
        pcfg.module.enable_inpaint = True
        for idx, sa in enumerate(self.titleBar.stageActions):
            sa.setChecked(pcfg.module.stage_enabled(idx))
        pcfg.module.run_preset_name = 'Inpaint'

    def on_video_translator(self):
        """Open the Video translator dialog (Pipeline → Video translator...)."""
        from .video_translator_dialog import VideoTranslatorDialog
        # Keep a single instance so closing/hiding doesn't lose a running job's UI.
        dlg = getattr(self, "_video_translator_dlg", None)
        if dlg is None:
            dlg = VideoTranslatorDialog(self)
            self._video_translator_dlg = dlg
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def on_subtitle_file_translator(self):
        """Open subtitle file translator (Pipeline → Translate subtitle file…)."""
        from .subtitle_file_translator_dialog import SubtitleFileTranslatorDialog
        dlg = getattr(self, "_subtitle_file_translator_dlg", None)
        if dlg is None:
            dlg = SubtitleFileTranslatorDialog(self)
            self._subtitle_file_translator_dlg = dlg
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def on_video_subtitle_editor(self):
        """Open the Video Subtitle Editor (Pipeline → Video Subtitle Editor...)."""
        from .video_subtitle_editor import VideoSubtitleEditorWindow
        win = VideoSubtitleEditorWindow(self)
        win.show()

    def on_batch_export(self):
        """Export all pages as images (and optionally PDF) to a chosen folder."""
        if self.imgtrans_proj.is_empty:
            QMessageBox.information(self, self.tr('Export'), self.tr('Open a project first.'))
            return
        out_dir = QFileDialog.getExistingDirectory(self, self.tr('Select output folder'))
        if not out_dir:
            return
        self._do_batch_export(out_dir, ext=None)

    def on_batch_export_as(self):
        """Export all pages with chosen format (#126 Export as)."""
        if self.imgtrans_proj.is_empty:
            QMessageBox.information(self, self.tr('Export'), self.tr('Open a project first.'))
            return
        initial = self.imgtrans_proj.directory if self.imgtrans_proj.directory else None
        dlg = ExportFormatDialog(self, initial_dir=initial)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        export_as_zip = dlg.get_export_as_zip()
        zip_path = (dlg.get_zip_path() or '').strip()
        if export_as_zip and not zip_path:
            QMessageBox.information(self, self.tr('Export'), self.tr('Choose a path for the ZIP file.'))
            return
        out_dir = dlg.get_folder()
        if not export_as_zip and not out_dir:
            QMessageBox.information(self, self.tr('Export'), self.tr('Select an output folder.'))
            return
        temp_export_dir = None
        if export_as_zip:
            import tempfile
            temp_export_dir = tempfile.mkdtemp(prefix='ballons_export_')
            out_dir = temp_export_dir
        try:
            self._do_batch_export(
                out_dir,
                ext=dlg.get_extension(),
                also_pdf=dlg.get_also_pdf(),
                show_message=not export_as_zip,
                clean_after_export=dlg.get_clean_after_export() if not export_as_zip else False,
            )
            if export_as_zip and zip_path and osp.isdir(out_dir):
                import zipfile
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                    for root, _dirs, files in os.walk(out_dir):
                        for f in files:
                            path = osp.join(root, f)
                            zf.write(path, osp.relpath(path, out_dir))
                msg = self.tr('Exported as ZIP: {}').format(zip_path)
                if dlg.get_clean_after_export():
                    self.imgtrans_proj.clean_mask_and_inpainted_cache()
                    msg += '\n' + self.tr('Cache cleaned.')
                QMessageBox.information(self, self.tr('Export'), msg)
        finally:
            if temp_export_dir and osp.isdir(temp_export_dir):
                try:
                    import shutil
                    shutil.rmtree(temp_export_dir, ignore_errors=True)
                except Exception:
                    pass

    def _do_batch_export(self, out_dir: str, ext: str = None, also_pdf: bool = False, show_message: bool = True, clean_after_export: bool = False):
        try:
            from utils.io_utils import imread, imwrite
            result_dir = self.imgtrans_proj.result_dir()
            exported = 0
            missing = []
            exported_paths = []  # (pagename, path) in page order for PDF
            page_order = list(self.imgtrans_proj.pages.keys())
            for i, pagename in enumerate(page_order):
                result_path = self.imgtrans_proj.get_result_path(pagename)
                if osp.exists(result_path):
                    use_ext = ext if ext else osp.splitext(result_path)[1]
                    if use_ext not in ('.png', '.jpg', '.jpeg', '.webp', '.jxl'):
                        use_ext = '.png'
                    # Export with consistent 001, 002, 003 naming (same as manga download / natural sort)
                    dest = osp.join(out_dir, f"{i + 1:03d}{use_ext}")
                    img = imread(result_path)
                    kw = {'ext': use_ext, 'quality': pcfg.imgsave_quality}
                    if use_ext == '.webp' and getattr(pcfg, 'imgsave_webp_lossless', False):
                        kw['webp_lossless'] = True
                    imwrite(dest, img, **kw)
                    exported += 1
                    exported_paths.append((pagename, dest))
                else:
                    missing.append(pagename)
            msg = self.tr('Exported {0} page(s) to {1}.').format(exported, out_dir)
            if also_pdf and exported_paths:
                pdf_path = osp.join(out_dir, 'exported.pdf')
                try:
                    import img2pdf
                    with open(pdf_path, 'wb') as f:
                        # img2pdf expects image paths; order by page order
                        img_paths = [p for _, p in exported_paths]
                        f.write(img2pdf.convert(img_paths))
                    msg += '\n' + self.tr('PDF saved: {}').format(pdf_path)
                except ImportError:
                    msg += '\n' + self.tr('PDF export skipped. Install img2pdf: pip install img2pdf')
                except Exception as e:
                    LOGGER.exception(e)
                    msg += '\n' + self.tr('PDF export failed: {}').format(str(e))
            if missing:
                msg += '\n' + self.tr('Missing result for {0} page(s). Run pipeline first.').format(len(missing))
            if clean_after_export and self.imgtrans_proj is not None:
                self.imgtrans_proj.clean_mask_and_inpainted_cache()
                msg += '\n' + self.tr('Cache cleaned.')
            if show_message:
                QMessageBox.information(self, self.tr('Export'), msg)
        except Exception as e:
            LOGGER.exception(e)
            create_error_dialog(e, self.tr('Batch export failed'), 'BatchExport')

    def on_validate_project(self):
        """Check project: missing images, invalid JSON, duplicate/overlapping blocks."""
        if self.imgtrans_proj.is_empty:
            QMessageBox.information(self, self.tr('Check project'), self.tr('Open a project first.'))
            return
        report = []
        proj_dir = self.imgtrans_proj.directory
        json_path = self.imgtrans_proj.proj_path
        if not osp.exists(json_path):
            report.append(self.tr('Project JSON not found: {}').format(json_path))
        else:
            try:
                import json
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                pages = data.get('pages') or data.get('image_info') or {}
                if not pages:
                    report.append(self.tr('No pages in project JSON.'))
            except Exception as e:
                report.append(self.tr('Invalid project JSON: {}').format(str(e)))
        for pagename in list(self.imgtrans_proj.pages.keys()):
            img_path = osp.join(proj_dir, pagename)
            if not osp.exists(img_path):
                report.append(self.tr('Missing image: {}').format(pagename))

        # Duplicate/overlapping block check
        from utils.imgproc_utils import union_area
        overlap_report = []
        for pagename in list(self.imgtrans_proj.pages.keys()):
            blks = self.imgtrans_proj.pages.get(pagename) or []
            n = len(blks)
            pairs = []
            for i in range(n):
                for j in range(i + 1, n):
                    blk_i = blks[i]
                    blk_j = blks[j]
                    xyxy_i = getattr(blk_i, 'xyxy', None) if not isinstance(blk_i, dict) else blk_i.get('xyxy')
                    xyxy_j = getattr(blk_j, 'xyxy', None) if not isinstance(blk_j, dict) else blk_j.get('xyxy')
                    if xyxy_i is None and isinstance(blk_i, (list, tuple)) and len(blk_i) >= 4:
                        xyxy_i = list(blk_i)[:4]
                    if xyxy_j is None and isinstance(blk_j, (list, tuple)) and len(blk_j) >= 4:
                        xyxy_j = list(blk_j)[:4]
                    if xyxy_i and xyxy_j and len(xyxy_i) >= 4 and len(xyxy_j) >= 4:
                        inter = union_area(list(xyxy_i)[:4], list(xyxy_j)[:4])
                        if inter > 0:
                            pairs.append((i, j))
            if pairs:
                overlap_report.append(self.tr('Page "{}": {} overlapping block pair(s): {}').format(
                    pagename, len(pairs), ', '.join(f'({a},{b})' for a, b in pairs[:10]) + (' ...' if len(pairs) > 10 else '')
                ))
        if overlap_report:
            report.append('')
            report.append(self.tr('Duplicate/overlapping blocks:'))
            report.extend(overlap_report)

        if not report:
            report.append(self.tr('No issues found.'))
        msg = '\n'.join(report)
        dlg = QMessageBox(self)
        dlg.setWindowTitle(self.tr('Check project'))
        dlg.setText(msg)
        dlg.setStandardButtons(QMessageBox.StandardButton.Ok)
        dlg.exec()


    def on_req_update_pagetext(self):
        if self.canvas.text_change_unsaved():
            self.st_manager.updateTextBlkList()

    def on_req_move_page(self, page_name: str, force_save=False):
        ori_save = self.save_on_page_changed
        self.save_on_page_changed = False
        current_img = self.imgtrans_proj.current_img
        if current_img == page_name and not force_save:
            return
        if current_img not in self.global_search_widget.page_set:
            if self.canvas.projstate_unsaved: 
                self.saveCurrentPage()
        else:
            self.saveCurrentPage(save_rst_only=True)
        self.pageList.setCurrentRow(self.imgtrans_proj.pagename2idx(page_name))
        self.save_on_page_changed = ori_save

    def on_search_result_item_clicked(self, pagename: str, blk_idx: int, is_src: bool, start: int, end: int):
        idx = self.imgtrans_proj.pagename2idx(pagename)
        self.pageList.setCurrentRow(idx)
        pw = self.st_manager.pairwidget_list[blk_idx]
        edit = pw.e_source if is_src else pw.e_trans
        edit.setFocus()
        edit.ensure_scene_visible.emit()
        cursor = QTextCursor(edit.document())
        cursor.setPosition(start)
        cursor.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
        edit.setTextCursor(cursor)

    def shortcutEscape(self):
        if self.canvas.search_widget.isVisible():
            self.canvas.search_widget.hide()
        elif self.canvas.editing_textblkitem is not None and self.canvas.editing_textblkitem.isEditing():
            self.canvas.editing_textblkitem.endEdit()
        elif self.canvas.image_edit_mode == ImageEditMode.TextEraserTool:
            self.drawingPanel.setCurrentToolByName('hand')
            if hasattr(self.canvas, '_text_eraser_selected_blocks'):
                self.canvas._text_eraser_selected_blocks = None
        elif self.canvas.drawMode() and self.canvas.cancel_rect_selection():
            pass  # rect selection cancelled (#126)

    def setPaintMode(self):
        if self.bottomBar.paintChecker.isChecked():
            if self.rightComicTransStackPanel.isHidden():
                self.rightComicTransStackPanel.show()
            self.rightComicTransStackPanel.setCurrentIndex(0)
            self.canvas.setPaintMode(True)
            self.bottomBar.originalSlider.show()
            self.bottomBar.textlayerSlider.show()
            self.bottomBar.textblockChecker.hide()
        else:
            self.canvas.setPaintMode(False)
            self.rightComicTransStackPanel.setHidden(True)
        self.st_manager.setTextEditMode(False)

    def setTextEditMode(self):
        if self.bottomBar.texteditChecker.isChecked():
            if self.rightComicTransStackPanel.isHidden():
                self.rightComicTransStackPanel.show()
            self.bottomBar.textblockChecker.show()
            self.rightComicTransStackPanel.setCurrentIndex(1)
            self.st_manager.setTextEditMode(True)
            self.setTextBlockMode()
        else:
            self.bottomBar.textblockChecker.hide()
            self.rightComicTransStackPanel.setHidden(True)
            self.st_manager.setTextEditMode(False)
        self.canvas.setPaintMode(False)

    def setTextBlockMode(self):
        mode = self.bottomBar.textblockChecker.isChecked()
        self.canvas.setTextBlockMode(mode)
        pcfg.imgtrans_textblock = mode
        self.st_manager.showTextblkItemRect(mode)

    def manual_save(self):
        if self.leftBar.imgTransChecker.isChecked()\
            and self.imgtrans_proj.directory is not None:
            LOGGER.debug('Manually saving...')
            self.saveCurrentPage(update_scene_text=True, save_proj=True, restore_interface=True, save_rst_only=False)

    def close_project_and_show_welcome(self):
        """Close current project and switch to welcome screen. Prompts to save if unsaved."""
        if self.imgtrans_proj.is_empty:
            self._show_welcome_screen()
            return
        if self.canvas.projstate_unsaved:
            ret = QMessageBox.question(
                self,
                self.tr("Unsaved changes"),
                self.tr("Save project before closing?"),
                QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Save,
            )
            if ret == QMessageBox.StandardButton.Cancel:
                return
            if ret == QMessageBox.StandardButton.Save:
                self.saveCurrentPage(update_scene_text=True, save_proj=True, restore_interface=True, save_rst_only=False)
        self.imgtrans_proj.pages = {}
        self.imgtrans_proj._pagename2idx = {}
        self.imgtrans_proj._idx2pagename = {}
        self.imgtrans_proj.current_img = None
        self.imgtrans_proj.directory = None
        self.imgtrans_proj.proj_path = None
        self.imgtrans_proj.img_array = None
        self.imgtrans_proj.mask_array = None
        self.imgtrans_proj.inpainted_array = None
        self.st_manager.clearSceneTextitems()
        self.pageList.clear()
        self.titleBar.setTitleContent(proj_name="", page_name="")
        self.canvas.clear_undostack(update_saved_step=True)
        if hasattr(self.welcomeWidget, "set_recent_projects"):
            self.welcomeWidget.set_recent_projects(getattr(self.leftBar, "recent_proj_list", []) or [])
        self.centralStackWidget.setCurrentIndex(0)

    def saveCurrentPage(self, update_scene_text=True, save_proj=True, restore_interface=False, save_rst_only=False, keep_exist_as_backup=False):
        
        if not self.imgtrans_proj.img_valid:
            return
        
        if restore_interface:
            set_canvas_focus = self.canvas.hasFocus()
            sel_textitem = self.canvas.selected_text_items()
            n_sel_textitems = len(sel_textitem)
            editing_textitem = None
            if n_sel_textitems == 1 and sel_textitem[0].isEditing():
                editing_textitem = sel_textitem[0]
        
        if update_scene_text:
            self.st_manager.updateTextBlkList()
        
        if self.rightComicTransStackPanel.isHidden():
            self.bottomBar.texteditChecker.click()

        restore_textblock_mode = False
        if pcfg.imgtrans_textblock:
            restore_textblock_mode = True
            self.bottomBar.textblockChecker.click()

        hide_tsc = False
        if self.st_manager.txtblkShapeControl.isVisible():
            hide_tsc = True
            self.st_manager.txtblkShapeControl.hide()

        if not osp.exists(self.imgtrans_proj.result_dir()):
            os.makedirs(self.imgtrans_proj.result_dir())

        if save_proj:
            try:
                self.imgtrans_proj.save(keep_exist_as_backup=keep_exist_as_backup)
                if not save_rst_only:
                    mask_path = self.imgtrans_proj.get_mask_path()
                    mask_array = self.imgtrans_proj.mask_array
                    if mask_array is not None:
                        self.imsave_thread.saveImg(mask_path, mask_array, save_params={'ext': pcfg.intermediate_imgsave_ext})
                    inpainted_path = self.imgtrans_proj.get_inpainted_path()
                    if self.canvas.drawingLayer.drawed():
                        inpainted = self.canvas.base_pixmap.copy()
                        painter = QPainter(inpainted)
                        painter.drawPixmap(0, 0, self.canvas.drawingLayer.get_drawed_pixmap())
                        painter.end()
                    else:
                        inpainted = self.imgtrans_proj.inpainted_array
                    if inpainted is not None:
                        self.imsave_thread.saveImg(inpainted_path, inpainted, save_params={'ext': pcfg.intermediate_imgsave_ext}, keep_alpha=self.imgtrans_proj.current_has_alpha())
            except Exception as e:
                LOGGER.error(f"Failed to save project files: {e}")

        # Render the final result image properly
        try:
            img = self.canvas.render_result_img()
            # Optional final upscale (Section 6: 2x output for nicer export)
            if getattr(pcfg.module, 'image_upscale_final', False):
                try:
                    factor = float(getattr(pcfg.module, 'image_upscale_final_factor', 2.0) or 2.0)
                    policy = (getattr(pcfg.module, 'upscale_policy_final', None) or 'lanczos').strip().lower()
                    if factor > 1.0 and policy != 'none':
                        from utils.image_upscale import apply_upscale_final
                        if hasattr(img, 'bits'):
                            img = pixmap2ndarray(img, keep_alpha=self.imgtrans_proj.current_has_alpha())
                        img = apply_upscale_final(img, factor=factor, policy=policy)
                except Exception as e:
                    LOGGER.warning("Final upscale failed: %s", e)
            # Optional: lightweight colorization for grayscale pages (Section 6+Colorization).
            if getattr(pcfg.module, 'enable_colorization', False):
                try:
                    if hasattr(img, 'bits'):
                        img = pixmap2ndarray(img, keep_alpha=self.imgtrans_proj.current_has_alpha())
                    strength = float(getattr(pcfg.module, 'colorization_strength', 0.6) or 0.6)
                    backend = getattr(pcfg.module, 'colorization_backend', 'simple') or 'simple'
                    img = apply_colorization(img, strength=strength, backend=backend)
                except Exception as e:
                    LOGGER.warning("Colorization failed: %s", e)
            imsave_path = self.imgtrans_proj.get_result_path(self.imgtrans_proj.current_img)
            save_params = {'ext': pcfg.imgsave_ext, 'quality': pcfg.imgsave_quality}
            if pcfg.imgsave_ext == '.webp' and getattr(pcfg, 'imgsave_webp_lossless', False):
                save_params['webp_lossless'] = True
            self.imsave_thread.saveImg(imsave_path, img, self.imgtrans_proj.current_img, save_params=save_params, keep_alpha=self.imgtrans_proj.current_has_alpha())
        except Exception as e:
            LOGGER.error(f"Failed to render and save result image: {e}")
            
        self.canvas.setProjSaveState(False)
        self.canvas.update_saved_undostep()

        if restore_interface:
            if restore_textblock_mode:
                self.bottomBar.textblockChecker.click()
            if hide_tsc:
                self.st_manager.txtblkShapeControl.show()
            if set_canvas_focus:
                self.canvas.setFocus()
            if n_sel_textitems > 0:
                self.canvas.block_selection_signal = True
                for blk in sel_textitem:
                    blk.setSelected(True)
                self.st_manager.on_incanvas_selection_changed()
                self.canvas.block_selection_signal = False
            if editing_textitem is not None:
                editing_textitem.startEdit()
        
    def to_trans_config(self):
        self.leftBar.configChecker.setChecked(True)
        self.configPanel.focusOnTranslator()

    def to_inpaint_config(self):
        self.leftBar.configChecker.setChecked(True)
        self.configPanel.focusOnInpaint()

    def to_ocr_config(self):
        self.leftBar.configChecker.setChecked(True)
        self.configPanel.focusOnOCR()

    def to_detect_config(self):
        self.leftBar.configChecker.setChecked(True)
        self.configPanel.focusOnDetect()

    def on_textdet_changed(self):
        module = self.bottomBar.textdet_selector.selector.currentText()
        tgt_selector = self.configPanel.detect_config_panel.module_combobox
        if tgt_selector.currentText() != module and module in GET_VALID_TEXTDETECTORS():
            tgt_selector.setCurrentText(module)

    def on_ocr_changed(self):
        module = self.bottomBar.ocr_selector.selector.currentText()
        tgt_selector = self.configPanel.ocr_config_panel.module_combobox
        if tgt_selector.currentText() != module and module in GET_VALID_OCR():
            tgt_selector.setCurrentText(module)

    def on_trans_changed(self):
        module = self.bottomBar.trans_selector.selector.currentText()
        tgt_selector = self.configPanel.trans_config_panel.module_combobox
        if tgt_selector.currentText() != module and module in GET_VALID_TRANSLATORS():
            tgt_selector.setCurrentText(module)

    def on_trans_src_changed(self):
        sender = self.sender()
        text = sender.currentText()
        key = lang_display_to_key(text)
        translator = self.module_manager.translator
        if translator is not None:
            translator.set_source(key)
        pcfg.module.translate_source = key
        try:
            idx = translator.supported_src_list.index(key) if translator else 0
        except (ValueError, AttributeError):
            idx = 0
        combobox = self.configPanel.trans_config_panel.source_combobox
        if sender != combobox:
            combobox.blockSignals(True)
            combobox.setCurrentIndex(idx)
            combobox.blockSignals(False)
        combobox = self.bottomBar.trans_selector.src_selector
        if sender != combobox:
            combobox.blockSignals(True)
            combobox.setCurrentIndex(idx)
            combobox.blockSignals(False)

    def on_trans_tgt_changed(self):
        sender = self.sender()
        text = sender.currentText()
        key = lang_display_to_key(text)
        translator = self.module_manager.translator
        if translator is not None:
            translator.set_target(key)
        pcfg.module.translate_target = key
        try:
            idx = translator.supported_tgt_list.index(key) if translator else 0
        except (ValueError, AttributeError):
            idx = 0
        combobox = self.configPanel.trans_config_panel.target_combobox
        if sender != combobox:
            combobox.blockSignals(True)
            combobox.setCurrentIndex(idx)
            combobox.blockSignals(False)
        combobox = self.bottomBar.trans_selector.tgt_selector
        if sender != combobox:
            combobox.blockSignals(True)
            combobox.setCurrentIndex(idx)
            combobox.blockSignals(False)

    def on_inpaint_changed(self):
        module = self.bottomBar.inpaint_selector.selector.currentText()
        tgt_selector = self.configPanel.inpaint_config_panel.module_combobox
        if tgt_selector.currentText() != module and module in GET_VALID_INPAINTERS():
            tgt_selector.setCurrentText(module)

    def on_transpagebtn_pressed(self, run_target: bool):
        page_key = self.imgtrans_proj.current_img
        if page_key is None:
            return

        blkitem_list = self.st_manager.textblk_item_list

        if len(blkitem_list) < 1:
            return
        
        self.translateBlkitemList(blkitem_list, -1)


    def translateBlkitemList(self, blkitem_list: List, mode: int) -> bool:

        tgt_img = self.imgtrans_proj.img_array
        if tgt_img is None:
            return False
        tgt_mask = self.imgtrans_proj.mask_array
        
        if len(blkitem_list) < 1:
            return False
        
        self.global_search_widget.set_document_edited()
        
        im_h, im_w = tgt_img.shape[:2]

        blk_list, blk_ids = [], []
        for blkitem in blkitem_list:
            blk: TextBlock = blkitem.blk
            blk._bounding_rect = blkitem.absBoundingRect()
            blk.text = self.st_manager.pairwidget_list[blkitem.idx].e_source.toPlainText()
            blk_ids.append(blkitem.idx)
            blk.set_lines_by_xywh(blk._bounding_rect, angle=-blk.angle, x_range=[0, im_w-1], y_range=[0, im_h-1], adjust_bbox=True)
            blk_list.append(blk)

        self.module_manager.runBlktransPipeline(blk_list, tgt_img, mode, blk_ids, tgt_mask = tgt_mask)
        return True


    def finishTranslatePage(self, page_key):
        if page_key == self.imgtrans_proj.current_img:
            self.st_manager.updateTranslation()

    def on_imgtrans_pipeline_finished(self):
        if getattr(self, '_detection_only_restore', None) is not None:
            det, ocr, trans, inp = self._detection_only_restore
            pcfg.module.enable_detect = det
            pcfg.module.enable_ocr = ocr
            pcfg.module.enable_translate = trans
            pcfg.module.enable_inpaint = inp
            for idx, sa in enumerate(self.titleBar.stageActions):
                sa.setChecked(pcfg.module.stage_enabled(idx))
            self._detection_only_restore = None
        if getattr(self, '_run_stages_restore', None) is not None:
            det, ocr, trans, inp = self._run_stages_restore
            pcfg.module.enable_detect = det
            pcfg.module.enable_ocr = ocr
            pcfg.module.enable_translate = trans
            pcfg.module.enable_inpaint = inp
            for idx, sa in enumerate(self.titleBar.stageActions):
                sa.setChecked(pcfg.module.stage_enabled(idx))
            self._run_stages_restore = None
        self.backup_blkstyles.clear()
        self._run_imgtrans_wo_textstyle_update = False
        self.postprocess_mt_toggle = True
        if pcfg.module.empty_runcache and not (shared.HEADLESS or shared.HEADLESS_CONTINUOUS):
            self.module_manager.unload_all_models()
            try:
                from utils.pipeline_cache import get_pipeline_cache
                cache = get_pipeline_cache(True)
                if cache is not None:
                    cache.clear()
            except Exception:
                pass
        elif getattr(pcfg, 'release_caches_after_batch', False) and not (shared.HEADLESS or shared.HEADLESS_CONTINUOUS):
            import gc
            self.module_manager.unload_all_models()
            try:
                from utils.pipeline_cache import get_pipeline_cache
                cache = get_pipeline_cache(True)
                if cache is not None:
                    cache.clear()
            except Exception:
                pass
            gc.collect()
        if not (pcfg.module.empty_runcache or getattr(pcfg, 'release_caches_after_batch', False)):
            self._reset_idle_unload_timer()
        if shared.args.export_translation_txt:
            self.on_export_txt('translation')
        if shared.args.export_source_txt:
            self.on_export_txt('source')
        if shared.HEADLESS or shared.HEADLESS_CONTINUOUS or getattr(self, 'exec_dirs', None) is not None:
            # If this project dir was extracted from a ZIP, copy results out before moving on.
            try:
                if self._current_batch_dir:
                    handled, job, copied = self._zip_batch.on_project_finished(self._current_batch_dir)
                    if handled and job is not None:
                        LOGGER.info(
                            f'ZIP batch: copied {copied} file(s) for {self._current_batch_dir} -> {job.output_root}'
                        )
                        finished_jobs = self._zip_batch.finalize_finished_jobs()
                        for fj in finished_jobs:
                            LOGGER.info(
                                f'ZIP batch finished: {fj.zip_path} -> {fj.output_root}'
                            )
            except Exception as e:
                LOGGER.warning(f'ZIP batch output copy failed: {e}')
            self.run_next_dir()
            return
        self.maybe_auto_run_region_merge()
        if not (shared.HEADLESS or shared.HEADLESS_CONTINUOUS):
            report = self.module_manager.get_last_batch_report()
            self._last_batch_report = report
            if report and report.get("skipped") and len(report["skipped"]) > 0:
                self._show_batch_report_dialog()
            self.titleBar.show_batch_report_action.setEnabled(
                bool(self._last_batch_report and len(self._last_batch_report.get("skipped", {})) > 0)
            )

    def _show_batch_report_dialog(self):
        """Show the batch report dialog with last report (if any)."""
        if not hasattr(self, '_batch_report_dialog') or self._batch_report_dialog is None:
            from .batch_report_dialog import BatchReportDialog
            self._batch_report_dialog = BatchReportDialog(self)
            self._batch_report_dialog.page_activated.connect(self._go_to_page_from_report)
        self._batch_report_dialog.set_report(self._last_batch_report)
        self._batch_report_dialog.show()
        self._batch_report_dialog.raise_()
        self._batch_report_dialog.activateWindow()

    def _go_to_page_from_report(self, page_key: str):
        """Switch to the given page (from batch report double-click)."""
        if self.imgtrans_proj.is_empty or page_key not in self.imgtrans_proj.pages:
            return
        self.imgtrans_proj.set_current_img(page_key)
        idx = self.imgtrans_proj.pagename2idx(page_key)
        if idx >= 0 and self.pageList.count() > idx:
            self.page_changing = True
            self.pageList.setCurrentRow(idx)
            self.page_changing = False
        self.canvas.clear_undostack(update_saved_step=True)
        self.canvas.updateCanvas()
        self.st_manager.updateSceneTextitems()
        self.titleBar.setTitleContent(page_name=self.imgtrans_proj.current_img)
        self.module_manager.handle_page_changed()
        self.drawingPanel.handle_page_changed()

    def maybe_auto_run_region_merge(self):
        """If configured, run Region merge after pipeline (all pages or current page)."""
        mode = getattr(pcfg, 'auto_region_merge_after_run', 'never')
        if mode == 'never' or self.imgtrans_proj.is_empty:
            return
        if not hasattr(self, 'merge_dialog') or self.merge_dialog is None:
            from .merge_dialog import MergeDialog
            self.merge_dialog = MergeDialog(self)
            self.merge_dialog.run_current_clicked.connect(lambda: self.run_merge_task(on_current=True))
            self.merge_dialog.run_all_clicked.connect(lambda: self.run_merge_task(on_current=False))
        config = self.merge_dialog.get_config()
        if mode == 'all_pages':
            img_list = list(self.imgtrans_proj.pages.keys())
            if not img_list:
                return
            json_path = self.imgtrans_proj.proj_path
            if not json_path or not osp.exists(json_path):
                return
            self.run_merge_all_async(json_path, img_list, config)
        elif mode == 'current_page':
            self.run_merge_task(on_current=True)

    def postprocess_translations(self, blk_list: List[TextBlock]) -> None:
        src_is_cjk = is_cjk(pcfg.module.translate_source)
        tgt_is_cjk = is_cjk(pcfg.module.translate_target)
        if tgt_is_cjk:
            for blk in blk_list:
                if src_is_cjk:
                    blk.translation = full_len(blk.translation)
                else:
                    blk.translation = half_len(blk.translation)
                    blk.translation = re.sub(r'([?.!"])\s+', r'\1', blk.translation)    # remove spaces following punctuations
        else:
            for blk in blk_list:
                if blk.vertical:
                    blk.alignment = TextAlignment.Center
                blk.translation = half_len(blk.translation)
                blk.vertical = False

        for blk in blk_list:
            blk.translation = self.mtSubWidget.sub_text(blk.translation)
            if pcfg.let_uppercase_flag:
                blk.translation = blk.translation.upper()

    def on_pagtrans_finished(self, page_index: int):
        blk_list = self.imgtrans_proj.get_blklist_byidx(page_index)
        ffmt_list = None
        if len(self.backup_blkstyles) == self.imgtrans_proj.num_pages and len(self.backup_blkstyles[page_index]) == len(blk_list):
            ffmt_list: List[FontFormat] = self.backup_blkstyles[page_index]

        self.postprocess_translations(blk_list)
                
        # override font format if necessary
        override_fnt_size = pcfg.let_fntsize_flag == 1
        override_fnt_stroke = pcfg.let_fntstroke_flag == 1
        override_fnt_color = pcfg.let_fntcolor_flag == 1
        override_fnt_scolor = pcfg.let_fnt_scolor_flag == 1
        override_alignment = pcfg.let_alignment_flag == 1
        override_effect = pcfg.let_fnteffect_flag == 1
        override_writing_mode = pcfg.let_writing_mode_flag == 1
        override_font_family = pcfg.let_family_flag == 1
        gf = self.textPanel.formatpanel.global_format

        inpaint_only = pcfg.module.enable_inpaint
        inpaint_only = inpaint_only and not (pcfg.module.enable_detect or pcfg.module.enable_ocr or pcfg.module.enable_translate)
        
        if not inpaint_only:
            for ii, blk in enumerate(blk_list):
                if self._run_imgtrans_wo_textstyle_update and ffmt_list is not None:
                    blk.fontformat.merge(ffmt_list[ii])
                else:
                    if override_fnt_size or \
                        blk.font_size < 0:  # fall back to global font size if font size is not valid, it will be set to -1 for detected blocks
                        blk.font_size = gf.font_size
                    elif blk._detected_font_size > 0 and not pcfg.module.enable_detect:
                        blk.font_size = blk._detected_font_size
                    if override_fnt_stroke:
                        blk.stroke_width = gf.stroke_width
                    elif pcfg.module.enable_ocr:
                        blk.recalulate_stroke_width()
                    if override_fnt_color:
                        blk.set_font_colors(fg_colors=gf.frgb)
                    if override_fnt_scolor:
                        blk.set_font_colors(bg_colors=gf.srgb)
                    if override_alignment:
                        blk.alignment = gf.alignment
                    elif pcfg.module.enable_detect and not blk.src_is_vertical:
                        blk.recalulate_alignment()
                    if override_effect:
                        blk.opacity = gf.opacity
                        blk.shadow_color = gf.shadow_color
                        blk.shadow_radius = gf.shadow_radius
                        blk.shadow_strength = gf.shadow_strength
                        blk.shadow_offset = gf.shadow_offset
                    if override_writing_mode:
                        blk.vertical = gf.vertical
                    if override_font_family or blk.font_family is None:
                        blk.font_family = gf.font_family
                        if blk.rich_text:
                            blk.rich_text = set_html_family(blk.rich_text, gf.font_family)
                    
                    blk.line_spacing = gf.line_spacing
                    blk.letter_spacing = gf.letter_spacing
                    blk.italic = gf.italic
                    blk.bold = gf.bold
                    blk.underline = gf.underline
                    blk.strikethrough = getattr(gf, 'strikethrough', False)
                    sw = blk.stroke_width
                    if sw > 0 and pcfg.module.enable_ocr and pcfg.module.enable_detect and not override_fnt_size:
                        blk.font_size = blk.font_size / (1 + sw)

            self.st_manager.auto_textlayout_flag = (
                pcfg.let_autolayout_flag
                and (pcfg.let_fntsize_flag == 0)
                and (pcfg.module.enable_detect or pcfg.module.enable_ocr or pcfg.module.enable_translate)
            )
        else:
            self.st_manager.auto_textlayout_flag = False

        if page_index != self.pageList.currentIndex().row():
            self.pageList.setCurrentRow(page_index)
        else:
            self.imgtrans_proj.set_current_img_byidx(page_index)
            self.canvas.updateCanvas()
            self.st_manager.updateSceneTextitems()

        # Run auto layout (and balloon shape Auto) on all blocks so text boxes match bubble shape after Run.
        if self.st_manager.auto_textlayout_flag:
            self.st_manager.run_auto_layout_on_current_page_once()

        if not pcfg.module.enable_detect and pcfg.module.enable_translate:
            # Skip squeeze when auto layout ran; layout_textblk already set box size and squeeze would narrow/stretch boxes
            if not self.st_manager.auto_textlayout_flag:
                for blkitem in self.st_manager.textblk_item_list:
                    blkitem.squeezeBoundingRect()

        if page_index + 1 == self.imgtrans_proj.num_pages:
            self.st_manager.auto_textlayout_flag = False

        # save proj file on page trans finished
        self.imgtrans_proj.save()

        self.saveCurrentPage(False, False)

    def on_savestate_changed(self, unsaved: bool):
        save_state = self.tr('unsaved') if unsaved else self.tr('saved')
        self.titleBar.setTitleContent(save_state=save_state)

    def on_textstack_changed(self):
        if not self.page_changing:
            self.global_search_widget.set_document_edited()

    def on_run_blktrans(self, mode: int):
        blkitem_list = self.canvas.selected_text_items()
        self.translateBlkitemList(blkitem_list, mode)

    def on_blktrans_finished(self, mode: int, blk_ids: List[int]):

        if len(blk_ids) < 1:
            return
        
        blkitem_list = [self.st_manager.textblk_item_list[idx] for idx in blk_ids]

        pairw_list = []
        for blk in blkitem_list:
            pairw_list.append(self.st_manager.pairwidget_list[blk.idx])
        self.canvas.push_undo_command(RunBlkTransCommand(self.canvas, blkitem_list, pairw_list, mode))

    def on_imgtrans_progressbox_showed(self):
        msg_size = self.module_manager.progress_msgbox.size()
        size = self.size()
        p = self.mapToGlobal(QPoint(size.width() - msg_size.width(),
                                    size.height() - msg_size.height()))
        self.module_manager.progress_msgbox.move(p)

    def on_closebtn_clicked(self):
        if self.imsave_thread.isRunning():
            self.imsave_thread.finished.connect(self.close)
            mb = FrameLessMessageBox()
            mb.setText(self.tr('Saving image...'))
            self.imsave_thread.finished.connect(mb.close)
            mb.exec()
            return
        self.close()

    def on_display_lang_changed(self, lang: str):
        if lang != pcfg.display_lang:
            pcfg.display_lang = lang
            self.set_display_lang(lang)

    def on_dev_mode_changed(self):
        """Refresh detector/OCR/translator dropdowns after dev_mode toggle."""
        valid_det = GET_VALID_TEXTDETECTORS()
        valid_ocr = GET_VALID_OCR()
        valid_trans = GET_VALID_TRANSLATORS()
        self.bottomBar.textdet_selector.selector.clear()
        self.bottomBar.textdet_selector.selector.addItems(valid_det)
        self.bottomBar.textdet_selector.selector.setCurrentText(
            pcfg.module.textdetector if pcfg.module.textdetector in valid_det else (valid_det[0] if valid_det else '')
        )
        self.bottomBar.ocr_selector.selector.clear()
        self.bottomBar.ocr_selector.selector.addItems(valid_ocr)
        self.bottomBar.ocr_selector.selector.setCurrentText(
            pcfg.module.ocr if pcfg.module.ocr in valid_ocr else (valid_ocr[0] if valid_ocr else '')
        )
        self.bottomBar.trans_selector.selector.clear()
        self.bottomBar.trans_selector.selector.addItems(valid_trans)
        self.bottomBar.trans_selector.selector.setCurrentText(
            pcfg.module.translator if pcfg.module.translator in valid_trans else (valid_trans[0] if valid_trans else '')
        )
        if hasattr(self, 'module_manager') and self.module_manager is not None:
            self.module_manager.refresh_module_dropdowns()
    
    def run_imgtrans(self):
        if not self.imgtrans_proj.is_all_pages_no_text and not pcfg.module.keep_exist_textlines:
            if getattr(pcfg, 'confirm_before_run', True):
                msgBox = QMessageBox(self)
                msgBox.setIcon(QMessageBox.Question)
                msgBox.setWindowTitle(self.tr('Confirmation'))
                msgBox.setText(self.tr('"Run" will clear previous results, "Continue" will try to run from previous progress'))

                restart_btn = msgBox.addButton(self.tr('Run'), QMessageBox.YesRole)
                continue_btn = msgBox.addButton(self.tr('Continue'), QMessageBox.AcceptRole)
                cancel_btn = msgBox.addButton(self.tr('Cancel'), QMessageBox.RejectRole)

                msgBox.setDefaultButton(continue_btn)
                msgBox.exec_()

                clicked_button = msgBox.clickedButton()
                if clicked_button == cancel_btn:
                    return
                elif clicked_button == continue_btn:
                    self.on_run_imgtrans(continue_mode=True)
                    return
        self.on_run_imgtrans()

    def run_imgtrans_wo_textstyle_update(self):
        self._run_imgtrans_wo_textstyle_update = True
        self.run_imgtrans()

    def on_run_imgtrans(self, continue_mode=False, pages_to_process=None, skip_ignored_in_batch=None):
        self._idle_unload_timer.stop()
        self.backup_blkstyles.clear()

        if self.bottomBar.textblockChecker.isChecked():
            self.bottomBar.textblockChecker.click()
        self.postprocess_mt_toggle = False

        all_disabled = pcfg.module.all_stages_disabled()
        skip_ignored = skip_ignored_in_batch if skip_ignored_in_batch is not None else getattr(pcfg, 'skip_ignored_in_run', True)

        # 继续模式：先检查哪些页面需要处理
        if continue_mode:
            pages_to_process = []
            for page_name in self.imgtrans_proj.pages:
                if not self.imgtrans_proj.get_page_progress(page_name):
                    pages_to_process.append(page_name)
            if skip_ignored:
                pages_to_process = [p for p in pages_to_process if not self.imgtrans_proj.is_page_ignored(p)]
            if len(pages_to_process) == 0:
                return
        elif pages_to_process is None:
            if getattr(pcfg, 'manual_mode', False) and self.imgtrans_proj.current_img:
                pages_to_process = [self.imgtrans_proj.current_img]
                self.imgtrans_proj.set_page_progress(pages_to_process[0], 0)
            else:
                pages_to_process = list(self.imgtrans_proj.pages.keys())
                if skip_ignored:
                    pages_to_process = [p for p in pages_to_process if not self.imgtrans_proj.is_page_ignored(p)]
                if not pages_to_process:
                    return
                for page_name in pages_to_process:
                    self.imgtrans_proj.set_page_progress(page_name, 0)
        else:
            for page_name in pages_to_process:
                if page_name in self.imgtrans_proj.pages:
                    self.imgtrans_proj.set_page_progress(page_name, 0)
        
        if pcfg.module.enable_detect:
            for page in self.imgtrans_proj.pages:
                if not pcfg.module.keep_exist_textlines:
                    if not pages_to_process:
                        # 没有指定pages_to_process，清空所有页面
                        self.imgtrans_proj.pages[page].clear()
        else:
            self.st_manager.updateTextBlkList()
            textblk: TextBlock = None
            for page_name, blklist in self.imgtrans_proj.pages.items():
                # 如果指定了pages_to_process，跳过不需要处理的页面
                if pages_to_process and page_name not in pages_to_process:
                    continue
                    
                ffmt_list = []
                self.backup_blkstyles.append(ffmt_list)
                for textblk in blklist:
                    if not pcfg.module.enable_detect:
                        ffmt_list.append(textblk.fontformat.deepcopy())
                    # 继续模式且没有指定pages_to_process时：跳过已有文本的文本块
                    if continue_mode and not pages_to_process and textblk.text and len(textblk.text) > 0:
                        continue
                    if pcfg.module.enable_ocr:
                        textblk.text = []
                        textblk.set_font_colors((0, 0, 0), (0, 0, 0))
                    if pcfg.module.enable_translate or (all_disabled and not self._run_imgtrans_wo_textstyle_update) or pcfg.module.enable_ocr:
                        textblk.rich_text = ''
                    textblk.vertical = textblk.src_is_vertical
        
        # 如果有指定pages_to_process或者是continue_mode，则传递页面列表
        self.module_manager.runImgtransPipeline(pages_to_process if (pages_to_process or continue_mode) else None)

    def on_transpanel_changed(self):
        self.canvas.editor_index = self.rightComicTransStackPanel.currentIndex()
        if self.rightComicTransStackPanel.currentIndex() != 0:
            self.canvas.restoreDefaultCursor()
            self.canvas.clearDrawingStroke()
        if not self.canvas.textEditMode() and self.canvas.search_widget.isVisible():
            self.canvas.search_widget.hide()
        self.canvas.updateLayers()

    def import_tstyles(self):
        ddir = osp.dirname(pcfg.text_styles_path)
        p = QFileDialog.getOpenFileName(self, self.tr("Import Text Styles"), ddir, None, "(.json)")
        if not isinstance(p, str):
            p = p[0]
        if p == '':
            return
        try:
            load_textstyle_from(p, raise_exception=True)
            save_config()
            self.textPanel.formatpanel.textstyle_panel.setStyles(text_styles)
            self.textPanel.formatpanel.textstyle_panel.set_active_style_by_name(
                getattr(pcfg, 'default_text_style_name', '') or ''
            )
        except Exception as e:
            create_error_dialog(e, self.tr(f'Failed to load from {p}'))

    def export_tstyles(self):
        ddir = osp.dirname(pcfg.text_styles_path)
        savep = QFileDialog.getSaveFileName(self, self.tr("Save Text Styles"), ddir, None, "(.json)")
        if not isinstance(savep, str):
            savep = savep[0]
        if savep == '':
            return
        suffix = Path(savep).suffix
        if suffix != '.json':
            if suffix == '':
                savep = savep + '.json'
            else:
                savep = savep.replace(suffix, '.json')
        oldp = pcfg.text_styles_path
        try:
            pcfg.text_styles_path = savep
            save_text_styles(raise_exception=True)
            save_config()
        except Exception as e:
            create_error_dialog(e, self.tr(f'Failed save to {savep}'))
            pcfg.text_styles_path = oldp

    def fold_textarea(self, fold: bool):
        pcfg.fold_textarea = fold
        self.textPanel.textEditList.setFoldTextarea(fold)

    def show_source_text(self, show: bool):
        pcfg.show_source_text = show
        self.textPanel.textEditList.setSourceVisible(show)

    def show_trans_text(self, show: bool):
        pcfg.show_trans_text = show
        self.textPanel.textEditList.setTransVisible(show)

    def on_export_doc(self):
        if self.canvas.text_change_unsaved():
            self.st_manager.updateTextBlkList()
        self.export_doc_thread.exportAsDoc(self.imgtrans_proj)

    def on_export_current_page_as(self):
        """Export current page as image: result if available, else inpainted, else original (#1134)."""
        self._export_or_download_page(mode=None)

    def on_download_page_requested(self, mode: str):
        """Handle canvas right-click Download image: result | inpainted | original."""
        self._export_or_download_page(mode=mode)

    def _export_or_download_page(self, mode: str = None):
        """Export or download current page. mode: None = result→inpainted→original; 'result'|'inpainted'|'original' = canvas Download menu."""
        if self.imgtrans_proj.is_empty or not self.imgtrans_proj.current_img:
            QMessageBox.information(self, self.tr('Export'), self.tr('Open a project and select a page first.'))
            return
        from utils.io_utils import imread, imwrite
        current_img = self.imgtrans_proj.current_img
        result_path = self.imgtrans_proj.get_result_path(current_img)
        inpainted_path = self.imgtrans_proj.get_inpainted_path(current_img)
        orig_path = osp.join(self.imgtrans_proj.directory, current_img)
        img = None
        source_kind = None

        def get_inpainted():
            if self.imgtrans_proj.inpainted_valid and self.imgtrans_proj.inpainted_array is not None:
                arr = np.array(self.imgtrans_proj.inpainted_array)
                if arr.ndim == 2:
                    arr = np.stack([arr] * 3, axis=-1)
                elif arr.shape[-1] == 4:
                    arr = arr[:, :, :3].copy()
                return arr
            if inpainted_path and osp.exists(inpainted_path):
                arr = imread(inpainted_path)
                if arr is not None and arr.ndim == 3 and arr.shape[-1] == 4:
                    arr = arr[:, :, :3].copy()
                return arr
            return None

        def get_original():
            if orig_path and osp.exists(orig_path):
                return imread(orig_path)
            return None

        if mode == 'original':
            img = get_original()
            source_kind = self.tr('original')
        elif mode == 'inpainted':
            img = get_inpainted()
            source_kind = self.tr('inpainted')
        elif mode == 'result':
            if osp.exists(result_path):
                img = imread(result_path)
                source_kind = self.tr('result')
            elif self.imgtrans_proj.inpainted_valid and self.imgtrans_proj.inpainted_array is not None:
                try:
                    qimg = self.canvas.render_result_img()
                    if qimg is not None and not qimg.isNull():
                        img = pixmap2ndarray(qimg, keep_alpha=False)
                        if img is not None and img.ndim == 3 and img.shape[-1] == 4:
                            img = img[:, :, :3].copy()
                        source_kind = self.tr('result')
                except Exception as e:
                    LOGGER.exception(e)
            if img is None:
                img = get_inpainted()
                source_kind = self.tr('inpainted (no text yet)')
            if img is None:
                img = get_original()
                source_kind = self.tr('original')
        else:
            # mode is None: result → inpainted → original
            if osp.exists(result_path):
                img = imread(result_path)
                source_kind = self.tr('result')
            if img is None:
                img = get_inpainted()
                source_kind = self.tr('inpainted')
            if img is None:
                img = get_original()
                source_kind = self.tr('original')

        if img is None:
            QMessageBox.information(self, self.tr('Export'), self.tr('No image available for current page.'))
            return
        flt = 'PNG (*.png);;JPEG (*.jpg);;WebP (*.webp);;JXL (*.jxl)'
        default_name = osp.splitext(current_img)[0] + '.png'
        path, sel = QFileDialog.getSaveFileName(self, self.tr('Export current page as'), default_name, flt)
        if not path:
            return
        try:
            ext = osp.splitext(path)[1].lower()
            if ext not in ('.png', '.jpg', '.jpeg', '.webp', '.jxl'):
                ext = '.png'
            kw = {'ext': ext, 'quality': pcfg.imgsave_quality}
            if ext == '.webp' and getattr(pcfg, 'imgsave_webp_lossless', False):
                kw['webp_lossless'] = True
            imwrite(path, img, **kw)
            msg = self.tr('Saved to {}').format(path)
            if source_kind:
                msg += ' (' + source_kind + ')'
            QMessageBox.information(self, self.tr('Export'), msg)
        except Exception as e:
            LOGGER.exception(e)
            create_error_dialog(e, self.tr('Export failed'))

    def on_import_doc(self):
        self.import_doc_thread.importDoc(self.imgtrans_proj)

    def on_export_txt(self, dump_target, suffix='.txt'):
        try:
            self.imgtrans_proj.dump_txt(dump_target=dump_target, suffix=suffix)
            create_info_dialog(self.tr('Text file exported to ') + self.imgtrans_proj.dump_txt_path(dump_target, suffix))
        except Exception as e:
            create_error_dialog(e, self.tr('Failed to export as TEXT file'))

    def on_export_lptxt(self):
        """Export translations to LPtxt format (for auto-labeling tools)."""
        if not self.imgtrans_proj or not self.imgtrans_proj.directory:
            QMessageBox.information(self, self.tr('Export'), self.tr('Open a project first.'))
            return
        reply = QMessageBox.question(
            self,
            self.tr('Export translation to LPtxt'),
            self.tr('Include font and size info in the LPtxt output?'),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Cancel:
            return
        include_font_info = reply == QMessageBox.StandardButton.Yes
        try:
            path = self.imgtrans_proj.dump_lptxt(include_font_info=include_font_info)
            create_info_dialog(self.tr('LPtxt exported to: ') + path)
        except Exception as e:
            create_error_dialog(e, self.tr('Failed to export LPtxt'))

    def on_import_trans_txt(self):
        try:
            selected_file = ''
            dialog = QFileDialog()
            selected_file = str(dialog.getOpenFileUrl(self.parent(), self.tr('Import *.md/*.txt'), filter="*.txt *.md *.TXT *.MD")[0].toLocalFile())
            if not osp.exists(selected_file):
                return

            all_matched, match_rst = self.imgtrans_proj.load_translation_from_txt(selected_file)
            matched_pages = match_rst['matched_pages']

            if self.imgtrans_proj.current_img in matched_pages:
                self.canvas.clear_undostack(update_saved_step=True)
                self.st_manager.updateSceneTextitems()

            if all_matched:
                msg = self.tr('Translation imported and matched successfully.')
            else:
                msg = self.tr('Imported txt file not fully matched with current project, please make sure source txt file structured like results from \"export TXT/markdown\"')
                if len(match_rst['missing_pages']) > 0:
                    msg += '\n' + self.tr('Missing pages: ') + '\n'
                    msg += '\n'.join(match_rst['missing_pages'])
                if len(match_rst['unexpected_pages']) > 0:
                    msg += '\n' + self.tr('Unexpected pages: ') + '\n'
                    msg += '\n'.join(match_rst['unexpected_pages'])
                if len(match_rst['unmatched_pages']) > 0:
                    msg += '\n' + self.tr('Unmatched pages: ') + '\n'
                    msg += '\n'.join(match_rst['unmatched_pages'])
                msg = msg.strip()

            for pagename in matched_pages:
                for blk in self.imgtrans_proj.pages[pagename]:
                    blk.translation = self.mtSubWidget.sub_text(blk.translation)
            
            create_info_dialog(msg)

        except Exception as e:
            create_error_dialog(e, self.tr('Failed to import translation from ') + selected_file)

    def on_reveal_file(self):
        current_img_path = self.imgtrans_proj.current_img_path()
        if sys.platform == 'win32':
            # qprocess seems to fuck up with "\""
            p = "\""+str(Path(current_img_path))+"\""
            subprocess.Popen("explorer.exe /select,"+p, shell=True)
        elif sys.platform == 'darwin':
            p = "\""+current_img_path+"\""
            subprocess.Popen("open -R "+p, shell=True)

    def on_translate_images(self, image_names: list):
        if not image_names:
            return
        self.on_run_imgtrans(pages_to_process=image_names)

    def on_run_ocr_images(self, image_names: list):
        if not image_names:
            return
        self._run_stages_restore = (
            pcfg.module.enable_detect,
            pcfg.module.enable_ocr,
            pcfg.module.enable_translate,
            pcfg.module.enable_inpaint,
        )
        pcfg.module.enable_detect = False
        pcfg.module.enable_ocr = True
        pcfg.module.enable_translate = False
        pcfg.module.enable_inpaint = False
        for idx, sa in enumerate(self.titleBar.stageActions):
            sa.setChecked(pcfg.module.stage_enabled(idx))
        self.on_run_imgtrans(pages_to_process=image_names)

    def on_run_detect_images(self, image_names: list):
        if not image_names:
            return
        self._run_stages_restore = (
            pcfg.module.enable_detect,
            pcfg.module.enable_ocr,
            pcfg.module.enable_translate,
            pcfg.module.enable_inpaint,
        )
        pcfg.module.enable_detect = True
        pcfg.module.enable_ocr = False
        pcfg.module.enable_translate = False
        pcfg.module.enable_inpaint = False
        for idx, sa in enumerate(self.titleBar.stageActions):
            sa.setChecked(pcfg.module.stage_enabled(idx))
        self.on_run_imgtrans(pages_to_process=image_names)

    def on_run_translate_images(self, image_names: list):
        if not image_names:
            return
        self._run_stages_restore = (
            pcfg.module.enable_detect,
            pcfg.module.enable_ocr,
            pcfg.module.enable_translate,
            pcfg.module.enable_inpaint,
        )
        pcfg.module.enable_detect = False
        pcfg.module.enable_ocr = False
        pcfg.module.enable_translate = True
        pcfg.module.enable_inpaint = False
        for idx, sa in enumerate(self.titleBar.stageActions):
            sa.setChecked(pcfg.module.stage_enabled(idx))
        self.on_run_imgtrans(pages_to_process=image_names)

    def on_run_inpaint_images(self, image_names: list):
        if not image_names:
            return
        self._run_stages_restore = (
            pcfg.module.enable_detect,
            pcfg.module.enable_ocr,
            pcfg.module.enable_translate,
            pcfg.module.enable_inpaint,
        )
        pcfg.module.enable_detect = False
        pcfg.module.enable_ocr = False
        pcfg.module.enable_translate = False
        pcfg.module.enable_inpaint = True
        for idx, sa in enumerate(self.titleBar.stageActions):
            sa.setChecked(pcfg.module.stage_enabled(idx))
        self.on_run_imgtrans(pages_to_process=image_names)

    def on_toggle_page_ignored(self, pagenames: list, ignored: bool):
        """Mark selected pages as ignored (skip in full run) or include in run."""
        if not pagenames or self.imgtrans_proj.is_empty:
            return
        for name in pagenames:
            if name in self.imgtrans_proj.pages:
                self.imgtrans_proj.set_page_ignored(name, ignored)
        if self.imgtrans_proj.directory:
            self.imgtrans_proj.save()
            self.canvas.setProjSaveState(False)
        self._update_page_list_ignored_style()

    def _update_page_list_ignored_style(self):
        """Update page list items to show ignored state: tooltip, muted text, and darkened/lightened row background."""
        from qtpy.QtGui import QBrush, QColor
        default_fg = self.pageList.palette().brush(self.pageList.foregroundRole())
        default_bg = self.pageList.palette().brush(self.pageList.backgroundRole())
        # Ignored: muted foreground and distinct row background (darkened in light theme, lightened in dark theme)
        if getattr(pcfg, 'darkmode', False):
            ignored_fg = QBrush(QColor(120, 120, 120))
            ignored_bg = QBrush(QColor(45, 45, 48))  # darker than list bg
        else:
            ignored_fg = QBrush(QColor(100, 100, 100))
            ignored_bg = QBrush(QColor(232, 232, 234))  # light gray row
        for i in range(self.pageList.count()):
            item = self.pageList.item(i)
            if item is None:
                continue
            name = item.text()
            is_ignored = self.imgtrans_proj.is_page_ignored(name) if not self.imgtrans_proj.is_empty else False
            if is_ignored:
                item.setToolTip(self.tr('Ignored in run (right-click → Include in run to process)'))
                item.setForeground(ignored_fg)
                item.setBackground(ignored_bg)
            else:
                item.setToolTip('')
                item.setForeground(default_fg)
                item.setBackground(default_bg)

    def on_remove_images(self, image_names: list):
        if not image_names:
            return
        if len(image_names) == 1:
            confirm_msg = self.tr('Are you sure you want to remove "{}" from the project?').format(image_names[0])
        else:
            confirm_msg = self.tr('Are you sure you want to remove {} images from the project?').format(len(image_names))
        reply = QMessageBox.question(
            self, self.tr('Confirm Removal'), confirm_msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        current_img = self.imgtrans_proj.current_img
        need_switch = current_img in image_names
        next_img = None
        for img_name in self.imgtrans_proj.pages.keys():
            if img_name not in image_names:
                next_img = img_name
                break
        if need_switch and next_img:
            if self.canvas.projstate_unsaved:
                self.saveCurrentPage()
            self.imgtrans_proj.set_current_img(next_img)
        for img_name in image_names:
            if img_name in self.imgtrans_proj.pages:
                del self.imgtrans_proj.pages[img_name]
            if img_name in self.imgtrans_proj._pagename2idx:
                idx = self.imgtrans_proj._pagename2idx[img_name]
                del self.imgtrans_proj._pagename2idx[img_name]
                if idx in self.imgtrans_proj._idx2pagename:
                    del self.imgtrans_proj._idx2pagename[idx]
        self.imgtrans_proj._pagename2idx = {}
        self.imgtrans_proj._idx2pagename = {}
        for ii, imgname in enumerate(self.imgtrans_proj.pages.keys()):
            self.imgtrans_proj._pagename2idx[imgname] = ii
            self.imgtrans_proj._idx2pagename[ii] = imgname
        if len(self.imgtrans_proj.pages) == 0:
            self.imgtrans_proj.set_current_img(None)
            self.canvas.clear_undostack(update_saved_step=True)
            self.canvas.drop_files = []
            self.canvas.drop_folder = None
            self.canvas.updateCanvas()
            self.st_manager.clearSceneTextitems()
            self.titleBar.setTitleContent(page_name="")
        elif need_switch and next_img:
            self.canvas.clear_undostack(update_saved_step=True)
            self.canvas.drop_files = []
            self.canvas.drop_folder = None
            self.canvas.updateCanvas()
            self.st_manager.updateSceneTextitems()
            self.titleBar.setTitleContent(page_name=self.imgtrans_proj.current_img)
        self.imgtrans_proj.save()
        self.updatePageList()

    def on_set_gsearch_widget(self):
        setup = self.leftBar.globalSearchChecker.isChecked()
        if setup:
            if self.leftStackWidget.isHidden():
                self.leftStackWidget.show()
            self.leftBar.showPageListLabel.setChecked(False)
            self.leftStackWidget.setCurrentWidget(self.global_search_widget)
        else:
            self.leftStackWidget.hide()

    def on_fin_export_doc(self):
        msg = QMessageBox()
        msg.setText(self.tr('Export to ') + self.imgtrans_proj.doc_path())
        msg.exec_()

    def on_fin_import_doc(self):
        self.st_manager.updateSceneTextitems()

    def on_global_replace_finished(self):
        rt = self.global_search_widget.replace_thread
        self.canvas.push_text_command(
            GlobalRepalceAllCommand(rt.sceneitem_list, rt.background_list, rt.target_text, self.imgtrans_proj)
        )
        rt.sceneitem_list = None
        rt.background_list = None

    def on_darkmode_triggered(self):
        pcfg.darkmode = self.titleBar.darkModeAction.isChecked()
        self.titleBar.themeLightAction.setChecked(not pcfg.darkmode)
        self.titleBar.themeDarkAction.setChecked(pcfg.darkmode)
        self.resetStyleSheet()

    def _on_theme_light_triggered(self):
        pcfg.darkmode = False
        self.titleBar.darkModeAction.setChecked(False)
        self.titleBar.themeLightAction.setChecked(True)
        self.titleBar.themeDarkAction.setChecked(False)
        self.resetStyleSheet()

    def _on_theme_dark_triggered(self):
        pcfg.darkmode = True
        self.titleBar.darkModeAction.setChecked(True)
        self.titleBar.themeLightAction.setChecked(False)
        self.titleBar.themeDarkAction.setChecked(True)
        self.resetStyleSheet()

    def _on_bubbly_ui_triggered(self):
        pcfg.bubbly_ui = self.titleBar.bubblyUIAction.isChecked()
        self.resetStyleSheet()
        self.save_config()

    def on_config_darkmode_changed(self, checked: bool):
        self.titleBar.darkModeAction.setChecked(checked)
        self.titleBar.themeLightAction.setChecked(not checked)
        self.titleBar.themeDarkAction.setChecked(checked)
        self.resetStyleSheet()
        self.save_config()

    def on_config_bubbly_ui_changed(self, checked: bool):
        self.titleBar.bubblyUIAction.setChecked(checked)
        self.resetStyleSheet()
        self.save_config()

    def _on_config_custom_cursor_changed(self):
        self._apply_custom_cursor()
        self.save_config()

    def ocr_postprocess(self, textblocks: List[TextBlock], img, ocr_module=None, **kwargs):
        for blk in textblocks:
            text = blk.get_text()
            blk.text = self.ocrSubWidget.sub_text(text)

        # 字体检测：在 OCR 完成后按配置执行（按需导入以减少启动开销）
        try:
            if pcfg.module.ocr_font_detect:
                try:
                    from utils import font_detect
                    for blk in textblocks:
                        try:
                            name, conf = font_detect.detect_font_from_block(img, blk)
                            blk._detected_font_name = name
                            blk._detected_font_confidence = float(conf)
                        except Exception:
                            # don't break the pipeline on detector errors
                            blk._detected_font_name = ''
                            blk._detected_font_confidence = 0.0
                except Exception:
                    # failed to import or run detector
                    pass
        except Exception:
            pass

    def translate_preprocess(self, translations: List[str] = None, textblocks: List[TextBlock] = None, translator = None, source_text:list = []):
        for i in range(len(source_text)):
            source_text[i] = self.mtPreSubWidget.sub_text(source_text[i])

    def translate_postprocess(self, translations: List[str] = None, textblocks: List[TextBlock] = None, translator = None):
        if not self.postprocess_mt_toggle:
            return
        
        for ii, tr in enumerate(translations):
            translations[ii] = self.mtSubWidget.sub_text(tr)

    def on_copy_src(self):
        blks = self.canvas.selected_text_items()
        if len(blks) == 0:
            return
        
        if isinstance(self.module_manager.translator, GPTTranslator):
            src_list = [self.st_manager.pairwidget_list[blk.idx].e_source.toPlainText() for blk in blks]
            src_txt = ''
            for (prompt, num_src) in self.module_manager.translator._assemble_prompts(src_list, max_tokens=4294967295):
                src_txt += prompt
            src_txt = src_txt.strip()
        else:
            src_list = [self.st_manager.pairwidget_list[blk.idx].e_source.toPlainText().strip().replace('\n', ' ') for blk in blks]
            src_txt = '\n'.join(src_list)

        self.st_manager.app_clipborad.setText(src_txt, QClipboard.Mode.Clipboard)

    def on_paste_src(self):
        blks = self.canvas.selected_text_items()
        if len(blks) == 0:
            return

        src_widget_list = [self.st_manager.pairwidget_list[blk.idx].e_source for blk in blks]
        text_list = self.st_manager.app_clipborad.text().split('\n')
        
        n_paragraph = min(len(src_widget_list), len(text_list))
        if n_paragraph < 1:
            return
        
        src_widget_list = src_widget_list[:n_paragraph]
        text_list = text_list[:n_paragraph]

        self.canvas.push_undo_command(PasteSrcItemsCommand(src_widget_list, text_list))

    def on_copy_trans(self):
        blks = self.canvas.selected_text_items()
        if len(blks) == 0:
            return
        trans_list = [self.st_manager.pairwidget_list[blk.idx].e_trans.toPlainText().strip().replace('\n', ' ') for blk in blks]
        self.st_manager.app_clipborad.setText('\n'.join(trans_list), QClipboard.Mode.Clipboard)

    def on_paste_trans(self):
        blks = self.canvas.selected_text_items()
        if len(blks) == 0:
            return
        trans_widget_list = [self.st_manager.pairwidget_list[blk.idx].e_trans for blk in blks]
        text_list = self.st_manager.app_clipborad.text().split('\n')
        n_paragraph = min(len(trans_widget_list), len(text_list))
        if n_paragraph < 1:
            return
        trans_widget_list = trans_widget_list[:n_paragraph]
        text_list = text_list[:n_paragraph]
        for w, t in zip(trans_widget_list, text_list):
            w.setPlainText(t)
        self.canvas.setProjSaveState(False)

    def on_clear_src(self):
        blks = self.canvas.selected_text_items()
        for blk in blks:
            self.st_manager.pairwidget_list[blk.idx].e_source.setPlainText('')
        if blks:
            self.canvas.setProjSaveState(False)

    def on_clear_trans(self):
        blks = self.canvas.selected_text_items()
        for blk in blks:
            self.st_manager.pairwidget_list[blk.idx].e_trans.setPlainText('')
        if blks:
            self.canvas.setProjSaveState(False)

    def on_select_all_canvas(self):
        # Comic trans (canvas) view is centralStackWidget index 1; index 0 = welcome, 2 = config
        if self.centralStackWidget.currentIndex() != 1:
            return
        self.st_manager.set_blkitems_selection(True)

    def on_spell_check_src(self):
        """Open spell check panel and run check on source text so misspelled words are listed."""
        try:
            from utils.ocr_spellcheck import _init_enchant
            if not _init_enchant():
                raise RuntimeError("enchant not available")
        except Exception:
            QMessageBox.warning(
                self,
                self.tr("Spell check"),
                self.tr("Spell check requires pyenchant. Install it and a system dictionary (e.g. en_US)."),
            )
            return
        if self.centralStackWidget.currentIndex() != 0:
            return
        if self.rightComicTransStackPanel.isHidden():
            self.rightComicTransStackPanel.show()
            self.bottomBar.paintChecker.setChecked(True)
        self.rightComicTransStackPanel.setCurrentIndex(2)
        self.spellCheckPanel.run_check(source_not_translation=True)

    def on_spell_check_trans(self):
        """Open spell check panel and run check on translation so misspelled words are listed."""
        try:
            from utils.ocr_spellcheck import _init_enchant
            if not _init_enchant():
                raise RuntimeError("enchant not available")
        except Exception:
            QMessageBox.warning(
                self,
                self.tr("Spell check"),
                self.tr("Spell check requires pyenchant. Install it and a system dictionary (e.g. en_US)."),
            )
            return
        if self.centralStackWidget.currentIndex() != 0:
            return
        if self.rightComicTransStackPanel.isHidden():
            self.rightComicTransStackPanel.show()
            self.bottomBar.paintChecker.setChecked(True)
        self.rightComicTransStackPanel.setCurrentIndex(2)
        self.spellCheckPanel.run_check(source_not_translation=False)

    def on_trim_whitespace(self):
        """Trim leading/trailing whitespace from each line in selected blocks (source and translation)."""
        blks = self.canvas.selected_text_items()
        if not blks:
            return
        for blkitem in blks:
            blk = blkitem.blk
            pw = self.st_manager.pairwidget_list[blkitem.idx]
            src = pw.e_source.toPlainText()
            trans = pw.e_trans.toPlainText()
            src_trimmed = "\n".join(line.strip() for line in src.split("\n"))
            trans_trimmed = "\n".join(line.strip() for line in trans.split("\n"))
            if getattr(blk, "text", None) and isinstance(blk.text, list):
                blk.text = [line.strip() for line in blk.text]
            blk.translation = trans_trimmed
            pw.e_source.setPlainText(src_trimmed)
            pw.e_trans.setPlainText(trans_trimmed)
        self.canvas.setProjSaveState(False)

    def on_to_uppercase(self):
        """Convert source and translation of selected blocks to uppercase."""
        blks = self.canvas.selected_text_items()
        if not blks:
            return
        for blkitem in blks:
            blk = blkitem.blk
            pw = self.st_manager.pairwidget_list[blkitem.idx]
            src = pw.e_source.toPlainText().upper()
            trans = pw.e_trans.toPlainText().upper()
            if getattr(blk, "text", None) and isinstance(blk.text, list):
                blk.text = [line.upper() for line in blk.text]
            blk.translation = trans
            pw.e_source.setPlainText(src)
            pw.e_trans.setPlainText(trans)
        self.canvas.setProjSaveState(False)

    def on_to_lowercase(self):
        """Convert source and translation of selected blocks to lowercase."""
        blks = self.canvas.selected_text_items()
        if not blks:
            return
        for blkitem in blks:
            blk = blkitem.blk
            pw = self.st_manager.pairwidget_list[blkitem.idx]
            src = pw.e_source.toPlainText().lower()
            trans = pw.e_trans.toPlainText().lower()
            if getattr(blk, "text", None) and isinstance(blk.text, list):
                blk.text = [line.lower() for line in blk.text]
            blk.translation = trans
            pw.e_source.setPlainText(src)
            pw.e_trans.setPlainText(trans)
        self.canvas.setProjSaveState(False)

    def on_toggle_strikethrough(self):
        """Toggle strikethrough on selected text blocks."""
        blks = self.canvas.selected_text_items()
        if not blks:
            return
        gf = self.textPanel.formatpanel.global_format
        new_val = not getattr(blks[0].fontformat, 'strikethrough', False)
        from ui import fontformat_commands as FC
        FC.ffmt_change_strikethrough('strikethrough', new_val, gf, True, blks)
        self.textPanel.formatpanel.formatBtnGroup.strikethroughBtn.setChecked(new_val)
        self.st_manager.updateSceneTextitems()
        self.canvas.setProjSaveState(False)

    def on_set_gradient_type(self, gradient_type: int):
        """Set gradient type (0=Linear, 1=Radial) on selected text blocks."""
        blks = self.canvas.selected_text_items()
        if not blks:
            return
        gf = self.textPanel.formatpanel.global_format
        from ui import fontformat_commands as FC
        FC.ffmt_change_gradient_type('gradient_type', gradient_type, gf, True, blks)
        self.st_manager.updateSceneTextitems()
        self.canvas.setProjSaveState(False)

    def on_set_text_on_path(self, mode: int):
        """Set text on path (0=None, 1=Circular, 2=Arc) on selected text blocks."""
        blks = self.canvas.selected_text_items()
        if not blks:
            return
        gf = self.textPanel.formatpanel.global_format
        from ui import fontformat_commands as FC
        FC.ffmt_change_text_on_path('text_on_path', [mode] * len(blks), gf, True, blks)
        self.st_manager.updateSceneTextitems()
        self.canvas.setProjSaveState(False)

    def on_run_detect_region(self, rect, replace_page: bool = False):
        if not self.imgtrans_proj.img_valid or self.imgtrans_proj.img_array is None:
            return
        page_name = self.imgtrans_proj.current_img
        if not page_name:
            return
        self.module_manager.run_detect_region(rect, self.imgtrans_proj.img_array, page_name, replace_if_full_page=replace_page)

    def on_detect_region_finished(self, page_name: str, blk_list: List, replace: bool = False):
        if page_name != self.imgtrans_proj.current_img:
            return
        if replace:
            # Replace all blocks on the page (e.g. "Detect text on page")
            old_blks = self.imgtrans_proj.pages.get(page_name, [])
            self.imgtrans_proj.pages[page_name] = list(blk_list)
            # Old blocks are TextBlocks (no .idx); widgets and canvas items are in same order as page, so use position index.
            n_old = len(old_blks)
            to_remove = [self.st_manager.textblk_item_list[i] for i in range(n_old) if i < len(self.st_manager.textblk_item_list)]
            to_remove_widgets = [self.st_manager.pairwidget_list[i] for i in range(n_old) if i < len(self.st_manager.pairwidget_list)]
            if to_remove:
                self.st_manager.deleteTextblkItemList(to_remove, to_remove_widgets)
            im_h, im_w = self.imgtrans_proj.img_array.shape[:2]
            for blk in blk_list:
                if getattr(blk, 'lines', None) and len(blk.lines) > 0:
                    examine_textblk(blk, im_w, im_h, sort=True)
            for blk in blk_list:
                self.st_manager.addTextBlock(blk)
        elif blk_list:
            im_h, im_w = self.imgtrans_proj.img_array.shape[:2]
            for blk in blk_list:
                if getattr(blk, 'lines', None) and len(blk.lines) > 0:
                    examine_textblk(blk, im_w, im_h, sort=True)
            self.imgtrans_proj.pages[page_name].extend(blk_list)
            for blk in blk_list:
                self.st_manager.addTextBlock(blk)
        self.canvas.setProjSaveState(False)
        self.global_search_widget.set_document_edited()

    def on_merge_selected_blocks(self):
        blks = self.canvas.selected_text_items()
        if len(blks) < 2:
            return
        blks = sorted(blks, key=lambda b: b.idx)
        first = blks[0]
        page_name = self.imgtrans_proj.current_img
        if not page_name:
            return
        src_parts = [self.st_manager.pairwidget_list[b.idx].e_source.toPlainText() for b in blks]
        trans_parts = [self.st_manager.pairwidget_list[b.idx].e_trans.toPlainText() for b in blks]
        first.blk.text = ['\n'.join(src_parts)]
        first.blk.translation = '\n'.join(trans_parts)
        self.st_manager.pairwidget_list[first.idx].e_source.setPlainText('\n'.join(src_parts))
        self.st_manager.pairwidget_list[first.idx].e_trans.setPlainText('\n'.join(trans_parts))
        indices_to_remove = sorted([b.idx for b in blks[1:]], reverse=True)
        for i in indices_to_remove:
            self.imgtrans_proj.pages[page_name].pop(i)
        to_remove_items = blks[1:]
        to_remove_widgets = [self.st_manager.pairwidget_list[b.idx] for b in blks[1:]]
        self.st_manager.deleteTextblkItemList(to_remove_items, to_remove_widgets)
        self.canvas.setProjSaveState(False)
        self.global_search_widget.set_document_edited()

    def on_split_selected_regions(self):
        """Try to split each selected region into multiple regions using image gaps (e.g. two bubbles in one box)."""
        blks = self.canvas.selected_text_items()
        if not blks:
            return
        page_name = self.imgtrans_proj.current_img
        if not page_name or not self.imgtrans_proj.img_valid or self.imgtrans_proj.img_array is None:
            return
        img = self.imgtrans_proj.img_array
        if img.ndim == 3:
            crop_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            crop_img = img
        h_img, w_img = img.shape[:2]
        # Process in descending index so indices stay valid
        for blk_item in sorted(blks, key=lambda b: b.idx, reverse=True):
            blk = blk_item.blk
            x1, y1, x2, y2 = blk.xyxy
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w_img, int(x2)), min(h_img, int(y2))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = crop_img[y1:y2, x1:x2]
            if crop.size < 100:
                continue
            try:
                span_list, _ = split_textblock(crop, crop_ratio=0.2, shrink=True, discard=True, recheck=False)
            except Exception:
                continue
            if not span_list or len(span_list) <= 1:
                continue
            new_blks = []
            for span in span_list:
                if span.top is None or span.bottom is None or span.left is None or span.right is None:
                    continue
                ox1 = x1 + max(0, span.left)
                oy1 = y1 + max(0, span.top)
                ox2 = x1 + min(crop.shape[1], span.right)
                oy2 = y1 + min(crop.shape[0], span.bottom)
                if ox2 <= ox1 or oy2 <= oy1:
                    continue
                new_blk = copy.copy(blk)
                new_blk.xyxy = [ox1, oy1, ox2, oy2]
                new_blk.lines = [[[ox1, oy1], [ox2, oy1], [ox2, oy2], [ox1, oy2]]]
                new_blk.text = [""]
                new_blk.translation = ""
                new_blks.append(new_blk)
            if len(new_blks) <= 1:
                continue
            idx = blk_item.idx
            self.imgtrans_proj.pages[page_name].pop(idx)
            self.st_manager.deleteTextblkItemList([blk_item], [self.st_manager.pairwidget_list[idx]])
            for i, nb in enumerate(new_blks):
                self.imgtrans_proj.pages[page_name].insert(idx + i, nb)
            self.st_manager.insertTextBlocksAt(idx, new_blks)
        self.canvas.setProjSaveState(False)
        self.global_search_widget.set_document_edited()

    def on_move_blocks_up(self):
        blks = self.canvas.selected_text_items()
        if len(blks) != 1:
            return
        idx = blks[0].idx
        if idx <= 0:
            return
        self.st_manager.swap_block_positions(idx - 1, idx)
        self.canvas.setProjSaveState(False)

    def on_move_blocks_down(self):
        blks = self.canvas.selected_text_items()
        if len(blks) != 1:
            return
        idx = blks[0].idx
        n = len(self.st_manager.textblk_item_list)
        if idx >= n - 1:
            return
        self.st_manager.swap_block_positions(idx, idx + 1)
        self.canvas.setProjSaveState(False)

    def on_import_image_to_blk(self):
        """PR #1070: Import image as foreground overlay for the single selected empty block."""
        if self.imgtrans_proj.is_empty or not self.imgtrans_proj.directory:
            return
        sel = self.canvas.selected_text_items()
        if len(sel) != 1 or not sel[0].document().isEmpty():
            return
        item = sel[0]
        blk = getattr(item, 'blk', None)
        if blk is None:
            return
        flt = self.tr("Images") + " (*.png *.jpg *.jpeg *.bmp *.webp);;" + self.tr("All files") + " (*.*)"
        path, _ = QFileDialog.getOpenFileName(self, self.tr("Import Image"), None, flt)
        if not path or not osp.isfile(path):
            return
        proj_dir = self.imgtrans_proj.directory
        overlay_dir = osp.join(proj_dir, "overlays")
        try:
            os.makedirs(overlay_dir, exist_ok=True)
        except OSError:
            create_error_dialog(self, self.tr("Could not create overlays folder."))
            return
        base = osp.splitext(osp.basename(self.imgtrans_proj.current_img or "page"))[0]
        ext = osp.splitext(path)[1] or ".png"
        dest_name = f"{base}_blk{item.idx}{ext}"
        dest_path = osp.join(overlay_dir, dest_name)
        try:
            shutil.copy2(path, dest_path)
        except OSError as e:
            create_error_dialog(self, self.tr("Could not copy image: ") + str(e))
            return
        rel_path = osp.join("overlays", dest_name)
        blk.foreground_image_path = rel_path
        item.invalidate_foreground_cache()
        item.update()
        self.imgtrans_proj.save()
        self.canvas.setProjSaveState(False)

    def on_clear_overlay(self):
        """PR #1070: Clear foreground image overlay from the selected block."""
        sel = self.canvas.selected_text_items()
        if len(sel) != 1:
            return
        item = sel[0]
        blk = getattr(item, 'blk', None)
        if blk is None or not getattr(blk, 'foreground_image_path', None):
            return
        blk.foreground_image_path = None
        item.invalidate_foreground_cache()
        item.update()
        if not self.imgtrans_proj.is_empty and self.imgtrans_proj.directory:
            self.imgtrans_proj.save()
            self.canvas.setProjSaveState(False)
    
    def run_batch(self, exec_dirs: Union[List, str], **kwargs):
        if not isinstance(exec_dirs, List):
            exec_dirs = [x.strip() for x in exec_dirs.split(',') if x.strip()]
        valid_dirs = []
        from utils.validation import normalize_zip_file_input, validate_batch_input_path
        for d in exec_dirs:
            d = normalize_zip_file_input(d)
            if not d:
                continue
            try:
                path, kind = validate_batch_input_path(d)
            except ValueError as e:
                LOGGER.warning(str(e))
                continue
            if kind == "zip":
                if not osp.isfile(path) or not path.lower().endswith(".zip"):
                    continue
                try:
                    job = self._zip_batch.add_zip(path)
                    if not job.project_dirs:
                        LOGGER.warning(f'ZIP contains no images: {path}')
                        continue
                    LOGGER.info(
                        f'ZIP batch: extracted {osp.basename(path)} -> {job.extracted_root} '
                        f'({len(job.project_dirs)} folder(s) with images), output -> {job.output_root}'
                    )
                    valid_dirs.extend(job.project_dirs)
                except Exception as e:
                    LOGGER.warning(f'Failed to extract ZIP {path}: {e}')
                    continue
            else:
                valid_dirs.append(path)
        self.exec_dirs = valid_dirs
        self._batch_cancelled = False
        self._batch_skip_ignored = kwargs.get('skip_ignored_pages', getattr(pcfg, 'skip_ignored_in_run', True))
        self._batch_run_indices = None  # after first item, use same page indices for rest of batch
        self.run_next_dir()

    def run_next_dir(self):
        if len(self.exec_dirs) == 0:
            while self.imsave_thread.isRunning():
                time.sleep(0.1)
            LOGGER.info('finished translating all dirs.')
            try:
                self._zip_batch.cleanup_all()
            except Exception:
                pass
            if self._batch_cancelled:
                self.batch_queue_cancelled.emit()
                self._batch_cancelled = False
            else:
                self.batch_queue_empty.emit()
            self._batch_run_indices = None
            if shared.HEADLESS_CONTINUOUS:
                LOGGER.info('Enter next dirs to translate (comma-separated), or "exit" to quit.')
                try:
                    new_exec_dirs = input()
                except (EOFError, KeyboardInterrupt):
                    new_exec_dirs = 'exit'
                if new_exec_dirs.strip().lower() == 'exit':
                    LOGGER.info('Exiting.')
                    self.app.quit()
                    return
                self.run_batch(new_exec_dirs)
                return
            if shared.HEADLESS:
                self.app.quit()
            return
        d = self.exec_dirs.pop(0)
        self._current_batch_dir = d
        self.batch_queue_item_started.emit(d)
        LOGGER.info(f'translating {d} ...')
        self.openDir(d)
        all_pages = list(self.imgtrans_proj.pages.keys())
        skip_ignored = getattr(self, '_batch_skip_ignored', True)
        if getattr(self, '_batch_run_indices', None) is not None:
            # Use same page indices as first item (e.g. skip page 1 in every chapter)
            pages_to_process = [all_pages[i] for i in self._batch_run_indices if i < len(all_pages)]
        else:
            # First item: compute from this project's ignore state, then store indices for rest of batch
            if skip_ignored:
                pages_to_process = [p for p in all_pages if not self.imgtrans_proj.is_page_ignored(p)]
            else:
                pages_to_process = list(all_pages)
            self._batch_run_indices = [all_pages.index(p) for p in pages_to_process]
        shared.pbar = {}
        npages = len(pages_to_process)
        if npages > 0:
            if pcfg.module.enable_detect:
                shared.pbar['detect'] = tqdm(range(npages), desc="Text Detection")
            if pcfg.module.enable_ocr:
                shared.pbar['ocr'] = tqdm(range(npages), desc="OCR")
            if pcfg.module.enable_translate:
                shared.pbar['translate'] = tqdm(range(npages), desc="Translation")
            if pcfg.module.enable_inpaint:
                shared.pbar['inpaint'] = tqdm(range(npages), desc="Inpaint")
            self.on_run_imgtrans(
                skip_ignored_in_batch=skip_ignored,
                pages_to_process=pages_to_process,
            )

    def setupRegisterWidget(self):
        self.titleBar.viewMenu.addSeparator()
        for cfg_name in shared.config_name_to_view_widget:
            d = shared.config_name_to_view_widget[cfg_name]
            widget: ViewWidget = d['widget']
            action = QAction(widget.action_name, self.titleBar)
            action.setCheckable(True)
            visible = getattr(pcfg, cfg_name)
            action.setChecked(visible)
            action.triggered.connect(self.action_set_view_visible)
            self.titleBar.viewMenu.addAction(action)
            d['action'] = action
            shared.action_to_view_config_name[action] = cfg_name
            widget.set_expend_area(expend=getattr(pcfg, widget.config_expand_name), set_config=False)
            widget.view_hide_btn_clicked.connect(self.on_hide_view_widget)
            widget.setVisible(visible)

        if hasattr(self, 'textPanel') and hasattr(self.textPanel, 'formatpanel'):
            self.textPanel.formatpanel.showGlobalFontFormatBtn.setVisible(
                not getattr(pcfg, 'show_text_style_preset', True)
            )

        # Section 10: Canvas view mode (original / debug / translated / normal) + Fit to width
        self.titleBar.viewMenu.addSeparator()
        from qtpy.QtGui import QActionGroup
        self._canvas_view_group = QActionGroup(self)
        self._canvas_view_group.setExclusive(True)
        mode_actions = []
        for mode, label in [
            ("normal", self.tr("Canvas: Normal")),
            ("original", self.tr("Canvas: Original")),
            ("debug", self.tr("Canvas: Debug (boxes/masks)")),
            ("translated", self.tr("Canvas: Translated")),
        ]:
            act = QAction(label, self.titleBar)
            act.setCheckable(True)
            act.setChecked(getattr(pcfg, "canvas_view_mode", "normal") == mode)
            act.setProperty("canvas_view_mode", mode)
            act.triggered.connect(self._on_canvas_view_mode_triggered)
            self._canvas_view_group.addAction(act)
            self.titleBar.viewMenu.addAction(act)
            mode_actions.append((mode, act))
        self._canvas_view_mode_actions = dict(mode_actions)
        fit_action = QAction(self.tr("Canvas: Fit to width"), self.titleBar)
        fit_action.setToolTip(self.tr("Zoom canvas so image width fits the view."))
        fit_action.triggered.connect(self._on_canvas_fit_to_width)
        self.titleBar.viewMenu.addAction(fit_action)

    def _on_canvas_view_mode_triggered(self):
        action = self.sender()
        if not isinstance(action, QAction):
            return
        mode = action.property("canvas_view_mode")
        if mode:
            pcfg.canvas_view_mode = mode
            self.canvas.updateLayers()

    def _on_canvas_fit_to_width(self):
        self.canvas.fitToWidth()

    def action_set_view_visible(self):
        action: QAction = self.sender()
        show = action.isChecked()
        cfg_name = shared.action_to_view_config_name[action]
        widget: ViewWidget = shared.config_name_to_view_widget[cfg_name]['widget']
        widget.setVisible(show)
        setattr(pcfg, cfg_name, show)
        if cfg_name == 'show_text_style_preset' and hasattr(self, 'textPanel'):
            self.textPanel.formatpanel.showGlobalFontFormatBtn.setVisible(not show)
        if cfg_name == 'text_advanced_format_panel' and hasattr(self, 'textPanel'):
            self.textPanel.formatpanel.showAdvancedFontFormatBtn.setVisible(not show)

    def on_hide_view_widget(self, cfg_name: str):
        d = shared.config_name_to_view_widget[cfg_name]
        widget: ViewWidget = d['widget']
        widget.setVisible(False)
        action: QAction = d['action']
        action.setChecked(False)
        setattr(pcfg, cfg_name, False)
        if cfg_name == 'show_text_style_preset' and hasattr(self, 'textPanel'):
            self.textPanel.formatpanel.showGlobalFontFormatBtn.setVisible(True)
        if cfg_name == 'text_advanced_format_panel' and hasattr(self, 'textPanel'):
            self.textPanel.formatpanel.showAdvancedFontFormatBtn.setVisible(True)