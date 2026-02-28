import os.path as osp
import os, re, traceback, sys
from typing import List, Union
from pathlib import Path
import subprocess
from functools import partial
import time
import cv2

from tqdm import tqdm
from qtpy.QtWidgets import QAction, QFileDialog, QMenu, QHBoxLayout, QVBoxLayout, QApplication, QStackedWidget, QSplitter, QListWidget, QShortcut, QListWidgetItem, QMessageBox, QTextEdit, QPlainTextEdit
from qtpy.QtCore import Qt, QPoint, QSize, QEvent, Signal, QTimer
from qtpy.QtGui import QContextMenuEvent, QTextCursor, QGuiApplication, QIcon, QCloseEvent, QKeySequence, QKeyEvent, QPainter, QClipboard, QImage

from utils.logger import logger as LOGGER
from utils.text_processing import is_cjk, full_len, half_len
from utils.textblock import TextBlock, TextAlignment, examine_textblk
from utils import shared
from utils.message import create_error_dialog, create_info_dialog
from modules.translators.trans_chatgpt import GPTTranslator
from modules import GET_VALID_TEXTDETECTORS, GET_VALID_INPAINTERS, GET_VALID_TRANSLATORS, GET_VALID_OCR
from .misc import parse_stylesheet, set_html_family, QKEY
from utils.config import ProgramConfig, pcfg, save_config, text_styles, save_text_styles, load_textstyle_from, FontFormat
from utils.shortcuts import get_shortcut
from utils.proj_imgtrans import ProjImgTrans
from .canvas import Canvas
from .configpanel import ConfigPanel
from .module_manager import ModuleManager
from .textedit_area import SourceTextEdit, SelectTextMiniMenu, TransTextEdit
from .drawingpanel import DrawingPanel
from .scenetext_manager import SceneTextManager, TextPanel, PasteSrcItemsCommand
from .mainwindowbars import TitleBar, LeftBar, BottomBar
from .io_thread import ImgSaveThread, ImportDocThread, ExportDocThread
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

class PageListView(QListWidget):

    reveal_file = Signal()

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setIconSize(QSize(shared.PAGELIST_THUMBNAIL_SIZE, shared.PAGELIST_THUMBNAIL_SIZE))

    def contextMenuEvent(self, e: QContextMenuEvent):
        menu = QMenu()
        reveal_act = menu.addAction(self.tr('Reveal in File Explorer'))
        rst = menu.exec_(e.globalPos())

        if rst == reveal_act:
            self.reveal_file.emit()

        return super().contextMenuEvent(e)

mainwindow_cls = Widget if shared.HEADLESS else FramelessWindow
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
    
    def __init__(self, app: QApplication, config: ProgramConfig, open_dir='', **exec_args) -> None:
        super().__init__()

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
            self.OpenProj(open_dir)
        elif pcfg.open_recent_on_startup:
            if len(self.leftBar.recent_proj_list) > 0:
                proj_dir = self.leftBar.recent_proj_list[0]
                if osp.exists(proj_dir):
                    self.OpenProj(proj_dir)

        if shared.HEADLESS:
            self.run_batch(**exec_args)
        elif exec_args.get('exec_dirs') and str(exec_args.get('exec_dirs')).strip():
            self.run_batch(**exec_args)

        if shared.ON_MACOS:
            # https://bugreports.qt.io/browse/QTBUG-133215
            self.hideSystemTitleBar()
            self.showMaximized()

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

    def setupUi(self):
        screen_size = QGuiApplication.primaryScreen().geometry().size()
        self.setMinimumWidth(screen_size.width() // 2)
        self.configPanel = ConfigPanel(self)
        self.configPanel.trans_config_panel.show_pre_MT_keyword_window.connect(self.show_pre_MT_keyword_window)
        self.configPanel.trans_config_panel.show_MT_keyword_window.connect(self.show_MT_keyword_window)
        self.configPanel.trans_config_panel.show_OCR_keyword_window.connect(self.show_OCR_keyword_window)

        self.leftBar = LeftBar(self)
        self.leftBar.showPageListLabel.clicked.connect(self.pageLabelStateChanged)
        self.leftBar.imgTransChecked.connect(self.setupImgTransUI)
        self.leftBar.configChecked.connect(self.setupConfigUI)
        self.leftBar.globalSearchChecker.clicked.connect(self.on_set_gsearch_widget)
        self.leftBar.open_dir.connect(self.OpenProj)
        self.leftBar.open_json_proj.connect(self.openJsonProj)
        self.leftBar.save_proj.connect(self.manual_save)
        self.leftBar.export_doc.connect(self.on_export_doc)
        self.leftBar.import_doc.connect(self.on_import_doc)
        self.leftBar.export_src_txt.connect(lambda : self.on_export_txt(dump_target='source'))
        self.leftBar.export_trans_txt.connect(lambda : self.on_export_txt(dump_target='translation'))
        self.leftBar.export_src_md.connect(lambda : self.on_export_txt(dump_target='source', suffix='.md'))
        self.leftBar.export_trans_md.connect(lambda : self.on_export_txt(dump_target='translation', suffix='.md'))
        self.leftBar.import_trans_txt.connect(self.on_import_trans_txt)

        self.pageList = PageListView()
        self.pageList.reveal_file.connect(self.on_reveal_file)
        self.pageList.setHidden(True)
        self.pageList.currentItemChanged.connect(self.pageListCurrentItemChanged)

        self.leftStackWidget = QStackedWidget(self)
        self.leftStackWidget.addWidget(self.pageList)

        self.global_search_widget = GlobalSearchWidget(self.leftStackWidget)
        self.global_search_widget.req_update_pagetext.connect(self.on_req_update_pagetext)
        self.global_search_widget.req_move_page.connect(self.on_req_move_page)
        self.imsave_thread.img_writed.connect(self.global_search_widget.on_img_writed)
        self.global_search_widget.search_tree.result_item_clicked.connect(self.on_search_result_item_clicked)
        self.leftStackWidget.addWidget(self.global_search_widget)
        
        self.centralStackWidget = QStackedWidget(self)
        
        self.titleBar = TitleBar(self)
        self.titleBar.closebtn_clicked.connect(self.on_closebtn_clicked)
        self.titleBar.display_lang_changed.connect(self.on_display_lang_changed)
        self.bottomBar = BottomBar(self)
        self.bottomBar.textedit_checkchanged.connect(self.setTextEditMode)
        self.bottomBar.paintmode_checkchanged.connect(self.setPaintMode)
        self.bottomBar.textblock_checkchanged.connect(self.setTextBlockMode)

        mainHLayout = QHBoxLayout()
        mainHLayout.addWidget(self.leftBar)
        mainHLayout.addWidget(self.centralStackWidget)
        mainHLayout.setContentsMargins(0, 0, 0, 0)
        mainHLayout.setSpacing(0)

        # set up canvas
        SW.canvas = self.canvas = Canvas()
        self.canvas.imgtrans_proj = self.imgtrans_proj
        self.canvas.gv.hide_canvas.connect(self.onHideCanvas)
        self.canvas.proj_savestate_changed.connect(self.on_savestate_changed)
        self.canvas.textstack_changed.connect(self.on_textstack_changed)
        self.canvas.run_blktrans.connect(self.on_run_blktrans)
        self.canvas.drop_open_folder.connect(self.dropOpenDir)
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
        self.canvas.run_detect_region.connect(self.on_run_detect_region)
        self.canvas.merge_selected_blocks_signal.connect(self.on_merge_selected_blocks)
        self.canvas.move_blocks_up_signal.connect(self.on_move_blocks_up)
        self.canvas.move_blocks_down_signal.connect(self.on_move_blocks_down)

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
        self.rightComicTransStackPanel = QStackedWidget(self)
        self.rightComicTransStackPanel.addWidget(self.drawingPanel)
        self.rightComicTransStackPanel.addWidget(self.textPanel)
        self.rightComicTransStackPanel.currentChanged.connect(self.on_transpanel_changed)

        self.comicTransSplitter = QSplitter(Qt.Orientation.Horizontal)
        self.comicTransSplitter.addWidget(self.leftStackWidget)
        self.comicTransSplitter.addWidget(self.canvas.gv)
        self.comicTransSplitter.addWidget(self.rightComicTransStackPanel)

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
        self.imgtrans_progress_msgbox = ImgtransProgressMessageBox()
        self.resetStyleSheet()

    def on_finish_setdetector(self):
        module_manager = self.module_manager
        if module_manager.textdetector is not None:
            name = module_manager.textdetector.name
            pcfg.module.textdetector = name
            self.configPanel.detect_config_panel.setDetector(name)
            self.bottomBar.textdet_selector.setSelectedValue(name)
            LOGGER.info('Text detector set to {}'.format(name))

    def on_finish_setocr(self):
        module_manager = self.module_manager
        if module_manager.ocr is not None:
            name = module_manager.ocr.name
            pcfg.module.ocr = name
            self.configPanel.ocr_config_panel.setOCR(name)
            self.bottomBar.ocr_selector.setSelectedValue(name)
            LOGGER.info('OCR set to {}'.format(name))

    def on_finish_setinpainter(self):
        module_manager = self.module_manager
        if module_manager.inpainter is not None:
            name = module_manager.inpainter.name
            pcfg.module.inpainter = name
            self.configPanel.inpaint_config_panel.setInpainter(name)
            self.bottomBar.inpaint_selector.setSelectedValue(name)
            LOGGER.info('Inpainter set to {}'.format(name))

    def on_finish_settranslator(self):
        module_manager = self.module_manager
        translator = module_manager.translator
        if translator is not None:
            name = translator.name
            pcfg.module.translator = name
            self.bottomBar.trans_selector.finishSetTranslator(translator)
            self.configPanel.trans_config_panel.finishSetTranslator(translator)
            LOGGER.info('Translator set to {}'.format(name))
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

        self.leftBar.run_imgtrans_clicked.connect(self.run_imgtrans)

        self.titleBar.darkModeAction.setChecked(pcfg.darkmode)

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
        self.configPanel.display_lang_changed.connect(self.on_display_lang_changed)
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

    def setupImgTransUI(self):
        self.centralStackWidget.setCurrentIndex(0)
        if self.leftBar.needleftStackWidget():
            self.leftStackWidget.show()
        else:
            self.leftStackWidget.hide()

    def setupConfigUI(self):
        self.centralStackWidget.setCurrentIndex(1)

    def set_display_lang(self, lang: str):
        self.retranslateUI()

    def OpenProj(self, proj_path: str):
        if osp.isdir(proj_path):
            self.openDir(proj_path)
        else:
            self.openJsonProj(proj_path)
        
        if pcfg.let_textstyle_indep_flag and not shared.HEADLESS:
            self.load_textstyle_from_proj_dir(from_proj=True)

    def load_textstyle_from_proj_dir(self, from_proj=False):
        if from_proj:
            text_style_path = osp.join(self.imgtrans_proj.directory, 'textstyles.json')
        else:
            text_style_path = 'config/textstyles/default.json'
        if osp.exists(text_style_path):
            load_textstyle_from(text_style_path)
            self.textPanel.formatpanel.textstyle_panel.setStyles(text_styles)
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
            self.opening_dir = False
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

    def openJsonProj(self, json_path: str):
        try:
            self.opening_dir = True
            self.imgtrans_proj.load_from_json(json_path)
            self.st_manager.clearSceneTextitems()
            self.leftBar.updateRecentProjList(self.imgtrans_proj.proj_path)
            self.updatePageList()
            self.titleBar.setTitleContent(osp.basename(self.imgtrans_proj.proj_path))
            self.opening_dir = False
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

    def pageLabelStateChanged(self):
        setup = self.leftBar.showPageListLabel.isChecked()
        if setup:
            if self.leftStackWidget.isHidden():
                self.leftStackWidget.show()
            if self.leftBar.globalSearchChecker.isChecked():
                self.leftBar.globalSearchChecker.setChecked(False)
            self.leftStackWidget.setCurrentWidget(self.pageList)
        else:
            self.leftStackWidget.hide()
        pcfg.show_page_list = setup
        save_config()

    def closeEvent(self, event: QCloseEvent) -> None:
        if not self.imgtrans_proj.is_empty:
            self.conditional_save(keep_exist_as_backup=True)
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
        save_config()

    def on_set_default_format_requested(self):
        """Save current global font format to config so it is used as default for new projects/sessions."""
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
        self.titleBar.redo_trigger.connect(self.on_redo)
        self.titleBar.undo_trigger.connect(self.on_undo)
        self.titleBar.page_search_trigger.connect(self.on_page_search)
        self.titleBar.global_search_trigger.connect(self.on_global_search)
        self.titleBar.replacePreMTkeyword_trigger.connect(self.show_pre_MT_keyword_window)
        self.titleBar.replaceMTkeyword_trigger.connect(self.show_MT_keyword_window)
        self.titleBar.replaceOCRkeyword_trigger.connect(self.show_OCR_keyword_window)
        self.titleBar.run_trigger.connect(self.leftBar.runImgtransBtn.click)
        self.titleBar.run_woupdate_textstyle_trigger.connect(self.run_imgtrans_wo_textstyle_update)
        self.titleBar.translate_page_trigger.connect(self.on_transpagebtn_pressed)
        self.titleBar.enable_module.connect(self.on_enable_module)
        self.titleBar.importtstyle_trigger.connect(self.import_tstyles)
        self.titleBar.exporttstyle_trigger.connect(self.export_tstyles)
        self.titleBar.darkmode_trigger.connect(self.on_darkmode_triggered)
        self.titleBar.merge_tool_trigger.connect(self.on_open_merge_tool)
        self.titleBar.re_run_detection_only_trigger.connect(self.on_re_run_detection_only)
        self.titleBar.re_run_ocr_only_trigger.connect(self.on_re_run_ocr_only)
        self.titleBar.batch_export_trigger.connect(self.on_batch_export)
        self.titleBar.validate_project_trigger.connect(self.on_validate_project)
        self.titleBar.manga_source_trigger.connect(self.on_open_manga_source)
        self.titleBar.manage_models_trigger.connect(self.on_open_manage_models)
        self.titleBar.run_preset_full_trigger.connect(self.on_run_preset_full)
        self.titleBar.run_preset_detect_ocr_trigger.connect(self.on_run_preset_detect_ocr)
        self.titleBar.run_preset_translate_trigger.connect(self.on_run_preset_translate)
        self.titleBar.run_preset_inpaint_trigger.connect(self.on_run_preset_inpaint)
        self.titleBar.keyboard_shortcuts_trigger.connect(self.open_shortcuts_dialog)

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
        _mk_shortcut("canvas.select_all", "Ctrl+A", self.shortcutSelectAll)
        _mk_shortcut("canvas.escape", "Escape", self.shortcutEscape)
        _mk_shortcut("format.bold", "Ctrl+B", self.shortcutBold)
        _mk_shortcut("format.italic", "Ctrl+I", self.shortcutItalic)
        _mk_shortcut("format.underline", "Ctrl+U", self.shortcutUnderline)
        _mk_shortcut("canvas.delete_line", "Delete", self.shortcutDelete)

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

    def shortcutCtrlD(self):
        if self.centralStackWidget.currentIndex() == 0:
            if self.drawingPanel.isVisible():
                if self.drawingPanel.currentTool == self.drawingPanel.rectTool:
                    self.drawingPanel.rectPanel.delete_btn.click()
            elif self.canvas.textEditMode():
                self.canvas.delete_textblks.emit(0)

    def shortcutSelectAll(self):
        if self.centralStackWidget.currentIndex() == 0:
            if self.textPanel.isVisible():
                self.st_manager.set_blkitems_selection(True)

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

    def on_open_manage_models(self):
        """Open the Manage models dialog (check downloaded models, download selected)."""
        dlg = ModelManagerDialog(self)
        dlg.exec()

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
                labels = set(s.get('label', '') for s in initial_shapes)
                detail_msg = f"No blocks were merged.\nTotal: {initial_count} text blocks.\nLabel types: {', '.join(labels) or 'none'}\n\n"
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

    def on_batch_export(self):
        """Export all pages as images (and optionally PDF) to a chosen folder."""
        if self.imgtrans_proj.is_empty:
            QMessageBox.information(self, self.tr('Export'), self.tr('Open a project first.'))
            return
        out_dir = QFileDialog.getExistingDirectory(self, self.tr('Select output folder'))
        if not out_dir:
            return
        try:
            from utils.io_utils import imread, imwrite
            result_dir = self.imgtrans_proj.result_dir()
            exported = 0
            missing = []
            for pagename in self.imgtrans_proj.pages:
                result_path = self.imgtrans_proj.get_result_path(pagename)
                if osp.exists(result_path):
                    ext = osp.splitext(result_path)[1]
                    dest = osp.join(out_dir, osp.splitext(pagename)[0] + ext)
                    img = imread(result_path)
                    kw = {'ext': ext if ext in ('.png', '.jpg', '.jpeg', '.webp', '.jxl') else '.png', 'quality': pcfg.imgsave_quality}
                    if kw['ext'] == '.webp' and getattr(pcfg, 'imgsave_webp_lossless', False):
                        kw['webp_lossless'] = True
                    imwrite(dest, img, **kw)
                    exported += 1
                else:
                    missing.append(pagename)
            msg = self.tr('Exported {0} page(s) to {1}.').format(exported, out_dir)
            if missing:
                msg += '\n' + self.tr('Missing result for {0} page(s). Run pipeline first.').format(len(missing))
            QMessageBox.information(self, self.tr('Export'), msg)
        except Exception as e:
            LOGGER.exception(e)
            create_error_dialog(e, self.tr('Batch export failed'), 'BatchExport')

    def on_validate_project(self):
        """Check project: missing images, invalid JSON, duplicate blocks."""
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
        translator = self.module_manager.translator
        if translator is not None:
            translator.set_source(text)
        pcfg.module.translate_source = text
        combobox = self.configPanel.trans_config_panel.source_combobox
        if sender != combobox:
            combobox.blockSignals(True)
            combobox.setCurrentText(text)
            combobox.blockSignals(False)
        combobox = self.bottomBar.trans_selector.src_selector
        if sender != combobox:
            combobox.blockSignals(True)
            combobox.setCurrentText(text)
            combobox.blockSignals(False)

    def on_trans_tgt_changed(self):
        sender = self.sender()
        text = sender.currentText()
        translator = self.module_manager.translator
        if translator is not None:
            translator.set_target(text)
        pcfg.module.translate_target = text
        combobox = self.configPanel.trans_config_panel.target_combobox
        if sender != combobox:
            combobox.blockSignals(True)
            combobox.setCurrentText(text)
            combobox.blockSignals(False)
        combobox = self.bottomBar.trans_selector.tgt_selector
        if sender != combobox:
            combobox.blockSignals(True)
            combobox.setCurrentText(text)
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
        self.backup_blkstyles.clear()
        self._run_imgtrans_wo_textstyle_update = False
        self.postprocess_mt_toggle = True
        if pcfg.module.empty_runcache and not shared.HEADLESS:
            self.module_manager.unload_all_models()
        else:
            self._reset_idle_unload_timer()
        if shared.args.export_translation_txt:
            self.on_export_txt('translation')
        if shared.args.export_source_txt:
            self.on_export_txt('source')
        if shared.HEADLESS or getattr(self, 'exec_dirs', None) is not None:
            self.run_next_dir()

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

            self.st_manager.auto_textlayout_flag = pcfg.let_autolayout_flag and \
                (pcfg.module.enable_detect or pcfg.module.enable_translate)
        
        if page_index != self.pageList.currentIndex().row():
            self.pageList.setCurrentRow(page_index)
        else:
            self.imgtrans_proj.set_current_img_byidx(page_index)
            self.canvas.updateCanvas()
            self.st_manager.updateSceneTextitems()

        if not pcfg.module.enable_detect and pcfg.module.enable_translate:
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

    def on_run_imgtrans(self, continue_mode=False):
        self._idle_unload_timer.stop()
        self.backup_blkstyles.clear()

        if self.bottomBar.textblockChecker.isChecked():
            self.bottomBar.textblockChecker.click()
        self.postprocess_mt_toggle = False

        all_disabled = pcfg.module.all_stages_disabled()
        
        pages_to_process = []
        
        # 继续模式：先检查哪些页面需要处理
        if continue_mode:
            for page_name in self.imgtrans_proj.pages:
                if not self.imgtrans_proj.get_page_progress(page_name):
                    pages_to_process.append(page_name)
            if len(pages_to_process) == 0:
                return
        else:
            for page_name in self.imgtrans_proj.pages:
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

    def on_import_doc(self):
        self.import_doc_thread.importDoc(self.imgtrans_proj)

    def on_export_txt(self, dump_target, suffix='.txt'):
        try:
            self.imgtrans_proj.dump_txt(dump_target=dump_target, suffix=suffix)
            create_info_dialog(self.tr('Text file exported to ') + self.imgtrans_proj.dump_txt_path(dump_target, suffix))
        except Exception as e:
            create_error_dialog(e, self.tr('Failed to export as TEXT file'))

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
        self.resetStyleSheet()

    def on_config_darkmode_changed(self, checked: bool):
        self.titleBar.darkModeAction.setChecked(checked)
        self.resetStyleSheet()
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
        for blk in self.st_manager.textblk_item_list:
            blk.setSelected(True)
        self.st_manager.on_incanvas_selection_changed()

    def on_spell_check_src(self):
        """Run spell check on source text of selected blocks."""
        blks = self.canvas.selected_text_items()
        if not blks:
            return
        try:
            from utils.ocr_spellcheck import spell_check_line
        except Exception:
            QMessageBox.warning(
                self,
                self.tr("Spell check"),
                self.tr("Spell check requires pyenchant. Install it and a system dictionary (e.g. en_US)."),
            )
            return
        for blkitem in blks:
            blk = blkitem.blk
            if getattr(blk, "text", None) and isinstance(blk.text, list):
                blk.text = [spell_check_line(line) for line in blk.text]
                self.st_manager.pairwidget_list[blkitem.idx].e_source.setPlainText("\n".join(blk.text))
        if blks:
            self.canvas.setProjSaveState(False)

    def on_spell_check_trans(self):
        """Run spell check on translation text of selected blocks."""
        blks = self.canvas.selected_text_items()
        if not blks:
            return
        try:
            from utils.ocr_spellcheck import spell_check_line
        except Exception:
            QMessageBox.warning(
                self,
                self.tr("Spell check"),
                self.tr("Spell check requires pyenchant. Install it and a system dictionary (e.g. en_US)."),
            )
            return
        for blkitem in blks:
            blk = blkitem.blk
            trans = (getattr(blk, "translation", None) or "") or self.st_manager.pairwidget_list[blkitem.idx].e_trans.toPlainText()
            lines = trans.split("\n")
            corrected = [spell_check_line(line) for line in lines]
            new_trans = "\n".join(corrected)
            blk.translation = new_trans
            self.st_manager.pairwidget_list[blkitem.idx].e_trans.setPlainText(new_trans)
        if blks:
            self.canvas.setProjSaveState(False)

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

    def on_run_detect_region(self, rect):
        if not self.imgtrans_proj.img_valid or self.imgtrans_proj.img_array is None:
            return
        page_name = self.imgtrans_proj.current_img
        if not page_name:
            return
        self.module_manager.run_detect_region(rect, self.imgtrans_proj.img_array, page_name)

    def on_detect_region_finished(self, page_name: str, blk_list: List):
        if page_name != self.imgtrans_proj.current_img or not blk_list:
            return
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
    
    def run_batch(self, exec_dirs: Union[List, str], **kwargs):
        if not isinstance(exec_dirs, List):
            exec_dirs = [x.strip() for x in exec_dirs.split(',') if x.strip()]
        valid_dirs = []
        for d in exec_dirs:
            if osp.exists(d):
                valid_dirs.append(d)
            else:
                LOGGER.warning(f'target directory {d} does not exist.')
        self.exec_dirs = valid_dirs
        self.run_next_dir()

    def run_next_dir(self):
        if len(self.exec_dirs) == 0:
            while self.imsave_thread.isRunning():
                time.sleep(0.1)
            LOGGER.info('finished translating all dirs.')
            if shared.HEADLESS:
                self.app.quit()
            return
        d = self.exec_dirs.pop(0)
        
        LOGGER.info(f'translating {d} ...')
        self.openDir(d)
        shared.pbar = {}
        npages = len(self.imgtrans_proj.pages)
        if npages > 0:
            if pcfg.module.enable_detect:
                shared.pbar['detect'] = tqdm(range(npages), desc="Text Detection")
            if pcfg.module.enable_ocr:
                shared.pbar['ocr'] = tqdm(range(npages), desc="OCR")
            if pcfg.module.enable_translate:
                shared.pbar['translate'] = tqdm(range(npages), desc="Translation")
            if pcfg.module.enable_inpaint:
                shared.pbar['inpaint'] = tqdm(range(npages), desc="Inpaint")
        self.on_run_imgtrans()

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
        except:
            if exception_type in shared.showed_exception:
                shared.showed_exception.remove(exception_type)
            LOGGER.error('Failed to create error dialog')
            LOGGER.error(traceback.format_exc())

    def on_create_infodialog(self, info_dict: dict):
        QMessageBox.StandardButton.NoButton
        dialog = MessageBox(**info_dict)
        dialog.show()   # exec_ will block main thread

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

    def register_view_widget(self, widget: ViewWidget):
        assert widget.config_name not in shared.config_name_to_view_widget
        d = {'widget': widget}
        shared.config_name_to_view_widget[widget.config_name] = d

    def action_set_view_visible(self):
        action: QAction = self.sender()
        show = action.isChecked()
        cfg_name = shared.action_to_view_config_name[action]
        widget: ViewWidget = shared.config_name_to_view_widget[cfg_name]['widget']
        widget.setVisible(show)
        setattr(pcfg, cfg_name, show)

    def on_hide_view_widget(self, cfg_name: str):
        d = shared.config_name_to_view_widget[cfg_name]
        widget: ViewWidget = d['widget']
        widget.setVisible(False)
        action: QAction = d['action']
        action.setChecked(False)
        setattr(pcfg, cfg_name, False)