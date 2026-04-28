from typing import List, Union, Tuple

from qtpy.QtCore import Qt, Signal, QSize, QEvent, QItemSelection
from qtpy.QtGui import QStandardItem, QStandardItemModel, QMouseEvent, QFont, QIntValidator, QValidator, QFocusEvent, QColor
from qtpy.QtWidgets import (QPushButton, QKeySequenceEdit, QLayout, QGridLayout, QHBoxLayout, QVBoxLayout,
    QTreeView, QWidget, QLabel, QSizePolicy, QSpacerItem, QCheckBox, QSplitter, QScrollArea, QLineEdit,
    QSpinBox, QComboBox, QDoubleSpinBox, QColorDialog, QMessageBox, QFileDialog)

from .custom_widget import ConfigComboBox, Widget, DangoSwitch
from .custom_widget.smooth_scroll import SmoothScrollArea
from .custom_widget.focus_ring import FocusRingFrame
from .context_menu_config_dialog import ContextMenuConfigDialog
from utils.config import pcfg
from utils import shared as C
from utils.shared import CONFIG_FONTSIZE_CONTENT, CONFIG_FONTSIZE_HEADER, CONFIG_FONTSIZE_TABLE, CONFIG_COMBOBOX_SHORT, CONFIG_COMBOBOX_LONG, CONFIG_COMBOBOX_MIDEAN, DISPLAY_LANGUAGE_MAP
from .module_parse_widgets import InpaintConfigPanel, TextDetectConfigPanel, TranslatorConfigPanel, OCRConfigPanel


def _config_font_size(base_size: float) -> float:
    """Apply config panel font scale (accessibility)."""
    scale = getattr(pcfg, 'config_panel_font_scale', 1.0)
    if scale <= 0 or scale > 5:
        scale = 1.0
    return max(1.0, base_size * scale)

class CustomIntValidator(QIntValidator):

    def __init__(self, bottom: int, top: int, ndigits: int = None, parent = None):
        super().__init__(bottom=bottom, top=top, parent=parent)
        self.ndigits = ndigits

    def validate(self, s: str, pos: int) -> object:
        if not s.isnumeric():
            if s != '':
                return (QValidator.State.Invalid, s, pos)
            else:
                return (QValidator.State.Intermediate, s, pos)
            
        s_ori = s
        d = int(s)
        s = str(d)
        if len(s) != len(s_ori):
            pos -= len(s_ori) - len(s)
        if len(s) > self.ndigits:
            ndel = len(s) - self.ndigits
            s = s[ndel:]
            pos -= ndel
        else:
            if d > self.top():
                if s[-1] == '0':
                    d = self.top()
                else:
                    d = d % self.top()
            d = max(d, self.bottom())
            s = str(d)
        return (QValidator.State.Acceptable, s, pos)


class PercentageLineEdit(QLineEdit):

    finish_edited = Signal(str)

    def __init__(self, default_value: str = '100', parent=None) -> None:
        super().__init__(default_value, parent=parent)
        validator = CustomIntValidator(0, 101, 3)
        self.setValidator(validator)
        self.textEdited.connect(self.on_text_edited)
        self._edited = False

    def on_text_edited(self):
        self._edited = True

    def focusOutEvent(self, e: QFocusEvent) -> None:
        if self._edited:
            text = self.text()
            if not text.isnumeric():
                text = '100'
                self.setText(text)
            self.finish_edited.emit(text)

        return super().focusOutEvent(e)


class ConfigTextLabel(QLabel):
    def __init__(self, text: str, fontsize: int, font_weight: int = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setText(text)
        font = self.font()
        if font_weight is not None:
            font.setWeight(font_weight)
        size = max(1, fontsize) if fontsize <= 0 else fontsize
        font.setPointSizeF(float(size))
        font.setPointSize(max(1, int(size)))  # avoid QFont::setPointSize <= 0 (Windows default -1)
        self.setFont(font)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        self.setOpenExternalLinks(True)

    def setActiveBackground(self):
        self.setStyleSheet("background-color:rgba(30, 147, 229, 51);")


class ConfigSubBlock(Widget):
    pressed = Signal(int, int)
    def __init__(self, widget: Union[QWidget, QLayout], name: str = None, discription: str = None, vertical_layout=True, insert_stretch: bool = False, content_margins = (24, 6, 24, 6)) -> None:
        super().__init__()
        self.idx0: int = None
        self.idx1: int = None
        if vertical_layout:
            layout = QVBoxLayout(self)
        else:
            layout = QHBoxLayout(self)
        self.name = name
        if name is not None:
            textlabel = ConfigTextLabel(name, _config_font_size(CONFIG_FONTSIZE_CONTENT), QFont.Weight.Normal)
            self.name_label = textlabel
            layout.addWidget(textlabel)
        if discription is not None:
            layout.addWidget(ConfigTextLabel(discription, _config_font_size(CONFIG_FONTSIZE_CONTENT)-2))
        if insert_stretch:
            layout.insertStretch(-1)
        if isinstance(widget, QWidget):
            layout.addWidget(widget)
        else:
            layout.addLayout(widget)
        self.widget = widget
        self.setContentsMargins(*content_margins)

    def setIdx(self, idx0: int, idx1: int) -> None:
        self.idx0 = idx0
        self.idx1 = idx1

    def enterEvent(self, e: QEvent) -> None:
        self.pressed.emit(self.idx0, self.idx1)
        return super().enterEvent(e)
    

def combobox_with_label(sel: List[str], name: str, discription: str = None, vertical_layout: bool = False, target_block: QWidget = None, fix_size: bool = True, parent: QWidget = None, insert_stretch: bool = False) -> Tuple[ConfigComboBox, QWidget]:
    combox = ConfigComboBox(fix_size=fix_size, scrollWidget=parent)
    combox.addItems(sel)
    if target_block is None:
        sublock = ConfigSubBlock(combox, name, discription, vertical_layout=vertical_layout, insert_stretch=insert_stretch)
        sublock.layout().setAlignment(Qt.AlignmentFlag.AlignLeft)
        sublock.layout().setSpacing(20)
        return combox, sublock
    else:
        layout = target_block.layout()
        layout.addSpacing(20)
        layout.addWidget(ConfigTextLabel(name, _config_font_size(CONFIG_FONTSIZE_CONTENT), QFont.Weight.Normal))
        layout.addWidget(combox)
        return combox, target_block
    
def checkbox_with_label(name: str, discription: str = None, target_block: QWidget = None):
    checkbox = QCheckBox()
    if discription is not None:
        font = checkbox.font()
        size = max(1.0, _config_font_size(CONFIG_FONTSIZE_CONTENT) * 0.8)
        font.setPointSizeF(size)
        font.setPointSize(max(1, int(size)))  # avoid QFont::setPointSize <= 0
        checkbox.setFont(font)
        checkbox.setText(discription)
        vertical_layout = True
    else:
        vertical_layout = False
        from .custom_widget.widget import _sanitize_font
        checkbox.setFont(_sanitize_font(checkbox.font()))

    if target_block is None:
        sublock = ConfigSubBlock(checkbox, name, vertical_layout=vertical_layout)
        if vertical_layout is False:
            sublock.layout().addItem(QSpacerItem(0, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding))
        target_block = sublock
    return checkbox, target_block
    


class ConfigBlock(Widget):
    sublock_pressed = Signal(int, int)

    def __init__(self, header: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.header = ConfigTextLabel(header, _config_font_size(CONFIG_FONTSIZE_HEADER))
        self.vlayout = QVBoxLayout(self)
        self.vlayout.addWidget(self.header)
        self.setContentsMargins(24, 24, 24, 24)
        self.label_list = []
        self.subblock_list = []
        self.index: int = 0

    def setIndex(self, index: int):
        self.index = index

    def addLineEdit(self, name: str = None, discription: str = None, vertical_layout: bool = False):
        le = QLineEdit()
        le.setFixedWidth(CONFIG_COMBOBOX_MIDEAN)
        le.setFixedHeight(45)
        sublock = ConfigSubBlock(le, name, discription, vertical_layout)
        if vertical_layout is False:
            sublock.layout().addItem(QSpacerItem(0, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding))
        self.addSublock(sublock)
        sublock.layout().setSpacing(20)
        return le, sublock

    def addTextLabel(self, text: str = None):
        label = ConfigTextLabel(text, _config_font_size(CONFIG_FONTSIZE_HEADER))
        self.vlayout.addWidget(label)
        self.label_list.append(label)

    def addSectionDescription(self, text: str):
        """Short section-level description (Phase 2.3 UI plan)."""
        desc = ConfigTextLabel(text, max(1, _config_font_size(CONFIG_FONTSIZE_CONTENT) - 2))
        desc.setStyleSheet("color: gray;")
        self.vlayout.addWidget(desc)

    def addSublock(self, sublock: ConfigSubBlock):
        self.vlayout.addWidget(sublock)
        sublock.setIdx(self.index, len(self.label_list)-1)
        sublock.pressed.connect(lambda idx0, idx1: self.sublock_pressed.emit(idx0, idx1))
        self.subblock_list.append(sublock)

    def addCombobox(self, sel: List[str], name: str, discription: str = None, vertical_layout: bool = False, target_block: QWidget = None, fix_size: bool = True) -> Tuple[ConfigComboBox, QWidget]:
        combox, sublock = combobox_with_label(sel, name, discription, vertical_layout, target_block, fix_size, parent=self)
        if target_block is None:
            self.addSublock(sublock)
        return combox, sublock

    def addBlockWidget(self, widget: Union[QWidget, QLayout], name: str = None, discription: str = None, vertical_layout: bool = False) -> ConfigSubBlock:
        sublock = ConfigSubBlock(widget, name, discription, vertical_layout)
        self.addSublock(sublock)
        return sublock

    def addCheckBox(self, name: str, discription: str = None, target_block: ConfigSubBlock = None) -> QCheckBox:
        checkbox, sublock = checkbox_with_label(name, discription, target_block)
        if target_block is None:
            self.addSublock(sublock)
        return checkbox, sublock

    def getSubBlockbyIdx(self, idx: int) -> ConfigSubBlock:
        return self.subblock_list[idx]


class ConfigContent(SmoothScrollArea):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setSmoothScrollDuration(getattr(pcfg, 'smooth_scroll_duration_ms', 0))
        self.setMotionBlurOnScroll(getattr(pcfg, 'motion_blur_on_scroll', False))
        self.config_block_list: List[ConfigBlock] = []
        self.scrollContent = Widget()
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.setWidget(self.scrollContent)
        vlayout = QVBoxLayout()
        vlayout.setContentsMargins(0, 0, 0, 0)
        vlayout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.scrollContent.setLayout(vlayout)
        self.setWidgetResizable(True)
        self.setContentsMargins(0, 0, 0, 0)
        self.vlayout = vlayout
        self.active_label: ConfigTextLabel = None

    def addConfigBlock(self, block: ConfigBlock):
        self.vlayout.addWidget(block)
        self.config_block_list.append(block)

    def setActiveLabel(self, idx0: int, idx1: int):
        if self.active_label is not None:
            self.deactiveLabel()
        block = self.config_block_list[idx0]
        if idx1 >= 0:
            self.active_label = block.label_list[idx1]
        else:
            self.active_label = block.header
        self.active_label.setActiveBackground()
        if C.USE_PYSIDE6:
            self.ensureWidgetVisible(self.active_label, ymargin=self.active_label.height() * 7)
        else:
            self.ensureWidgetVisible(self.active_label, yMargin=self.active_label.height() * 7)

    def deactiveLabel(self):
        if self.active_label is not None:
            self.active_label.setStyleSheet("")
            self.active_label = None


class TableItem(QStandardItem):
    def __init__(self, text, fontsize):
        super().__init__()
        font = self.font()
        size = max(1, fontsize) if fontsize <= 0 else fontsize
        font.setPointSizeF(float(size))
        font.setPointSize(max(1, int(size)))  # avoid QFont::setPointSize <= 0
        self.setFont(font)
        self.setText(text)
        self.setEditable(False)

    def setBold(self, bold: bool):
        font = self.font()
        if font.pointSize() <= 0 or font.pointSizeF() <= 0:
            font.setPointSizeF(10.0)
            font.setPointSize(10)
        font.setBold(bold)
        self.setFont(font)


class TreeModel(QStandardItemModel):
    # https://stackoverflow.com/questions/32229314/pyqt-how-can-i-set-row-heights-of-qtreeview
    def data(self, index, role):
        if not index.isValid():
            return None
        if role == Qt.ItemDataRole.SizeHintRole:
            size = QSize()
            item = self.itemFromIndex(index)
            size.setHeight(max(1, item.font().pointSize()) + 20)
            return size
        else:
            return super().data(index, role)


class ConfigTable(QTreeView):
    tableitem_pressed = Signal(int, int)
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        treeModel = TreeModel()
        self.tm = treeModel
        self.setModel(treeModel)
        self.selected: TableItem = None
        self.last_selected: TableItem = None
        self.setHeaderHidden(True)
        self.setMinimumWidth(260)

    def showEvent(self, event):
        # Avoid QFont::setPointSize <= 0 (-1) from default font on Windows
        from .custom_widget.widget import _sanitize_font
        f = self.font()
        if f.pointSizeF() <= 0 or f.pointSize() <= 0:
            self.setFont(_sanitize_font(f))
        super().showEvent(event)

    def addHeader(self, header: str) -> TableItem:
        rootNode = self.model().invisibleRootItem()
        ti = TableItem(header, _config_font_size(CONFIG_FONTSIZE_TABLE))
        rootNode.appendRow(ti)
        return ti

    def selectionChanged(self, selected: QItemSelection, deselected: QItemSelection) -> None:
        dis = deselected.indexes()
        sel = selected.indexes()
        model = self.model()
        self.last_selected = model.itemFromIndex(dis[0]) \
            if len(dis) > 0 else None
        
        self.selected = model.itemFromIndex(sel[0]) \
            if len(sel) > 0 else None
        for i in deselected.indexes():
            self.model().itemFromIndex(i).setBold(False)
        
        index = self.currentIndex()
        if index.isValid():
            self.model().itemFromIndex(index).setBold(True)
        super().selectionChanged(selected, deselected)

    def setCurrentItem(self, idx0, idx1):
        parent = self.tm.item(idx0, 0)
        if parent is None:
            return
        child = parent.child(idx1)
        if child is None:
            return
        index = child.index()
        self.setCurrentIndex(index)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        super().mousePressEvent(event)
        if self.selected is not None:
            parent = self.selected.parent()
            if parent is None:
                idx1 = -1
                idx0 = self.selected.row()
            else:
                idx1 = self.selected.row()
                idx0 = parent.row()
            self.tableitem_pressed.emit(idx0, idx1)


class ConfigPanel(Widget):

    save_config = Signal()
    unload_models = Signal()
    reload_textstyle = Signal(bool)
    show_only_custom_font = Signal(bool)
    darkmode_changed = Signal(bool)
    bubbly_ui_changed = Signal(bool)
    custom_cursor_changed = Signal()
    display_lang_changed = Signal(str)
    dev_mode_changed = Signal()
    manual_mode_changed = Signal()

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setObjectName("ConfigPanel")
        self.configTable = ConfigTable()
        self.configTable.tableitem_pressed.connect(self.onTableItemPressed)
        self.configTreeFocusRing = FocusRingFrame()
        self.configTreeFocusRing.setChild(self.configTable)
        self.configContent = ConfigContent()
        # Ensure content has enough width so Typesetting combos and Test translator/OCR buttons don't get cut off
        self.configContent.setMinimumWidth(420)
        w = self.configContent.widget()
        if w is not None:
            w.setMinimumWidth(520)
        dlConfigPanel, dltableitem = self.addConfigBlock(self.tr('DL Module'))
        generalConfigPanel, generalTableItem = self.addConfigBlock(self.tr('General'))
        
        label_text_det = self.tr('Text Detection')
        label_text_ocr = self.tr('OCR')
        label_inpaint = self.tr('Inpaint')
        label_translator = self.tr('Translator')
        label_startup = self.tr('Startup')
        label_display = self.tr('Display')
        label_typesetting = self.tr('Typesetting')
        label_ocr_result = self.tr('OCR result')
        label_save = self.tr('Save')
        label_saladict = self.tr('SalaDict')
        label_canvas = self.tr('Canvas')
        label_integrations = self.tr('Integrations')
    
        dltableitem.appendRows([
            TableItem(self.tr('Image upscaling'), _config_font_size(CONFIG_FONTSIZE_TABLE)),
            TableItem(label_text_det, _config_font_size(CONFIG_FONTSIZE_TABLE)),
            TableItem(label_text_ocr, _config_font_size(CONFIG_FONTSIZE_TABLE)),
            TableItem(label_inpaint, _config_font_size(CONFIG_FONTSIZE_TABLE)),
            TableItem(label_translator, _config_font_size(CONFIG_FONTSIZE_TABLE)),
        ])
        generalTableItem.appendRows([
            TableItem(label_startup, _config_font_size(CONFIG_FONTSIZE_TABLE)),
            TableItem(label_display, _config_font_size(CONFIG_FONTSIZE_TABLE)),
            TableItem(label_typesetting, _config_font_size(CONFIG_FONTSIZE_TABLE)),
            TableItem(label_ocr_result, _config_font_size(CONFIG_FONTSIZE_TABLE)),
            TableItem(label_save, _config_font_size(CONFIG_FONTSIZE_TABLE)),
            TableItem(label_canvas, _config_font_size(CONFIG_FONTSIZE_TABLE)),
            TableItem(label_integrations, _config_font_size(CONFIG_FONTSIZE_TABLE)),
        ])
        
        self.load_model_checker, msublock = checkbox_with_label(self.tr('Load models on demand'), discription=self.tr('Load models on demand to save memory.'))
        self.load_model_checker.stateChanged.connect(self.on_load_model_changed)
        dlConfigPanel.vlayout.addWidget(msublock)
        self.default_device_combobox = QComboBox()
        try:
            from modules.base import AVAILABLE_DEVICES
            self.default_device_combobox.addItem(self.tr('Default (use module default)'))
            self.default_device_combobox.addItems(AVAILABLE_DEVICES)
        except Exception:
            self.default_device_combobox.addItems([self.tr('Default (use module default)'), 'cpu', 'cuda'])
        self.default_device_combobox.currentIndexChanged.connect(self.on_default_device_index_changed)
        sublock = ConfigSubBlock(self.default_device_combobox, self.tr('Default device'), discription=self.tr('Preferred device for DL modules when set to Default.'))
        dlConfigPanel.vlayout.addWidget(sublock)
        self.empty_runcache_checker, msublock = checkbox_with_label(self.tr('Empty cache after RUN'), discription=self.tr('Empty cache after RUN to save memory.'))
        dlConfigPanel.vlayout.addWidget(msublock)
        self.empty_runcache_checker.stateChanged.connect(self.on_runcache_changed)
        self.release_caches_after_batch_checker, rcb_sublock = checkbox_with_label(
            self.tr('Release model caches after batch'),
            discription=self.tr('After pipeline finishes (all pages), unload models and clear caches to free RAM. Like "Empty cache after RUN" but only after a full run.')
        )
        dlConfigPanel.vlayout.addWidget(rcb_sublock)
        self.release_caches_after_batch_checker.stateChanged.connect(self.on_release_caches_after_batch_changed)
        self.skip_already_translated_checker, skip_sublock = checkbox_with_label(
            self.tr('Skip already translated'),
            discription=self.tr('Do not call translator for pages where every block already has a translation.')
        )
        dlConfigPanel.vlayout.addWidget(skip_sublock)
        self.skip_already_translated_checker.stateChanged.connect(self._on_skip_already_translated_changed)
        self.ocr_cache_enabled_checker, ocr_cache_sublock = checkbox_with_label(
            self.tr('Enable OCR cache'),
            discription=self.tr('Reuse OCR results for the same image/model/language in this session. Reduces redundant OCR runs.')
        )
        dlConfigPanel.vlayout.addWidget(ocr_cache_sublock)
        self.ocr_cache_enabled_checker.stateChanged.connect(self._on_ocr_cache_enabled_changed)
        self.ocr_auto_by_language_checker, ocr_auto_sublock = checkbox_with_label(
            self.tr('Auto OCR by source language'),
            discription=self.tr('Select OCR module by source language (e.g. Japanese → manga_ocr). Requires language mapping in pipeline.')
        )
        dlConfigPanel.vlayout.addWidget(ocr_auto_sublock)
        self.ocr_auto_by_language_checker.stateChanged.connect(self._on_ocr_auto_by_language_changed)
        self.show_module_tier_badges_checker, tier_sublock = checkbox_with_label(
            self.tr('Show module tier badge in selector tooltip'),
            discription=self.tr('Show Stable/Beta/Experimental/External-heavy badge in detector/OCR/inpainter/translator dropdown tooltips.')
        )
        dlConfigPanel.vlayout.addWidget(tier_sublock)
        self.show_module_tier_badges_checker.stateChanged.connect(self._on_show_module_tier_badges_changed)
        self.merge_nearby_blocks_checker, merge_sublock = checkbox_with_label(
            self.tr('Merge nearby blocks (collision)'),
            discription=self.tr('Merge detection boxes that are close (collision-based). Use when OCR returns many small boxes.')
        )
        dlConfigPanel.vlayout.addWidget(merge_sublock)
        self.merge_nearby_blocks_checker.stateChanged.connect(self._on_merge_nearby_blocks_changed)
        self.merge_nearby_blocks_gap_spin = QDoubleSpinBox()
        self.merge_nearby_blocks_gap_spin.setRange(0.3, 3.0)
        self.merge_nearby_blocks_gap_spin.setSingleStep(0.1)
        self.merge_nearby_blocks_gap_spin.setValue(float(getattr(pcfg.module, 'merge_nearby_blocks_gap_ratio', 1.5)))
        self.merge_nearby_blocks_gap_spin.valueChanged.connect(self._on_merge_nearby_blocks_gap_changed)
        merge_gap_sublock = ConfigSubBlock(
            self.merge_nearby_blocks_gap_spin,
            self.tr('Merge gap ratio'),
            discription=self.tr('Vertical expansion ratio for horizontal merge (e.g. 1.5 for typical merge).')
        )
        dlConfigPanel.vlayout.addWidget(merge_gap_sublock)
        self.merge_nearby_blocks_min_blocks_spin = QSpinBox()
        self.merge_nearby_blocks_min_blocks_spin.setRange(5, 60)
        self.merge_nearby_blocks_min_blocks_spin.setValue(int(getattr(pcfg.module, 'merge_nearby_blocks_min_blocks', 18)))
        self.merge_nearby_blocks_min_blocks_spin.valueChanged.connect(self._on_merge_nearby_blocks_min_blocks_changed)
        merge_min_sublock = ConfigSubBlock(
            self.merge_nearby_blocks_min_blocks_spin,
            self.tr('Min blocks to merge'),
            discription=self.tr('Only run merge when page has at least this many blocks (avoids merging normal bubble layouts; default 18).')
        )
        dlConfigPanel.vlayout.addWidget(merge_min_sublock)
        self.unload_model_btn = QPushButton(parent=self)
        self.unload_model_btn.setFixedWidth(500)
        self.unload_model_btn.setText(self.tr('Unload All Models'))
        self.unload_model_btn.clicked.connect(self.unload_models)
        msublock.layout().addWidget(self.unload_model_btn)
        self.check_download_models_btn = QPushButton(parent=self)
        self.check_download_models_btn.setFixedWidth(500)
        self.check_download_models_btn.setText(self.tr('Check / Download module files'))
        self.check_download_models_btn.setToolTip(self.tr('Check and download required model files for all detectors, OCR, inpainters, and translators. Run this if a module fails to load.'))
        self.check_download_models_btn.clicked.connect(self.on_check_download_module_files)
        msublock.layout().addWidget(self.check_download_models_btn)
        self.unload_after_idle_spin = QSpinBox()
        self.unload_after_idle_spin.setRange(0, 120)
        self.unload_after_idle_spin.setValue(getattr(pcfg, 'unload_after_idle_minutes', 0))
        self.unload_after_idle_spin.setSpecialValueText(self.tr('Off'))
        self.unload_after_idle_spin.valueChanged.connect(self.on_unload_after_idle_changed)
        sublock_idle = ConfigSubBlock(self.unload_after_idle_spin, self.tr('Unload models after idle (minutes)'), discription=self.tr('0 = disabled. Unload DL models after N minutes of no Run or canvas activity.'))
        dlConfigPanel.vlayout.addWidget(sublock_idle)

        dlConfigPanel.addTextLabel(self.tr('Image upscaling (Section 6)'))
        self.image_upscale_initial_checker, _ = checkbox_with_label(
            self.tr('Initial upscale before detection/OCR'),
            discription=self.tr('Upscale page before pipeline (improves small text). Results are scaled back to original size.')
        )
        self.image_upscale_initial_checker.stateChanged.connect(self._on_image_upscale_initial_changed)
        dlConfigPanel.vlayout.addWidget(_)
        self.image_upscale_initial_factor_spin = QDoubleSpinBox()
        self.image_upscale_initial_factor_spin.setRange(1.5, 4.0)
        self.image_upscale_initial_factor_spin.setSingleStep(0.5)
        self.image_upscale_initial_factor_spin.setValue(getattr(pcfg.module, 'image_upscale_initial_factor', 2.0))
        self.image_upscale_initial_factor_spin.valueChanged.connect(self._on_image_upscale_initial_factor_changed)
        sublock_ini = ConfigSubBlock(self.image_upscale_initial_factor_spin, self.tr('Initial upscale factor'), discription=self.tr('e.g. 2 = 2x before detection/OCR.'))
        dlConfigPanel.vlayout.addWidget(sublock_ini)
        self.image_upscale_final_checker, _ = checkbox_with_label(
            self.tr('Final upscale when saving result'),
            discription=self.tr('Upscale result image when saving (e.g. 2x for nicer export).')
        )
        self.image_upscale_final_checker.stateChanged.connect(self._on_image_upscale_final_changed)
        dlConfigPanel.vlayout.addWidget(_)
        self.image_upscale_final_factor_spin = QDoubleSpinBox()
        self.image_upscale_final_factor_spin.setRange(1.5, 4.0)
        self.image_upscale_final_factor_spin.setSingleStep(0.5)
        self.image_upscale_final_factor_spin.setValue(getattr(pcfg.module, 'image_upscale_final_factor', 2.0))
        self.image_upscale_final_factor_spin.valueChanged.connect(self._on_image_upscale_final_factor_changed)
        sublock_fin = ConfigSubBlock(self.image_upscale_final_factor_spin, self.tr('Final upscale factor'), discription=self.tr('e.g. 2 = 2x when saving result.'))
        dlConfigPanel.vlayout.addWidget(sublock_fin)
        self.processing_scale_checker, _ = checkbox_with_label(
            self.tr('Auto-scale pipeline params by image area'),
            discription=self.tr('Scale padding/gaps by image size so settings behave consistently across resolutions.')
        )
        self.processing_scale_checker.stateChanged.connect(self._on_processing_scale_changed)
        dlConfigPanel.vlayout.addWidget(_)
        # Optional: lightweight colorization when saving final result (experimental).
        self.colorization_checker, _ = checkbox_with_label(
            self.tr('Colorize final result (experimental)'),
            discription=self.tr('Apply a light colorization to grayscale pages when saving the final result. Does not affect original or inpainted layers.')
        )
        self.colorization_checker.stateChanged.connect(self._on_colorization_changed)
        dlConfigPanel.vlayout.addWidget(_)
        # Backend selection for colorization (different “models” / palettes).
        self.colorization_backend_combobox = QComboBox()
        self.colorization_backend_combobox.addItems([
            self.tr('Soft (Twilight, manga-friendly)'),
            self.tr('Vibrant (Magma)'),
            self.tr('Cool (Ocean)'),
        ])
        sublock_color_backend = ConfigSubBlock(
            self.colorization_backend_combobox,
            self.tr('Colorization style'),
            discription=self.tr('Choose the colorization palette: softer vs more vibrant or cool-toned.')
        )
        self.colorization_backend_combobox.currentIndexChanged.connect(self._on_colorization_backend_changed)
        dlConfigPanel.vlayout.addWidget(sublock_color_backend)
        self.colorization_strength_spin = QDoubleSpinBox()
        self.colorization_strength_spin.setRange(0.1, 1.0)
        self.colorization_strength_spin.setSingleStep(0.1)
        self.colorization_strength_spin.setDecimals(2)
        self.colorization_strength_spin.setValue(float(getattr(pcfg.module, 'colorization_strength', 0.6)))
        self.colorization_strength_spin.valueChanged.connect(self._on_colorization_strength_changed)
        sublock_color = ConfigSubBlock(
            self.colorization_strength_spin,
            self.tr('Colorization strength'),
            discription=self.tr('Blend between grayscale (0) and fully colorized (1). Recommended 0.5–0.8.')
        )
        dlConfigPanel.vlayout.addWidget(sublock_color)
        self.ocr_upscale_min_side_spin = QSpinBox()
        self.ocr_upscale_min_side_spin.setRange(0, 2048)
        self.ocr_upscale_min_side_spin.setSpecialValueText(self.tr('Off'))
        self.ocr_upscale_min_side_spin.setValue(getattr(pcfg.module, 'ocr_upscale_min_side', 0))
        self.ocr_upscale_min_side_spin.valueChanged.connect(self._on_ocr_upscale_min_side_changed)
        sublock_ocr_upscale = ConfigSubBlock(self.ocr_upscale_min_side_spin, self.tr('OCR crop upscale min side (global)'), discription=self.tr('If >0, upscale small crops so longer side >= this before OCR (e.g. 512). Per-OCR params override when set.'))
        dlConfigPanel.vlayout.addWidget(sublock_ocr_upscale)
        self.inpaint_spill_after_spin = QSpinBox()
        self.inpaint_spill_after_spin.setRange(0, 64)
        self.inpaint_spill_after_spin.setSpecialValueText(self.tr('Off'))
        self.inpaint_spill_after_spin.setValue(getattr(pcfg.module, 'inpaint_spill_to_disk_after_blocks', 0))
        self.inpaint_spill_after_spin.valueChanged.connect(self._on_inpaint_spill_after_changed)
        sublock_spill = ConfigSubBlock(self.inpaint_spill_after_spin, self.tr('Inpaint: spill to disk after N blocks'), discription=self.tr('When >0, write intermediate result to temp file every N blocks to reduce peak RAM/VRAM on long pages (e.g. 8 or 12).'))
        dlConfigPanel.vlayout.addWidget(sublock_spill)

        dlConfigPanel.addTextLabel(label_text_det)
        self.detect_config_panel = TextDetectConfigPanel(self.tr('Detector'), scrollWidget=self)
        self.detect_sub_block = dlConfigPanel.addBlockWidget(self.detect_config_panel)
        self.detect_config_panel.keep_existing_checker.clicked.connect(self.on_keepline_clicked)

        dlConfigPanel.addTextLabel(label_text_ocr)
        self.ocr_config_panel = OCRConfigPanel(self.tr('OCR'), scrollWidget=self)
        self.ocr_sub_block = dlConfigPanel.addBlockWidget(self.ocr_config_panel)

        dlConfigPanel.addTextLabel(label_inpaint)
        self.inpaint_config_panel = InpaintConfigPanel(self.tr('Inpainter'), scrollWidget=self)
        self.inpaint_sub_block = dlConfigPanel.addBlockWidget(self.inpaint_config_panel)

        dlConfigPanel.addTextLabel(label_translator)
        self.trans_config_panel = TranslatorConfigPanel(label_translator, scrollWidget=self)
        self.trans_sub_block = dlConfigPanel.addBlockWidget(self.trans_config_panel)

        generalConfigPanel.addTextLabel(label_startup)
        generalConfigPanel.addSectionDescription(self.tr("Reopen last project, confirm before run, recent list limit."))
        self.open_on_startup_checker, _ = generalConfigPanel.addCheckBox(self.tr('Reopen last project on startup'))
        self.open_on_startup_checker.stateChanged.connect(self.on_open_onstartup_changed)

        self.show_welcome_screen_checker, _ = generalConfigPanel.addCheckBox(
            self.tr('Show welcome screen when no project is open'),
            discription=self.tr('On startup, when no project is opened, show the welcome screen to open or create a project (manhua-translator / Komakun style).')
        )
        self.show_welcome_screen_checker.stateChanged.connect(self.on_show_welcome_screen_changed)

        self.auto_update_from_github_checker, _ = generalConfigPanel.addCheckBox(
            self.tr('Auto update from GitHub on startup'),
            discription=self.tr('Check for updates and pull from GitHub when the app starts. Can cause issues or bad results (e.g. merge conflicts, broken code, unexpected behavior). Use only if you understand the risks. Your config and local files are not overwritten.')
        )
        self.auto_update_from_github_checker.stateChanged.connect(self.on_auto_update_from_github_changed)

        self.dev_mode_checker, _ = generalConfigPanel.addCheckBox(
            self.tr('Dev mode (show all modules)'),
            discription=self.tr(
                'For testing and debugging. When enabled: (1) Detector / OCR / translator dropdowns show all modules, including not-yet-downloaded or incompatible. '
                '(2) Console/py.exe window shows more log output (DEBUG and third-party INFO). When disabled: only ready-to-use modules are listed and console shows INFO and above.'
            )
        )
        self.dev_mode_checker.stateChanged.connect(self.on_dev_mode_changed)

        self.recent_proj_list_max_spin = QSpinBox()
        self.recent_proj_list_max_spin.setRange(5, 30)
        self.recent_proj_list_max_spin.setValue(14)
        sublock = ConfigSubBlock(self.recent_proj_list_max_spin, self.tr('Recent projects limit (5–30)'), discription=self.tr('Maximum number of recent projects in the list.'))
        generalConfigPanel.addSublock(sublock)
        self.recent_proj_list_max_spin.valueChanged.connect(self.on_recent_proj_list_max_changed)

        self.confirm_before_run_checker, _ = generalConfigPanel.addCheckBox(self.tr('Confirm before Run'), discription=self.tr('Show Run / Continue / Cancel dialog when clicking Run.'))
        self.confirm_before_run_checker.stateChanged.connect(self.on_confirm_before_run_changed)

        self.manual_mode_checker, _ = generalConfigPanel.addCheckBox(
            self.tr('Manual mode'),
            discription=self.tr('When on, Run processes the current page only (comic-translate style). Useful for step-by-step workflow.')
        )
        self.manual_mode_checker.stateChanged.connect(self.on_manual_mode_changed)

        self.skip_ignored_in_run_checker, _ = generalConfigPanel.addCheckBox(
            self.tr('Skip ignored pages in run'),
            discription=self.tr('When on, full run and batch queue skip pages marked as "Ignore in run" in the page list context menu.')
        )
        self.skip_ignored_in_run_checker.stateChanged.connect(self.on_skip_ignored_in_run_changed)

        self.auto_region_merge_combobox = QComboBox()
        self.auto_region_merge_combobox.addItem(self.tr('Never'), 'never')
        self.auto_region_merge_combobox.addItem(self.tr('After run: on all pages'), 'all_pages')
        self.auto_region_merge_combobox.addItem(self.tr('After run: on current page only'), 'current_page')
        self.auto_region_merge_combobox.currentIndexChanged.connect(self.on_auto_region_merge_changed)
        sublock_arm = ConfigSubBlock(
            self.auto_region_merge_combobox,
            self.tr('After Run: Region merge'),
            discription=self.tr('Automatically run the Region merge tool after a pipeline run. Uses settings from Tools → Region merge tool.')
        )
        generalConfigPanel.addSublock(sublock_arm)

        generalConfigPanel.addTextLabel(label_display)
        self.darkmode_checker = DangoSwitch(self)
        self.darkmode_checker.setChecked(bool(getattr(pcfg, 'darkmode', False)))
        darkmode_sublock = ConfigSubBlock(
            self.darkmode_checker,
            self.tr('Dark mode'),
            discription=self.tr('Use dark theme. Restart or toggle View → Dark Mode to apply fully.'),
            vertical_layout=False,
            insert_stretch=True
        )
        generalConfigPanel.addSublock(darkmode_sublock)
        self.darkmode_checker.checkedChanged.connect(self.on_darkmode_checker_changed)

        self.bubbly_ui_checker = DangoSwitch(self)
        self.bubbly_ui_checker.setChecked(bool(getattr(pcfg, 'bubbly_ui', True)))
        bubbly_ui_sublock = ConfigSubBlock(
            self.bubbly_ui_checker,
            self.tr('Bubbly UI'),
            discription=self.tr('Rounder corners, gradients, softer look. Same as View → Bubbly UI.'),
            vertical_layout=False,
            insert_stretch=True
        )
        generalConfigPanel.addSublock(bubbly_ui_sublock)
        self.bubbly_ui_checker.checkedChanged.connect(self.on_bubbly_ui_checker_changed)

        self.smooth_scroll_spin = QSpinBox()
        self.smooth_scroll_spin.setRange(0, 400)
        self.smooth_scroll_spin.setSingleStep(20)
        self.smooth_scroll_spin.setValue(getattr(pcfg, 'smooth_scroll_duration_ms', 0))
        self.smooth_scroll_spin.setSpecialValueText(self.tr('Off'))
        smooth_scroll_sublock = ConfigSubBlock(
            self.smooth_scroll_spin,
            self.tr('Smooth scroll (ms)'),
            discription=self.tr('Animate scroll position on wheel for an enhanced feel. 0 = off. 80–200 = subtle. Applies to config and dialogs.'),
            vertical_layout=False
        )
        generalConfigPanel.addSublock(smooth_scroll_sublock)
        self.smooth_scroll_spin.valueChanged.connect(self.on_smooth_scroll_changed)

        self.motion_blur_on_scroll_checker = QCheckBox()
        self.motion_blur_on_scroll_checker.setChecked(bool(getattr(pcfg, 'motion_blur_on_scroll', False)))
        motion_blur_sublock = ConfigSubBlock(
            self.motion_blur_on_scroll_checker,
            self.tr('Motion blur on scroll'),
            discription=self.tr('Briefly blur content during scroll (can be costly on some systems).'),
            vertical_layout=False,
            insert_stretch=True
        )
        generalConfigPanel.addSublock(motion_blur_sublock)
        self.motion_blur_on_scroll_checker.stateChanged.connect(self.on_motion_blur_on_scroll_changed)

        self.reduce_motion_checker = QCheckBox()
        self.reduce_motion_checker.setChecked(bool(getattr(pcfg, 'reduce_motion', False)))
        reduce_motion_sublock = ConfigSubBlock(
            self.reduce_motion_checker,
            self.tr('Reduce motion'),
            discription=self.tr('Shorter or no UI animations (accessibility / preference).'),
            vertical_layout=False,
            insert_stretch=True
        )
        generalConfigPanel.addSublock(reduce_motion_sublock)
        self.reduce_motion_checker.stateChanged.connect(self.on_reduce_motion_changed)

        self.use_custom_cursor_checker = DangoSwitch(self)
        self.use_custom_cursor_checker.setChecked(bool(getattr(pcfg, 'use_custom_cursor', False)))
        self.custom_cursor_path_edit = QLineEdit()
        self.custom_cursor_path_edit.setPlaceholderText(self.tr('Path to cursor image (.png, .cur)'))
        self.custom_cursor_path_edit.setText(getattr(pcfg, 'custom_cursor_path', '') or '')
        self.custom_cursor_path_edit.setClearButtonEnabled(True)
        self.custom_cursor_browse_btn = QPushButton(self.tr('Browse...'))
        self.custom_cursor_browse_btn.clicked.connect(self._on_custom_cursor_browse)
        cursor_path_layout = QHBoxLayout()
        cursor_path_layout.addWidget(self.custom_cursor_path_edit)
        cursor_path_layout.addWidget(self.custom_cursor_browse_btn)
        cursor_path_widget = Widget()
        cursor_path_widget.setLayout(cursor_path_layout)
        use_cursor_sublock = ConfigSubBlock(
            self.use_custom_cursor_checker,
            self.tr('Use custom cursor'),
            discription=self.tr('Use a custom mouse cursor from file. Set path below.'),
            vertical_layout=False,
            insert_stretch=True
        )
        generalConfigPanel.addSublock(use_cursor_sublock)
        self.use_custom_cursor_checker.checkedChanged.connect(self.on_use_custom_cursor_changed)
        cursor_path_sublock = ConfigSubBlock(
            cursor_path_widget,
            self.tr('Custom cursor path'),
            discription=self.tr('PNG or CUR file. Applied when "Use custom cursor" is on.'),
            vertical_layout=False
        )
        generalConfigPanel.addSublock(cursor_path_sublock)
        self.custom_cursor_path_edit.textChanged.connect(self._on_custom_cursor_path_text_changed)
        self.custom_cursor_path_edit.editingFinished.connect(self._on_custom_cursor_path_editing_finished)

        self.display_lang_combobox = QComboBox()
        self.display_lang_combobox.addItems(list(DISPLAY_LANGUAGE_MAP.keys()))
        self.display_lang_combobox.setCurrentText(self._display_lang_to_label(getattr(pcfg, 'display_lang', C.DEFAULT_DISPLAY_LANG)))
        self.display_lang_combobox.currentTextChanged.connect(self.on_display_lang_combobox_changed)
        sublock = ConfigSubBlock(self.display_lang_combobox, self.tr('Display language'), discription=self.tr('UI language. Same as View → Display Language.'))
        generalConfigPanel.addSublock(sublock)

        self.config_font_scale_spin = QDoubleSpinBox()
        self.config_font_scale_spin.setRange(0.8, 1.5)
        self.config_font_scale_spin.setSingleStep(0.1)
        self.config_font_scale_spin.setValue(getattr(pcfg, 'config_panel_font_scale', 1.0))
        self.config_font_scale_spin.valueChanged.connect(self.on_config_font_scale_changed)
        sublock = ConfigSubBlock(self.config_font_scale_spin, self.tr('Config panel font scale (0.8–1.5)'), discription=self.tr('Scale font size in this Config panel. Reopen Config to apply.'))
        generalConfigPanel.addSublock(sublock)

        self.logical_dpi_spin = QSpinBox()
        self.logical_dpi_spin.setRange(0, 300)
        self.logical_dpi_spin.setValue(0)
        self.logical_dpi_spin.setSpecialValueText(self.tr('System default'))
        sublock = ConfigSubBlock(self.logical_dpi_spin, self.tr('Logical DPI (restart to apply)'), discription=self.tr('0 = use system. Use 96 or 72 if font scaling is wrong.'))
        generalConfigPanel.addSublock(sublock)
        self.logical_dpi_spin.valueChanged.connect(self.on_logical_dpi_changed)

        generalConfigPanel.addTextLabel(label_typesetting)

        text_fit_auto = self.tr('Auto fit to box')
        text_fit_fixed = self.tr('Fixed size (use font size list)')
        self.text_box_format_combox, sublock = combobox_with_label(
            [text_fit_auto, text_fit_fixed],
            self.tr('Text in box'),
            discription=self.tr('How text is fitted in each text box. Auto fit scales font size so text fits the box while keeping line structure (see issue #1077). Fixed size uses the font size list below.'),
            parent=self,
            insert_stretch=True
        )
        self.text_box_format_combox.setToolTip(self.tr('Auto fit: program decides font size so text fits the balloon. Fixed: use the font size list.'))
        generalConfigPanel.addSublock(sublock)
        self.text_box_format_combox.activated.connect(self.on_text_box_format_changed)

        dec_program_str = self.tr('decide by program')
        use_global_str = self.tr('use global setting')

        global_fntfmt_widget = QWidget()
        global_fntfmt_layout = QGridLayout(global_fntfmt_widget)
        global_fntfmt_layout.setSpacing(0)
        global_fntfmt_widget.setContentsMargins(0, 0, 0, 0)

        b = generalConfigPanel.addBlockWidget(global_fntfmt_widget)
        b.layout().setContentsMargins(0, 0, 0, 0)
        b.setContentsMargins(0, 0, 0, 0)
        self.let_fntsize_combox, sublock = combobox_with_label([dec_program_str, use_global_str], self.tr('Font Size'), parent=self, insert_stretch=True)
        global_fntfmt_layout.addWidget(sublock, 0, 0)

        self.let_fntsize_combox.activated.connect(self.on_fntsize_flag_changed)
        self.let_fntstroke_combox, sublock = combobox_with_label([dec_program_str, use_global_str], self.tr('Stroke Size'), parent=self, insert_stretch=True)
        self.let_fntstroke_combox.activated.connect(self.on_fntstroke_flag_changed)
        global_fntfmt_layout.addWidget(sublock, 1, 0)

        self.let_fntcolor_combox, sublock = combobox_with_label([dec_program_str, use_global_str], self.tr('Font Color'), parent=self, insert_stretch=True)
        self.let_fntcolor_combox.activated.connect(self.on_fontcolor_flag_changed)
        global_fntfmt_layout.addWidget(sublock, 2, 0)
        self.let_fnt_scolor_combox, sublock = combobox_with_label([dec_program_str, use_global_str], self.tr('Stroke Color'), parent=self, insert_stretch=True)
        self.let_fnt_scolor_combox.activated.connect(self.on_font_scolor_flag_changed)
        global_fntfmt_layout.addWidget(sublock, 3, 0)

        self.default_stroke_width_spin = QDoubleSpinBox(self)
        self.default_stroke_width_spin.setRange(0, 5)
        self.default_stroke_width_spin.setSingleStep(0.1)
        self.default_stroke_width_spin.setDecimals(2)
        self.default_stroke_width_spin.setValue(getattr(pcfg.global_fontformat, 'stroke_width', 0) or 0)
        self.default_stroke_width_spin.valueChanged.connect(self.on_default_stroke_width_changed)
        sublock_sw = ConfigSubBlock(self.default_stroke_width_spin, self.tr('Default stroke width'), discription=self.tr('Global default when "use global setting" is selected for Stroke Size. 0 = no stroke.'))
        global_fntfmt_layout.addWidget(sublock_sw, 4, 0)

        # Default text box corner radius / shape: 0 = rectangle, >0 = rounded (large values → circle-like for square boxes).
        self.default_box_corner_radius_spin = QDoubleSpinBox(self)
        self.default_box_corner_radius_spin.setRange(0, 999)
        self.default_box_corner_radius_spin.setSingleStep(1.0)
        self.default_box_corner_radius_spin.setDecimals(1)
        self.default_box_corner_radius_spin.setValue(float(getattr(pcfg.global_fontformat, 'text_box_corner_radius', 0.0) or 0.0))
        self.default_box_corner_radius_spin.valueChanged.connect(self.on_default_box_corner_radius_changed)
        sublock_br = ConfigSubBlock(
            self.default_box_corner_radius_spin,
            self.tr('Default box corner radius'),
            discription=self.tr('0 = square/rectangle. >0 = rounded rectangle; for circle-like boxes use a large value (radius ≈ half of box size). Applies to newly created text boxes.')
        )
        global_fntfmt_layout.addWidget(sublock_br, 5, 0)

        self.default_stroke_color_btn = QPushButton(self)
        self.default_stroke_color_btn.setFixedWidth(80)
        self._update_default_stroke_color_button()
        self.default_stroke_color_btn.clicked.connect(self.on_default_stroke_color_clicked)
        sublock_sc = ConfigSubBlock(self.default_stroke_color_btn, self.tr('Default stroke color'), discription=self.tr('Global default when "use global setting" is selected for Stroke Color.'))
        global_fntfmt_layout.addWidget(sublock_sc, 6, 0)

        self.let_effect_combox, sublock = combobox_with_label([dec_program_str, use_global_str], self.tr('Effect'), parent=self, insert_stretch=True)
        self.let_effect_combox.activated.connect(self.on_effect_flag_changed)
        global_fntfmt_layout.addWidget(sublock, 7, 0)
        self.let_alignment_combox, sublock = combobox_with_label([dec_program_str, use_global_str], self.tr('Alignment'), parent=self, insert_stretch=True)
        self.let_alignment_combox.activated.connect(self.on_alignment_flag_changed)
        global_fntfmt_layout.addWidget(sublock, 8, 0)

        self.let_writing_mode_combox, sublock = combobox_with_label([dec_program_str, use_global_str], self.tr('Writing-mode'), parent=self, insert_stretch=True)
        self.let_writing_mode_combox.activated.connect(self.on_writing_mode_flag_changed)
        global_fntfmt_layout.addWidget(sublock, 9, 0)
        self.let_family_combox, sublock = combobox_with_label([self.tr('Keep existing'), self.tr('Always use global setting')], self.tr('Font Family'), parent=self, insert_stretch=True)
        self.let_family_combox.activated.connect(self.on_family_flag_changed)
        global_fntfmt_layout.addWidget(sublock, 10, 0)

        global_fntfmt_layout.addItem(QSpacerItem(0, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding), 11, 0)

        self.let_autolayout_checker, sublock = generalConfigPanel.addCheckBox(self.tr('Auto layout'),
                discription=self.tr('When on: scale font size to fit the speech bubble and wrap lines to the balloon shape. When off: use global font size and still wrap to the balloon (text may overflow if too long). Works with "Text in box" = Auto fit to box.'))

        self.let_autolayout_checker.stateChanged.connect(self.on_autolayout_changed)

        self.layout_constrain_to_bubble_checker, layout_sublock = generalConfigPanel.addCheckBox(
            self.tr('Constrain text box to bubble'),
            discription=self.tr('Keep the text box size within the detected bubble and do not move it. When on, the box will not grow past the bubble or shift position after layout.')
        )
        self.layout_constrain_to_bubble_checker.stateChanged.connect(self._on_layout_constrain_to_bubble_changed)

        self.layout_center_in_bubble_after_autolayout_checker, _ = generalConfigPanel.addCheckBox(
            self.tr('Center in bubble after auto layout'),
            discription=self.tr('After auto layout, move each text box so its center aligns with the bubble center. Skips boxes that are close to another (combined/overlapping bubbles).')
        )
        self.layout_center_in_bubble_after_autolayout_checker.stateChanged.connect(self._on_layout_center_in_bubble_after_autolayout_changed)
        self.layout_center_in_bubble_min_gap_spin = QDoubleSpinBox()
        self.layout_center_in_bubble_min_gap_spin.setRange(0.0, 200.0)
        self.layout_center_in_bubble_min_gap_spin.setSingleStep(10.0)
        self.layout_center_in_bubble_min_gap_spin.setDecimals(0)
        self.layout_center_in_bubble_min_gap_spin.setValue(float(getattr(pcfg.module, 'layout_center_in_bubble_min_gap_px', 40.0)))
        self.layout_center_in_bubble_min_gap_spin.valueChanged.connect(self._on_layout_center_in_bubble_min_gap_px_changed)
        generalConfigPanel.addSublock(ConfigSubBlock(
            self.layout_center_in_bubble_min_gap_spin,
            self.tr('Center in bubble: skip if another box within (px)'),
            discription=self.tr('Do not center a box if another text box is within this many pixels (avoids pulling multiple boxes to the same spot in combined bubbles).')
        ))

        self.layout_check_overflow_after_layout_checker, _ = generalConfigPanel.addCheckBox(
            self.tr('Check overflow after layout'),
            discription=self.tr('After layout, check if the text box or lines extend outside the bubble. Shrink box to bubble and/or scale font down to fix (no model).')
        )
        self.layout_check_overflow_after_layout_checker.stateChanged.connect(self._on_layout_check_overflow_after_layout_changed)
        self.layout_box_size_check_model_id_edit = QLineEdit()
        self.layout_box_size_check_model_id_edit.setPlaceholderText(self.tr('Use "builtin" for zero-shot CLIP, or a Hugging Face model ID. Leave empty to skip.'))
        self.layout_box_size_check_model_id_edit.textChanged.connect(self._on_layout_box_size_check_model_id_changed)
        generalConfigPanel.addSublock(ConfigSubBlock(
            self.layout_box_size_check_model_id_edit,
            self.tr('Box size check model ID (secondary)'),
            discription=self.tr('Built-in: use "builtin" (zero-shot CLIP). Custom: image-classification model with labels too_large, too_small, ok. Empty = geometric checks only.')
        ))

        self.layout_optimal_breaks_checker, _ = generalConfigPanel.addCheckBox(
            self.tr('Optimal line breaks'),
            discription=self.tr('Use dynamic programming to choose better line breaks (non-CJK). Reduces awkward mid-word wraps.')
        )
        self.layout_optimal_breaks_checker.stateChanged.connect(self._on_layout_optimal_breaks_changed)

        self.layout_hyphenation_checker, _ = generalConfigPanel.addCheckBox(
            self.tr('Hyphenation'),
            discription=self.tr('Allow hyphenation when breaking long words (non-CJK). Works with Optimal line breaks.')
        )
        self.layout_hyphenation_checker.stateChanged.connect(self._on_layout_hyphenation_changed)

        self.optimize_line_breaks_checker, _ = generalConfigPanel.addCheckBox(
            self.tr('Optimize line breaks (fewer lines)'),
            discription=self.tr('Try slightly larger font / fewer lines so text fits with fewer breaks. Experimental.')
        )
        self.optimize_line_breaks_checker.stateChanged.connect(self._on_optimize_line_breaks_changed)

        # Layout penalty tuning (advanced): exposed so users can tweak behavior if needed.
        self.layout_short_line_penalty_spin = QDoubleSpinBox()
        self.layout_short_line_penalty_spin.setRange(0.0, 400.0)
        self.layout_short_line_penalty_spin.setSingleStep(10.0)
        self.layout_short_line_penalty_spin.setDecimals(1)
        self.layout_short_line_penalty_spin.setValue(float(getattr(pcfg.module, 'layout_short_line_penalty', 80.0)))
        self.layout_short_line_penalty_spin.valueChanged.connect(self._on_layout_short_line_penalty_changed)
        short_line_sublock = ConfigSubBlock(
            self.layout_short_line_penalty_spin,
            self.tr('Short line penalty'),
            discription=self.tr('Penalty per very short non-final line in auto layout (higher = avoid 1–2 word stub lines).')
        )
        generalConfigPanel.addSublock(short_line_sublock)

        self.layout_height_overflow_penalty_spin = QDoubleSpinBox()
        self.layout_height_overflow_penalty_spin.setRange(0.0, 2000.0)
        self.layout_height_overflow_penalty_spin.setSingleStep(50.0)
        self.layout_height_overflow_penalty_spin.setDecimals(1)
        self.layout_height_overflow_penalty_spin.setValue(float(getattr(pcfg.module, 'layout_height_overflow_penalty', 360.0)))
        self.layout_height_overflow_penalty_spin.valueChanged.connect(self._on_layout_height_overflow_penalty_changed)
        height_penalty_sublock = ConfigSubBlock(
            self.layout_height_overflow_penalty_spin,
            self.tr('Height overflow penalty'),
            discription=self.tr('When a layout would exceed bubble height, this penalty is applied. Lower (e.g. 400–500) = fewer lines, larger font, allow some overflow. Higher (e.g. 1000+) = strict fit, more lines, smaller font.')
        )
        generalConfigPanel.addSublock(height_penalty_sublock)

        # 2.1 Font scaling to fit bubble: min/max font size (pt), fit-to-bubble toggle
        self.layout_font_size_min_spin = QDoubleSpinBox()
        self.layout_font_size_min_spin.setRange(4.0, 48.0)
        self.layout_font_size_min_spin.setSingleStep(1.0)
        self.layout_font_size_min_spin.setDecimals(1)
        self.layout_font_size_min_spin.setValue(float(getattr(pcfg.module, 'layout_font_size_min', 8.0)))
        self.layout_font_size_min_spin.valueChanged.connect(self._on_layout_font_size_min_changed)
        generalConfigPanel.addSublock(ConfigSubBlock(
            self.layout_font_size_min_spin,
            self.tr('Layout font size min (pt)'),
            discription=self.tr('Minimum font size (points) for auto layout. Text will not be scaled below this.')
        ))
        self.layout_font_size_max_spin = QDoubleSpinBox()
        self.layout_font_size_max_spin.setRange(12.0, 120.0)
        self.layout_font_size_max_spin.setSingleStep(2.0)
        self.layout_font_size_max_spin.setDecimals(1)
        self.layout_font_size_max_spin.setValue(float(getattr(pcfg.module, 'layout_font_size_max', 72.0)))
        self.layout_font_size_max_spin.valueChanged.connect(self._on_layout_font_size_max_changed)
        generalConfigPanel.addSublock(ConfigSubBlock(
            self.layout_font_size_max_spin,
            self.tr('Layout font size max (pt)'),
            discription=self.tr('Maximum font size (points) for auto layout. Prevents text from growing too large.')
        ))
        self.layout_font_fit_bubble_checker, _ = generalConfigPanel.addCheckBox(
            self.tr('Scale font to fit bubble'),
            discription=self.tr('When on, scale laid-out text to fit inside the bubble (ratio-based); when off, only apply line-break scaling. Font size is always clamped to min/max above.')
        )
        self.layout_font_fit_bubble_checker.stateChanged.connect(self._on_layout_font_fit_bubble_changed)
        self.layout_font_binary_search_checker, _ = generalConfigPanel.addCheckBox(
            self.tr('Binary search font size'),
            discription=self.tr('Find the largest font size that fits the bubble by trying multiple sizes (more accurate, slower). Only when "Scale font to fit bubble" is on and non-CJK.')
        )
        self.layout_font_binary_search_checker.stateChanged.connect(self._on_layout_font_binary_search_changed)
        # Diamond-Text: balloon shape (auto, round, elongated, narrow, diamond, square, bevel, pentagon, point)
        self.layout_balloon_shape_combo, _ = generalConfigPanel.addCombobox(
            [
                self.tr('Auto'), self.tr('Round'), self.tr('Elongated'), self.tr('Narrow'),
                self.tr('Diamond'), self.tr('Square'), self.tr('Bevel'), self.tr('Pentagon'), self.tr('Point'),
            ],
            self.tr('Balloon shape (Diamond-Text)'),
            discription=self.tr('Hint for bubble shape: Auto detects from aspect ratio; other options set insets and line-length scoring (e.g. round/square/bevel = more uniform line lengths).')
        )
        self.layout_balloon_shape_combo.activated.connect(self._on_layout_balloon_shape_changed)
        # When Auto: pick which method(s) to use and in what order
        self.layout_balloon_shape_auto_method_combo, _ = generalConfigPanel.addCombobox(
            [
                self.tr('Aspect ratio only'),
                self.tr('Contour only'),
                self.tr('Model only'),
                self.tr('Model, then contour'),
                self.tr('Model, then aspect ratio'),
                self.tr('Contour, then aspect ratio'),
                self.tr('Model, then contour, then aspect ratio'),
            ],
            self.tr('Auto shape method'),
            discription=self.tr('When balloon shape is Auto, which method(s) to use. Contour = mask-based (OpenCV). Model = HF image-classification (set model ID below).')
        )
        self.layout_balloon_shape_auto_method_combo.activated.connect(self._on_layout_balloon_shape_auto_method_changed)
        self.layout_balloon_shape_model_id_edit = QLineEdit()
        self.layout_balloon_shape_model_id_edit.setPlaceholderText(self.tr('Default: prithivMLmods/Geometric-Shapes-Classification (92.9M). Lighter: 0-ma/vit-geometric-shapes-tiny'))
        self.layout_balloon_shape_model_id_edit.textChanged.connect(self._on_layout_balloon_shape_model_id_changed)
        generalConfigPanel.addSublock(ConfigSubBlock(
            self.layout_balloon_shape_model_id_edit,
            self.tr('Balloon shape model ID'),
            discription=self.tr('Default is prithivMLmods/Geometric-Shapes-Classification (SigLIP2-base, 92.9M params, 8 shapes, ~99% accuracy). Use 0-ma/vit-geometric-shapes-tiny for a lighter 5.5M model. Wide bubbles are forced to oval when the model says square.')
        ))
        # Minimum line width so short text (e.g. "pluck!") is not broken into 2–3 char lines
        self.layout_min_line_width_spin = QSpinBox()
        self.layout_min_line_width_spin.setRange(40, 400)
        self.layout_min_line_width_spin.setSingleStep(10)
        self.layout_min_line_width_spin.setValue(int(float(getattr(pcfg.module, 'layout_min_line_width_px', 80.0))))
        self.layout_min_line_width_spin.valueChanged.connect(self._on_layout_min_line_width_changed)
        generalConfigPanel.addSublock(ConfigSubBlock(
            self.layout_min_line_width_spin,
            self.tr('Minimum line width (px)'),
            discription=self.tr('Layout never uses a narrower line width; avoids short text (e.g. "pluck!") breaking into 2–3 character lines.')
        ))
        # Max line width fraction for free-standing text (no bubble): lower = more lines
        self.layout_max_line_width_frac_no_bubble_spin = QDoubleSpinBox()
        self.layout_max_line_width_frac_no_bubble_spin.setRange(0.50, 1.00)
        self.layout_max_line_width_frac_no_bubble_spin.setSingleStep(0.02)
        self.layout_max_line_width_frac_no_bubble_spin.setDecimals(2)
        self.layout_max_line_width_frac_no_bubble_spin.setValue(float(getattr(pcfg.module, 'layout_max_line_width_frac_no_bubble', 0.78)))
        self.layout_max_line_width_frac_no_bubble_spin.valueChanged.connect(self._on_layout_max_line_width_frac_no_bubble_changed)
        generalConfigPanel.addSublock(ConfigSubBlock(
            self.layout_max_line_width_frac_no_bubble_spin,
            self.tr('Max line width fraction (no bubble)'),
            discription=self.tr('For text boxes outside bubbles: max line width = box width × this. Lower values give more, shorter lines (e.g. 0.78).')
        ))
        # 2.3 Stub penalty for 1-word lines
        self.layout_stub_penalty_1word_spin = QDoubleSpinBox()
        self.layout_stub_penalty_1word_spin.setRange(200.0, 3000.0)
        self.layout_stub_penalty_1word_spin.setSingleStep(100.0)
        self.layout_stub_penalty_1word_spin.setDecimals(0)
        self.layout_stub_penalty_1word_spin.setValue(float(getattr(pcfg.module, 'layout_stub_penalty_1word', 2000.0)))
        self.layout_stub_penalty_1word_spin.valueChanged.connect(self._on_layout_stub_penalty_1word_changed)
        generalConfigPanel.addSublock(ConfigSubBlock(
            self.layout_stub_penalty_1word_spin,
            self.tr('1-word line penalty'),
            discription=self.tr('Penalty for a single word on its own line in layout scoring (higher = strongly avoid e.g. "the" alone).')
        ))
        self.layout_panel_preserve_line_breaks_checker, _ = generalConfigPanel.addCheckBox(
            self.tr('Translation panel: preserve line breaks'),
            discription=self.tr('When on, the translation panel does not wrap by width so line breaks match the canvas bubble (rendering parity).')
        )
        self.layout_panel_preserve_line_breaks_checker.stateChanged.connect(self._on_layout_panel_preserve_line_breaks_changed)

        self.let_uppercase_checker, _ = generalConfigPanel.addCheckBox(self.tr('To uppercase'))
        self.let_uppercase_checker.stateChanged.connect(self.on_uppercase_changed)

        self.let_textstyle_indep_checker, _ = generalConfigPanel.addCheckBox(self.tr('Independent text styles for each projects'))
        self.let_textstyle_indep_checker.stateChanged.connect(self.on_textstyle_indep_changed)

        self.let_show_only_custom_fonts, sublock = generalConfigPanel.addCheckBox(self.tr("Show only custom fonts"))
        self.let_show_only_custom_fonts.stateChanged.connect(self.on_show_only_custom_fonts)

        generalConfigPanel.addTextLabel(label_ocr_result)
        self.ocr_spell_check_checker, _ = generalConfigPanel.addCheckBox(
            self.tr('Spell check / Auto-correct OCR result'),
            discription=self.tr('After OCR runs, correct misspelled words using a spell checker when there is a single suggestion (e.g. "teh" → "the"). Requires pyenchant and a system dictionary (e.g. en_US). Enable this in General to auto-correct OCR text.')
        )
        self.ocr_spell_check_checker.stateChanged.connect(self.on_ocr_spell_check_changed)

        generalConfigPanel.addTextLabel(label_save)
        self.rst_imgformat_combobox, imsave_sublock = generalConfigPanel.addCombobox(['PNG', 'JPG', 'WEBP', 'JXL'], self.tr('Result image format'))
        self.rst_imgformat_combobox.activated.connect(self.on_rst_imgformat_changed)
        self.rst_imgquality_edit = PercentageLineEdit('100')
        self.rst_imgquality_edit.setFixedWidth(CONFIG_COMBOBOX_SHORT)
        self.rst_imgquality_edit.finish_edited.connect(self.on_edit_quality_changed)

        sublock = ConfigSubBlock(self.rst_imgquality_edit, self.tr('Quality'), vertical_layout=False)
        sublock.layout().setAlignment(Qt.AlignmentFlag.AlignLeft)
        sublock.layout().insertStretch(-1)
        imsave_sublock.layout().addWidget(sublock)

        self.rst_webp_lossless_checker, _ = generalConfigPanel.addCheckBox(self.tr('WebP lossless'), None, imsave_sublock)
        self.rst_webp_lossless_checker.setToolTip(self.tr('Use lossless WebP when result format is WebP (ignores quality).'))
        self.rst_webp_lossless_checker.stateChanged.connect(self.on_webp_lossless_changed)
        self.rst_imgformat_combobox.currentTextChanged.connect(self._update_webp_lossless_visibility)

        self.intermediate_imgformat_combobox, intermediate_imsave_sublock = generalConfigPanel.addCombobox(['PNG', 'JXL'], self.tr('Intermediate image format'))
        self.intermediate_imgformat_combobox.activated.connect(self.on_intermediate_imgformat_changed)

        generalConfigPanel.addTextLabel(label_canvas)
        self.context_menu_btn = QPushButton(self.tr("Context menu options..."))
        self.context_menu_btn.setToolTip(self.tr("Choose which actions appear in the canvas right-click menu."))
        self.context_menu_btn.clicked.connect(self._open_context_menu_config)
        sublock_ctx = ConfigSubBlock(self.context_menu_btn, self.tr("Right-click menu"), discription=self.tr("Same as View → Context menu options. Show or hide actions in the canvas context menu by category."))
        generalConfigPanel.addSublock(sublock_ctx)

        generalConfigPanel.addTextLabel(label_integrations)

        sublock = ConfigSubBlock(ConfigTextLabel(self.tr("<a href=\"https://github.com/dmMaze/BallonsTranslator/tree/master/doc/saladict.md\">Installation guide</a>"), _config_font_size(CONFIG_FONTSIZE_CONTENT) - 2), vertical_layout=False)
        sublock.layout().insertStretch(-1)
        generalConfigPanel.addSublock(sublock)

        self.selectext_minimenu_checker, _ = generalConfigPanel.addCheckBox(self.tr('Show mini menu when selecting text.'))
        self.selectext_minimenu_checker.stateChanged.connect(self.on_selectext_minimenu_changed)
        self.saladict_shortcut = QKeySequenceEdit("ALT+W", self)
        self.saladict_shortcut.keySequenceChanged.connect(self.on_saladict_shortcut_changed)
        self.saladict_shortcut.setFixedWidth(CONFIG_COMBOBOX_MIDEAN)

        sublock = ConfigSubBlock(self.saladict_shortcut, self.tr("Shortcut"), vertical_layout=False)
        sublock.layout().insertStretch(-1)
        generalConfigPanel.addSublock(sublock)
        self.searchurl_combobox, _ = generalConfigPanel.addCombobox(["https://www.google.com/search?q=", "https://www.bing.com/search?q=", "https://duckduckgo.com/?q=", "https://yandex.com/search/?text=", "http://www.baidu.com/s?wd=", "https://search.yahoo.com/search;?p=", "https://www.urbandictionary.com/define.php?term="], self.tr("Search Engines"), fix_size=False)
        self.searchurl_combobox.setEditable(True)
        self.searchurl_combobox.setFixedWidth(CONFIG_COMBOBOX_LONG)
        self.searchurl_combobox.currentTextChanged.connect(self.on_searchurl_changed)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.configTreeFocusRing)
        splitter.addWidget(self.configContent)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        hlayout = QHBoxLayout(self)

        hlayout.addWidget(splitter)
        hlayout.setSpacing(0)
        hlayout.setContentsMargins(0, 0, 0, 0)

        self.configTable.expandAll()

    def on_load_model_changed(self):
        pcfg.module.load_model_on_demand = self.load_model_checker.isChecked()

    def on_runcache_changed(self):
        pcfg.module.empty_runcache = self.empty_runcache_checker.isChecked()

    def on_release_caches_after_batch_changed(self):
        pcfg.release_caches_after_batch = self.release_caches_after_batch_checker.isChecked()

    def _on_ocr_cache_enabled_changed(self):
        pcfg.module.ocr_cache_enabled = self.ocr_cache_enabled_checker.isChecked()

    def _on_ocr_auto_by_language_changed(self):
        pcfg.module.ocr_auto_by_language = self.ocr_auto_by_language_checker.isChecked()

    def _on_skip_already_translated_changed(self):
        pcfg.module.skip_already_translated = self.skip_already_translated_checker.isChecked()

    def _on_show_module_tier_badges_changed(self):
        pcfg.show_module_tier_badges_in_tooltips = self.show_module_tier_badges_checker.isChecked()

    def _on_merge_nearby_blocks_changed(self):
        pcfg.module.merge_nearby_blocks_collision = self.merge_nearby_blocks_checker.isChecked()

    def _on_merge_nearby_blocks_gap_changed(self, value):
        pcfg.module.merge_nearby_blocks_gap_ratio = float(value)

    def _on_merge_nearby_blocks_min_blocks_changed(self, value):
        pcfg.module.merge_nearby_blocks_min_blocks = int(value)

    def on_keepline_clicked(self):
        pcfg.module.keep_exist_textlines = self.detect_config_panel.keep_existing_checker.isChecked()

    def addConfigBlock(self, header: str) -> Tuple[ConfigBlock, TableItem]:
        cb = ConfigBlock(header, parent=self)
        cb.sublock_pressed.connect(self.onSublockPressed)
        self.configContent.addConfigBlock(cb)
        cb.setIndex(len(self.configContent.config_block_list)-1)
        ti = self.configTable.addHeader(header)
        return cb, ti

    def onSublockPressed(self, idx0, idx1):
        self.configTable.setCurrentItem(idx0, idx1)
        self.configContent.deactiveLabel()

    def onTableItemPressed(self, idx0, idx1):
        self.configContent.setActiveLabel(idx0, idx1)

    def on_open_onstartup_changed(self):
        pcfg.open_recent_on_startup = self.open_on_startup_checker.isChecked()

    def on_show_welcome_screen_changed(self):
        pcfg.show_welcome_screen = self.show_welcome_screen_checker.isChecked()

    def on_auto_update_from_github_changed(self):
        pcfg.auto_update_from_github = self.auto_update_from_github_checker.isChecked()

    def on_dev_mode_changed(self):
        pcfg.dev_mode = self.dev_mode_checker.isChecked()
        try:
            from utils.logger import apply_dev_mode_logging
            apply_dev_mode_logging(pcfg.dev_mode)
        except Exception:
            pass
        self.dev_mode_changed.emit()

    def on_recent_proj_list_max_changed(self, value: int):
        pcfg.recent_proj_list_max = value

    def on_logical_dpi_changed(self, value: int):
        pcfg.logical_dpi = value

    def on_confirm_before_run_changed(self):
        pcfg.confirm_before_run = self.confirm_before_run_checker.isChecked()

    def on_manual_mode_changed(self):
        pcfg.manual_mode = self.manual_mode_checker.isChecked()
        self.manual_mode_changed.emit()

    def on_skip_ignored_in_run_changed(self):
        pcfg.skip_ignored_in_run = self.skip_ignored_in_run_checker.isChecked()

    def _open_context_menu_config(self):
        dlg = ContextMenuConfigDialog(self)
        dlg.exec()

    def on_auto_region_merge_changed(self):
        idx = self.auto_region_merge_combobox.currentIndex()
        pcfg.auto_region_merge_after_run = self.auto_region_merge_combobox.itemData(idx) or 'never'

    def on_darkmode_checker_changed(self):
        pcfg.darkmode = self.darkmode_checker.isChecked()
        self.darkmode_changed.emit(pcfg.darkmode)

    def on_bubbly_ui_checker_changed(self):
        pcfg.bubbly_ui = self.bubbly_ui_checker.isChecked()
        self.bubbly_ui_changed.emit(pcfg.bubbly_ui)

    def on_smooth_scroll_changed(self, value: int):
        pcfg.smooth_scroll_duration_ms = value
        if hasattr(self, 'configContent') and hasattr(self.configContent, 'setSmoothScrollDuration'):
            self.configContent.setSmoothScrollDuration(value)

    def on_motion_blur_on_scroll_changed(self):
        pcfg.motion_blur_on_scroll = self.motion_blur_on_scroll_checker.isChecked()
        if hasattr(self, 'configContent') and hasattr(self.configContent, 'setMotionBlurOnScroll'):
            self.configContent.setMotionBlurOnScroll(pcfg.motion_blur_on_scroll)

    def on_reduce_motion_changed(self):
        pcfg.reduce_motion = self.reduce_motion_checker.isChecked()

    def on_use_custom_cursor_changed(self):
        pcfg.use_custom_cursor = self.use_custom_cursor_checker.isChecked()
        pcfg.custom_cursor_path = self.custom_cursor_path_edit.text().strip()
        self.custom_cursor_changed.emit()

    def _on_custom_cursor_browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr('Select cursor image'),
            self.custom_cursor_path_edit.text() or '',
            self.tr('Cursor images (*.png *.cur *.ico);;All files (*)')
        )
        if path:
            self.custom_cursor_path_edit.setText(path)
            pcfg.custom_cursor_path = path
            if self.use_custom_cursor_checker.isChecked():
                self.custom_cursor_changed.emit()

    def _on_custom_cursor_path_text_changed(self, text: str):
        pcfg.custom_cursor_path = text.strip()

    def _on_custom_cursor_path_editing_finished(self):
        if self.use_custom_cursor_checker.isChecked():
            self.custom_cursor_changed.emit()

    def _display_lang_to_label(self, code: str) -> str:
        for label, c in DISPLAY_LANGUAGE_MAP.items():
            if c == code:
                return label
        return list(DISPLAY_LANGUAGE_MAP.keys())[0] if DISPLAY_LANGUAGE_MAP else 'English'

    def on_display_lang_combobox_changed(self, text: str):
        if text not in DISPLAY_LANGUAGE_MAP:
            return
        code = DISPLAY_LANGUAGE_MAP[text]
        pcfg.display_lang = code
        self.display_lang_changed.emit(code)

    def on_config_font_scale_changed(self, value: float):
        pcfg.config_panel_font_scale = value

    def on_ocr_spell_check_changed(self):
        pcfg.ocr_spell_check = self.ocr_spell_check_checker.isChecked()

    def on_default_device_index_changed(self, index: int):
        if index == 0:
            pcfg.default_device = ''
        else:
            pcfg.default_device = self.default_device_combobox.currentText()

    def on_unload_after_idle_changed(self, value: int):
        pcfg.unload_after_idle_minutes = value

    def _on_image_upscale_initial_changed(self):
        pcfg.module.image_upscale_initial = self.image_upscale_initial_checker.isChecked()

    def _on_image_upscale_initial_factor_changed(self, value: float):
        pcfg.module.image_upscale_initial_factor = value

    def _on_image_upscale_final_changed(self):
        pcfg.module.image_upscale_final = self.image_upscale_final_checker.isChecked()

    def _on_image_upscale_final_factor_changed(self, value: float):
        pcfg.module.image_upscale_final_factor = value

    def _on_processing_scale_changed(self):
        pcfg.module.processing_scale_enabled = self.processing_scale_checker.isChecked()

    def _on_colorization_changed(self):
        pcfg.module.enable_colorization = self.colorization_checker.isChecked()

    def _on_colorization_strength_changed(self, value: float):
        pcfg.module.colorization_strength = float(value)

    def _on_colorization_backend_changed(self, index: int):
        # Map combo index to backend key
        if index == 0:
            backend = "simple"          # Twilight
        elif index == 1:
            backend = "manga_vibrant"   # Magma
        else:
            backend = "cool"            # Ocean
        pcfg.module.colorization_backend = backend

    def _on_ocr_upscale_min_side_changed(self, value: int):
        pcfg.module.ocr_upscale_min_side = value

    def _on_inpaint_spill_after_changed(self, value: int):
        pcfg.module.inpaint_spill_to_disk_after_blocks = value

    def on_check_download_module_files(self):
        try:
            from modules.prepare_local_files import download_and_check_module_files
            download_and_check_module_files()
            QMessageBox.information(
                self,
                self.tr('Check / Download module files'),
                self.tr('Finished. If any module reported missing files, check the console/log and save them to the specified paths, then restart or switch module.')
            )
        except Exception as e:
            QMessageBox.warning(
                self,
                self.tr('Check / Download module files'),
                self.tr('Error: ') + str(e)
            )

    def on_fntsize_flag_changed(self):
        pcfg.let_fntsize_flag = self.let_fntsize_combox.currentIndex()

    def on_fntstroke_flag_changed(self):
        pcfg.let_fntstroke_flag = self.let_fntstroke_combox.currentIndex()

    def _update_default_stroke_color_button(self):
        gf = pcfg.global_fontformat
        srgb = getattr(gf, 'srgb', [0, 0, 0]) or [0, 0, 0]
        r, g, b = int(srgb[0]) if len(srgb) > 0 else 0, int(srgb[1]) if len(srgb) > 1 else 0, int(srgb[2]) if len(srgb) > 2 else 0
        self.default_stroke_color_btn.setStyleSheet(f'background-color: rgb({r},{g},{b}); border: 1px solid #888;')

    def on_default_stroke_width_changed(self, value: float):
        pcfg.global_fontformat.stroke_width = value

    def on_default_box_corner_radius_changed(self, value: float):
        pcfg.global_fontformat.text_box_corner_radius = float(max(0.0, value))

    def on_default_stroke_color_clicked(self):
        gf = pcfg.global_fontformat
        srgb = getattr(gf, 'srgb', [0, 0, 0]) or [0, 0, 0]
        r, g, b = (srgb[0], srgb[1], srgb[2]) if len(srgb) >= 3 else (0, 0, 0)
        color = QColorDialog.getColor(QColor(r, g, b), self, self.tr('Default stroke color'))
        if color.isValid():
            pcfg.global_fontformat.srgb = [color.red(), color.green(), color.blue()]
            self._update_default_stroke_color_button()

    def on_autolayout_changed(self):
        pcfg.let_autolayout_flag = self.let_autolayout_checker.isChecked()

    def _on_layout_constrain_to_bubble_changed(self):
        pcfg.module.layout_constrain_to_bubble = self.layout_constrain_to_bubble_checker.isChecked()
        self.save_config.emit()

    def _on_layout_center_in_bubble_after_autolayout_changed(self):
        pcfg.module.layout_center_in_bubble_after_autolayout = self.layout_center_in_bubble_after_autolayout_checker.isChecked()
        self.save_config.emit()

    def _on_layout_center_in_bubble_min_gap_px_changed(self, value: float):
        pcfg.module.layout_center_in_bubble_min_gap_px = float(value)
        self.save_config.emit()

    def _on_layout_check_overflow_after_layout_changed(self):
        pcfg.module.layout_check_overflow_after_layout = self.layout_check_overflow_after_layout_checker.isChecked()
        self.save_config.emit()

    def _on_layout_box_size_check_model_id_changed(self, text: str):
        pcfg.module.layout_box_size_check_model_id = (text or "").strip()

    def _on_layout_optimal_breaks_changed(self):
        pcfg.module.layout_optimal_breaks = self.layout_optimal_breaks_checker.isChecked()
        self.save_config.emit()

    def _on_layout_hyphenation_changed(self):
        pcfg.module.layout_hyphenation = self.layout_hyphenation_checker.isChecked()
        self.save_config.emit()

    def _on_optimize_line_breaks_changed(self):
        pcfg.module.optimize_line_breaks = self.optimize_line_breaks_checker.isChecked()
        self.save_config.emit()

    def _on_layout_short_line_penalty_changed(self, value: float):
        pcfg.module.layout_short_line_penalty = float(value)
        self.save_config.emit()

    def _on_layout_height_overflow_penalty_changed(self, value: float):
        pcfg.module.layout_height_overflow_penalty = float(value)

    def _on_layout_font_size_min_changed(self, value: float):
        pcfg.module.layout_font_size_min = float(value)

    def _on_layout_font_size_max_changed(self, value: float):
        pcfg.module.layout_font_size_max = float(value)

    def _on_layout_font_fit_bubble_changed(self):
        pcfg.module.layout_font_fit_bubble = self.layout_font_fit_bubble_checker.isChecked()

    def _on_layout_font_binary_search_changed(self):
        pcfg.module.layout_font_binary_search = self.layout_font_binary_search_checker.isChecked()

    def _on_layout_balloon_shape_changed(self, index: int):
        pcfg.module.layout_balloon_shape = ("auto", "round", "elongated", "narrow", "diamond", "square", "bevel", "pentagon", "point")[index]

    def _on_layout_balloon_shape_auto_method_changed(self, index: int):
        pcfg.module.layout_balloon_shape_auto_method = (
            "aspect_ratio", "contour", "model", "model_contour", "model_ratio", "contour_ratio", "model_contour_ratio"
        )[index]

    def _on_layout_balloon_shape_model_id_changed(self, text: str):
        pcfg.module.layout_balloon_shape_model_id = (text or "").strip()

    def _on_layout_min_line_width_changed(self, value: int):
        pcfg.module.layout_min_line_width_px = float(value)

    def _on_layout_max_line_width_frac_no_bubble_changed(self, value: float):
        pcfg.module.layout_max_line_width_frac_no_bubble = float(value)

    def _on_layout_stub_penalty_1word_changed(self, value: float):
        pcfg.module.layout_stub_penalty_1word = float(value)

    def _on_layout_panel_preserve_line_breaks_changed(self):
        pcfg.module.layout_panel_preserve_line_breaks = self.layout_panel_preserve_line_breaks_checker.isChecked()
        self.save_config.emit()

    def on_text_box_format_changed(self):
        """Sync Font Size and Auto layout from the Text in box dropdown (issue #1077)."""
        idx = self.text_box_format_combox.currentIndex()
        if idx == 0:  # Auto fit to box
            self.let_fntsize_combox.setCurrentIndex(0)
            self.let_autolayout_checker.setChecked(True)
        else:  # Fixed size
            self.let_fntsize_combox.setCurrentIndex(1)
            self.let_autolayout_checker.setChecked(False)
        pcfg.let_fntsize_flag = self.let_fntsize_combox.currentIndex()
        pcfg.let_autolayout_flag = self.let_autolayout_checker.isChecked()

    def on_uppercase_changed(self):
        pcfg.let_uppercase_flag = self.let_uppercase_checker.isChecked()

    def on_textstyle_indep_changed(self):
        pcfg.let_textstyle_indep_flag = self.let_textstyle_indep_checker.isChecked()
        self.reload_textstyle.emit(pcfg.let_textstyle_indep_flag)

    def on_rst_imgformat_changed(self):
        pcfg.imgsave_ext = '.' + self.rst_imgformat_combobox.currentText().lower()

    def on_intermediate_imgformat_changed(self):
        pcfg.intermediate_imgsave_ext = '.' + self.intermediate_imgformat_combobox.currentText().lower()

    def on_edit_quality_changed(self, value: str):
        pcfg.imgsave_quality = int(value)

    def on_webp_lossless_changed(self):
        pcfg.imgsave_webp_lossless = self.rst_webp_lossless_checker.isChecked()

    def _update_webp_lossless_visibility(self):
        is_webp = self.rst_imgformat_combobox.currentText().upper() == 'WEBP'
        self.rst_webp_lossless_checker.setEnabled(is_webp)

    def on_selectext_minimenu_changed(self):
        pcfg.textselect_mini_menu = self.selectext_minimenu_checker.isChecked()

    def on_saladict_shortcut_changed(self):
        kstr = self.saladict_shortcut.keySequence().toString()
        if kstr:
            pcfg.saladict_shortcut = self.saladict_shortcut.keySequence().toString()

    def on_searchurl_changed(self):
        url = self.searchurl_combobox.currentText()
        pcfg.search_url = url

    def on_fontcolor_flag_changed(self):
        pcfg.let_fntcolor_flag = self.let_fntcolor_combox.currentIndex()

    def on_font_scolor_flag_changed(self):
        pcfg.let_fnt_scolor_flag = self.let_fnt_scolor_combox.currentIndex()

    def on_alignment_flag_changed(self):
        pcfg.let_alignment_flag = self.let_alignment_combox.currentIndex()

    def on_writing_mode_flag_changed(self):
        pcfg.let_writing_mode_flag = self.let_writing_mode_combox.currentIndex()

    def on_family_flag_changed(self):
        pcfg.let_family_flag = self.let_family_combox.currentIndex()

    def on_effect_flag_changed(self):
        pcfg.let_fnteffect_flag = self.let_effect_combox.currentIndex()

    def on_show_only_custom_fonts(self):
        pcfg.let_show_only_custom_fonts_flag = self.let_show_only_custom_fonts.isChecked()
        self.show_only_custom_font.emit(pcfg.let_show_only_custom_fonts_flag)

    def focusOnTranslator(self):
        idx0, idx1 = self.trans_sub_block.idx0, self.trans_sub_block.idx1
        self.configTable.setCurrentItem(idx0, idx1)
        self.configTable.tableitem_pressed.emit(idx0, idx1)

    def focusOnInpaint(self):
        idx0, idx1 = self.inpaint_sub_block.idx0, self.inpaint_sub_block.idx1
        self.configTable.setCurrentItem(idx0, idx1)
        self.configTable.tableitem_pressed.emit(idx0, idx1)

    def focusOnDetect(self):
        idx0, idx1 = self.detect_sub_block.idx0, self.detect_sub_block.idx1
        self.configTable.setCurrentItem(idx0, idx1)
        self.configTable.tableitem_pressed.emit(idx0, idx1)

    def focusOnOCR(self):
        idx0, idx1 = self.ocr_sub_block.idx0, self.ocr_sub_block.idx1
        self.configTable.setCurrentItem(idx0, idx1)
        self.configTable.tableitem_pressed.emit(idx0, idx1)

    def hideEvent(self, e) -> None:
        self.save_config.emit()
        return super().hideEvent(e)
        
    def setupConfig(self):
        self.blockSignals(True)

        if pcfg.open_recent_on_startup:
            self.open_on_startup_checker.setChecked(True)
        if getattr(pcfg, 'auto_update_from_github', False):
            self.auto_update_from_github_checker.setChecked(True)

        self.detect_config_panel.keep_existing_checker.setChecked(pcfg.module.keep_exist_textlines)
        if hasattr(self.detect_config_panel, 'dual_detect_checker'):
            self.detect_config_panel.dual_detect_checker.setChecked(getattr(pcfg.module, 'enable_dual_detect', False))
        if hasattr(self.detect_config_panel, 'secondary_detector_combobox'):
            self.detect_config_panel.secondary_detector_combobox.setCurrentText(getattr(pcfg.module, 'textdetector_secondary', '') or '')
        if hasattr(self.detect_config_panel, 'secondary_outside_bubble_only_checker'):
            self.detect_config_panel.secondary_outside_bubble_only_checker.setChecked(getattr(pcfg.module, 'secondary_detector_outside_bubble_only', False))
        if hasattr(self.detect_config_panel, 'tertiary_detect_checker'):
            self.detect_config_panel.tertiary_detect_checker.setChecked(getattr(pcfg.module, 'enable_tertiary_detect', False))
        if hasattr(self.detect_config_panel, 'tertiary_detector_combobox'):
            self.detect_config_panel.tertiary_detector_combobox.setCurrentText(getattr(pcfg.module, 'textdetector_tertiary', '') or '')
        if hasattr(self.detect_config_panel, 'allow_detection_box_rotation_checker'):
            self.detect_config_panel.allow_detection_box_rotation_checker.setChecked(getattr(pcfg.module, 'allow_detection_box_rotation', False))
        if hasattr(self.detect_config_panel, 'detection_rotation_threshold_spinbox'):
            self.detect_config_panel.detection_rotation_threshold_spinbox.setValue(int(getattr(pcfg.module, 'detection_rotation_threshold_degrees', 10.0)))
        if hasattr(self.inpaint_config_panel, 'inpaint_tile_size_spin'):
            self.inpaint_config_panel.inpaint_tile_size_spin.blockSignals(True)
            self.inpaint_config_panel.inpaint_tile_size_spin.setValue(getattr(pcfg.module, 'inpaint_tile_size', 0))
            self.inpaint_config_panel.inpaint_tile_size_spin.blockSignals(False)
        if hasattr(self.inpaint_config_panel, 'inpaint_tile_overlap_spin'):
            self.inpaint_config_panel.inpaint_tile_overlap_spin.blockSignals(True)
            self.inpaint_config_panel.inpaint_tile_overlap_spin.setValue(getattr(pcfg.module, 'inpaint_tile_overlap', 64))
            self.inpaint_config_panel.inpaint_tile_overlap_spin.blockSignals(False)
        if hasattr(self.inpaint_config_panel, 'exclude_labels_checker'):
            self.inpaint_config_panel.exclude_labels_checker.setChecked(getattr(pcfg.module, 'inpaint_exclude_labels_enabled', False))
            self.inpaint_config_panel.exclude_labels_edit.setEnabled(self.inpaint_config_panel.exclude_labels_checker.isChecked())
        if hasattr(self.inpaint_config_panel, 'exclude_labels_edit'):
            self.inpaint_config_panel.exclude_labels_edit.blockSignals(True)
            self.inpaint_config_panel.exclude_labels_edit.setText(getattr(pcfg.module, 'inpaint_exclude_labels', '') or '')
            self.inpaint_config_panel.exclude_labels_edit.blockSignals(False)
        if hasattr(self.inpaint_config_panel, 'full_image_checker'):
            self.inpaint_config_panel.full_image_checker.setChecked(getattr(pcfg.module, 'inpaint_full_image', False))
        self.let_effect_combox.setCurrentIndex(pcfg.let_fnteffect_flag)
        self.let_fntsize_combox.setCurrentIndex(pcfg.let_fntsize_flag)
        self.let_fntstroke_combox.setCurrentIndex(pcfg.let_fntstroke_flag)
        self.let_fntcolor_combox.setCurrentIndex(pcfg.let_fntcolor_flag)
        self.let_fnt_scolor_combox.setCurrentIndex(pcfg.let_fnt_scolor_flag)
        if hasattr(self, 'default_stroke_width_spin'):
            self.default_stroke_width_spin.blockSignals(True)
            self.default_stroke_width_spin.setValue(getattr(pcfg.global_fontformat, 'stroke_width', 0) or 0)
            self.default_stroke_width_spin.blockSignals(False)
        if hasattr(self, 'default_stroke_color_btn'):
            self._update_default_stroke_color_button()
        self.let_alignment_combox.setCurrentIndex(pcfg.let_alignment_flag)
        self.let_family_combox.setCurrentIndex(pcfg.let_family_flag)
        self.let_writing_mode_combox.setCurrentIndex(pcfg.let_writing_mode_flag)
        self.let_autolayout_checker.setChecked(pcfg.let_autolayout_flag)
        if hasattr(self, 'layout_constrain_to_bubble_checker'):
            self.layout_constrain_to_bubble_checker.setChecked(getattr(pcfg.module, 'layout_constrain_to_bubble', True))
        if hasattr(self, 'layout_center_in_bubble_after_autolayout_checker'):
            self.layout_center_in_bubble_after_autolayout_checker.setChecked(getattr(pcfg.module, 'layout_center_in_bubble_after_autolayout', True))
        if hasattr(self, 'layout_center_in_bubble_min_gap_spin'):
            self.layout_center_in_bubble_min_gap_spin.blockSignals(True)
            self.layout_center_in_bubble_min_gap_spin.setValue(float(getattr(pcfg.module, 'layout_center_in_bubble_min_gap_px', 40.0)))
            self.layout_center_in_bubble_min_gap_spin.blockSignals(False)
        if hasattr(self, 'layout_check_overflow_after_layout_checker'):
            self.layout_check_overflow_after_layout_checker.setChecked(getattr(pcfg.module, 'layout_check_overflow_after_layout', True))
        if hasattr(self, 'layout_box_size_check_model_id_edit'):
            self.layout_box_size_check_model_id_edit.blockSignals(True)
            self.layout_box_size_check_model_id_edit.setText(getattr(pcfg.module, 'layout_box_size_check_model_id', '') or '')
            self.layout_box_size_check_model_id_edit.blockSignals(False)
        if hasattr(self, 'layout_optimal_breaks_checker'):
            self.layout_optimal_breaks_checker.setChecked(getattr(pcfg.module, 'layout_optimal_breaks', True))
        if hasattr(self, 'layout_hyphenation_checker'):
            self.layout_hyphenation_checker.setChecked(getattr(pcfg.module, 'layout_hyphenation', True))
        if hasattr(self, 'optimize_line_breaks_checker'):
            self.optimize_line_breaks_checker.setChecked(getattr(pcfg.module, 'optimize_line_breaks', False))
        if hasattr(self, 'layout_short_line_penalty_spin'):
            self.layout_short_line_penalty_spin.blockSignals(True)
            self.layout_short_line_penalty_spin.setValue(float(getattr(pcfg.module, 'layout_short_line_penalty', 80.0)))
            self.layout_short_line_penalty_spin.blockSignals(False)
        if hasattr(self, 'layout_height_overflow_penalty_spin'):
            self.layout_height_overflow_penalty_spin.blockSignals(True)
            self.layout_height_overflow_penalty_spin.setValue(float(getattr(pcfg.module, 'layout_height_overflow_penalty', 360.0)))
            self.layout_height_overflow_penalty_spin.blockSignals(False)
        if hasattr(self, 'layout_font_size_min_spin'):
            self.layout_font_size_min_spin.blockSignals(True)
            self.layout_font_size_min_spin.setValue(float(getattr(pcfg.module, 'layout_font_size_min', 8.0)))
            self.layout_font_size_min_spin.blockSignals(False)
        if hasattr(self, 'layout_font_size_max_spin'):
            self.layout_font_size_max_spin.blockSignals(True)
            self.layout_font_size_max_spin.setValue(float(getattr(pcfg.module, 'layout_font_size_max', 72.0)))
            self.layout_font_size_max_spin.blockSignals(False)
        if hasattr(self, 'layout_font_fit_bubble_checker'):
            self.layout_font_fit_bubble_checker.setChecked(bool(getattr(pcfg.module, 'layout_font_fit_bubble', True)))
        if hasattr(self, 'layout_font_binary_search_checker'):
            self.layout_font_binary_search_checker.setChecked(bool(getattr(pcfg.module, 'layout_font_binary_search', False)))
        if hasattr(self, 'layout_balloon_shape_combo'):
            shape = (getattr(pcfg.module, 'layout_balloon_shape', 'auto') or 'auto').lower()
            idx = {"auto": 0, "round": 1, "elongated": 2, "narrow": 3, "diamond": 4, "square": 5, "bevel": 6, "pentagon": 7, "point": 8}.get(shape, 0)
            self.layout_balloon_shape_combo.blockSignals(True)
            self.layout_balloon_shape_combo.setCurrentIndex(idx)
            self.layout_balloon_shape_combo.blockSignals(False)
        if hasattr(self, 'layout_balloon_shape_auto_method_combo'):
            method = (getattr(pcfg.module, 'layout_balloon_shape_auto_method', 'model_contour') or 'model_contour').lower()
            idx = {"aspect_ratio": 0, "contour": 1, "model": 2, "model_contour": 3, "model_ratio": 4, "contour_ratio": 5, "model_contour_ratio": 6}.get(method, 3)
            self.layout_balloon_shape_auto_method_combo.blockSignals(True)
            self.layout_balloon_shape_auto_method_combo.setCurrentIndex(idx)
            self.layout_balloon_shape_auto_method_combo.blockSignals(False)
        if hasattr(self, 'layout_balloon_shape_model_id_edit'):
            self.layout_balloon_shape_model_id_edit.blockSignals(True)
            self.layout_balloon_shape_model_id_edit.setText(getattr(pcfg.module, 'layout_balloon_shape_model_id', '') or '')
            self.layout_balloon_shape_model_id_edit.blockSignals(False)
        if hasattr(self, 'layout_min_line_width_spin'):
            self.layout_min_line_width_spin.blockSignals(True)
            self.layout_min_line_width_spin.setValue(int(float(getattr(pcfg.module, 'layout_min_line_width_px', 80.0))))
            self.layout_min_line_width_spin.blockSignals(False)
        if hasattr(self, 'layout_max_line_width_frac_no_bubble_spin'):
            self.layout_max_line_width_frac_no_bubble_spin.blockSignals(True)
            self.layout_max_line_width_frac_no_bubble_spin.setValue(float(getattr(pcfg.module, 'layout_max_line_width_frac_no_bubble', 0.78)))
            self.layout_max_line_width_frac_no_bubble_spin.blockSignals(False)
        if hasattr(self, 'layout_stub_penalty_1word_spin'):
            self.layout_stub_penalty_1word_spin.blockSignals(True)
            self.layout_stub_penalty_1word_spin.setValue(float(getattr(pcfg.module, 'layout_stub_penalty_1word', 2000.0)))
            self.layout_stub_penalty_1word_spin.blockSignals(False)
        if hasattr(self, 'layout_panel_preserve_line_breaks_checker'):
            self.layout_panel_preserve_line_breaks_checker.setChecked(bool(getattr(pcfg.module, 'layout_panel_preserve_line_breaks', False)))
        # Keep Text in box dropdown in sync (Auto fit = decide by program + Auto layout)
        if pcfg.let_fntsize_flag == 0 and pcfg.let_autolayout_flag:
            self.text_box_format_combox.setCurrentIndex(0)
        else:
            self.text_box_format_combox.setCurrentIndex(1)
        self.selectext_minimenu_checker.setChecked(pcfg.textselect_mini_menu)
        self.let_uppercase_checker.setChecked(pcfg.let_uppercase_flag)
        self.let_textstyle_indep_checker.setChecked(pcfg.let_textstyle_indep_flag)
        self.saladict_shortcut.setKeySequence(pcfg.saladict_shortcut)
        self.searchurl_combobox.setCurrentText(pcfg.search_url)
        self.ocr_config_panel.restoreEmptyOCRChecker.setChecked(pcfg.restore_ocr_empty)
        self.rst_imgformat_combobox.setCurrentText(pcfg.imgsave_ext.replace('.', '').upper())
        self.intermediate_imgformat_combobox.setCurrentText(pcfg.intermediate_imgsave_ext.replace('.', '').upper())
        self.rst_imgquality_edit.setText(str(pcfg.imgsave_quality))
        self.rst_webp_lossless_checker.setChecked(getattr(pcfg, 'imgsave_webp_lossless', False))
        self._update_webp_lossless_visibility()
        self.load_model_checker.setChecked(pcfg.module.load_model_on_demand)
        self.empty_runcache_checker.setChecked(pcfg.module.empty_runcache)
        if hasattr(self, 'release_caches_after_batch_checker'):
            self.release_caches_after_batch_checker.setChecked(getattr(pcfg, 'release_caches_after_batch', False))
        if hasattr(self, 'ocr_cache_enabled_checker'):
            self.ocr_cache_enabled_checker.setChecked(getattr(pcfg.module, 'ocr_cache_enabled', True))
        if hasattr(self, 'ocr_auto_by_language_checker'):
            self.ocr_auto_by_language_checker.setChecked(getattr(pcfg.module, 'ocr_auto_by_language', False))
        if hasattr(self, 'show_module_tier_badges_checker'):
            self.show_module_tier_badges_checker.setChecked(getattr(pcfg, 'show_module_tier_badges_in_tooltips', True))
        if hasattr(self, 'skip_already_translated_checker'):
            self.skip_already_translated_checker.setChecked(getattr(pcfg.module, 'skip_already_translated', False))
        if hasattr(self, 'merge_nearby_blocks_checker'):
            self.merge_nearby_blocks_checker.setChecked(getattr(pcfg.module, 'merge_nearby_blocks_collision', False))
        if hasattr(self, 'merge_nearby_blocks_gap_spin'):
            self.merge_nearby_blocks_gap_spin.setValue(float(getattr(pcfg.module, 'merge_nearby_blocks_gap_ratio', 1.5)))
        if hasattr(self, 'merge_nearby_blocks_min_blocks_spin'):
            self.merge_nearby_blocks_min_blocks_spin.setValue(int(getattr(pcfg.module, 'merge_nearby_blocks_min_blocks', 18)))
        if hasattr(self, 'image_upscale_initial_checker'):
            self.image_upscale_initial_checker.setChecked(getattr(pcfg.module, 'image_upscale_initial', False))
        if hasattr(self, 'image_upscale_initial_factor_spin'):
            self.image_upscale_initial_factor_spin.setValue(float(getattr(pcfg.module, 'image_upscale_initial_factor', 2.0)))
        if hasattr(self, 'image_upscale_final_checker'):
            self.image_upscale_final_checker.setChecked(getattr(pcfg.module, 'image_upscale_final', False))
        if hasattr(self, 'image_upscale_final_factor_spin'):
            self.image_upscale_final_factor_spin.setValue(float(getattr(pcfg.module, 'image_upscale_final_factor', 2.0)))
        if hasattr(self, 'processing_scale_checker'):
            self.processing_scale_checker.setChecked(getattr(pcfg.module, 'processing_scale_enabled', True))
        if hasattr(self, 'colorization_checker'):
            self.colorization_checker.setChecked(getattr(pcfg.module, 'enable_colorization', False))
        if hasattr(self, 'colorization_strength_spin'):
            self.colorization_strength_spin.setValue(float(getattr(pcfg.module, 'colorization_strength', 0.6)))
        if hasattr(self, 'colorization_backend_combobox'):
            backend = getattr(pcfg.module, 'colorization_backend', 'simple')
            if backend in ('simple', 'twilight', 'manga_soft'):
                idx = 0
            elif backend in ('manga_vibrant', 'magma', 'warm'):
                idx = 1
            else:
                idx = 2
            self.colorization_backend_combobox.blockSignals(True)
            self.colorization_backend_combobox.setCurrentIndex(idx)
            self.colorization_backend_combobox.blockSignals(False)
        if hasattr(self, 'ocr_upscale_min_side_spin'):
            self.ocr_upscale_min_side_spin.setValue(int(getattr(pcfg.module, 'ocr_upscale_min_side', 0)))
        if hasattr(self, 'inpaint_spill_after_spin'):
            self.inpaint_spill_after_spin.setValue(int(getattr(pcfg.module, 'inpaint_spill_to_disk_after_blocks', 0)))
        self.let_show_only_custom_fonts.setChecked(pcfg.let_show_only_custom_fonts_flag)
        self.recent_proj_list_max_spin.setValue(getattr(pcfg, 'recent_proj_list_max', 14))
        self.show_welcome_screen_checker.setChecked(getattr(pcfg, 'show_welcome_screen', True))
        self.dev_mode_checker.setChecked(getattr(pcfg, 'dev_mode', False))
        self.logical_dpi_spin.setValue(getattr(pcfg, 'logical_dpi', 0))
        self.confirm_before_run_checker.setChecked(getattr(pcfg, 'confirm_before_run', True))
        if hasattr(self, 'manual_mode_checker'):
            self.manual_mode_checker.setChecked(getattr(pcfg, 'manual_mode', False))
        if hasattr(self, 'skip_ignored_in_run_checker'):
            self.skip_ignored_in_run_checker.setChecked(getattr(pcfg, 'skip_ignored_in_run', True))
        if hasattr(self, 'smooth_scroll_spin'):
            self.smooth_scroll_spin.setValue(getattr(pcfg, 'smooth_scroll_duration_ms', 0))
            if hasattr(self, 'configContent') and hasattr(self.configContent, 'setSmoothScrollDuration'):
                self.configContent.setSmoothScrollDuration(getattr(pcfg, 'smooth_scroll_duration_ms', 0))
        if hasattr(self, 'motion_blur_on_scroll_checker'):
            self.motion_blur_on_scroll_checker.setChecked(getattr(pcfg, 'motion_blur_on_scroll', False))
            if hasattr(self, 'configContent') and hasattr(self.configContent, 'setMotionBlurOnScroll'):
                self.configContent.setMotionBlurOnScroll(getattr(pcfg, 'motion_blur_on_scroll', False))
        if hasattr(self, 'reduce_motion_checker'):
            self.reduce_motion_checker.setChecked(getattr(pcfg, 'reduce_motion', False))
        arm = getattr(pcfg, 'auto_region_merge_after_run', 'never')
        arm_idx = self.auto_region_merge_combobox.findData(arm)
        if arm_idx >= 0:
            self.auto_region_merge_combobox.blockSignals(True)
            self.auto_region_merge_combobox.setCurrentIndex(arm_idx)
            self.auto_region_merge_combobox.blockSignals(False)
        self.darkmode_checker.setChecked(pcfg.darkmode)
        self.bubbly_ui_checker.setChecked(getattr(pcfg, 'bubbly_ui', True))
        self.use_custom_cursor_checker.setChecked(bool(getattr(pcfg, 'use_custom_cursor', False)))
        self.custom_cursor_path_edit.blockSignals(True)
        self.custom_cursor_path_edit.setText(getattr(pcfg, 'custom_cursor_path', '') or '')
        self.custom_cursor_path_edit.blockSignals(False)
        self.display_lang_combobox.blockSignals(True)
        self.display_lang_combobox.setCurrentText(self._display_lang_to_label(getattr(pcfg, 'display_lang', C.DEFAULT_DISPLAY_LANG)))
        self.display_lang_combobox.blockSignals(False)
        self.config_font_scale_spin.setValue(getattr(pcfg, 'config_panel_font_scale', 1.0))
        self.ocr_spell_check_checker.setChecked(getattr(pcfg, 'ocr_spell_check', False))
        dd = getattr(pcfg, 'default_device', '') or ''
        if dd == '':
            self.default_device_combobox.setCurrentIndex(0)
        else:
            idx = self.default_device_combobox.findText(dd)
            if idx >= 0:
                self.default_device_combobox.setCurrentIndex(idx)
            else:
                self.default_device_combobox.setCurrentIndex(0)
        self.unload_after_idle_spin.setValue(getattr(pcfg, 'unload_after_idle_minutes', 0))

        self.blockSignals(False)
