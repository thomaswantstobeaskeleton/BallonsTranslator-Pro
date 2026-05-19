import copy
import sys
from typing import List

from qtpy.QtWidgets import QLineEdit, QSizePolicy, QHBoxLayout, QVBoxLayout, QFrame, QFontComboBox, QApplication, QPushButton, QLabel, QGroupBox, QCheckBox, QSlider, QComboBox, QDoubleSpinBox, QInputDialog, QFileDialog, QMessageBox
from qtpy.QtCore import Signal, Qt
from qtpy.QtGui import QFocusEvent, QMouseEvent, QTextCursor, QKeyEvent

from utils import shared
from utils import config as C
from utils.config import pcfg
from utils.fontformat import FontFormat, px2pt, LineSpacingType
from utils.text_rendering import MANGA_PRESETS, fit_font_size_to_box, manga_presets, merge_font_fallback_chain, missing_glyphs_after_fallback, normalize_fit_mode, normalize_writing_mode, normalize_line_break_strategy, preset_from_font_format, preset_id_from_label, plan_typography_cleanup, smart_fit_text_to_box
from utils.text_masking import masked_text_warnings, mask_effective_box
from .custom_widget import Widget, ColorPickerLabel, ClickableLabel, CheckableLabel, TextCheckerLabel, AlignmentChecker, QFontChecker, SizeComboBox, SizeControlLabel
from .custom_widget.flow_layout import FlowLayout
from .textitem import TextBlkItem
from .text_advanced_format import TextAdvancedFormatPanel
from .text_style_presets import TextStylePresetPanel
from . import funcmaps as FM
from . import shared_widget as SW


class LineEdit(QLineEdit):

    return_pressed_wochange = Signal()
    return_pressed = Signal()

    def __init__(self, content: str = None, parent = None):
        super().__init__(content, parent)
        self.textChanged.connect(self.on_text_changed)
        self._text_changed = False
        self.editingFinished.connect(self.on_editing_finished)
        # self.returnPressed.connect(self.on_return_pressed)

    def on_text_changed(self):
        self._text_changed = True

    def on_editing_finished(self):
        self._text_changed = False

    def focusOutEvent(self, e: QFocusEvent) -> None:
        self._text_changed = False
        return super().focusOutEvent(e)

    def keyPressEvent(self, e: QKeyEvent) -> None:
        super().keyPressEvent(e)
        if e.key() == Qt.Key.Key_Return:
            self.return_pressed.emit()
            if not self._text_changed:
                self.return_pressed_wochange.emit()


class IncrementalBtn(QPushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFixedSize(13, 13)


def _set_combo_tooltip_to_current(combo: QComboBox, prefix: str = ""):
    text = combo.currentText()
    combo.setToolTip((prefix + "\n" if prefix else "") + text)


def _make_format_group(title: str, widgets: list, parent=None, tooltip: str = None) -> QGroupBox:
    group = QGroupBox(title, parent)
    group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
    if tooltip:
        group.setToolTip(tooltip)
    layout = FlowLayout(group, isTight=True)
    layout.setContentsMargins(8, 8, 8, 8)
    layout.setHorizontalSpacing(8)
    layout.setVerticalSpacing(6)
    for widget in widgets:
        widget.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        layout.addWidget(widget)
    return group


def _font_list_from_config(attr: str) -> List[str]:
    seen = set()
    out: List[str] = []
    for part in str(getattr(pcfg, attr, '') or '').split(','):
        family = part.strip()
        if family and family.lower() not in seen:
            seen.add(family.lower())
            out.append(family)
    return out


def _favorite_font_list() -> List[str]:
    return _font_list_from_config('render_favorite_fonts')


def _recent_font_list() -> List[str]:
    return _font_list_from_config('render_recent_fonts')


def _remember_recent_font(family: str):
    family = str(family or '').strip()
    if not family:
        return
    recent = [f for f in _recent_font_list() if f.lower() != family.lower()]
    recent.insert(0, family)
    pcfg.render_recent_fonts = ', '.join(recent[:16])
    C.save_config()


class AlignmentBtnGroup(QFrame):
    param_changed = Signal(str, int)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alignLeftChecker = AlignmentChecker(self)
        self.alignLeftChecker.clicked.connect(self.alignBtnPressed)
        self.alignCenterChecker = AlignmentChecker(self)
        self.alignCenterChecker.clicked.connect(self.alignBtnPressed)
        self.alignRightChecker = AlignmentChecker(self)
        self.alignRightChecker.clicked.connect(self.alignBtnPressed)
        self.alignLeftChecker.setObjectName("AlignLeftChecker")
        self.alignRightChecker.setObjectName("AlignRightChecker")
        self.alignCenterChecker.setObjectName("AlignCenterChecker")

        hlayout = QHBoxLayout(self)
        hlayout.addWidget(self.alignLeftChecker)
        hlayout.addWidget(self.alignCenterChecker)
        hlayout.addWidget(self.alignRightChecker)
        hlayout.setSpacing(2)

    def alignBtnPressed(self):
        btn = self.sender()
        if btn == self.alignLeftChecker:
            self.alignLeftChecker.setChecked(True)
            self.alignCenterChecker.setChecked(False)
            self.alignRightChecker.setChecked(False)
            self.param_changed.emit('alignment', 0)
        elif btn == self.alignRightChecker:
            self.alignRightChecker.setChecked(True)
            self.alignCenterChecker.setChecked(False)
            self.alignLeftChecker.setChecked(False)
            self.param_changed.emit('alignment', 2)
        else:
            self.alignCenterChecker.setChecked(True)
            self.alignLeftChecker.setChecked(False)
            self.alignRightChecker.setChecked(False)
            self.param_changed.emit('alignment', 1)
    
    def setAlignment(self, alignment: int):
        if alignment == 0:
            self.alignLeftChecker.setChecked(True)
            self.alignCenterChecker.setChecked(False)
            self.alignRightChecker.setChecked(False)
        elif alignment == 1:
            self.alignLeftChecker.setChecked(False)
            self.alignCenterChecker.setChecked(True)
            self.alignRightChecker.setChecked(False)
        else:
            self.alignLeftChecker.setChecked(False)
            self.alignCenterChecker.setChecked(False)
            self.alignRightChecker.setChecked(True)


class FormatGroupBtn(QFrame):
    param_changed = Signal(str, bool)
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.boldBtn = QFontChecker(self)
        self.boldBtn.setObjectName("FontBoldChecker")
        self.boldBtn.clicked.connect(self.setBold)
        self.italicBtn = QFontChecker(self)
        self.italicBtn.setObjectName("FontItalicChecker")
        self.italicBtn.clicked.connect(self.setItalic)
        self.underlineBtn = QFontChecker(self)
        self.underlineBtn.setObjectName("FontUnderlineChecker")
        self.underlineBtn.clicked.connect(self.setUnderline)
        self.strikethroughBtn = QFontChecker(self)
        self.strikethroughBtn.setObjectName("FontStrikethroughChecker")
        self.strikethroughBtn.clicked.connect(self.setStrikethrough)
        hlayout = QHBoxLayout(self)
        hlayout.addWidget(self.boldBtn)
        hlayout.addWidget(self.italicBtn)
        hlayout.addWidget(self.underlineBtn)
        hlayout.addWidget(self.strikethroughBtn)
        hlayout.setSpacing(8)

    def setBold(self):
        self.param_changed.emit('bold', self.boldBtn.isChecked())

    def setItalic(self):
        self.param_changed.emit('italic', self.italicBtn.isChecked())

    def setUnderline(self):
        self.param_changed.emit('underline', self.underlineBtn.isChecked())

    def setStrikethrough(self):
        self.param_changed.emit('strikethrough', self.strikethroughBtn.isChecked())
    

class FontSizeBox(QFrame):
    param_changed = Signal(str, float)
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.upBtn = IncrementalBtn(self)
        self.upBtn.setObjectName("FsizeIncrementUp")
        self.downBtn = IncrementalBtn(self)
        self.downBtn.setObjectName("FsizeIncrementDown")
        self.upBtn.clicked.connect(self.onUpBtnClicked)
        self.downBtn.clicked.connect(self.onDownBtnClicked)
        self.fcombobox = SizeComboBox([1, 1000], 'font_size', self)
        self.fcombobox.addItems([
            "5", "5.5", "6", "6.5", "7", "7.5", "8", "9", "10", "10.5",
            "11", "12", "14", "16", "18", "20", "22", "24", "26", "28",
            "30", "32", "34", "36", "40", "44", "48", "52", "56", "64", "72", "80", "93", "123", "163", "200"
        ])
        self.fcombobox.param_changed.connect(self.param_changed)

        hlayout = QHBoxLayout(self)
        vlayout = QVBoxLayout()
        vlayout.addWidget(self.upBtn)
        vlayout.addWidget(self.downBtn)
        vlayout.setContentsMargins(0, 0, 0, 0)
        vlayout.setSpacing(0)
        hlayout.addLayout(vlayout)
        hlayout.addWidget(self.fcombobox)
        hlayout.setSpacing(3)
        hlayout.setContentsMargins(0, 0, 0, 0)

    def getFontSize(self) -> str:
        return self.fcombobox.currentText()

    def onUpBtnClicked(self):
        raito = 1.25
        size = self.getFontSize()
        multi_size=False
        if "+" in size:
            size = size.strip("+")
            multi_size=True
        size = float(size)
        newsize = int(round(size * raito))
        if newsize == size:
            newsize += 1
        newsize = min(1000, newsize)
        if newsize != size:
            if not multi_size:
                self.param_changed.emit('font_size', newsize)
                self.fcombobox.setCurrentText(str(newsize))
            else:
                self.param_changed.emit('rel_font_size', raito)
                self.fcombobox.setCurrentText(str(newsize)+"+")

    def onDownBtnClicked(self):
        raito = 0.75
        size = self.getFontSize()
        multi_size=False
        if "+" in size:
            size = size.strip("+")
            multi_size=True
        size = float(size)
        newsize = int(round(size * raito))
        if newsize == size:
            newsize -= 1
        newsize = max(1, newsize)
        if newsize != size:
            if not multi_size:
                self.param_changed.emit('font_size', newsize)
                self.fcombobox.setCurrentText(str(newsize))
            else:
                self.param_changed.emit('rel_font_size', raito)
                self.fcombobox.setCurrentText(str(newsize)+"+")
    

class FontFamilyComboBox(QFontComboBox):
    param_changed = Signal(str, object)
    def __init__(self, emit_if_focused=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.currentFontChanged.connect(self.on_fontfamily_changed)
        self.lineedit = lineedit = LineEdit(parent=self)
        lineedit.return_pressed.connect(self.on_return_pressed)
        self.setLineEdit(lineedit)
        self.emit_if_focused = emit_if_focused
        self.return_pressed = False
        # #35: support long font names in dropdown
        self.setMinimumWidth(220)
        self.setMaxVisibleItems(20)
        
    def apply_fontfamily(self):
        ffamily = self.currentFont().family()
        if ffamily in shared.FONT_FAMILIES:
            _remember_recent_font(ffamily)
            self.param_changed.emit('font_family', ffamily)

    def update_font_list(self, font_list):
        self.currentFontChanged.disconnect(self.on_fontfamily_changed)
        current_font = self.currentFont().family()
        self.clear()
        self.addItems(font_list)
        self.addItems([current_font])
        self.setCurrentText(current_font)
        self.currentFontChanged.connect(self.on_fontfamily_changed)

    def on_return_pressed(self):
        self.return_pressed = True
        self.apply_fontfamily()

    def on_fontfamily_changed(self):
        if self.return_pressed:
            self.return_pressed = False
        else:
            self.apply_fontfamily()


class FontFormatPanel(Widget):
    
    apply_global_to_all_blocks_requested = Signal()
    set_default_format_requested = Signal()
    textblk_item: TextBlkItem = None
    text_cursor: QTextCursor = None
    global_format: FontFormat = None
    restoring_textblk: bool = False

    def __init__(self, app: QApplication, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.app = app

        self.vlayout = QVBoxLayout(self)
        self.vlayout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.familybox = FontFamilyComboBox(emit_if_focused=True, parent=self)
        self.familybox.setContentsMargins(0, 0, 0, 0)
        self.familybox.setObjectName("FontFamilyBox")
        self.familybox.setToolTip(self.tr("Font Family"))
        self.familybox.param_changed.connect(self.on_param_changed)
        self.familybox.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.favoriteFontCombo = QComboBox(self)
        self.favoriteFontCombo.setObjectName("FavoriteFontCombo")
        self.favoriteFontCombo.setMinimumWidth(170)
        self.favoriteFontCombo.setToolTip(self.tr("Favorite lettering fonts from Settings → Favorite lettering fonts. Choosing one applies it to the current selection/style."))
        self.favoriteFontCombo.currentIndexChanged.connect(self._on_favorite_font_selected)
        self.addFavoriteFontBtn = QPushButton(self.tr("★"), self)
        self.addFavoriteFontBtn.setObjectName("AddFavoriteFontButton")
        self.addFavoriteFontBtn.setToolTip(self.tr("Add the current font family to favorite lettering fonts."))
        self.addFavoriteFontBtn.setFixedWidth(32)
        self.addFavoriteFontBtn.clicked.connect(self._on_add_favorite_font_clicked)
        self._refresh_favorite_fonts()

        self.fontsizebox = FontSizeBox(self)
        self.fontsizebox.setToolTip(self.tr("Font Size"))
        self.fontsizebox.setObjectName("FontSizeBox")
        self.fontsizebox.fcombobox.setToolTip(self.tr("Change font size"))
        self.fontsizebox.param_changed.connect(self.on_param_changed)
        
        self.lineSpacingLabel = SizeControlLabel(self, direction=1, transparent_bg=False)
        self.lineSpacingLabel.setObjectName("lineSpacingLabel")
        self.lineSpacingLabel.size_ctrl_changed.connect(self.onLineSpacingCtrlChanged)
        self.lineSpacingLabel.btn_released.connect(lambda : self.on_param_changed('line_spacing', self.lineSpacingBox.value()))

        self.lineSpacingBox = SizeComboBox([0, 100], 'line_spacing', self)
        self.lineSpacingBox.addItems(["1.0", "1.1", "1.2"])
        self.lineSpacingBox.setToolTip(self.tr("Change line spacing"))
        self.lineSpacingBox.param_changed.connect(self.on_param_changed)
        
        self.colorPicker = ColorPickerLabel(self, param_name='frgb')
        self.colorPicker.setToolTip(self.tr("Change font color"))
        self.colorPicker.changingColor.connect(self.changingColor)
        self.colorPicker.colorChanged.connect(self.onColorLabelChanged)
        self.colorPicker.apply_color.connect(self.on_apply_color)

        self.alignBtnGroup = AlignmentBtnGroup(self)
        self.alignBtnGroup.param_changed.connect(self.on_param_changed)

        self.formatBtnGroup = FormatGroupBtn(self)
        self.formatBtnGroup.param_changed.connect(self.on_param_changed)

        self.verticalChecker = QFontChecker(self)
        self.verticalChecker.setObjectName("FontVerticalChecker")
        self.verticalChecker.clicked.connect(lambda : self.on_param_changed('vertical', self.verticalChecker.isChecked()))

        self.autoFitFontSizeChecker = QFontChecker(self)
        self.autoFitFontSizeChecker.setObjectName("AutoFitFontSizeChecker")
        self.autoFitFontSizeChecker.setToolTip(self.tr("Auto fit font size to block: scale font so text fits the bounding box when layout runs."))
        self.autoFitFontSizeChecker.clicked.connect(lambda : self.on_param_changed('auto_fit_font_size', self.autoFitFontSizeChecker.isChecked()))

        self.writingModeCombo = QComboBox(self)
        self.writingModeCombo.setObjectName("WritingModeCombo")
        self.writingModeCombo.addItem(self.tr("Writing: Auto"), "auto")
        self.writingModeCombo.addItem(self.tr("Horizontal LTR"), "horizontal_ltr")
        self.writingModeCombo.addItem(self.tr("Vertical RL"), "vertical_rl")
        self.writingModeCombo.addItem(self.tr("RTL"), "rtl")
        self.writingModeCombo.setMinimumWidth(150)
        self.writingModeCombo.setToolTip(self.tr("Current selection/style writing mode. Auto uses script and text-box geometry; project defaults live in Settings → Text Rendering Defaults."))
        self.writingModeCombo.currentIndexChanged.connect(self._on_writing_mode_changed)
        self.writingModeCombo.currentTextChanged.connect(lambda _t: _set_combo_tooltip_to_current(self.writingModeCombo, self.tr("Current selection/style writing mode.")))

        self.fitModeCombo = QComboBox(self)
        self.fitModeCombo.setObjectName("FitModeCombo")
        self.fitModeCombo.addItem(self.tr("Fit: Shrink"), "shrink")
        self.fitModeCombo.addItem(self.tr("Expand to fill"), "expand")
        self.fitModeCombo.addItem(self.tr("Preserve size"), "preserve")
        self.fitModeCombo.addItem(self.tr("Balance lines"), "balance")
        self.fitModeCombo.setMinimumWidth(140)
        self.fitModeCombo.setToolTip(self.tr("Current selection/style fitting policy used by auto-layout and layout review."))
        self.fitModeCombo.currentIndexChanged.connect(self._on_fit_mode_changed)
        self.fitModeCombo.currentTextChanged.connect(lambda _t: _set_combo_tooltip_to_current(self.fitModeCombo, self.tr("Current selection/style fitting policy.")))

        self.lineBreakCombo = QComboBox(self)
        self.lineBreakCombo.setObjectName("LineBreakStrategyCombo")
        self.lineBreakCombo.addItem(self.tr("Break: Auto"), "auto")
        self.lineBreakCombo.addItem(self.tr("Strict CJK"), "cjk_strict")
        self.lineBreakCombo.addItem(self.tr("Balanced"), "balanced")
        self.lineBreakCombo.addItem(self.tr("Loose SFX"), "loose")
        self.lineBreakCombo.setMinimumWidth(135)
        self.lineBreakCombo.setToolTip(self.tr("Current selection/style line-break strategy: CJK kinsoku, balanced dangling-line cleanup, or loose SFX wrapping."))
        self.lineBreakCombo.currentIndexChanged.connect(self._on_line_break_changed)
        self.lineBreakCombo.currentTextChanged.connect(lambda _t: _set_combo_tooltip_to_current(self.lineBreakCombo, self.tr("Current selection/style line-break strategy.")))

        self.mangaPresetCombo = QComboBox(self)
        self.mangaPresetCombo.setObjectName("MangaPresetCombo")
        self.mangaPresetCombo.setMinimumWidth(150)
        self.mangaPresetCombo.setToolTip(self.tr("Apply manga lettering presets to the current selection/style: stroke, spacing, writing mode, fit mode, alignment, and padding."))
        self.mangaPresetCombo.currentIndexChanged.connect(self._on_manga_preset_changed)
        self.mangaPresetCombo.currentTextChanged.connect(lambda _t: _set_combo_tooltip_to_current(self.mangaPresetCombo, self.tr("Manga lettering preset for current selection/style.")))
        self.saveMangaPresetBtn = QPushButton(self.tr("Save preset"), self)
        self.saveMangaPresetBtn.setToolTip(self.tr("Save the current style/textbox formatting as a reusable manga lettering preset."))
        self.saveMangaPresetBtn.clicked.connect(self._on_save_manga_preset_clicked)
        self.importMangaPresetsBtn = QPushButton(self.tr("Import presets"), self)
        self.importMangaPresetsBtn.setToolTip(self.tr("Import a shared manga lettering preset JSON pack."))
        self.importMangaPresetsBtn.clicked.connect(self._on_import_manga_presets_clicked)
        self.exportMangaPresetsBtn = QPushButton(self.tr("Export presets"), self)
        self.exportMangaPresetsBtn.setToolTip(self.tr("Export your custom manga lettering presets as a reusable JSON pack."))
        self.exportMangaPresetsBtn.clicked.connect(self._on_export_manga_presets_clicked)
        self._refresh_manga_presets()

        self.fallbackChainEdit = LineEdit(parent=self)
        self.fallbackChainEdit.setObjectName("FallbackFontChainEdit")
        self.fallbackChainEdit.setPlaceholderText(self.tr("Current style fallback fonts"))
        self.fallbackChainEdit.setMinimumWidth(190)
        self.fallbackChainEdit.setToolTip(self.tr("Current selection/style fallback fonts. Empty uses Settings → Text Rendering Defaults fallback chains."))
        self.fallbackChainEdit.editingFinished.connect(self._on_fallback_chain_changed)

        self.fontFallbackWarningLabel = QLabel(self)
        self.fontFallbackWarningLabel.setObjectName("FontFallbackWarningLabel")
        self.fontFallbackWarningLabel.setWordWrap(True)
        self.fontFallbackWarningLabel.setToolTip(self.tr("Shows missing-glyph diagnostics after configured fallback chains are considered."))

        self.formatScopeLabel = QLabel(self)
        self.formatScopeLabel.setObjectName("FormatScopeLabel")
        self.formatScopeLabel.setWordWrap(True)
        self.formatScopeLabel.setToolTip(self.tr("Shows whether the controls edit the global text style or the currently selected textbox/style override."))

        self.useProjectDefaultsBtn = QPushButton(self.tr("Use project text defaults"), self)
        self.useProjectDefaultsBtn.setToolTip(self.tr("Apply Settings → Project text rendering defaults to the current style/selection: writing mode, fit mode, line breaks, padding, and clear fallback override."))
        self.useProjectDefaultsBtn.clicked.connect(self._on_use_project_text_defaults)

        self.clearFallbackOverrideBtn = QPushButton(self.tr("Use global fallback fonts"), self)
        self.clearFallbackOverrideBtn.setToolTip(self.tr("Clear this style/textbox fallback chain so renderer per-script fallback fonts from Settings are used."))
        self.clearFallbackOverrideBtn.clicked.connect(self._on_clear_fallback_override)

        self.resetPathEffectsBtn = QPushButton(self.tr("Reset path/warp effects"), self)
        self.resetPathEffectsBtn.setToolTip(self.tr("Reset text-on-path, warp, and rounded-box effects for the current style/selection."))
        self.resetPathEffectsBtn.clicked.connect(self._on_reset_path_effects)

        self.letteringDiagnosticsLabel = QLabel(self)
        self.letteringDiagnosticsLabel.setObjectName("LetteringDiagnosticsLabel")
        self.letteringDiagnosticsLabel.setWordWrap(True)
        self.letteringDiagnosticsLabel.setToolTip(self.tr("Live estimate for the active text box/style: writing mode, fit result, overflow, fallback glyphs, and quality score."))
        self.applyDiagnosticsFixesBtn = QPushButton(self.tr("Apply diagnostics fixes"), self)
        self.applyDiagnosticsFixesBtn.setToolTip(self.tr("Apply safe typography polish and smart fit to the selected textbox: writing mode, punctuation, line breaks, padding, fallback chain, and font size."))
        self.applyDiagnosticsFixesBtn.clicked.connect(self._on_apply_diagnostics_fixes)

        self.textPaddingSpinBox = QDoubleSpinBox(self)
        self.textPaddingSpinBox.setObjectName("TextPaddingSpinBox")
        self.textPaddingSpinBox.setRange(0.0, 64.0)
        self.textPaddingSpinBox.setSingleStep(1.0)
        self.textPaddingSpinBox.setDecimals(1)
        self.textPaddingSpinBox.setSuffix(" px")
        self.textPaddingSpinBox.setToolTip(self.tr("Extra text inset/padding to avoid clipping strokes and punctuation."))
        self.textPaddingSpinBox.valueChanged.connect(self._on_text_padding_changed)

        self.fitMinFontSizeSpinBox = QDoubleSpinBox(self)
        self.fitMinFontSizeSpinBox.setObjectName("FitMinFontSizeSpinBox")
        self.fitMinFontSizeSpinBox.setRange(0.0, 200.0)
        self.fitMinFontSizeSpinBox.setSingleStep(1.0)
        self.fitMinFontSizeSpinBox.setDecimals(1)
        self.fitMinFontSizeSpinBox.setPrefix(self.tr("Min "))
        self.fitMinFontSizeSpinBox.setSuffix(" px")
        self.fitMinFontSizeSpinBox.setToolTip(self.tr("Current selection/style fit minimum font size. 0 = use project engine default."))
        self.fitMinFontSizeSpinBox.valueChanged.connect(self._on_fit_font_size_min_changed)

        self.fitMaxFontSizeSpinBox = QDoubleSpinBox(self)
        self.fitMaxFontSizeSpinBox.setObjectName("FitMaxFontSizeSpinBox")
        self.fitMaxFontSizeSpinBox.setRange(0.0, 300.0)
        self.fitMaxFontSizeSpinBox.setSingleStep(1.0)
        self.fitMaxFontSizeSpinBox.setDecimals(1)
        self.fitMaxFontSizeSpinBox.setPrefix(self.tr("Max "))
        self.fitMaxFontSizeSpinBox.setSuffix(" px")
        self.fitMaxFontSizeSpinBox.setToolTip(self.tr("Current selection/style fit maximum font size. 0 = use project engine default."))
        self.fitMaxFontSizeSpinBox.valueChanged.connect(self._on_fit_font_size_max_changed)

        self.textOnPathCombo = QComboBox(self)
        self.textOnPathCombo.setObjectName("TextOnPathCombo")
        self.textOnPathCombo.addItems([self.tr("Text on path: None"), self.tr("Circular"), self.tr("Arc")])
        self.textOnPathCombo.setMinimumWidth(150)
        self.textOnPathCombo.setToolTip(self.tr("Current selection/style: draw text along a circle or arc (e.g. for balloons, SFX)."))
        self.textOnPathCombo.currentIndexChanged.connect(self._on_text_on_path_combo_changed)

        self.textOnPathArcDegreesSpinBox = QDoubleSpinBox(self)
        self.textOnPathArcDegreesSpinBox.setObjectName("TextOnPathArcDegrees")
        self.textOnPathArcDegreesSpinBox.setRange(30, 360)
        self.textOnPathArcDegreesSpinBox.setValue(180)
        self.textOnPathArcDegreesSpinBox.setSuffix("°")
        self.textOnPathArcDegreesSpinBox.setToolTip(self.tr("Arc span in degrees (for Arc text on path)."))
        self.textOnPathArcDegreesSpinBox.setMinimumWidth(70)
        self.textOnPathArcDegreesSpinBox.valueChanged.connect(self._on_text_on_path_arc_degrees_changed)

        self.warpStyleCombo = QComboBox(self)
        self.warpStyleCombo.setObjectName("WarpStyleCombo")
        self.warpStyleCombo.addItems([
            self.tr("Warp: None"), self.tr("Arc"), self.tr("Arch"), self.tr("Bulge"), self.tr("Flag")
        ])
        self.warpStyleCombo.setMinimumWidth(125)
        self.warpStyleCombo.setToolTip(self.tr("Current selection/style Photoshop-like text warp (#1093)."))
        self.warpStyleCombo.currentIndexChanged.connect(self._on_warp_style_changed)

        self.warpStrengthSpinBox = QDoubleSpinBox(self)
        self.warpStrengthSpinBox.setObjectName("WarpStrength")
        self.warpStrengthSpinBox.setRange(0.1, 1.0)
        self.warpStrengthSpinBox.setSingleStep(0.1)
        self.warpStrengthSpinBox.setValue(0.5)
        self.warpStrengthSpinBox.setToolTip(self.tr("Warp intensity."))
        self.warpStrengthSpinBox.setMinimumWidth(55)
        self.warpStrengthSpinBox.valueChanged.connect(self._on_warp_strength_changed)

        self.boxCornerRadiusSpinBox = QDoubleSpinBox(self)
        self.boxCornerRadiusSpinBox.setObjectName("BoxCornerRadius")
        self.boxCornerRadiusSpinBox.setRange(0, 50)
        self.boxCornerRadiusSpinBox.setSingleStep(2)
        self.boxCornerRadiusSpinBox.setDecimals(1)
        self.boxCornerRadiusSpinBox.setValue(0)
        self.boxCornerRadiusSpinBox.setSuffix(" px")
        self.boxCornerRadiusSpinBox.setToolTip(self.tr("Rounded text box corners. 0 = sharp (rectangle)."))
        self.boxCornerRadiusSpinBox.setMinimumWidth(60)
        self.boxCornerRadiusSpinBox.valueChanged.connect(self._on_box_corner_radius_changed)

        self.strokeWidthBox = SizeComboBox([0, 10], 'stroke_width', self)
        self.strokeWidthBox.addItems(["0.1"])
        self.strokeWidthBox.setToolTip(self.tr("Change stroke width"))
        self.strokeWidthBox.param_changed.connect(self.on_param_changed)

        self.fontStrokeLabel = SizeControlLabel(self, 0, self.tr("Stroke"))
        self.fontStrokeLabel.setObjectName("fontStrokeLabel")
        font = self.fontStrokeLabel.font()
        font.setPointSizeF(max(1.0, shared.CONFIG_FONTSIZE_CONTENT * 0.95))
        self.fontStrokeLabel.setFont(font)
        self.fontStrokeLabel.size_ctrl_changed.connect(self.strokeWidthBox.changeByDelta)
        self.fontStrokeLabel.btn_released.connect(lambda : self.on_param_changed('stroke_width', self.strokeWidthBox.value()))
        
        self.strokeColorPicker = ColorPickerLabel(self, param_name='srgb')
        self.strokeColorPicker.setToolTip(self.tr("Change stroke color"))
        self.strokeColorPicker.changingColor.connect(self.changingColor)
        self.strokeColorPicker.colorChanged.connect(self.onColorLabelChanged)
        self.strokeColorPicker.apply_color.connect(self.on_apply_color)

        stroke_hlayout = QHBoxLayout()
        stroke_hlayout.addWidget(self.fontStrokeLabel)
        stroke_hlayout.addWidget(self.strokeWidthBox)
        stroke_hlayout.addWidget(self.strokeColorPicker)
        stroke_hlayout.setSpacing(shared.WIDGET_SPACING_CLOSE)

        self.secondaryStrokeWidthBox = SizeComboBox([0, 10], 'secondary_stroke_width', self)
        self.secondaryStrokeWidthBox.addItems(["0.0", "0.12", "0.18", "0.25"])
        self.secondaryStrokeWidthBox.setToolTip(self.tr("Back/second outline width for manga SFX and high-contrast lettering"))
        self.secondaryStrokeWidthBox.param_changed.connect(self.on_param_changed)
        self.secondaryStrokeLabel = SizeControlLabel(self, 0, self.tr("Back stroke"))
        self.secondaryStrokeLabel.setObjectName("secondaryStrokeLabel")
        font = self.secondaryStrokeLabel.font()
        font.setPointSizeF(max(1.0, shared.CONFIG_FONTSIZE_CONTENT * 0.95))
        self.secondaryStrokeLabel.setFont(font)
        self.secondaryStrokeLabel.size_ctrl_changed.connect(self.secondaryStrokeWidthBox.changeByDelta)
        self.secondaryStrokeLabel.btn_released.connect(lambda : self.on_param_changed('secondary_stroke_width', self.secondaryStrokeWidthBox.value()))
        self.secondaryStrokeColorPicker = ColorPickerLabel(self, param_name='secondary_srgb')
        self.secondaryStrokeColorPicker.setToolTip(self.tr("Change back/second outline color"))
        self.secondaryStrokeColorPicker.changingColor.connect(self.changingColor)
        self.secondaryStrokeColorPicker.colorChanged.connect(self.onColorLabelChanged)
        self.secondaryStrokeColorPicker.apply_color.connect(self.on_apply_color)
        secondary_stroke_hlayout = QHBoxLayout()
        secondary_stroke_hlayout.addWidget(self.secondaryStrokeLabel)
        secondary_stroke_hlayout.addWidget(self.secondaryStrokeWidthBox)
        secondary_stroke_hlayout.addWidget(self.secondaryStrokeColorPicker)
        secondary_stroke_hlayout.setSpacing(shared.WIDGET_SPACING_CLOSE)

        self.letterSpacingBox = SizeComboBox([0, 10], "letter_spacing", self)
        self.letterSpacingBox.addItems(["0.0"])
        self.letterSpacingBox.setToolTip(self.tr("Change letter spacing"))
        self.letterSpacingBox.setMinimumWidth(int(self.letterSpacingBox.height() * 2.5))
        self.letterSpacingBox.param_changed.connect(self.on_param_changed)

        self.letterSpacingLabel = SizeControlLabel(self, direction=0, transparent_bg=False)
        self.letterSpacingLabel.setObjectName("letterSpacingLabel")
        self.letterSpacingLabel.size_ctrl_changed.connect(self.letterSpacingBox.changeByDelta)
        self.letterSpacingLabel.btn_released.connect(lambda : self.on_param_changed('letter_spacing', self.letterSpacingBox.value()))

        lettersp_hlayout = QHBoxLayout()
        lettersp_hlayout.addWidget(self.letterSpacingLabel)
        lettersp_hlayout.addWidget(self.letterSpacingBox)
        lettersp_hlayout.setSpacing(shared.WIDGET_SPACING_CLOSE)

        self.opacityMainLabel = SizeControlLabel(self, direction=1, transparent_bg=False, text=self.tr('Opacity'))
        self.opacityMainLabel.setObjectName("opacityMainLabel")
        self.opacityMainLabel.setMinimumWidth(52)
        self.opacityMainBox = SizeComboBox([0, 1], 'opacity', self, init_value=1.)
        self.opacityMainBox.addItems(["0.5", "0.75", "1.0"])
        self.opacityMainBox.setToolTip(self.tr("Text opacity (quick access)"))
        self.opacityMainBox.param_changed.connect(self.on_param_changed)
        self.opacityMainLabel.size_ctrl_changed.connect(lambda x: self.opacityMainBox.changeByDelta(x, multiplier=0.02))
        self.opacityMainLabel.btn_released.connect(lambda : self.on_param_changed('opacity', self.opacityMainBox.value()))
        opacity_hlayout = QHBoxLayout()
        opacity_hlayout.addWidget(self.opacityMainLabel)
        opacity_hlayout.addWidget(self.opacityMainBox)
        opacity_hlayout.setSpacing(shared.WIDGET_SPACING_CLOSE)
        
        self.global_fontfmt_str = self.tr("Global Font Format")
        self.textstyle_panel = TextStylePresetPanel(
            self.global_fontfmt_str,
            config_name='show_text_style_preset',
            config_expand_name='expand_tstyle_panel'
        )
        self.textstyle_panel.active_text_style_label_changed.connect(self.on_active_textstyle_label_changed)
        self.textstyle_panel.active_stylename_edited.connect(self.on_active_stylename_edited)

        self.applyToAllBlocksBtn = QPushButton(self.tr("Apply to all blocks"))
        self.applyToAllBlocksBtn.setToolTip(self.tr("Apply current global font format to every text block on this page."))
        self.applyToAllBlocksBtn.clicked.connect(self.apply_global_to_all_blocks_requested.emit)
        self.saveAsDefaultBtn = QPushButton(self.tr("Save as default"))
        self.saveAsDefaultBtn.setToolTip(self.tr("Save current global font format as the default for new projects and sessions."))
        self.saveAsDefaultBtn.clicked.connect(self.set_default_format_requested.emit)

        self.showGlobalFontFormatBtn = QPushButton(self.tr("Show Global Font Format"))
        self.showGlobalFontFormatBtn.setToolTip(self.tr("Show the Global Font Format section (style presets) again."))
        self.showGlobalFontFormatBtn.clicked.connect(self._on_show_global_font_format_clicked)
        self.showGlobalFontFormatBtn.setVisible(not getattr(pcfg, "show_text_style_preset", True))

        self.showAdvancedFontFormatBtn = QPushButton(self.tr("Show Advanced Font Format"))
        self.showAdvancedFontFormatBtn.setToolTip(self.tr("Show the Advanced Text Format section again."))
        self.showAdvancedFontFormatBtn.clicked.connect(self._on_show_advanced_font_format_clicked)
        self.showAdvancedFontFormatBtn.setVisible(not getattr(pcfg, "text_advanced_format_panel", True))

        self.textadvancedfmt_panel = TextAdvancedFormatPanel(
            self.tr('Advanced Text Format'),
            config_name='text_advanced_format_panel',
            config_expand_name='expand_tadvanced_panel',
            on_format_changed=self.on_param_changed
        )
        color_label = self.textadvancedfmt_panel.shadow_group.color_label
        color_label.changingColor.connect(self.changingColor)
        color_label.colorChanged.connect(self.onColorLabelChanged)
        color_label.apply_color.connect(self.on_apply_color)

        color_label = self.textadvancedfmt_panel.gradient_group.start_picker
        color_label.changingColor.connect(self.changingColor)
        color_label.colorChanged.connect(self.onColorLabelChanged)
        color_label.apply_color.connect(self.on_apply_color)
        
        color_label = self.textadvancedfmt_panel.gradient_group.end_picker
        color_label.changingColor.connect(self.changingColor)
        color_label.colorChanged.connect(self.onColorLabelChanged)
        color_label.apply_color.connect(self.on_apply_color)
        
        self.foldTextBtn = CheckableLabel(self.tr("Unfold"), self.tr("Fold"), False)
        self.sourceBtn = TextCheckerLabel(self.tr("Source"))
        self.transBtn = TextCheckerLabel(self.tr("Translation"))

        FONTFORMAT_SPACING = 10

        vl0 = QVBoxLayout()
        vl0.addWidget(self.showGlobalFontFormatBtn)
        vl0.addWidget(self.textstyle_panel.view_widget)
        vl0.addWidget(self.applyToAllBlocksBtn)
        vl0.addWidget(self.saveAsDefaultBtn)
        vl0.addWidget(self.showAdvancedFontFormatBtn)
        vl0.addWidget(self.textadvancedfmt_panel.view_widget)
        vl0.setSpacing(0)
        vl0.setContentsMargins(0, 0, 0, 0)
        hl1 = QHBoxLayout()
        hl1.addWidget(self.familybox)
        hl1.addWidget(self.favoriteFontCombo)
        hl1.addWidget(self.addFavoriteFontBtn)
        hl1.addWidget(self.fontsizebox)
        hl1.addWidget(self.lineSpacingLabel)
        hl1.addWidget(self.lineSpacingBox)
        hl1.setSpacing(4)
        hl1.setContentsMargins(0, 12, 0, 0)
        hl2 = QHBoxLayout()
        hl2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hl2.addWidget(self.colorPicker)
        hl2.addWidget(self.alignBtnGroup)
        hl2.addWidget(self.formatBtnGroup)
        hl2.addWidget(self.verticalChecker)
        hl2.addWidget(self.autoFitFontSizeChecker)
        hl2.setSpacing(FONTFORMAT_SPACING)
        hl2.setContentsMargins(4, 0, 4, 0)
        layout_group = _make_format_group(
            self.tr("Selection layout overrides"),
            [self.writingModeCombo, self.fitModeCombo, self.lineBreakCombo, self.mangaPresetCombo, self.saveMangaPresetBtn, self.importMangaPresetsBtn, self.exportMangaPresetsBtn, self.textPaddingSpinBox, self.fitMinFontSizeSpinBox, self.fitMaxFontSizeSpinBox],
            self,
            self.tr("Current style/textbox controls. Project defaults are in Settings → Project text rendering defaults.")
        )
        fallback_group = _make_format_group(
            self.tr("Selection font fallback"),
            [self.fallbackChainEdit, self.clearFallbackOverrideBtn],
            self,
            self.tr("Empty fallback chain means use Settings fallback fonts.")
        )
        effects_group = _make_format_group(
            self.tr("Selection shape/path effects"),
            [self.textOnPathCombo, self.textOnPathArcDegreesSpinBox, self.warpStyleCombo, self.warpStrengthSpinBox, self.boxCornerRadiusSpinBox, self.resetPathEffectsBtn],
            self,
            self.tr("Optional decorative effects for the current style/textbox.")
        )
        defaults_group = _make_format_group(
            self.tr("Selection defaults / reset"),
            [self.useProjectDefaultsBtn],
            self,
            self.tr("Apply project text defaults without changing font family, size, colors, or stroke.")
        )
        hl2b = QVBoxLayout()
        hl2b.addWidget(self.formatScopeLabel)
        hl2b.addWidget(self.letteringDiagnosticsLabel)
        hl2b.addWidget(self.applyDiagnosticsFixesBtn)
        hl2b.addWidget(layout_group)
        hl2b.addWidget(fallback_group)
        hl2b.addWidget(effects_group)
        hl2b.addWidget(defaults_group)
        hl2b.setSpacing(6)
        hl2b.setContentsMargins(4, 4, 4, 0)
        hl3 = QHBoxLayout()
        hl3.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hl3.addLayout(stroke_hlayout)
        hl3.addLayout(secondary_stroke_hlayout)
        hl3.addLayout(lettersp_hlayout)
        hl3.addLayout(opacity_hlayout)
        hl3.addWidget(self.fontFallbackWarningLabel)
        hl3.setContentsMargins(6, 0, 6, 0)
        hl3.setSpacing(16)
        hl4 = QHBoxLayout()
        hl4.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hl4.addWidget(self.foldTextBtn)
        hl4.addWidget(self.sourceBtn)
        hl4.addWidget(self.transBtn)
        hl4.setStretch(0, 1)
        hl4.setStretch(1, 1)
        hl4.setStretch(2, 1)
        hl4.setContentsMargins(0, 12, 0, 0)
        hl4.setSpacing(0)

        # Main format controls first so they stay visible when sections above are collapsed/hidden
        self.vlayout.addLayout(hl1)
        self.vlayout.addLayout(hl2)
        self.vlayout.addLayout(hl2b)
        self.vlayout.addLayout(hl3)
        self.vlayout.addLayout(hl4)
        self.vlayout.addLayout(vl0)
        self.vlayout.setContentsMargins(0, 0, 7, 0)
        self.vlayout.setSpacing(0)

        # Ensure format controls remain visible when panel is narrow or sections are collapsed
        self.setMinimumHeight(360)
        self.setMinimumWidth(330)

        self.focusOnColorDialog = False
        C.active_format = self.global_format

    def _on_show_global_font_format_clicked(self):
        """Restore the Global Font Format (style presets) section when it was hidden."""
        pcfg.show_text_style_preset = True
        self.textstyle_panel.view_widget.setVisible(True)
        self.showGlobalFontFormatBtn.setVisible(False)
        if hasattr(shared, "config_name_to_view_widget") and "show_text_style_preset" in shared.config_name_to_view_widget:
            d = shared.config_name_to_view_widget["show_text_style_preset"]
            if "action" in d and d["action"] is not None:
                d["action"].setChecked(True)

    def _on_show_advanced_font_format_clicked(self):
        """Restore the Advanced Text Format section when it was hidden."""
        pcfg.text_advanced_format_panel = True
        self.textadvancedfmt_panel.view_widget.setVisible(True)
        self.showAdvancedFontFormatBtn.setVisible(False)
        if hasattr(shared, "config_name_to_view_widget") and "text_advanced_format_panel" in shared.config_name_to_view_widget:
            d = shared.config_name_to_view_widget["text_advanced_format_panel"]
            if "action" in d and d["action"] is not None:
                d["action"].setChecked(True)

    def global_mode(self):
        return id(C.active_format) == id(self.global_format)
    
    def active_text_style_label(self):
        return self.textstyle_panel.active_text_style_label

    def active_text_style_format(self):
        af = self.active_text_style_label()
        if af is not None:
            return af.fontfmt
        else:
            return None

    def _refresh_manga_presets(self):
        if not hasattr(self, 'mangaPresetCombo'):
            return
        current = self.mangaPresetCombo.currentData()
        self.mangaPresetCombo.blockSignals(True)
        self.mangaPresetCombo.clear()
        self.mangaPresetCombo.addItem(self.tr("Preset: none"), "")
        presets = manga_presets(pcfg)
        for preset_id, preset in presets.items():
            label = str(preset.get("label", preset_id) or preset_id)
            if preset_id.startswith('custom:'):
                label = self.tr('Custom: ') + label
            self.mangaPresetCombo.addItem(self.tr(label), preset_id)
        idx = self.mangaPresetCombo.findData(current) if current else 0
        self.mangaPresetCombo.setCurrentIndex(idx if idx >= 0 else 0)
        self.mangaPresetCombo.blockSignals(False)

    def _on_save_manga_preset_clicked(self):
        fmt = self._current_format_for_reset()
        if fmt is None:
            return
        default_name = getattr(fmt, '_style_name', '') or self.tr('Custom manga preset')
        name, ok = QInputDialog.getText(self, self.tr('Save manga lettering preset'), self.tr('Preset name:'), text=default_name)
        if not ok:
            return
        name = str(name or '').strip()
        if not name:
            return
        custom = dict(getattr(pcfg, 'render_custom_manga_presets', {}) or {})
        preset_id = preset_id_from_label(name, list(MANGA_PRESETS.keys()) + list(custom.keys()))
        custom[preset_id] = preset_from_font_format(fmt, label=name)
        pcfg.render_custom_manga_presets = custom
        C.save_config()
        self._refresh_manga_presets()
        self._set_combo_by_data(self.mangaPresetCombo, preset_id)
        self.on_param_changed('manga_preset', preset_id)

    def _on_export_manga_presets_clicked(self):
        from utils.rendering_preset_io import preset_font_diagnostics, write_preset_pack
        default_path = 'manga_lettering_presets.json'
        path, _ = QFileDialog.getSaveFileName(self, self.tr('Export manga lettering presets'), default_path, self.tr('Preset packs (*.json)'))
        if not path:
            return
        pack = write_preset_pack(pcfg, path)
        diagnostics = preset_font_diagnostics(pack.get('presets', {}), getattr(shared, 'FONT_FAMILIES', None) or [])
        msg = self.tr('Exported {0} custom preset(s) to:\n{1}').format(len(pack.get('presets', {}) or {}), pack.get('path', path))
        missing = diagnostics.get('missing', {}) if diagnostics.get('checked') else {}
        if missing:
            msg += '\n\n' + self.tr('Missing font families on this machine: ') + ', '.join(sorted(set(missing.values())))
        QMessageBox.information(self, self.tr('Export presets'), msg)

    def _on_import_manga_presets_clicked(self):
        from utils.rendering_preset_io import import_preset_pack
        path, _ = QFileDialog.getOpenFileName(self, self.tr('Import manga lettering presets'), '', self.tr('Preset packs (*.json)'))
        if not path:
            return
        result = import_preset_pack(pcfg, path, overwrite=False)
        C.save_config()
        self._refresh_manga_presets()
        QMessageBox.information(
            self,
            self.tr('Import presets'),
            self.tr('Imported {0} preset(s). Total custom presets: {1}.').format(result.get('imported_count', 0), result.get('total_custom_presets', 0))
        )

    def _refresh_favorite_fonts(self):
        if not hasattr(self, 'favoriteFontCombo'):
            return
        current = self.favoriteFontCombo.currentData()
        self.favoriteFontCombo.blockSignals(True)
        self.favoriteFontCombo.clear()
        self.favoriteFontCombo.addItem(self.tr('Favorites / recent'), '')
        favorites = _favorite_font_list()
        for family in favorites:
            self.favoriteFontCombo.addItem(self.tr('★ {0}').format(family), family)
        fav_lower = {f.lower() for f in favorites}
        for family in _recent_font_list():
            if family.lower() not in fav_lower:
                self.favoriteFontCombo.addItem(self.tr('Recent: {0}').format(family), family)
        idx = self.favoriteFontCombo.findData(current) if current else 0
        self.favoriteFontCombo.setCurrentIndex(idx if idx >= 0 else 0)
        self.favoriteFontCombo.blockSignals(False)

    def _on_favorite_font_selected(self, index: int):
        family = str(self.favoriteFontCombo.itemData(index) or '').strip()
        if not family:
            return
        self.familybox.setCurrentText(family)
        self.on_param_changed('font_family', family)
        self.favoriteFontCombo.blockSignals(True)
        self.favoriteFontCombo.setCurrentIndex(0)
        self.favoriteFontCombo.blockSignals(False)

    def _on_add_favorite_font_clicked(self):
        family = self.familybox.currentFont().family() or self.familybox.currentText()
        family = str(family or '').strip()
        if not family:
            return
        favorites = _favorite_font_list()
        if family.lower() not in {f.lower() for f in favorites}:
            favorites.insert(0, family)
            pcfg.render_favorite_fonts = ', '.join(favorites[:24])
            C.save_config()
        self._refresh_favorite_fonts()

    def on_param_changed(self, param_name: str, value):
        func = FM.handle_ffmt_change.get(param_name)
        func_kwargs = {}
        if param_name in {'font_size', 'rel_font_size'}:
            func_kwargs['clip_size'] = True
        if self.global_mode():
            func(param_name, value, self.global_format, is_global=True, **func_kwargs)
            self.update_text_style_label()
        else:
            func(param_name, value, C.active_format, is_global=False, blkitems=self.textblk_item, set_focus=True, **func_kwargs)

    def _set_combo_by_data(self, combo: QComboBox, value):
        idx = combo.findData(value)
        combo.setCurrentIndex(idx if idx >= 0 else 0)

    def _on_writing_mode_changed(self, index: int):
        self.on_param_changed('writing_mode', self.writingModeCombo.itemData(index) or 'auto')

    def _on_fit_mode_changed(self, index: int):
        self.on_param_changed('fit_mode', self.fitModeCombo.itemData(index) or 'shrink')

    def _on_line_break_changed(self, index: int):
        self.on_param_changed('line_break_strategy', self.lineBreakCombo.itemData(index) or 'auto')

    def _on_manga_preset_changed(self, index: int):
        preset = self.mangaPresetCombo.itemData(index) or ''
        if preset:
            self.on_param_changed('manga_preset', preset)

    def _on_fallback_chain_changed(self):
        self.on_param_changed('fallback_font_chain', self.fallbackChainEdit.text().strip())

    def _on_text_padding_changed(self, value: float):
        self.on_param_changed('text_padding', float(value))

    def _on_fit_font_size_min_changed(self, value: float):
        self.on_param_changed('fit_font_size_min', float(value))

    def _on_fit_font_size_max_changed(self, value: float):
        self.on_param_changed('fit_font_size_max', float(value))

    def _on_text_on_path_combo_changed(self, index: int):
        self.textOnPathArcDegreesSpinBox.setVisible(index == 2)
        self.on_param_changed('text_on_path', index)

    def _on_text_on_path_arc_degrees_changed(self, value: float):
        self.on_param_changed('text_on_path_arc_degrees', value)

    def _on_warp_style_changed(self, index: int):
        self.on_param_changed('warp_style', index)
        self.warpStrengthSpinBox.setVisible(index > 0)

    def _on_warp_strength_changed(self, value: float):
        self.on_param_changed('warp_strength', value)

    def _on_box_corner_radius_changed(self, value: float):
        self.on_param_changed('text_box_corner_radius', value)

    def _current_format_for_reset(self):
        return self.global_format if self.global_mode() else C.active_format

    def _apply_format_params(self, params):
        for param_name, value in params:
            self.on_param_changed(param_name, value)
        fmt = self._current_format_for_reset()
        if fmt is not None:
            self.set_active_format(fmt)

    def _on_use_project_text_defaults(self):
        self._apply_format_params([
            ('writing_mode', normalize_writing_mode(getattr(pcfg, 'render_default_writing_mode', 'auto'))),
            ('fit_mode', normalize_fit_mode(getattr(pcfg, 'render_default_fit_mode', 'shrink'))),
            ('line_break_strategy', normalize_line_break_strategy(getattr(pcfg, 'render_default_line_break_strategy', 'auto'))),
            ('text_padding', float(getattr(pcfg, 'render_default_text_padding', 0.0) or 0.0)),
            ('stroke_width', float(getattr(pcfg, 'render_default_stroke_width', 0.0) or 0.0)),
            ('secondary_stroke_width', float(getattr(pcfg, 'render_default_secondary_stroke_width', 0.0) or 0.0)),
            ('secondary_srgb', getattr(pcfg, 'render_default_secondary_stroke_color', [255, 255, 255]) or [255, 255, 255]),
            ('shadow_radius', float(getattr(pcfg, 'render_default_shadow_radius', 0.0) or 0.0)),
            ('fallback_font_chain', ''),
        ])

    def _on_clear_fallback_override(self):
        self._apply_format_params([('fallback_font_chain', '')])

    def _on_reset_path_effects(self):
        self._apply_format_params([
            ('text_on_path', 0),
            ('text_on_path_arc_degrees', 180.0),
            ('warp_style', 0),
            ('warp_strength', 0.5),
            ('text_box_corner_radius', 0.0),
        ])


    def _on_apply_diagnostics_fixes(self):
        if self.textblk_item is None:
            self.letteringDiagnosticsLabel.setText(self.tr("Select a textbox before applying diagnostics fixes."))
            return
        fmt = self._current_format_for_reset()
        if fmt is None:
            return
        try:
            text = self.textblk_item.toPlainText()
            rect = self.textblk_item.absBoundingRect(qrect=True)
            box = (float(rect.width()), float(rect.height()))
            cleanup = plan_typography_cleanup(
                text,
                float(getattr(fmt, 'font_size', 24.0) or 24.0),
                box,
                getattr(fmt, 'writing_mode', 'auto'),
                getattr(fmt, 'fit_mode', 'shrink'),
                getattr(fmt, 'line_break_strategy', 'auto'),
                line_spacing=float(getattr(fmt, 'line_spacing', 1.15) or 1.15),
                letter_spacing=float(getattr(fmt, 'letter_spacing', 1.0) or 1.0),
                text_padding=float(getattr(fmt, 'text_padding', 0.0) or 0.0),
                font_family=getattr(fmt, 'font_family', ''),
                fallback_font_chain=getattr(fmt, 'fallback_font_chain', ''),
                config_obj=pcfg,
            )
            if cleanup.text and cleanup.text != text:
                if hasattr(self.textblk_item, 'setPlainTextAndKeepUndoStack'):
                    self.textblk_item.setPlainTextAndKeepUndoStack(cleanup.text)
                else:
                    self.textblk_item.setPlainText(cleanup.text)
                if getattr(self.textblk_item, 'blk', None) is not None:
                    self.textblk_item.blk.translation = cleanup.text
                text = cleanup.text
            params = [
                ('writing_mode', cleanup.writing_mode),
                ('fit_mode', cleanup.fit_mode),
                ('line_break_strategy', cleanup.line_break_strategy),
                ('line_spacing', cleanup.line_spacing),
                ('letter_spacing', cleanup.letter_spacing),
                ('text_padding', cleanup.text_padding),
                ('fallback_font_chain', cleanup.fallback_font_chain),
            ]
            smart = smart_fit_text_to_box(
                text,
                float(getattr(fmt, 'font_size', 24.0) or 24.0),
                box,
                cleanup.writing_mode,
                cleanup.fit_mode,
                min_font_size=float(getattr(fmt, 'fit_font_size_min', 0.0) or getattr(getattr(pcfg, 'module', None), 'layout_font_size_min', 6.0)),
                max_font_size=float(getattr(fmt, 'fit_font_size_max', 0.0) or getattr(getattr(pcfg, 'module', None), 'layout_font_size_max', 96.0)),
                line_spacing=cleanup.line_spacing,
                letter_spacing=cleanup.letter_spacing,
                padding=cleanup.text_padding,
                stroke_width=float(getattr(fmt, 'stroke_width', 0.0) or 0.0),
                secondary_stroke_width=float(getattr(fmt, 'secondary_stroke_width', 0.0) or 0.0),
                line_break_strategy=cleanup.line_break_strategy,
                shadow_radius=float(getattr(fmt, 'shadow_radius', 0.0) or 0.0),
                shadow_offset=getattr(fmt, 'shadow_offset', [0.0, 0.0]) or [0.0, 0.0],
            )
            if smart.text and smart.text != text:
                if hasattr(self.textblk_item, 'setPlainTextAndKeepUndoStack'):
                    self.textblk_item.setPlainTextAndKeepUndoStack(smart.text)
                else:
                    self.textblk_item.setPlainText(smart.text)
                if getattr(self.textblk_item, 'blk', None) is not None:
                    self.textblk_item.blk.translation = smart.text
            cur_font_size = float(getattr(fmt, 'font_size', 24.0) or 24.0)
            target_font_size = float(smart.font_size or cur_font_size)
            # Guardrail: diagnostics "quick fix" should not nuke layout by extreme jumps in one click.
            min_step = max(6.0, cur_font_size * 0.65)
            max_step = min(120.0, cur_font_size * 1.35)
            clamped_font_size = max(min_step, min(max_step, target_font_size))
            if abs(clamped_font_size - cur_font_size) > 0.2:
                params.append(('font_size', float(clamped_font_size)))
                if abs(clamped_font_size - target_font_size) > 0.2:
                    (smart.actions or []).append(self.tr('font-size-clamped'))
            if smart.letter_spacing != getattr(fmt, 'letter_spacing', smart.letter_spacing):
                params.append(('letter_spacing', float(smart.letter_spacing)))
            if smart.line_spacing != getattr(fmt, 'line_spacing', smart.line_spacing):
                params.append(('line_spacing', float(smart.line_spacing)))
            self._apply_format_params(params)
            canvas = getattr(SW, 'canvas', None)
            if canvas is not None:
                canvas.setProjSaveState(True)
            self.letteringDiagnosticsLabel.setText(self.tr("Applied diagnostics fixes: {0} · font {1:.1f}px→{2:.1f}px").format(', '.join((cleanup.actions or []) + (smart.actions or [])) or self.tr('none'), cur_font_size, float(getattr(fmt, 'font_size', cur_font_size) or cur_font_size)))
        except Exception as exc:
            self.letteringDiagnosticsLabel.setText(self.tr("Diagnostics fix failed: {0}").format(exc))

    def _update_format_scope_label(self):
        if self.global_mode():
            self.formatScopeLabel.setText(self.tr("Editing Global Font Format / active style defaults. Use Apply to all blocks for a page-wide change."))
        else:
            self.formatScopeLabel.setText(self.tr("Editing the current selected textbox/style override. Project defaults remain unchanged."))

    def update_text_style_label(self):
        if self.global_mode():
            active_text_style_label = self.active_text_style_label()
            if active_text_style_label is not None:
                active_text_style_label.update_style(self.global_format)

    def changingColor(self):
        self.focusOnColorDialog = True

    def onColorLabelChanged(self, is_valid=True):
        self.focusOnColorDialog = False
        if is_valid:
            sender: ColorPickerLabel = self.sender()
            rgb = sender.rgb()
            self.on_param_changed(sender.param_name, rgb)

    def on_apply_color(self, param_name, rgb):
        self.on_param_changed(param_name, rgb)

    def onLineSpacingCtrlChanged(self, delta: int):
        if C.active_format.line_spacing_type == LineSpacingType.Distance:
            mul = 0.1
        else:
            mul = 0.01
        self.lineSpacingBox.setValue(self.lineSpacingBox.value() + delta * mul)


    def _update_lettering_diagnostics(self, font_format: FontFormat):
        text = ''
        box = (0.0, 0.0)
        if self.textblk_item is not None:
            try:
                text = self.textblk_item.toPlainText()
            except Exception:
                text = ''
            try:
                rect = self.textblk_item.absBoundingRect(qrect=True)
                box = (float(rect.width()), float(rect.height()))
            except Exception:
                box = (0.0, 0.0)
        if not text:
            self.letteringDiagnosticsLabel.setText(self.tr("Lettering diagnostics: select a textbox to see fit, fallback, and overflow status."))
            return
        try:
            fitted_size, _text_out, diag = fit_font_size_to_box(
                text,
                float(getattr(font_format, 'font_size', 24.0) or 24.0),
                box,
                getattr(font_format, 'fit_mode', 'shrink'),
                getattr(font_format, 'writing_mode', 'auto'),
                min_font_size=float(getattr(font_format, 'fit_font_size_min', 0.0) or getattr(getattr(pcfg, 'module', None), 'layout_font_size_min', 6.0)),
                max_font_size=float(getattr(font_format, 'fit_font_size_max', 0.0) or getattr(getattr(pcfg, 'module', None), 'layout_font_size_max', 96.0)),
                line_spacing=float(getattr(font_format, 'line_spacing', 1.15) or 1.15),
                letter_spacing=float(getattr(font_format, 'letter_spacing', 1.0) or 1.0),
                padding=float(getattr(font_format, 'text_padding', 0.0) or 0.0),
                stroke_width=float(getattr(font_format, 'stroke_width', 0.0) or 0.0),
                secondary_stroke_width=float(getattr(font_format, 'secondary_stroke_width', 0.0) or 0.0),
                line_break_strategy=getattr(font_format, 'line_break_strategy', 'auto'),
            )
            missing = missing_glyphs_after_fallback(getattr(font_format, 'font_family', ''), text, pcfg, getattr(font_format, 'fallback_font_chain', ''))
            mask = getattr(getattr(self.textblk_item, 'blk', None), 'text_mask', None)
            mask_diag = masked_text_warnings(mask, getattr(font_format, 'text_padding', 0.0))
            effective = mask_effective_box(mask, box, getattr(font_format, 'text_padding', 0.0))
            cleanup = plan_typography_cleanup(
                text,
                float(getattr(font_format, 'font_size', 24.0) or 24.0),
                box,
                getattr(font_format, 'writing_mode', 'auto'),
                getattr(font_format, 'fit_mode', 'shrink'),
                getattr(font_format, 'line_break_strategy', 'auto'),
                line_spacing=float(getattr(font_format, 'line_spacing', 1.15) or 1.15),
                letter_spacing=float(getattr(font_format, 'letter_spacing', 1.0) or 1.0),
                text_padding=float(getattr(font_format, 'text_padding', 0.0) or 0.0),
                font_family=getattr(font_format, 'font_family', ''),
                fallback_font_chain=getattr(font_format, 'fallback_font_chain', ''),
                config_obj=pcfg,
                balance=False,
            )
            cleanup_note = ','.join(cleanup.actions[:3]) if cleanup.actions else self.tr('none')
            line_quality = (diag.to_dict().get('line_break_quality') or {}) if hasattr(diag, 'to_dict') else {}
            line_note = self.tr('balanced')
            if line_quality.get('needs_balance'):
                line_note = self.tr('rebalance')
            elif line_quality.get('raggedness') is not None:
                line_note = self.tr('ragged {0:.2f}').format(float(line_quality.get('raggedness', 0.0) or 0.0))
            mask_note = self.tr('mask none')
            if mask_diag.get('warning'):
                mask_note = self.tr('mask {coverage:.0%}, safe {w:.0f}×{h:.0f}').format(
                    coverage=float(mask_diag.get('coverage', 0.0) or 0.0),
                    w=float(effective.get('width', box[0]) or box[0]),
                    h=float(effective.get('height', box[1]) or box[1]),
                )
            status = self.tr("OK") if not diag.overflow and not missing and 'safe' not in mask_note else self.tr("Needs review")
            self.letteringDiagnosticsLabel.setText(
                self.tr("Lettering diagnostics: {status} · mode {mode} · fit {fit:.1f}px · quality {quality:.2f} · lines {lines} · polish {polish} · missing {missing} · {mask}").format(
                    status=status,
                    mode=diag.resolved_writing_mode,
                    fit=fitted_size,
                    quality=getattr(diag, 'quality_score', 1.0),
                    lines=line_note,
                    polish=cleanup_note,
                    missing=''.join(missing) if missing else self.tr('none'),
                    mask=mask_note,
                )
            )
        except Exception as exc:
            self.letteringDiagnosticsLabel.setText(self.tr("Lettering diagnostics unavailable: {0}").format(exc))


    def _update_font_fallback_warning(self, font_format: FontFormat):
        text = ''
        if self.textblk_item is not None:
            try:
                text = self.textblk_item.toPlainText()
            except Exception:
                text = ''
        if not text:
            self.fontFallbackWarningLabel.setText('')
            return
        missing = missing_glyphs_after_fallback(
            getattr(font_format, 'font_family', ''), text, pcfg, getattr(font_format, 'fallback_font_chain', '')
        )
        chain = merge_font_fallback_chain(
            getattr(font_format, 'font_family', ''), text, pcfg, getattr(font_format, 'fallback_font_chain', '')
        )
        if missing:
            self.fontFallbackWarningLabel.setText(self.tr('Missing glyphs: ') + ''.join(missing))
            self.fontFallbackWarningLabel.setToolTip(self.tr('No configured font renders these characters. Chain: ') + ', '.join(chain))
        else:
            self.fontFallbackWarningLabel.setText(self.tr('Fallback OK') if len(chain) > 1 else '')
            self.fontFallbackWarningLabel.setToolTip(self.tr('Merged font chain: ') + ', '.join(chain))

    def set_active_format(self, font_format: FontFormat, multi_size=False):
        C.active_format = font_format
        self.familybox.blockSignals(True)
        font_size = round(font_format.font_size, 1)
        if int(font_size) == font_size:
            font_size = str(int(font_size))
        else:
            font_size = f'{font_size:.1f}'
        if multi_size:
            font_size += "+"
        self.fontsizebox.fcombobox.setCurrentText(font_size)
        self.familybox.setCurrentText(font_format.font_family)
        self._refresh_favorite_fonts()
        self.colorPicker.setPickerColor(font_format.foreground_color())
        self.strokeColorPicker.setPickerColor(font_format.stroke_color())
        self.strokeWidthBox.setValue(font_format.stroke_width)
        self.lineSpacingBox.setValue(font_format.line_spacing)
        self.letterSpacingBox.setValue(font_format.letter_spacing)
        self.verticalChecker.setChecked(font_format.vertical)
        self.writingModeCombo.blockSignals(True)
        self._set_combo_by_data(self.writingModeCombo, normalize_writing_mode(getattr(font_format, 'writing_mode', 'auto')))
        self.writingModeCombo.blockSignals(False)
        self.fitModeCombo.blockSignals(True)
        self._set_combo_by_data(self.fitModeCombo, normalize_fit_mode(getattr(font_format, 'fit_mode', 'shrink')))
        self.fitModeCombo.blockSignals(False)
        self.lineBreakCombo.blockSignals(True)
        self._set_combo_by_data(self.lineBreakCombo, normalize_line_break_strategy(getattr(font_format, 'line_break_strategy', 'auto')))
        self.lineBreakCombo.blockSignals(False)
        self._refresh_manga_presets()
        self.mangaPresetCombo.blockSignals(True)
        self._set_combo_by_data(self.mangaPresetCombo, getattr(font_format, 'manga_preset', '') or '')
        self.mangaPresetCombo.blockSignals(False)
        self.fallbackChainEdit.blockSignals(True)
        self.fallbackChainEdit.setText(str(getattr(font_format, 'fallback_font_chain', '') or ''))
        self.fallbackChainEdit.blockSignals(False)
        self._update_font_fallback_warning(font_format)
        self._update_lettering_diagnostics(font_format)
        self.textPaddingSpinBox.blockSignals(True)
        self.textPaddingSpinBox.setValue(float(getattr(font_format, 'text_padding', 0.0) or 0.0))
        self.textPaddingSpinBox.blockSignals(False)
        self.fitMinFontSizeSpinBox.blockSignals(True)
        self.fitMinFontSizeSpinBox.setValue(float(getattr(font_format, 'fit_font_size_min', 0.0) or 0.0))
        self.fitMinFontSizeSpinBox.blockSignals(False)
        self.fitMaxFontSizeSpinBox.blockSignals(True)
        self.fitMaxFontSizeSpinBox.setValue(float(getattr(font_format, 'fit_font_size_max', 0.0) or 0.0))
        self.fitMaxFontSizeSpinBox.blockSignals(False)
        self.autoFitFontSizeChecker.setChecked(bool(getattr(font_format, 'auto_fit_font_size', False)))
        self.opacityMainBox.setValue(font_format.opacity)
        text_on_path = getattr(font_format, 'text_on_path', 0)
        self.textOnPathCombo.blockSignals(True)
        self.textOnPathCombo.setCurrentIndex(min(2, max(0, int(text_on_path))))
        self.textOnPathCombo.blockSignals(False)
        self.textOnPathArcDegreesSpinBox.blockSignals(True)
        self.textOnPathArcDegreesSpinBox.setValue(float(getattr(font_format, 'text_on_path_arc_degrees', 180.0)))
        self.textOnPathArcDegreesSpinBox.blockSignals(False)
        self.textOnPathArcDegreesSpinBox.setVisible(text_on_path == 2)
        warp_style = getattr(font_format, 'warp_style', 0)
        self.warpStyleCombo.blockSignals(True)
        self.warpStyleCombo.setCurrentIndex(min(4, max(0, int(warp_style))))
        self.warpStyleCombo.blockSignals(False)
        self.warpStrengthSpinBox.blockSignals(True)
        self.warpStrengthSpinBox.setValue(float(getattr(font_format, 'warp_strength', 0.5)))
        self.warpStrengthSpinBox.blockSignals(False)
        self.warpStrengthSpinBox.setVisible(warp_style > 0)
        self.boxCornerRadiusSpinBox.blockSignals(True)
        self.boxCornerRadiusSpinBox.setValue(float(getattr(font_format, 'text_box_corner_radius', 0.0)))
        self.boxCornerRadiusSpinBox.blockSignals(False)
        self.formatBtnGroup.boldBtn.setChecked(font_format.bold)
        self.formatBtnGroup.underlineBtn.setChecked(font_format.underline)
        self.formatBtnGroup.strikethroughBtn.setChecked(font_format.strikethrough)
        self.formatBtnGroup.italicBtn.setChecked(font_format.italic)
        self.alignBtnGroup.setAlignment(font_format.alignment)
        self._update_format_scope_label()
        
        self.familybox.blockSignals(False)
        self.textadvancedfmt_panel.set_active_format(font_format)

    def set_globalfmt_title(self):
        active_text_style_label = self.active_text_style_label()
        if active_text_style_label is None:
            self.textstyle_panel.setTitle(self.global_fontfmt_str)
        else:
            title = self.global_fontfmt_str + ' - ' + active_text_style_label.fontfmt._style_name
            valid_title = self.textstyle_panel.elidedText(title)
            self.textstyle_panel.setTitle(valid_title)


    def deactivate_style_label(self):
        if self.active_text_style_label() is not None:
            self.textstyle_panel.on_stylelabel_activated(False)


    def on_active_textstyle_label_changed(self):
        '''
        merge activate textstyle into global format
        '''
        active_text_style_label = self.active_text_style_label()
        if active_text_style_label is not None:
            updated_keys = self.global_format.merge(active_text_style_label.fontfmt, compare=True)
            if self.global_mode() and len(updated_keys) > 0:
                self.set_active_format(self.global_format)
            self.set_globalfmt_title()
        else:
            if self.global_mode():
                self.set_globalfmt_title()

    def on_active_stylename_edited(self):
        if self.global_mode():
            self.set_globalfmt_title()

    def set_textblk_item(self, textblk_item: TextBlkItem = None, multi_select:bool=False):
        if textblk_item is None:
            focus_w = self.app.focusWidget()
            focus_p = None if focus_w is None else focus_w.parentWidget()
            focus_on_fmtoptions = False
            if self.focusOnColorDialog:
                focus_on_fmtoptions = True
            elif focus_p:
                if focus_p == self or focus_p.parentWidget() == self:
                    focus_on_fmtoptions = True
            if not focus_on_fmtoptions:
                # Store the current text block's format before switching to global
                if self.textblk_item is not None:
                    # Save all format properties including gradient state
                    self.textblk_item.fontformat = copy.deepcopy(C.active_format)
                self.textblk_item = None
                self.set_active_format(self.global_format, multi_select)
                self.set_globalfmt_title()
            
        else:
            if not self.restoring_textblk:
                blk_fmt = textblk_item.get_fontformat()
                # Preserve gradient properties from the text block's format
                if hasattr(textblk_item.fontformat, 'gradient_enabled'):
                    blk_fmt.gradient_enabled = textblk_item.fontformat.gradient_enabled
                    blk_fmt.gradient_start_color = textblk_item.fontformat.gradient_start_color
                    blk_fmt.gradient_end_color = textblk_item.fontformat.gradient_end_color
                    blk_fmt.gradient_angle = textblk_item.fontformat.gradient_angle
                    blk_fmt.gradient_size = textblk_item.fontformat.gradient_size
                self.textblk_item = textblk_item
                multi_size = not textblk_item.isEditing() and textblk_item.isMultiFontSize()
                self.set_active_format(blk_fmt, multi_size)
                self.textstyle_panel.setTitle(f'TextBlock #{textblk_item.idx}')
