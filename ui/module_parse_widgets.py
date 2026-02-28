from typing import List, Callable

from modules import GET_VALID_INPAINTERS, GET_VALID_TEXTDETECTORS, GET_VALID_TRANSLATORS, GET_VALID_OCR, \
    BaseTranslator, DEFAULT_DEVICE, GPUINTENSIVE_SET
from utils.logger import logger as LOGGER
from .custom_widget import ConfigComboBox, ParamComboBox, NoBorderPushBtn, ParamNameLabel
from utils.shared import CONFIG_COMBOBOX_LONG, size2width, CONFIG_COMBOBOX_SHORT, CONFIG_COMBOBOX_HEIGHT
from utils.config import pcfg

from qtpy.QtWidgets import QPlainTextEdit, QHBoxLayout, QVBoxLayout, QWidget, QLabel, QCheckBox, QLineEdit, QGridLayout, QPushButton, QMessageBox
from qtpy.QtCore import Qt, Signal, QTimer
from qtpy.QtGui import QDoubleValidator


class ParamCheckGroup(QWidget):

    paramwidget_edited = Signal(str, dict)

    def __init__(self, param_key, check_group: dict, parent=None) -> None:
        super().__init__(parent=parent)
        self.param_key = param_key
        layout = QHBoxLayout(self)
        self.label2widget = {}
        for k, v in check_group.items():
            checker = QCheckBox(text=k, parent=self)
            checker.setChecked(v)
            layout.addWidget(checker)
            self.label2widget[k] = checker
            checker.clicked.connect(self.on_checker_clicked)

    def on_checker_clicked(self):
        new_state_dict = {}
        w = QCheckBox()
        for k, w in self.label2widget.items():
            new_state_dict[k] = w.isChecked()
        self.paramwidget_edited.emit(self.param_key, new_state_dict)


class ParamLineEditor(QLineEdit):
    
    paramwidget_edited = Signal(str, str)
    def __init__(self, param_key: str, force_digital, size='short', *args, **kwargs) -> None:
        super().__init__( *args, **kwargs)
        self.param_key = param_key
        self.setFixedWidth(size2width(size))
        self.setFixedHeight(CONFIG_COMBOBOX_HEIGHT)
        self.textChanged.connect(self.on_text_changed)

        if force_digital:
            validator = QDoubleValidator()
            self.setValidator(validator)

    def on_text_changed(self):
        self.paramwidget_edited.emit(self.param_key, self.text())

class ParamEditor(QPlainTextEdit):
    
    paramwidget_edited = Signal(str, str)
    def __init__(self, param_key: str, *args, **kwargs) -> None:
        super().__init__( *args, **kwargs)
        self.param_key = param_key

        if param_key == 'chat sample':
            self.setFixedWidth(int(CONFIG_COMBOBOX_LONG * 1.2))
            self.setFixedHeight(200)
        else:
            self.setFixedWidth(CONFIG_COMBOBOX_LONG)
            self.setFixedHeight(100)
        # self.setFixedHeight(CONFIG_COMBOBOX_HEIGHT)
        self.textChanged.connect(self.on_text_changed)

    def on_text_changed(self):
        self.paramwidget_edited.emit(self.param_key, self.text())

    def setText(self, text: str):
        self.setPlainText(text)

    def text(self):
        return self.toPlainText()


class ParamCheckerBox(QWidget):
    checker_changed = Signal(bool)
    paramwidget_edited = Signal(str, str)
    def __init__(self, param_key: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_key = param_key
        self.checker = QCheckBox()
        name_label = ParamNameLabel(param_key)
        hlayout = QHBoxLayout(self)
        hlayout.addWidget(name_label)
        hlayout.addWidget(self.checker)
        hlayout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.checker.stateChanged.connect(self.on_checker_changed)

    def on_checker_changed(self):
        is_checked = self.checker.isChecked()
        self.checker_changed.emit(is_checked)
        checked = 'true' if is_checked else 'false'
        self.paramwidget_edited.emit(self.param_key, checked)


class ParamCheckBox(QCheckBox):
    paramwidget_edited = Signal(str, bool)
    def __init__(self, param_key: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_key = param_key
        self.stateChanged.connect(self.on_checker_changed)

    def on_checker_changed(self):
        self.paramwidget_edited.emit(self.param_key, self.isChecked())


def get_param_display_name(param_key: str, param_dict: dict = None):
    if param_dict is not None and isinstance(param_dict, dict):
        if 'display_name' in param_dict:
            return param_dict['display_name']
    return param_key


class ParamPushButton(QPushButton):
    paramwidget_edited = Signal(str, str)
    def __init__(self, param_key: str, param_dict: dict = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.param_key = param_key
        self.setText(get_param_display_name(param_key, param_dict))
        self.clicked.connect(self.on_clicked)

    def on_clicked(self):
        self.paramwidget_edited.emit(self.param_key, '')


class ParamWidget(QWidget):

    paramwidget_edited = Signal(str, dict)
    def __init__(self, params, scrollWidget: QWidget = None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        layout = QHBoxLayout(self)
        self.param_layout = param_layout = QGridLayout()
        param_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        param_layout.setContentsMargins(0, 0, 0, 0)
        param_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addLayout(param_layout)
        layout.addStretch(-1)

        if 'description' in params:
            self.setToolTip(params['description'])

        for ii, param_key in enumerate(params):
            if param_key == 'description' or param_key.startswith('__'):
                continue
            display_param_name = param_key

            require_label = True
            is_str = isinstance(params[param_key], str)
            is_digital = isinstance(params[param_key], float) or isinstance(params[param_key], int)
            param_widget = None

            if isinstance(params[param_key], bool):
                param_widget = ParamCheckBox(param_key)
                val = params[param_key]
                param_widget.setChecked(val)
                param_widget.paramwidget_edited.connect(self.on_paramwidget_edited)

            elif is_str or is_digital:
                param_widget = ParamLineEditor(param_key, force_digital=is_digital)
                val = params[param_key]
                if is_digital:
                    val = str(val)
                param_widget.setText(val)
                param_widget.paramwidget_edited.connect(self.on_paramwidget_edited)

            elif isinstance(params[param_key], dict):
                param_dict = params[param_key]
                display_param_name = get_param_display_name(param_key, param_dict)
                value = params[param_key]['value']
                param_widget = None  # Ensure initialization
                param_type = param_dict['type'] if 'type' in param_dict else 'line_editor'
                flush_btn = param_dict.get('flush_btn', False)
                path_selector = param_dict.get('path_selector', False)
                param_size = param_dict.get('size', 'short')
                if param_type == 'selector':
                    if 'url' in param_key:
                        size = size2width('median')
                    else:
                        size = size2width(param_size)

                    options_list = list(param_dict['options'])
                    if param_key == 'device':
                        default_label = self.tr('Default')
                        if default_label not in options_list:
                            options_list = [default_label] + options_list
                        effective_default = (getattr(pcfg, 'default_device', None) or '').strip() or DEFAULT_DEVICE
                        if not str(value).strip() or value == effective_default:
                            value = default_label
                    param_widget = ParamComboBox(
                        param_key, options_list, size=size, scrollWidget=scrollWidget, flush_btn=flush_btn, path_selector=path_selector)

                    if param_key == 'device' and DEFAULT_DEVICE == 'cpu':
                        param_dict['value'] = 'cpu'
                        for ii, device in enumerate(options_list):
                            if device in GPUINTENSIVE_SET:
                                model = param_widget.model()
                                item = model.item(ii, 0)
                                item.setEnabled(False)
                    param_widget.setCurrentText(str(value))
                    param_widget.setEditable(param_dict.get('editable', False))

                elif param_type == 'editor':
                    param_widget = ParamEditor(param_key)
                    param_widget.setText(value)

                elif param_type == 'checkbox':
                    param_widget = ParamCheckBox(param_key)
                    if isinstance(value, str):
                        value = value.lower().strip() == 'true'
                        params[param_key]['value'] = value
                    param_widget.setChecked(value)

                elif param_type == 'pushbtn':
                    param_widget = ParamPushButton(param_key, param_dict)
                    require_label = False

                elif param_type == 'line_editor':
                    param_widget = ParamLineEditor(param_key, force_digital=is_digital)
                    param_widget.setText(str(value))

                elif param_type == 'check_group':
                    param_widget = ParamCheckGroup(param_key, check_group=value)

                if param_widget is not None:
                    param_widget.paramwidget_edited.connect(self.on_paramwidget_edited)
                    if 'description' in param_dict:
                        param_widget.setToolTip(param_dict['description'])

            widget_idx = 0
            if require_label:
                param_label = ParamNameLabel(display_param_name)
                param_layout.addWidget(param_label, ii, 0)
                widget_idx = 1
            if param_widget is not None:
                pw_lo = None
                if hasattr(param_widget, 'flush_btn') or hasattr(param_widget, 'path_select_btn'):
                    pw_lo = QHBoxLayout()
                    pw_lo.addWidget(param_widget)
                if hasattr(param_widget, 'flush_btn'):
                    pw_lo.addWidget(param_widget.flush_btn)
                    param_widget.flushbtn_clicked.connect(self.on_flushbtn_clicked)
                if hasattr(param_widget, 'path_select_btn'):
                    pw_lo.addWidget(param_widget.path_select_btn)
                    param_widget.pathbtn_clicked.connect(self.on_pathbtn_clicked)
                if pw_lo is None:
                    param_layout.addWidget(param_widget, ii, widget_idx)
                else:
                    param_layout.addLayout(pw_lo, ii, widget_idx)
            else:
                v = params[param_key]
                raise ValueError(f"Failed to initialize widget for key-value pair: {param_key}-{v}")
            
    def on_flushbtn_clicked(self):
        paramw: ParamComboBox = self.sender()
        content_dict = {'content': '', 'widget': paramw, 'flush': True}
        self.paramwidget_edited.emit(paramw.param_key, content_dict)

    def on_pathbtn_clicked(self):
        paramw: ParamComboBox = self.sender()
        content_dict = {'content': '', 'widget': paramw, 'select_path': True}
        self.paramwidget_edited.emit(paramw.param_key, content_dict)

    def on_paramwidget_edited(self, param_key, param_content):
        if param_key == 'device' and param_content == self.tr('Default'):
            param_content = (getattr(pcfg, 'default_device', None) or '').strip() or DEFAULT_DEVICE
        content_dict = {'content': param_content}
        self.paramwidget_edited.emit(param_key, content_dict)

class ModuleParseWidgets(QWidget):
    def addModulesParamWidgets(self, ocr_instance):
        self.params = ocr_instance.get_params()
        self.on_module_changed()

    def on_module_changed(self):
        self.updateModuleParamWidget()

    def updateModuleParamWidget(self):
        widget = ParamWidget(self.params, scrollWidget=self)
        layout = QVBoxLayout()
        layout.addWidget(widget)
        self.setLayout(layout)

class ModuleConfigParseWidget(QWidget):
    module_changed = Signal(str)
    paramwidget_edited = Signal(str, dict)
    def __init__(self, module_name: str, get_valid_module_keys: Callable, scrollWidget: QWidget, add_from: int = 1, *args, **kwargs) -> None:
        super().__init__( *args, **kwargs)
        self.get_valid_module_keys = get_valid_module_keys
        self.module_combobox = ConfigComboBox(scrollWidget=scrollWidget)
        self.params_layout = QHBoxLayout()
        self.params_layout.setContentsMargins(0, 0, 0, 0)

        p_layout = QHBoxLayout()
        p_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.module_label = ParamNameLabel(module_name)
        p_layout.addWidget(self.module_label)
        p_layout.addWidget(self.module_combobox)
        p_layout.addStretch(-1)
        self.p_layout = p_layout

        layout = QVBoxLayout(self)
        self.param_widget_map = {}
        layout.addLayout(p_layout) 
        layout.addLayout(self.params_layout)
        layout.setSpacing(30)
        self.vlayout = layout

        self.visibleWidget: QWidget = None
        self.module_dict: dict = {}

    def addModulesParamWidgets(self, module_dict: dict):
        invalid_module_keys = []
        valid_modulekeys = self.get_valid_module_keys()

        num_widgets_before = len(self.param_widget_map)

        for module in module_dict:
            if module not in valid_modulekeys:
                invalid_module_keys.append(module)
                continue

            if module in self.param_widget_map:
                LOGGER.warning(f'duplicated module key: {module}')
                continue

            self.module_combobox.addItem(module)
            params = module_dict[module]
            if params is not None:
                self.param_widget_map[module] = None

        if len(invalid_module_keys) > 0:
            LOGGER.warning(F'Invalid module keys: {invalid_module_keys}')
            for ik in invalid_module_keys:
                module_dict.pop(ik)

        self.module_dict = module_dict

        num_widgets_after = len(self.param_widget_map)
        if num_widgets_before == 0 and num_widgets_after > 0:
            self.on_module_changed()
            self.module_combobox.currentTextChanged.connect(self.on_module_changed)

    def setModule(self, module: str):
        self.blockSignals(True)
        self.module_combobox.setCurrentText(module)
        self.updateModuleParamWidget()
        self.blockSignals(False)

    def updateModuleParamWidget(self):
        module = self.module_combobox.currentText()
        if self.visibleWidget is not None:
            self.visibleWidget.hide()
        if module in self.param_widget_map:
            widget: QWidget = self.param_widget_map[module]
            if widget is None:
                # lazy load widgets
                params = self.module_dict[module]
                widget = ParamWidget(params, scrollWidget=self)
                widget.paramwidget_edited.connect(self.paramwidget_edited)
                self.param_widget_map[module] = widget
                self.params_layout.addWidget(widget)
            else:
                widget.show()
            self.visibleWidget = widget

    def on_module_changed(self):
        self.updateModuleParamWidget()
        self.module_changed.emit(self.module_combobox.currentText())


class TranslatorConfigPanel(ModuleConfigParseWidget):

    show_pre_MT_keyword_window = Signal()
    show_MT_keyword_window = Signal()
    show_OCR_keyword_window = Signal()
    show_translation_context_requested = Signal()
    test_translator_clicked = Signal()

    def __init__(self, module_name, scrollWidget: QWidget = None, *args, **kwargs) -> None:
        super().__init__(module_name, GET_VALID_TRANSLATORS, scrollWidget=scrollWidget, *args, **kwargs)
        self.translator_changed = self.module_changed
    
        self.source_combobox = ConfigComboBox(scrollWidget=scrollWidget)
        self.target_combobox = ConfigComboBox(scrollWidget=scrollWidget)
        self.testTranslatorBtn = QPushButton(self.tr("Test translator"), self)
        self.testTranslatorBtn.setToolTip(self.tr("Test if the current translator is available (e.g. API key, network)."))
        self.testTranslatorBtn.clicked.connect(self.test_translator_clicked.emit)
        self.replacePreMTkeywordBtn = NoBorderPushBtn(self.tr("Keyword substitution for machine translation source text"), self)
        self.replacePreMTkeywordBtn.clicked.connect(self.show_pre_MT_keyword_window)
        self.replacePreMTkeywordBtn.setFixedWidth(500)
        self.replaceMTkeywordBtn = NoBorderPushBtn(self.tr("Keyword substitution for machine translation"), self)
        self.replaceMTkeywordBtn.clicked.connect(self.show_MT_keyword_window)
        self.replaceMTkeywordBtn.setFixedWidth(500)
        self.replaceOCRkeywordBtn = NoBorderPushBtn(self.tr("Keyword substitution for source text"), self)
        self.replaceOCRkeywordBtn.clicked.connect(self.show_OCR_keyword_window)
        self.replaceOCRkeywordBtn.setFixedWidth(500)
        self.translationContextBtn = NoBorderPushBtn(self.tr("Translation context (project)..."), self)
        self.translationContextBtn.setToolTip(self.tr("Set series path and glossary for this project (cross-chapter consistency)."))
        self.translationContextBtn.clicked.connect(self.show_translation_context_requested.emit)
        self.translationContextBtn.setFixedWidth(500)

        st_layout = QHBoxLayout()
        st_layout.setSpacing(15)
        st_layout.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        st_layout.addWidget(ParamNameLabel(self.tr('Source')))
        st_layout.addWidget(self.source_combobox)
        st_layout.addWidget(ParamNameLabel(self.tr('Target')))
        st_layout.addWidget(self.target_combobox)
        
        self.vlayout.insertLayout(1, st_layout) 
        self.vlayout.addWidget(self.testTranslatorBtn)
        self.vlayout.addWidget(self.replaceOCRkeywordBtn)
        self.vlayout.addWidget(self.translationContextBtn)
        self.vlayout.addWidget(self.replacePreMTkeywordBtn)
        self.vlayout.addWidget(self.replaceMTkeywordBtn)

    def finishSetTranslator(self, translator: BaseTranslator):
        self.source_combobox.blockSignals(True)
        self.target_combobox.blockSignals(True)
        self.module_combobox.blockSignals(True)

        self.source_combobox.clear()
        self.target_combobox.clear()

        self.source_combobox.addItems(translator.supported_src_list)
        self.target_combobox.addItems(translator.supported_tgt_list)
        self.module_combobox.setCurrentText(translator.name)
        self.source_combobox.setCurrentText(translator.lang_source)
        self.target_combobox.setCurrentText(translator.lang_target)
        self.updateModuleParamWidget()
        self.source_combobox.blockSignals(False)
        self.target_combobox.blockSignals(False)
        self.module_combobox.blockSignals(False)


class InpaintConfigPanel(ModuleConfigParseWidget):
    def __init__(self, module_name: str, scrollWidget: QWidget = None, *args, **kwargs) -> None:
        super().__init__(module_name, GET_VALID_INPAINTERS, scrollWidget = scrollWidget, *args, **kwargs)
        self.inpainter_changed = self.module_changed
        self.setInpainter = self.setModule
        self.needInpaintChecker = ParamCheckerBox(self.tr('Let the program decide whether it is necessary to use the selected inpaint method.'))
        self.vlayout.addWidget(self.needInpaintChecker)

    def showEvent(self, e) -> None:
        self.p_layout.insertWidget(1, self.module_combobox)
        super().showEvent(e)

    def hideEvent(self, e) -> None:
        self.p_layout.removeWidget(self.module_combobox)
        return super().hideEvent(e)

class TextDetectConfigPanel(ModuleConfigParseWidget):
    def __init__(self, module_name: str, scrollWidget: QWidget = None, *args, **kwargs) -> None:
        super().__init__(module_name, GET_VALID_TEXTDETECTORS, scrollWidget = scrollWidget, *args, **kwargs)
        self.detector_changed = self.module_changed
        self.setDetector = self.setModule
        self.keep_existing_checker = QCheckBox(text=self.tr('Keep Existing Lines'))
        self.p_layout.insertWidget(2, self.keep_existing_checker)
        self.dual_detect_checker = QCheckBox(self.tr('Run second detector (dual detect)'))
        self.dual_detect_checker.setToolTip(self.tr('Run a second text detector and merge results to catch more regions (e.g. one good at bubbles, one at captions).'))
        self.dual_detect_checker.setChecked(getattr(pcfg.module, 'enable_dual_detect', False))
        self.dual_detect_checker.clicked.connect(self._on_dual_detect_changed)
        self.secondary_detector_combobox = ConfigComboBox(scrollWidget=scrollWidget)
        self.secondary_detector_combobox.addItem('')
        self.secondary_detector_combobox.addItems(GET_VALID_TEXTDETECTORS())
        self.secondary_detector_combobox.setCurrentText(getattr(pcfg.module, 'textdetector_secondary', '') or '')
        self.secondary_detector_combobox.currentTextChanged.connect(self._on_secondary_detector_changed)
        dual_hl = QHBoxLayout()
        dual_hl.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        dual_hl.addWidget(self.dual_detect_checker)
        dual_hl.addWidget(QLabel(self.tr('Secondary:')))
        dual_hl.addWidget(self.secondary_detector_combobox)
        self.vlayout.addLayout(dual_hl)
        # Secondary detector params (shown when dual detect is on and a secondary is selected)
        self.secondary_params_container = QWidget(self)
        secondary_params_outer = QVBoxLayout(self.secondary_params_container)
        secondary_params_outer.setContentsMargins(0, 0, 0, 0)
        secondary_params_outer.addWidget(QLabel(self.tr('Secondary detector params:')))
        self.secondary_params_layout = QHBoxLayout()
        self.secondary_params_layout.setContentsMargins(0, 0, 0, 0)
        secondary_params_outer.addLayout(self.secondary_params_layout)
        self.secondary_param_widget_map = {}
        self.secondary_visible_widget = None
        self.vlayout.addWidget(self.secondary_params_container)
        self._update_secondary_params_visibility()

    def _on_dual_detect_changed(self):
        pcfg.module.enable_dual_detect = self.dual_detect_checker.isChecked()
        self._update_secondary_params_visibility()

    def _on_secondary_detector_changed(self, name: str):
        pcfg.module.textdetector_secondary = (name or '').strip()
        self._update_secondary_params_visibility()

    def _update_secondary_params_visibility(self):
        dual = self.dual_detect_checker.isChecked()
        sec = (self.secondary_detector_combobox.currentText() or '').strip()
        if dual and sec and sec in getattr(self, 'module_dict', {}):
            self.secondary_params_container.show()
            self._update_secondary_param_widget()
        else:
            self.secondary_params_container.hide()

    def _update_secondary_param_widget(self):
        sec_name = (self.secondary_detector_combobox.currentText() or '').strip()
        if self.secondary_visible_widget is not None:
            self.secondary_visible_widget.hide()
            self.secondary_params_layout.removeWidget(self.secondary_visible_widget)
            self.secondary_visible_widget = None
        if not sec_name or not getattr(self, 'module_dict', None) or sec_name not in self.module_dict:
            return
        if sec_name in self.secondary_param_widget_map:
            widget = self.secondary_param_widget_map[sec_name]
        else:
            params = self.module_dict[sec_name]
            if params is None:
                return
            widget = ParamWidget(params, scrollWidget=self)
            widget.paramwidget_edited.connect(self._on_secondary_param_edited)
            self.secondary_param_widget_map[sec_name] = widget
        self.secondary_params_layout.addWidget(widget)
        widget.show()
        self.secondary_visible_widget = widget

    def _on_secondary_param_edited(self, param_key: str, param_content: dict):
        sec_name = (self.secondary_detector_combobox.currentText() or '').strip()
        if not sec_name:
            return
        tparams = getattr(pcfg.module, 'textdetector_params', None) or {}
        if sec_name not in tparams or param_key not in tparams[sec_name]:
            return
        val = param_content.get('content')
        if val is None:
            return
        params = tparams[sec_name]
        if isinstance(params.get(param_key), dict):
            params[param_key]['value'] = val
        else:
            params[param_key] = val

    def addModulesParamWidgets(self, module_dict: dict):
        super().addModulesParamWidgets(module_dict)
        self._update_secondary_params_visibility()

class OCRConfigPanel(ModuleConfigParseWidget):
    def __init__(self, module_name: str, scrollWidget: QWidget = None, *args, **kwargs) -> None:
        super().__init__(module_name, GET_VALID_OCR, scrollWidget = scrollWidget, *args, **kwargs)
        self.ocr_changed = self.module_changed
        self.setOCR = self.setModule
        self.restoreEmptyOCRChecker = QCheckBox(self.tr("Delete and restore region where OCR return empty string."), self)
        self.restoreEmptyOCRChecker.clicked.connect(self.on_restore_empty_ocr)
        self.vlayout.addWidget(self.restoreEmptyOCRChecker)
        # 字体检测选项
        self.fontDetectChecker = QCheckBox(self.tr("Font Detection"), self)
        self.fontDetectChecker.setToolTip(self.tr("Detect font properties (e.g. bold, italic) from the image after OCR. Useful for manga/comics."))
        self.fontDetectChecker.setChecked(pcfg.module.ocr_font_detect)
        self.fontDetectChecker.clicked.connect(self.on_fontdetect_changed)
        self.vlayout.addWidget(self.fontDetectChecker)

    def on_restore_empty_ocr(self):
        pcfg.restore_ocr_empty = self.restoreEmptyOCRChecker.isChecked()

    def on_fontdetect_changed(self):
        pcfg.module.ocr_font_detect = self.fontDetectChecker.isChecked()