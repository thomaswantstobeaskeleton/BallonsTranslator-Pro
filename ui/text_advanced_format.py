from typing import Any, Callable

from qtpy.QtWidgets import QSizePolicy, QVBoxLayout, QPushButton, QGroupBox, QLabel, QHBoxLayout, QWidget, QGridLayout
from qtpy.QtCore import Signal, Qt

from .custom_widget import SmallColorPickerLabel, SmallParamLabel, PanelArea, SmallSizeControlLabel, SmallSizeComboBox, SmallParamLabel, SmallSizeComboBox, SmallComboBox, TextCheckerLabel
from utils.fontformat import FontFormat


class TextShadowGroup(QGroupBox):
    def __init__(self, on_param_changed: Callable = None, title=None):
        super().__init__(title=title)
        self.on_param_changed = on_param_changed
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

        self.xoffset_box = SmallSizeComboBox([-2, 2], 'shadow_xoffset', self)
        self.xoffset_box.setToolTip(self.tr("Set X offset"))
        self.xoffset_box.param_changed.connect(self.on_offset_changed)
        self.xoffset_label = SmallSizeControlLabel(self, direction=1, text='X', alignment=Qt.AlignmentFlag.AlignCenter)
        self.xoffset_label.setFixedWidth(56)
        self.xoffset_label.size_ctrl_changed.connect(self.xoffset_box.changeByDelta)
        self.xoffset_label.btn_released.connect(self.on_offset_changed)
        xoffset_layout = QHBoxLayout()
        xoffset_layout.addWidget(self.xoffset_label)
        xoffset_layout.addWidget(self.xoffset_box)

        self.yoffset_box = SmallSizeComboBox([-2, 2], 'shadow_yoffset', self)
        self.yoffset_box.setToolTip(self.tr("Set Y offset"))
        self.yoffset_box.param_changed.connect(self.on_offset_changed)
        self.yoffset_label = SmallSizeControlLabel(self, direction=1, text='Y', alignment=Qt.AlignmentFlag.AlignCenter)
        self.yoffset_label.size_ctrl_changed.connect(self.yoffset_box.changeByDelta)
        self.yoffset_label.btn_released.connect(self.on_offset_changed)
        yoffset_layout = QHBoxLayout()
        yoffset_layout.addWidget(self.yoffset_label)
        yoffset_layout.addWidget(self.yoffset_box)

        self.color_label = SmallColorPickerLabel(self, param_name='shadow_color')

        self.strength_box = SmallSizeComboBox([0, 3], 'shadow_strength', self)
        self.strength_box.setToolTip(self.tr("Set Shadow Strength"))
        self.strength_box.param_changed.connect(self.on_param_changed)
        self.strength_label = SmallSizeControlLabel(self, direction=1, text=self.tr('Strength'), alignment=Qt.AlignmentFlag.AlignCenter)
        self.strength_label.setFixedWidth(56)
        self.strength_label.size_ctrl_changed.connect(lambda x : self.strength_box.changeByDelta(x, multiplier=0.03))
        self.strength_label.btn_released.connect(lambda : self.on_param_changed('shadow_strength', self.strength_box.value()))
        strength_layout = QHBoxLayout()
        strength_layout.addWidget(self.strength_label)
        strength_layout.addWidget(self.strength_box)

        self.radius_box = SmallSizeComboBox([0, 2], 'shadow_radius', self)
        self.radius_box.setToolTip(self.tr("Set Shadow Radius"))
        self.radius_box.param_changed.connect(self.on_param_changed)
        self.radius_label = SmallSizeControlLabel(self, direction=1, text=self.tr('Radius'), alignment=Qt.AlignmentFlag.AlignCenter)
        self.radius_label.size_ctrl_changed.connect(self.radius_box.changeByDelta)
        self.radius_label.btn_released.connect(lambda : self.on_param_changed('shadow_radius', self.radius_box.value()))
        radius_layout = QHBoxLayout()
        radius_layout.addWidget(self.radius_label)
        radius_layout.addWidget(self.radius_box)

        hlayout2 = QHBoxLayout()
        color_cell = QWidget(self)
        color_cell.setFixedWidth(52)
        color_cell.setAutoFillBackground(False)
        color_cell.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        color_cell_layout = QHBoxLayout(color_cell)
        color_cell_layout.setContentsMargins(0, 0, 0, 0)
        color_cell_layout.addWidget(self.color_label)
        hlayout2.addWidget(color_cell)
        hlayout2.addLayout(strength_layout)
        hlayout2.addLayout(radius_layout)

        offset_label = SmallParamLabel(self.tr('Offset'))
        offset_label.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)
        offset_label.setFixedWidth(52)
        offset_row = QHBoxLayout()
        offset_row.addWidget(offset_label)
        offset_row.addLayout(xoffset_layout)
        offset_row.addLayout(yoffset_layout)

        layout = QVBoxLayout(self)
        layout.addLayout(offset_row)
        layout.addLayout(hlayout2)

    def on_offset_changed(self, *args, **kwargs):
        self.on_param_changed('shadow_offset', [self.xoffset_box.value(), self.yoffset_box.value()])


class TextGradientGroup(QGroupBox):
    def __init__(self, on_param_changed: Callable = None):
        super().__init__()
        self.setTitle(self.tr('Gradient'))
        self.on_param_changed = on_param_changed
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

        self.start_picker = SmallColorPickerLabel(self, param_name='gradient_start_color')
        start_picker_label = SmallParamLabel(self.tr('Start Color'), alignment=Qt.AlignmentFlag.AlignCenter)
        start_picker_label.setFixedWidth(72)

        self.end_picker = SmallColorPickerLabel(self, param_name='gradient_end_color')
        end_picker_label = SmallParamLabel(self.tr('End Color'), alignment=Qt.AlignmentFlag.AlignCenter)

        self.enable_checker = TextCheckerLabel(self.tr('Enable'))
        self.enable_checker.checkStateChanged.connect(lambda checked: self.on_param_changed('gradient_enabled', checked))

        self.type_combobox = SmallComboBox(parent=self, options=[self.tr('Linear'), self.tr('Radial')])
        self.type_combobox.setToolTip(self.tr("Gradient type: Linear or Radial"))
        self.type_combobox.activated.connect(self.on_gradient_type_changed)
        type_label = SmallParamLabel(self.tr('Type'))
        type_label.setFixedWidth(72)

        self.size_box = SmallSizeComboBox([0.5, 2], 'gradient_size', self)
        self.size_box.setToolTip(self.tr("Set Gradient Size"))
        self.size_box.param_changed.connect(self.on_param_changed)
        self.size_label = SmallSizeControlLabel(self, direction=1, text=self.tr('Size'), alignment=Qt.AlignmentFlag.AlignCenter)
        self.size_label.size_ctrl_changed.connect(lambda x : self.size_box.changeByDelta(x, multiplier=0.02))
        self.size_label.btn_released.connect(lambda : self.on_param_changed('gradient_size', self.size_box.value()))

        self.angle_box = SmallSizeComboBox([0, 359], 'gradient_angle', self)
        self.angle_box.setToolTip(self.tr("Set Gradient Angle"))
        self.angle_box.param_changed.connect(self.on_param_changed)
        self.angle_label = SmallSizeControlLabel(self, direction=1, text=self.tr('Angle'), alignment=Qt.AlignmentFlag.AlignCenter)
        self.angle_label.setFixedHeight(20)
        self.angle_label.size_ctrl_changed.connect(lambda x : self.angle_box.changeByDelta(x, multiplier=1))
        self.angle_label.btn_released.connect(lambda : self.on_param_changed('gradient_angle', self.angle_box.value()))
        angle_label_cell = QWidget(self)
        angle_label_cell.setFixedSize(72, 20)
        angle_label_cell.setAutoFillBackground(False)
        angle_label_cell.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        angle_label_cell.setStyleSheet("background: transparent; border: none;")
        angle_label_cell_layout = QHBoxLayout(angle_label_cell)
        angle_label_cell_layout.setContentsMargins(0, 0, 0, 0)
        angle_label_cell_layout.addWidget(self.angle_label)

        # Use QGridLayout so the label column (0) and control column (1) align across all rows
        grid = QGridLayout()
        grid.setColumnMinimumWidth(0, 72)
        grid.addWidget(start_picker_label, 0, 0)
        grid.addWidget(self.start_picker, 0, 1)
        grid.addWidget(end_picker_label, 0, 2)
        grid.addWidget(self.end_picker, 0, 3)
        grid.addWidget(self.enable_checker, 0, 4)
        grid.addWidget(type_label, 1, 0)
        grid.addWidget(self.type_combobox, 1, 1, 1, 1, Qt.AlignmentFlag.AlignLeft)
        grid.addWidget(angle_label_cell, 2, 0)
        grid.addWidget(self.angle_box, 2, 1)
        grid.addWidget(self.size_label, 2, 2)
        grid.addWidget(self.size_box, 2, 3)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.addLayout(grid)

    def on_gradient_type_changed(self):
        self.on_param_changed('gradient_type', self.type_combobox.currentIndex())


class TextAdvancedFormatPanel(PanelArea):

    param_changed = Signal(str, object)

    def __init__(self, panel_name: str, config_name: str, config_expand_name: str, on_format_changed: Callable):
        super().__init__(panel_name, config_name, config_expand_name)

        self.active_format: FontFormat = None
        self.on_format_changed = on_format_changed

        self.linespacing_type_combobox = SmallComboBox(
            parent=self,
            options=[
                self.tr("Proportional"),
                self.tr("Distance")
            ]
        )
        self.linespacing_type_combobox.activated.connect(self.on_linespacing_type_changed)
        linespacing_type_label = SmallParamLabel(self.tr('Line Spacing Type'))
        linespacing_type_layout = QHBoxLayout()
        linespacing_type_layout.addWidget(linespacing_type_label)
        linespacing_type_layout.addWidget(self.linespacing_type_combobox)
        linespacing_type_layout.addStretch(1)

        self.opacity_box = SmallSizeComboBox([0, 1], 'opacity', self, init_value=1.)
        self.opacity_box.setToolTip(self.tr("Set Text Opacity"))
        self.opacity_box.param_changed.connect(self.on_format_changed)
        self.opacity_label = SmallSizeControlLabel(self, direction=1, text=self.tr('Opacity'), alignment=Qt.AlignmentFlag.AlignCenter)
        self.opacity_label.size_ctrl_changed.connect(self.opacity_box.changeByDelta)
        self.opacity_label.btn_released.connect(lambda : self.on_format_changed('opacity', self.opacity_box.value()))
        opacity_layout = QHBoxLayout()
        opacity_layout.addWidget(self.opacity_label)
        opacity_layout.addWidget(self.opacity_box)
        opacity_layout.addStretch(1)

        self.font_weight_values = [300, 400, 500, 600, 700]
        self.font_weight_combobox = SmallComboBox(
            parent=self,
            options=[self.tr('Light'), self.tr('Normal'), self.tr('Medium'), self.tr('SemiBold'), self.tr('Bold')]
        )
        self.font_weight_combobox.setToolTip(self.tr("Font weight (100–900)"))
        self.font_weight_combobox.activated.connect(self.on_font_weight_changed)
        font_weight_label = SmallParamLabel(self.tr('Font Weight'))
        font_weight_layout = QHBoxLayout()
        font_weight_layout.addWidget(font_weight_label)
        font_weight_layout.addWidget(self.font_weight_combobox)
        font_weight_layout.addStretch(1)

        # self.tate_chu_yoko_checker = QFontChecker()
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        self.scrollContent.after_resized.connect(self.adjuset_size)

        self.shadow_group = TextShadowGroup(self.on_format_changed, title=self.tr('Shadow'))
        self.shadow_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)

        self.gradient_group = TextGradientGroup(self.on_format_changed)
        self.gradient_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)

        # Allow horizontal scroll when content is wider than panel (e.g. narrow side panel)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Stack rows vertically so each control has room; avoids overflow and "pushed left" look
        vlayout = QVBoxLayout()
        vlayout.addLayout(linespacing_type_layout)
        vlayout.addLayout(opacity_layout)
        vlayout.addLayout(font_weight_layout)
        vlayout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.blend_mode_combobox = SmallComboBox(
            parent=self,
            options=[self.tr('Normal'), self.tr('Multiply'), self.tr('Screen'), self.tr('Overlay'), self.tr('Darken'), self.tr('Lighten')]
        )
        self.blend_mode_combobox.setToolTip(self.tr("Blend mode for compositing with the page image"))
        self.blend_mode_combobox.activated.connect(self.on_blend_mode_changed)
        blend_label = SmallParamLabel(self.tr('Blend Mode'))
        blend_layout = QHBoxLayout()
        blend_layout.addWidget(blend_label)
        blend_layout.addWidget(self.blend_mode_combobox)
        blend_layout.addStretch(1)

        self.outline_only_checker = TextCheckerLabel(self.tr('Outline only'))
        self.outline_only_checker.setToolTip(self.tr("Draw stroke only, no fill (set stroke width > 0)"))
        self.outline_only_checker.checkStateChanged.connect(lambda s: self.on_format_changed('outline_only', self.outline_only_checker.isChecked()))

        self.overlay_opacity_box = SmallSizeComboBox([0, 1], 'overlay_opacity', self, init_value=1.)
        self.overlay_opacity_box.setToolTip(self.tr("Foreground overlay image opacity"))
        self.overlay_opacity_box.param_changed.connect(self.on_format_changed)
        overlay_opacity_label = SmallParamLabel(self.tr('Overlay opacity'))
        overlay_opacity_layout = QHBoxLayout()
        overlay_opacity_layout.addWidget(overlay_opacity_label)
        overlay_opacity_layout.addWidget(self.overlay_opacity_box)
        overlay_opacity_layout.addStretch(1)

        self.skew_x_box = SmallSizeComboBox([-0.5, 0.5], 'skew_x', self, init_value=0.)
        self.skew_x_box.param_changed.connect(self.on_format_changed)
        skew_x_label = SmallParamLabel(self.tr('Skew X'))
        self.skew_y_box = SmallSizeComboBox([-0.5, 0.5], 'skew_y', self, init_value=0.)
        self.skew_y_box.param_changed.connect(self.on_format_changed)
        skew_y_label = SmallParamLabel(self.tr('Skew Y'))
        skew_layout = QHBoxLayout()
        skew_layout.addWidget(skew_x_label)
        skew_layout.addWidget(self.skew_x_box)
        skew_layout.addWidget(skew_y_label)
        skew_layout.addWidget(self.skew_y_box)
        skew_layout.addStretch(1)

        vlayout.addLayout(blend_layout)
        vlayout.addLayout(overlay_opacity_layout)
        vlayout.addLayout(skew_layout)
        vlayout.addWidget(self.outline_only_checker)
        vlayout.addWidget(self.shadow_group)
        vlayout.addWidget(self.gradient_group)

        self.setContentLayout(vlayout)
        self.vlayout = vlayout

    def adjuset_size(self):
        # Cap height so the panel doesn't grow unbounded; content can scroll vertically if needed
        self.setMaximumHeight(420)

    def on_linespacing_type_changed(self):
        self.on_format_changed('line_spacing_type', self.linespacing_type_combobox.currentIndex())

    def on_font_weight_changed(self):
        idx = self.font_weight_combobox.currentIndex()
        if 0 <= idx < len(self.font_weight_values):
            self.on_format_changed('font_weight', self.font_weight_values[idx])

    def on_blend_mode_changed(self):
        self.on_format_changed('blend_mode', self.blend_mode_combobox.currentIndex())

    def set_active_format(self, font_format: FontFormat):
        self.active_format = font_format
        self.linespacing_type_combobox.setCurrentIndex(font_format.line_spacing_type)

        # Map font_weight to combo index (closest match)
        w = font_format.font_weight
        if w is None:
            w = 400
        idx = min(range(len(self.font_weight_values)), key=lambda i: abs(self.font_weight_values[i] - w))
        self.font_weight_combobox.setCurrentIndex(idx)

        self.shadow_group.color_label.setPickerColor(font_format.shadow_color)
        self.shadow_group.strength_box.setValue(font_format.shadow_strength)
        self.shadow_group.radius_box.setValue(font_format.shadow_radius)
        self.shadow_group.xoffset_box.setValue(font_format.shadow_offset[0])
        self.shadow_group.yoffset_box.setValue(font_format.shadow_offset[1])

        self.gradient_group.size_box.setValue(font_format.gradient_size)
        self.gradient_group.angle_box.setValue(font_format.gradient_angle)
        self.gradient_group.enable_checker.setCheckState(font_format.gradient_enabled)
        self.gradient_group.type_combobox.setCurrentIndex(getattr(font_format, 'gradient_type', 0))
        self.gradient_group.start_picker.setPickerColor(font_format.gradient_start_color)
        self.gradient_group.end_picker.setPickerColor(font_format.gradient_end_color)
        self.blend_mode_combobox.setCurrentIndex(max(0, min(getattr(font_format, 'blend_mode', 0), 5)))
        self.outline_only_checker.setCheckState(getattr(font_format, 'outline_only', False))
        self.overlay_opacity_box.setValue(getattr(font_format, 'overlay_opacity', 1.0))
        self.skew_x_box.setValue(getattr(font_format, 'skew_x', 0.0))
        self.skew_y_box.setValue(getattr(font_format, 'skew_y', 0.0))
        # self.tate_chu_yoko_checker.setChecked(font_format.font)