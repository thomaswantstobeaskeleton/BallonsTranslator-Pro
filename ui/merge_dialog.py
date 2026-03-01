from qtpy.QtWidgets import QDialog, QVBoxLayout, QGroupBox, QFormLayout, QComboBox, QLineEdit, QPlainTextEdit, QCheckBox, QSpinBox, QLabel, QRadioButton, QButtonGroup, QHBoxLayout, QPushButton
from qtpy.QtCore import Signal, Qt
from qtpy.QtWidgets import QSizePolicy
from qtpy.QtGui import QShowEvent, QCloseEvent

from utils.config import pcfg

class MergeDialog(QDialog):
    # Signals: emitted when user clicks run buttons
    run_current_clicked = Signal()  # Run on current file
    run_all_clicked = Signal()      # Run on all files

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Region merge tool settings")
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.adjustSize()
        self.setWindowFlags(
            Qt.Window |
            Qt.WindowMinimizeButtonHint |
            Qt.WindowMaximizeButtonHint |
            Qt.WindowCloseButtonHint
        )

        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(6)
        self.layout.setContentsMargins(8, 8, 8, 8)

        # --- Mappings for merge logic (data values unchanged) ---
        self.merge_mode_map = {
            "Vertical": "VERTICAL",
            "Horizontal": "HORIZONTAL",
            "Vertical then horizontal": "VERTICAL_THEN_HORIZONTAL",
            "Horizontal then vertical": "HORIZONTAL_THEN_VERTICAL",
            "None": "NONE",
        }
        self.label_strategy_map = {
            "Prefer shorter label": "PREFER_SHORTER",
            "Use first box's label": "FIRST",
            "Combine labels (label1+label2)": "COMBINE",
            "Prefer non-default label": "PREFER_NON_DEFAULT",
        }

        # --- Main Settings --- #
        main_group = QGroupBox("Main settings")
        main_layout = QFormLayout(main_group)
        main_layout.setSpacing(4)
        main_layout.setContentsMargins(8, 6, 8, 6)

        self.merge_mode = QComboBox()
        for text, data in self.merge_mode_map.items():
            self.merge_mode.addItem(text, userData=data)
        main_layout.addRow("Merge mode:", self.merge_mode)
        self.layout.addWidget(main_group)

        # --- Text reading order (by label) ---
        reading_order_group = QGroupBox("Text merge order (by label)")
        reading_order_layout = QFormLayout(reading_order_group)
        reading_order_layout.setSpacing(4)
        reading_order_layout.setContentsMargins(8, 6, 8, 6)

        self.ltr_labels_edit = QLineEdit()
        self.ltr_labels_edit.setPlaceholderText("label1,label2,...")
        self.rtl_labels_edit = QLineEdit()
        self.rtl_labels_edit.setText("balloon,qipao,shuqing")
        self.ttb_labels_edit = QLineEdit()
        self.ttb_labels_edit.setText("changfangtiao,hengxie")

        reading_order_layout.addRow("Left-to-right (LTR) labels:", self.ltr_labels_edit)
        reading_order_layout.addRow("Right-to-left (RTL) labels:", self.rtl_labels_edit)
        reading_order_layout.addRow("Top-to-bottom (TTB) labels:", self.ttb_labels_edit)

        self.layout.addWidget(reading_order_group)

        # --- Label rules --- #
        label_group = QGroupBox("Label merge rules")
        label_layout = QFormLayout(label_group)
        label_layout.setSpacing(4)
        label_layout.setContentsMargins(8, 6, 8, 6)

        self.label_merge_strategy = QComboBox()
        for text, data in self.label_strategy_map.items():
            self.label_merge_strategy.addItem(text, userData=data)
        label_layout.addRow("Label merge strategy:", self.label_merge_strategy)

        self.enable_exclude_labels = QCheckBox("Enable exclude-from-merge labels (blacklist)")
        self.enable_exclude_labels.setChecked(True)
        label_layout.addRow(self.enable_exclude_labels)

        self.exclude_labels = QLineEdit()
        self.exclude_labels.setText("other")
        self.exclude_labels.setPlaceholderText("e.g. label1,label2")
        label_layout.addRow("Blacklist labels:", self.exclude_labels)

        self.enable_exclude_labels.toggled.connect(self.exclude_labels.setEnabled)

        self.require_same_label = QCheckBox("Require same label to merge")
        label_layout.addRow(self.require_same_label)

        self.use_specific_groups = QCheckBox("Merge only within specific label groups")
        self.specific_groups_edit = QPlainTextEdit()
        self.specific_groups_edit.setPlaceholderText("One group per line, labels comma-separated\n e.g.:\nballoon,balloon2\nqipao,qipao2")
        self.specific_groups_edit.setPlainText("balloon\nqipao\nshuqing\nchangfangtiao\nhengxie")
        self.specific_groups_edit.setMinimumHeight(100)
        self.specific_groups_edit.setMaximumHeight(120)
        self.specific_groups_edit.setEnabled(False)
        self.use_specific_groups.toggled.connect(self.specific_groups_edit.setEnabled)
        self.use_specific_groups.toggled.connect(lambda checked: self.require_same_label.setDisabled(checked))

        label_layout.addRow(self.use_specific_groups)
        label_layout.addRow(self.specific_groups_edit)

        self.layout.addWidget(label_group)

        # --- Geometric rules ---
        geo_group = QGroupBox("Geometry merge parameters")
        geo_layout = QFormLayout(geo_group)
        geo_layout.setSpacing(4)
        geo_layout.setContentsMargins(8, 6, 8, 6)

        self.max_vertical_gap = QSpinBox()
        self.max_vertical_gap.setRange(-100, 1000)
        self.max_vertical_gap.setValue(10)
        self.max_vertical_gap.setToolTip("Max gap between boxes (px). 0 = must touch; negative = require overlap (e.g. -15 = at least 15 px overlap).")
        self.min_width_overlap_ratio = QSpinBox()
        self.min_width_overlap_ratio.setRange(0, 100)
        self.min_width_overlap_ratio.setValue(90)
        self.min_width_overlap_ratio.setSuffix(" %")
        self.min_width_overlap_ratio.setToolTip("Min horizontal overlap (%). Higher = boxes must align more (e.g. 98 = almost fully overlapped).")

        self.max_horizontal_gap = QSpinBox()
        self.max_horizontal_gap.setRange(-100, 1000)
        self.max_horizontal_gap.setValue(10)
        self.max_horizontal_gap.setToolTip("Max gap between boxes (px). 0 = must touch; negative = require overlap (e.g. -15 = at least 15 px overlap).")
        self.min_height_overlap_ratio = QSpinBox()
        self.min_height_overlap_ratio.setRange(0, 100)
        self.min_height_overlap_ratio.setValue(90)
        self.min_height_overlap_ratio.setSuffix(" %")
        self.min_height_overlap_ratio.setToolTip("Min vertical overlap (%). Higher = boxes must align more (e.g. 98 = almost fully overlapped).")

        geo_layout.addRow(QLabel("<b>Vertical merge (up-down)</b>"))
        geo_layout.addRow("Max vertical gap (px):", self.max_vertical_gap)
        geo_layout.addRow("Min horizontal overlap ratio:", self.min_width_overlap_ratio)
        geo_layout.addRow(QLabel("<b>Horizontal merge (left-right)</b>"))
        geo_layout.addRow("Max horizontal gap (px):", self.max_horizontal_gap)
        geo_layout.addRow("Min vertical overlap ratio:", self.min_height_overlap_ratio)
        geo_strict_label = QLabel(self.tr("Strict (one box per bubble): set both gaps to 0 or negative (e.g. -10), overlap to 98–100%."))
        geo_strict_label.setWordWrap(True)
        geo_strict_label.setStyleSheet("color: gray; font-size: 0.9em;")
        geo_layout.addRow(geo_strict_label)

        self.layout.addWidget(geo_group)

        # --- Advanced options --- #
        advanced_group = QGroupBox("Advanced options")
        advanced_layout = QVBoxLayout(advanced_group)
        advanced_layout.setSpacing(4)
        advanced_layout.setContentsMargins(8, 6, 8, 6)
        self.allow_negative_gap = QCheckBox("Allow negative gap (overlapping boxes)")
        self.allow_negative_gap.setChecked(True)
        advanced_layout.addWidget(self.allow_negative_gap)

        self.layout.addWidget(advanced_group)

        # --- Merge result type --- #
        result_type_group = QGroupBox("Merge result type")
        result_type_layout = QVBoxLayout(result_type_group)
        result_type_layout.setSpacing(4)
        result_type_layout.setContentsMargins(8, 6, 8, 6)

        self.output_type_group = QButtonGroup(self)
        self.radio_output_rectangle = QRadioButton("Axis-aligned rectangle")
        self.radio_output_rotation = QRadioButton("Rotated rectangle")

        self.radio_output_rectangle.setChecked(True)

        self.output_type_group.addButton(self.radio_output_rectangle, 1)
        self.output_type_group.addButton(self.radio_output_rotation, 2)

        result_type_layout.addWidget(self.radio_output_rectangle)
        result_type_layout.addWidget(self.radio_output_rotation)

        self.layout.addWidget(result_type_group)

        # --- Buttons --- #
        button_layout = QHBoxLayout()
        self.run_current_button = QPushButton("Run on current page")
        self.run_all_button = QPushButton("Run on all pages")
        self.cancel_button = QPushButton("Cancel")

        button_layout.addWidget(self.run_current_button)
        button_layout.addWidget(self.run_all_button)
        button_layout.addWidget(self.cancel_button)
        button_layout.addStretch()

        self.run_current_button.clicked.connect(self.on_run_current)
        self.run_all_button.clicked.connect(self.on_run_all)
        self.cancel_button.clicked.connect(self.reject)

        self.layout.addLayout(button_layout)

    def on_run_current(self):
        self.run_current_clicked.emit()

    def on_run_all(self):
        self.run_all_clicked.emit()

    def get_serializable_settings(self):
        """Return a JSON-serializable dict of current dialog state for saving."""
        excluded = self.exclude_labels.text().strip()
        excluded_list = [l.strip() for l in excluded.split(",") if l.strip()]
        return {
            "merge_mode": self.merge_mode.currentData(),
            "ltr_labels": self.ltr_labels_edit.text().strip(),
            "rtl_labels": self.rtl_labels_edit.text().strip(),
            "ttb_labels": self.ttb_labels_edit.text().strip(),
            "label_merge_strategy": self.label_merge_strategy.currentData(),
            "enable_exclude_labels": self.enable_exclude_labels.isChecked(),
            "exclude_labels": self.exclude_labels.text().strip(),
            "require_same_label": self.require_same_label.isChecked(),
            "use_specific_groups": self.use_specific_groups.isChecked(),
            "specific_groups_text": self.specific_groups_edit.toPlainText().strip(),
            "max_vertical_gap": self.max_vertical_gap.value(),
            "min_width_overlap_ratio": self.min_width_overlap_ratio.value(),
            "max_horizontal_gap": self.max_horizontal_gap.value(),
            "min_height_overlap_ratio": self.min_height_overlap_ratio.value(),
            "allow_negative_gap": self.allow_negative_gap.isChecked(),
            "output_shape_type": "rectangle" if self.output_type_group.checkedId() == 1 else "rotation",
        }

    def save_settings_to_pcfg(self):
        """Write current dialog state to pcfg (call before save_config)."""
        pcfg.region_merge_settings = self.get_serializable_settings()

    def set_settings(self, d):
        """Restore dialog state from a saved dict (e.g. pcfg.region_merge_settings)."""
        if not d or not isinstance(d, dict):
            return
        mode = d.get("merge_mode")
        if mode is not None:
            idx = self.merge_mode.findData(mode)
            if idx >= 0:
                self.merge_mode.setCurrentIndex(idx)
        if "ltr_labels" in d:
            self.ltr_labels_edit.setText(d.get("ltr_labels", ""))
        if "rtl_labels" in d:
            self.rtl_labels_edit.setText(d.get("rtl_labels", ""))
        if "ttb_labels" in d:
            self.ttb_labels_edit.setText(d.get("ttb_labels", ""))
        strat = d.get("label_merge_strategy")
        if strat is not None:
            idx = self.label_merge_strategy.findData(strat)
            if idx >= 0:
                self.label_merge_strategy.setCurrentIndex(idx)
        self.enable_exclude_labels.setChecked(d.get("enable_exclude_labels", True))
        if "exclude_labels" in d:
            self.exclude_labels.setText(d.get("exclude_labels", ""))
        self.require_same_label.setChecked(d.get("require_same_label", False))
        self.use_specific_groups.setChecked(d.get("use_specific_groups", False))
        if "specific_groups_text" in d:
            self.specific_groups_edit.setPlainText(d.get("specific_groups_text", ""))
        if "max_vertical_gap" in d:
            self.max_vertical_gap.setValue(int(d["max_vertical_gap"]))
        if "min_width_overlap_ratio" in d:
            self.min_width_overlap_ratio.setValue(int(d["min_width_overlap_ratio"]))
        if "max_horizontal_gap" in d:
            self.max_horizontal_gap.setValue(int(d["max_horizontal_gap"]))
        if "min_height_overlap_ratio" in d:
            self.min_height_overlap_ratio.setValue(int(d["min_height_overlap_ratio"]))
        self.allow_negative_gap.setChecked(d.get("allow_negative_gap", True))
        out = d.get("output_shape_type", "rectangle")
        if out == "rotation":
            self.radio_output_rotation.setChecked(True)
        else:
            self.radio_output_rectangle.setChecked(True)

    def showEvent(self, event: QShowEvent):
        super().showEvent(event)
        self.set_settings(getattr(pcfg, "region_merge_settings", {}))

    def closeEvent(self, event: QCloseEvent):
        self.save_settings_to_pcfg()
        super().closeEvent(event)

    def get_config(self):
        """Return merge config from dialog."""
        config = {}
        config["MERGE_MODE"] = self.merge_mode.currentData()
        config["READING_DIRECTION"] = "LTR"

        per_label_directions = {}
        for label in [l.strip() for l in self.ltr_labels_edit.text().split(',') if l.strip()]:
            per_label_directions[label] = 'LTR'
        for label in [l.strip() for l in self.rtl_labels_edit.text().split(',') if l.strip()]:
            per_label_directions[label] = 'RTL'
        for label in [l.strip() for l in self.ttb_labels_edit.text().split(',') if l.strip()]:
            per_label_directions[label] = 'TTB'
        config["PER_LABEL_DIRECTIONS"] = per_label_directions

        if self.enable_exclude_labels.isChecked():
            excluded = self.exclude_labels.text().strip()
            config["LABELS_TO_EXCLUDE_FROM_MERGE"] = set(l.strip() for l in excluded.split(",") if l.strip())
        else:
            config["LABELS_TO_EXCLUDE_FROM_MERGE"] = set()

        config["USE_SPECIFIC_MERGE_GROUPS"] = self.use_specific_groups.isChecked()
        if config["USE_SPECIFIC_MERGE_GROUPS"]:
            groups_text = self.specific_groups_edit.toPlainText().strip()
            groups = []
            for line in groups_text.split('\n'):
                if line.strip():
                    groups.append([l.strip() for l in line.split(',')])
            config["SPECIFIC_MERGE_GROUPS"] = groups
            config["REQUIRE_SAME_LABEL"] = False
        else:
            config["SPECIFIC_MERGE_GROUPS"] = []
            config["REQUIRE_SAME_LABEL"] = self.require_same_label.isChecked()

        config["LABEL_MERGE_STRATEGY"] = self.label_merge_strategy.currentData()

        config["VERTICAL_MERGE_PARAMS"] = {
            "max_vertical_gap": self.max_vertical_gap.value(),
            "min_width_overlap_ratio": self.min_width_overlap_ratio.value(),
            "overlap_epsilon": 1e-6
        }

        config["HORIZONTAL_MERGE_PARAMS"] = {
            "max_horizontal_gap": self.max_horizontal_gap.value(),
            "min_height_overlap_ratio": self.min_height_overlap_ratio.value(),
            "overlap_epsilon": 1e-6
        }

        config["ADVANCED_MERGE_OPTIONS"] = {
            "allow_negative_gap": self.allow_negative_gap.isChecked(),
            "debug_mode": False
        }

        config["OUTPUT_SHAPE_TYPE"] = "rectangle" if self.output_type_group.checkedId() == 1 else "rotation"

        return config
