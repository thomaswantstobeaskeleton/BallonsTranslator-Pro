import os.path as osp
from typing import List

from qtpy.QtCore import QObject, QThread, Signal, Qt
from qtpy.QtGui import QFontDatabase
from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from utils import shared
from utils.google_fonts import install_google_font_family


POPULAR_GOOGLE_FONTS = [
    "Bangers", "Comic Neue", "Noto Sans", "Noto Sans JP", "Noto Sans KR", "Noto Sans SC",
    "Noto Serif JP", "Roboto", "Roboto Condensed", "Open Sans", "Lato", "Montserrat",
    "Oswald", "Raleway", "Poppins", "Inter", "Nunito", "M PLUS Rounded 1c", "Kosugi Maru",
    "Zen Maru Gothic", "Mochiy Pop One", "Permanent Marker", "Patrick Hand", "Architects Daughter",
]


class GoogleFontInstallWorker(QObject):
    finished = Signal(list, str)
    failed = Signal(str)

    def __init__(self, family: str, target_dir: str):
        super().__init__()
        self.family = family
        self.target_dir = target_dir

    def run(self):
        try:
            installed = install_google_font_family(self.family, self.target_dir)
            self.finished.emit(installed, self.target_dir)
        except Exception as exc:
            self.failed.emit(str(exc))


class GoogleFontInstallDialog(QDialog):
    font_installed = Signal(list, list, str)

    def __init__(self, parent=None, target_dir: str = None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Install Google Font"))
        self.setMinimumWidth(560)
        self.target_dir = target_dir or osp.join(shared.PROGRAM_PATH, "fonts", "google")
        self._thread = None
        self._worker = None

        intro = QLabel(self.tr("Search or type a Google Fonts family name, then install it into the app font folder."), self)
        intro.setWordWrap(True)

        self.family_combo = QComboBox(self)
        self.family_combo.setEditable(True)
        self.family_combo.setInsertPolicy(getattr(getattr(QComboBox, "InsertPolicy", QComboBox), "NoInsert"))
        self.family_combo.addItems(POPULAR_GOOGLE_FONTS)
        self.family_combo.setCurrentText("")
        self.family_combo.lineEdit().setPlaceholderText(self.tr("Example: Bangers, Noto Sans JP, Comic Neue"))
        self.family_combo.setMinimumWidth(320)
        self.family_combo.setToolTip(self.tr("Google Fonts family name. The installer downloads the official ZIP for this family."))
        try:
            self.family_combo.completer().setFilterMode(Qt.MatchFlag.MatchContains)
            self.family_combo.completer().setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        except Exception:
            pass

        self.folder_edit = QLineEdit(self.target_dir, self)
        self.folder_edit.setReadOnly(True)
        self.folder_edit.setToolTip(self.tr("Installed .ttf/.otf files are saved here."))

        self.status = QLabel(self.tr("Ready. Requires internet access to fonts.google.com."), self)
        self.status.setWordWrap(True)

        self.details = QPlainTextEdit(self)
        self.details.setReadOnly(True)
        self.details.setMaximumHeight(130)
        self.details.setPlaceholderText(self.tr("Install details and errors will appear here."))

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.addRow(self.tr("Font family"), self.family_combo)
        form.addRow(self.tr("Install folder"), self.folder_edit)

        self.install_btn = QPushButton(self.tr("Install Font"), self)
        self.install_btn.clicked.connect(self._install)
        self.close_btn = QPushButton(self.tr("Close"), self)
        self.close_btn.clicked.connect(self.reject)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self.install_btn)
        buttons.addWidget(self.close_btn)

        layout = QVBoxLayout(self)
        layout.addWidget(intro)
        layout.addLayout(form)
        layout.addWidget(self.status)
        layout.addWidget(self.details)
        layout.addLayout(buttons)

    def _family(self) -> str:
        return (self.family_combo.currentText() or "").strip()

    def _install(self):
        family = self._family()
        if not family:
            QMessageBox.warning(self, self.tr("Missing font family"), self.tr("Type or choose a Google Fonts family name first."))
            return
        self.install_btn.setEnabled(False)
        self.status.setText(self.tr(f"Installing '{family}'..."))
        self.details.setPlainText(self.tr("Downloading from Google Fonts and registering with Qt..."))
        self._thread = QThread(self)
        self._worker = GoogleFontInstallWorker(family, self.target_dir)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    def _on_finished(self, installed: List[str], target_dir: str):
        registered = []
        registration_failures = []
        for fp in installed:
            idx = QFontDatabase.addApplicationFont(fp)
            if idx < 0:
                registration_failures.append(fp)
                continue
            for fam_name in QFontDatabase.applicationFontFamilies(idx):
                if fam_name not in registered:
                    registered.append(fam_name)
                if fam_name not in shared.CUSTOM_FONTS:
                    shared.CUSTOM_FONTS.append(fam_name)
                if isinstance(shared.FONT_FAMILIES, set):
                    shared.FONT_FAMILIES.add(fam_name)
        self.install_btn.setEnabled(True)
        self.status.setText(self.tr(f"Installed {len(installed)} font file(s) to {target_dir}."))
        lines = [self.tr("Registered families:")]
        lines.extend([f"  • {name}" for name in (registered or [self.tr("None registered by Qt")])])
        if registration_failures:
            lines.append("")
            lines.append(self.tr("Qt could not register these files; restart the app or verify the font files:"))
            lines.extend([f"  • {fp}" for fp in registration_failures])
        lines.append("")
        lines.append(self.tr("Files:"))
        lines.extend([f"  • {fp}" for fp in installed])
        self.details.setPlainText("\n".join(lines))
        self.font_installed.emit(installed, registered, target_dir)
        QMessageBox.information(self, self.tr("Google Font Installed"), self.tr(f"Installed '{self._family()}' to:\n{target_dir}"))

    def _on_failed(self, message: str):
        self.install_btn.setEnabled(True)
        self.status.setText(self.tr("Install failed."))
        self.details.setPlainText(message)
        QMessageBox.warning(self, self.tr("Google Font Install Failed"), self.tr("Could not install the font. Details:\n") + message)
