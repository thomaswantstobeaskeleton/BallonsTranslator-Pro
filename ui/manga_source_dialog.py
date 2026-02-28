"""
Dialog to search for manga/manhua/manhwa titles, list chapters, and download selected chapters.
Uses MangaDex API by default. Opens from Tools → Manga / Comic source...
"""
from __future__ import annotations

import os
import os.path as osp
import re
from typing import Any, List, Optional

from qtpy.QtCore import Qt, Signal, QThread, QObject
from qtpy.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QListWidget,
    QListWidgetItem,
    QComboBox,
    QCheckBox,
    QGroupBox,
    QFileDialog,
    QProgressBar,
    QSplitter,
    QSizePolicy,
    QMessageBox,
    QAbstractItemView,
    QDoubleSpinBox,
)

from utils.manga_sources import MangaDexClient
from utils.config import pcfg, save_config, ProgramConfig
from utils.logger import logger as LOGGER


def _sanitize_filename(name: str) -> str:
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    return name.strip(". ") or "unnamed"


def _extract_mangadex_chapter_id(url_or_id: str) -> Optional[str]:
    """Extract MangaDex chapter UUID from URL or return as-is if already a UUID."""
    s = (url_or_id or "").strip()
    if not s:
        return None
    uuid_match = re.search(
        r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', s, re.IGNORECASE
    )
    if uuid_match:
        return uuid_match.group(0)
    return None


class MangaSourceWorker(QObject):
    """Runs API calls and downloads in a background thread."""
    search_finished = Signal(list)   # list of {id, title, description}
    feed_finished = Signal(list)     # list of {id, chapter, display, ...}
    error = Signal(str)
    download_progress = Signal(int, int, str)  # current, total, chapter_display
    download_finished = Signal(str)  # path to folder that was downloaded
    run_search = Signal(str)
    run_feed = Signal(str, str)
    run_download = Signal(object)  # (chapter_infos, base_dir, manga_title, data_saver)
    run_load_chapter_url = Signal(str)

    def __init__(self):
        super().__init__()
        self._client = MangaDexClient(timeout=30, request_delay=getattr(pcfg, 'manga_source_request_delay', 0.3))
        self._abort = False
        self.run_search.connect(self.do_search)
        self.run_feed.connect(self.do_feed)
        self.run_download.connect(self._do_download)
        self.run_load_chapter_url.connect(self.do_load_chapter_by_url)

    def abort(self):
        self._abort = True

    def do_search(self, title: str):
        self._abort = False
        self._client.request_delay = getattr(pcfg, 'manga_source_request_delay', 0.3)
        try:
            results = self._client.search(title, limit=25)
            self.search_finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))

    def do_feed(self, manga_id: str, lang: str):
        self._abort = False
        self._client.request_delay = getattr(pcfg, 'manga_source_request_delay', 0.3)
        try:
            chapters = self._client.get_feed(manga_id, translated_language=lang, limit=500, order="asc")
            self.feed_finished.emit(chapters)
        except Exception as e:
            self.error.emit(str(e))

    def do_download(
        self,
        chapter_infos: List[dict],
        base_dir: str,
        manga_title: str,
        data_saver: bool,
    ):
        self._abort = False
        self._client.request_delay = getattr(pcfg, 'manga_source_request_delay', 0.3)
        manga_safe = _sanitize_filename(manga_title)
        parent_dir = osp.join(base_dir, manga_safe)
        os.makedirs(parent_dir, exist_ok=True)
        total_ch = len(chapter_infos)
        for i, ch in enumerate(chapter_infos):
            if self._abort:
                self.error.emit("Download aborted.")
                return
            ch_id = ch.get("id")
            display = ch.get("display", "Ch.?")
            ch_safe = _sanitize_filename(display)
            save_dir = osp.join(parent_dir, ch_safe)
            try:
                result = self._client.download_chapter(
                    ch_id,
                    save_dir,
                    data_saver=data_saver,
                    on_progress=lambda cur, tot, fn: None,
                )
                if result:
                    self.download_progress.emit(i + 1, total_ch, display)
                else:
                    self.error.emit("Failed to download " + display)
                    return
            except Exception as e:
                self.error.emit(f"{display}: {e}")
                return
        self.download_finished.emit(parent_dir)

    def _do_download(self, payload: tuple):
        """Slot for run_download: payload = (chapter_infos, base_dir, manga_title, data_saver)."""
        self.do_download(payload[0], payload[1], payload[2], payload[3])

    def do_load_chapter_by_url(self, url_or_id: str):
        """Load a single chapter by MangaDex chapter URL or UUID. Emits feed_finished([ch]) or error."""
        self._abort = False
        self._client.request_delay = getattr(pcfg, 'manga_source_request_delay', 0.3)
        ch_id = _extract_mangadex_chapter_id(url_or_id)
        if not ch_id:
            self.error.emit("Invalid MangaDex chapter URL or ID.")
            return
        try:
            ch = self._client.get_chapter_by_id(ch_id)
            if ch:
                self.feed_finished.emit([ch])
            else:
                self.error.emit("Chapter not found.")
        except Exception as e:
            self.error.emit(str(e))


class MangaSourceDialog(QDialog):
    """Search manga, load chapters, download selected. Emits open_folder_requested(path)."""
    open_folder_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Manga / Comic source"))
        self.setMinimumSize(520, 480)
        self.resize(600, 560)
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowMaximizeButtonHint
            | Qt.WindowType.WindowCloseButtonHint
        )

        self._worker = MangaSourceWorker()
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.start()

        self._manga_results: List[dict] = []
        self._chapters: List[dict] = []
        self._selected_manga: Optional[dict] = None
        saved_dir = (getattr(pcfg, 'manga_source_download_dir', None) or '').strip()
        self._download_base_dir = (
            saved_dir
            if saved_dir
            else ProgramConfig.default_downloaded_chapters_dir()
        )
        self._is_url_source = False

        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(10, 10, 10, 10)

        # --- Source & Search ---
        search_group = QGroupBox(self.tr("Search"))
        search_layout = QVBoxLayout(search_group)
        row = QHBoxLayout()
        self._source_combo = QComboBox()
        self._source_combo.addItem("MangaDex", "mangadex")
        self._source_combo.addItem(self.tr("MangaDex (by chapter URL)"), "mangadex_url")
        self._source_combo.setToolTip(self.tr("Manga source. Use chapter URL to load a single chapter by link."))
        self._source_combo.currentIndexChanged.connect(self._on_source_changed)
        row.addWidget(QLabel(self.tr("Source:")))
        row.addWidget(self._source_combo)
        row.addStretch()
        search_layout.addLayout(row)
        row2 = QHBoxLayout()
        self._search_edit = QLineEdit()
        self._search_edit.setPlaceholderText(self.tr("Enter manga title..."))
        self._search_edit.returnPressed.connect(self._on_search)
        row2.addWidget(self._search_edit)
        self._search_btn = QPushButton(self.tr("Search"))
        self._search_btn.clicked.connect(self._on_search)
        row2.addWidget(self._search_btn)
        search_layout.addLayout(row2)
        row_url = QHBoxLayout()
        self._url_edit = QLineEdit()
        self._url_edit.setPlaceholderText(self.tr("Paste MangaDex chapter URL or chapter UUID..."))
        self._url_edit.returnPressed.connect(self._on_load_chapter_url)
        row_url.addWidget(self._url_edit)
        self._load_chapter_url_btn = QPushButton(self.tr("Load chapter"))
        self._load_chapter_url_btn.clicked.connect(self._on_load_chapter_url)
        row_url.addWidget(self._load_chapter_url_btn)
        search_layout.addLayout(row_url)
        self._url_row_widgets = [self._url_edit, self._load_chapter_url_btn]
        self._search_row_widgets = [self._search_edit, self._search_btn]
        layout.addWidget(search_group)

        # --- Results (manga list) ---
        results_group = QGroupBox(self.tr("Results"))
        results_layout = QVBoxLayout(results_group)
        self._results_list = QListWidget()
        self._results_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._results_list.setMinimumHeight(100)
        self._results_list.currentRowChanged.connect(self._on_result_selected)
        results_layout.addWidget(self._results_list)
        layout.addWidget(results_group)

        # --- Chapters ---
        ch_group = QGroupBox(self.tr("Chapters"))
        ch_layout = QVBoxLayout(ch_group)
        row_ch = QHBoxLayout()
        self._lang_combo = QComboBox()
        for code, label in [
            ("en", "English"),
            ("ja", "Japanese"),
            ("zh-hans", "Chinese (Simplified)"),
            ("zh-hant", "Chinese (Traditional)"),
            ("ko", "Korean"),
        ]:
            self._lang_combo.addItem(label, code)
        row_ch.addWidget(QLabel(self.tr("Language:")))
        row_ch.addWidget(self._lang_combo)
        self._lang_combo.currentIndexChanged.connect(self._save_config_to_pcfg)
        self._data_saver_check = QCheckBox(self.tr("Use data-saver (smaller images)"))
        self._data_saver_check.setToolTip(self.tr("Faster and smaller files; lower resolution."))
        row_ch.addWidget(self._data_saver_check)
        self._data_saver_check.toggled.connect(self._save_config_to_pcfg)
        row_ch.addStretch()
        self._delay_spin = QDoubleSpinBox()
        self._delay_spin.setRange(0.0, 2.0)
        self._delay_spin.setSingleStep(0.1)
        self._delay_spin.setSuffix(" s")
        self._delay_spin.setToolTip(self.tr("Delay between API requests (rate limiting)."))
        self._delay_spin.setValue(getattr(pcfg, 'manga_source_request_delay', 0.3))
        self._delay_spin.valueChanged.connect(self._on_delay_changed)
        row_ch.addWidget(QLabel(self.tr("Request delay:")))
        row_ch.addWidget(self._delay_spin)
        ch_layout.addLayout(row_ch)
        btn_row = QHBoxLayout()
        self._load_ch_btn = QPushButton(self.tr("Load chapters"))
        self._load_ch_btn.clicked.connect(self._on_load_chapters)
        self._load_ch_btn.setEnabled(False)
        self._select_all_btn = QPushButton(self.tr("Select all"))
        self._select_all_btn.clicked.connect(self._on_select_all_chapters)
        self._deselect_all_btn = QPushButton(self.tr("Deselect all"))
        self._deselect_all_btn.clicked.connect(self._on_deselect_all_chapters)
        btn_row.addWidget(self._load_ch_btn)
        btn_row.addWidget(self._select_all_btn)
        btn_row.addWidget(self._deselect_all_btn)
        btn_row.addStretch()
        ch_layout.addLayout(btn_row)
        self._chapters_list = QListWidget()
        self._chapters_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._chapters_list.setMinimumHeight(120)
        ch_layout.addWidget(self._chapters_list)
        layout.addWidget(ch_group)

        # --- Download ---
        dl_group = QGroupBox(self.tr("Download"))
        dl_layout = QVBoxLayout(dl_group)
        row_dl = QHBoxLayout()
        self._folder_edit = QLineEdit()
        self._folder_edit.setReadOnly(True)
        self._folder_edit.setPlaceholderText(self.tr("Choose folder..."))
        self._folder_edit.setText(self._download_base_dir)
        row_dl.addWidget(self._folder_edit)
        self._browse_btn = QPushButton(self.tr("Browse..."))
        self._browse_btn.clicked.connect(self._on_browse)
        row_dl.addWidget(self._browse_btn)
        dl_layout.addLayout(row_dl)
        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        dl_layout.addWidget(self._status_label)
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(True)
        dl_layout.addWidget(self._progress_bar)
        self._open_after_download_check = QCheckBox(self.tr("Open in BallonsTranslator after download"))
        self._open_after_download_check.setToolTip(self.tr("When checked, the first chapter folder will open in BallonsTranslator automatically when download finishes."))
        self._open_after_download_check.setChecked(getattr(pcfg, 'manga_source_open_after_download', False))
        self._open_after_download_check.stateChanged.connect(self._save_config_to_pcfg)
        dl_layout.addWidget(self._open_after_download_check)
        btn_dl_row = QHBoxLayout()
        self._download_btn = QPushButton(self.tr("Download selected chapters"))
        self._download_btn.clicked.connect(self._on_download)
        self._open_folder_btn = QPushButton(self.tr("Open folder in BallonsTranslator"))
        self._open_folder_btn.clicked.connect(self._on_open_folder)
        self._open_folder_btn.setEnabled(False)
        self._last_download_path: Optional[str] = None
        self._last_download_first_chapter: Optional[str] = None
        btn_dl_row.addWidget(self._download_btn)
        btn_dl_row.addWidget(self._open_folder_btn)
        btn_dl_row.addStretch()
        dl_layout.addLayout(btn_dl_row)
        layout.addWidget(dl_group)

        # --- Connect worker signals ---
        self._worker.search_finished.connect(self._on_search_finished)
        self._worker.feed_finished.connect(self._on_feed_finished)
        self._worker.error.connect(self._on_worker_error)
        self._worker.download_progress.connect(self._on_download_progress)
        self._worker.download_finished.connect(self._on_download_finished)

        # Load persisted config
        lang = getattr(pcfg, 'manga_source_lang', 'en')
        idx = self._lang_combo.findData(lang)
        if idx >= 0:
            self._lang_combo.setCurrentIndex(idx)
        self._data_saver_check.setChecked(getattr(pcfg, 'manga_source_data_saver', False))
        self._folder_edit.setText(self._download_base_dir)
        self._delay_spin.setValue(getattr(pcfg, 'manga_source_request_delay', 0.3))
        self._open_after_download_check.setChecked(getattr(pcfg, 'manga_source_open_after_download', False))
        self._on_source_changed(self._source_combo.currentIndex())

    def _on_source_changed(self, index: int):
        source = self._source_combo.currentData() if index >= 0 else None
        self._is_url_source = source == "mangadex_url"
        for w in self._search_row_widgets:
            w.setVisible(not self._is_url_source)
        for w in self._url_row_widgets:
            w.setVisible(self._is_url_source)
        if self._is_url_source:
            self._results_list.clear()
            self._manga_results = []
            self._load_ch_btn.setEnabled(False)
        self._status_label.setText("" if self._is_url_source else self._status_label.text())

    def _on_load_chapter_url(self):
        url = self._url_edit.text().strip()
        if not url:
            self._status_label.setText(self.tr("Enter a MangaDex chapter URL or UUID."))
            return
        self._status_label.setText(self.tr("Loading chapter..."))
        self._load_chapter_url_btn.setEnabled(False)
        self._worker.run_load_chapter_url.emit(url)

    def _on_search(self):
        title = self._search_edit.text().strip()
        if not title:
            self._status_label.setText(self.tr("Enter a title to search."))
            return
        self._status_label.setText(self.tr("Searching..."))
        self._search_btn.setEnabled(False)
        self._worker.run_search.emit(title)

    def _on_search_finished(self, results: list):
        self._search_btn.setEnabled(True)
        self._manga_results = results
        self._results_list.clear()
        for r in results:
            item = QListWidgetItem(r.get("title", "?"))
            item.setData(Qt.ItemDataRole.UserRole, r)
            self._results_list.addItem(item)
        if not results:
            self._status_label.setText(self.tr("No results found."))
        else:
            self._status_label.setText(self.tr("Select a title, then click Load chapters."))
        self._load_ch_btn.setEnabled(bool(results))

    def _on_result_selected(self, row: int):
        self._load_ch_btn.setEnabled(row >= 0 and row < len(self._manga_results))
        if row >= 0 and row < len(self._manga_results):
            self._selected_manga = self._manga_results[row]

    def _on_load_chapters(self):
        if not self._selected_manga:
            cur = self._results_list.currentRow()
            if 0 <= cur < len(self._manga_results):
                self._selected_manga = self._manga_results[cur]
        if not self._selected_manga:
            self._status_label.setText(self.tr("Select a manga from the results first."))
            return
        manga_id = self._selected_manga.get("id")
        if not manga_id:
            return
        lang = self._lang_combo.currentData() or "en"
        self._status_label.setText(self.tr("Loading chapters..."))
        self._load_ch_btn.setEnabled(False)
        self._worker.run_feed.emit(manga_id, lang)

    def _on_feed_finished(self, chapters: list):
        self._load_ch_btn.setEnabled(True)
        if self._is_url_source:
            self._load_chapter_url_btn.setEnabled(True)
        self._chapters = chapters
        if self._is_url_source and len(chapters) == 1:
            self._selected_manga = {"id": None, "title": self.tr("Chapter from URL")}
        self._chapters_list.clear()
        for ch in chapters:
            item = QListWidgetItem(ch.get("display", "Ch.?"))
            item.setData(Qt.ItemDataRole.UserRole, ch)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            self._chapters_list.addItem(item)
        if self._is_url_source:
            if chapters:
                self._chapters_list.item(0).setCheckState(Qt.CheckState.Checked)
            self._status_label.setText(
                self.tr("Loaded 1 chapter. Check to download, then choose folder and click Download.") if chapters else self._status_label.text()
            )
        else:
            self._status_label.setText(
                self.tr("Loaded {0} chapter(s). Check the ones to download.").format(len(chapters))
            )

    def _on_select_all_chapters(self):
        for i in range(self._chapters_list.count()):
            self._chapters_list.item(i).setCheckState(Qt.CheckState.Checked)

    def _on_deselect_all_chapters(self):
        for i in range(self._chapters_list.count()):
            self._chapters_list.item(i).setCheckState(Qt.CheckState.Unchecked)

    def _on_browse(self):
        d = QFileDialog.getExistingDirectory(self, self.tr("Choose download folder"), self._download_base_dir)
        if d:
            self._download_base_dir = d
            self._folder_edit.setText(d)
            self._save_config_to_pcfg()

    def _save_config_to_pcfg(self):
        """Write current dialog values to pcfg and persist to disk."""
        pcfg.manga_source_lang = self._lang_combo.currentData() or "en"
        pcfg.manga_source_data_saver = self._data_saver_check.isChecked()
        pcfg.manga_source_download_dir = self._folder_edit.text().strip() or self._download_base_dir
        pcfg.manga_source_request_delay = self._delay_spin.value()
        pcfg.manga_source_open_after_download = self._open_after_download_check.isChecked()
        try:
            save_config()
        except Exception:
            pass

    def _on_delay_changed(self, value: float):
        pcfg.manga_source_request_delay = value
        try:
            save_config()
        except Exception:
            pass

    def _on_download(self):
        if not self._selected_manga:
            self._status_label.setText(self.tr("Select a manga and load chapters first."))
            return
        checked = []
        for i in range(self._chapters_list.count()):
            item = self._chapters_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                ch = item.data(Qt.ItemDataRole.UserRole)
                if ch:
                    checked.append(ch)
        if not checked:
            self._status_label.setText(self.tr("Select at least one chapter to download."))
            return
        base_dir = self._folder_edit.text().strip() or self._download_base_dir
        if not base_dir or not osp.isdir(base_dir):
            self._status_label.setText(self.tr("Choose a valid download folder."))
            return
        self._progress_bar.setRange(0, len(checked))
        self._progress_bar.setValue(0)
        self._status_label.setText(self.tr("Downloading..."))
        self._download_btn.setEnabled(False)
        self._worker.run_download.emit((
            checked,
            base_dir,
            self._selected_manga.get("title", "manga"),
            self._data_saver_check.isChecked(),
        ))

    def _on_download_progress(self, current: int, total: int, display: str):
        self._progress_bar.setMaximum(total)
        self._progress_bar.setValue(current)
        self._status_label.setText(self.tr("Downloaded {0} / {1}: {2}").format(current, total, display))

    def _on_download_finished(self, path: str):
        self._download_btn.setEnabled(True)
        self._progress_bar.setValue(self._progress_bar.maximum())
        self._status_label.setText(self.tr("Download complete: {0}").format(path))
        self._last_download_path = path
        self._last_download_first_chapter = None
        if path and osp.isdir(path):
            for name in sorted(os.listdir(path)):
                sub = osp.join(path, name)
                if osp.isdir(sub):
                    self._last_download_first_chapter = sub
                    break
        self._open_folder_btn.setEnabled(True)
        QMessageBox.information(
            self,
            self.tr("Download complete"),
            self.tr("Chapters saved to:\n{0}\n\nYou can open a chapter folder in BallonsTranslator to translate.").format(path),
        )
        if self._open_after_download_check.isChecked():
            to_open = self._last_download_first_chapter or self._last_download_path
            if to_open and osp.isdir(to_open):
                self.open_folder_requested.emit(to_open)
                self.accept()

    def _on_open_folder(self):
        to_open = self._last_download_first_chapter or self._last_download_path
        if to_open and osp.isdir(to_open):
            self.open_folder_requested.emit(to_open)
            self.accept()

    def _on_worker_error(self, msg: str):
        self._search_btn.setEnabled(True)
        self._load_ch_btn.setEnabled(True)
        self._load_chapter_url_btn.setEnabled(True)
        self._download_btn.setEnabled(True)
        self._status_label.setText(msg)
        QMessageBox.warning(self, self.tr("Error"), msg)

    def closeEvent(self, event):
        self._save_config_to_pcfg()
        self._worker.abort()
        self._thread.quit()
        self._thread.wait(2000)
        super().closeEvent(event)
