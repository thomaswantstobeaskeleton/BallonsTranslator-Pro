"""
Dialog to search for manga/manhua/manhwa titles, list chapters, and download selected chapters.
Uses MangaDex API by default. Opens from Tools → Manga / Comic source...
"""
from __future__ import annotations

import os
import os.path as osp
import re
import threading
from typing import Any, List, Optional

from qtpy.QtCore import Qt, Signal, QThread, QObject, QTimer
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

from utils.manga_sources import (
    MangaDexClient,
    ComickSourceClient,
    GomangaApiClient,
    ManhwaReaderClient,
    GenericChapterUrlClient,
    MangaForFreeClient,
    ToonGodClient,
    MangaNatoClient,
    MangaFireClient,
    NaruRawClient,
    ManhwaRawClient,
    OneKkkClient,
)
from utils.manga_sources.generic_chapter_url import ensure_playwright_chromium_installed
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


# Source combo: (display label, source_id). Manhwa Reader is shown only when API is up.
# generic_chapter_url and *_url sources use paste-chapter-URL flow (same backend).
GENERIC_CHAPTER_SOURCE_IDS = ("generic_chapter_url", "mangafire_url", "manganato_url", "raws_manhwa_manhua_url")
SOURCE_OPTIONS: List[tuple] = [
    ("MangaDex", "mangadex"),
    ("MangaDex (raw / original language)", "mangadex_raw"),
    ("MangaDex (by chapter URL)", "mangadex_url"),
    ("Comick", "comick"),
    ("GOMANGA", "gomanga"),
    ("Manhwa Reader", "manhwa_reader"),
    ("MangaForFree", "mangaforfree"),
    ("ToonGod", "toongod"),
    ("MangaNato", "manganato"),
    ("MangaFire", "mangafire"),
    ("NaruRaw (Japanese raw)", "naruraw"),
    ("ManhwaRaw (Korean raw)", "manhwaraw"),
    ("1kkk (Chinese manhua)", "onekkk"),
    ("Generic (chapter URL)", "generic_chapter_url"),
    ("MangaFire (chapter URL)", "mangafire_url"),
    ("MangaNato (chapter URL)", "manganato_url"),
    ("Raws / Manhwa / Manhua (chapter URL)", "raws_manhwa_manhua_url"),
    ("Local folder", "local_folder"),
]


def _mangadex_original_language_code(ui_code: str) -> str:
    """Map UI language code to MangaDex API originalLanguage (e.g. zh-hans -> zh)."""
    if not ui_code:
        return "ja"
    if ui_code == "zh-hans":
        return "zh"
    if ui_code == "zh-hant":
        return "zh-hk"
    return ui_code


class MangaSourceWorker(QObject):
    """Runs API calls and downloads in a background thread."""
    search_finished = Signal(list)   # list of {id, title, description}
    feed_finished = Signal(list)     # list of {id, chapter, display, ...}
    error = Signal(str)
    download_progress = Signal(int, int, str)  # current, total, chapter_display
    download_finished = Signal(str)  # path to folder that was downloaded
    manhwa_reader_available = Signal(bool)  # True if Manhwa Reader API is up
    run_search = Signal(str, str, str, bool, bool)   # title, source_id, lang, use_playwright, headless
    run_feed = Signal(str, str, str, bool, bool)  # manga_id, lang, source_id, use_playwright, headless
    run_download = Signal(object)  # (chapter_infos, base_dir, manga_title, data_saver, source_id [, use_playwright, headless])
    run_load_chapter_url = Signal(str)
    run_load_generic_chapter_url = Signal(str, bool, bool)  # chapter_url, use_playwright, headless
    run_check_manhwa_reader = Signal()

    def __init__(self):
        super().__init__()
        delay = getattr(pcfg, 'manga_source_request_delay', 0.3)
        self._mangadex = MangaDexClient(timeout=30, request_delay=delay)
        self._comick = ComickSourceClient(timeout=30, request_delay=delay)
        self._gomanga = GomangaApiClient(timeout=30, request_delay=delay)
        self._manhwa_reader = ManhwaReaderClient(timeout=30, request_delay=delay)
        self._generic_chapter = GenericChapterUrlClient(timeout=25, request_delay=delay)
        self._mangaforfree = MangaForFreeClient(base_url="https://mangaforfree.com", timeout=25, request_delay=delay)
        self._toongod = ToonGodClient(base_url="https://toongod.org", timeout=25, request_delay=delay)
        self._manganato = MangaNatoClient(base_url="https://manganato.com", timeout=25, request_delay=delay)
        self._mangafire = MangaFireClient(base_url="https://mangafire.to", timeout=25, request_delay=delay)
        self._naruraw = NaruRawClient(base_url="https://naruraw.net", timeout=25, request_delay=delay)
        self._manhwaraw = ManhwaRawClient(base_url="https://manhwaraw.club", timeout=25, request_delay=delay)
        self._onekkk = OneKkkClient(base_url="https://www.1kkk.com", timeout=25, request_delay=delay)
        self._abort = False
        self.run_search.connect(self.do_search)
        self.run_feed.connect(self.do_feed)
        self.run_download.connect(self._do_download)
        self.run_load_chapter_url.connect(self.do_load_chapter_by_url)
        self.run_load_generic_chapter_url.connect(self.do_load_generic_chapter_url)
        self.run_check_manhwa_reader.connect(self._do_check_manhwa_reader)

    def abort(self):
        self._abort = True

    def _do_check_manhwa_reader(self):
        try:
            ok = self._manhwa_reader.is_available(timeout_override=8)
            self.manhwa_reader_available.emit(ok)
        except Exception:
            self.manhwa_reader_available.emit(False)

    def do_search(self, title: str, source_id: str, lang: str = "", use_playwright: bool = False, headless: bool = True):
        self._abort = False
        delay = getattr(pcfg, 'manga_source_request_delay', 0.3)
        if source_id == "comick":
            self._comick.request_delay = delay
            try:
                results = self._comick.search(title, limit=25)
                self.search_finished.emit(results)
            except Exception as e:
                self.error.emit(str(e))
        elif source_id == "gomanga":
            self._gomanga.request_delay = delay
            try:
                results = self._gomanga.search(title, limit=25)
                self.search_finished.emit(results)
            except Exception as e:
                self.error.emit(str(e))
        elif source_id == "manhwa_reader":
            self._manhwa_reader.request_delay = delay
            try:
                results = self._manhwa_reader.search(title, limit=25)
                self.search_finished.emit(results)
            except Exception as e:
                self.error.emit(str(e))
        elif source_id == "mangaforfree":
            self._mangaforfree.request_delay = delay
            try:
                results = self._mangaforfree.search(title, limit=25, use_playwright=use_playwright, headless=headless)
                self.search_finished.emit(results)
            except Exception as e:
                self.error.emit(str(e))
        elif source_id == "toongod":
            self._toongod.request_delay = delay
            try:
                results = self._toongod.search(title, limit=25, use_playwright=use_playwright, headless=headless)
                self.search_finished.emit(results)
            except Exception as e:
                self.error.emit(str(e))
        elif source_id == "manganato":
            self._manganato.request_delay = delay
            try:
                results = self._manganato.search(title, limit=25, use_playwright=use_playwright, headless=headless)
                self.search_finished.emit(results)
            except Exception as e:
                self.error.emit(str(e))
        elif source_id == "mangafire":
            self._mangafire.request_delay = delay
            try:
                results = self._mangafire.search(title, limit=25, use_playwright=use_playwright, headless=headless)
                self.search_finished.emit(results)
            except Exception as e:
                self.error.emit(str(e))
        elif source_id == "naruraw":
            self._naruraw.request_delay = delay
            try:
                results = self._naruraw.search(title, limit=25, use_playwright=use_playwright, headless=headless)
                self.search_finished.emit(results)
            except Exception as e:
                self.error.emit(str(e))
        elif source_id == "manhwaraw":
            self._manhwaraw.request_delay = delay
            try:
                results = self._manhwaraw.search(title, limit=25, use_playwright=use_playwright, headless=headless)
                self.search_finished.emit(results)
            except Exception as e:
                self.error.emit(str(e))
        elif source_id == "onekkk":
            self._onekkk.request_delay = delay
            try:
                results = self._onekkk.search(title, limit=25, use_playwright=use_playwright, headless=headless)
                self.search_finished.emit(results)
            except Exception as e:
                self.error.emit(str(e))
        elif source_id == "mangadex_raw":
            self._mangadex.request_delay = delay
            try:
                raw_lang = _mangadex_original_language_code(lang or "ja")
                results = self._mangadex.search(
                    title, limit=25, original_language=raw_lang
                )
                self.search_finished.emit(results)
            except Exception as e:
                self.error.emit(str(e))
        else:
            self._mangadex.request_delay = delay
            try:
                results = self._mangadex.search(title, limit=25)
                self.search_finished.emit(results)
            except Exception as e:
                self.error.emit(str(e))

    def do_feed(self, manga_id: str, lang: str, source_id: str, use_playwright: bool = False, headless: bool = True):
        self._abort = False
        delay = getattr(pcfg, 'manga_source_request_delay', 0.3)
        if source_id == "comick":
            self._comick.request_delay = delay
            try:
                chapters = self._comick.get_feed(manga_id, translated_language=lang, limit=500, order="asc")
                self.feed_finished.emit(chapters)
            except Exception as e:
                self.error.emit(str(e))
        elif source_id == "gomanga":
            self._gomanga.request_delay = delay
            try:
                chapters = self._gomanga.get_feed(manga_id, translated_language=lang, limit=500, order="asc")
                self.feed_finished.emit(chapters)
            except Exception as e:
                self.error.emit(str(e))
        elif source_id == "manhwa_reader":
            self._manhwa_reader.request_delay = delay
            try:
                chapters = self._manhwa_reader.get_feed(manga_id, translated_language=lang, limit=500, order="asc")
                self.feed_finished.emit(chapters)
            except Exception as e:
                self.error.emit(str(e))
        elif source_id == "mangaforfree":
            self._mangaforfree.request_delay = delay
            try:
                chapters = self._mangaforfree.get_feed(manga_id, translated_language=lang, limit=500, order="asc", use_playwright=use_playwright, headless=headless)
                self.feed_finished.emit(chapters)
            except Exception as e:
                self.error.emit(str(e))
        elif source_id == "toongod":
            self._toongod.request_delay = delay
            try:
                chapters = self._toongod.get_feed(manga_id, translated_language=lang, limit=500, order="asc", use_playwright=use_playwright, headless=headless)
                self.feed_finished.emit(chapters)
            except Exception as e:
                self.error.emit(str(e))
        elif source_id == "manganato":
            self._manganato.request_delay = delay
            try:
                chapters = self._manganato.get_feed(manga_id, translated_language=lang, limit=500, order="asc", use_playwright=use_playwright, headless=headless)
                self.feed_finished.emit(chapters)
            except Exception as e:
                self.error.emit(str(e))
        elif source_id == "mangafire":
            self._mangafire.request_delay = delay
            try:
                chapters = self._mangafire.get_feed(manga_id, translated_language=lang, limit=500, order="asc", use_playwright=use_playwright, headless=headless)
                self.feed_finished.emit(chapters)
            except Exception as e:
                self.error.emit(str(e))
        elif source_id == "naruraw":
            self._naruraw.request_delay = delay
            try:
                chapters = self._naruraw.get_feed(manga_id, translated_language=lang, limit=500, order="asc", use_playwright=use_playwright, headless=headless)
                self.feed_finished.emit(chapters)
            except Exception as e:
                self.error.emit(str(e))
        elif source_id == "manhwaraw":
            self._manhwaraw.request_delay = delay
            try:
                chapters = self._manhwaraw.get_feed(manga_id, translated_language=lang, limit=500, order="asc", use_playwright=use_playwright, headless=headless)
                self.feed_finished.emit(chapters)
            except Exception as e:
                self.error.emit(str(e))
        elif source_id == "onekkk":
            self._onekkk.request_delay = delay
            try:
                chapters = self._onekkk.get_feed(manga_id, translated_language=lang, limit=500, order="asc", use_playwright=use_playwright, headless=headless)
                self.feed_finished.emit(chapters)
            except Exception as e:
                self.error.emit(str(e))
        elif source_id == "mangadex_raw":
            self._mangadex.request_delay = delay
            try:
                feed_lang = lang or "ja"
                if feed_lang == "zh-hans":
                    feed_lang = "zh"
                elif feed_lang == "zh-hant":
                    feed_lang = "zh-hk"
                chapters = self._mangadex.get_feed(manga_id, translated_language=feed_lang, limit=500, order="asc")
                self.feed_finished.emit(chapters)
            except Exception as e:
                self.error.emit(str(e))
        else:
            self._mangadex.request_delay = delay
            try:
                chapters = self._mangadex.get_feed(manga_id, translated_language=lang, limit=500, order="asc")
                self.feed_finished.emit(chapters)
            except Exception as e:
                self.error.emit(str(e))

    def do_download(
        self,
        chapter_infos: List[dict],
        base_dir: str,
        manga_title: str,
        data_saver: bool,
        source_id: str = "mangadex",
        use_playwright: bool = False,
        headless: bool = True,
    ):
        if source_id == "comick":
            self.error.emit(
                "Download is not supported for Comick source. The API does not provide image URLs. "
                "Use MangaDex for download, or open the chapter link in your browser."
            )
            return
        if source_id in ("mangaforfree", "toongod", "manganato", "mangafire", "naruraw", "manhwaraw", "onekkk") or source_id in GENERIC_CHAPTER_SOURCE_IDS:
            self._do_download_generic_chapter_url(
                chapter_infos, base_dir, manga_title, data_saver, use_playwright=use_playwright, headless=headless
            )
            return
        self._abort = False
        delay = getattr(pcfg, 'manga_source_request_delay', 0.3)
        if source_id == "gomanga":
            client = self._gomanga
        elif source_id == "manhwa_reader":
            client = self._manhwa_reader
        else:
            client = self._mangadex
        client.request_delay = delay
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
                result = client.download_chapter(
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
        """Slot for run_download: payload = (chapter_infos, base_dir, manga_title, data_saver, source_id [, use_playwright, headless])."""
        if len(payload) >= 7:
            self.do_download(payload[0], payload[1], payload[2], payload[3], payload[4], payload[5], payload[6])
        elif len(payload) >= 6:
            self.do_download(payload[0], payload[1], payload[2], payload[3], payload[4], payload[5])
        elif len(payload) >= 5:
            self.do_download(payload[0], payload[1], payload[2], payload[3], payload[4])
        else:
            self.do_download(payload[0], payload[1], payload[2], payload[3], "mangadex")

    def do_load_chapter_by_url(self, url_or_id: str):
        """Load a single chapter by MangaDex chapter URL or UUID. Emits feed_finished([ch]) or error."""
        self._abort = False
        self._mangadex.request_delay = getattr(pcfg, 'manga_source_request_delay', 0.3)
        ch_id = _extract_mangadex_chapter_id(url_or_id)
        if not ch_id:
            self.error.emit("Invalid MangaDex chapter URL or ID.")
            return
        try:
            ch = self._mangadex.get_chapter_by_id(ch_id)
            if ch:
                self.feed_finished.emit([ch])
            else:
                self.error.emit("Chapter not found.")
        except Exception as e:
            self.error.emit(str(e))

    def do_load_generic_chapter_url(self, chapter_url: str, use_playwright: bool = False, headless: bool = True):
        """Load a single chapter by generic chapter page URL. Fetches HTML (or Playwright if use_playwright), extracts image count; emits feed_finished([ch]) or error."""
        self._abort = False
        delay = getattr(pcfg, 'manga_source_request_delay', 0.3)
        self._generic_chapter.request_delay = delay
        if not (chapter_url or chapter_url.strip()):
            self.error.emit("Enter a chapter page URL.")
            return
        try:
            urls = self._generic_chapter.get_chapter_images(chapter_url.strip(), use_playwright=use_playwright, headless=headless)
            if not urls:
                self.error.emit(
                    "No images found at this URL. The site may require JavaScript (e.g. Cloudflare); try another source or use a browser extension."
                )
                return
            ch = {
                "id": chapter_url.strip(),
                "display": "Chapter (%d images)" % len(urls),
                "image_count": len(urls),
            }
            self.feed_finished.emit([ch])
        except Exception as e:
            self.error.emit(str(e))

    def _do_download_generic_chapter_url(
        self,
        chapter_infos: List[dict],
        base_dir: str,
        manga_title: str,
        data_saver: bool,
        use_playwright: bool = False,
        headless: bool = True,
    ):
        """Download chapters using GenericChapterUrlClient (chapter_infos[].id = chapter URL).
        Used for Generic (chapter URL), MangaForFree, and ToonGod."""
        self._abort = False
        delay = getattr(pcfg, 'manga_source_request_delay', 0.3)
        self._generic_chapter.request_delay = delay
        manga_safe = _sanitize_filename(manga_title)
        parent_dir = osp.join(base_dir, manga_safe)
        os.makedirs(parent_dir, exist_ok=True)
        total_ch = len(chapter_infos)
        for i, ch in enumerate(chapter_infos):
            if self._abort:
                self.error.emit("Download aborted.")
                return
            ch_url = ch.get("id")
            display = ch.get("display", "Chapter")
            ch_safe = _sanitize_filename(display)
            save_dir = osp.join(parent_dir, ch_safe)
            try:
                result = self._generic_chapter.download_chapter(
                    ch_url,
                    save_dir,
                    manga_id=manga_safe,
                    chapter_id=ch_safe,
                    on_progress=None,
                    use_playwright=use_playwright,
                    headless=headless,
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
        self._is_local_folder = False
        self._is_comick = False
        self._is_generic_chapter_url = False
        # Manhwa Reader is hidden until we confirm the API is up
        self._unavailable_sources: set = {"manhwa_reader"}

        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(10, 10, 10, 10)

        # --- Source & Search ---
        search_group = QGroupBox(self.tr("Search"))
        search_layout = QVBoxLayout(search_group)
        row = QHBoxLayout()
        self._source_combo = QComboBox()
        self._populate_source_combo()
        self._source_combo.setToolTip(
            self.tr("Manga source. MangaDex: translated chapters. MangaDex (raw): search by original language and download raw chapters to translate. GOMANGA and Manhwa Reader support search and download. Comick supports search and chapter list only. Local folder opens a folder of images as a project.")
        )
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
        self._use_playwright_check = QCheckBox(self.tr("Use browser (Playwright) for JS-heavy sites"))
        self._use_playwright_check.setToolTip(
            self.tr("Enable for sites that load images with JavaScript or use Cloudflare. Requires: pip install playwright && playwright install chromium")
        )
        self._use_playwright_check.toggled.connect(self._on_playwright_toggled)
        row_url.addWidget(self._use_playwright_check)
        self._playwright_headless_check = QCheckBox(self.tr("Run browser in background (hidden)"))
        self._playwright_headless_check.setToolTip(
            self.tr("When checked, the browser runs without a visible window so it does not interfere with the menu. Uncheck to show the browser window (e.g. for debugging).")
        )
        self._playwright_headless_check.setChecked(getattr(pcfg, 'manga_source_playwright_headless', True))
        self._playwright_headless_check.toggled.connect(self._save_config_to_pcfg)
        row_url.addWidget(self._playwright_headless_check)
        self._install_chromium_btn = QPushButton(self.tr("Install Chromium"))
        self._install_chromium_btn.setToolTip(self.tr("Ensure Playwright and Chromium are installed. Run this if browser features fail."))
        self._install_chromium_btn.clicked.connect(self._on_install_chromium)
        row_url.addWidget(self._install_chromium_btn)
        search_layout.addLayout(row_url)
        self._url_row_widgets = [self._url_edit, self._load_chapter_url_btn]
        self._search_row_widgets = [self._search_edit, self._search_btn]
        row_local = QHBoxLayout()
        self._local_folder_label = QLabel(self.tr("Open a folder of images to use as a project."))
        self._open_local_folder_btn = QPushButton(self.tr("Open folder in BallonsTranslator"))
        self._open_local_folder_btn.clicked.connect(self._on_open_local_folder)
        row_local.addWidget(self._local_folder_label)
        row_local.addWidget(self._open_local_folder_btn)
        row_local.addStretch()
        search_layout.addLayout(row_local)
        self._local_folder_widgets = [self._local_folder_label, self._open_local_folder_btn]
        layout.addWidget(search_group)

        # --- Results (manga list) ---
        results_group = QGroupBox(self.tr("Results"))
        self._results_group = results_group
        results_layout = QVBoxLayout(results_group)
        self._results_list = QListWidget()
        self._results_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._results_list.setMinimumHeight(100)
        self._results_list.currentRowChanged.connect(self._on_result_selected)
        results_layout.addWidget(self._results_list)
        layout.addWidget(results_group)

        # --- Chapters ---
        ch_group = QGroupBox(self.tr("Chapters"))
        self._ch_group = ch_group
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
        self._lang_label = QLabel(self.tr("Language:"))
        row_ch.addWidget(self._lang_label)
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
        self._dl_group = dl_group
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
        self._worker.manhwa_reader_available.connect(self._on_manhwa_reader_availability)

        # Load persisted config
        lang = getattr(pcfg, 'manga_source_lang', 'en')
        idx = self._lang_combo.findData(lang)
        if idx >= 0:
            self._lang_combo.setCurrentIndex(idx)
        self._data_saver_check.setChecked(getattr(pcfg, 'manga_source_data_saver', False))
        self._folder_edit.setText(self._download_base_dir)
        self._delay_spin.setValue(getattr(pcfg, 'manga_source_request_delay', 0.3))
        self._open_after_download_check.setChecked(getattr(pcfg, 'manga_source_open_after_download', False))
        self._playwright_headless_check.setChecked(getattr(pcfg, 'manga_source_playwright_headless', True))
        self._on_source_changed(self._source_combo.currentIndex())
        QTimer.singleShot(100, self._start_manhwa_reader_check)
        self._on_playwright_toggled(self._use_playwright_check.isChecked())

    def _on_playwright_toggled(self, checked: bool):
        """Keep dialog on top when using browser so it does not hide behind the browser window."""
        effective = bool(checked and self._use_playwright_check.isVisible())
        base = (
            Qt.WindowType.Window
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowMaximizeButtonHint
            | Qt.WindowType.WindowCloseButtonHint
        )
        if effective:
            self.setWindowFlags(base | Qt.WindowType.WindowStaysOnTopHint)
        else:
            self.setWindowFlags(base)
        self.show()

    def _populate_source_combo(self):
        current_id = self._source_combo.currentData() if self._source_combo.currentIndex() >= 0 else None
        self._source_combo.blockSignals(True)
        self._source_combo.clear()
        for label, source_id in SOURCE_OPTIONS:
            if source_id not in self._unavailable_sources:
                self._source_combo.addItem(self.tr(label), source_id)
        idx = self._source_combo.findData(current_id)
        if idx >= 0:
            self._source_combo.setCurrentIndex(idx)
        self._source_combo.blockSignals(False)

    def _start_manhwa_reader_check(self):
        self._worker.run_check_manhwa_reader.emit()

    def _on_manhwa_reader_availability(self, available: bool):
        if available:
            self._unavailable_sources.discard("manhwa_reader")
        else:
            self._unavailable_sources.add("manhwa_reader")
        self._populate_source_combo()

    def _on_source_changed(self, index: int):
        source = self._source_combo.currentData() if index >= 0 else None
        self._is_url_source = source == "mangadex_url"
        self._is_local_folder = source == "local_folder"
        self._is_comick = source == "comick"
        self._is_generic_chapter_url = source in GENERIC_CHAPTER_SOURCE_IDS
        is_raw = source == "mangadex_raw"
        if is_raw:
            self._lang_label.setText(self.tr("Raw language (chapters to load):"))
            if self._lang_combo.currentData() == "en":
                idx = self._lang_combo.findData("ja")
                if idx >= 0:
                    self._lang_combo.setCurrentIndex(idx)
        else:
            self._lang_label.setText(self.tr("Language:"))
        for w in self._search_row_widgets:
            w.setVisible(not self._is_url_source and not self._is_local_folder and not self._is_generic_chapter_url)
        for w in self._url_row_widgets:
            w.setVisible(self._is_url_source or self._is_generic_chapter_url)
        self._use_playwright_check.setVisible(
            self._is_generic_chapter_url or source in ("mangaforfree", "toongod", "manganato", "mangafire", "naruraw", "manhwaraw", "onekkk")
        )
        headless_visible = self._use_playwright_check.isVisible()
        self._playwright_headless_check.setVisible(headless_visible)
        self._install_chromium_btn.setVisible(headless_visible)
        if self._is_generic_chapter_url:
            if source == "mangafire_url":
                self._url_edit.setPlaceholderText(self.tr("Paste MangaFire chapter page URL..."))
            elif source == "manganato_url":
                self._url_edit.setPlaceholderText(self.tr("Paste MangaNato chapter page URL..."))
            elif source == "raws_manhwa_manhua_url":
                self._url_edit.setPlaceholderText(self.tr("Paste chapter URL (raw manga, manhwa, manhua sites)..."))
            else:
                self._url_edit.setPlaceholderText(self.tr("Paste chapter page URL (HTML with images)..."))
        else:
            self._url_edit.setPlaceholderText(self.tr("Paste MangaDex chapter URL or chapter UUID..."))
        for w in self._local_folder_widgets:
            w.setVisible(self._is_local_folder)
        self._results_group.setVisible(not self._is_local_folder)
        self._ch_group.setVisible(not self._is_local_folder)
        self._dl_group.setVisible(not self._is_local_folder)
        if self._is_url_source:
            self._results_list.clear()
            self._manga_results = []
            self._load_ch_btn.setEnabled(False)
        if self._is_generic_chapter_url:
            self._results_list.clear()
            self._manga_results = []
            self._selected_manga = {"id": None, "title": self.tr("Chapter from URL")}
            self._load_ch_btn.setEnabled(False)
        if self._is_comick:
            self._download_btn.setToolTip(self.tr("Download is not supported for Comick. Use MangaDex to download, or open chapter links in your browser."))
        else:
            self._download_btn.setToolTip("")
        if self._is_local_folder:
            self._status_label.setText(self.tr("Click the button to open a folder of images as a project."))
        elif self._is_url_source:
            self._status_label.setText("" if self._is_url_source else self._status_label.text())
        elif self._is_generic_chapter_url:
            self._status_label.setText(self.tr("Paste a chapter page URL and click Load chapter. For JS-heavy sites enable \"Use browser (Playwright)\"."))
        elif source in ("mangaforfree", "toongod", "manganato", "mangafire", "naruraw", "manhwaraw", "onekkk"):
            self._status_label.setText(self.tr("Search by title, load chapters, then download. Enable \"Use browser (Playwright)\" if the site uses Cloudflare or lazy-loaded images."))
        self._on_playwright_toggled(self._use_playwright_check.isChecked())

    def _on_load_chapter_url(self):
        url = self._url_edit.text().strip()
        if self._is_generic_chapter_url:
            if not url:
                self._status_label.setText(self.tr("Enter a chapter page URL."))
                return
            self._status_label.setText(self.tr("Loading chapter..."))
            self._load_chapter_url_btn.setEnabled(False)
            self._worker.run_load_generic_chapter_url.emit(url, self._use_playwright_check.isChecked(), self._playwright_headless_check.isChecked())
            return
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
        source_id = self._source_combo.currentData() or "mangadex"
        if source_id in ("mangadex_url", "local_folder"):
            return
        self._status_label.setText(self.tr("Searching..."))
        self._search_btn.setEnabled(False)
        lang = self._lang_combo.currentData() or "en"
        use_playwright = self._use_playwright_check.isChecked() if source_id in ("mangaforfree", "toongod", "manganato", "mangafire", "naruraw", "manhwaraw", "onekkk") else False
        headless = self._playwright_headless_check.isChecked()
        self._worker.run_search.emit(title, source_id, lang, use_playwright, headless)

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
        source_id = self._source_combo.currentData() or "mangadex"
        if source_id in ("mangadex_url", "local_folder"):
            return
        lang = self._lang_combo.currentData() or "en"
        self._status_label.setText(self.tr("Loading chapters..."))
        self._load_ch_btn.setEnabled(False)
        use_playwright = self._use_playwright_check.isChecked() if source_id in ("mangaforfree", "toongod", "manganato", "mangafire", "naruraw", "manhwaraw", "onekkk") else False
        headless = self._playwright_headless_check.isChecked()
        self._worker.run_feed.emit(manga_id, lang, source_id, use_playwright, headless)

    def _on_feed_finished(self, chapters: list):
        self._load_ch_btn.setEnabled(True)
        if self._is_url_source or self._is_generic_chapter_url:
            self._load_chapter_url_btn.setEnabled(True)
        self._chapters = chapters
        if self._is_url_source and len(chapters) == 1:
            self._selected_manga = {"id": None, "title": self.tr("Chapter from URL")}
        if self._is_generic_chapter_url and len(chapters) == 1:
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
        pcfg.manga_source_playwright_headless = self._playwright_headless_check.isChecked()
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
        source_id = self._source_combo.currentData() or "mangadex"
        payload = (
            checked,
            base_dir,
            self._selected_manga.get("title", "manga"),
            self._data_saver_check.isChecked(),
            source_id,
        )
        if source_id in ("mangaforfree", "toongod", "manganato", "mangafire", "naruraw", "manhwaraw", "onekkk") or source_id in GENERIC_CHAPTER_SOURCE_IDS:
            payload = payload + (self._use_playwright_check.isChecked(), self._playwright_headless_check.isChecked())
        self._worker.run_download.emit(payload)

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

    def _on_open_local_folder(self):
        start = self._folder_edit.text().strip() or self._download_base_dir or ""
        folder = QFileDialog.getExistingDirectory(self, self.tr("Open folder in BallonsTranslator"), start)
        if folder and osp.isdir(folder):
            self.open_folder_requested.emit(folder)
            self.accept()

    def _on_install_chromium(self):
        """Run playwright install chromium in a background thread and show result."""
        self._install_chromium_btn.setEnabled(False)
        self._status_label.setText(self.tr("Installing Chromium..."))
        result_holder: List[tuple] = []

        def run():
            result_holder.append(ensure_playwright_chromium_installed())

        thread = threading.Thread(target=run, daemon=True)
        thread.start()

        def check():
            if not thread.is_alive():
                self._install_chromium_btn.setEnabled(True)
                if result_holder:
                    ok, msg = result_holder[0]
                    self._status_label.setText(msg)
                    if ok:
                        QMessageBox.information(self, self.tr("Install Chromium"), msg)
                    else:
                        QMessageBox.warning(self, self.tr("Install Chromium"), msg)
                return
            QTimer.singleShot(200, check)

        QTimer.singleShot(200, check)

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
