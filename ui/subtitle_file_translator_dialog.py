"""
Translate standalone .srt or timestamped .txt subtitle files using the app's configured translator.

Recommended: Config → Translator → **LLM_API_Translator** with provider **OpenRouter** and your model.
Uses the same JSON batching options as the video translator (video_translator_nlp_chunk_size / max_workers).
"""
from __future__ import annotations

import os.path as osp
import traceback

from qtpy.QtCore import Qt, QThread, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from modules import GET_VALID_TRANSLATORS, TRANSLATORS
from modules.translators.base import LANGMAP_GLOBAL, lang_display_label, lang_display_to_key
from modules.translators.exceptions import CriticalTranslationError
from modules.video_translator import clear_translator_video_nlp_parallel, configure_translator_video_nlp_parallel
from utils.config import pcfg
from utils.logger import logger as LOGGER
from utils.series_context_store import get_series_context_dir, load_recent_context
from utils.subtitle_cue_io import (
    FormatHint,
    SubtitleCue,
    parse_subtitle_content,
    write_srt,
    write_timestamped_txt,
)

from ui.translation_context_dialog import _text_to_glossary


def _llm_translator_series_path_default() -> str:
    """Default series context ID from Config → LLM_API_Translator, if set."""
    try:
        tp = (pcfg.module.translator_params.get("LLM_API_Translator") or {})
        sc = tp.get("series_context_path") or {}
        if isinstance(sc, dict):
            return (sc.get("value") or "").strip()
    except Exception:
        pass
    return ""


class SubtitleFileTranslateThread(QThread):
    progress = Signal(int, int)
    finished_ok = Signal(list)
    failed = Signal(str)

    def __init__(
        self,
        texts: list,
        lang_source: str,
        lang_target: str,
        glossary_hint: str,
        series_context_path: str = "",
        project_glossary: list | None = None,
        use_recent_series_context: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self.texts = texts
        self.lang_source = lang_source
        self.lang_target = lang_target
        self.glossary_hint = (glossary_hint or "").strip()
        self.series_context_path = (series_context_path or "").strip()
        self.project_glossary = project_glossary or []
        self.use_recent_series_context = bool(use_recent_series_context)

    def run(self):
        n = len(self.texts)
        self.progress.emit(0, max(n, 1))
        translator = None
        try:
            name = pcfg.module.translator
            valid = GET_VALID_TRANSLATORS()
            if name not in valid:
                self.failed.emit(
                    self.tr("Translator '%s' is not available. Check Config → Translator.") % name
                )
                return
            if name not in TRANSLATORS.module_dict:
                self.failed.emit(self.tr("Translator '%s' is not registered.") % name)
                return
            params = (pcfg.module.translator_params.get(name) or {}).copy()
            params["lang_source"] = self.lang_source
            params["lang_target"] = self.lang_target
            try:
                translator = TRANSLATORS.module_dict[name](**params)
            except Exception as e:
                self.failed.emit(
                    self.tr("Could not start translator: {0}").format(e)
                )
                return

            setattr(translator, "_current_page_key", "subtitle_file")
            configure_translator_video_nlp_parallel(translator, pcfg.module)
            if self.glossary_hint:
                setattr(translator, "_video_glossary_hint", self.glossary_hint)

            ui_series = self.series_context_path
            resolved_dir = ""
            if ui_series:
                resolved_dir = get_series_context_dir(ui_series)
            else:
                getter = getattr(translator, "_get_series_context_path", None)
                resolved_dir = (getter() or "") if callable(getter) else ""

            previous_pages: list = []
            if self.use_recent_series_context and resolved_dir:
                try:
                    n_prev = max(
                        1,
                        int(getattr(translator, "context_previous_pages_count", 1) or 1),
                    )
                except (TypeError, ValueError):
                    n_prev = 1
                previous_pages = load_recent_context(resolved_dir, max_pages=n_prev)

            series_kw = None
            if ui_series:
                series_kw = ui_series
            elif self.use_recent_series_context and resolved_dir:
                # Match glossary/recent store when only "use recent" is on (path from translator param).
                try:
                    p = (translator.get_param_value("series_context_path") or "").strip()
                except Exception:
                    p = ""
                series_kw = p if p else None

            translator.set_translation_context(
                previous_pages=previous_pages,
                project_glossary=list(self.project_glossary),
                series_context_path=series_kw,
                next_page=None,
            )

            out = translator.translate(list(self.texts))
            if not isinstance(out, list) or len(out) != n:
                self.failed.emit(
                    self.tr("Translator returned %s lines; expected %s.")
                    % (len(out) if isinstance(out, list) else "?", n)
                )
                return
            self.progress.emit(n, n)
            self.finished_ok.emit(out)
        except CriticalTranslationError as e:
            self.failed.emit(str(e) or self.tr("Translation failed (API error)."))
        except Exception as e:
            LOGGER.error("Subtitle file translate: %s\n%s", e, traceback.format_exc())
            self.failed.emit(f"{type(e).__name__}: {e}")
        finally:
            if translator is not None:
                clear_translator_video_nlp_parallel(translator)
                for attr in ("_current_page_key", "_video_glossary_hint"):
                    try:
                        delattr(translator, attr)
                    except Exception:
                        pass
            try:
                del translator
            except Exception:
                pass


class SubtitleFileTranslatorDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Translate subtitle file…"))
        self.resize(580, 680)
        self._input_path = ""
        self._cues: list[SubtitleCue] = []
        self._detected_format: str = "srt"
        self._thread: SubtitleFileTranslateThread | None = None

        root = QVBoxLayout(self)

        hint = QLabel(
            self.tr(
                "Uses the translator selected in Config (recommended: LLM_API_Translator + OpenRouter). "
                "Batching follows Video translator NLP settings (chunk size / parallel workers)."
            )
        )
        hint.setWordWrap(True)
        root.addWidget(hint)

        file_row = QHBoxLayout()
        self.path_label = QLabel(self.tr("(no file)"))
        self.path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.btn_open = QPushButton(self.tr("Open…"))
        self.btn_open.clicked.connect(self._on_open)
        file_row.addWidget(self.path_label, 1)
        file_row.addWidget(self.btn_open)
        root.addLayout(file_row)

        form_box = QGroupBox(self.tr("Input"))
        form = QFormLayout(form_box)
        self.format_combo = QComboBox()
        self.format_combo.addItem(self.tr("Auto-detect"), "auto")
        self.format_combo.addItem(self.tr("SubRip (.srt)"), "srt")
        self.format_combo.addItem(self.tr("Timestamped text (.txt)"), "txt")
        self.format_combo.setToolTip(
            self.tr(
                "Timestamped text: one cue per line, e.g. [00:01:02,345 --> 00:01:05,678] Hello"
            )
        )
        form.addRow(self.tr("Format:"), self.format_combo)

        self.src_combo = QComboBox()
        self.tgt_combo = QComboBox()
        for lang in LANGMAP_GLOBAL:
            self.src_combo.addItem(lang_display_label(lang), lang)
            self.tgt_combo.addItem(lang_display_label(lang), lang)
        self._select_lang_combo(self.src_combo, getattr(pcfg.module, "translate_source", "日本語"))
        self._select_lang_combo(self.tgt_combo, getattr(pcfg.module, "translate_target", "English"))
        form.addRow(self.tr("Source language:"), self.src_combo)
        form.addRow(self.tr("Target language:"), self.tgt_combo)
        root.addWidget(form_box)

        out_box = QGroupBox(self.tr("Output"))
        out_form = QFormLayout(out_box)
        self.out_format_combo = QComboBox()
        self.out_format_combo.addItem(self.tr("SRT (.srt)"), "srt")
        self.out_format_combo.addItem(self.tr("Timestamped text (.txt)"), "txt")
        out_form.addRow(self.tr("Save as:"), self.out_format_combo)
        root.addWidget(out_box)

        ctx_box = QGroupBox(self.tr("Series translation context"))
        ctx_v = QVBoxLayout(ctx_box)
        ctx_v.addWidget(
            QLabel(
                self.tr(
                    "Loads series glossary (glossary.txt under data/translation_context/). "
                    "Leave blank to use the translator’s Series context path from Config."
                )
            )
        )
        ser_row = QHBoxLayout()
        self.series_context_edit = QLineEdit()
        self.series_context_edit.setPlaceholderText(self.tr("e.g. default, urban_immortal, or a subfolder path"))
        self.series_context_edit.setText(_llm_translator_series_path_default())
        try:
            self.series_context_edit.setClearButtonEnabled(True)
        except Exception:
            pass
        ser_row.addWidget(self.series_context_edit)
        ctx_v.addLayout(ser_row)
        self.check_series_recent = QCheckBox(
            self.tr("Include recent pages from series store (recent_context.json)")
        )
        self.check_series_recent.setToolTip(
            self.tr(
                "Uses the same “previous pages” count as the translator config. "
                "Requires an existing series folder with recent_context.json (e.g. from comic translation)."
            )
        )
        ctx_v.addWidget(self.check_series_recent)
        ctx_v.addWidget(
            QLabel(
                self.tr(
                    "Extra project glossary for this run only (merged with series + translator; one line per entry):"
                )
            )
        )
        self.project_glossary_edit = QPlainTextEdit()
        self.project_glossary_edit.setPlaceholderText(
            self.tr("e.g. 主角名 -> Hero Name\n门派 -> sect")
        )
        self.project_glossary_edit.setMaximumHeight(90)
        ctx_v.addWidget(self.project_glossary_edit)
        root.addWidget(ctx_box)

        gloss_box = QGroupBox(self.tr("Optional subtitle / model hint (video-style)"))
        gv = QVBoxLayout(gloss_box)
        self.glossary_edit = QPlainTextEdit()
        self.glossary_edit.setPlaceholderText(
            self.tr("Short notes or terms appended to the prompt (like the video translator glossary hint).")
        )
        self.glossary_edit.setMaximumHeight(100)
        gv.addWidget(self.glossary_edit)
        root.addWidget(gloss_box)

        self.stats_label = QLabel(self.tr("No cues loaded."))
        root.addWidget(self.stats_label)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        root.addWidget(self.progress)

        btn_row = QHBoxLayout()
        self.btn_translate = QPushButton(self.tr("Translate and save as…"))
        self.btn_translate.setEnabled(False)
        self.btn_translate.clicked.connect(self._on_translate_save)
        self.btn_close = QPushButton(self.tr("Close"))
        self.btn_close.clicked.connect(self.close)
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_translate)
        btn_row.addWidget(self.btn_close)
        root.addLayout(btn_row)

    def _select_lang_combo(self, combo: QComboBox, key: str):
        for i in range(combo.count()):
            if combo.itemData(i) == key:
                combo.setCurrentIndex(i)
                return
        combo.setCurrentIndex(0)

    def _on_open(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Open subtitle file"),
            "",
            self.tr("Subtitles (*.srt *.txt);;All files (*.*)"),
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except OSError as e:
            QMessageBox.warning(self, self.tr("Open file"), str(e))
            return
        prefer: FormatHint = self.format_combo.currentData() or "auto"
        try:
            cues, fmt = parse_subtitle_content(content, prefer=prefer)
        except Exception as e:
            QMessageBox.warning(self, self.tr("Parse error"), str(e))
            return
        self._input_path = path
        self._cues = cues
        self._detected_format = fmt
        self.path_label.setText(path)
        self.stats_label.setText(
            self.tr("{0} cue(s) loaded (detected format: {1}).").format(len(cues), fmt.upper())
        )
        if self.out_format_combo.currentIndex() == 0 and fmt == "txt":
            self.out_format_combo.setCurrentIndex(1)
        elif self.out_format_combo.currentIndex() == 1 and fmt == "srt":
            self.out_format_combo.setCurrentIndex(0)
        self.btn_translate.setEnabled(len(cues) > 0)
        if len(cues) == 0:
            QMessageBox.information(
                self,
                self.tr("No cues"),
                self.tr("No subtitle cues found. Try another format option or check the file."),
            )

    def _on_translate_save(self):
        if self._thread and self._thread.isRunning():
            return
        if not self._cues:
            return
        tr_name = pcfg.module.translator
        if tr_name != "LLM_API_Translator":
            r = QMessageBox.question(
                self,
                self.tr("Translator"),
                self.tr(
                    "Current translator is '%s'. OpenRouter works best with LLM_API_Translator. Continue anyway?"
                )
                % tr_name,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if r != QMessageBox.StandardButton.Yes:
                return

        default_name = osp.splitext(osp.basename(self._input_path or "subtitles"))[0]
        out_kind = self.out_format_combo.currentData() or "srt"
        ext = ".srt" if out_kind == "srt" else ".txt"
        suggested = default_name + ".translated" + ext
        out_path, _ = QFileDialog.getSaveFileName(
            self,
            self.tr("Save translated subtitles"),
            suggested,
            self.tr("SRT (*.srt);;Text (*.txt);;All files (*.*)"),
        )
        if not out_path:
            return
        if out_kind == "srt" and not out_path.lower().endswith(".srt"):
            out_path += ".srt"
        if out_kind == "txt" and not (out_path.lower().endswith(".txt")):
            out_path += ".txt"

        texts = [c.text for c in self._cues]
        src = self.src_combo.currentData()
        tgt = self.tgt_combo.currentData()
        if not isinstance(src, str) or not src:
            src = lang_display_to_key(self.src_combo.currentText())
        if not isinstance(tgt, str) or not tgt:
            tgt = lang_display_to_key(self.tgt_combo.currentText())
        glossary = self.glossary_edit.toPlainText()
        proj_glossary = _text_to_glossary(self.project_glossary_edit.toPlainText())

        self.btn_translate.setEnabled(False)
        self.btn_open.setEnabled(False)
        self.progress.setRange(0, max(len(texts), 1))
        self.progress.setValue(0)

        self._thread = SubtitleFileTranslateThread(
            texts,
            src,
            tgt,
            glossary,
            series_context_path=self.series_context_edit.text(),
            project_glossary=proj_glossary,
            use_recent_series_context=self.check_series_recent.isChecked(),
            parent=self,
        )
        self._thread.progress.connect(self._on_progress)
        self._thread.finished_ok.connect(lambda lines: self._on_done_write(out_path, out_kind, lines))
        self._thread.failed.connect(self._on_failed)
        self._thread.start()

    def _on_progress(self, cur: int, total: int):
        self.progress.setMaximum(max(total, 1))
        self.progress.setValue(min(cur, total))

    def _on_done_write(self, out_path: str, out_kind: str, lines: list):
        self.btn_translate.setEnabled(True)
        self.btn_open.setEnabled(True)
        try:
            if out_kind == "srt":
                body = write_srt(self._cues, lines)
            else:
                body = write_timestamped_txt(self._cues, lines)
            with open(out_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(body)
        except OSError as e:
            QMessageBox.warning(self, self.tr("Save"), str(e))
            return
        QMessageBox.information(
            self,
            self.tr("Done"),
            self.tr("Wrote translated file to:\n{0}").format(out_path),
        )

    def _on_failed(self, msg: str):
        self.btn_translate.setEnabled(True)
        self.btn_open.setEnabled(True)
        self.progress.setValue(0)
        QMessageBox.warning(self, self.tr("Translation failed"), msg)

    def closeEvent(self, event):
        if self._thread and self._thread.isRunning():
            r = QMessageBox.question(
                self,
                self.tr("Close"),
                self.tr("Translation is still running. Close anyway?"),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if r != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        super().closeEvent(event)
