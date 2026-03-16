"""
Video translator dialog: UI and worker thread to translate hardcoded subtitles in videos
using the same detect / OCR / translate / inpaint pipeline as the main app.
"""
from __future__ import annotations

import os
import os.path as osp
import tempfile
from typing import Optional

from qtpy.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog,
    QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox, QProgressBar, QGroupBox, QGridLayout,
    QComboBox, QWidget, QPlainTextEdit, QListWidget, QListWidgetItem, QAbstractItemView,
    QScrollArea, QFrame, QSizePolicy,
)
from qtpy.QtCore import Qt, QThread, Signal, QEvent
from qtpy.QtGui import QIcon, QImage, QPixmap, QCloseEvent

from utils.config import pcfg, save_config
from utils.logger import logger as LOGGER


def _get_video_module(module_type: str, name: str):
    from modules import TEXTDETECTORS, OCR, TRANSLATORS, INPAINTERS
    if module_type == "textdetector":
        reg, cfg = TEXTDETECTORS, pcfg.module.textdetector_params
    elif module_type == "ocr":
        reg, cfg = OCR, pcfg.module.ocr_params
    elif module_type == "translator":
        reg, cfg = TRANSLATORS, pcfg.module.translator_params
    elif module_type == "inpainter":
        reg, cfg = INPAINTERS, pcfg.module.inpainter_params
    else:
        raise ValueError(module_type)
    if name not in reg.module_dict:
        raise RuntimeError(f"Module {module_type}/{name} not found.")
    params = (cfg.get(name) or {}).copy()
    if module_type == "translator":
        # BaseTranslator requires lang_source and lang_target (from app config)
        params.setdefault("lang_source", getattr(pcfg.module, "translate_source", "日本語"))
        params.setdefault("lang_target", getattr(pcfg.module, "translate_target", "简体中文"))
    if params:
        return reg.module_dict[name](**params)
    return reg.module_dict[name]()


def _get_source_bitrate_kbps(video_path: str, ffmpeg_exe: str = "") -> int:
    """Get source video bitrate in kbps via ffprobe. Uses stream then format bit_rate; if missing or
    very low (<1000), estimates from file size and duration. Returns 0 if not detectable."""
    import subprocess
    exe = "ffprobe"
    if (ffmpeg_exe or "").strip():
        base = (ffmpeg_exe or "").strip()
        d = osp.dirname(base)
        name = "ffprobe.exe" if base.lower().endswith(".exe") else "ffprobe"
        exe = osp.join(d, name) if d else name
    def _parse_kbps(bps: int) -> int | None:
        if bps <= 0:
            return None
        kbps = bps // 1000
        if kbps < 500 or kbps > 50000:
            return None
        return kbps

    try:
        # Stream bit_rate (video stream)
        out = subprocess.run(
            [
                exe, "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=bit_rate", "-of", "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode == 0 and out.stdout and out.stdout.strip():
            kbps = _parse_kbps(int(out.stdout.strip()))
            if kbps is not None and kbps >= 1000:
                return kbps
        # Format bit_rate (container)
        out = subprocess.run(
            [
                exe, "-v", "error", "-show_entries", "format=bit_rate",
                "-of", "default=noprint_wrappers=1:nokey=1", video_path,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode == 0 and out.stdout and out.stdout.strip():
            kbps = _parse_kbps(int(out.stdout.strip()))
            if kbps is not None and kbps >= 1000:
                return kbps
        # Fallback: estimate from file size and duration (metadata often wrong for stream/format)
        out = subprocess.run(
            [
                exe, "-v", "error", "-show_entries", "format=duration,size",
                "-of", "default=noprint_wrappers=1:nokey=1", video_path,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode == 0 and out.stdout and out.stdout.strip():
            lines = [s.strip() for s in out.stdout.strip().splitlines() if s.strip()]
            duration = size = None
            if len(lines) >= 2:
                try:
                    duration = float(lines[0])
                    size = int(float(lines[1]))
                except (ValueError, IndexError):
                    pass
            if duration is not None and size is not None and duration > 0:
                bps = int((size * 8) / duration)
                kbps = _parse_kbps(bps)
                if kbps is not None:
                    return kbps
        # Last resort: duration from ffprobe, size from filesystem
        out = subprocess.run(
            [exe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", video_path],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode == 0 and out.stdout and osp.isfile(video_path):
            try:
                duration = float(out.stdout.strip().split()[0] or 0)
                if duration > 0:
                    bps = int((osp.getsize(video_path) * 8) / duration)
                    kbps = _parse_kbps(bps)
                    if kbps is not None:
                        return kbps
            except (ValueError, IndexError, OSError):
                pass
    except Exception:
        pass
    return 0


def _normalize_fps(cap, reported_fps: float, total_frames: int) -> float:
    """Return a sane FPS for encoding. OpenCV CAP_PROP_FPS can be wrong for some codecs/VFR.
    When reported FPS is suspicious (0, >200, or <15), try FPS = frame_count / duration (seek to end).
    Never return below 30 fps: bogus 1-5 from underreported frame count becomes 30."""
    fps = reported_fps or 0.0
    if (fps <= 0 or fps > 200 or fps < 15.0) and cap and total_frames > 0:
        try:
            cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1.0)
            end_ms = cap.get(cv2.CAP_PROP_POS_MSEC) or 0
            cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 0.0)
            if end_ms > 0:
                from_fps = total_frames / (end_ms / 1000.0)
                if 1.0 <= from_fps <= 200.0:
                    return max(30.0, min(120.0, round(from_fps, 2)))
        except Exception:
            pass
    if fps <= 0:
        return 30.0
    if fps < 15.0:
        return 30.0
    return fps


def _scene_change(prev_gray, curr_gray, threshold: float) -> bool:
    """True if histogram diff exceeds threshold (new scene)."""
    if prev_gray is None or curr_gray is None:
        return True
    import cv2
    prev_hist = cv2.calcHist([prev_gray], [0], None, [64], [0, 256])
    curr_hist = cv2.calcHist([curr_gray], [0], None, [64], [0, 256])
    cv2.normalize(prev_hist, prev_hist, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(curr_hist, curr_hist, 0, 1, cv2.NORM_MINMAX)
    diff = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_BHATTACHARYYA)
    return diff * 100.0 > threshold


def _temporal_blend(prev: "np.ndarray", curr: "np.ndarray", alpha: float, bottom_frac: float) -> "np.ndarray":
    """Blend curr with prev in the bottom_frac band; alpha = weight of prev."""
    import numpy as np
    if prev is None or prev.shape != curr.shape or alpha <= 0:
        return curr
    h, w = curr.shape[:2]
    y_start = int(h * (1.0 - bottom_frac))
    y_start = max(0, min(y_start, h - 1))
    out = curr.copy()
    blend_roi = out[y_start:h, :].astype(np.float32)
    prev_roi = prev[y_start:h, :].astype(np.float32)
    blend_roi = (1.0 - alpha) * blend_roi + alpha * prev_roi
    np.clip(blend_roi, 0, 255, out=blend_roi)
    out[y_start:h, :] = blend_roi.astype(np.uint8)
    return out


def _frame_to_sec(frames: float, fps: float) -> float:
    return frames / fps if fps > 0 else 0


def _ts_srt(frames: float, fps: float) -> str:
    """SRT timestamp: HH:MM:SS,mmm (comma for decimal)."""
    s = _frame_to_sec(frames, fps)
    m = int(s // 60)
    s = s % 60
    h = m // 60
    m = m % 60
    return ("%02d:%02d:%06.3f" % (h, m, s)).replace(".", ",")


def _ts_vtt(frames: float, fps: float) -> str:
    """WebVTT timestamp: HH:MM:SS.mmm (period for decimal)."""
    s = _frame_to_sec(frames, fps)
    m = int(s // 60)
    s = s % 60
    h = m // 60
    m = m % 60
    return "%02d:%02d:%06.3f" % (h, m, s)


def _ts_ass(frames: float, fps: float) -> str:
    """ASS timestamp: H:MM:SS.cc (centiseconds)."""
    s = _frame_to_sec(frames, fps)
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec_frac = s % 60
    sec_int = int(sec_frac)
    cs = int(round((sec_frac - sec_int) * 100))
    return "%d:%02d:%02d.%02d" % (h, m, sec_int, min(99, cs))


def _write_srt(path: str, entries: list, fps: float) -> None:
    """Write SRT file. entries: list of (start_frame, end_frame, text)."""
    with open(path, "w", encoding="utf-8") as f:
        for i, (start, end, text) in enumerate(entries, 1):
            text = (text or "").strip().replace("\n", " ")
            if not text:
                text = ""
            f.write("%d\n%s --> %s\n%s\n\n" % (i, _ts_srt(start, fps), _ts_srt(end, fps), text))


def _write_vtt(path: str, entries: list, fps: float) -> None:
    """Write WebVTT file. entries: list of (start_frame, end_frame, text)."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for i, (start, end, text) in enumerate(entries, 1):
            text = (text or "").strip().replace("\n", " ")
            if not text:
                text = ""
            f.write("%d\n%s --> %s\n%s\n\n" % (i, _ts_vtt(start, fps), _ts_vtt(end, fps), text))


def _write_ass(path: str, entries: list, fps: float) -> None:
    """Write ASS file. entries: list of (start_frame, end_frame, text)."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("[Script Info]\nTitle: BallonsTranslator\n\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, ShadowColour, BackColour, Bold, Italic, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\nStyle: Default,Arial,20,&H00FFFFFF,&H00000000,&H00000000,&H80000000,0,0,1,2,0,2,10,10,10,1\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
        for start, end, text in entries:
            text = (text or "").strip().replace("\n", "\\N")
            if not text:
                text = ""
            # Escape ASS: replace \ with \\ and { with {{ and } with }}
            text = text.replace("\\", "\\\\").replace("{", "{{").replace("}", "}}")
            f.write("Dialogue: 0,%s,%s,Default,,0,0,0,,%s\n" % (_ts_ass(start, fps), _ts_ass(end, fps), text))


class VideoTranslateThread(QThread):
    """Worker that runs the pipeline on each sampled frame and writes the output video."""
    progress = Signal(int, int)  # current, total
    finished_ok = Signal(str)    # output path
    failed = Signal(str)
    # Live preview (OCR path): frame_index, frame_bgr (numpy copy), list of (source_text, translated_text)
    frame_preview_updated = Signal(int, object, list)

    def __init__(
        self,
        input_path: str,
        output_path: str,
        sample_every_frames: int,
        enable_detect: bool,
        enable_ocr: bool,
        enable_translate: bool,
        enable_inpaint: bool,
        use_scene_detection: bool = False,
        scene_threshold: float = 30.0,
        temporal_smoothing: bool = False,
        temporal_alpha: float = 0.25,
        use_ffmpeg: bool = False,
        ffmpeg_path: str = "",
        ffmpeg_crf: int = 18,
        video_bitrate_kbps: int = 0,
        skip_detect: bool = False,
        export_srt: bool = False,
        region_preset: str = "full",
        soft_subs_only: bool = False,
        inpaint_only_soft_subs: bool = False,
        mux_srt_into_video: bool = False,
        source: str = "ocr",
        asr_model: str = "base",
        asr_device: str = "cuda",
        asr_language: str = "",
        asr_vad_filter: bool = True,
        asr_sentence_break: bool = False,
        asr_audio_separation: bool = False,
        export_ass: bool = False,
        export_vtt: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self.input_path = input_path
        self.output_path = output_path
        self.sample_every_frames = max(1, sample_every_frames)
        self.enable_detect = enable_detect
        self.enable_ocr = enable_ocr
        self.enable_translate = enable_translate
        self.enable_inpaint = enable_inpaint
        self.use_scene_detection = use_scene_detection
        self.scene_threshold = float(scene_threshold)
        self.temporal_smoothing = temporal_smoothing
        self.temporal_alpha = float(temporal_alpha)
        self.use_ffmpeg = use_ffmpeg
        self.ffmpeg_path = (ffmpeg_path or "").strip()
        self.ffmpeg_crf = max(0, min(51, int(ffmpeg_crf)))
        self.video_bitrate_kbps = max(0, int(video_bitrate_kbps))
        self.skip_detect = skip_detect
        self.export_srt = export_srt
        self.region_preset = (region_preset or "full").strip().lower()
        self.soft_subs_only = bool(soft_subs_only)
        self.inpaint_only_soft_subs = bool(inpaint_only_soft_subs)
        self.mux_srt_into_video = bool(mux_srt_into_video)
        self.source = (source or "ocr").strip().lower()
        self.asr_model = (asr_model or "base").strip()
        self.asr_device = (asr_device or "cuda").strip().lower()
        self.asr_language = (asr_language or "").strip()
        self.asr_vad_filter = bool(asr_vad_filter)
        self.asr_sentence_break = bool(asr_sentence_break)
        self.asr_audio_separation = bool(asr_audio_separation)
        self.export_ass = bool(export_ass)
        self.export_vtt = bool(export_vtt)
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def _run_asr_pipeline(self):
        """Source = Audio (ASR): extract audio, transcribe, translate, then render frames with timed subs or soft SRT only."""
        import cv2
        import subprocess
        import numpy as np
        from modules.video_translator import _draw_timed_subs_on_image
        from modules.audio_transcribe import (
            HAS_FASTER_WHISPER,
            extract_audio_from_video,
            transcribe_audio,
        )
        from utils.textblock import TextBlock

        try:
            if not HAS_FASTER_WHISPER:
                self.failed.emit(
                    "ASR requires faster-whisper. Install with: pip install faster-whisper"
                )
                return
            cfg = pcfg.module
            ffmpeg_exe = (self.ffmpeg_path or getattr(cfg, "video_translator_ffmpeg_path", "") or "ffmpeg").strip()
            extract_path = extract_audio_from_video(self.input_path, ffmpeg_path=ffmpeg_exe)
            if not extract_path or not os.path.isfile(extract_path):
                self.failed.emit("Could not extract audio from video (check FFmpeg path).")
                return
            audio_path = extract_path
            vocals_path = None
            if self.asr_audio_separation:
                try:
                    from modules.audio_separate import separate_vocals
                    vocals_path = separate_vocals(extract_path, device=self.asr_device)
                    if vocals_path and os.path.isfile(vocals_path):
                        audio_path = vocals_path
                    else:
                        LOGGER.warning("Vocal separation failed or demucs not installed; using original audio. Install with: pip install demucs")
                except Exception as e:
                    LOGGER.warning("Vocal separation failed: %s. Using original audio.", e)
            try:
                segments = transcribe_audio(
                    audio_path,
                    model_size=self.asr_model,
                    device=self.asr_device,
                    language=self.asr_language or None,
                    vad_filter=self.asr_vad_filter,
                )
            finally:
                for p in (extract_path, vocals_path):
                    if p and os.path.isfile(p) and p.startswith(tempfile.gettempdir()):
                        try:
                            os.remove(p)
                        except Exception:
                            pass
            if not segments:
                self.failed.emit("No speech detected in audio.")
                return
            if self._cancel:
                self.failed.emit("Cancelled.")
                return
            translator = _get_video_module("translator", cfg.translator) if self.enable_translate else None
            if translator is not None:
                setattr(translator, "_video_glossary_hint", (getattr(cfg, "video_translator_glossary", None) or "").strip())
                sp = (getattr(cfg, "video_translator_series_context_path", None) or "").strip()
                if sp and hasattr(translator, "set_translation_context"):
                    translator.set_translation_context(series_context_path=sp)
                if getattr(translator, "get_param_value", None) and translator.get_param_value("correct_asr_with_llm"):
                    texts = [t for _s, _e, t in segments]
                    try:
                        corrected = translator.correct_asr_texts(texts)
                        if corrected and len(corrected) == len(segments):
                            segments = [(s, e, c) for (s, e, _), c in zip(segments, corrected)]
                    except Exception as e:
                        LOGGER.warning("ASR correction failed (using original): %s", e)
                if self.asr_sentence_break and hasattr(translator, "sentence_break_segments"):
                    try:
                        segments = translator.sentence_break_segments(segments)
                    except Exception as e:
                        LOGGER.warning("Sentence break failed (using original): %s", e)
            if translator is not None and hasattr(translator, "load_model"):
                translator.load_model()
            segments_with_trans = []
            if translator is not None and self.enable_translate:
                blk_list = []
                for _s, _e, text in segments:
                    blk = TextBlock()
                    blk.text = [text or ""]
                    blk.lines = []
                    blk.xyxy = [0, 0, 1, 1]
                    blk_list.append(blk)
                try:
                    setattr(translator, "_current_page_key", "video_asr")
                    setattr(translator, "_current_page_image", None)
                    translator.translate_textblk_lst(blk_list)
                except Exception as e:
                    LOGGER.warning("ASR translate batch failed: %s", e)
                for (s, e, _), blk in zip(segments, blk_list):
                    trans = (getattr(blk, "translation", None) or blk.get_text() or "").strip()
                    segments_with_trans.append((s, e, trans))
            else:
                segments_with_trans = [(s, e, (t or "").strip()) for s, e, t in segments]

            cap = cv2.VideoCapture(self.input_path)
            if not cap.isOpened():
                self.failed.emit("Could not open input video for ASR output.")
                return
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            if fps <= 0:
                fps = 25.0
            elif fps < 15.0:
                fps = 24.0
            if fps <= 0:
                fps = 25.0
            elif fps < 15.0:
                fps = 24.0
            # Some codecs report very low or bogus FPS (e.g. 1–5). Clamp to a sane minimum
            # so partial outputs and cancelled runs don't look like 4 FPS slideshows.
            if fps <= 0:
                fps = 25.0
            elif fps < 15.0:
                fps = 24.0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if w <= 0 or h <= 0:
                cap.release()
                self.failed.emit("Invalid video size.")
                return
            if total <= 0 and fps > 0:
                dur = cap.get(cv2.CAP_PROP_POS_MSEC) or 0
                total = int(dur * fps / 1000.0) or -1 if dur > 0 else -1
            src_bps = cap.get(cv2.CAP_PROP_BITRATE) or 0
            ocv_kbps = int(src_bps / 1000) if src_bps > 0 else 0
            source_bitrate_kbps = max(500, min(50000, ocv_kbps)) if ocv_kbps >= 1000 else 0
            if source_bitrate_kbps <= 0:
                source_bitrate_kbps = _get_source_bitrate_kbps(self.input_path, self.ffmpeg_path)

            use_ffmpeg = self.use_ffmpeg
            ffmpeg_proc = None
            if use_ffmpeg:
                try:
                    ffmpeg_exe = self.ffmpeg_path or "ffmpeg"
                    effective_kbps = self.video_bitrate_kbps if self.video_bitrate_kbps > 0 else source_bitrate_kbps
                    if effective_kbps > 0:
                        ffmpeg_cmd = [
                            ffmpeg_exe, "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                            "-s", "%dx%d" % (w, h), "-r", str(fps), "-i", "pipe:0",
                            "-c:v", "libx264", "-preset", "medium",
                            "-b:v", "%dk" % effective_kbps,
                            "-minrate", "%dk" % effective_kbps,
                            "-maxrate", "%dk" % effective_kbps,
                            "-bufsize", "%dk" % (effective_kbps * 2),
                            "-x264-params", "nal-hrd=cbr",
                            "-r", str(fps),
                            self.output_path,
                        ]
                    else:
                        ffmpeg_cmd = [
                            ffmpeg_exe, "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                            "-s", "%dx%d" % (w, h), "-r", str(fps), "-i", "pipe:0",
                            "-c:v", "libx264", "-preset", "medium", "-crf", str(self.ffmpeg_crf),
                            "-r", str(fps),
                            self.output_path,
                        ]
                    ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
                except Exception as e:
                    LOGGER.warning("FFmpeg not available for ASR output: %s. Using OpenCV.", e)
                    use_ffmpeg = False
            out = None
            if not use_ffmpeg:
                codec = (getattr(cfg, "video_translator_output_codec", None) or "mp4v").strip() or "mp4v"
                if len(codec) != 4:
                    codec = "mp4v"
                fourcc = cv2.VideoWriter_fourcc(*codec)
                out = cv2.VideoWriter(self.output_path, fourcc, fps, (w, h))
                if not out.isOpened() and codec != "avc1":
                    out = cv2.VideoWriter(self.output_path, cv2.VideoWriter_fourcc(*"avc1"), fps, (w, h))
                if not out.isOpened():
                    cap.release()
                    self.failed.emit("Could not create output video.")
                    return

            style = (getattr(cfg, "video_translator_subtitle_style", None) or "default").strip().lower()
            if style not in ("anime", "documentary"):
                style = "default"
            n = 0
            step = 10 if total <= 10000 else (500 if total <= 100000 else 2000)
            while True:
                if self._cancel:
                    break
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                time_sec = n / fps if fps > 0 else 0
                if not self.soft_subs_only:
                    _draw_timed_subs_on_image(frame, time_sec, segments_with_trans, style=style)
                if use_ffmpeg and ffmpeg_proc and ffmpeg_proc.stdin:
                    try:
                        ffmpeg_proc.stdin.write(frame.tobytes())
                    except Exception:
                        pass
                elif out is not None:
                    out.write(frame)
                n += 1
                if total > 0 and (n % step == 0 or n == total):
                    self.progress.emit(n, total)
                elif total <= 0 and n % 30 == 0:
                    self.progress.emit(n, 0)
            cap.release()
            if out is not None:
                out.release()
            if use_ffmpeg and ffmpeg_proc and ffmpeg_proc.stdin:
                try:
                    ffmpeg_proc.stdin.close()
                    ffmpeg_proc.wait(timeout=max(600, (total or 0) // 1000))
                except Exception:
                    try:
                        ffmpeg_proc.terminate()
                    except Exception:
                        pass

            # Only write sidecar SRT/ASS/VTT when not burning in (avoids double subtitles when player auto-loads SRT)
            if self.soft_subs_only:
                srt_entries = [(int(s * fps), int(e * fps), text) for s, e, text in segments_with_trans]
                base_path = osp.splitext(self.output_path)[0]
                if srt_entries:
                    _write_srt(base_path + ".srt", srt_entries, fps)
                    if self.export_ass:
                        _write_ass(base_path + ".ass", srt_entries, fps)
                    if self.export_vtt:
                        _write_vtt(base_path + ".vtt", srt_entries, fps)

            if osp.isfile(self.output_path):
                self.finished_ok.emit(self.output_path)
            else:
                self.failed.emit("Cancelled.")
        except Exception as e:
            LOGGER.exception("ASR pipeline failed")
            self.failed.emit(str(e))

    def _run_existing_subs_pipeline(self):
        """Source = Existing subtitles: load sidecar or embedded subs, translate, then render frames with timed subs or soft SRT only."""
        import cv2
        import subprocess
        from modules.video_translator import _draw_timed_subs_on_image
        from modules.video_subtitle_extract import load_existing_subtitles
        from utils.textblock import TextBlock

        try:
            cfg = pcfg.module
            ffmpeg_exe = (self.ffmpeg_path or getattr(cfg, "video_translator_ffmpeg_path", "") or "ffmpeg").strip()
            segments = load_existing_subtitles(self.input_path, ffmpeg_path=ffmpeg_exe)
            if not segments:
                self.failed.emit("No existing subtitles found (no sidecar .srt/.ass/.vtt and no embedded subtitle stream).")
                return
            if self._cancel:
                self.failed.emit("Cancelled.")
                return
            translator = _get_video_module("translator", cfg.translator) if self.enable_translate else None
            if translator is not None:
                setattr(translator, "_video_glossary_hint", (getattr(cfg, "video_translator_glossary", None) or "").strip())
                sp = (getattr(cfg, "video_translator_series_context_path", None) or "").strip()
                if sp and hasattr(translator, "set_translation_context"):
                    translator.set_translation_context(series_context_path=sp)
            if translator is not None and hasattr(translator, "load_model"):
                translator.load_model()
            segments_with_trans = []
            if translator is not None and self.enable_translate:
                blk_list = []
                for _s, _e, text in segments:
                    blk = TextBlock()
                    blk.text = [text or ""]
                    blk.lines = []
                    blk.xyxy = [0, 0, 1, 1]
                    blk_list.append(blk)
                try:
                    setattr(translator, "_current_page_key", "video_existing_subs")
                    setattr(translator, "_current_page_image", None)
                    translator.translate_textblk_lst(blk_list)
                except Exception as e:
                    LOGGER.warning("Existing subs translate batch failed: %s", e)
                for (s, e, _), blk in zip(segments, blk_list):
                    trans = (getattr(blk, "translation", None) or blk.get_text() or "").strip()
                    segments_with_trans.append((s, e, trans))
            else:
                segments_with_trans = [(s, e, (t or "").strip()) for s, e, t in segments]

            cap = cv2.VideoCapture(self.input_path)
            if not cap.isOpened():
                self.failed.emit("Could not open input video for output.")
                return
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if w <= 0 or h <= 0:
                cap.release()
                self.failed.emit("Invalid video size.")
                return
            if total <= 0 and fps > 0:
                dur = cap.get(cv2.CAP_PROP_POS_MSEC) or 0
                total = int(dur * fps / 1000.0) or -1 if dur > 0 else -1
            src_bps = cap.get(cv2.CAP_PROP_BITRATE) or 0
            ocv_kbps = int(src_bps / 1000) if src_bps > 0 else 0
            source_bitrate_kbps = max(500, min(50000, ocv_kbps)) if ocv_kbps >= 1000 else 0
            if source_bitrate_kbps <= 0:
                source_bitrate_kbps = _get_source_bitrate_kbps(self.input_path, self.ffmpeg_path)

            use_ffmpeg = self.use_ffmpeg
            ffmpeg_proc = None
            if use_ffmpeg:
                try:
                    ffmpeg_exe = self.ffmpeg_path or "ffmpeg"
                    effective_kbps = self.video_bitrate_kbps if self.video_bitrate_kbps > 0 else source_bitrate_kbps
                    if effective_kbps > 0:
                        ffmpeg_cmd = [
                            ffmpeg_exe, "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                            "-s", "%dx%d" % (w, h), "-r", str(fps), "-i", "pipe:0",
                            "-c:v", "libx264", "-preset", "medium",
                            "-b:v", "%dk" % effective_kbps,
                            "-minrate", "%dk" % effective_kbps,
                            "-maxrate", "%dk" % effective_kbps,
                            "-bufsize", "%dk" % (effective_kbps * 2),
                            "-x264-params", "nal-hrd=cbr",
                            "-r", str(fps),
                            self.output_path,
                        ]
                    else:
                        ffmpeg_cmd = [
                            ffmpeg_exe, "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                            "-s", "%dx%d" % (w, h), "-r", str(fps), "-i", "pipe:0",
                            "-c:v", "libx264", "-preset", "medium", "-crf", str(self.ffmpeg_crf),
                            "-r", str(fps),
                            self.output_path,
                        ]
                    ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
                except Exception as e:
                    LOGGER.warning("FFmpeg not available for existing subs output: %s. Using OpenCV.", e)
                    use_ffmpeg = False
            out = None
            if not use_ffmpeg:
                codec = (getattr(cfg, "video_translator_output_codec", None) or "mp4v").strip() or "mp4v"
                if len(codec) != 4:
                    codec = "mp4v"
                fourcc = cv2.VideoWriter_fourcc(*codec)
                out = cv2.VideoWriter(self.output_path, fourcc, fps, (w, h))
                if not out.isOpened() and codec != "avc1":
                    out = cv2.VideoWriter(self.output_path, cv2.VideoWriter_fourcc(*"avc1"), fps, (w, h))
                if not out.isOpened():
                    cap.release()
                    self.failed.emit("Could not create output video.")
                    return

            style = (getattr(cfg, "video_translator_subtitle_style", None) or "default").strip().lower()
            if style not in ("anime", "documentary"):
                style = "default"
            n = 0
            step = 10 if total <= 10000 else (500 if total <= 100000 else 2000)
            while True:
                if self._cancel:
                    break
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                time_sec = n / fps if fps > 0 else 0
                if not self.soft_subs_only:
                    _draw_timed_subs_on_image(frame, time_sec, segments_with_trans, style=style)
                if use_ffmpeg and ffmpeg_proc and ffmpeg_proc.stdin:
                    try:
                        ffmpeg_proc.stdin.write(frame.tobytes())
                    except Exception:
                        pass
                elif out is not None:
                    out.write(frame)
                n += 1
                if total > 0 and (n % step == 0 or n == total):
                    self.progress.emit(n, total)
                elif total <= 0 and n % 30 == 0:
                    self.progress.emit(n, 0)
            cap.release()
            if out is not None:
                out.release()
            if use_ffmpeg and ffmpeg_proc and ffmpeg_proc.stdin:
                try:
                    ffmpeg_proc.stdin.close()
                    ffmpeg_proc.wait(timeout=max(600, (total or 0) // 1000))
                except Exception:
                    try:
                        ffmpeg_proc.terminate()
                    except Exception:
                        pass

            # Only write sidecar SRT/ASS/VTT when not burning in (avoids double subtitles when player auto-loads SRT)
            if self.soft_subs_only:
                srt_entries = [(int(s * fps), int(e * fps), text) for s, e, text in segments_with_trans]
                base_path = osp.splitext(self.output_path)[0]
                if srt_entries:
                    _write_srt(base_path + ".srt", srt_entries, fps)
                    if self.export_ass:
                        _write_ass(base_path + ".ass", srt_entries, fps)
                    if self.export_vtt:
                        _write_vtt(base_path + ".vtt", srt_entries, fps)

            if osp.isfile(self.output_path):
                self.finished_ok.emit(self.output_path)
            else:
                self.failed.emit("Cancelled.")
        except Exception as e:
            LOGGER.exception("Existing subtitles pipeline failed")
            self.failed.emit(str(e))

    def run(self):
        import cv2
        import subprocess
        import numpy as np
        from modules.video_translator import run_one_frame_pipeline, _region_fraction_from_preset

        try:
            src = getattr(self, "source", "ocr").strip().lower()
            if src == "asr":
                self._run_asr_pipeline()
                return
            if src == "existing_subs":
                self._run_existing_subs_pipeline()
                return

            cap = cv2.VideoCapture(self.input_path)
            if not cap.isOpened():
                self.failed.emit("Could not open input video.")
                return
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            fps = _normalize_fps(cap, cap.get(cv2.CAP_PROP_FPS) or 0.0, total)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if w <= 0 or h <= 0:
                cap.release()
                self.failed.emit("Invalid video size.")
                return
            if total <= 0 and fps > 0:
                dur = cap.get(cv2.CAP_PROP_POS_MSEC) or 0
                if dur <= 0:
                    total = -1
                else:
                    total = int(dur * fps / 1000.0) or -1
            # Source bitrate: OpenCV often returns 0 or wrong low values (e.g. 500); use ffprobe/size-duration when < 1000 kbps
            src_bps = cap.get(cv2.CAP_PROP_BITRATE) or 0
            ocv_kbps = int(src_bps / 1000) if src_bps > 0 else 0
            source_bitrate_kbps = max(500, min(50000, ocv_kbps)) if ocv_kbps >= 1000 else 0
            if source_bitrate_kbps <= 0:
                source_bitrate_kbps = _get_source_bitrate_kbps(self.input_path, self.ffmpeg_path)

            cfg = pcfg.module
            detector = _get_video_module("textdetector", cfg.textdetector) if (self.enable_detect and not self.skip_detect) else None
            ocr = _get_video_module("ocr", cfg.ocr) if self.enable_ocr else None
            translator = _get_video_module("translator", cfg.translator) if self.enable_translate else None
            inpainter = _get_video_module("inpainter", cfg.inpainter) if self.enable_inpaint else None
            if translator is not None:
                setattr(translator, "_video_glossary_hint", (getattr(cfg, "video_translator_glossary", None) or "").strip())
                sp = (getattr(cfg, "video_translator_series_context_path", None) or "").strip()
                if sp and hasattr(translator, "set_translation_context"):
                    translator.set_translation_context(series_context_path=sp)

            if self.enable_detect and not self.skip_detect and detector is not None and hasattr(detector, "load_model"):
                detector.load_model()
            if self.enable_ocr and ocr is not None and hasattr(ocr, "load_model"):
                ocr.load_model()
            if self.enable_inpaint and inpainter is not None and hasattr(inpainter, "load_model"):
                inpainter.load_model()
            if self.enable_translate and translator is not None and hasattr(translator, "load_model"):
                translator.load_model()

            # Flow fixer (optional): local Ollama/LM Studio or OpenRouter (second model) to improve flow
            flow_fixer = None
            flow_fixer_context = 20
            if getattr(cfg, "video_translator_flow_fixer_enabled", False):
                fixer_name = (getattr(cfg, "video_translator_flow_fixer", None) or "none").strip().lower()
                flow_fixer_context = max(1, min(50, int(getattr(cfg, "video_translator_flow_fixer_context_lines", 20))))
                if fixer_name and fixer_name != "none":
                    try:
                        from modules.flow_fixer import get_flow_fixer
                        if fixer_name == "openrouter":
                            flow_fixer = get_flow_fixer(
                                "openrouter",
                                api_key=(getattr(cfg, "video_translator_flow_fixer_openrouter_apikey", None) or "").strip(),
                                model=(getattr(cfg, "video_translator_flow_fixer_openrouter_model", None) or "google/gemma-3n-e2b-it:free").strip(),
                                max_tokens=int(getattr(cfg, "video_translator_flow_fixer_max_tokens", 256)),
                                timeout=float(getattr(cfg, "video_translator_flow_fixer_timeout", 30.0)),
                            )
                        elif fixer_name == "openai":
                            flow_fixer = get_flow_fixer(
                                "openai",
                                api_key=(getattr(cfg, "video_translator_flow_fixer_openai_apikey", None) or "").strip(),
                                model=(getattr(cfg, "video_translator_flow_fixer_openai_model", None) or "gpt-4o-mini").strip(),
                                max_tokens=int(getattr(cfg, "video_translator_flow_fixer_max_tokens", 256)),
                                timeout=float(getattr(cfg, "video_translator_flow_fixer_timeout", 30.0)),
                            )
                        else:
                            flow_fixer = get_flow_fixer(
                                fixer_name,
                                server_url=(getattr(cfg, "video_translator_flow_fixer_server_url", None) or "http://localhost:1234/v1").strip(),
                                model=(getattr(cfg, "video_translator_flow_fixer_model", None) or "local").strip(),
                                max_tokens=int(getattr(cfg, "video_translator_flow_fixer_max_tokens", 256)),
                                timeout=float(getattr(cfg, "video_translator_flow_fixer_timeout", 30.0)),
                            )
                    except Exception as e:
                        LOGGER.warning("Flow fixer not available: %s", e)
                        flow_fixer = None
            if flow_fixer is None:
                from modules.flow_fixer import get_flow_fixer
                flow_fixer = get_flow_fixer("none")

            # FFmpeg writer (optional)
            ffmpeg_proc = None
            use_ffmpeg = self.use_ffmpeg
            if use_ffmpeg:
                try:
                    ffmpeg_exe = self.ffmpeg_path if self.ffmpeg_path else "ffmpeg"
                    effective_kbps = self.video_bitrate_kbps if self.video_bitrate_kbps > 0 else source_bitrate_kbps
                    if effective_kbps > 0:
                        ffmpeg_cmd = [
                            ffmpeg_exe, "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                            "-s", "%dx%d" % (w, h), "-r", str(fps), "-i", "pipe:0",
                            "-c:v", "libx264", "-preset", "medium",
                            "-b:v", "%dk" % effective_kbps,
                            "-minrate", "%dk" % effective_kbps,
                            "-maxrate", "%dk" % effective_kbps,
                            "-bufsize", "%dk" % (effective_kbps * 2),
                            "-x264-params", "nal-hrd=cbr",
                            "-r", str(fps),
                            self.output_path,
                        ]
                    else:
                        ffmpeg_cmd = [
                            ffmpeg_exe, "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                            "-s", "%dx%d" % (w, h), "-r", str(fps), "-i", "pipe:0",
                            "-c:v", "libx264", "-preset", "medium", "-crf", str(self.ffmpeg_crf),
                            "-r", str(fps),
                            self.output_path,
                        ]
                    ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
                except Exception as e:
                    LOGGER.warning(
                        "FFmpeg not available: %s. Either uncheck 'Use FFmpeg' in Video translator Advanced to use OpenCV, or install FFmpeg and add it to PATH, or set 'FFmpeg path' to the full path to ffmpeg.exe (e.g. C:\\ffmpeg\\bin\\ffmpeg.exe). Using OpenCV for output.",
                        e,
                    )
                    use_ffmpeg = False

            if not use_ffmpeg:
                codec = (getattr(cfg, "video_translator_output_codec", None) or "mp4v").strip() or "mp4v"
                if len(codec) != 4:
                    codec = "mp4v"
                fourcc = cv2.VideoWriter_fourcc(*codec)
                out = cv2.VideoWriter(self.output_path, fourcc, fps, (w, h))
                if not out.isOpened() and codec != "avc1":
                    fourcc = cv2.VideoWriter_fourcc(*"avc1")
                    out = cv2.VideoWriter(self.output_path, fourcc, fps, (w, h))
                if not out.isOpened():
                    cap.release()
                    self.failed.emit("Could not create output video (try different output path or codec).")
                    return
            else:
                out = None

            def write_frame(frame):
                if use_ffmpeg and ffmpeg_proc and ffmpeg_proc.stdin:
                    try:
                        ffmpeg_proc.stdin.write(frame.tobytes())
                    except Exception:
                        pass
                elif out is not None:
                    out.write(frame)

            cached_result = None
            prev_gray = None
            prev_result = None
            last_pipeline_n = -self.sample_every_frames - 1  # so we run on n=0 and respect "process every N" minimum interval
            n = 0
            srt_entries = []
            current_preview_texts = []
            preview_throttle = 5
            video_previous_subtitles = []
            max_video_context = 10
            frac = _region_fraction_from_preset(cfg) if self.temporal_smoothing else 0.0
            if frac <= 0:
                frac = 0.2

            # Frame pacing: one cap.read() per iteration, one write_frame() — output frame count
            # matches input. Pipeline runs at most every sample_every_frames (so "Process every N" is never exceeded).
            while True:
                if self._cancel:
                    break
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
                # Run pipeline every sample_every_frames, or on scene change (throttled: at least sample_every_frames since last run)
                periodic = n % self.sample_every_frames == 0
                if self.use_scene_detection:
                    scene_changed = _scene_change(prev_gray, curr_gray, self.scene_threshold)
                    run_pipeline = periodic or (scene_changed and (n - last_pipeline_n) >= self.sample_every_frames)
                else:
                    run_pipeline = periodic
                prev_gray = curr_gray

                if run_pipeline:
                    last_pipeline_n = n
                    try:
                        if translator is not None and hasattr(translator, "set_translation_context"):
                            translator.set_translation_context(
                                previous_pages=video_previous_subtitles,
                                series_context_path=sp if sp else None,
                            )
                        # When inpaint_only_soft_subs, do not draw subtitles on frames (video gets SRT only)
                        draw_subs = not self.inpaint_only_soft_subs
                        out_frame, blk_list, _ = run_one_frame_pipeline(
                            frame,
                            detector, ocr, translator, inpainter,
                            self.enable_detect, self.enable_ocr, self.enable_translate, self.enable_inpaint,
                            cfg=cfg,
                            skip_detect=self.skip_detect,
                            draw_subtitles=draw_subs,
                        )
                        if translator is not None and blk_list:
                            video_previous_subtitles.append({
                                "sources": [b.get_text() if hasattr(b, "get_text") else (b.text or [""])[0] for b in blk_list],
                                "translations": [str(getattr(b, "translation", "") or "").strip() for b in blk_list],
                            })
                            if len(video_previous_subtitles) > max_video_context:
                                video_previous_subtitles = video_previous_subtitles[-max_video_context:]
                            # Apply optional revised_previous from model (flow fix; no extra API call)
                            rev = getattr(translator, "_last_revised_previous", None)
                            if rev and isinstance(rev, list) and len(rev) > 0 and len(video_previous_subtitles) >= len(rev):
                                k = len(rev)
                                for i in range(k):
                                    idx = -k + i
                                    if i < len(rev) and rev[i]:
                                        video_previous_subtitles[idx]["translations"] = [rev[i].strip()]
                                # Update SRT/ASS/VTT text for the revised segments (rev[0]=oldest of k, rev[k-1]=newest)
                                if (self.export_srt or self.soft_subs_only or self.inpaint_only_soft_subs) and len(srt_entries) >= k:
                                    for i in range(k):
                                        if i < len(rev) and rev[i]:
                                            srt_entries[-(k - i)] = (srt_entries[-(k - i)][0], srt_entries[-(k - i)][1], rev[i].strip())
                                # Update translation cache so future reuse uses revised text
                                cache = getattr(translator, "_video_frame_cache", None)
                                if cache is not None:
                                    for i in range(k):
                                        seg = video_previous_subtitles[-k + i]
                                        srcs = seg.get("sources") or []
                                        if i < len(rev) and rev[i] and srcs:
                                            cache[tuple(srcs)] = [rev[i].strip()]
                                setattr(translator, "_last_revised_previous", None)
                        # Local flow fixer (Ollama/LM Studio): improve flow without extra cloud API calls
                        if flow_fixer is not None and blk_list and getattr(flow_fixer, "improve_flow", None):
                            try:
                                new_translations = [str(getattr(b, "translation", "") or "").strip() for b in blk_list]
                                previous_entries = video_previous_subtitles[-flow_fixer_context:] if video_previous_subtitles else []
                                revised_prev, revised_new = flow_fixer.improve_flow(
                                    [dict(e) for e in previous_entries],
                                    new_translations,
                                    target_lang="en",
                                )
                                if revised_new and len(revised_new) == len(blk_list):
                                    for i, b in enumerate(blk_list):
                                        if i < len(revised_new):
                                            b.translation = revised_new[i]
                                    if revised_prev is not None and len(revised_prev) > 0 and len(video_previous_subtitles) >= len(revised_prev):
                                        k = len(revised_prev)
                                        for i in range(k):
                                            video_previous_subtitles[-(k - i)] = revised_prev[i]
                                        if (self.export_srt or self.soft_subs_only or self.inpaint_only_soft_subs) and len(srt_entries) >= k:
                                            for i in range(k):
                                                ent = revised_prev[i]
                                                txt = " ".join((ent.get("translations") or [])).strip()
                                                if txt and i < k:
                                                    idx = -(k - i)
                                                    srt_entries[idx] = (srt_entries[idx][0], srt_entries[idx][1], txt)
                                        cache = getattr(translator, "_video_frame_cache", None) if translator else None
                                        if cache is not None:
                                            for i in range(k):
                                                seg = video_previous_subtitles[-k + i]
                                                srcs = seg.get("sources") or []
                                                trans = (seg.get("translations") or [])
                                                if srcs and trans:
                                                    cache[tuple(srcs)] = [str(t).strip() for t in trans]
                            except Exception as e:
                                LOGGER.debug("Flow fixer failed: %s", e)
                        if self.temporal_smoothing and prev_result is not None:
                            out_frame = _temporal_blend(prev_result, out_frame, self.temporal_alpha, frac)
                        cached_result = out_frame
                        prev_result = out_frame.copy()
                        if (self.export_srt or self.soft_subs_only or self.inpaint_only_soft_subs) and blk_list:
                            text = " ".join((getattr(b, "translation", None) or "").strip() for b in blk_list).strip()
                            if text:
                                if srt_entries and srt_entries[-1][2] == text:
                                    # Same text as last segment: just extend its end frame
                                    srt_entries[-1] = (srt_entries[-1][0], n + self.sample_every_frames, srt_entries[-1][2])
                                else:
                                    # Close previous segment at this frame, start a new one
                                    if srt_entries:
                                        srt_entries[-1] = (srt_entries[-1][0], n, srt_entries[-1][2])
                                    srt_entries.append((n, n + self.sample_every_frames, text))
                        current_preview_texts = [(b.get_text() if hasattr(b, "get_text") else (b.text or [""])[0], (getattr(b, "translation", None) or "").strip()) for b in (blk_list or [])]
                        self.frame_preview_updated.emit(n, out_frame.copy(), current_preview_texts)
                    except Exception as e:
                        LOGGER.warning("Frame %d pipeline failed: %s", n, e)
                        cached_result = frame.copy()
                        prev_result = cached_result.copy()

                elif cached_result is not None and n % preview_throttle == 0:
                    self.frame_preview_updated.emit(n, cached_result.copy(), current_preview_texts)

                if self.soft_subs_only:
                    # Original video only; subs in SRT/ASS/VTT (no inpainting, no burn-in)
                    write_frame(frame)
                elif self.inpaint_only_soft_subs and cached_result is not None:
                    # Inpainted video, no text drawn; subs in SRT/ASS/VTT only (no double subs in video)
                    if cached_result.shape[1] != w or cached_result.shape[0] != h:
                        cached_result = cv2.resize(cached_result, (w, h), interpolation=cv2.INTER_LINEAR)
                    write_frame(cached_result)
                elif cached_result is not None:
                    if cached_result.shape[1] != w or cached_result.shape[0] != h:
                        cached_result = cv2.resize(cached_result, (w, h), interpolation=cv2.INTER_LINEAR)
                    write_frame(cached_result)
                else:
                    write_frame(frame)

                n += 1
                # Throttle progress for long videos to avoid UI overload (e.g. 17h = 1.5M+ frames)
                if total > 0:
                    step = 10 if total <= 10000 else (500 if total <= 100000 else 2000)
                    if n % step == 0 or n == total:
                        self.progress.emit(n, total)
                elif total <= 0 and n % 30 == 0:
                    self.progress.emit(n, 0)

            cap.release()
            if not use_ffmpeg and out is not None:
                out.release()
            if use_ffmpeg and ffmpeg_proc and ffmpeg_proc.stdin:
                try:
                    ffmpeg_proc.stdin.close()
                    # Long videos: allow time for FFmpeg to finish encoding (e.g. 17h output)
                    wait_timeout = max(600, total // 1000) if total > 0 else 3600
                    ffmpeg_proc.wait(timeout=wait_timeout)
                except subprocess.TimeoutExpired:
                    try:
                        ffmpeg_proc.terminate()
                        ffmpeg_proc.wait(timeout=60)
                    except Exception:
                        pass
                except Exception:
                    try:
                        ffmpeg_proc.terminate()
                    except Exception:
                        pass

            # Only write sidecar SRT/ASS/VTT when not burning in (avoids double subtitles when player auto-loads SRT)
            if (self.soft_subs_only or self.inpaint_only_soft_subs) and srt_entries:
                # Last segment ends at frame n (exclusive): visible on [start, n-1]
                srt_entries[-1] = (srt_entries[-1][0], n, srt_entries[-1][2])
                base_path = osp.splitext(self.output_path)[0]
                srt_path = base_path + ".srt"
                _write_srt(srt_path, srt_entries, fps)
                if self.export_ass:
                    _write_ass(base_path + ".ass", srt_entries, fps)
                if self.export_vtt:
                    _write_vtt(base_path + ".vtt", srt_entries, fps)
                # Optionally mux SRT into the video as a subtitle stream (inpaint-only mode only; never with burn-in)
                if self.inpaint_only_soft_subs and self.mux_srt_into_video and osp.isfile(srt_path) and osp.isfile(self.output_path):
                    try:
                        ffmpeg_exe = self.ffmpeg_path if self.ffmpeg_path else "ffmpeg"
                        muxed_path = self.output_path + ".muxed.mp4"
                        subprocess.run([
                            ffmpeg_exe, "-y", "-i", self.output_path, "-i", srt_path,
                            "-c:v", "copy", "-c:s", "mov_text", "-metadata:s:s:0", "language=eng",
                            muxed_path,
                        ], check=True, capture_output=True, timeout=300)
                        os.replace(muxed_path, self.output_path)
                    except Exception as e:
                        LOGGER.warning("Could not mux SRT into video: %s. Output video and SRT file are still saved.", e)

            if osp.isfile(self.output_path):
                self.finished_ok.emit(self.output_path)
            else:
                self.failed.emit("Cancelled.")
        except Exception as e:
            LOGGER.exception("Video translate failed")
            self.failed.emit(str(e))


class VideoTranslatorDialog(QDialog):
    """Dialog to configure and run video translation (hardcoded subtitles -> detect, OCR, translate, inpaint)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.tr("Video translator"))
        self.setMinimumWidth(640)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.thread: Optional[VideoTranslateThread] = None
        self._batch_jobs: list = []  # [(input_path, output_path), ...]
        self._batch_index: int = 0

        main_layout = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setMinimumHeight(400)
        content_widget = QWidget()
        content = QVBoxLayout(content_widget)
        content.setContentsMargins(0, 0, 0, 0)
        scroll.setWidget(content_widget)
        main_layout.addWidget(scroll)

        # Input / output
        g = QGroupBox(self.tr("Video files"))
        gl = QGridLayout(g)
        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText(self.tr("Input video path"))
        self.input_btn = QPushButton(self.tr("Browse..."))
        self.input_btn.clicked.connect(self._browse_input)
        self.input_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        gl.addWidget(QLabel(self.tr("Input:")), 0, 0)
        gl.addWidget(self.input_edit, 0, 1)
        gl.addWidget(self.input_btn, 0, 2)
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText(self.tr("Output video path (default: input_translated.mp4)"))
        self.output_btn = QPushButton(self.tr("Browse..."))
        self.output_btn.clicked.connect(self._browse_output)
        self.output_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        gl.addWidget(QLabel(self.tr("Output:")), 1, 0)
        gl.addWidget(self.output_edit, 1, 1)
        gl.addWidget(self.output_btn, 1, 2)
        gl.setColumnStretch(1, 1)
        content.addWidget(g)

        # Batch
        g_batch = QGroupBox(self.tr("Batch (optional)"))
        g_batch_l = QVBoxLayout(g_batch)
        self.batch_list = QListWidget()
        self.batch_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.batch_list.setMaximumHeight(100)
        g_batch_l.addWidget(QLabel(self.tr("Add video files; when list is non-empty, Run processes all to the output directory below.")))
        g_batch_l.addWidget(self.batch_list)
        batch_btn_row = QHBoxLayout()
        self.batch_add_btn = QPushButton(self.tr("Add files..."))
        self.batch_add_btn.clicked.connect(self._batch_add_files)
        self.batch_folder_btn = QPushButton(self.tr("Add folder..."))
        self.batch_folder_btn.clicked.connect(self._batch_add_folder)
        self.batch_remove_btn = QPushButton(self.tr("Remove"))
        self.batch_remove_btn.clicked.connect(self._batch_remove)
        self.batch_clear_btn = QPushButton(self.tr("Clear"))
        self.batch_clear_btn.clicked.connect(lambda: self.batch_list.clear())
        batch_btn_row.addWidget(self.batch_add_btn)
        batch_btn_row.addWidget(self.batch_folder_btn)
        batch_btn_row.addWidget(self.batch_remove_btn)
        batch_btn_row.addWidget(self.batch_clear_btn)
        batch_btn_row.addStretch()
        g_batch_l.addLayout(batch_btn_row)
        batch_out_row = QHBoxLayout()
        batch_out_row.addWidget(QLabel(self.tr("Output directory:")))
        self.batch_output_edit = QLineEdit()
        self.batch_output_edit.setPlaceholderText(self.tr("All batch outputs go here (e.g. input_translated.mp4)"))
        self.batch_output_edit.setText((getattr(pcfg.module, "video_translator_last_batch_output_dir", None) or "").strip())
        self.batch_output_edit.textChanged.connect(self._save_options)
        batch_out_row.addWidget(self.batch_output_edit, 1)
        self.batch_output_btn = QPushButton(self.tr("Browse..."))
        self.batch_output_btn.clicked.connect(self._browse_batch_output)
        self.batch_output_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        batch_out_row.addWidget(self.batch_output_btn, 0)
        g_batch_l.addLayout(batch_out_row)
        content.addWidget(g_batch)

        # Single column so Pipeline & Source get full width and never collapse (was two-col; browse buttons and left content disappeared when narrow)
        main_col = QVBoxLayout()
        content.addLayout(main_col)

        # Pipeline & sampling
        _pipe_title = self.tr("Pipeline & sampling") or "Pipeline & sampling"
        g2 = QGroupBox(_pipe_title)
        g2l = QVBoxLayout(g2)
        _src_label = self.tr("Source:") or "Source:"
        g2l.addWidget(QLabel(_src_label))
        self.source_combo = QComboBox()
        _src_ocr = self.tr("Hardcoded subtitles (OCR)") or "Hardcoded subtitles (OCR)"
        _src_asr = self.tr("Audio (speech-to-text / ASR)") or "Audio (speech-to-text / ASR)"
        _src_subs = self.tr("Existing subtitles (file or embedded)") or "Existing subtitles (file or embedded)"
        self.source_combo.addItems([_src_ocr, _src_asr, _src_subs])
        _src = (getattr(pcfg.module, "video_translator_source", None) or "ocr").strip().lower()
        self.source_combo.setCurrentIndex({"asr": 1, "existing_subs": 2}.get(_src, 0))
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        self.source_combo.setToolTip(self.tr("OCR = detect and read text on screen. ASR = transcribe from audio (faster-whisper). Existing = sidecar .srt/.ass/.vtt or embedded subtitle stream."))
        g2l.addWidget(self.source_combo)
        self.asr_options_widget = QWidget()
        asr_row = QGridLayout(self.asr_options_widget)
        asr_row.addWidget(QLabel(self.tr("ASR model:")), 0, 0)
        self.asr_model_combo = QComboBox()
        self.asr_model_combo.addItems(["tiny", "base", "small", "medium", "large-v2", "large-v3"])
        _asr_model = (getattr(pcfg.module, "video_translator_asr_model", None) or "base").strip()
        idx = self.asr_model_combo.findText(_asr_model)
        self.asr_model_combo.setCurrentIndex(max(0, idx))
        self.asr_model_combo.currentIndexChanged.connect(self._save_options)
        asr_row.addWidget(self.asr_model_combo, 0, 1)
        asr_row.addWidget(QLabel(self.tr("Device:")), 1, 0)
        self.asr_device_combo = QComboBox()
        self.asr_device_combo.addItems(["cuda", "cpu"])
        _asr_dev = (getattr(pcfg.module, "video_translator_asr_device", None) or "cuda").strip().lower()
        self.asr_device_combo.setCurrentIndex(1 if _asr_dev == "cpu" else 0)
        self.asr_device_combo.currentIndexChanged.connect(self._save_options)
        asr_row.addWidget(self.asr_device_combo, 1, 1)
        asr_row.addWidget(QLabel(self.tr("Language (empty=auto):")), 2, 0)
        self.asr_lang_edit = QLineEdit()
        self.asr_lang_edit.setPlaceholderText("ja, en, zh, ...")
        self.asr_lang_edit.setText((getattr(pcfg.module, "video_translator_asr_language", None) or "").strip())
        self.asr_lang_edit.textChanged.connect(self._save_options)
        asr_row.addWidget(self.asr_lang_edit, 2, 1)
        self.check_asr_vad = QCheckBox(self.tr("VAD filter (reduce hallucinations)"))
        self.check_asr_vad.setChecked(bool(getattr(pcfg.module, "video_translator_asr_vad_filter", True)))
        self.check_asr_vad.stateChanged.connect(self._save_options)
        self.check_asr_vad.setToolTip(self.tr("Filter non-speech segments to reduce ASR hallucinations (VideoCaptioner-style). Recommended on."))
        asr_row.addWidget(self.check_asr_vad, 3, 0, 1, 2)
        self.check_asr_sentence_break = QCheckBox(self.tr("Smart sentence break (LLM)"))
        self.check_asr_sentence_break.setChecked(bool(getattr(pcfg.module, "video_translator_asr_sentence_break", False)))
        self.check_asr_sentence_break.stateChanged.connect(self._save_options)
        self.check_asr_sentence_break.setToolTip(self.tr("Use LLM to merge/split ASR segments into natural sentences. Requires translator. VideoCaptioner-style 智能断句."))
        asr_row.addWidget(self.check_asr_sentence_break, 4, 0, 1, 2)
        self.check_asr_audio_separation = QCheckBox(self.tr("Separate vocals (reduce music noise)"))
        self.check_asr_audio_separation.setChecked(bool(getattr(pcfg.module, "video_translator_asr_audio_separation", False)))
        self.check_asr_audio_separation.stateChanged.connect(self._save_options)
        self.check_asr_audio_separation.setToolTip(self.tr("Use demucs to separate vocals before ASR. Improves accuracy on noisy audio. Optional: pip install demucs."))
        asr_row.addWidget(self.check_asr_audio_separation, 5, 0, 1, 2)
        g2l.addWidget(self.asr_options_widget)
        row = QHBoxLayout()
        _process_every_label = QLabel(self.tr("Process every:") or "Process every:")
        _process_every_label.setToolTip(self.tr(
            "Run detect/OCR/translate/inpaint only every N frames; other frames reuse the last result. "
            "Output video FPS is always the same as input (every frame is written). "
            "Lower N = subtitles update more often but N times more work (e.g. 5 = much slower than 30). "
            "Recommended: 24–30 for speed; use temporal smoothing to reduce flicker."
        ) or "Run pipeline every N frames; other frames reuse last result. Lower N = slower. Recommended: 24–30.")
        row.addWidget(_process_every_label)
        self.sample_spin = QSpinBox()
        self.sample_spin.setRange(1, 300)
        self.sample_spin.setValue(max(1, int(getattr(pcfg.module, "video_translator_sample_every_frames", 30))))
        self.sample_spin.setSuffix(self.tr(" frames") or " frames")
        self.sample_spin.setToolTip(_process_every_label.toolTip())
        self.sample_spin.valueChanged.connect(self._save_options)
        row.addWidget(self.sample_spin)
        row.addStretch()
        g2l.addLayout(row)
        self.check_detect = QCheckBox(self.tr("Detection"))
        self.check_detect.setChecked(bool(getattr(pcfg.module, "video_translator_enable_detect", True)))
        self.check_detect.stateChanged.connect(self._save_options)
        self.check_ocr = QCheckBox(self.tr("OCR"))
        self.check_ocr.setChecked(bool(getattr(pcfg.module, "video_translator_enable_ocr", True)))
        self.check_ocr.stateChanged.connect(self._save_options)
        self.check_translate = QCheckBox(self.tr("Translation"))
        self.check_translate.setChecked(bool(getattr(pcfg.module, "video_translator_enable_translate", True)))
        self.check_translate.stateChanged.connect(self._save_options)
        self.check_inpaint = QCheckBox(self.tr("Inpainting"))
        self.check_inpaint.setChecked(bool(getattr(pcfg.module, "video_translator_enable_inpaint", True)))
        self.check_inpaint.stateChanged.connect(self._save_options)
        row2 = QHBoxLayout()
        row2.addWidget(self.check_detect)
        row2.addWidget(self.check_ocr)
        row2.addWidget(self.check_translate)
        row2.addWidget(self.check_inpaint)
        g2l.addLayout(row2)
        main_col.addWidget(g2)

        # Region & output codec (inspired by subtitle ROI in video-subtitle-extractor / subtitle-quality tools)
        g3 = QGroupBox(self.tr("Subtitle region & output"))
        g3l = QGridLayout(g3)
        g3l.addWidget(QLabel(self.tr("Subtitle region:")), 0, 0)
        self.region_combo = QComboBox()
        self.region_combo.addItems([
            self.tr("Full frame"),
            self.tr("Bottom 15%"),
            self.tr("Bottom 20%"),
            self.tr("Bottom 25%"),
            self.tr("Bottom 30%"),
        ])
        _region_to_idx = {"full": 0, "bottom_15": 1, "bottom_20": 2, "bottom_25": 3, "bottom_30": 4}
        _region = (getattr(pcfg.module, "video_translator_region_preset", None) or "full").strip().lower()
        self.region_combo.setCurrentIndex(_region_to_idx.get(_region, 0))
        self.region_combo.currentIndexChanged.connect(self._save_options)
        self.region_combo.setToolTip(self.tr("Only process text blocks in this area (e.g. Bottom 20%% = subtitle band). Speeds up and reduces false positives. Inspired by ROI in video-subtitle-extractor / subtitle-quality tools."))
        g3l.addWidget(self.region_combo, 0, 1)
        g3l.addWidget(QLabel(self.tr("Output codec (FourCC):")), 1, 0)
        self.codec_edit = QLineEdit()
        self.codec_edit.setPlaceholderText("mp4v")
        self.codec_edit.setMaxLength(4)
        self.codec_edit.setToolTip(self.tr("OpenCV FourCC: mp4v (default), avc1 (H.264), XVID, etc. Leave default if playback has issues."))
        self.codec_edit.setText((getattr(pcfg.module, "video_translator_output_codec", None) or "mp4v").strip() or "mp4v")
        self.codec_edit.textChanged.connect(self._save_options)
        g3l.addWidget(self.codec_edit, 1, 1)
        g3l.addWidget(QLabel(self.tr("Burn-in style:")), 2, 0)
        self.style_combo = QComboBox()
        self.style_combo.addItems([self.tr("Default"), self.tr("Anime (larger)"), self.tr("Documentary (smaller)")])
        _style = (getattr(pcfg.module, "video_translator_subtitle_style", None) or "default").strip().lower()
        self.style_combo.setCurrentIndex({"anime": 1, "documentary": 2}.get(_style, 0))
        self.style_combo.currentIndexChanged.connect(self._save_options)
        self.style_combo.setToolTip(self.tr("Subtitle look when burning in. Anime = larger text; Documentary = smaller. Inspired by VideoCaptioner."))
        g3l.addWidget(self.style_combo, 2, 1)
        g3l.addWidget(QLabel(self.tr("Subtitle font:")), 3, 0)
        self.subtitle_font_edit = QLineEdit()
        self.subtitle_font_edit.setPlaceholderText(self.tr("Empty = Arial / default"))
        self.subtitle_font_edit.setText((getattr(pcfg.module, "video_translator_subtitle_font", None) or "").strip())
        self.subtitle_font_edit.setToolTip(self.tr("Optional path to a .ttf or .otf file for burn-in subtitles. Leave empty to use the default font (Arial/DejaVu)."))
        self.subtitle_font_edit.textChanged.connect(self._save_options)
        g3l.addWidget(self.subtitle_font_edit, 3, 1)
        _font_browse_btn = QPushButton(self.tr("Browse..."))
        _font_browse_btn.clicked.connect(self._browse_subtitle_font)
        _font_browse_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        g3l.addWidget(_font_browse_btn, 3, 2)
        g3l.setColumnStretch(1, 1)
        self.check_soft_subs = QCheckBox(self.tr("Soft subtitles only (no burn-in; output video + SRT, very fast)"))
        self.check_soft_subs.setChecked(bool(getattr(pcfg.module, "video_translator_soft_subs_only", False)))
        self.check_soft_subs.stateChanged.connect(self._on_soft_subs_changed)
        self.check_soft_subs.setToolTip(self.tr("Output original video unchanged and translated subtitles in SRT. No inpainting or text drawn on video; play with a player that loads the SRT. Much faster (VideoCaptioner-style)."))
        g3l.addWidget(self.check_soft_subs, 4, 0, 1, 2)
        self.check_inpaint_only_soft = QCheckBox(self.tr("Inpaint only (no burn-in; SRT/ASS/VTT only)"))
        self.check_inpaint_only_soft.setChecked(bool(getattr(pcfg.module, "video_translator_inpaint_only_soft_subs", False)))
        self.check_inpaint_only_soft.stateChanged.connect(self._on_inpaint_only_soft_changed)
        self.check_inpaint_only_soft.setToolTip(self.tr("Run full pipeline (detect, OCR, translate, inpaint) but do not draw subtitles on the video. Output: inpainted video + SRT/ASS/VTT files. Use so the video never has both hardcoded and soft subs at once."))
        g3l.addWidget(self.check_inpaint_only_soft, 5, 0, 1, 2)
        self.check_mux_srt = QCheckBox(self.tr("Mux SRT into video (inpaint-only mode)"))
        self.check_mux_srt.setChecked(bool(getattr(pcfg.module, "video_translator_mux_srt_into_video", False)))
        self.check_mux_srt.stateChanged.connect(self._save_options)
        self.check_mux_srt.setToolTip(self.tr("When 'Inpaint only' is on: add the SRT as a subtitle stream inside the output video file. Requires FFmpeg. If off, SRT is only a sidecar file."))
        g3l.addWidget(self.check_mux_srt, 6, 0, 1, 2)
        self.check_mux_srt.setEnabled(self.check_inpaint_only_soft.isChecked())
        main_col.addWidget(g3)

        # Advanced: scene, smoothing, FFmpeg, export
        g4_row = QHBoxLayout()
        # Left: Scene & smoothing
        g4a = QGroupBox(self.tr("Scene & smoothing"))
        g4al = QGridLayout(g4a)
        self.check_scene = QCheckBox(self.tr("Scene detection (pipeline only on scene changes)"))
        self.check_scene.setChecked(bool(getattr(pcfg.module, "video_translator_use_scene_detection", False)))
        self.check_scene.stateChanged.connect(self._save_options)
        g4al.addWidget(self.check_scene, 0, 0, 1, 2)
        g4al.addWidget(QLabel(self.tr("Scene threshold:")), 1, 0)
        self.scene_threshold_spin = QSpinBox()
        self.scene_threshold_spin.setRange(5, 100)
        self.scene_threshold_spin.setValue(int(float(getattr(pcfg.module, "video_translator_scene_threshold", 30))))
        self.scene_threshold_spin.setSuffix(" (higher = fewer cuts)")
        self.scene_threshold_spin.valueChanged.connect(self._save_options)
        g4al.addWidget(self.scene_threshold_spin, 1, 1)
        self.check_temporal = QCheckBox(self.tr("Temporal smoothing (blend with previous frame)"))
        self.check_temporal.setChecked(bool(getattr(pcfg.module, "video_translator_temporal_smoothing", False)))
        self.check_temporal.stateChanged.connect(self._save_options)
        g4al.addWidget(self.check_temporal, 2, 0, 1, 2)
        g4al.addWidget(QLabel(self.tr("Blend weight:")), 3, 0)
        self.temporal_alpha_spin = QDoubleSpinBox()
        self.temporal_alpha_spin.setRange(0.05, 0.95)
        self.temporal_alpha_spin.setSingleStep(0.05)
        self.temporal_alpha_spin.setValue(float(getattr(pcfg.module, "video_translator_temporal_alpha", 0.25)))
        self.temporal_alpha_spin.valueChanged.connect(self._save_options)
        g4al.addWidget(self.temporal_alpha_spin, 3, 1)
        g4_row.addWidget(g4a, 1)
        # Right: FFmpeg & export
        g4b = QGroupBox(self.tr("FFmpeg & export"))
        g4bl = QGridLayout(g4b)
        self.check_ffmpeg = QCheckBox(self.tr("Use FFmpeg (libx264, better compatibility)"))
        self.check_ffmpeg.setChecked(bool(getattr(pcfg.module, "video_translator_use_ffmpeg", False)))
        self.check_ffmpeg.stateChanged.connect(self._save_options)
        g4bl.addWidget(self.check_ffmpeg, 0, 0, 1, 2)
        g4bl.addWidget(QLabel(self.tr("FFmpeg path:")), 1, 0)
        self.ffmpeg_path_edit = QLineEdit()
        self.ffmpeg_path_edit.setPlaceholderText(self.tr("Empty = use PATH"))
        self.ffmpeg_path_edit.setText((getattr(pcfg.module, "video_translator_ffmpeg_path", None) or "").strip())
        self.ffmpeg_path_edit.setToolTip(self.tr("If FFmpeg is not on PATH, set full path to ffmpeg.exe."))
        self.ffmpeg_path_edit.textChanged.connect(self._save_options)
        g4bl.addWidget(self.ffmpeg_path_edit, 1, 1)
        ffmpeg_browse_btn = QPushButton(self.tr("Browse..."))
        ffmpeg_browse_btn.clicked.connect(self._browse_ffmpeg)
        ffmpeg_browse_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        g4bl.addWidget(ffmpeg_browse_btn, 1, 2)
        g4bl.setColumnStretch(1, 1)
        g4bl.addWidget(QLabel(self.tr("CRF (0–51):")), 2, 0)
        self.ffmpeg_crf_spin = QSpinBox()
        self.ffmpeg_crf_spin.setRange(0, 51)
        self.ffmpeg_crf_spin.setValue(int(getattr(pcfg.module, "video_translator_ffmpeg_crf", 18)))
        self.ffmpeg_crf_spin.setToolTip(self.tr("Lower = better quality (18 = high; 23 = medium). Ignored when Target bitrate is set."))
        self.ffmpeg_crf_spin.valueChanged.connect(self._save_options)
        g4bl.addWidget(self.ffmpeg_crf_spin, 2, 1)
        g4bl.addWidget(QLabel(self.tr("Target bitrate (kbps, 0=CRF only):")), 2, 2)
        self.video_bitrate_spin = QSpinBox()
        self.video_bitrate_spin.setRange(0, 50000)
        self.video_bitrate_spin.setValue(int(getattr(pcfg.module, "video_translator_video_bitrate_kbps", 0)))
        self.video_bitrate_spin.setSpecialValueText("0 (CRF)")
        self.video_bitrate_spin.setToolTip(self.tr("0 = use source video bitrate when detectable (same as original), else use CRF. Set a value (e.g. 9600) to override. Only when FFmpeg is enabled."))
        self.video_bitrate_spin.valueChanged.connect(self._save_options)
        g4bl.addWidget(self.video_bitrate_spin, 2, 3)
        self.check_skip_detect = QCheckBox(self.tr("Skip detection (fixed region only)"))
        self.check_skip_detect.setChecked(bool(getattr(pcfg.module, "video_translator_skip_detect", False)))
        self.check_skip_detect.setToolTip(self.tr("Use one block for subtitle region; no text detection. Set region above (e.g. Bottom 20%%)."))
        self.check_skip_detect.stateChanged.connect(self._on_skip_detect_changed)
        g4bl.addWidget(self.check_skip_detect, 3, 0, 1, 2)
        self.check_export_srt = QCheckBox(self.tr("Export SRT"))
        self.check_export_srt.setChecked(bool(getattr(pcfg.module, "video_translator_export_srt", False)))
        self.check_export_srt.stateChanged.connect(self._save_options)
        self.check_export_srt.setToolTip(self.tr("Write SRT file. Only available when using Soft subs only or Inpaint only (no burn-in); disabled when burning in to avoid double subtitles in players."))
        g4bl.addWidget(self.check_export_srt, 4, 0, 1, 2)
        self.check_export_ass = QCheckBox(self.tr("Export ASS"))
        self.check_export_ass.setChecked(bool(getattr(pcfg.module, "video_translator_export_ass", False)))
        self.check_export_ass.stateChanged.connect(self._save_options)
        self.check_export_ass.setToolTip(self.tr("Write ASS subtitle file (same timing as SRT)."))
        g4bl.addWidget(self.check_export_ass, 5, 0, 1, 2)
        self.check_export_vtt = QCheckBox(self.tr("Export WebVTT"))
        self.check_export_vtt.setChecked(bool(getattr(pcfg.module, "video_translator_export_vtt", False)))
        self.check_export_vtt.stateChanged.connect(self._save_options)
        self.check_export_vtt.setToolTip(self.tr("Write WebVTT subtitle file (same timing as SRT)."))
        g4bl.addWidget(self.check_export_vtt, 6, 0, 1, 2)
        g4_row.addWidget(g4b, 1)
        main_col.addLayout(g4_row)
        self._update_export_subs_enabled()

        # Context: glossary hint + series context path (same as project translation context) (right column)
        g_ctx = QGroupBox(self.tr("Context (optional)"))
        g_ctx_l = QVBoxLayout(g_ctx)
        g_ctx_l.addWidget(QLabel(self.tr("Glossary / script hint:")))
        self.glossary_edit = QPlainTextEdit()
        self.glossary_edit.setPlaceholderText(self.tr("Terminology, names, script excerpt or correction hints…"))
        self.glossary_edit.setMaximumHeight(100)
        self.glossary_edit.setToolTip(self.tr("Optional. Sent to the LLM for OCR/ASR correction and translation (e.g. terms, names, script excerpt). VideoCaptioner-style."))
        self.glossary_edit.setPlainText((getattr(pcfg.module, "video_translator_glossary", None) or "").strip())
        self.glossary_edit.textChanged.connect(self._save_options)
        g_ctx_l.addWidget(self.glossary_edit)
        ctx_row = QHBoxLayout()
        ctx_row.addWidget(QLabel(self.tr("Series context path:")))
        self.series_context_edit = QLineEdit()
        self.series_context_edit.setPlaceholderText(self.tr("Folder or series ID (e.g. my_anime); leave empty to use translator's config"))
        self.series_context_edit.setToolTip(self.tr("Optional. Same as project series context: loads glossary and recent context from data/translation_context/<id>/ or the given folder. Improves consistency across videos."))
        self.series_context_edit.setText((getattr(pcfg.module, "video_translator_series_context_path", None) or "").strip())
        self.series_context_edit.textChanged.connect(self._save_options)
        ctx_row.addWidget(self.series_context_edit, 1)
        self.series_context_btn = QPushButton(self.tr("Browse..."))
        self.series_context_btn.clicked.connect(self._browse_series_context)
        self.series_context_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        ctx_row.addWidget(self.series_context_btn, 0)
        g_ctx_l.addLayout(ctx_row)
        main_col.addWidget(g_ctx)

        # Flow fixer: local model or OpenRouter (second model) to improve subtitle flow
        g_flow = QGroupBox(self.tr("Flow fixer (optional)"))
        g_flow_l = QGridLayout(g_flow)
        self.check_flow_fixer = QCheckBox(self.tr("Use flow fixer (smooth subtitle flow after translation)"))
        self.check_flow_fixer.setChecked(bool(getattr(pcfg.module, "video_translator_flow_fixer_enabled", False)))
        self.check_flow_fixer.setToolTip(self.tr("After translation, a second pass can improve flow. Local = free (Ollama/LM Studio). OpenAI = use ChatGPT credits (gpt-4o-mini is cheap). See docs/FLOW_FIXER_SETUP.md for model list."))
        self.check_flow_fixer.stateChanged.connect(self._save_options)
        g_flow_l.addWidget(self.check_flow_fixer, 0, 0, 1, 2)
        g_flow_l.addWidget(QLabel(self.tr("Flow fixer:")), 1, 0)
        self.flow_fixer_combo = QComboBox()
        self.flow_fixer_combo.addItems([
            self.tr("None"),
            self.tr("Local server (Ollama or LM Studio)"),
            self.tr("OpenRouter (second model)"),
            self.tr("OpenAI / ChatGPT (use credits)"),
        ])
        _flow_name = (getattr(pcfg.module, "video_translator_flow_fixer", None) or "none").strip().lower()
        self.flow_fixer_combo.setCurrentIndex(
            3 if _flow_name == "openai" else (2 if _flow_name == "openrouter" else (1 if _flow_name == "local_server" else 0))
        )
        self.flow_fixer_combo.currentIndexChanged.connect(self._save_options)
        g_flow_l.addWidget(self.flow_fixer_combo, 1, 1)
        g_flow_l.addWidget(QLabel(self.tr("Server URL:")), 2, 0)
        self.flow_fixer_url_edit = QLineEdit()
        self.flow_fixer_url_edit.setPlaceholderText("http://localhost:1234/v1")
        self.flow_fixer_url_edit.setText((getattr(pcfg.module, "video_translator_flow_fixer_server_url", None) or "http://localhost:1234/v1").strip())
        self.flow_fixer_url_edit.setToolTip(self.tr("LM Studio: http://localhost:1234/v1  —  Ollama: http://localhost:11434/v1"))
        self.flow_fixer_url_edit.textChanged.connect(self._save_options)
        g_flow_l.addWidget(self.flow_fixer_url_edit, 2, 1)
        g_flow_l.addWidget(QLabel(self.tr("Model (local):")), 3, 0)
        self.flow_fixer_model_edit = QLineEdit()
        self.flow_fixer_model_edit.setPlaceholderText("local")
        self.flow_fixer_model_edit.setText((getattr(pcfg.module, "video_translator_flow_fixer_model", None) or "local").strip())
        self.flow_fixer_model_edit.setToolTip(self.tr("Ollama: e.g. qwen2.5:3b, phi3:mini, llama3.2:3b. LM Studio: model name when server runs (often \"local\"). Good local: Qwen2.5-3B, Phi-3-mini, Llama-3.2-3B."))
        self.flow_fixer_model_edit.textChanged.connect(self._save_options)
        g_flow_l.addWidget(self.flow_fixer_model_edit, 3, 1)
        g_flow_l.addWidget(QLabel(self.tr("OpenRouter API key:")), 5, 0)
        self.flow_fixer_openrouter_apikey_edit = QLineEdit()
        self.flow_fixer_openrouter_apikey_edit.setPlaceholderText("Same as translator or paste key")
        self.flow_fixer_openrouter_apikey_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.flow_fixer_openrouter_apikey_edit.setText((getattr(pcfg.module, "video_translator_flow_fixer_openrouter_apikey", None) or "").strip())
        self.flow_fixer_openrouter_apikey_edit.setToolTip(self.tr("OpenRouter API key for the flow-fixer model. Can be the same key as your main translator."))
        self.flow_fixer_openrouter_apikey_edit.textChanged.connect(self._save_options)
        g_flow_l.addWidget(self.flow_fixer_openrouter_apikey_edit, 5, 1)
        g_flow_l.addWidget(QLabel(self.tr("OpenRouter model:")), 6, 0)
        self.flow_fixer_openrouter_model_edit = QLineEdit()
        self.flow_fixer_openrouter_model_edit.setPlaceholderText("google/gemma-3n-e2b-it:free")
        self.flow_fixer_openrouter_model_edit.setText((getattr(pcfg.module, "video_translator_flow_fixer_openrouter_model", None) or "google/gemma-3n-e2b-it:free").strip())
        self.flow_fixer_openrouter_model_edit.setToolTip(self.tr("Small/free model for flow only. Examples: google/gemma-3n-e2b-it:free, qwen/qwen3-4b:free"))
        self.flow_fixer_openrouter_model_edit.textChanged.connect(self._save_options)
        g_flow_l.addWidget(self.flow_fixer_openrouter_model_edit, 6, 1)
        g_flow_l.addWidget(QLabel(self.tr("OpenAI API key:")), 7, 0)
        self.flow_fixer_openai_apikey_edit = QLineEdit()
        self.flow_fixer_openai_apikey_edit.setPlaceholderText("sk-... (platform.openai.com API key)")
        self.flow_fixer_openai_apikey_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.flow_fixer_openai_apikey_edit.setText((getattr(pcfg.module, "video_translator_flow_fixer_openai_apikey", None) or "").strip())
        self.flow_fixer_openai_apikey_edit.setToolTip(self.tr("OpenAI API key (ChatGPT credits). Get one at platform.openai.com → API keys. Use gpt-4o-mini or gpt-3.5-turbo for cheap flow passes."))
        self.flow_fixer_openai_apikey_edit.textChanged.connect(self._save_options)
        g_flow_l.addWidget(self.flow_fixer_openai_apikey_edit, 7, 1)
        g_flow_l.addWidget(QLabel(self.tr("OpenAI model:")), 8, 0)
        self.flow_fixer_openai_model_edit = QLineEdit()
        self.flow_fixer_openai_model_edit.setPlaceholderText("gpt-4o-mini")
        self.flow_fixer_openai_model_edit.setText((getattr(pcfg.module, "video_translator_flow_fixer_openai_model", None) or "gpt-4o-mini").strip())
        self.flow_fixer_openai_model_edit.setToolTip(self.tr("Cheap and good: gpt-4o-mini (default). Even cheaper: gpt-3.5-turbo. Uses your ChatGPT/OpenAI credits."))
        self.flow_fixer_openai_model_edit.textChanged.connect(self._save_options)
        g_flow_l.addWidget(self.flow_fixer_openai_model_edit, 8, 1)
        g_flow_l.addWidget(QLabel(self.tr("Max tokens:")), 4, 0)
        self.flow_fixer_max_tokens_spin = QSpinBox()
        self.flow_fixer_max_tokens_spin.setRange(64, 999999)
        self.flow_fixer_max_tokens_spin.setValue(int(getattr(pcfg.module, "video_translator_flow_fixer_max_tokens", 256)))
        self.flow_fixer_max_tokens_spin.valueChanged.connect(self._save_options)
        g_flow_l.addWidget(self.flow_fixer_max_tokens_spin, 4, 1)
        g_flow_l.addWidget(QLabel(self.tr("Context lines:")), 9, 0)
        self.flow_fixer_context_spin = QSpinBox()
        self.flow_fixer_context_spin.setRange(1, 50)
        self.flow_fixer_context_spin.setValue(int(getattr(pcfg.module, "video_translator_flow_fixer_context_lines", 20)))
        self.flow_fixer_context_spin.setToolTip(self.tr("How many previous subtitle lines to send for flow (1–50). Higher = more context; may trigger summarization when long. Default 20."))
        self.flow_fixer_context_spin.valueChanged.connect(self._save_options)
        g_flow_l.addWidget(self.flow_fixer_context_spin, 9, 1)
        g_flow_l.setColumnStretch(1, 1)
        main_col.addWidget(g_flow)

        # Recommended for long videos / anime
        tips_row = QHBoxLayout()
        long_tips = self.tr(
            "Long videos (e.g. 17h): Use FFmpeg, Scene detection, and optionally Skip detection + Bottom 20%% for anime."
        )
        self.tips_label = QLabel(long_tips)
        self.tips_label.setWordWrap(True)
        self.tips_label.setStyleSheet("color: gray; font-size: 0.9em;")
        self.tips_label.setToolTip(
            self.tr(
                "Recommended for long anime: Enable Scene detection (threshold 25–35), Skip detection + Subtitle region Bottom 20%%, "
                "Use FFmpeg (CRF 23), Process every 24–30 frames. Temporal smoothing 0.2–0.3 reduces flicker. All models work; "
                "progress is throttled for very long runs. If you see 'CUDA out of memory' with inpainting, set Inpainter device to CPU or lower inpaint_size in Config → Inpainting."
            )
        )
        tips_row.addWidget(self.tips_label)
        self.apply_long_anime_btn = QPushButton(self.tr("Apply recommended (long anime)"))
        self.apply_long_anime_btn.setToolTip(self.tr("Set options optimized for long anime (e.g. Rebirth of the Urban Immortal Cultivator)."))
        self.apply_long_anime_btn.clicked.connect(self._apply_recommended_long_anime)
        tips_row.addWidget(self.apply_long_anime_btn)
        content.addLayout(tips_row)

        # When Skip detection is on, Detection is not used
        self.check_detect.setEnabled(not self.check_skip_detect.isChecked())

        # Live preview (OCR pipeline: current frame + detected/translated subtitles)
        g_preview = QGroupBox(self.tr("Live preview (current frame)"))
        g_preview_l = QHBoxLayout(g_preview)
        self.preview_frame_label = QLabel()
        self.preview_frame_label.setMinimumSize(320, 180)
        self.preview_frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_frame_label.setStyleSheet("background-color: #1a1a1a; color: #666;")
        self.preview_frame_label.setText(self.tr("Frame and subtitles will appear here during OCR run"))
        self.preview_frame_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        g_preview_l.addWidget(self.preview_frame_label, 1)
        # Text preview (detected / translated lines) – use a read-only plain text editor
        # so users can select and copy subtitles during the run.
        self.preview_text_edit = QPlainTextEdit()
        self.preview_text_edit.setReadOnly(True)
        self.preview_text_edit.setUndoRedoEnabled(False)
        self.preview_text_edit.setMinimumWidth(200)
        self.preview_text_edit.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.preview_text_edit.setStyleSheet("color: #ccc; font-size: 0.9em; background: #202020;")
        self.preview_text_edit.setPlainText(self.tr("Detected / translated lines appear here during run."))
        g_preview_l.addWidget(self.preview_text_edit, 1)
        content.addWidget(g_preview)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.status_label = QLabel("")
        content.addWidget(self.status_label)
        content.addWidget(self.progress_bar)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.edit_subs_btn = QPushButton(self.tr("Edit subtitles..."))
        self.edit_subs_btn.setToolTip(self.tr("Open Video Subtitle Editor to cut, edit captions, and export."))
        self.edit_subs_btn.clicked.connect(self._open_subtitle_editor)
        btn_layout.addWidget(self.edit_subs_btn)
        self.run_btn = QPushButton(self.tr("Run"))
        self.run_btn.clicked.connect(self._run)
        self.cancel_btn = QPushButton(self.tr("Cancel"))
        self.cancel_btn.clicked.connect(self._cancel_run)
        self.close_btn = QPushButton(self.tr("Close"))
        self.close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(self.run_btn)
        btn_layout.addWidget(self.cancel_btn)
        btn_layout.addWidget(self.close_btn)
        content.addLayout(btn_layout)

        self.cancel_btn.setEnabled(False)
        self._on_source_changed()  # Initial visibility/enabled state (after all widgets exist)
        self._load_paths()

    def _load_paths(self):
        """Restore last used paths from config."""
        inp = (getattr(pcfg.module, "video_translator_last_input_path", None) or "").strip()
        out = (getattr(pcfg.module, "video_translator_last_output_path", None) or "").strip()
        if inp:
            self.input_edit.setText(inp)
        if out:
            self.output_edit.setText(out)

    def _browse_series_context(self):
        """Pick a folder for series context (glossary/recent context). User can also type a series ID (e.g. my_anime) manually."""
        start = (self.series_context_edit.text() or "").strip()
        if not start or not osp.isdir(start):
            try:
                from utils.series_context_store import DEFAULT_SERIES_CONTEXT_DIR
                start = DEFAULT_SERIES_CONTEXT_DIR if osp.isdir(DEFAULT_SERIES_CONTEXT_DIR) else osp.dirname(DEFAULT_SERIES_CONTEXT_DIR)
            except Exception:
                start = ""
        path = QFileDialog.getExistingDirectory(self, self.tr("Series context folder"), start)
        if path:
            self.series_context_edit.setText(path)
            self._save_options()

    def _open_subtitle_editor(self):
        """Open Video Subtitle Editor with current input or output video and optional sidecar SRT."""
        from .video_subtitle_editor import VideoSubtitleEditorWindow
        out_path = (self.output_edit.text() or "").strip()
        inp_path = (self.input_edit.text() or "").strip()
        video_path = out_path if out_path and osp.isfile(out_path) else (inp_path if inp_path and osp.isfile(inp_path) else "")
        sub_path = ""
        if video_path:
            base = osp.splitext(video_path)[0]
            for ext in (".srt", ".ass", ".vtt"):
                p = base + ext
                if osp.isfile(p):
                    sub_path = p
                    break
        win = VideoSubtitleEditorWindow(self, video_path=video_path, subtitle_path=sub_path)
        win.show()

    def _region_preset_value(self) -> str:
        idx = self.region_combo.currentIndex()
        return ("full", "bottom_15", "bottom_20", "bottom_25", "bottom_30")[min(max(0, idx), 4)]

    def _on_source_changed(self):
        """When Source is ASR, show ASR options; when OCR, enable pipeline options; when Existing subs, disable pipeline."""
        idx = self.source_combo.currentIndex()
        is_asr = idx == 1
        is_timed = is_asr or idx == 2  # ASR or Existing subtitles: no per-frame pipeline
        self.asr_options_widget.setVisible(is_asr)
        self.sample_spin.setEnabled(not is_timed)
        self.check_detect.setEnabled(not is_timed and not self.check_skip_detect.isChecked())
        self.check_ocr.setEnabled(not is_timed)
        self.check_inpaint.setEnabled(not is_timed)
        self._save_options()

    def _on_skip_detect_changed(self):
        """When Skip detection is checked, disable Detection checkbox (detector not used)."""
        skip = self.check_skip_detect.isChecked()
        self.check_detect.setEnabled(not skip and self.source_combo.currentIndex() == 0)
        self._save_options()

    def _update_export_subs_enabled(self):
        """Export SRT/ASS/VTT only when not burning in (soft subs only or inpaint only), to avoid double subtitles."""
        no_burn_in = self.check_soft_subs.isChecked() or self.check_inpaint_only_soft.isChecked()
        self.check_export_srt.setEnabled(no_burn_in)
        self.check_export_ass.setEnabled(no_burn_in)
        self.check_export_vtt.setEnabled(no_burn_in)

    def _on_soft_subs_changed(self):
        """When Soft subtitles only is checked, SRT is always exported; inpainting is skipped for output."""
        soft = self.check_soft_subs.isChecked()
        if soft:
            self.check_export_srt.setChecked(True)
            self.check_inpaint_only_soft.setChecked(False)
        self._update_export_subs_enabled()
        self._save_options()

    def _on_inpaint_only_soft_changed(self):
        """When Inpaint only (no burn-in) is checked, ensure soft_subs_only is off; enable/disable Mux SRT and export options."""
        inpaint_only = self.check_inpaint_only_soft.isChecked()
        if inpaint_only:
            self.check_soft_subs.setChecked(False)
            self.check_export_srt.setChecked(True)
        self.check_mux_srt.setEnabled(inpaint_only)
        self._update_export_subs_enabled()
        self._save_options()

    def _apply_recommended_long_anime(self):
        """Apply settings recommended for long anime (e.g. 10–17h): scene detection, skip detect + bottom region, FFmpeg, temporal smoothing."""
        self.sample_spin.setValue(30)
        self.region_combo.setCurrentIndex(2)  # Bottom 20%
        self.check_scene.setChecked(True)
        self.scene_threshold_spin.setValue(30)
        self.check_temporal.setChecked(True)
        self.temporal_alpha_spin.setValue(0.25)
        self.check_ffmpeg.setChecked(True)
        self.ffmpeg_crf_spin.setValue(18)
        self.check_skip_detect.setChecked(True)  # Also disables Detection via _on_skip_detect_changed
        self.check_ocr.setChecked(True)
        self.check_translate.setChecked(True)
        self.check_inpaint.setChecked(True)
        self._save_options()

    def _save_options(self):
        """Persist pipeline options to config and save."""
        pcfg.module.video_translator_sample_every_frames = self.sample_spin.value()
        pcfg.module.video_translator_enable_detect = self.check_detect.isChecked()
        pcfg.module.video_translator_enable_ocr = self.check_ocr.isChecked()
        pcfg.module.video_translator_enable_translate = self.check_translate.isChecked()
        pcfg.module.video_translator_enable_inpaint = self.check_inpaint.isChecked()
        pcfg.module.video_translator_region_preset = self._region_preset_value()
        pcfg.module.video_translator_output_codec = (self.codec_edit.text() or "").strip() or "mp4v"
        pcfg.module.video_translator_use_scene_detection = self.check_scene.isChecked()
        pcfg.module.video_translator_scene_threshold = float(self.scene_threshold_spin.value())
        pcfg.module.video_translator_temporal_smoothing = self.check_temporal.isChecked()
        pcfg.module.video_translator_temporal_alpha = float(self.temporal_alpha_spin.value())
        pcfg.module.video_translator_use_ffmpeg = self.check_ffmpeg.isChecked()
        pcfg.module.video_translator_ffmpeg_path = (self.ffmpeg_path_edit.text() or "").strip()
        pcfg.module.video_translator_ffmpeg_crf = int(self.ffmpeg_crf_spin.value())
        pcfg.module.video_translator_video_bitrate_kbps = int(self.video_bitrate_spin.value())
        pcfg.module.video_translator_skip_detect = self.check_skip_detect.isChecked()
        pcfg.module.video_translator_export_srt = self.check_export_srt.isChecked()
        style_idx = self.style_combo.currentIndex()
        pcfg.module.video_translator_subtitle_style = ("default", "anime", "documentary")[min(style_idx, 2)]
        pcfg.module.video_translator_subtitle_font = (self.subtitle_font_edit.text() or "").strip()
        _idx = self.source_combo.currentIndex()
        pcfg.module.video_translator_source = ("asr" if _idx == 1 else "existing_subs" if _idx == 2 else "ocr")
        pcfg.module.video_translator_asr_model = (self.asr_model_combo.currentText() or "base").strip()
        pcfg.module.video_translator_asr_device = (self.asr_device_combo.currentText() or "cuda").strip().lower()
        pcfg.module.video_translator_asr_language = (self.asr_lang_edit.text() or "").strip()
        pcfg.module.video_translator_asr_vad_filter = self.check_asr_vad.isChecked()
        pcfg.module.video_translator_asr_sentence_break = self.check_asr_sentence_break.isChecked()
        pcfg.module.video_translator_asr_audio_separation = self.check_asr_audio_separation.isChecked()
        pcfg.module.video_translator_export_ass = self.check_export_ass.isChecked()
        pcfg.module.video_translator_export_vtt = self.check_export_vtt.isChecked()
        pcfg.module.video_translator_glossary = (self.glossary_edit.toPlainText() or "").strip()
        pcfg.module.video_translator_series_context_path = (self.series_context_edit.text() or "").strip()
        pcfg.module.video_translator_last_batch_output_dir = (self.batch_output_edit.text() or "").strip()
        pcfg.module.video_translator_soft_subs_only = self.check_soft_subs.isChecked()
        pcfg.module.video_translator_inpaint_only_soft_subs = self.check_inpaint_only_soft.isChecked()
        pcfg.module.video_translator_mux_srt_into_video = self.check_mux_srt.isChecked()
        pcfg.module.video_translator_flow_fixer_enabled = self.check_flow_fixer.isChecked()
        idx = self.flow_fixer_combo.currentIndex()
        pcfg.module.video_translator_flow_fixer = "openai" if idx == 3 else ("openrouter" if idx == 2 else ("local_server" if idx == 1 else "none"))
        pcfg.module.video_translator_flow_fixer_server_url = (self.flow_fixer_url_edit.text() or "").strip() or "http://localhost:1234/v1"
        pcfg.module.video_translator_flow_fixer_model = (self.flow_fixer_model_edit.text() or "").strip() or "local"
        pcfg.module.video_translator_flow_fixer_max_tokens = self.flow_fixer_max_tokens_spin.value()
        pcfg.module.video_translator_flow_fixer_context_lines = self.flow_fixer_context_spin.value()
        pcfg.module.video_translator_flow_fixer_timeout = float(getattr(pcfg.module, "video_translator_flow_fixer_timeout", 30.0))
        pcfg.module.video_translator_flow_fixer_openrouter_apikey = (self.flow_fixer_openrouter_apikey_edit.text() or "").strip()
        pcfg.module.video_translator_flow_fixer_openrouter_model = (self.flow_fixer_openrouter_model_edit.text() or "").strip() or "google/gemma-3n-e2b-it:free"
        pcfg.module.video_translator_flow_fixer_openai_apikey = (self.flow_fixer_openai_apikey_edit.text() or "").strip()
        pcfg.module.video_translator_flow_fixer_openai_model = (self.flow_fixer_openai_model_edit.text() or "").strip() or "gpt-4o-mini"
        save_config()

    def _save_paths(self, inp: str, out: str):
        """Persist last used paths to config and save."""
        pcfg.module.video_translator_last_input_path = inp or ""
        pcfg.module.video_translator_last_output_path = out or ""
        save_config()

    def _browse_input(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Select input video"),
            "",
            self.tr("Video files (*.mp4 *.avi *.mkv *.mov *.webm);;All files (*)"),
        )
        if path:
            self.input_edit.setText(path)
            if not self.output_edit.text().strip():
                base, ext = osp.splitext(path)
                self.output_edit.setText(base + "_translated.mp4")

    def _browse_output(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            self.tr("Select output video"),
            self.output_edit.text() or "",
            self.tr("Video files (*.mp4 *.avi);;All files (*)"),
        )
        if path:
            self.output_edit.setText(path)

    def _batch_add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, self.tr("Add video files"), "",
            self.tr("Video files (*.mp4 *.mkv *.avi *.mov *.webm);;All files (*)"),
        )
        for p in paths:
            if p and osp.isfile(p) and self.batch_list.findItems(p, Qt.MatchFlag.MatchExactly) == []:
                self.batch_list.addItem(p)

    def _batch_add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, self.tr("Add folder with videos"))
        if not folder or not osp.isdir(folder):
            return
        exts = (".mp4", ".mkv", ".avi", ".mov", ".webm")
        for name in os.listdir(folder):
            if os.path.splitext(name)[1].lower() in exts:
                p = osp.join(folder, name)
                if self.batch_list.findItems(p, Qt.MatchFlag.MatchExactly) == []:
                    self.batch_list.addItem(p)

    def _batch_remove(self):
        for item in self.batch_list.selectedItems():
            self.batch_list.takeItem(self.batch_list.row(item))

    def _browse_batch_output(self):
        path = QFileDialog.getExistingDirectory(
            self, self.tr("Batch output directory"), self.batch_output_edit.text() or "",
        )
        if path:
            self.batch_output_edit.setText(path)

    def _browse_ffmpeg(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Select ffmpeg executable"),
            self.ffmpeg_path_edit.text() or "",
            self.tr("Executables (ffmpeg.exe);;All files (*)"),
        )
        if path:
            self.ffmpeg_path_edit.setText(path)

    def _browse_subtitle_font(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            self.tr("Select subtitle font"),
            self.subtitle_font_edit.text() or "",
            self.tr("Fonts (*.ttf *.otf);;All files (*)"),
        )
        if path:
            self.subtitle_font_edit.setText(path)

    def _run(self):
        self._save_options()
        batch_count = self.batch_list.count()
        if batch_count > 0:
            out_dir = (self.batch_output_edit.text() or "").strip()
            if not out_dir or not osp.isdir(out_dir):
                self.status_label.setText(self.tr("For batch, select a valid output directory."))
                return
            self._batch_jobs = []
            for i in range(batch_count):
                inp = (self.batch_list.item(i).text() or "").strip()
                if not inp or not osp.isfile(inp):
                    continue
                base = osp.splitext(osp.basename(inp))[0]
                out = osp.join(out_dir, base + "_translated.mp4")
                self._batch_jobs.append((inp, out))
            if not self._batch_jobs:
                self.status_label.setText(self.tr("No valid files in batch list."))
                return
            pcfg.module.video_translator_last_batch_output_dir = out_dir
            save_config()
            self.progress_bar.setValue(0)
            self.run_btn.setEnabled(False)
            self.cancel_btn.setEnabled(True)
            self._batch_index = 0
            self._start_next_batch_job()
            return

        self._batch_jobs = []
        inp = self.input_edit.text().strip()
        if not inp or not osp.isfile(inp):
            self.status_label.setText(self.tr("Please select a valid input video file."))
            return
        out = self.output_edit.text().strip()
        if not out:
            base, _ = osp.splitext(inp)
            out = base + "_translated.mp4"
            self.output_edit.setText(out)
        out_dir = osp.dirname(out)
        if out_dir and not osp.isdir(out_dir):
            self.status_label.setText(self.tr("Output directory does not exist."))
            return

        self._save_paths(inp, out)

        self.progress_bar.setValue(0)
        self.status_label.setText(self.tr("Running pipeline..."))
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)

        soft_subs = self.check_soft_subs.isChecked()
        _idx = self.source_combo.currentIndex()
        src = "asr" if _idx == 1 else "existing_subs" if _idx == 2 else "ocr"
        self.thread = VideoTranslateThread(
            inp,
            out,
            self.sample_spin.value(),
            self.check_detect.isChecked(),
            self.check_ocr.isChecked(),
            self.check_translate.isChecked(),
            False if soft_subs else self.check_inpaint.isChecked(),
            use_scene_detection=self.check_scene.isChecked(),
            scene_threshold=float(self.scene_threshold_spin.value()),
            temporal_smoothing=self.check_temporal.isChecked(),
            temporal_alpha=float(self.temporal_alpha_spin.value()),
            use_ffmpeg=self.check_ffmpeg.isChecked(),
            ffmpeg_path=(self.ffmpeg_path_edit.text() or "").strip(),
            ffmpeg_crf=int(self.ffmpeg_crf_spin.value()),
            video_bitrate_kbps=int(self.video_bitrate_spin.value()),
            skip_detect=self.check_skip_detect.isChecked(),
            export_srt=self.check_export_srt.isChecked(),
            region_preset=self._region_preset_value(),
            soft_subs_only=soft_subs,
            inpaint_only_soft_subs=self.check_inpaint_only_soft.isChecked(),
            mux_srt_into_video=self.check_mux_srt.isChecked(),
            source=src,
            asr_model=(self.asr_model_combo.currentText() or "base").strip(),
            asr_device=(self.asr_device_combo.currentText() or "cuda").strip().lower(),
            asr_language=(self.asr_lang_edit.text() or "").strip(),
            asr_vad_filter=self.check_asr_vad.isChecked(),
            asr_sentence_break=self.check_asr_sentence_break.isChecked(),
            asr_audio_separation=self.check_asr_audio_separation.isChecked(),
            export_ass=self.check_export_ass.isChecked(),
            export_vtt=self.check_export_vtt.isChecked(),
        )
        self.thread.progress.connect(self._on_progress)
        self.thread.finished_ok.connect(self._on_finished_ok)
        self.thread.failed.connect(self._on_failed)
        self.thread.finished.connect(self._on_thread_finished)
        self.thread.frame_preview_updated.connect(self._on_frame_preview_updated)
        self.thread.start()

    def _start_next_batch_job(self):
        if not self._batch_jobs or self._batch_index >= len(self._batch_jobs):
            self.run_btn.setEnabled(True)
            self.cancel_btn.setEnabled(False)
            self.status_label.setText(self.tr("Batch done."))
            return
        inp, out = self._batch_jobs[self._batch_index]
        total = len(self._batch_jobs)
        self.status_label.setText(self.tr("File {} / {}: {}").format(self._batch_index + 1, total, osp.basename(inp)))
        soft_subs = self.check_soft_subs.isChecked()
        _idx = self.source_combo.currentIndex()
        src = "asr" if _idx == 1 else "existing_subs" if _idx == 2 else "ocr"
        self.thread = VideoTranslateThread(
            inp, out,
            self.sample_spin.value(),
            self.check_detect.isChecked(),
            self.check_ocr.isChecked(),
            self.check_translate.isChecked(),
            False if soft_subs else self.check_inpaint.isChecked(),
            use_scene_detection=self.check_scene.isChecked(),
            scene_threshold=float(self.scene_threshold_spin.value()),
            temporal_smoothing=self.check_temporal.isChecked(),
            temporal_alpha=float(self.temporal_alpha_spin.value()),
            use_ffmpeg=self.check_ffmpeg.isChecked(),
            ffmpeg_path=(self.ffmpeg_path_edit.text() or "").strip(),
            ffmpeg_crf=int(self.ffmpeg_crf_spin.value()),
            video_bitrate_kbps=int(self.video_bitrate_spin.value()),
            skip_detect=self.check_skip_detect.isChecked(),
            export_srt=self.check_export_srt.isChecked(),
            region_preset=self._region_preset_value(),
            soft_subs_only=soft_subs,
            inpaint_only_soft_subs=self.check_inpaint_only_soft.isChecked(),
            mux_srt_into_video=self.check_mux_srt.isChecked(),
            source=src,
            asr_model=(self.asr_model_combo.currentText() or "base").strip(),
            asr_device=(self.asr_device_combo.currentText() or "cuda").strip().lower(),
            asr_language=(self.asr_lang_edit.text() or "").strip(),
            asr_vad_filter=self.check_asr_vad.isChecked(),
            asr_sentence_break=self.check_asr_sentence_break.isChecked(),
            asr_audio_separation=self.check_asr_audio_separation.isChecked(),
            export_ass=self.check_export_ass.isChecked(),
            export_vtt=self.check_export_vtt.isChecked(),
        )
        self.thread.progress.connect(self._on_progress)
        self.thread.finished_ok.connect(self._on_finished_ok)
        self.thread.failed.connect(self._on_failed)
        self.thread.finished.connect(self._on_batch_job_finished)
        self.thread.frame_preview_updated.connect(self._on_frame_preview_updated)
        self.thread.start()

    def _on_batch_job_finished(self):
        self._batch_index += 1
        if self._batch_index < len(self._batch_jobs) and not getattr(self.thread, "_cancel", False):
            self._start_next_batch_job()
        else:
            self.run_btn.setEnabled(True)
            self.cancel_btn.setEnabled(False)
            self.status_label.setText(self.tr("Batch done."))

    def _on_frame_preview_updated(self, frame_index: int, frame_bgr, subtitle_lines: list):
        """Update live preview with current frame and detected/translated subtitles (OCR path)."""
        if frame_bgr is not None:
            try:
                import cv2
                import numpy as np
                f = np.asarray(frame_bgr).copy()
                h, w = f.shape[:2]
                rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                bytes_per_line = rgb.strides[0]
                img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                pix = QPixmap.fromImage(img)
                self.preview_frame_label.setPixmap(
                    pix.scaled(self.preview_frame_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                )
            except Exception as e:
                LOGGER.debug("Preview frame update: %s", e)
        lines = []
        for src, trans in (subtitle_lines or []):
            if (src or "").strip() or (trans or "").strip():
                lines.append("%s → %s" % ((src or "").strip() or "-", (trans or "").strip() or "-"))
        text = "\n".join(lines) if lines else self.tr("No text on this frame.")
        self.preview_text_edit.setPlainText(text)

    def _on_progress(self, current: int, total: int):
        if total > 0:
            self.progress_bar.setValue(int(100 * current / total))
            self.progress_bar.setRange(0, 100)
        else:
            self.progress_bar.setRange(0, 0)
        if self._batch_jobs and self._batch_index < len(self._batch_jobs):
            self.status_label.setText(self.tr("File {} / {}: frame {} / {}").format(
                self._batch_index + 1, len(self._batch_jobs), current, total if total > 0 else "?"))
        else:
            self.status_label.setText(self.tr("Frame {} / {}").format(current, total if total > 0 else "?"))

    def _on_finished_ok(self, path: str):
        if not (self._batch_jobs and self._batch_index <= len(self._batch_jobs)):
            self.status_label.setText(self.tr("Done. Output: {}").format(path))

    def _on_failed(self, msg: str):
        self.status_label.setText(self.tr("Error: {}").format(msg))

    def _on_thread_finished(self):
        if not self._batch_jobs:
            self.run_btn.setEnabled(True)
            self.cancel_btn.setEnabled(False)

    def closeEvent(self, event: QCloseEvent):
        """Close (X or Close button) does not stop the pipeline; only Cancel does."""
        if self.thread and self.thread.isRunning():
            event.accept()
            self.accept()
            return
        super().closeEvent(event)

    def _cancel_run(self):
        """Cancel button: stop the running pipeline."""
        if self.thread and self.thread.isRunning():
            self.thread.cancel()
