"""
Video translator dialog: UI and worker thread to translate hardcoded subtitles in videos
using the same detect / OCR / translate / inpaint pipeline as the main app.
"""
from __future__ import annotations

import os
import os.path as osp
import tempfile
import hashlib
import re
import threading
import copy
import concurrent.futures
import queue as queue_module
from typing import Optional

from qtpy.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog,
    QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox, QProgressBar, QGroupBox, QGridLayout,
    QComboBox, QWidget, QPlainTextEdit, QListWidget, QListWidgetItem, QAbstractItemView,
    QScrollArea, QFrame, QSizePolicy,
)
from qtpy.QtCore import Qt, QThread, Signal, QEvent, QTimer
from qtpy.QtGui import QIcon, QImage, QPixmap, QCloseEvent, QKeySequence, QShowEvent, QResizeEvent
from qtpy.QtWidgets import QShortcut

from utils.config import pcfg, save_config
from utils.logger import logger as LOGGER


def _set_thread_priority_above_normal() -> None:
    """Raise current thread priority so the pipeline is not starved by I/O threads. Windows only."""
    try:
        if hasattr(os, "name") and os.name == "nt":
            import ctypes
            # THREAD_PRIORITY_ABOVE_NORMAL = 1
            ctypes.windll.kernel32.SetThreadPriority(ctypes.windll.kernel32.GetCurrentThread(), 1)
    except Exception:
        pass


def _set_thread_priority_below_normal() -> None:
    """Lower current thread priority so I/O threads don't starve the pipeline. Windows only."""
    try:
        if hasattr(os, "name") and os.name == "nt":
            import ctypes
            # THREAD_PRIORITY_BELOW_NORMAL = -1
            ctypes.windll.kernel32.SetThreadPriority(ctypes.windll.kernel32.GetCurrentThread(), -1)
    except Exception:
        pass


_REGION_PLACEHOLDER_RE = re.compile(
    r"^\s*(?:\[\s*)?(?:reg(?:i|l|1)on)\s*[:#-]?\s*\d+\s*(?:\]\s*)?$",
    re.IGNORECASE,
)


def _is_region_placeholder_text(text: str) -> bool:
    """Some translators/models occasionally output placeholders like '[region 1]' / 'Region 1'. Treat those as empty."""
    try:
        return bool(_REGION_PLACEHOLDER_RE.match((text or "").strip()))
    except Exception:
        return False


_SUBTITLE_JUNK_RE = re.compile(
    r"^\s*(?:\d+\.\s*)?(?:line\s*\d+\s+content|line\s+content|line\s*[a-z0-9]+|text\s*[a-z0-9]*|subtitle\s*[a-z0-9]*|hello(?:\s+world)?)\s*$",
    re.IGNORECASE,
)


def _is_invalid_subtitle_text(text: str) -> bool:
    """Filter placeholder/meta/junk lines so they are never added to timed cues."""
    t = (text or "").strip()
    if not t:
        return True
    if _is_region_placeholder_text(t):
        return True
    tl = t.lower()
    if tl in (
        "...",
        "corrected_text",
        "corrected text",
        "no source text or current translation provided for review.",
    ):
        return True
    if _SUBTITLE_JUNK_RE.match(tl):
        return True
    return False


_SENTENCE_END_RE = re.compile(r"[.!?。！？…]+(?:[\"'”’）\]\s]*)$")


def _merge_segments_until_sentence_end(
    segments: list[tuple[float, float, str]],
    max_merge_seconds: float = 8.0,
) -> list[tuple[float, float, str]]:
    """
    Rule-based cue merge: accumulate short adjacent cues until sentence-ending punctuation.
    Helps avoid translating half-sentences line-by-line (Issue-inspired, non-LLM sentence batching).
    """
    if not segments:
        return segments
    out: list[tuple[float, float, str]] = []
    cur_start: Optional[float] = None
    cur_end: Optional[float] = None
    parts: list[str] = []
    max_merge_seconds = max(0.5, float(max_merge_seconds or 8.0))

    def _flush():
        nonlocal cur_start, cur_end, parts
        if cur_start is None or cur_end is None:
            cur_start, cur_end, parts = None, None, []
            return
        text = " ".join(p.strip() for p in parts if (p or "").strip()).strip()
        if text:
            out.append((float(cur_start), float(cur_end), text))
        cur_start, cur_end, parts = None, None, []

    for s, e, t in segments:
        txt = (t or "").strip()
        if not txt:
            continue
        if cur_start is None:
            cur_start, cur_end, parts = float(s), float(e), [txt]
        else:
            cur_end = float(e)
            parts.append(txt)
        merged_text = " ".join(parts).strip()
        reached_end = bool(_SENTENCE_END_RE.search(merged_text))
        duration = float(cur_end) - float(cur_start)
        if reached_end or duration >= max_merge_seconds:
            _flush()
    _flush()
    return out


def _align_blk_list_to_segments(
    segments: list,
    blk_list: list,
    TextBlock_cls: type,
) -> None:
    """
    After translate_textblk_lst, enforce len(blk_list) == len(segments) so timings stay aligned
    (pad with source-only blocks or trim extras; log mismatches).
    """
    n = len(segments)
    if len(blk_list) == n:
        return
    if len(blk_list) < n:
        LOGGER.warning(
            "Subtitle blocks (%d) fewer than cues (%d); padding with source-only blocks.",
            len(blk_list),
            n,
        )
        while len(blk_list) < n:
            i = len(blk_list)
            _s, _e, text = segments[i]
            blk = TextBlock_cls()
            blk.text = [text or ""]
            blk.lines = []
            blk.xyxy = [0, 0, 1, 1]
            blk_list.append(blk)
    else:
        LOGGER.warning(
            "Subtitle blocks (%d) more than cues (%d); trimming extras to prevent timeline drift.",
            len(blk_list),
            n,
        )
        del blk_list[n:]


def _apply_continuation_comma_fix(revised_prev, revised_new, previous_entries, video_previous_subtitles, srt_entries=None):
    """
    When the flow fixer returns unchanged text, apply a rule-based fix: if the last
    previous line has no trailing comma and the first new line is a continuation
    (e.g. 'Taken away from Earth...'), add a comma to the previous line and optionally
    'I was ' to the new line so the pair reads as one sentence. Mutates in place.
    When mutating video_previous_subtitles, pass srt_entries to update the last SRT segment text.
    Returns (n_prev_fixed, n_new_fixed) for logging.
    """
    n_prev_fixed, n_new_fixed = 0, 0
    if not revised_new:
        return n_prev_fixed, n_new_fixed
    first_new = (revised_new[0] or "").strip()
    if not first_new:
        return n_prev_fixed, n_new_fixed
    first_lower = first_new.lower()
    continuation_starts = (
        first_new.startswith("Taken ") or first_new.startswith("Taken away")
        or first_new.startswith("Reached ") or first_new.startswith("Because of ")
        or first_new.startswith("Thus ") or first_new.startswith("From this ")
        or first_new.startswith("From the ") or first_new.startswith("Called ")
        or first_new.startswith("Left ") or first_new.startswith("Bringing ")
        or first_new.startswith("Having ")
        or (first_new.startswith("By ") and len(first_new) > 10)
        or first_lower.startswith("and ") or first_lower.startswith("but ")
    )
    if not continuation_starts:
        return n_prev_fixed, n_new_fixed
    # Get last previous line
    last_prev_line = None
    last_prev_source = None  # (revised_prev, idx_ent, idx_trans) or ("subs", idx_ent, idx_trans)
    if revised_prev and len(revised_prev) > 0:
        ent = revised_prev[-1]
        trans = ent.get("translations") or []
        if trans:
            last_prev_line = (trans[-1] or "").strip()
            last_prev_source = ("revised_prev", len(revised_prev) - 1, len(trans) - 1)
    if last_prev_line is None and previous_entries and len(previous_entries) > 0:
        ent = previous_entries[-1]
        trans = ent.get("translations") or []
        if trans:
            last_prev_line = (trans[-1] or "").strip()
            last_prev_source = ("previous_entries", len(previous_entries) - 1, len(trans) - 1)
    if last_prev_line is None and video_previous_subtitles and len(video_previous_subtitles) >= 2:
        ent = video_previous_subtitles[-2]
        trans = ent.get("translations") or []
        if trans:
            last_prev_line = (trans[-1] or "").strip()
            last_prev_source = ("video_subs", -2, len(trans) - 1)
    if not last_prev_line or last_prev_source is None:
        return n_prev_fixed, n_new_fixed
    if last_prev_line.rstrip().endswith(",") or last_prev_line.rstrip().endswith(";"):
        return n_prev_fixed, n_new_fixed
    if last_prev_line.rstrip().endswith(".") or last_prev_line.rstrip().endswith("?") or last_prev_line.rstrip().endswith("!"):
        return n_prev_fixed, n_new_fixed
    # Add comma to last previous line
    fixed_prev = last_prev_line.rstrip()
    if not fixed_prev.endswith(","):
        fixed_prev = fixed_prev + ","
        kind, idx_ent, idx_trans = last_prev_source
        if kind == "revised_prev" and revised_prev is not None:
            revised_prev[idx_ent]["translations"][idx_trans] = fixed_prev
            n_prev_fixed = 1
        elif kind == "video_subs" and video_previous_subtitles and len(video_previous_subtitles) >= 2:
            ent = video_previous_subtitles[-2]
            trans = ent.get("translations") or []
            if trans:
                trans[-1] = fixed_prev
                if "translations" not in ent:
                    ent["translations"] = trans
                n_prev_fixed = 1
                if srt_entries and len(srt_entries) >= 1:
                    srt_entries[-1] = (srt_entries[-1][0], srt_entries[-1][1], " ".join(trans).strip())
    # Optionally add "I was " to first new line when it's passive and context is first person
    recent_text = ""
    if previous_entries:
        for e in previous_entries[-5:]:
            recent_text += " " + " ".join((e.get("translations") or []))
    if (
        first_new.startswith("Taken ") or first_new.startswith("Taken away")
    ) and (" I " in recent_text or " I'm " in recent_text or " my " in recent_text) and not first_new.lower().startswith("i was "):
        revised_new[0] = "I was " + first_new
        n_new_fixed = 1
    return n_prev_fixed, n_new_fixed


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


def _get_source_fps_ffprobe(video_path: str, ffmpeg_exe: str = "") -> float:
    """Get source video FPS via ffprobe (avg_frame_rate). Returns 0.0 if not detectable.
    Parses rationals like 24/1, 30000/1001, 24000/1001."""
    import subprocess
    exe = "ffprobe"
    if (ffmpeg_exe or "").strip():
        base = (ffmpeg_exe or "").strip()
        d = osp.dirname(base)
        name = "ffprobe.exe" if base.lower().endswith(".exe") else "ffprobe"
        exe = osp.join(d, name) if d else name
    try:
        out = subprocess.run(
            [
                exe, "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=avg_frame_rate,r_frame_rate",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode != 0 or not out.stdout:
            return 0.0
        lines = [s.strip() for s in out.stdout.strip().splitlines() if s.strip()]
        for raw in (lines[0:1] if lines else []):
            if "/" in raw:
                num, den = raw.split("/", 1)
                try:
                    n, d = float(num.strip()), float(den.strip())
                    if d > 0 and 1.0 <= (n / d) <= 200.0:
                        return round(n / d, 4)
                except (ValueError, ZeroDivisionError):
                    pass
            try:
                v = float(raw)
                if 1.0 <= v <= 200.0:
                    return round(v, 4)
            except ValueError:
                pass
    except Exception:
        pass
    return 0.0


def _normalize_fps(cap, reported_fps: float, total_frames: int, video_path: str = "", ffmpeg_exe: str = "") -> float:
    """Return FPS for encoding that matches the source. Preserves 24, 23.976, 25, 29.97, 30, etc.
    Prefers ffprobe when video_path given; else OpenCV; else frame_count/duration. Never forces 30 for 24fps sources."""
    import cv2
    fps = reported_fps or 0.0
    # Prefer ffprobe so output framerate matches source exactly (avoids wrong duration/speed)
    if video_path:
        probe = _get_source_fps_ffprobe(video_path, ffmpeg_exe)
        if probe > 0:
            return probe
    # OpenCV reported FPS: use if sane; only treat 0 or (0,6) as bogus
    if (fps <= 0 or fps > 200 or (0 < fps < 6.0)) and cap and total_frames > 0:
        try:
            cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1.0)
            end_ms = cap.get(cv2.CAP_PROP_POS_MSEC) or 0
            cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 0.0)
            if end_ms > 0:
                from_fps = total_frames / (end_ms / 1000.0)
                if 1.0 <= from_fps <= 200.0:
                    return round(from_fps, 4)
        except Exception:
            pass
    if fps <= 0 or (0 < fps < 6.0):
        return 24.0  # safe default; do not force 30
    if fps > 200:
        return 120.0
    return round(fps, 4)


def _probe_ffmpeg_hw_encoders(ffmpeg_exe: str) -> dict:
    """Run ffmpeg -encoders and return which GPU encoders are available. Returns {'nvenc': bool, 'qsv': bool}."""
    import subprocess
    exe = (ffmpeg_exe or "ffmpeg").strip()
    if not exe:
        return {"nvenc": False, "qsv": False}
    out = subprocess.run(
        [exe, "-encoders"],
        capture_output=True,
        text=True,
        timeout=15,
        creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
    )
    text = (out.stdout or "") + (out.stderr or "")
    nvenc = "h264_nvenc" in text and "H.264" in text
    qsv = "h264_qsv" in text and "H.264" in text
    return {"nvenc": nvenc, "qsv": qsv}


def _build_ffmpeg_encode_cmd(
    ffmpeg_exe: str,
    w: int,
    h: int,
    fps: float,
    output_path: str,
    use_bitrate: bool,
    effective_kbps: int,
    crf: int,
    preset: str,
    hw_encoder: str,
) -> list:
    """Build full FFmpeg command list for raw BGR24 pipe input -> encoded output.
    hw_encoder: 'none' | 'nvenc' | 'qsv' | 'auto'. Preset used for libx264 only; GPU uses internal presets/CQ."""
    exe = (ffmpeg_exe or "ffmpeg").strip() or "ffmpeg"
    base = [
        exe, "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", "%dx%d" % (w, h), "-framerate", str(fps), "-i", "pipe:0",
    ]
    chosen = (hw_encoder or "none").strip().lower()
    if chosen == "auto":
        encoders = _probe_ffmpeg_hw_encoders(ffmpeg_exe)
        chosen = "nvenc" if encoders.get("nvenc") else ("qsv" if encoders.get("qsv") else "none")
    if chosen == "nvenc":
        encoders = _probe_ffmpeg_hw_encoders(ffmpeg_exe)
        if not encoders.get("nvenc"):
            chosen = "none"
    elif chosen == "qsv":
        encoders = _probe_ffmpeg_hw_encoders(ffmpeg_exe)
        if not encoders.get("qsv"):
            chosen = "none"

    if chosen == "nvenc":
        # NVENC: BGR24 -> yuv420p (CPU) -> h264_nvenc. -cq 1-51 (like CRF) or -b:v
        if use_bitrate and effective_kbps > 0:
            ext = ["-vf", "format=yuv420p", "-c:v", "h264_nvenc", "-b:v", "%dk" % effective_kbps, "-r", str(fps), "-pix_fmt", "yuv420p"]
        else:
            cq = max(1, min(51, crf))
            ext = ["-vf", "format=yuv420p", "-c:v", "h264_nvenc", "-cq", str(cq), "-r", str(fps), "-pix_fmt", "yuv420p"]
        base.extend(ext)
    elif chosen == "qsv":
        # QSV: BGR24 -> nv12 (CPU) -> h264_qsv. -global_quality 1-51 (like CRF)
        ext = ["-vf", "format=nv12", "-c:v", "h264_qsv", "-global_quality", str(max(1, min(51, crf))), "-r", str(fps)]
        if use_bitrate and effective_kbps > 0:
            ext = ["-vf", "format=nv12", "-c:v", "h264_qsv", "-b:v", "%dk" % effective_kbps, "-r", str(fps)]
        base.extend(ext)
    else:
        # libx264
        _presets = ("ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow", "placebo")
        effective_preset = (preset or "medium").strip().lower()
        if effective_preset not in _presets:
            effective_preset = "medium"
        if use_bitrate and effective_kbps > 0:
            base.extend([
                "-c:v", "libx264", "-preset", effective_preset,
                "-b:v", "%dk" % effective_kbps, "-minrate", "%dk" % effective_kbps,
                "-maxrate", "%dk" % effective_kbps, "-bufsize", "%dk" % (effective_kbps * 2),
                "-x264-params", "nal-hrd=cbr", "-r", str(fps), "-pix_fmt", "yuv420p",
            ])
        else:
            base.extend(["-c:v", "libx264", "-preset", effective_preset, "-crf", str(crf), "-r", str(fps), "-pix_fmt", "yuv420p"])
    if output_path.lower().endswith((".mp4", ".mov")):
        base.extend(["-movflags", "+faststart"])
    base.append(output_path)
    return base


def _scene_change(prev_gray, curr_gray, threshold: float, bottom_frac: float = 1.0) -> bool:
    """True if histogram diff exceeds threshold (new scene).
    Cost is kept low by downscaling and by computing histogram only on the subtitle band ROI.
    """
    if prev_gray is None or curr_gray is None:
        return True
    import cv2
    try:
        bottom_frac = float(bottom_frac)
    except Exception:
        bottom_frac = 1.0
    if bottom_frac <= 0:
        bottom_frac = 1.0
    bottom_frac = min(1.0, max(0.0, bottom_frac))
    # Lighter scene detection: downscale to max 320 px before histogram when frame is large
    max_side = 320
    h, w = prev_gray.shape[:2]
    prev_full, curr_full = prev_gray, curr_gray
    y_start = max(0, int(h * (1.0 - bottom_frac)))
    if y_start > 0:
        prev_gray = prev_gray[y_start:h, :]
        curr_gray = curr_gray[y_start:h, :]
        # Safety: if crop is empty due to bad params, fall back to full frame
        if prev_gray is None or curr_gray is None or prev_gray.size == 0 or curr_gray.size == 0:
            prev_gray = prev_full
            curr_gray = curr_full
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        prev_gray = cv2.resize(prev_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        curr_gray = cv2.resize(curr_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
    prev_hist = cv2.calcHist([prev_gray], [0], None, [64], [0, 256])
    curr_hist = cv2.calcHist([curr_gray], [0], None, [64], [0, 256])
    cv2.normalize(prev_hist, prev_hist, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(curr_hist, curr_hist, 0, 1, cv2.NORM_MINMAX)
    diff = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_BHATTACHARYYA)
    return diff * 100.0 > threshold


def _subtitle_region_small_gray(frame: "np.ndarray", bottom_frac: float) -> "np.ndarray | None":
    """Return a small grayscale thumbnail of the subtitle band ROI for cheap comparisons."""
    if frame is None or frame.size == 0:
        return None
    import cv2
    try:
        bottom_frac = float(bottom_frac)
    except Exception:
        bottom_frac = 1.0
    if bottom_frac <= 0:
        bottom_frac = 1.0
    bottom_frac = min(1.0, max(0.0, bottom_frac))
    h, w = frame.shape[:2]
    y_start = max(0, int(h * (1.0 - bottom_frac)))
    crop = frame[y_start:h, :]
    if crop is None or crop.size == 0:
        return None
    # Downscale to small size for fast hashing/diff (e.g. 64 px wide)
    small_w = 64
    small_h = max(4, int(crop.shape[0] * small_w / max(1, crop.shape[1])))
    small = cv2.resize(crop, (small_w, small_h), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
    return gray


def _subtitle_region_content_hash(frame: "np.ndarray", bottom_frac: float) -> bytes:
    """Cheap hash of the subtitle band thumbnail (used by two-stage keyframes)."""
    import hashlib
    gray = _subtitle_region_small_gray(frame, bottom_frac)
    if gray is None:
        return b""
    return hashlib.md5(gray.tobytes()).digest()


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


def _composite_cached_subs_on_frame(
    frame: "np.ndarray",
    cached_result: "np.ndarray",
    cached_source_frame: "np.ndarray",
    blk_list: list,
    h: int,
    w: int,
) -> "np.ndarray":
    """Paste cached subtitle regions (inpainted + drawn text) onto the current frame so motion stays at full fps."""
    if (
        frame is None
        or cached_result is None
        or not blk_list
        or frame.shape[:2] != cached_result.shape[:2]
    ):
        return frame.copy() if frame is not None else None
    out = frame.copy()
    for blk in blk_list:
        xyxy = getattr(blk, "xyxy", None)
        if not xyxy or len(xyxy) < 4:
            continue
        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
        x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
        y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            continue
        out[y1:y2, x1:x2] = cached_result[y1:y2, x1:x2]
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
            # Ensure every cue has at least 1-frame duration; some players glitch on 0/negative-length cues.
            try:
                if end is None or start is None:
                    continue
                end = max(int(end), int(start) + 1)
                start = int(start)
            except Exception:
                pass
            text = (text or "").strip().replace("\n", " ")
            if not text:
                text = ""
            f.write("%d\n%s --> %s\n%s\n\n" % (i, _ts_srt(start, fps), _ts_srt(end, fps), text))


def _write_vtt(path: str, entries: list, fps: float) -> None:
    """Write WebVTT file. entries: list of (start_frame, end_frame, text)."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for i, (start, end, text) in enumerate(entries, 1):
            # Ensure every cue has at least 1-frame duration; prevents 0-length cues.
            try:
                if end is None or start is None:
                    continue
                end = max(int(end), int(start) + 1)
                start = int(start)
            except Exception:
                pass
            text = (text or "").strip().replace("\n", " ")
            if not text:
                text = ""
            f.write("%d\n%s --> %s\n%s\n\n" % (i, _ts_vtt(start, fps), _ts_vtt(end, fps), text))


def _write_ass(path: str, entries: list, fps: float, width: int | None = None, height: int | None = None) -> None:
    """Write ASS file. entries: list of (start_frame, end_frame, text).
    If width/height given, write PlayResX/PlayResY in [Script Info] for correct scaling (matches video resolution)."""
    with open(path, "w", encoding="utf-8") as f:
        script_info = "[Script Info]\nTitle: BallonsTranslator\n"
        if width is not None and height is not None and width > 0 and height > 0:
            script_info += "PlayResX: %d\nPlayResY: %d\n" % (width, height)
        script_info += "\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, ShadowColour, BackColour, Bold, Italic, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\nStyle: Default,Arial,20,&H00FFFFFF,&H00000000,&H00000000,&H80000000,0,0,1,2,0,2,10,10,10,1\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        f.write(script_info)
        for start, end, text in entries:
            # Ensure at least 1-frame duration.
            try:
                if end is None or start is None:
                    continue
                end = max(int(end), int(start) + 1)
                start = int(start)
            except Exception:
                pass
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
        prefetch_frames: int = 0,
        two_stage_keyframes: bool = False,
        background_writer: bool = False,
        use_two_pass_ocr_burn_in: bool = True,
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
        self.prefetch_frames = max(0, min(4, int(prefetch_frames)))
        self.two_stage_keyframes = bool(two_stage_keyframes)
        self.background_writer = bool(background_writer)
        self.use_two_pass_ocr_burn_in = bool(use_two_pass_ocr_burn_in)
        self._progress_phase = ""
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def _run_asr_pipeline(self):
        """Source = Audio (ASR): strict stages — transcribe → subtitle NLP (correction / translate) → video synthesis.

        Stages: (1) audio extract + optional vocal separation + ASR segments;
        (2) optional LLM ASR correction, optional sentence break, batched translation (see video_translator_nlp_* for chunk/workers);
        (3) decode video frames, timed burn-in or soft subs, encode. No translation interleaved with per-frame decode.
        """
        import cv2
        import subprocess
        import numpy as np
        from modules.video_translator import (
            _draw_timed_subs_on_image,
            subtitle_black_box_draw_kwargs_from_cfg,
            configure_translator_video_nlp_parallel,
            clear_translator_video_nlp_parallel,
        )
        from modules.audio_transcribe import (
            HAS_FASTER_WHISPER,
            extract_audio_from_video,
            transcribe_audio,
        )
        from utils.textblock import TextBlock
        from utils.textblock import examine_textblk, remove_contained_boxes, deduplicate_primary_boxes
        from modules.inpaint.base import build_mask_with_resolved_overlaps

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
                chunk_s = float(getattr(cfg, "video_translator_asr_chunk_seconds", 0) or 0)
                threshold_s = float(
                    getattr(cfg, "video_translator_asr_long_audio_threshold_seconds", 0) or 0
                )
                resume_ck = bool(getattr(cfg, "video_translator_asr_checkpoint_resume", True))
                checkpoint_path = None
                if chunk_s > 0 and threshold_s > 0 and resume_ck:
                    vid_key = (self.input_path or audio_path).encode("utf-8", errors="ignore")
                    h = hashlib.md5(
                        vid_key
                        + f"|{chunk_s}|{threshold_s}|{self.asr_model}|{self.asr_device}|{self.asr_language or ''}|{int(self.asr_audio_separation)}".encode(
                            "utf-8", errors="ignore"
                        )
                    ).hexdigest()[:24]
                    checkpoint_path = os.path.join(
                        tempfile.gettempdir(), f"bt_asr_ckpt_{h}.json"
                    )
                segments = transcribe_audio(
                    audio_path,
                    model_size=self.asr_model,
                    device=self.asr_device,
                    language=self.asr_language or None,
                    vad_filter=self.asr_vad_filter,
                    chunk_seconds=chunk_s,
                    long_audio_threshold_seconds=threshold_s,
                    ffmpeg_path=ffmpeg_exe,
                    checkpoint_path=checkpoint_path if resume_ck else None,
                    resume_checkpoint=resume_ck,
                    source_video_path=(self.input_path or None),
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
                        use_flow_fixer_for_corrections = bool(getattr(cfg, "video_translator_use_flow_fixer_for_corrections", False))
                        if use_flow_fixer_for_corrections and getattr(cfg, "video_translator_flow_fixer_enabled", False):
                            from modules.flow_fixer import get_flow_fixer
                            from modules.flow_fixer.corrections import correct_asr_via_fixer
                            fixer_name = (getattr(cfg, "video_translator_flow_fixer", None) or "none").strip().lower()
                            flow_fixer_asr = None
                            if fixer_name == "openrouter":
                                flow_fixer_asr = get_flow_fixer(
                                    "openrouter",
                                    api_key=(getattr(cfg, "video_translator_flow_fixer_openrouter_apikey", None) or "").strip(),
                                    model=(getattr(cfg, "video_translator_flow_fixer_openrouter_model", None) or "google/gemma-3n-e2b-it:free").strip(),
                                    max_tokens=int(getattr(cfg, "video_translator_flow_fixer_max_tokens", 256)),
                                    timeout=float(getattr(cfg, "video_translator_flow_fixer_timeout", 30.0)),
                                    enable_reasoning=bool(getattr(cfg, "video_translator_flow_fixer_enable_reasoning", False)),
                                    reasoning_effort=(getattr(cfg, "video_translator_flow_fixer_reasoning_effort", None) or "medium"),
                                )
                            elif fixer_name == "openai":
                                flow_fixer_asr = get_flow_fixer(
                                    "openai",
                                    api_key=(getattr(cfg, "video_translator_flow_fixer_openai_apikey", None) or "").strip(),
                                    model=(getattr(cfg, "video_translator_flow_fixer_openai_model", None) or "gpt-4o-mini").strip(),
                                    max_tokens=int(getattr(cfg, "video_translator_flow_fixer_max_tokens", 256)),
                                    timeout=float(getattr(cfg, "video_translator_flow_fixer_timeout", 30.0)),
                                    enable_reasoning=bool(getattr(cfg, "video_translator_flow_fixer_enable_reasoning", False)),
                                    reasoning_effort=(getattr(cfg, "video_translator_flow_fixer_reasoning_effort", None) or "medium"),
                                )
                            elif fixer_name == "local_server":
                                local_model = (getattr(cfg, "video_translator_flow_fixer_model", None) or "").strip()
                                if local_model:
                                    flow_fixer_asr = get_flow_fixer(
                                        fixer_name,
                                        server_url=(getattr(cfg, "video_translator_flow_fixer_server_url", None) or "http://localhost:1234/v1").strip(),
                                        model=local_model,
                                        max_tokens=int(getattr(cfg, "video_translator_flow_fixer_max_tokens", 256)),
                                        timeout=float(getattr(cfg, "video_translator_flow_fixer_timeout", 30.0)),
                                        enable_reasoning=bool(getattr(cfg, "video_translator_flow_fixer_enable_reasoning", False)),
                                        reasoning_effort=(getattr(cfg, "video_translator_flow_fixer_reasoning_effort", None) or "medium"),
                                    )
                            if flow_fixer_asr and getattr(flow_fixer_asr, "request_completion", None):
                                glossary = (getattr(cfg, "video_translator_glossary", None) or "").strip()
                                corrected = correct_asr_via_fixer(flow_fixer_asr, texts, glossary=glossary)
                                if corrected and len(corrected) == len(segments):
                                    segments = [(s, e, c) for (s, e, _), c in zip(segments, corrected)]
                            else:
                                corrected = translator.correct_asr_texts(texts)
                                if corrected and len(corrected) == len(segments):
                                    segments = [(s, e, c) for (s, e, _), c in zip(segments, corrected)]
                        else:
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
            if bool(getattr(cfg, "video_translator_sentence_merge_by_punctuation", True)):
                try:
                    segments = _merge_segments_until_sentence_end(
                        segments,
                        max_merge_seconds=float(getattr(cfg, "video_translator_sentence_merge_max_seconds", 8.0)),
                    )
                except Exception as e:
                    LOGGER.debug("Rule-based sentence merge failed (using original): %s", e)
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
                configure_translator_video_nlp_parallel(translator, cfg)
                try:
                    setattr(translator, "_current_page_key", "video_asr")
                    setattr(translator, "_current_page_image", None)
                    translator.translate_textblk_lst(blk_list)
                except Exception as e:
                    LOGGER.warning("ASR translate batch failed: %s", e)
                finally:
                    clear_translator_video_nlp_parallel(translator)
                _align_blk_list_to_segments(segments, blk_list, TextBlock)
                for (s, e, _), blk in zip(segments, blk_list):
                    trans = (getattr(blk, "translation", None) or blk.get_text() or "").strip()
                    if _is_region_placeholder_text(trans):
                        trans = ""
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
                    effective_preset = (getattr(cfg, "video_translator_ffmpeg_preset", None) or "medium").strip()
                    hw_encoder = (getattr(cfg, "video_translator_ffmpeg_hw_encoder", None) or "none").strip().lower()
                    ffmpeg_cmd = _build_ffmpeg_encode_cmd(
                        ffmpeg_exe, w, h, fps, self.output_path,
                        use_bitrate=(effective_kbps > 0), effective_kbps=effective_kbps,
                        crf=self.ffmpeg_crf, preset=effective_preset, hw_encoder=hw_encoder,
                    )
                    ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
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
            asr_guided_detect_inpaint = bool(getattr(cfg, "video_translator_asr_guided_detect_inpaint", False))
            asr_guided_midpoint_refresh = bool(getattr(cfg, "video_translator_asr_guided_midpoint_refresh", True))
            guided_detector = None
            guided_inpainter = None
            guided_frac = 0.0
            guided_active_idx = -1
            guided_mid_refreshed = False
            guided_blks = []
            if asr_guided_detect_inpaint and self.enable_inpaint and self.enable_detect and (not self.soft_subs_only):
                try:
                    from modules.video_translator import _region_fraction_from_preset
                    guided_detector = _get_video_module("textdetector", cfg.textdetector)
                    guided_inpainter = _get_video_module("inpainter", cfg.inpainter)
                    if guided_detector is not None and hasattr(guided_detector, "load_model"):
                        guided_detector.load_model()
                    if guided_inpainter is not None and hasattr(guided_inpainter, "load_model"):
                        guided_inpainter.load_model()
                    guided_frac = float(_region_fraction_from_preset(cfg) or 0.0)
                except Exception as e:
                    LOGGER.warning("ASR-guided detect/inpaint init failed; using subtitle-only ASR render: %s", e)
                    guided_detector = None
                    guided_inpainter = None

            def _detect_guided_blocks(frame_bgr: np.ndarray):
                if guided_detector is None:
                    return []
                try:
                    h2, w2 = frame_bgr.shape[:2]
                    _m, bl = guided_detector.detect(frame_bgr, None)
                    bl = bl or []
                    if bl:
                        for b in bl:
                            if getattr(b, "lines", None) and len(b.lines) > 0:
                                examine_textblk(b, w2, h2, sort=True)
                        bl = remove_contained_boxes(bl)
                        bl = deduplicate_primary_boxes(bl, iou_threshold=0.5)
                    if guided_frac > 0.0 and h2 > 0:
                        y_min_region = h2 * (1.0 - guided_frac)
                        kept = []
                        for b in bl:
                            xyxy = getattr(b, "xyxy", None)
                            if not xyxy or len(xyxy) < 4:
                                continue
                            y1, y2 = float(xyxy[1]), float(xyxy[3])
                            cy = (y1 + y2) * 0.5
                            if cy >= y_min_region and y1 < h2:
                                kept.append(b)
                        bl = kept
                    return bl
                except Exception:
                    return []
            n = 0
            prev_active = []  # list of active subtitle texts on previous frame
            step = 25 if total <= 10000 else (50 if total <= 100000 else 200)
            while True:
                if self._cancel:
                    break
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                time_sec = n / fps if fps > 0 else 0
                if guided_detector is not None and guided_inpainter is not None:
                    active_idx = -1
                    active_seg = None
                    for i_seg, (s, e, t) in enumerate(segments_with_trans):
                        txt = (t or "").strip()
                        if txt and (not _is_invalid_subtitle_text(txt)) and float(s) <= float(time_sec) < float(e):
                            active_idx = i_seg
                            active_seg = (float(s), float(e), txt)
                            break
                    if active_idx >= 0 and active_seg is not None:
                        if active_idx != guided_active_idx:
                            guided_active_idx = active_idx
                            guided_mid_refreshed = False
                            guided_blks = _detect_guided_blocks(frame)
                        elif asr_guided_midpoint_refresh and (not guided_mid_refreshed):
                            s0, e0, _t0 = active_seg
                            if float(time_sec) >= ((s0 + e0) * 0.5):
                                guided_blks = _detect_guided_blocks(frame)
                                guided_mid_refreshed = True
                        if guided_blks:
                            try:
                                hh, ww = frame.shape[:2]
                                msk = build_mask_with_resolved_overlaps(guided_blks, ww, hh, text_blocks_for_nudge=None)
                                if msk is not None and msk.size > 0 and np.count_nonzero(msk > 127) > 0:
                                    frame = guided_inpainter.inpaint(frame, msk, guided_blks)
                            except Exception:
                                pass
                    else:
                        guided_active_idx = -1
                        guided_mid_refreshed = False
                        guided_blks = []
                if not self.soft_subs_only:
                    # Log start/end transitions for hardcoded (burn-in) subtitles (only when active set changes).
                    try:
                        active_now = []
                        for s, e, t in segments_with_trans:
                            txt = (t or "").strip()
                            if txt and (not _is_invalid_subtitle_text(txt)) and float(s) <= float(time_sec) < float(e):
                                active_now.append(txt)
                        if active_now != prev_active:
                            ended = [t for t in prev_active if t not in active_now]
                            started = [t for t in active_now if t not in prev_active]
                            for t in ended:
                                LOGGER.info("ASR subtitle ended @ %.3fs: %r", float(time_sec), t)
                            for t in started:
                                # Find the segment time window for this text (best-effort)
                                seg_s, seg_e = None, None
                                for s, e, txt in segments_with_trans:
                                    if (txt or "").strip() == t:
                                        seg_s, seg_e = float(s), float(e)
                                        break
                                if seg_s is not None and seg_e is not None:
                                    LOGGER.info("ASR subtitle started @ %.3fs (%.3f-%.3fs): %r", float(time_sec), seg_s, seg_e, t)
                                else:
                                    LOGGER.info("ASR subtitle started @ %.3fs: %r", float(time_sec), t)
                            prev_active = active_now
                    except Exception:
                        pass
                    # Stack back-to-back/overlapping subs and give very short cues a 1-frame overlap at boundaries
                    # to avoid flicker/overwrite when end/start are extremely close.
                    dt = (1.0 / fps) if fps > 0 else 0.04
                    overlap_frames = min(15, max(5, int(fps * 0.5))) if fps > 0 else 10
                    overlap_sec = overlap_frames / fps if fps > 0 else 0.4
                    segs_for_draw = []
                    for s, e, t in segments_with_trans:
                        try:
                            if abs(float(e) - float(time_sec)) <= dt and (float(e) - float(s)) <= overlap_sec:
                                segs_for_draw.append((float(s), float(time_sec) + dt, t))
                            else:
                                segs_for_draw.append((s, e, t))
                        except Exception:
                            segs_for_draw.append((s, e, t))
                    _draw_timed_subs_on_image(
                        frame,
                        time_sec,
                        segs_for_draw,
                        style=style,
                        stack_multiple_lines=True,
                        **subtitle_black_box_draw_kwargs_from_cfg(cfg),
                    )
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
                srt_entries = [
                    (int(s * fps), int(e * fps), text)
                    for s, e, text in segments_with_trans
                    if (text or "").strip() and not _is_invalid_subtitle_text(text)
                ]
                base_path = osp.splitext(self.output_path)[0]
                if srt_entries:
                    _write_srt(base_path + ".srt", srt_entries, fps)
                    if self.export_ass:
                        _write_ass(base_path + ".ass", srt_entries, fps, w, h)
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
        from modules.video_translator import (
            _draw_timed_subs_on_image,
            subtitle_black_box_draw_kwargs_from_cfg,
            configure_translator_video_nlp_parallel,
            clear_translator_video_nlp_parallel,
        )
        from modules.video_subtitle_extract import load_existing_subtitles
        from utils.textblock import TextBlock

        try:
            cfg = pcfg.module
            ffmpeg_exe = (self.ffmpeg_path or getattr(cfg, "video_translator_ffmpeg_path", "") or "ffmpeg").strip()
            segments = load_existing_subtitles(self.input_path, ffmpeg_path=ffmpeg_exe)
            if not segments:
                self.failed.emit("No existing subtitles found (no sidecar .srt/.ass/.vtt and no embedded subtitle stream).")
                return
            if bool(getattr(cfg, "video_translator_sentence_merge_by_punctuation", True)):
                try:
                    segments = _merge_segments_until_sentence_end(
                        segments,
                        max_merge_seconds=float(getattr(cfg, "video_translator_sentence_merge_max_seconds", 8.0)),
                    )
                except Exception as e:
                    LOGGER.debug("Existing subs rule-based sentence merge failed (using original): %s", e)
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
                configure_translator_video_nlp_parallel(translator, cfg)
                try:
                    setattr(translator, "_current_page_key", "video_existing_subs")
                    setattr(translator, "_current_page_image", None)
                    translator.translate_textblk_lst(blk_list)
                except Exception as e:
                    LOGGER.warning("Existing subs translate batch failed: %s", e)
                finally:
                    clear_translator_video_nlp_parallel(translator)
                _align_blk_list_to_segments(segments, blk_list, TextBlock)
                for (s, e, _), blk in zip(segments, blk_list):
                    trans = (getattr(blk, "translation", None) or blk.get_text() or "").strip()
                    if _is_region_placeholder_text(trans):
                        trans = ""
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
                    effective_preset = (getattr(cfg, "video_translator_ffmpeg_preset", None) or "medium").strip()
                    hw_encoder = (getattr(cfg, "video_translator_ffmpeg_hw_encoder", None) or "none").strip().lower()
                    ffmpeg_cmd = _build_ffmpeg_encode_cmd(
                        ffmpeg_exe, w, h, fps, self.output_path,
                        use_bitrate=(effective_kbps > 0), effective_kbps=effective_kbps,
                        crf=self.ffmpeg_crf, preset=effective_preset, hw_encoder=hw_encoder,
                    )
                    ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
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
            prev_active = []  # list of active subtitle texts on previous frame
            step = 25 if total <= 10000 else (50 if total <= 100000 else 200)
            while True:
                if self._cancel:
                    break
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                time_sec = n / fps if fps > 0 else 0
                if not self.soft_subs_only:
                    # Log start/end transitions for hardcoded (burn-in) subtitles (only when active set changes).
                    try:
                        active_now = []
                        for s, e, t in segments_with_trans:
                            txt = (t or "").strip()
                            if txt and (not _is_invalid_subtitle_text(txt)) and float(s) <= float(time_sec) < float(e):
                                active_now.append(txt)
                        if active_now != prev_active:
                            ended = [t for t in prev_active if t not in active_now]
                            started = [t for t in active_now if t not in prev_active]
                            for t in ended:
                                LOGGER.info("Existing-sub subtitle ended @ %.3fs: %r", float(time_sec), t)
                            for t in started:
                                seg_s, seg_e = None, None
                                for s, e, txt in segments_with_trans:
                                    if (txt or "").strip() == t:
                                        seg_s, seg_e = float(s), float(e)
                                        break
                                if seg_s is not None and seg_e is not None:
                                    LOGGER.info("Existing-sub subtitle started @ %.3fs (%.3f-%.3fs): %r", float(time_sec), seg_s, seg_e, t)
                                else:
                                    LOGGER.info("Existing-sub subtitle started @ %.3fs: %r", float(time_sec), t)
                            prev_active = active_now
                    except Exception:
                        pass
                    # Stack back-to-back/overlapping subs and give very short cues a 1-frame overlap at boundaries
                    # to avoid flicker/overwrite when end/start are extremely close.
                    dt = (1.0 / fps) if fps > 0 else 0.04
                    overlap_frames = min(15, max(5, int(fps * 0.5))) if fps > 0 else 10
                    overlap_sec = overlap_frames / fps if fps > 0 else 0.4
                    segs_for_draw = []
                    for s, e, t in segments_with_trans:
                        try:
                            if abs(float(e) - float(time_sec)) <= dt and (float(e) - float(s)) <= overlap_sec:
                                segs_for_draw.append((float(s), float(time_sec) + dt, t))
                            else:
                                segs_for_draw.append((s, e, t))
                        except Exception:
                            segs_for_draw.append((s, e, t))
                    _draw_timed_subs_on_image(
                        frame,
                        time_sec,
                        segs_for_draw,
                        style=style,
                        stack_multiple_lines=True,
                        **subtitle_black_box_draw_kwargs_from_cfg(cfg),
                    )
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
                srt_entries = [
                    (int(s * fps), int(e * fps), text)
                    for s, e, text in segments_with_trans
                    if (text or "").strip() and not _is_invalid_subtitle_text(text)
                ]
                base_path = osp.splitext(self.output_path)[0]
                if srt_entries:
                    _write_srt(base_path + ".srt", srt_entries, fps)
                    if self.export_ass:
                        _write_ass(base_path + ".ass", srt_entries, fps, w, h)
                    if self.export_vtt:
                        _write_vtt(base_path + ".vtt", srt_entries, fps)

            if osp.isfile(self.output_path):
                self.finished_ok.emit(self.output_path)
            else:
                self.failed.emit("Cancelled.")
        except Exception as e:
            LOGGER.exception("Existing subtitles pipeline failed")
            self.failed.emit(str(e))

    def _run_ocr_two_pass_pipeline(self):
        """
        OCR hardcoded subtitles, two-pass:
        1) detect/OCR/inpaint + async translation collection (no burn-in draw in pass 1)
        2) burn-in timed subtitles on the inpainted intermediate video
        """
        import cv2
        import subprocess
        import numpy as np
        from utils.textblock import TextBlock
        from modules.video_translator import (
            run_one_frame_pipeline,
            _draw_timed_subs_on_image,
            subtitle_black_box_draw_kwargs_from_cfg,
            _region_fraction_from_preset,
            translate_video_textblk_list,
        )

        try:
            self._progress_phase = "Pass 1/2 (scan + inpaint)"
            cfg = pcfg.module
            cap = cv2.VideoCapture(self.input_path)
            if not cap.isOpened():
                self.failed.emit("Could not open input video.")
                return
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            fps = _normalize_fps(
                cap,
                cap.get(cv2.CAP_PROP_FPS) or 0.0,
                total,
                video_path=self.input_path,
                ffmpeg_exe=self.ffmpeg_path or "",
            )
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if w <= 0 or h <= 0:
                cap.release()
                self.failed.emit("Could not open input video or no frames.")
                return
            if total <= 0 and fps > 0:
                total = -1

            source_bitrate_kbps = _get_source_bitrate_kbps(self.input_path, self.ffmpeg_path) or 0

            detector = _get_video_module("textdetector", cfg.textdetector) if (self.enable_detect and not self.skip_detect) else None
            ocr = _get_video_module("ocr", cfg.ocr) if self.enable_ocr else None
            translator = _get_video_module("translator", cfg.translator) if self.enable_translate else None
            inpainter = _get_video_module("inpainter", cfg.inpainter) if self.enable_inpaint else None

            if self.enable_detect and not self.skip_detect and detector is not None and hasattr(detector, "load_model"):
                detector.load_model()
            if self.enable_ocr and ocr is not None and hasattr(ocr, "load_model"):
                ocr.load_model()
            if self.enable_inpaint and inpainter is not None and hasattr(inpainter, "load_model"):
                inpainter.load_model()
            if self.enable_translate and translator is not None and hasattr(translator, "load_model"):
                translator.load_model()

            if ocr is not None:
                setattr(ocr, "_video_frame_ocr_cache", None)
                setattr(ocr, "_video_frame_ocr_cache_order", None)
            if translator is not None:
                setattr(translator, "_video_frame_cache", None)
                setattr(translator, "_video_recent_text_cache", None)
                setattr(translator, "_video_glossary_hint", (getattr(cfg, "video_translator_glossary", None) or "").strip())
                sp = (getattr(cfg, "video_translator_series_context_path", None) or "").strip()
                if sp and hasattr(translator, "set_translation_context"):
                    translator.set_translation_context(series_context_path=sp)
            else:
                sp = ""

            flow_fixer_review = None
            if getattr(cfg, "video_translator_flow_fixer_enabled", False):
                try:
                    from modules.flow_fixer import get_flow_fixer
                    fixer_name = (getattr(cfg, "video_translator_flow_fixer", None) or "none").strip().lower()
                    if fixer_name == "openrouter":
                        flow_fixer_review = get_flow_fixer(
                            "openrouter",
                            api_key=(getattr(cfg, "video_translator_flow_fixer_openrouter_apikey", None) or "").strip(),
                            model=(getattr(cfg, "video_translator_flow_fixer_openrouter_model", None) or "google/gemma-3n-e2b-it:free").strip(),
                            max_tokens=int(getattr(cfg, "video_translator_flow_fixer_max_tokens", 256)),
                            timeout=float(getattr(cfg, "video_translator_flow_fixer_timeout", 30.0)),
                            enable_reasoning=bool(getattr(cfg, "video_translator_flow_fixer_enable_reasoning", False)),
                            reasoning_effort=(getattr(cfg, "video_translator_flow_fixer_reasoning_effort", None) or "medium"),
                        )
                    elif fixer_name == "openai":
                        flow_fixer_review = get_flow_fixer(
                            "openai",
                            api_key=(getattr(cfg, "video_translator_flow_fixer_openai_apikey", None) or "").strip(),
                            model=(getattr(cfg, "video_translator_flow_fixer_openai_model", None) or "gpt-4o-mini").strip(),
                            max_tokens=int(getattr(cfg, "video_translator_flow_fixer_max_tokens", 256)),
                            timeout=float(getattr(cfg, "video_translator_flow_fixer_timeout", 30.0)),
                            enable_reasoning=bool(getattr(cfg, "video_translator_flow_fixer_enable_reasoning", False)),
                            reasoning_effort=(getattr(cfg, "video_translator_flow_fixer_reasoning_effort", None) or "medium"),
                        )
                    elif fixer_name == "local_server":
                        local_model = (getattr(cfg, "video_translator_flow_fixer_model", None) or "").strip()
                        if local_model:
                            flow_fixer_review = get_flow_fixer(
                                "local_server",
                                server_url=(getattr(cfg, "video_translator_flow_fixer_server_url", None) or "http://localhost:1234/v1").strip(),
                                model=local_model,
                                max_tokens=int(getattr(cfg, "video_translator_flow_fixer_max_tokens", 256)),
                                timeout=float(getattr(cfg, "video_translator_flow_fixer_timeout", 30.0)),
                                enable_reasoning=bool(getattr(cfg, "video_translator_flow_fixer_enable_reasoning", False)),
                                reasoning_effort=(getattr(cfg, "video_translator_flow_fixer_reasoning_effort", None) or "medium"),
                            )
                except Exception:
                    flow_fixer_review = None

            # Intermediate inpainted video (no burned subtitles).
            tmp_fd, tmp_inpaint_path = tempfile.mkstemp(prefix="bt_inpaint_", suffix=".mp4")
            os.close(tmp_fd)

            use_ffmpeg_pass1 = self.use_ffmpeg
            ffmpeg_proc1 = None
            if use_ffmpeg_pass1:
                try:
                    ffmpeg_exe = self.ffmpeg_path if self.ffmpeg_path else "ffmpeg"
                    effective_kbps = self.video_bitrate_kbps if self.video_bitrate_kbps > 0 else source_bitrate_kbps
                    effective_preset = (getattr(cfg, "video_translator_ffmpeg_preset", None) or "medium").strip()
                    hw_encoder = (getattr(cfg, "video_translator_ffmpeg_hw_encoder", None) or "none").strip().lower()
                    ffmpeg_cmd = _build_ffmpeg_encode_cmd(
                        ffmpeg_exe,
                        w,
                        h,
                        fps,
                        tmp_inpaint_path,
                        use_bitrate=(effective_kbps > 0),
                        effective_kbps=effective_kbps,
                        crf=self.ffmpeg_crf,
                        preset=effective_preset,
                        hw_encoder=hw_encoder,
                    )
                    ffmpeg_proc1 = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
                except Exception as e:
                    LOGGER.warning("Pass1 FFmpeg unavailable (%s), using OpenCV writer.", e)
                    use_ffmpeg_pass1 = False

            out1 = None
            if not use_ffmpeg_pass1:
                codec = (getattr(cfg, "video_translator_output_codec", None) or "mp4v").strip() or "mp4v"
                if len(codec) != 4:
                    codec = "mp4v"
                out1 = cv2.VideoWriter(tmp_inpaint_path, cv2.VideoWriter_fourcc(*codec), fps, (w, h))
                if not out1.isOpened() and codec != "avc1":
                    out1 = cv2.VideoWriter(tmp_inpaint_path, cv2.VideoWriter_fourcc(*"avc1"), fps, (w, h))
                if not out1.isOpened():
                    cap.release()
                    self.failed.emit("Could not create intermediate video.")
                    return

            # Translation worker (single thread): keeps ordering/context stable while decode/pipeline continues.
            translation_jobs = []  # (frame_idx, future)
            translation_history = []
            try:
                translation_history_max = int(
                    translator.get_param_value("context_previous_pages")
                ) if (translator is not None and hasattr(translator, "get_param_value")) else 10
            except Exception:
                translation_history_max = 10
            translation_history_max = max(10, min(50, translation_history_max))
            trans_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1) if translator is not None else None

            def _filtered_context_entries(entries: list, max_items: int) -> list:
                """Keep only useful recent subtitle context entries for flow continuity."""
                out = []
                for e in entries or []:
                    if not isinstance(e, dict):
                        continue
                    srcs = [str(s or "").strip() for s in (e.get("sources") or [])]
                    trans = [str(t or "").strip() for t in (e.get("translations") or [])]
                    clean_trans = [t for t in trans if t and not _is_invalid_subtitle_text(t)]
                    if not clean_trans:
                        continue
                    clean_srcs = [s for s in srcs if s and not _is_invalid_subtitle_text(s)]
                    out.append({"sources": clean_srcs, "translations": clean_trans})
                return out[-max_items:] if len(out) > max_items else out

            def _clone_blk_for_translation(src_blk: TextBlock) -> TextBlock:
                b = TextBlock()
                try:
                    b.xyxy = list(getattr(src_blk, "xyxy", []) or [])
                except Exception:
                    b.xyxy = []
                try:
                    b.lines = copy.deepcopy(getattr(src_blk, "lines", []) or [])
                except Exception:
                    b.lines = []
                txt = (src_blk.get_text() if hasattr(src_blk, "get_text") else "") or ""
                b.text = [txt]
                b.translation = ""
                return b

            def _translate_job(frame_idx: int, blk_payload: list[TextBlock]):
                if translator is None or not blk_payload:
                    return frame_idx, []
                try:
                    if hasattr(translator, "set_translation_context"):
                        translator.set_translation_context(
                            previous_pages=_filtered_context_entries(translation_history, translation_history_max),
                            series_context_path=sp if sp else None,
                        )
                except Exception:
                    pass
                translate_video_textblk_list(
                    translator=translator,
                    blk_list=blk_payload,
                    cfg=cfg,
                    frame_index=frame_idx,
                    img=None,
                    flow_fixer=None,
                    use_flow_fixer_for_corrections=False,
                )
                # Built-in flow continuity (without Flow Fixer):
                # apply model-provided revised_previous to recent history so subsequent
                # frames inherit corrected phrasing/tense/pronouns.
                try:
                    rev = getattr(translator, "_last_revised_previous", None)
                    if rev and isinstance(rev, list) and len(rev) > 0 and len(translation_history) >= len(rev):
                        k = len(rev)
                        for i in range(k):
                            idx = -k + i
                            if i < len(rev) and rev[i]:
                                translation_history[idx]["translations"] = [str(rev[i]).strip()]
                        setattr(translator, "_last_revised_previous", None)
                except Exception:
                    pass
                srcs = []
                trans = []
                for b in blk_payload:
                    s = (b.get_text() if hasattr(b, "get_text") else "") or ""
                    t = (getattr(b, "translation", None) or "").strip()
                    srcs.append(s)
                    trans.append(t)
                if srcs or trans:
                    translation_history.append({"sources": srcs, "translations": trans})
                    if len(translation_history) > translation_history_max:
                        del translation_history[:-translation_history_max]
                return frame_idx, trans

            cached_result = None
            cached_blk_list = None
            cached_source_frame = None
            prev_gray = None
            last_pipeline_n = -self.sample_every_frames - 1
            last_content_hash = None
            last_band_small_gray = None
            subtitle_frac = _region_fraction_from_preset(cfg)
            subtitle_roi_frac = subtitle_frac if subtitle_frac > 0 else 1.0
            scene_roi_frac = subtitle_roi_frac
            two_stage_force_every_frames = int(getattr(cfg, "video_translator_two_stage_force_refresh_every_frames", 0) or 0)
            two_stage_new_line_diff_threshold = float(getattr(cfg, "video_translator_two_stage_new_line_diff_threshold", 8.0) or 8.0)
            n = 0
            while True:
                if self._cancel:
                    break
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
                pending_two_stage_hash = None
                pending_two_stage_small_gray = None
                periodic = n % self.sample_every_frames == 0
                if self.use_scene_detection:
                    scene_changed = _scene_change(prev_gray, curr_gray, self.scene_threshold, bottom_frac=scene_roi_frac)
                    run_pipeline = periodic or (scene_changed and (n - last_pipeline_n) >= self.sample_every_frames)
                else:
                    run_pipeline = periodic
                prev_gray = curr_gray

                if run_pipeline and self.two_stage_keyframes:
                    curr_small_gray = _subtitle_region_small_gray(frame, subtitle_roi_frac)
                    if curr_small_gray is not None:
                        curr_hash = hashlib.md5(curr_small_gray.tobytes()).digest()
                        forced_due = (
                            two_stage_force_every_frames > 0
                            and last_pipeline_n >= 0
                            and (n - last_pipeline_n) >= two_stage_force_every_frames
                        )
                        should_skip = False
                        if last_content_hash is not None and curr_hash == last_content_hash and not forced_due:
                            if two_stage_new_line_diff_threshold > 0 and last_band_small_gray is not None:
                                diff = float(
                                    np.mean(
                                        np.abs(
                                            curr_small_gray.astype(np.int16)
                                            - last_band_small_gray.astype(np.int16)
                                        )
                                    )
                                )
                                should_skip = diff <= two_stage_new_line_diff_threshold
                            else:
                                should_skip = True
                        if should_skip:
                            run_pipeline = False
                        else:
                            pending_two_stage_hash = curr_hash
                            pending_two_stage_small_gray = curr_small_gray

                if run_pipeline:
                    last_pipeline_n = n
                    out_frame, blk_list, _ = run_one_frame_pipeline(
                        frame,
                        detector,
                        ocr,
                        None,  # translation deferred to worker
                        inpainter,
                        self.enable_detect,
                        self.enable_ocr,
                        False,  # no translation in pass 1
                        self.enable_inpaint,
                        cfg=cfg,
                        skip_detect=self.skip_detect,
                        draw_subtitles=False,
                        detect_roi_xyxy=None,
                        frame_index=n,
                        flow_fixer=None,
                        use_flow_fixer_for_corrections=False,
                    )
                    cached_result = out_frame
                    cached_blk_list = blk_list
                    cached_source_frame = frame.copy()
                    if pending_two_stage_hash is not None:
                        last_content_hash = pending_two_stage_hash
                        last_band_small_gray = pending_two_stage_small_gray

                    if trans_executor is not None and blk_list:
                        payload = [_clone_blk_for_translation(b) for b in blk_list]
                        fut = trans_executor.submit(_translate_job, n, payload)
                        translation_jobs.append((n, fut))
                if cached_result is not None and cached_blk_list is not None:
                    comp = _composite_cached_subs_on_frame(
                        frame,
                        cached_result,
                        cached_source_frame,
                        cached_blk_list,
                        h,
                        w,
                    )
                    write_fr = comp if comp is not None else frame
                elif cached_result is not None:
                    write_fr = cached_result
                else:
                    write_fr = frame

                if use_ffmpeg_pass1 and ffmpeg_proc1 and ffmpeg_proc1.stdin:
                    try:
                        ffmpeg_proc1.stdin.write(write_fr.tobytes())
                    except Exception:
                        pass
                elif out1 is not None:
                    out1.write(write_fr)

                if total > 0:
                    step = 25 if total <= 10000 else (50 if total <= 100000 else 200)
                    if n % step == 0 or n == total:
                        self.progress.emit(min(total * 2, n), total * 2)
                elif n % 30 == 0:
                    self.progress.emit(n, 0)
                n += 1

            cap.release()
            if out1 is not None:
                out1.release()
            if use_ffmpeg_pass1 and ffmpeg_proc1 and ffmpeg_proc1.stdin:
                try:
                    ffmpeg_proc1.stdin.close()
                    ffmpeg_proc1.wait(timeout=max(600, (total or 0) // 1000))
                except Exception:
                    try:
                        ffmpeg_proc1.terminate()
                    except Exception:
                        pass

            if trans_executor is not None:
                trans_executor.shutdown(wait=True)

            # Build timed cues after all translations finish.
            srt_entries = []
            jobs_sorted = sorted(translation_jobs, key=lambda x: x[0])
            for frame_idx, fut in jobs_sorted:
                try:
                    _, trans = fut.result()
                except Exception:
                    trans = []
                parts = []
                for t in trans:
                    tt = (t or "").strip()
                    if tt and not _is_invalid_subtitle_text(tt):
                        parts.append(tt)
                text = " ".join(parts).strip()
                if not text:
                    continue
                seg_end = frame_idx + max(1, self.sample_every_frames)
                if srt_entries and srt_entries[-1][2] == text:
                    srt_entries[-1] = (srt_entries[-1][0], seg_end, srt_entries[-1][2])
                else:
                    if srt_entries:
                        srt_entries[-1] = (srt_entries[-1][0], frame_idx, srt_entries[-1][2])
                    srt_entries.append((frame_idx, seg_end, text))
            if srt_entries:
                srt_entries[-1] = (srt_entries[-1][0], max(srt_entries[-1][1], n), srt_entries[-1][2])

            # Optional post-run global subtitle flow review before burn-in pass (dedicated flow fixer or main LLM).
            post_review_fixer = None
            try:
                from modules.flow_fixer.translator_flow_fixer import resolve_post_review_flow_fixer

                post_review_fixer = resolve_post_review_flow_fixer(cfg, translator, flow_fixer_review)
            except Exception as e:
                LOGGER.debug("Post-review flow fixer resolve failed: %s", e)
            if srt_entries and post_review_fixer is not None:
                post_review_enabled = bool(getattr(cfg, "video_translator_post_review_enabled", True))
                post_review_on_cancel = bool(getattr(cfg, "video_translator_post_review_apply_on_cancel", True))
                can_run = post_review_enabled and ((not self._cancel) or post_review_on_cancel)
                if can_run:
                    try:
                        from modules.flow_fixer.corrections import improve_subtitle_timeline_via_fixer

                        original_texts = [str(e[2] or "").strip() for e in srt_entries]
                        chunk_size = int(getattr(cfg, "video_translator_post_review_chunk_size", 80) or 80)
                        ctx_lines = int(getattr(cfg, "video_translator_post_review_context_lines", 20) or 20)
                        _tl = "en"
                        if translator is not None:
                            _tl = str(getattr(translator, "lang_target", None) or "en")
                        revised_texts = improve_subtitle_timeline_via_fixer(
                            flow_fixer=post_review_fixer,
                            timeline_texts=original_texts,
                            target_lang=_tl,
                            chunk_size=chunk_size,
                            context_lines=ctx_lines,
                        )
                        if revised_texts and len(revised_texts) == len(srt_entries):
                            n_changed = 0
                            updated = []
                            for i, (start_f, end_f, old_text) in enumerate(srt_entries):
                                new_text = str(revised_texts[i] or "").strip()
                                if not new_text:
                                    new_text = str(old_text or "").strip()
                                if new_text != str(old_text or "").strip():
                                    n_changed += 1
                                updated.append((start_f, end_f, new_text))
                            srt_entries = updated
                            if n_changed > 0:
                                LOGGER.info(
                                    "Post-run subtitle flow review changed %d/%d segment(s) before burn-in.",
                                    n_changed,
                                    len(srt_entries),
                                )
                    except Exception as e:
                        LOGGER.debug("Post-run subtitle flow review failed before burn-in: %s", e)

            # Pass 2: burn-in translated timed cues onto inpainted intermediate.
            self._progress_phase = "Pass 2/2 (burn-in)"
            cap2 = cv2.VideoCapture(tmp_inpaint_path)
            if not cap2.isOpened():
                self.failed.emit("Could not open intermediate video for burn-in.")
                return

            use_ffmpeg_pass2 = self.use_ffmpeg
            ffmpeg_proc2 = None
            if use_ffmpeg_pass2:
                try:
                    ffmpeg_exe = self.ffmpeg_path if self.ffmpeg_path else "ffmpeg"
                    effective_kbps = self.video_bitrate_kbps if self.video_bitrate_kbps > 0 else source_bitrate_kbps
                    effective_preset = (getattr(cfg, "video_translator_ffmpeg_preset", None) or "medium").strip()
                    hw_encoder = (getattr(cfg, "video_translator_ffmpeg_hw_encoder", None) or "none").strip().lower()
                    ffmpeg_cmd = _build_ffmpeg_encode_cmd(
                        ffmpeg_exe,
                        w,
                        h,
                        fps,
                        self.output_path,
                        use_bitrate=(effective_kbps > 0),
                        effective_kbps=effective_kbps,
                        crf=self.ffmpeg_crf,
                        preset=effective_preset,
                        hw_encoder=hw_encoder,
                    )
                    ffmpeg_proc2 = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
                except Exception as e:
                    LOGGER.warning("Pass2 FFmpeg unavailable (%s), using OpenCV writer.", e)
                    use_ffmpeg_pass2 = False

            out2 = None
            if not use_ffmpeg_pass2:
                codec = (getattr(cfg, "video_translator_output_codec", None) or "mp4v").strip() or "mp4v"
                if len(codec) != 4:
                    codec = "mp4v"
                out2 = cv2.VideoWriter(self.output_path, cv2.VideoWriter_fourcc(*codec), fps, (w, h))
                if not out2.isOpened() and codec != "avc1":
                    out2 = cv2.VideoWriter(self.output_path, cv2.VideoWriter_fourcc(*"avc1"), fps, (w, h))
                if not out2.isOpened():
                    cap2.release()
                    self.failed.emit("Could not create output video.")
                    return

            m = 0
            while True:
                if self._cancel:
                    break
                ret2, fr2 = cap2.read()
                if not ret2 or fr2 is None:
                    break
                if srt_entries:
                    overlap_frames = min(15, max(5, int(fps * 0.5)))
                    segments_for_draw = []
                    for (start_f, end_f, text) in srt_entries:
                        if start_f <= m < end_f:
                            segments_for_draw.append((start_f / fps, end_f / fps, text))
                        elif end_f == m and (end_f - start_f) <= overlap_frames:
                            segments_for_draw.append((start_f / fps, (m + 1) / fps, text))
                    _draw_timed_subs_on_image(
                        fr2,
                        m / fps if fps > 0 else 0.0,
                        segments_for_draw,
                        stack_multiple_lines=True,
                        **subtitle_black_box_draw_kwargs_from_cfg(cfg),
                    )

                if use_ffmpeg_pass2 and ffmpeg_proc2 and ffmpeg_proc2.stdin:
                    try:
                        ffmpeg_proc2.stdin.write(fr2.tobytes())
                    except Exception:
                        pass
                elif out2 is not None:
                    out2.write(fr2)
                if total > 0:
                    step2 = 10 if total <= 10000 else (500 if total <= 100000 else 2000)
                    if m % step2 == 0 or m == total:
                        self.progress.emit(min(total * 2, total + m), total * 2)
                elif m % 30 == 0:
                    self.progress.emit(m, 0)
                m += 1

            cap2.release()
            if out2 is not None:
                out2.release()
            if use_ffmpeg_pass2 and ffmpeg_proc2 and ffmpeg_proc2.stdin:
                try:
                    ffmpeg_proc2.stdin.close()
                    ffmpeg_proc2.wait(timeout=max(600, (total or 0) // 1000))
                except Exception:
                    try:
                        ffmpeg_proc2.terminate()
                    except Exception:
                        pass

            # Burn-in mode: remove sidecar if exists (avoid double subtitles)
            base_path = osp.splitext(self.output_path)[0]
            for ext in (".srt", ".ass", ".vtt"):
                p = base_path + ext
                if osp.isfile(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass

            try:
                if osp.isfile(tmp_inpaint_path):
                    os.remove(tmp_inpaint_path)
            except Exception:
                pass

            if osp.isfile(self.output_path):
                self._progress_phase = ""
                self.finished_ok.emit(self.output_path)
            else:
                self._progress_phase = ""
                self.failed.emit("Cancelled.")
        except Exception as e:
            self._progress_phase = ""
            LOGGER.exception("Two-pass OCR pipeline failed")
            self.failed.emit(str(e))

    def run(self):
        import cv2
        import subprocess
        import numpy as np
        from modules.video_translator import (
            run_one_frame_pipeline,
            _region_fraction_from_preset,
            _draw_text_on_image,
            _draw_timed_subs_on_image,
            subtitle_black_box_draw_kwargs_from_cfg,
        )

        try:
            src = getattr(self, "source", "ocr").strip().lower()
            if src == "asr":
                self._run_asr_pipeline()
                return
            if src == "existing_subs":
                self._run_existing_subs_pipeline()
                return
            # OCR hardcoded subtitles + burn-in: run fast two-pass pipeline so translation never blocks frame scanning.
            if (
                src == "ocr"
                and (not self.soft_subs_only)
                and (not self.inpaint_only_soft_subs)
                and self.enable_translate
                and self.use_two_pass_ocr_burn_in
            ):
                self._run_ocr_two_pass_pipeline()
                return

            # Pipeline thread: raise priority so decode/encode threads don't throttle it (Windows).
            _set_thread_priority_above_normal()

            # Always decode in a separate thread so slow I/O never blocks the pipeline.
            frame_queue = queue_module.Queue(maxsize=max(2, self.prefetch_frames + 2))

            def _frame_reader():
                _set_thread_priority_below_normal()
                cap_r = cv2.VideoCapture(self.input_path)
                if not cap_r.isOpened():
                    frame_queue.put(("meta", 0.0, 0, 0, 0))
                    frame_queue.put((False, None, 0))
                    return
                total_r = int(cap_r.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
                fps_r = _normalize_fps(
                    cap_r, cap_r.get(cv2.CAP_PROP_FPS) or 0.0, total_r,
                    video_path=self.input_path,
                    ffmpeg_exe=self.ffmpeg_path or "",
                )
                ret, frame = cap_r.read()
                if ret and frame is not None:
                    h_r, w_r = frame.shape[:2]
                    frame_queue.put(("meta", fps_r, total_r, w_r, h_r))
                    frame_queue.put((True, frame, 0))
                else:
                    frame_queue.put(("meta", fps_r, total_r, 0, 0))
                    frame_queue.put((False, None, 0))
                    cap_r.release()
                    return
                idx = 1
                while not self._cancel:
                    ret, frame = cap_r.read()
                    frame_queue.put((ret, frame, idx))
                    if not ret:
                        break
                    idx += 1
                cap_r.release()

            reader_thread = threading.Thread(target=_frame_reader, daemon=True, name="VideoDecode")
            reader_thread.start()
            meta = frame_queue.get()
            if not (isinstance(meta, tuple) and len(meta) >= 5 and meta[0] == "meta"):
                reader_thread.join(timeout=2)
                self.failed.emit("Could not open input video (prefetch).")
                return
            fps, total, w, h = meta[1], meta[2], meta[3], meta[4]
            total = int(total)
            if w <= 0 or h <= 0:
                reader_thread.join(timeout=2)
                self.failed.emit("Could not open input video or no frames.")
                return
            if total <= 0 and fps > 0:
                total = -1
            source_bitrate_kbps = _get_source_bitrate_kbps(self.input_path, self.ffmpeg_path) or 0

            cap = None  # Decode is in reader thread; main loop never reads from cap
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
                                enable_reasoning=bool(getattr(cfg, "video_translator_flow_fixer_enable_reasoning", False)),
                                reasoning_effort=(getattr(cfg, "video_translator_flow_fixer_reasoning_effort", None) or "medium"),
                            )
                        elif fixer_name == "openai":
                            flow_fixer = get_flow_fixer(
                                "openai",
                                api_key=(getattr(cfg, "video_translator_flow_fixer_openai_apikey", None) or "").strip(),
                                model=(getattr(cfg, "video_translator_flow_fixer_openai_model", None) or "gpt-4o-mini").strip(),
                                max_tokens=int(getattr(cfg, "video_translator_flow_fixer_max_tokens", 256)),
                                timeout=float(getattr(cfg, "video_translator_flow_fixer_timeout", 30.0)),
                                enable_reasoning=bool(getattr(cfg, "video_translator_flow_fixer_enable_reasoning", False)),
                                reasoning_effort=(getattr(cfg, "video_translator_flow_fixer_reasoning_effort", None) or "medium"),
                            )
                        else:
                            # Local (Ollama / LM Studio): require non-empty model name so user picks the right one
                            local_model = (getattr(cfg, "video_translator_flow_fixer_model", None) or "").strip()
                            if not local_model:
                                LOGGER.info(
                                    "Flow fixer (local): model name is empty. Enter the exact model name in Video translator options (e.g. the name shown in LM Studio) to use flow fixer."
                                )
                            else:
                                flow_fixer = get_flow_fixer(
                                    fixer_name,
                                    server_url=(getattr(cfg, "video_translator_flow_fixer_server_url", None) or "http://localhost:1234/v1").strip(),
                                    model=local_model,
                                    max_tokens=int(getattr(cfg, "video_translator_flow_fixer_max_tokens", 256)),
                                    timeout=float(getattr(cfg, "video_translator_flow_fixer_timeout", 30.0)),
                                    enable_reasoning=bool(getattr(cfg, "video_translator_flow_fixer_enable_reasoning", False)),
                                    reasoning_effort=(getattr(cfg, "video_translator_flow_fixer_reasoning_effort", None) or "medium"),
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
                    effective_preset = (getattr(cfg, "video_translator_ffmpeg_preset", None) or "medium").strip()
                    hw_encoder = (getattr(cfg, "video_translator_ffmpeg_hw_encoder", None) or "none").strip().lower()
                    ffmpeg_cmd = _build_ffmpeg_encode_cmd(
                        ffmpeg_exe, w, h, fps, self.output_path,
                        use_bitrate=(effective_kbps > 0),
                        effective_kbps=effective_kbps,
                        crf=self.ffmpeg_crf,
                        preset=effective_preset,
                        hw_encoder=hw_encoder,
                    )
                    ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
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
                    reader_thread.join(timeout=2)
                    self.failed.emit("Could not create output video (try different output path or codec).")
                    return
            else:
                out = None

            # Always encode in a separate thread so slow I/O never blocks the pipeline.
            write_queue = queue_module.Queue(maxsize=90)

            def _writer_loop():
                _set_thread_priority_below_normal()
                pending = {}
                next_index = 0
                while True:
                    item = write_queue.get()
                    if item[0] is None:
                        break
                    idx, fr = item
                    if fr is not None:
                        pending[idx] = fr
                    while next_index in pending:
                        f = pending.pop(next_index)
                        if use_ffmpeg and ffmpeg_proc and ffmpeg_proc.stdin:
                            try:
                                ffmpeg_proc.stdin.write(f.tobytes())
                            except Exception:
                                pass
                        elif out is not None:
                            out.write(f)
                        next_index += 1

            writer_thread = threading.Thread(target=_writer_loop, daemon=True, name="VideoEncode")
            writer_thread.start()

            def write_frame(frame_index, frame):
                try:
                    write_queue.put((frame_index, frame.copy() if frame is not None else None), block=True, timeout=300)
                except Exception:
                    pass

            # When burning in, remove any existing sidecar SRT/ASS/VTT so the player does not show double subtitles
            if not self.soft_subs_only and not self.inpaint_only_soft_subs:
                base_path = osp.splitext(self.output_path)[0]
                for ext in (".srt", ".ass", ".vtt"):
                    p = base_path + ext
                    if osp.isfile(p):
                        try:
                            os.remove(p)
                        except Exception:
                            pass

            cached_result = None
            cached_blk_list = None  # blocks from last pipeline run (for compositing so motion stays full fps)
            cached_source_frame = None  # raw frame at last "good" pipeline tick (used to validate reuse)
            cached_inpainted = None  # inpainted frame without text (for srt-based burn-in with stacking)
            # Keep last "good" subtitles so a single missed detection doesn't make subs flicker off.
            last_good_cached_result = None
            last_good_cached_blk_list = None
            last_good_cached_source_frame = None
            last_good_cached_inpainted = None
            miss_streak = 0
            cached_hold_until = -1  # frame index until which we keep compositing cached subs
            last_cache_clear_frame = -9999  # when we last set cached_blk_list = [], for limited last_good fallback
            prev_gray = None
            prev_result = None
            last_pipeline_n = -self.sample_every_frames - 1  # so we run on n=0 and respect "process every N" minimum interval
            last_content_hash = None  # for two-stage keyframes: skip pipeline when subtitle region unchanged
            last_band_small_gray = None  # for "new line" detection: cheap band diff
            n = 0
            srt_entries = []
            current_preview_texts = []
            preview_throttle = 5
            video_previous_subtitles = []
            max_video_context = 10
            # Memoize flow-fixer calls to avoid repeated model latency on identical translation sequences.
            flow_fixer_cache = {}
            flow_fixer_cache_order = []
            flow_fixer_cache_max = max(1, int(getattr(cfg, "video_translator_flow_fixer_cache_size", 200) or 200))
            # Clear per-region OCR cache and translator cache so a new video run doesn't reuse old results
            if ocr is not None:
                setattr(ocr, "_video_frame_ocr_cache", None)
                setattr(ocr, "_video_frame_ocr_cache_order", None)
            if translator is not None:
                setattr(translator, "_video_frame_cache", None)
                setattr(translator, "_video_recent_text_cache", None)

            subtitle_frac = _region_fraction_from_preset(cfg)
            subtitle_roi_frac = subtitle_frac if subtitle_frac > 0 else 1.0  # 1.0 => whole frame (preset "full")
            scene_roi_frac = subtitle_roi_frac
            two_stage_force_every_frames = int(getattr(cfg, "video_translator_two_stage_force_refresh_every_frames", 0) or 0)
            two_stage_new_line_diff_threshold = float(getattr(cfg, "video_translator_two_stage_new_line_diff_threshold", 8.0) or 8.0)
            adaptive_detector_roi = bool(getattr(cfg, "video_translator_adaptive_detector_roi", False))
            adaptive_detector_roi_padding_frac = float(getattr(cfg, "video_translator_adaptive_detector_roi_padding_frac", 0.15) or 0.15)
            adaptive_detector_roi_start_seconds = max(
                0.0,
                float(getattr(cfg, "video_translator_adaptive_detector_roi_start_seconds", 0.0) or 0.0),
            )
            auto_catch_subtitle_on_skipped_frames = bool(
                getattr(cfg, "video_translator_auto_catch_subtitle_on_skipped_frames", True)
            )
            auto_catch_diff_threshold = float(
                getattr(cfg, "video_translator_auto_catch_diff_threshold", 0.0) or 0.0
            )
            prev_auto_catch_small_gray = None

            frac = _region_fraction_from_preset(cfg) if self.temporal_smoothing else 0.0
            if frac <= 0:
                frac = 0.2

            # Frame pacing: one frame per iteration, one write_frame() — output frame count
            # matches input. Decode and encode run in separate threads so the pipeline is never blocked by I/O.
            while True:
                if self._cancel:
                    break
                item = frame_queue.get()
                ret, frame = item[0], item[1]
                n = item[2] if len(item) >= 3 else n
                if not ret or frame is None:
                    break
                curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
                pending_two_stage_hash = None
                pending_two_stage_small_gray = None
                # Run pipeline every sample_every_frames, or on scene change (throttled: at least sample_every_frames since last run)
                periodic = n % self.sample_every_frames == 0
                if self.use_scene_detection:
                    scene_changed = _scene_change(prev_gray, curr_gray, self.scene_threshold, bottom_frac=scene_roi_frac)
                    run_pipeline = periodic or (scene_changed and (n - last_pipeline_n) >= self.sample_every_frames)
                else:
                    run_pipeline = periodic
                prev_gray = curr_gray
                if (
                    not run_pipeline
                    and auto_catch_subtitle_on_skipped_frames
                    and self.enable_detect
                    and self.enable_ocr
                    and detector is not None
                ):
                    try:
                        small_gray = _subtitle_region_small_gray(frame, subtitle_roi_frac)
                        if small_gray is not None:
                            if prev_auto_catch_small_gray is not None:
                                import numpy as np
                                # Auto threshold scales with sampling interval (sample_every/fps):
                                # larger intervals need more aggressive "catch first appearance".
                                if fps > 0:
                                    interval_s = float(self.sample_every_frames) / float(fps)
                                else:
                                    interval_s = float(self.sample_every_frames) / 24.0
                                auto_thr = max(3.5, min(18.0, 4.0 + interval_s * 6.0))
                                thr = auto_catch_diff_threshold if auto_catch_diff_threshold > 0 else auto_thr
                                diff = float(
                                    np.mean(
                                        np.abs(
                                            small_gray.astype(np.int16)
                                            - prev_auto_catch_small_gray.astype(np.int16)
                                        )
                                    )
                                )
                                if diff >= thr:
                                    run_pipeline = True
                            prev_auto_catch_small_gray = small_gray
                    except Exception:
                        pass
                # Two-stage keyframes: skip full pipeline when subtitle region content hash unchanged,
                # with an additional "new line" pixel-diff guard + optional forced refresh.
                if run_pipeline and self.two_stage_keyframes:
                    curr_small_gray = _subtitle_region_small_gray(frame, subtitle_roi_frac)
                    if curr_small_gray is not None:
                        import hashlib
                        curr_hash = hashlib.md5(curr_small_gray.tobytes()).digest()

                        forced_due = (
                            two_stage_force_every_frames > 0
                            and last_pipeline_n >= 0
                            and (n - last_pipeline_n) >= two_stage_force_every_frames
                        )

                        should_skip = False
                        if last_content_hash is not None and curr_hash == last_content_hash and not forced_due:
                            if two_stage_new_line_diff_threshold > 0 and last_band_small_gray is not None:
                                import numpy as np
                                diff = float(
                                    np.mean(
                                        np.abs(
                                            curr_small_gray.astype(np.int16)
                                            - last_band_small_gray.astype(np.int16)
                                        )
                                    )
                                )
                                should_skip = diff <= two_stage_new_line_diff_threshold
                            else:
                                should_skip = True

                        if should_skip:
                            run_pipeline = False
                        else:
                            pending_two_stage_hash = curr_hash
                            pending_two_stage_small_gray = curr_small_gray

                if run_pipeline:
                    last_pipeline_n = n
                    try:
                        if translator is not None and hasattr(translator, "set_translation_context"):
                            translator.set_translation_context(
                                previous_pages=[
                                    {
                                        "sources": [s for s in (e.get("sources") or []) if str(s or "").strip()],
                                        "translations": [
                                            t
                                            for t in (e.get("translations") or [])
                                            if str(t or "").strip() and not _is_invalid_subtitle_text(str(t or "").strip())
                                        ],
                                    }
                                    for e in (video_previous_subtitles or [])
                                    if isinstance(e, dict)
                                    and any(
                                        str(t or "").strip() and not _is_invalid_subtitle_text(str(t or "").strip())
                                        for t in (e.get("translations") or [])
                                    )
                                ],
                                series_context_path=sp if sp else None,
                            )
                        # When inpaint_only_soft_subs, do not draw subtitles on frames (video gets SRT only)
                        draw_subs = not self.inpaint_only_soft_subs
                        # Single draw path: never draw in pipeline; always draw once in dialog so we never get double/layered subtitles
                        detect_roi_xyxy = None
                        curr_seconds = (float(n) / float(fps)) if fps and float(fps) > 0.0 else 0.0
                        adaptive_roi_ready = curr_seconds >= adaptive_detector_roi_start_seconds
                        if (
                            adaptive_detector_roi
                            and adaptive_roi_ready
                            and not self.skip_detect
                            and self.enable_detect
                            and detector is not None
                            and cached_blk_list
                        ):
                            try:
                                # Crop detector to a padded union of last subtitle boxes
                                roi_x1, roi_y1, roi_x2, roi_y2 = w, h, 0, 0
                                for b in cached_blk_list:
                                    xyxy = getattr(b, "xyxy", None)
                                    if not xyxy or len(xyxy) < 4:
                                        continue
                                    x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
                                    roi_x1 = min(roi_x1, x1)
                                    roi_y1 = min(roi_y1, y1)
                                    roi_x2 = max(roi_x2, x2)
                                    roi_y2 = max(roi_y2, y2)
                                if roi_x2 > roi_x1 and roi_y2 > roi_y1:
                                    roi_x1 = int(max(0, roi_x1))
                                    roi_y1 = int(max(0, roi_y1))
                                    roi_x2 = int(min(w, roi_x2))
                                    roi_y2 = int(min(h, roi_y2))
                                    # Clamp to subtitle band when ROI preset is not "full"
                                    if subtitle_roi_frac < 1.0:
                                        band_y_min = int(h * (1.0 - subtitle_roi_frac))
                                        roi_y1 = max(roi_y1, band_y_min)
                                    pad_x = int((roi_x2 - roi_x1) * adaptive_detector_roi_padding_frac)
                                    pad_y = int((roi_y2 - roi_y1) * adaptive_detector_roi_padding_frac)
                                    roi_x1 = max(0, roi_x1 - pad_x)
                                    roi_y1 = max(0, roi_y1 - pad_y)
                                    roi_x2 = min(w, roi_x2 + pad_x)
                                    roi_y2 = min(h, roi_y2 + pad_y)
                                    roi_w = roi_x2 - roi_x1
                                    roi_h = roi_y2 - roi_y1
                                    # Avoid tiny ROIs that might miss new subtitles
                                    if roi_w >= int(w * 0.15) and roi_h >= int(h * 0.07):
                                        detect_roi_xyxy = [roi_x1, roi_y1, roi_x2, roi_y2]
                            except Exception:
                                detect_roi_xyxy = None
                        use_flow_fixer_for_corrections = bool(getattr(cfg, "video_translator_use_flow_fixer_for_corrections", False))
                        out_frame, blk_list, _ = run_one_frame_pipeline(
                            frame,
                            detector, ocr, translator, inpainter,
                            self.enable_detect, self.enable_ocr, self.enable_translate, self.enable_inpaint,
                            cfg=cfg,
                            skip_detect=self.skip_detect,
                            draw_subtitles=False,
                            detect_roi_xyxy=detect_roi_xyxy,
                            frame_index=n,
                            flow_fixer=flow_fixer,
                            use_flow_fixer_for_corrections=use_flow_fixer_for_corrections,
                        )
                        # Snapshot inpainted frame before we draw text (for srt-based burn-in with stacking).
                        cached_inpainted = out_frame.copy() if out_frame is not None else None
                        did_append = False
                        if translator is not None and blk_list:
                            new_entry = {
                                "sources": [b.get_text() if hasattr(b, "get_text") else (b.text or [""])[0] for b in blk_list],
                                "translations": [str(getattr(b, "translation", "") or "").strip() for b in blk_list],
                            }
                            # Only append when subtitle content changed (avoid inflating n_prev with same cached line on multiple frames)
                            last = video_previous_subtitles[-1] if video_previous_subtitles else None
                            did_append = last is None or last.get("sources") != new_entry["sources"] or last.get("translations") != new_entry["translations"]
                            if did_append:
                                video_previous_subtitles.append(new_entry)
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
                                # IMPORTANT:
                                # Do NOT write revised_previous into srt_entries by index.
                                # srt_entries merges consecutive same-text segments, so its indices do not map 1:1
                                # with video_previous_subtitles. Index-updating can put flow-fixer versions on the
                                # wrong cue (1-frame pop / wrong-frame "dominance").
                                # Update translation cache so future reuse uses revised text (same key as pipeline: segment + texts)
                                cache = getattr(translator, "_video_frame_cache", None)
                                if cache is not None:
                                    segment_size = 300
                                    seg_idx = n // segment_size
                                    for i in range(k):
                                        seg = video_previous_subtitles[-k + i]
                                        srcs = seg.get("sources") or []
                                        if i < len(rev) and rev[i] and srcs:
                                            cache[(seg_idx, tuple(srcs))] = [rev[i].strip()]
                                setattr(translator, "_last_revised_previous", None)
                        # Flow fixer: only run after actual translation (not when reused cached translations)
                        if flow_fixer is not None and blk_list and getattr(flow_fixer, "improve_flow", None) and did_append:
                            try:
                                new_translations = [str(getattr(b, "translation", "") or "").strip() for b in blk_list]
                                # Previous entries = earlier frames only (we just appended so exclude current)
                                if video_previous_subtitles:
                                    previous_entries = video_previous_subtitles[-flow_fixer_context:-1]
                                else:
                                    previous_entries = []
                                key_prev = tuple(
                                    [" ".join((e.get("translations") or [])).strip() for e in previous_entries]
                                )
                                key_new = tuple(new_translations)
                                cache_key = ("v1", key_prev, key_new, "en")
                                cached = flow_fixer_cache.get(cache_key)
                                strict_single = bool(getattr(cfg, "video_translator_flow_fixer_strict_single_line_review", False))
                                if cached is not None:
                                    cached_prev, cached_new = cached
                                    revised_prev = [dict(e) for e in cached_prev] if cached_prev is not None else None
                                    revised_new = list(cached_new) if cached_new is not None else None
                                elif (not strict_single) and (not previous_entries) and len(new_translations) == 1:
                                    # No context to smooth; skip API call and keep translation as-is.
                                    revised_prev, revised_new = None, list(new_translations)
                                else:
                                    revised_prev, revised_new = flow_fixer.improve_flow(
                                        [dict(e) for e in previous_entries],
                                        new_translations,
                                        target_lang="en",
                                    )
                                # Store in cache when we computed (including skip-case) and not already cached
                                if cached is None and revised_new is not None:
                                    try:
                                        flow_fixer_cache[cache_key] = (
                                            [dict(e) for e in revised_prev] if revised_prev is not None else None,
                                            list(revised_new) if revised_new is not None else None,
                                        )
                                        flow_fixer_cache_order.append(cache_key)
                                        if len(flow_fixer_cache_order) > flow_fixer_cache_max:
                                            old_key = flow_fixer_cache_order.pop(0)
                                            flow_fixer_cache.pop(old_key, None)
                                    except Exception:
                                        pass
                                # Rule-based continuity fix when flow fixer returns unchanged text (e.g. local model echoes)
                                if revised_new and len(revised_new) == len(blk_list):
                                    comma_prev, comma_new = _apply_continuation_comma_fix(
                                        revised_prev, revised_new, previous_entries, video_previous_subtitles, srt_entries
                                    )
                                    if comma_prev or comma_new:
                                        LOGGER.info(
                                            "Continuation comma fix applied: %d previous, %d new line(s)",
                                            comma_prev, comma_new,
                                        )
                                if revised_new and len(revised_new) == len(blk_list):
                                    n_new_changed = 0
                                    new_changes = []  # list of (index, old, new) for logging
                                    for i, b in enumerate(blk_list):
                                        if i < len(revised_new):
                                            orig = (getattr(b, "translation", None) or "").strip()
                                            new_text = (revised_new[i] or "").strip()
                                            changed = new_text != orig
                                            if changed:
                                                n_new_changed += 1
                                                new_changes.append((i, orig, new_text))
                                            setattr(b, "flow_fixed", changed)
                                            b.translation = revised_new[i]
                                        else:
                                            setattr(b, "flow_fixed", False)
                                    # revised_prev corresponds to previous_entries only
                                    n_prev_changed = 0
                                    prev_changes = []  # list of (index, old, new) for logging
                                    if revised_prev is not None and len(revised_prev) > 0:
                                        k = len(revised_prev)
                                        if k <= len(previous_entries):
                                            for i in range(k):
                                                old_t = " ".join((previous_entries[i].get("translations") or [])).strip()
                                                new_t = " ".join((revised_prev[i].get("translations") or [])).strip()
                                                if old_t != new_t:
                                                    n_prev_changed += 1
                                                    prev_changes.append((i, old_t, new_t))
                                        need = k + 1  # we appended so current is at -1
                                        if len(video_previous_subtitles) >= need:
                                            for i in range(k):
                                                video_previous_subtitles[-(k + 1) + i] = revised_prev[i]
                                            # Do not write revised_prev into srt_entries: srt_entries merges
                                            # same-text segments so its indices do not match video_previous_subtitles.
                                            # Updating by index would put revised text into the wrong segment and cause
                                            # "flow fixer version" of a subtitle to appear on wrong frames (one-frame pop).
                                            cache = getattr(translator, "_video_frame_cache", None) if translator else None
                                            if cache is not None:
                                                segment_size = 300
                                                seg_idx = n // segment_size
                                                for i in range(k):
                                                    seg = video_previous_subtitles[-(k + 1) + i]
                                                    srcs = seg.get("sources") or []
                                                    trans = (seg.get("translations") or [])
                                                    if srcs and trans:
                                                        cache[(seg_idx, tuple(srcs))] = [str(t).strip() for t in trans]
                                    if n_prev_changed or n_new_changed:
                                        LOGGER.info(
                                            "Flow fixer changed %d previous and %d new line(s)",
                                            n_prev_changed, n_new_changed,
                                        )
                                        def _tr(s, max_len=120):
                                            s = (s or "").strip()
                                            return s if len(s) <= max_len else s[: max_len - 3] + "..."
                                        for idx, old_t, new_t in prev_changes:
                                            LOGGER.info(
                                                "Flow fixer [prev %d]: %r -> %r",
                                                idx, _tr(old_t), _tr(new_t),
                                            )
                                        for idx, old_t, new_t in new_changes:
                                            LOGGER.info(
                                                "Flow fixer [new %d]: %r -> %r",
                                                idx, _tr(old_t), _tr(new_t),
                                            )
                            except Exception as e:
                                LOGGER.debug("Flow fixer failed: %s", e)
                        # Single draw: burn-in subtitles once here (same style for all; no second layer)
                        if draw_subs and blk_list:
                            # Stabilize position: when subtitle text is unchanged, reuse previous frame's box so text doesn't jitter.
                            # Match by translation (not index) so detector reordering doesn't assign wrong positions and cause "text thrown" glitches.
                            if cached_blk_list is not None and len(cached_blk_list) == len(blk_list):
                                current_texts = [(getattr(b, "translation", None) or "").strip() for b in blk_list]
                                cached_texts = [(getattr(c, "translation", None) or "").strip() for c in cached_blk_list]
                                if set(current_texts) == set(cached_texts) and len(current_texts) == len(set(current_texts)):
                                    used = set()
                                    for b in blk_list:
                                        bt = (getattr(b, "translation", None) or "").strip()
                                        for i, c in enumerate(cached_blk_list):
                                            if i in used:
                                                continue
                                            ct = (getattr(c, "translation", None) or "").strip()
                                            if bt == ct:
                                                xy = getattr(c, "xyxy", None)
                                                if xy is not None and len(xy) >= 4:
                                                    b.xyxy = list(xy)
                                                used.add(i)
                                                break
                            style = (getattr(cfg, "video_translator_subtitle_style", None) or "default").strip().lower()
                            if style not in ("anime", "documentary"):
                                style = "default"
                            bb_mode = bool(getattr(cfg, "video_translator_subtitle_black_box_mode", False))
                            try:
                                bb_pad = int(getattr(cfg, "video_translator_subtitle_black_box_padding", 6) or 0)
                            except (TypeError, ValueError):
                                bb_pad = 6
                            bb_pad = max(0, min(64, bb_pad))
                            bb_bgr = (
                                int(getattr(cfg, "video_translator_subtitle_black_box_b", 0)),
                                int(getattr(cfg, "video_translator_subtitle_black_box_g", 0)),
                                int(getattr(cfg, "video_translator_subtitle_black_box_r", 0)),
                            )
                            _draw_text_on_image(
                                out_frame,
                                blk_list,
                                style=style,
                                black_box_behind_text=bb_mode,
                                black_box_padding=bb_pad,
                                black_box_bgr=bb_bgr,
                            )
                        if self.temporal_smoothing and prev_result is not None:
                            out_frame = _temporal_blend(prev_result, out_frame, self.temporal_alpha, frac)
                        # If we got no blocks (or empty translations), don't wipe cached subtitles — keep last good.
                        has_text = False
                        try:
                            has_text = bool(blk_list) and any((getattr(b, "translation", None) or "").strip() for b in blk_list)
                        except Exception:
                            has_text = bool(blk_list)
                        hold_window = max(2, self.sample_every_frames * 2)
                        if has_text:
                            miss_streak = 0
                            cached_result = out_frame
                            cached_blk_list = blk_list
                            cached_source_frame = frame.copy()
                            last_good_cached_result = out_frame
                            last_good_cached_blk_list = blk_list
                            last_good_cached_source_frame = cached_source_frame
                            last_good_cached_inpainted = cached_inpainted
                            prev_result = out_frame.copy()
                            # Hold the cached subtitle state for a short time so brief detector dropouts don't flicker.
                            # cached_hold_until is inclusive (we stop compositing when n > cached_hold_until).
                            # Align with SRT semantics where segment end is exclusive: visible on [start, end-1].
                            cached_hold_until = n + hold_window - 1
                            if pending_two_stage_hash is not None:
                                last_content_hash = pending_two_stage_hash
                                last_band_small_gray = pending_two_stage_small_gray
                        else:
                            # If detection missed once, we can keep last good to avoid flicker.
                            # Keep last good for a few misses to avoid flicker when sample_every_frames is higher.
                            miss_streak += 1
                            keep_miss_limit = max(3, self.sample_every_frames + 1)
                            if miss_streak <= keep_miss_limit and last_good_cached_result is not None:
                                cached_result = last_good_cached_result
                                cached_blk_list = last_good_cached_blk_list
                                cached_source_frame = last_good_cached_source_frame
                                cached_inpainted = last_good_cached_inpainted
                                prev_result = last_good_cached_result.copy() if hasattr(last_good_cached_result, "copy") else last_good_cached_result
                                # Also extend compositing TTL; otherwise subtitles can flicker off briefly
                                # when we reuse cached subtitles after a detector miss.
                                cached_hold_until = max(
                                    cached_hold_until,
                                    n + hold_window - 1,
                                )
                            else:
                                # After many misses, stop compositing — but keep last_good_* so we can still fall back when writing (avoids raw flash).
                                cached_result = out_frame
                                cached_blk_list = []  # stop compositing old subtitle regions
                                cached_source_frame = None
                                cached_inpainted = None
                                prev_result = out_frame.copy()
                                last_cache_clear_frame = n
                        if (self.export_srt or self.soft_subs_only or self.inpaint_only_soft_subs) and blk_list:
                            parts = []
                            for b in blk_list:
                                t = (getattr(b, "translation", None) or "").strip()
                                if t and not _is_invalid_subtitle_text(t):
                                    parts.append(t)
                            text = " ".join(parts).strip()
                            if text:
                                if srt_entries and srt_entries[-1][2] == text:
                                    # Same text as last segment: just extend its end frame
                                    srt_entries[-1] = (srt_entries[-1][0], n + self.sample_every_frames, srt_entries[-1][2])
                                else:
                                    # Close previous segment at this frame, start a new one
                                    if srt_entries:
                                        srt_entries[-1] = (srt_entries[-1][0], n, srt_entries[-1][2])
                                        try:
                                            LOGGER.info(
                                                "OCR subtitle ended @ %.3fs (frames %d-%d): %r",
                                                n / fps if fps > 0 else 0.0,
                                                int(srt_entries[-1][0]),
                                                int(srt_entries[-1][1]),
                                                srt_entries[-1][2],
                                            )
                                        except Exception:
                                            pass
                                    srt_entries.append((n, n + self.sample_every_frames, text))
                                    try:
                                        LOGGER.info(
                                            "OCR subtitle started @ %.3fs (frames %d-%d): %r",
                                            n / fps if fps > 0 else 0.0,
                                            int(srt_entries[-1][0]),
                                            int(srt_entries[-1][1]),
                                            srt_entries[-1][2],
                                        )
                                    except Exception:
                                        pass
                        current_preview_texts = [(b.get_text() if hasattr(b, "get_text") else (b.text or [""])[0], (getattr(b, "translation", None) or "").strip()) for b in (blk_list or [])]
                        self.frame_preview_updated.emit(n, out_frame.copy(), current_preview_texts)
                    except Exception as e:
                        LOGGER.warning("Frame %d pipeline failed: %s", n, e)
                        LOGGER.exception("Frame %d pipeline traceback", n)
                        # On failure, keep last good cached result if any, so subtitles don't blink off.
                        if last_good_cached_result is not None:
                            cached_result = last_good_cached_result
                            cached_blk_list = last_good_cached_blk_list
                            cached_source_frame = last_good_cached_source_frame
                            cached_inpainted = last_good_cached_inpainted
                            prev_result = last_good_cached_result.copy() if hasattr(last_good_cached_result, "copy") else last_good_cached_result
                        else:
                            cached_result = frame.copy()
                            prev_result = cached_result.copy()

                elif cached_result is not None and n % preview_throttle == 0:
                    # Preview should match what we write (composited/validated), otherwise it can appear to flicker.
                    try:
                        if (not run_pipeline) and (n > cached_hold_until):
                            self.frame_preview_updated.emit(n, frame.copy(), current_preview_texts)
                        elif (not run_pipeline) and cached_blk_list is not None:
                            prev_comp = _composite_cached_subs_on_frame(
                                frame, cached_result, cached_source_frame, cached_blk_list, h, w
                            )
                            self.frame_preview_updated.emit(n, (prev_comp if prev_comp is not None else cached_result).copy(), current_preview_texts)
                        else:
                            self.frame_preview_updated.emit(n, cached_result.copy(), current_preview_texts)
                    except Exception:
                        self.frame_preview_updated.emit(n, cached_result.copy(), current_preview_texts)

                if self.soft_subs_only:
                    # Original video only; subs in SRT/ASS/VTT (no inpainting, no burn-in)
                    write_frame(n, frame)
                elif self.inpaint_only_soft_subs and cached_result is not None:
                    # Inpainted video, no text drawn; subs in SRT/ASS/VTT only (no double subs in video)
                    if cached_result.shape[1] != w or cached_result.shape[0] != h:
                        cached_result = cv2.resize(cached_result, (w, h), interpolation=cv2.INTER_LINEAR)
                    write_frame(n, cached_result)
                elif cached_result is not None:
                    prefer_timed_burn_in = bool(
                        getattr(cfg, "video_translator_prefer_timed_burn_in", True)
                    )
                    # Full-fps motion: on non-pipeline frames paste cached subtitle regions onto current frame.
                    # Extend cache TTL when we use it so we don't flicker (subs off/on) when pipeline is skipped at keyframes (e.g. two-stage).
                    if not run_pipeline and cached_blk_list is not None:
                        cached_hold_until = max(
                            cached_hold_until,
                            n + max(2, self.sample_every_frames * 2) - 1,
                        )
                    # When we would write raw (no subs), use last good composited for a short fallback to avoid jitter.
                    # Keep stale subtitles only for about one "segment window" so wrong/old text can't leak.
                    fallback_hold = max(2, self.sample_every_frames * 2)
                    if prefer_timed_burn_in and srt_entries:
                        # Deterministic two-step burn-in:
                        # 1) build cue timeline (srt_entries) from OCR/translation keyframes
                        # 2) render active timed cues per frame (stack overlaps) on top of inpainted base
                        # This avoids cached text-region priority fights and one-frame text pops.
                        overlap_frames = min(15, max(5, int(fps * 0.5)))
                        segments_for_draw = []
                        for (start_f, end_f, text) in srt_entries:
                            if start_f <= n < end_f:
                                segments_for_draw.append((start_f / fps, end_f / fps, text))
                            elif end_f == n and (end_f - start_f) <= overlap_frames:
                                segments_for_draw.append((start_f / fps, (n + 1) / fps, text))
                        base = frame.copy()
                        if cached_blk_list and (run_pipeline or n <= cached_hold_until):
                            comp = _composite_cached_subs_on_frame(
                                frame,
                                cached_inpainted if cached_inpainted is not None else cached_result,
                                cached_source_frame,
                                cached_blk_list,
                                h,
                                w,
                            )
                            if comp is not None:
                                base = comp
                        elif (
                            (not run_pipeline)
                            and (n > cached_hold_until)
                            and last_good_cached_blk_list
                            and last_good_cached_result is not None
                            and (n <= last_cache_clear_frame + fallback_hold)
                        ):
                            comp = _composite_cached_subs_on_frame(
                                frame,
                                last_good_cached_inpainted if last_good_cached_inpainted is not None else last_good_cached_result,
                                last_good_cached_source_frame,
                                last_good_cached_blk_list,
                                h,
                                w,
                            )
                            if comp is not None:
                                base = comp
                        _draw_timed_subs_on_image(
                            base,
                            n / fps,
                            segments_for_draw,
                            stack_multiple_lines=True,
                            **subtitle_black_box_draw_kwargs_from_cfg(cfg),
                        )
                        write_frame(n, base)
                    elif (
                        (not run_pipeline)
                        and (n > cached_hold_until)
                        and last_good_cached_blk_list
                        and last_good_cached_result is not None
                        and (n <= last_cache_clear_frame + fallback_hold)
                    ):
                        comp = _composite_cached_subs_on_frame(
                            frame,
                            last_good_cached_result,
                            last_good_cached_source_frame,
                            last_good_cached_blk_list,
                            h,
                            w,
                        )
                        if comp is not None:
                            cached_hold_until = max(
                                cached_hold_until,
                                n + max(2, self.sample_every_frames * 2) - 1,
                            )
                            write_frame(n, comp)
                        else:
                            write_frame(n, frame)
                    # Stop pasting after TTL to avoid "stuck inpaint" when subtitles disappear/move.
                    elif (not run_pipeline) and (n > cached_hold_until):
                        write_frame(n, frame)
                    elif srt_entries and (cached_inpainted is not None or cached_result is not None) and (run_pipeline or (cached_blk_list is not None and n <= cached_hold_until)):
                        # SRT-based burn-in: only draw segments active at this frame (by integer frame index) to avoid wrong-frame/float-boundary subtitles.
                        overlap_frames = min(15, max(5, int(fps * 0.5)))
                        segments_for_draw = []
                        for (start_f, end_f, text) in srt_entries:
                            # Active at frame n: start_f <= n < end_f, or short segment ending exactly at n (extend by 1 frame for overlap).
                            if start_f <= n < end_f:
                                segments_for_draw.append((start_f / fps, end_f / fps, text))
                            elif end_f == n and (end_f - start_f) <= overlap_frames:
                                segments_for_draw.append((start_f / fps, (n + 1) / fps, text))
                        base = frame.copy()
                        if cached_blk_list:
                            comp = _composite_cached_subs_on_frame(
                                frame,
                                cached_inpainted if cached_inpainted is not None else cached_result,
                                cached_source_frame,
                                cached_blk_list,
                                h,
                                w,
                            )
                            if comp is not None:
                                base = comp
                        _draw_timed_subs_on_image(
                            base,
                            n / fps,
                            segments_for_draw,
                            stack_multiple_lines=True,
                            **subtitle_black_box_draw_kwargs_from_cfg(cfg),
                        )
                        write_frame(n, base)
                    elif not run_pipeline and cached_blk_list is not None:
                        comp = _composite_cached_subs_on_frame(
                            frame, cached_result, cached_source_frame, cached_blk_list, h, w
                        )
                        if comp is not None:
                            write_frame(n, comp)
                        else:
                            # Fallback: never write the full cached frame (it freezes motion and looks like low FPS).
                            write_frame(n, frame)
                    else:
                        if cached_result.shape[1] != w or cached_result.shape[0] != h:
                            cached_result = cv2.resize(cached_result, (w, h), interpolation=cv2.INTER_LINEAR)
                        write_frame(n, cached_result)
                else:
                    write_frame(n, frame)

                # Throttle progress for long videos to avoid UI overload (e.g. 17h = 1.5M+ frames)
                if total > 0:
                    step = 25 if total <= 10000 else (50 if total <= 100000 else 200)
                    if n % step == 0 or n == total:
                        self.progress.emit(n, total)
                elif total <= 0 and n % 30 == 0:
                    self.progress.emit(n, 0)

            reader_thread.join(timeout=10)
            try:
                write_queue.put((None, None), block=True, timeout=60)
                writer_thread.join(timeout=max(180, (total or 0) // 300))
            except Exception:
                pass
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

            # Optional post-run global subtitle flow review (timeline-level), applied before final sidecar write.
            post_review_fixer = None
            try:
                from modules.flow_fixer.translator_flow_fixer import resolve_post_review_flow_fixer

                post_review_fixer = resolve_post_review_flow_fixer(cfg, translator, flow_fixer)
            except Exception as e:
                LOGGER.debug("Post-review flow fixer resolve failed: %s", e)
            if srt_entries and post_review_fixer is not None:
                post_review_enabled = bool(getattr(cfg, "video_translator_post_review_enabled", True))
                post_review_on_cancel = bool(getattr(cfg, "video_translator_post_review_apply_on_cancel", True))
                can_run = post_review_enabled and ((not self._cancel) or post_review_on_cancel)
                if can_run:
                    try:
                        from modules.flow_fixer.corrections import improve_subtitle_timeline_via_fixer

                        original_texts = [str(e[2] or "").strip() for e in srt_entries]
                        if original_texts:
                            chunk_size = int(getattr(cfg, "video_translator_post_review_chunk_size", 80) or 80)
                            ctx_lines = int(getattr(cfg, "video_translator_post_review_context_lines", 20) or 20)
                            _tl = "en"
                            if translator is not None:
                                _tl = str(getattr(translator, "lang_target", None) or "en")
                            revised_texts = improve_subtitle_timeline_via_fixer(
                                flow_fixer=post_review_fixer,
                                timeline_texts=original_texts,
                                target_lang=_tl,
                                chunk_size=chunk_size,
                                context_lines=ctx_lines,
                            )
                            if revised_texts and len(revised_texts) == len(srt_entries):
                                n_changed = 0
                                updated = []
                                for i, (start_f, end_f, old_text) in enumerate(srt_entries):
                                    new_text = str(revised_texts[i] or "").strip()
                                    if not new_text:
                                        new_text = str(old_text or "").strip()
                                    if new_text != str(old_text or "").strip():
                                        n_changed += 1
                                    updated.append((start_f, end_f, new_text))
                                srt_entries = updated
                                if n_changed > 0:
                                    LOGGER.info(
                                        "Post-run subtitle flow review changed %d/%d segment(s).",
                                        n_changed,
                                        len(srt_entries),
                                    )
                    except Exception as e:
                        LOGGER.debug("Post-run subtitle flow review failed: %s", e)

            # Only write sidecar SRT/ASS/VTT when not burning in (avoids double subtitles when player auto-loads SRT)
            if (self.soft_subs_only or self.inpaint_only_soft_subs) and srt_entries:
                # Last segment ends at frame n (exclusive): visible on [start, n-1]
                srt_entries[-1] = (srt_entries[-1][0], n, srt_entries[-1][2])
                base_path = osp.splitext(self.output_path)[0]
                srt_path = base_path + ".srt"
                _write_srt(srt_path, srt_entries, fps)
                if self.export_ass:
                    _write_ass(base_path + ".ass", srt_entries, fps, w, h)
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
        # Persist window state across hide/show (close button hides the dialog).
        self._vt_was_maximized: bool = False
        self._vt_saved_geometry = None
        self._vt_first_show_done: bool = False
        self._scroll = None
        self.thread: Optional[VideoTranslateThread] = None
        self._batch_jobs: list = []  # [(input_path, output_path), ...]
        self._batch_index: int = 0

        main_layout = QVBoxLayout(self)
        scroll = QScrollArea()
        self._scroll = scroll
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setMinimumHeight(400)
        content_widget = QWidget()
        content_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
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
        self.asr_lang_edit.setPlaceholderText(self.tr("ja, en, zh, ..."))
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
        self.check_sentence_merge_punct = QCheckBox(self.tr("Merge cues until sentence punctuation (rule-based)"))
        self.check_sentence_merge_punct.setChecked(bool(getattr(pcfg.module, "video_translator_sentence_merge_by_punctuation", True)))
        self.check_sentence_merge_punct.stateChanged.connect(self._save_options)
        self.check_sentence_merge_punct.setToolTip(self.tr("Merge short adjacent cues until sentence-ending punctuation (., ?, !, 。！？). Reduces half-sentence mistranslations and request count without extra LLM calls."))
        asr_row.addWidget(self.check_sentence_merge_punct, 5, 0, 1, 2)
        asr_row.addWidget(QLabel(self.tr("Max merge duration (sec):")), 6, 0)
        self.sentence_merge_max_sec_spin = QDoubleSpinBox()
        self.sentence_merge_max_sec_spin.setRange(0.5, 30.0)
        self.sentence_merge_max_sec_spin.setSingleStep(0.5)
        self.sentence_merge_max_sec_spin.setValue(float(getattr(pcfg.module, "video_translator_sentence_merge_max_seconds", 8.0)))
        self.sentence_merge_max_sec_spin.valueChanged.connect(self._save_options)
        asr_row.addWidget(self.sentence_merge_max_sec_spin, 6, 1)
        self.check_asr_audio_separation = QCheckBox(self.tr("Separate vocals (reduce music noise)"))
        self.check_asr_audio_separation.setChecked(bool(getattr(pcfg.module, "video_translator_asr_audio_separation", False)))
        self.check_asr_audio_separation.stateChanged.connect(self._save_options)
        self.check_asr_audio_separation.setToolTip(self.tr("Use demucs to separate vocals before ASR. Improves accuracy on noisy audio. Optional: pip install demucs."))
        asr_row.addWidget(self.check_asr_audio_separation, 7, 0, 1, 2)
        self.check_asr_guided_detect_inpaint = QCheckBox(self.tr("ASR-guided detect/inpaint (experimental)"))
        self.check_asr_guided_detect_inpaint.setChecked(bool(getattr(pcfg.module, "video_translator_asr_guided_detect_inpaint", False)))
        self.check_asr_guided_detect_inpaint.stateChanged.connect(self._save_options)
        self.check_asr_guided_detect_inpaint.setToolTip(self.tr("Use ASR timings to drive vision: detect once at each subtitle segment start and reuse boxes until segment end, then inpaint only during active segments."))
        asr_row.addWidget(self.check_asr_guided_detect_inpaint, 8, 0, 1, 2)
        self.check_asr_guided_midpoint_refresh = QCheckBox(self.tr("Midpoint refresh in ASR-guided mode"))
        self.check_asr_guided_midpoint_refresh.setChecked(bool(getattr(pcfg.module, "video_translator_asr_guided_midpoint_refresh", True)))
        self.check_asr_guided_midpoint_refresh.stateChanged.connect(self._save_options)
        self.check_asr_guided_midpoint_refresh.setToolTip(self.tr("Refresh text detection once around the middle of each active subtitle segment to handle moving subtitles."))
        asr_row.addWidget(self.check_asr_guided_midpoint_refresh, 9, 0, 1, 2)
        asr_row.addWidget(QLabel(self.tr("Long-audio chunk size (sec, 0=off):")), 10, 0)
        self.asr_chunk_seconds_spin = QDoubleSpinBox()
        self.asr_chunk_seconds_spin.setRange(0.0, 14400.0)
        self.asr_chunk_seconds_spin.setDecimals(0)
        self.asr_chunk_seconds_spin.setSingleStep(60.0)
        self.asr_chunk_seconds_spin.setSpecialValueText(self.tr("Off"))
        self.asr_chunk_seconds_spin.setValue(
            max(
                0.0,
                float(getattr(pcfg.module, "video_translator_asr_chunk_seconds", 2400.0) or 0.0),
            )
        )
        self.asr_chunk_seconds_spin.valueChanged.connect(self._save_options)
        self.asr_chunk_seconds_spin.setToolTip(
            self.tr(
                "When > 0 and duration >= the threshold below, ASR runs in time slices (seconds per slice). "
                "Helps very long files and enables checkpoint resume. 0 = transcribe whole extract in one pass."
            )
        )
        asr_row.addWidget(self.asr_chunk_seconds_spin, 10, 1)
        asr_row.addWidget(QLabel(self.tr("Chunking min duration (sec, 0=never):")), 11, 0)
        self.asr_long_audio_threshold_spin = QDoubleSpinBox()
        self.asr_long_audio_threshold_spin.setRange(0.0, 864000.0)
        self.asr_long_audio_threshold_spin.setDecimals(0)
        self.asr_long_audio_threshold_spin.setSingleStep(60.0)
        self.asr_long_audio_threshold_spin.setSpecialValueText(self.tr("Never"))
        self.asr_long_audio_threshold_spin.setValue(
            max(
                0.0,
                float(
                    getattr(
                        pcfg.module, "video_translator_asr_long_audio_threshold_seconds", 5400.0
                    )
                    or 0.0
                ),
            )
        )
        self.asr_long_audio_threshold_spin.valueChanged.connect(self._save_options)
        self.asr_long_audio_threshold_spin.setToolTip(
            self.tr(
                "Minimum audio length (seconds) before chunking activates. 0 = never chunk by duration. "
                "Requires chunk size > 0."
            )
        )
        asr_row.addWidget(self.asr_long_audio_threshold_spin, 11, 1)
        self.check_asr_checkpoint_resume = QCheckBox(
            self.tr("Save/resume ASR chunk checkpoint (temp JSON)")
        )
        self.check_asr_checkpoint_resume.setChecked(
            bool(getattr(pcfg.module, "video_translator_asr_checkpoint_resume", True))
        )
        self.check_asr_checkpoint_resume.stateChanged.connect(self._save_options)
        self.check_asr_checkpoint_resume.setToolTip(
            self.tr(
                "When chunking is active, progress is saved after each slice so a failed run can resume. "
                "Checkpoint is tied to the input video path and ASR settings."
            )
        )
        asr_row.addWidget(self.check_asr_checkpoint_resume, 12, 0, 1, 2)
        g2l.addWidget(self.asr_options_widget)
        # Usage preset: one-click profiles for different video types and speed/quality trade-offs
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel(self.tr("Usage preset:")))
        self.usage_preset_combo = QComboBox()
        self.usage_preset_combo.addItems([
            self.tr("Custom (no change)"),
            self.tr("Don't miss text (quality)"),
            self.tr("Balanced (recommended)"),
            self.tr("Maximum speed"),
            self.tr("Anime / long series"),
            self.tr("Documentary / captions"),
        ])
        _usage = (getattr(pcfg.module, "video_translator_usage_preset", None) or "balanced").strip().lower()
        _usage_to_idx = {"custom": 0, "dont_miss_text": 1, "balanced": 2, "max_speed": 3, "anime": 4, "documentary": 5}
        self.usage_preset_combo.blockSignals(True)
        self.usage_preset_combo.setCurrentIndex(_usage_to_idx.get(_usage, 2))
        self.usage_preset_combo.blockSignals(False)
        self.usage_preset_combo.setToolTip(self.tr(
            "Apply a bundle of settings for different use cases. Custom leaves your current settings unchanged. "
            "Don't miss text = catch every subtitle (slower). Balanced = good default. Maximum speed = fastest. "
            "Anime = long series with bottom subs. Documentary = captions anywhere on screen."
        ))
        self.usage_preset_combo.currentIndexChanged.connect(self._on_usage_preset_changed)
        preset_row.addWidget(self.usage_preset_combo, 1)
        g2l.addLayout(preset_row)
        row = QHBoxLayout()
        _process_every_label = QLabel(self.tr("Process every:") or "Process every:")
        _process_every_label.setToolTip(self.tr(
            "Run detect/OCR/translate/inpaint only every N frames; other frames reuse the last result. "
            "Output video FPS is always the same as input (every frame is written). "
            "Lower N = subtitles update more often but more work.\n"
            "At 30 fps: use 1 to never miss text (every frame); 2–3 to catch quickly with less cost (~15–10 runs/sec); 5–6 for a good balance (~6–5 runs/sec). "
            "Higher values (e.g. 24–30) are faster but short-lived subtitles may be missed; use temporal smoothing to reduce flicker."
        ) or "Run pipeline every N frames. At 30fps: 1=best, 2–3=catch quickly, 5–6=balance. Higher= faster but may miss short subs.")
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
        self.check_subtitle_black_box = QCheckBox(
            self.tr("Subtitle bar: solid box behind text (no inpaint)")
        )
        self.check_subtitle_black_box.setChecked(
            bool(getattr(pcfg.module, "video_translator_subtitle_black_box_mode", False))
        )
        self.check_subtitle_black_box.stateChanged.connect(self._save_options)
        self.check_subtitle_black_box.setToolTip(
            self.tr(
                "Skips inpainting. Draws a filled rectangle behind each detected subtitle using the same "
                "line wrap as burn-in, so multi-line translations expand the bar. Only the text area is covered, "
                "not the whole bottom band. Color/padding: config keys video_translator_subtitle_black_box_*."
            )
        )
        row2b = QHBoxLayout()
        row2b.addWidget(self.check_subtitle_black_box)
        row2b.addStretch()
        g2l.addLayout(row2b)
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
        self.codec_edit.setPlaceholderText(self.tr("mp4v"))
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
        self.style_combo.setToolTip(
            self.tr(
                "Burn-in only: font size/position on the video. SRT sidecar files do not carry these styles; "
                "use ASS export + a player that reads styling, or burn-in for a fixed look."
            )
        )
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
        self.check_soft_subs.setToolTip(
            self.tr(
                "Soft subs: original video pixels unchanged; timing + text go to SRT (plain text/timing only). "
                "The player chooses font/size/color. Fast path; not the same as burn-in style above."
            )
        )
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
        g4al.addWidget(QLabel(self.tr("Prefetch frames:")), 4, 0)
        self.prefetch_spin = QSpinBox()
        self.prefetch_spin.setRange(0, 4)
        self.prefetch_spin.setValue(int(getattr(pcfg.module, "video_translator_prefetch_frames", 2)))
        self.prefetch_spin.setSpecialValueText(self.tr("0 (off)"))
        self.prefetch_spin.setToolTip(self.tr("Decode next frames in a separate thread (2–3 recommended) so I/O doesn't block the pipeline. 0 = off. OCR path only."))
        self.prefetch_spin.valueChanged.connect(self._save_options)
        g4al.addWidget(self.prefetch_spin, 4, 1)
        self.check_two_stage = QCheckBox(self.tr("Two-stage keyframes (skip pipeline when subtitle region unchanged)"))
        self.check_two_stage.setChecked(bool(getattr(pcfg.module, "video_translator_two_stage_keyframes", False)))
        self.check_two_stage.stateChanged.connect(self._save_options)
        self.check_two_stage.setToolTip(self.tr("Cheap content hash of the subtitle band; skip full pipeline when unchanged. Saves work when the same subtitle holds across many frames. OCR path only."))
        g4al.addWidget(self.check_two_stage, 5, 0, 1, 2)
        self.check_background_writer = QCheckBox(self.tr("Background writer (encode in separate thread)"))
        self.check_background_writer.setChecked(bool(getattr(pcfg.module, "video_translator_background_writer", True)))
        self.check_background_writer.stateChanged.connect(self._save_options)
        self.check_background_writer.setToolTip(self.tr("Saved for compatibility. OCR path always runs decode and encode in separate threads so the pipeline is never throttled by I/O."))
        g4al.addWidget(self.check_background_writer, 5, 2, 1, 2)
        self.check_two_pass_ocr_burn_in = QCheckBox(self.tr("Two-pass OCR burn-in (faster when translation is slow)"))
        self.check_two_pass_ocr_burn_in.setChecked(bool(getattr(pcfg.module, "video_translator_use_two_pass_ocr_burn_in", True)))
        self.check_two_pass_ocr_burn_in.stateChanged.connect(self._save_options)
        self.check_two_pass_ocr_burn_in.setToolTip(self.tr("Pass 1 runs detect/OCR/inpaint while collecting translations asynchronously; pass 2 burns timed subtitles. Increases throughput on slow translators. OCR hardcoded burn-in mode only."))
        g4al.addWidget(self.check_two_pass_ocr_burn_in, 6, 2, 1, 2)

        g4al.addWidget(QLabel(self.tr("Forced refresh (frames):")), 6, 0)
        self.two_stage_force_spin = QSpinBox()
        self.two_stage_force_spin.setRange(0, 5000)
        self.two_stage_force_spin.setValue(int(getattr(pcfg.module, "video_translator_two_stage_force_refresh_every_frames", 0)))
        self.two_stage_force_spin.setToolTip(self.tr("0 = off. If enabled, never skip two-stage indefinitely: force a full pipeline run at least every N frames."))
        self.two_stage_force_spin.valueChanged.connect(self._save_options)
        g4al.addWidget(self.two_stage_force_spin, 6, 1)

        g4al.addWidget(QLabel(self.tr("New-line diff threshold:")), 7, 0)
        self.two_stage_newline_diff_spin = QDoubleSpinBox()
        self.two_stage_newline_diff_spin.setRange(0.0, 60.0)
        self.two_stage_newline_diff_spin.setSingleStep(0.5)
        self.two_stage_newline_diff_spin.setValue(float(getattr(pcfg.module, "video_translator_two_stage_new_line_diff_threshold", 8.0)))
        self.two_stage_newline_diff_spin.setToolTip(self.tr("Used with two-stage keyframes. If the subtitle band pixel-diff indicates a new line, we run the pipeline even when the hash matches. Higher = fewer forced runs."))
        self.two_stage_newline_diff_spin.valueChanged.connect(self._save_options)
        g4al.addWidget(self.two_stage_newline_diff_spin, 7, 1)

        self.check_adaptive_detector_roi = QCheckBox(self.tr("Adaptive detector ROI (crop around last subs)"))
        self.check_adaptive_detector_roi.setChecked(bool(getattr(pcfg.module, "video_translator_adaptive_detector_roi", False)))
        self.check_adaptive_detector_roi.stateChanged.connect(self._save_options)
        self.check_adaptive_detector_roi.setToolTip(self.tr("When enabled and ROI is known from the last pipeline run, the detector runs on a cropped region to speed it up."))
        g4al.addWidget(self.check_adaptive_detector_roi, 8, 0, 1, 2)
        self.check_auto_catch_subs = QCheckBox(self.tr("Auto-catch subtitle appearance on skipped frames"))
        self.check_auto_catch_subs.setChecked(bool(getattr(pcfg.module, "video_translator_auto_catch_subtitle_on_skipped_frames", True)))
        self.check_auto_catch_subs.stateChanged.connect(self._save_options)
        self.check_auto_catch_subs.setToolTip(self.tr("When sample_every_frames skips a frame, monitor subtitle-band pixel changes and force a full pipeline run if a new subtitle likely appeared."))
        g4al.addWidget(self.check_auto_catch_subs, 8, 2, 1, 2)

        g4al.addWidget(QLabel(self.tr("Detector ROI padding:")), 9, 0)
        self.adaptive_roi_padding_spin = QDoubleSpinBox()
        self.adaptive_roi_padding_spin.setRange(0.0, 0.5)
        self.adaptive_roi_padding_spin.setSingleStep(0.05)
        self.adaptive_roi_padding_spin.setValue(float(getattr(pcfg.module, "video_translator_adaptive_detector_roi_padding_frac", 0.15)))
        self.adaptive_roi_padding_spin.valueChanged.connect(self._save_options)
        g4al.addWidget(self.adaptive_roi_padding_spin, 9, 1)
        g4al.addWidget(QLabel(self.tr("Adaptive ROI start (seconds):")), 10, 0)
        self.adaptive_roi_start_seconds_spin = QDoubleSpinBox()
        self.adaptive_roi_start_seconds_spin.setRange(0.0, 60.0)
        self.adaptive_roi_start_seconds_spin.setSingleStep(0.5)
        self.adaptive_roi_start_seconds_spin.setValue(float(getattr(pcfg.module, "video_translator_adaptive_detector_roi_start_seconds", 0.0)))
        self.adaptive_roi_start_seconds_spin.setToolTip(self.tr("Delay adaptive ROI until this time in the video. Useful to ignore early corner watermarks/registration text."))
        self.adaptive_roi_start_seconds_spin.valueChanged.connect(self._save_options)
        g4al.addWidget(self.adaptive_roi_start_seconds_spin, 10, 1)
        g4al.addWidget(QLabel(self.tr("Auto-catch diff threshold:")), 10, 2)
        self.auto_catch_diff_threshold_spin = QDoubleSpinBox()
        self.auto_catch_diff_threshold_spin.setRange(0.0, 30.0)
        self.auto_catch_diff_threshold_spin.setSingleStep(0.5)
        self.auto_catch_diff_threshold_spin.setValue(float(getattr(pcfg.module, "video_translator_auto_catch_diff_threshold", 0.0)))
        self.auto_catch_diff_threshold_spin.setSpecialValueText(self.tr("0 (auto)"))
        self.auto_catch_diff_threshold_spin.setToolTip(self.tr("0 = auto from sample_every_frames/fps. Higher values trigger fewer catch-up runs; lower values trigger more often."))
        self.auto_catch_diff_threshold_spin.valueChanged.connect(self._save_options)
        g4al.addWidget(self.auto_catch_diff_threshold_spin, 10, 3)

        self.check_overlap_inpaint = QCheckBox(self.tr("Overlap inpaint with OCR/translation"))
        self.check_overlap_inpaint.setChecked(bool(getattr(pcfg.module, "video_translator_overlap_inpaint", False)))
        self.check_overlap_inpaint.stateChanged.connect(self._save_options)
        self.check_overlap_inpaint.setToolTip(self.tr("Run inpainting in parallel with OCR/translation within the same pipeline tick (only when safe). May increase GPU contention."))
        g4al.addWidget(self.check_overlap_inpaint, 11, 0, 1, 2)
        self.check_ocr_temporal_stability = QCheckBox(self.tr("OCR temporal stability (reduce OCR jitter)"))
        self.check_ocr_temporal_stability.setChecked(bool(getattr(pcfg.module, "video_translator_ocr_temporal_stability", True)))
        self.check_ocr_temporal_stability.stateChanged.connect(self._save_options)
        self.check_ocr_temporal_stability.setToolTip(self.tr("Vote over recent frames for the same subtitle region before translation. Helps stop per-frame character flicker."))
        g4al.addWidget(self.check_ocr_temporal_stability, 12, 0, 1, 2)
        g4al.addWidget(QLabel(self.tr("OCR temporal window:")), 13, 0)
        self.ocr_temporal_window_spin = QSpinBox()
        self.ocr_temporal_window_spin.setRange(2, 12)
        self.ocr_temporal_window_spin.setValue(int(getattr(pcfg.module, "video_translator_ocr_temporal_window", 5)))
        self.ocr_temporal_window_spin.valueChanged.connect(self._save_options)
        g4al.addWidget(self.ocr_temporal_window_spin, 13, 1)
        g4al.addWidget(QLabel(self.tr("OCR temporal min votes:")), 14, 0)
        self.ocr_temporal_min_votes_spin = QSpinBox()
        self.ocr_temporal_min_votes_spin.setRange(2, 12)
        self.ocr_temporal_min_votes_spin.setValue(int(getattr(pcfg.module, "video_translator_ocr_temporal_min_votes", 2)))
        self.ocr_temporal_min_votes_spin.valueChanged.connect(self._save_options)
        g4al.addWidget(self.ocr_temporal_min_votes_spin, 14, 1)
        g4al.addWidget(QLabel(self.tr("OCR temporal geo quantization:")), 15, 0)
        self.ocr_temporal_geo_quant_spin = QSpinBox()
        self.ocr_temporal_geo_quant_spin.setRange(4, 64)
        self.ocr_temporal_geo_quant_spin.setValue(int(getattr(pcfg.module, "video_translator_ocr_temporal_geo_quantization", 24)))
        self.ocr_temporal_geo_quant_spin.valueChanged.connect(self._save_options)
        self.ocr_temporal_geo_quant_spin.setToolTip(self.tr("Region matching tolerance in pixels across frames. Higher = more stable matching; too high can over-merge nearby subtitle regions."))
        g4al.addWidget(self.ocr_temporal_geo_quant_spin, 15, 1)

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
        g4bl.addWidget(QLabel(self.tr("Encoding preset:")), 2, 2)
        self.ffmpeg_preset_combo = QComboBox()
        self.ffmpeg_preset_combo.addItems([
            "ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow", "placebo"
        ])
        _preset = (getattr(pcfg.module, "video_translator_ffmpeg_preset", None) or "medium").strip().lower()
        idx = self.ffmpeg_preset_combo.findText(_preset)
        self.ffmpeg_preset_combo.setCurrentIndex(idx if idx >= 0 else 5)  # 5 = medium
        self.ffmpeg_preset_combo.setToolTip(self.tr("Faster presets = quicker encoding, slightly larger file. veryfast/faster = good speed/quality trade-off."))
        self.ffmpeg_preset_combo.currentTextChanged.connect(self._save_options)
        g4bl.addWidget(self.ffmpeg_preset_combo, 2, 3)
        g4bl.addWidget(QLabel(self.tr("Hardware encoder:")), 3, 0)
        self.ffmpeg_hw_encoder_combo = QComboBox()
        self.ffmpeg_hw_encoder_combo.addItems([
            self.tr("None (libx264)"),
            self.tr("NVIDIA (NVENC)"),
            self.tr("Intel (QSV)"),
            self.tr("Auto (NVENC or QSV if available)"),
        ])
        _hw = (getattr(pcfg.module, "video_translator_ffmpeg_hw_encoder", None) or "none").strip().lower()
        self.ffmpeg_hw_encoder_combo.setCurrentIndex(
            3 if _hw == "auto" else (2 if _hw == "qsv" else (1 if _hw == "nvenc" else 0))
        )
        self.ffmpeg_hw_encoder_combo.setToolTip(self.tr("Use GPU encoder for faster encoding. Requires FFmpeg built with NVENC (NVIDIA) or QSV (Intel). Auto picks first available."))
        self.ffmpeg_hw_encoder_combo.currentIndexChanged.connect(self._save_options)
        g4bl.addWidget(self.ffmpeg_hw_encoder_combo, 3, 1, 1, 2)
        g4bl.addWidget(QLabel(self.tr("Target bitrate (kbps, 0=CRF only):")), 4, 0)
        self.video_bitrate_spin = QSpinBox()
        self.video_bitrate_spin.setRange(0, 50000)
        self.video_bitrate_spin.setValue(int(getattr(pcfg.module, "video_translator_video_bitrate_kbps", 0)))
        self.video_bitrate_spin.setSpecialValueText(self.tr("0 (CRF)"))
        self.video_bitrate_spin.setToolTip(self.tr("0 = use source video bitrate when detectable (same as original), else use CRF. Set a value (e.g. 9600) to override. Only when FFmpeg is enabled."))
        self.video_bitrate_spin.valueChanged.connect(self._save_options)
        g4bl.addWidget(self.video_bitrate_spin, 4, 1)
        self.check_skip_detect = QCheckBox(self.tr("Skip detection (fixed region only)"))
        self.check_skip_detect.setChecked(bool(getattr(pcfg.module, "video_translator_skip_detect", False)))
        self.check_skip_detect.setToolTip(self.tr("Use one block for subtitle region; no text detection. Set region above (e.g. Bottom 20%%)."))
        self.check_skip_detect.stateChanged.connect(self._on_skip_detect_changed)
        g4bl.addWidget(self.check_skip_detect, 5, 0, 1, 2)
        self.check_detect_no_inpaint = QCheckBox(self.tr("Detect-only mode (skip inpainting)"))
        self.check_detect_no_inpaint.setChecked(bool(getattr(pcfg.module, "video_translator_detect_no_inpaint", False)))
        self.check_detect_no_inpaint.setToolTip(self.tr("Run detection/OCR/translation normally, but do not run inpainting. Useful when you only want subtitle burn-in and no background cleanup."))
        self.check_detect_no_inpaint.stateChanged.connect(self._on_detect_no_inpaint_changed)
        g4bl.addWidget(self.check_detect_no_inpaint, 6, 0, 1, 2)
        self.check_bottom_band_native_mode = QCheckBox(self.tr("Bottom-band native mode (skip-detect)"))
        self.check_bottom_band_native_mode.setChecked(bool(getattr(pcfg.module, "video_translator_bottom_band_native_mode", False)))
        self.check_bottom_band_native_mode.stateChanged.connect(self._save_options)
        self.check_bottom_band_native_mode.setToolTip(self.tr("When Skip detection is on, use native bottom-band behavior for OCR/inpaint handling instead of fixed-block shortcuts."))
        g4bl.addWidget(self.check_bottom_band_native_mode, 5, 2, 1, 2)
        self.check_export_srt = QCheckBox(self.tr("Export SRT"))
        self.check_export_srt.setChecked(bool(getattr(pcfg.module, "video_translator_export_srt", False)))
        self.check_export_srt.stateChanged.connect(self._save_options)
        self.check_export_srt.setToolTip(
            self.tr(
                "SRT: standard sidecar format (cues + plain text). No advanced styling. "
                "Only when not burning in, to avoid double subtitles in players."
            )
        )
        g4bl.addWidget(self.check_export_srt, 7, 0, 1, 2)
        self.check_export_ass = QCheckBox(self.tr("Export ASS"))
        self.check_export_ass.setChecked(bool(getattr(pcfg.module, "video_translator_export_ass", False)))
        self.check_export_ass.stateChanged.connect(self._save_options)
        self.check_export_ass.setToolTip(
            self.tr(
                "ASS includes styling (font, size, colors). SRT/VTT are mostly timing + plain text. "
                "Use ASS if you need styles without burn-in."
            )
        )
        g4bl.addWidget(self.check_export_ass, 8, 0, 1, 2)
        self.check_export_vtt = QCheckBox(self.tr("Export WebVTT"))
        self.check_export_vtt.setChecked(bool(getattr(pcfg.module, "video_translator_export_vtt", False)))
        self.check_export_vtt.stateChanged.connect(self._save_options)
        self.check_export_vtt.setToolTip(self.tr("Write WebVTT subtitle file (same timing as SRT)."))
        g4bl.addWidget(self.check_export_vtt, 9, 0, 1, 2)
        self.subtitle_mode_note = QLabel(
            self.tr(
                "Note: Burn-in style (Default / Anime / Documentary) only affects text drawn on the video. "
                "Soft subtitles and exported SRT/ASS/VTT are separate: the player uses its own font for soft subs; "
                "ASS may include styling if the player supports it."
            )
        )
        self.subtitle_mode_note.setWordWrap(True)
        self.subtitle_mode_note.setObjectName("subtitleModeNote")
        g4bl.addWidget(self.subtitle_mode_note, 10, 0, 1, 3)
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
        self.check_llm_per_line_quality_fix = QCheckBox(self.tr("LLM per-line quality fix (empty/echo rescue)"))
        self.check_llm_per_line_quality_fix.setChecked(bool(getattr(pcfg.module, "video_translator_llm_per_line_quality_fix", True)))
        self.check_llm_per_line_quality_fix.stateChanged.connect(self._save_options)
        self.check_llm_per_line_quality_fix.setToolTip(self.tr("When enabled, retries problematic lines (empty or source-echo) with focused prompts. Improves quality but may add extra LLM calls."))
        g_ctx_l.addWidget(self.check_llm_per_line_quality_fix)
        nlp_batch_row = QHBoxLayout()
        nlp_batch_row.addWidget(QLabel(self.tr("LLM translation chunk size:")))
        self.nlp_chunk_spin = QSpinBox()
        self.nlp_chunk_spin.setRange(0, 500)
        self.nlp_chunk_spin.setMinimum(0)
        self.nlp_chunk_spin.setSpecialValueText(self.tr("Off (one request)"))
        self.nlp_chunk_spin.setValue(int(getattr(pcfg.module, "video_translator_nlp_chunk_size", 32) or 0))
        self.nlp_chunk_spin.setToolTip(
            self.tr(
                "Max subtitle lines per LLM API call for video ASR/existing subs and subtitle-file translate. "
                "0 = send all lines in one request (can be slow or hit limits). 20–40 is a practical range."
            )
        )
        self.nlp_chunk_spin.valueChanged.connect(self._save_options)
        nlp_batch_row.addWidget(self.nlp_chunk_spin)
        nlp_batch_row.addWidget(QLabel(self.tr("Parallel workers:")))
        self.nlp_max_workers_spin = QSpinBox()
        self.nlp_max_workers_spin.setRange(1, 16)
        self.nlp_max_workers_spin.setValue(max(1, int(getattr(pcfg.module, "video_translator_nlp_max_workers", 1) or 1)))
        self.nlp_max_workers_spin.setToolTip(
            self.tr(
                "When chunking produces multiple batches, run up to this many LLM requests at once. "
                "Increase only if your API allows concurrent calls (watch rate limits)."
            )
        )
        self.nlp_max_workers_spin.valueChanged.connect(self._save_options)
        nlp_batch_row.addWidget(self.nlp_max_workers_spin)
        nlp_batch_row.addStretch(1)
        g_ctx_l.addLayout(nlp_batch_row)
        self.check_qwen_aux_passes = QCheckBox(self.tr("Qwen3.5-4B: allow OCR-correction and reflection passes"))
        self.check_qwen_aux_passes.setChecked(bool(getattr(pcfg.module, "video_translator_qwen35_allow_aux_passes", False)))
        self.check_qwen_aux_passes.stateChanged.connect(self._save_options)
        self.check_qwen_aux_passes.setToolTip(self.tr("When enabled, do not skip Qwen3.5-4B LM Studio auxiliary passes (OCR correction/reflection). Can improve quality but may trigger verbose-thinking JSON failures and extra latency."))
        g_ctx_l.addWidget(self.check_qwen_aux_passes)
        self.check_lock_watermark_lines = QCheckBox(self.tr("Lock/ignore watermark-like lines by regex"))
        self.check_lock_watermark_lines.setChecked(bool(getattr(pcfg.module, "video_translator_lock_watermark_lines", False)))
        self.check_lock_watermark_lines.stateChanged.connect(self._save_options)
        self.check_lock_watermark_lines.setToolTip(self.tr("Do not translate lines matching the regex below (keeps source text). Useful for备案号/watermarks."))
        g_ctx_l.addWidget(self.check_lock_watermark_lines)
        g_ctx_l.addWidget(QLabel(self.tr("Watermark lock regex:")))
        self.lock_watermark_regex_edit = QLineEdit()
        self.lock_watermark_regex_edit.setText((getattr(pcfg.module, "video_translator_lock_watermark_regex", None) or r"备案号[:：]?\s*\d{8,}").strip())
        self.lock_watermark_regex_edit.setToolTip(self.tr("Python regex. Example: 备案号[:：]?\\s*\\d{8,}"))
        self.lock_watermark_regex_edit.textChanged.connect(self._save_options)
        g_ctx_l.addWidget(self.lock_watermark_regex_edit)
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
        self.flow_fixer_url_edit.setPlaceholderText(self.tr("http://localhost:1234/v1"))
        self.flow_fixer_url_edit.setText((getattr(pcfg.module, "video_translator_flow_fixer_server_url", None) or "http://localhost:1234/v1").strip())
        self.flow_fixer_url_edit.setToolTip(self.tr("LM Studio: http://localhost:1234/v1  —  Ollama: http://localhost:11434/v1. If you get invalid JSON or wrong revision count, see docs/FLOW_FIXER_SETUP.md (context size, max tokens, stop sequences)."))
        self.flow_fixer_url_edit.textChanged.connect(self._save_options)
        g_flow_l.addWidget(self.flow_fixer_url_edit, 2, 1)
        g_flow_l.addWidget(QLabel(self.tr("Model (local):")), 3, 0)
        self.flow_fixer_model_edit = QLineEdit()
        self.flow_fixer_model_edit.setPlaceholderText(self.tr("Type model name (required)"))
        self.flow_fixer_model_edit.setText((getattr(pcfg.module, "video_translator_flow_fixer_model", None) or "").strip())
        self.flow_fixer_model_edit.setToolTip(self.tr("Required: type the exact model name. LM Studio: use the name shown for the loaded model (e.g. llama-3.2-3b, Qwen2.5-3B-Instruct). Ollama: e.g. qwen2.5:3b, phi3:mini. Flow fixer is only used when this is non-empty; see docs/FLOW_FIXER_SETUP.md."))
        self.flow_fixer_model_edit.textChanged.connect(self._save_options)
        g_flow_l.addWidget(self.flow_fixer_model_edit, 3, 1)
        g_flow_l.addWidget(QLabel(self.tr("OpenRouter API key:")), 5, 0)
        self.flow_fixer_openrouter_apikey_edit = QLineEdit()
        self.flow_fixer_openrouter_apikey_edit.setPlaceholderText(self.tr("Same as translator or paste key"))
        self.flow_fixer_openrouter_apikey_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.flow_fixer_openrouter_apikey_edit.setText((getattr(pcfg.module, "video_translator_flow_fixer_openrouter_apikey", None) or "").strip())
        self.flow_fixer_openrouter_apikey_edit.setToolTip(self.tr("OpenRouter API key for the flow-fixer model. Can be the same key as your main translator."))
        self.flow_fixer_openrouter_apikey_edit.textChanged.connect(self._save_options)
        g_flow_l.addWidget(self.flow_fixer_openrouter_apikey_edit, 5, 1)
        g_flow_l.addWidget(QLabel(self.tr("OpenRouter model:")), 6, 0)
        self.flow_fixer_openrouter_model_edit = QLineEdit()
        self.flow_fixer_openrouter_model_edit.setPlaceholderText(self.tr("google/gemma-3n-e2b-it:free"))
        self.flow_fixer_openrouter_model_edit.setText((getattr(pcfg.module, "video_translator_flow_fixer_openrouter_model", None) or "google/gemma-3n-e2b-it:free").strip())
        self.flow_fixer_openrouter_model_edit.setToolTip(self.tr("Small/free model for flow only. Examples: google/gemma-3n-e2b-it:free, qwen/qwen3-4b:free"))
        self.flow_fixer_openrouter_model_edit.textChanged.connect(self._save_options)
        g_flow_l.addWidget(self.flow_fixer_openrouter_model_edit, 6, 1)
        g_flow_l.addWidget(QLabel(self.tr("OpenAI API key:")), 7, 0)
        self.flow_fixer_openai_apikey_edit = QLineEdit()
        self.flow_fixer_openai_apikey_edit.setPlaceholderText(self.tr("sk-... (platform.openai.com API key)"))
        self.flow_fixer_openai_apikey_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.flow_fixer_openai_apikey_edit.setText((getattr(pcfg.module, "video_translator_flow_fixer_openai_apikey", None) or "").strip())
        self.flow_fixer_openai_apikey_edit.setToolTip(self.tr("OpenAI API key (ChatGPT credits). Get one at platform.openai.com → API keys. Use gpt-4o-mini or gpt-3.5-turbo for cheap flow passes."))
        self.flow_fixer_openai_apikey_edit.textChanged.connect(self._save_options)
        g_flow_l.addWidget(self.flow_fixer_openai_apikey_edit, 7, 1)
        g_flow_l.addWidget(QLabel(self.tr("OpenAI model:")), 8, 0)
        self.flow_fixer_openai_model_edit = QLineEdit()
        self.flow_fixer_openai_model_edit.setPlaceholderText(self.tr("gpt-4o-mini"))
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
        self.check_flow_fixer_reasoning = QCheckBox(self.tr("Enable reasoning/thinking (if supported)"))
        self.check_flow_fixer_reasoning.setChecked(bool(getattr(pcfg.module, "video_translator_flow_fixer_enable_reasoning", False)))
        self.check_flow_fixer_reasoning.setToolTip(self.tr("Uses model reasoning controls for flow-fixer requests. If unsupported by a model/provider, it automatically retries without reasoning params."))
        self.check_flow_fixer_reasoning.stateChanged.connect(self._save_options)
        g_flow_l.addWidget(self.check_flow_fixer_reasoning, 10, 0, 1, 2)
        g_flow_l.addWidget(QLabel(self.tr("Reasoning effort:")), 11, 0)
        self.flow_fixer_reasoning_effort_combo = QComboBox()
        self.flow_fixer_reasoning_effort_combo.addItems(["low", "medium", "high"])
        _eff = (getattr(pcfg.module, "video_translator_flow_fixer_reasoning_effort", None) or "medium").strip().lower()
        self.flow_fixer_reasoning_effort_combo.setCurrentIndex(2 if _eff == "high" else (0 if _eff == "low" else 1))
        self.flow_fixer_reasoning_effort_combo.currentIndexChanged.connect(self._save_options)
        g_flow_l.addWidget(self.flow_fixer_reasoning_effort_combo, 11, 1)
        self.check_flow_fixer_for_corrections = QCheckBox(self.tr("Use flow fixer model for OCR/ASR correction and reflection"))
        self.check_flow_fixer_for_corrections.setChecked(bool(getattr(pcfg.module, "video_translator_use_flow_fixer_for_corrections", False)))
        self.check_flow_fixer_for_corrections.setToolTip(self.tr("When enabled, Correct OCR with LLM, Correct ASR with LLM, and Reflect translation use the same model as the flow fixer (local/OpenRouter/OpenAI) instead of the main translator."))
        self.check_flow_fixer_for_corrections.stateChanged.connect(self._save_options)
        g_flow_l.addWidget(self.check_flow_fixer_for_corrections, 12, 0, 1, 2)
        self.check_flow_fixer_strict_single = QCheckBox(self.tr("Strict flow fixer: always review single-line updates"))
        self.check_flow_fixer_strict_single.setChecked(bool(getattr(pcfg.module, "video_translator_flow_fixer_strict_single_line_review", False)))
        self.check_flow_fixer_strict_single.stateChanged.connect(self._save_options)
        self.check_flow_fixer_strict_single.setToolTip(self.tr("If enabled, flow fixer API is called even when there is no previous context and only one new line."))
        g_flow_l.addWidget(self.check_flow_fixer_strict_single, 12, 2, 1, 2)
        self.check_post_review = QCheckBox(self.tr("Post-run global subtitle flow review"))
        self.check_post_review.setChecked(bool(getattr(pcfg.module, "video_translator_post_review_enabled", True)))
        self.check_post_review.setToolTip(
            self.tr(
                "After translation, one pass over the full subtitle timeline for flow (pronouns, continuity, punctuation). "
                "Uses the dedicated flow-fixer model when configured; otherwise the main translator (LLM_API), if enabled below."
            )
        )
        self.check_post_review.stateChanged.connect(self._save_options)
        g_flow_l.addWidget(self.check_post_review, 13, 0, 1, 2)
        self.check_post_review_use_main_llm = QCheckBox(
            self.tr("Allow post-review via main translator when flow fixer is off")
        )
        self.check_post_review_use_main_llm.setChecked(
            bool(getattr(pcfg.module, "video_translator_post_review_use_main_translator", True))
        )
        self.check_post_review_use_main_llm.setToolTip(
            self.tr(
                "When no dedicated flow-fixer backend is set (or it is “none”), run the same timeline review using your "
                "normal translation model (extra API calls). Turn off to only review when a flow-fixer model is configured."
            )
        )
        self.check_post_review_use_main_llm.stateChanged.connect(self._save_options)
        g_flow_l.addWidget(self.check_post_review_use_main_llm, 14, 0, 1, 2)
        self.check_post_review_on_cancel = QCheckBox(self.tr("Apply post-run review on cancel (partial output)"))
        self.check_post_review_on_cancel.setChecked(bool(getattr(pcfg.module, "video_translator_post_review_apply_on_cancel", True)))
        self.check_post_review_on_cancel.stateChanged.connect(self._save_options)
        g_flow_l.addWidget(self.check_post_review_on_cancel, 15, 0, 1, 2)
        g_flow_l.addWidget(QLabel(self.tr("Post-review chunk size:")), 16, 0)
        self.post_review_chunk_spin = QSpinBox()
        self.post_review_chunk_spin.setRange(10, 500)
        self.post_review_chunk_spin.setValue(int(getattr(pcfg.module, "video_translator_post_review_chunk_size", 80)))
        self.post_review_chunk_spin.valueChanged.connect(self._save_options)
        g_flow_l.addWidget(self.post_review_chunk_spin, 16, 1)
        g_flow_l.addWidget(QLabel(self.tr("Post-review context lines:")), 17, 0)
        self.post_review_context_spin = QSpinBox()
        self.post_review_context_spin.setRange(1, 80)
        self.post_review_context_spin.setValue(int(getattr(pcfg.module, "video_translator_post_review_context_lines", 20)))
        self.post_review_context_spin.valueChanged.connect(self._save_options)
        g_flow_l.addWidget(self.post_review_context_spin, 17, 1)
        self.check_post_review_reasoning = QCheckBox(
            self.tr("Post-review / main-LLM flow pass: enable reasoning")
        )
        self.check_post_review_reasoning.setChecked(
            bool(getattr(pcfg.module, "video_translator_post_review_enable_reasoning", True))
        )
        self.check_post_review_reasoning.setToolTip(
            self.tr(
                "When the main translation model runs the timeline flow review, allow reasoning (thinking) tokens. "
                "Batched subtitle translation always runs without reasoning for reliable JSON."
            )
        )
        self.check_post_review_reasoning.stateChanged.connect(self._save_options)
        g_flow_l.addWidget(self.check_post_review_reasoning, 18, 0, 1, 2)
        g_flow_l.addWidget(QLabel(self.tr("Post-review reasoning effort:")), 19, 0)
        self.post_review_reasoning_effort_combo = QComboBox()
        self.post_review_reasoning_effort_combo.addItems(["low", "medium", "high"])
        _pre = (getattr(pcfg.module, "video_translator_post_review_reasoning_effort", None) or "medium").strip().lower()
        self.post_review_reasoning_effort_combo.setCurrentIndex(
            2 if _pre == "high" else (0 if _pre == "low" else 1)
        )
        self.post_review_reasoning_effort_combo.currentIndexChanged.connect(self._save_options)
        g_flow_l.addWidget(self.post_review_reasoning_effort_combo, 19, 1)
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
        self.apply_long_anime_btn = QPushButton(self.tr("Apply Anime preset"))
        self.apply_long_anime_btn.setToolTip(self.tr("Same as selecting 'Anime / long series' in the Usage preset dropdown above."))
        self.apply_long_anime_btn.clicked.connect(self._apply_recommended_long_anime)
        tips_row.addWidget(self.apply_long_anime_btn)
        content.addLayout(tips_row)

        # When Skip detection is on, Detection is not used
        self.check_detect.setEnabled(not self.check_skip_detect.isChecked())

        # Live preview (OCR pipeline: current frame + detected/translated subtitles)
        g_preview = QGroupBox(self.tr("Live preview (current frame)"))
        g_preview_l = QHBoxLayout(g_preview)
        # Left: frame preview + full screen button
        preview_frame_col = QVBoxLayout()
        self.preview_frame_label = QLabel()
        self.preview_frame_label.setMinimumSize(320, 180)
        self.preview_frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_frame_label.setStyleSheet("background-color: #1a1a1a; color: #666;")
        self.preview_frame_label.setText(self.tr("Frame and subtitles will appear here during OCR run"))
        self.preview_frame_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        preview_frame_col.addWidget(self.preview_frame_label, 1)
        self.preview_fullscreen_btn = QPushButton(self.tr("Full screen"))
        self.preview_fullscreen_btn.setToolTip(self.tr("Show current frame in full screen. Press Esc to close."))
        self.preview_fullscreen_btn.clicked.connect(self._show_preview_fullscreen)
        preview_frame_col.addWidget(self.preview_fullscreen_btn)
        g_preview_l.addLayout(preview_frame_col, 1)
        # Right: text preview + view history button
        preview_text_col = QVBoxLayout()
        preview_text_header = QHBoxLayout()
        preview_text_header.addWidget(QLabel(self.tr("Current frame translation:")))
        self.preview_history_btn = QPushButton(self.tr("View history…"))
        self.preview_history_btn.setToolTip(self.tr("Open a list of previous source text and translations so you can read how they flow together."))
        self.preview_history_btn.clicked.connect(self._show_preview_history)
        self.preview_ab_btn = QPushButton(self.tr("A/B compare models…"))
        self.preview_ab_btn.setToolTip(self.tr("Compare two translator models on sampled subtitle lines from this run."))
        self.preview_ab_btn.clicked.connect(self._show_ab_compare_dialog)
        preview_text_header.addStretch()
        preview_text_header.addWidget(self.preview_ab_btn)
        preview_text_header.addWidget(self.preview_history_btn)
        preview_text_col.addLayout(preview_text_header)
        self.preview_text_edit = QPlainTextEdit()
        self.preview_text_edit.setReadOnly(True)
        self.preview_text_edit.setUndoRedoEnabled(False)
        self.preview_text_edit.setMinimumWidth(200)
        self.preview_text_edit.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.preview_text_edit.setStyleSheet("color: #ccc; font-size: 0.9em; background: #202020;")
        self.preview_text_edit.setPlainText(self.tr("Detected / translated lines appear here during run."))
        preview_text_col.addWidget(self.preview_text_edit, 1)
        g_preview_l.addLayout(preview_text_col, 1)
        content.addWidget(g_preview)
        # Full-screen preview window (created on demand)
        self._preview_fullscreen_window = None
        # Subtitle history for "View history" (frame_index, list of (source, trans)); cleared when run starts
        self.preview_subtitle_history = []
        self._last_preview_subtitle_lines = None

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
        # Close should *not* cancel/stop a running job. We hide the dialog and keep state.
        self.close_btn.clicked.connect(self.close)
        btn_layout.addWidget(self.run_btn)
        btn_layout.addWidget(self.cancel_btn)
        btn_layout.addWidget(self.close_btn)
        content.addLayout(btn_layout)

        self.cancel_btn.setEnabled(False)
        # Progress context: normal vs two-pass OCR burn-in.
        self._progress_mode = "normal"
        self._two_pass_stage_guess = 1
        self._last_progress_current = -1
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
        self.check_inpaint.setEnabled((not is_timed) and (not self.check_detect_no_inpaint.isChecked()))
        self._save_options()

    def _on_skip_detect_changed(self):
        """When Skip detection is checked, disable Detection checkbox (detector not used)."""
        skip = self.check_skip_detect.isChecked()
        self.check_detect.setEnabled(not skip and self.source_combo.currentIndex() == 0)
        self._save_options()

    def _on_detect_no_inpaint_changed(self):
        no_inpaint = self.check_detect_no_inpaint.isChecked()
        # Keep the explicit inpaint checkbox value, but disable editing while override is active.
        self.check_inpaint.setEnabled((not no_inpaint) and self.source_combo.currentIndex() != 2)
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

    def _usage_preset_values(self, key: str) -> dict:
        """Return widget values for a usage preset. Keys: sample_every, region_idx, scene, scene_threshold, temporal, temporal_alpha, skip_detect, two_stage, two_stage_force, two_stage_diff, prefetch, ffmpeg, crf."""
        # region_idx: 0=full, 1=bottom_15, 2=bottom_20, 3=bottom_25, 4=bottom_30
        presets = {
            "dont_miss_text": {
                "sample_every": 2, "region_idx": 2, "scene": True, "scene_threshold": 28,
                "temporal": True, "temporal_alpha": 0.25, "skip_detect": False,
                "two_stage": True, "two_stage_force": 45, "two_stage_diff": 6.0,
                "prefetch": 3, "ffmpeg": True, "crf": 18,
            },
            "balanced": {
                "sample_every": 6, "region_idx": 2, "scene": True, "scene_threshold": 35,
                "temporal": True, "temporal_alpha": 0.25, "skip_detect": True,
                "two_stage": True, "two_stage_force": 72, "two_stage_diff": 7.0,
                "prefetch": 3, "ffmpeg": True, "crf": 23,
            },
            "max_speed": {
                "sample_every": 30, "region_idx": 2, "scene": True, "scene_threshold": 45,
                "temporal": True, "temporal_alpha": 0.28, "skip_detect": True,
                "two_stage": True, "two_stage_force": 90, "two_stage_diff": 10.0,
                "prefetch": 3, "ffmpeg": True, "crf": 23,
            },
            "anime": {
                "sample_every": 30, "region_idx": 2, "scene": True, "scene_threshold": 30,
                "temporal": True, "temporal_alpha": 0.25, "skip_detect": True,
                "two_stage": True, "two_stage_force": 60, "two_stage_diff": 8.0,
                "prefetch": 3, "ffmpeg": True, "crf": 18,
            },
            "documentary": {
                "sample_every": 6, "region_idx": 0, "scene": True, "scene_threshold": 35,
                "temporal": True, "temporal_alpha": 0.25, "skip_detect": False,
                "two_stage": True, "two_stage_force": 72, "two_stage_diff": 7.0,
                "prefetch": 3, "ffmpeg": True, "crf": 20,
            },
        }
        return presets.get(key, {})

    def _on_usage_preset_changed(self):
        idx = self.usage_preset_combo.currentIndex()
        key = ("custom", "dont_miss_text", "balanced", "max_speed", "anime", "documentary")[min(idx, 5)]
        if key == "custom":
            self._save_options()
            return
        vals = self._usage_preset_values(key)
        if not vals:
            self._save_options()
            return
        widgets = [
            self.sample_spin, self.region_combo, self.scene_threshold_spin,
            self.temporal_alpha_spin, self.two_stage_force_spin, self.two_stage_newline_diff_spin,
            self.prefetch_spin, self.ffmpeg_crf_spin,
        ]
        for w in widgets:
            w.blockSignals(True)
        self.sample_spin.setValue(vals.get("sample_every", self.sample_spin.value()))
        self.region_combo.setCurrentIndex(vals.get("region_idx", self.region_combo.currentIndex()))
        self.check_scene.setChecked(vals.get("scene", self.check_scene.isChecked()))
        self.scene_threshold_spin.setValue(vals.get("scene_threshold", self.scene_threshold_spin.value()))
        self.check_temporal.setChecked(vals.get("temporal", self.check_temporal.isChecked()))
        self.temporal_alpha_spin.setValue(vals.get("temporal_alpha", self.temporal_alpha_spin.value()))
        self.check_skip_detect.setChecked(vals.get("skip_detect", self.check_skip_detect.isChecked()))
        self._on_skip_detect_changed()
        self.check_two_stage.setChecked(vals.get("two_stage", self.check_two_stage.isChecked()))
        self.two_stage_force_spin.setValue(vals.get("two_stage_force", self.two_stage_force_spin.value()))
        self.two_stage_newline_diff_spin.setValue(vals.get("two_stage_diff", self.two_stage_newline_diff_spin.value()))
        self.prefetch_spin.setValue(vals.get("prefetch", self.prefetch_spin.value()))
        self.check_ffmpeg.setChecked(vals.get("ffmpeg", self.check_ffmpeg.isChecked()))
        self.ffmpeg_crf_spin.setValue(vals.get("crf", self.ffmpeg_crf_spin.value()))
        for w in widgets:
            w.blockSignals(False)
        self.check_detect.setEnabled(not self.check_skip_detect.isChecked())
        self._save_options()

    def _apply_recommended_long_anime(self):
        """Apply Anime preset (kept for compatibility; prefer Usage preset dropdown)."""
        self.usage_preset_combo.setCurrentIndex(4)  # Anime / long series
        self._on_usage_preset_changed()

    def _save_options(self):
        """Persist pipeline options to config and save."""
        _preset_idx = min(self.usage_preset_combo.currentIndex(), 5)
        pcfg.module.video_translator_usage_preset = ("custom", "dont_miss_text", "balanced", "max_speed", "anime", "documentary")[_preset_idx]
        pcfg.module.video_translator_sample_every_frames = self.sample_spin.value()
        pcfg.module.video_translator_enable_detect = self.check_detect.isChecked()
        pcfg.module.video_translator_enable_ocr = self.check_ocr.isChecked()
        pcfg.module.video_translator_enable_translate = self.check_translate.isChecked()
        pcfg.module.video_translator_enable_inpaint = self.check_inpaint.isChecked()
        pcfg.module.video_translator_subtitle_black_box_mode = self.check_subtitle_black_box.isChecked()
        pcfg.module.video_translator_region_preset = self._region_preset_value()
        pcfg.module.video_translator_output_codec = (self.codec_edit.text() or "").strip() or "mp4v"
        pcfg.module.video_translator_use_scene_detection = self.check_scene.isChecked()
        pcfg.module.video_translator_scene_threshold = float(self.scene_threshold_spin.value())
        pcfg.module.video_translator_temporal_smoothing = self.check_temporal.isChecked()
        pcfg.module.video_translator_temporal_alpha = float(self.temporal_alpha_spin.value())
        pcfg.module.video_translator_prefetch_frames = self.prefetch_spin.value()
        pcfg.module.video_translator_two_stage_keyframes = self.check_two_stage.isChecked()
        pcfg.module.video_translator_background_writer = self.check_background_writer.isChecked()
        pcfg.module.video_translator_use_two_pass_ocr_burn_in = self.check_two_pass_ocr_burn_in.isChecked()
        pcfg.module.video_translator_two_stage_force_refresh_every_frames = int(self.two_stage_force_spin.value())
        pcfg.module.video_translator_two_stage_new_line_diff_threshold = float(self.two_stage_newline_diff_spin.value())
        pcfg.module.video_translator_adaptive_detector_roi = self.check_adaptive_detector_roi.isChecked()
        pcfg.module.video_translator_adaptive_detector_roi_padding_frac = float(self.adaptive_roi_padding_spin.value())
        pcfg.module.video_translator_adaptive_detector_roi_start_seconds = float(self.adaptive_roi_start_seconds_spin.value())
        pcfg.module.video_translator_auto_catch_subtitle_on_skipped_frames = self.check_auto_catch_subs.isChecked()
        pcfg.module.video_translator_auto_catch_diff_threshold = float(self.auto_catch_diff_threshold_spin.value())
        pcfg.module.video_translator_overlap_inpaint = self.check_overlap_inpaint.isChecked()
        pcfg.module.video_translator_ocr_temporal_stability = self.check_ocr_temporal_stability.isChecked()
        pcfg.module.video_translator_ocr_temporal_window = int(self.ocr_temporal_window_spin.value())
        pcfg.module.video_translator_ocr_temporal_min_votes = int(self.ocr_temporal_min_votes_spin.value())
        pcfg.module.video_translator_ocr_temporal_geo_quantization = int(self.ocr_temporal_geo_quant_spin.value())
        pcfg.module.video_translator_use_ffmpeg = self.check_ffmpeg.isChecked()
        pcfg.module.video_translator_ffmpeg_path = (self.ffmpeg_path_edit.text() or "").strip()
        pcfg.module.video_translator_ffmpeg_crf = int(self.ffmpeg_crf_spin.value())
        pcfg.module.video_translator_ffmpeg_preset = (self.ffmpeg_preset_combo.currentText() or "medium").strip()
        _hw_idx = self.ffmpeg_hw_encoder_combo.currentIndex()
        pcfg.module.video_translator_ffmpeg_hw_encoder = ("auto" if _hw_idx == 3 else "qsv" if _hw_idx == 2 else "nvenc" if _hw_idx == 1 else "none")
        pcfg.module.video_translator_video_bitrate_kbps = int(self.video_bitrate_spin.value())
        pcfg.module.video_translator_skip_detect = self.check_skip_detect.isChecked()
        pcfg.module.video_translator_detect_no_inpaint = self.check_detect_no_inpaint.isChecked()
        pcfg.module.video_translator_bottom_band_native_mode = self.check_bottom_band_native_mode.isChecked()
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
        pcfg.module.video_translator_sentence_merge_by_punctuation = self.check_sentence_merge_punct.isChecked()
        pcfg.module.video_translator_sentence_merge_max_seconds = float(self.sentence_merge_max_sec_spin.value())
        pcfg.module.video_translator_asr_audio_separation = self.check_asr_audio_separation.isChecked()
        pcfg.module.video_translator_asr_guided_detect_inpaint = self.check_asr_guided_detect_inpaint.isChecked()
        pcfg.module.video_translator_asr_guided_midpoint_refresh = self.check_asr_guided_midpoint_refresh.isChecked()
        pcfg.module.video_translator_asr_chunk_seconds = float(self.asr_chunk_seconds_spin.value())
        pcfg.module.video_translator_asr_long_audio_threshold_seconds = float(
            self.asr_long_audio_threshold_spin.value()
        )
        pcfg.module.video_translator_asr_checkpoint_resume = self.check_asr_checkpoint_resume.isChecked()
        pcfg.module.video_translator_export_ass = self.check_export_ass.isChecked()
        pcfg.module.video_translator_export_vtt = self.check_export_vtt.isChecked()
        pcfg.module.video_translator_glossary = (self.glossary_edit.toPlainText() or "").strip()
        pcfg.module.video_translator_nlp_chunk_size = int(self.nlp_chunk_spin.value())
        pcfg.module.video_translator_nlp_max_workers = int(self.nlp_max_workers_spin.value())
        pcfg.module.video_translator_llm_per_line_quality_fix = self.check_llm_per_line_quality_fix.isChecked()
        pcfg.module.video_translator_qwen35_allow_aux_passes = self.check_qwen_aux_passes.isChecked()
        pcfg.module.video_translator_lock_watermark_lines = self.check_lock_watermark_lines.isChecked()
        pcfg.module.video_translator_lock_watermark_regex = (self.lock_watermark_regex_edit.text() or "").strip() or r"备案号[:：]?\s*\d{8,}"
        pcfg.module.video_translator_series_context_path = (self.series_context_edit.text() or "").strip()
        pcfg.module.video_translator_last_batch_output_dir = (self.batch_output_edit.text() or "").strip()
        pcfg.module.video_translator_soft_subs_only = self.check_soft_subs.isChecked()
        pcfg.module.video_translator_inpaint_only_soft_subs = self.check_inpaint_only_soft.isChecked()
        pcfg.module.video_translator_mux_srt_into_video = self.check_mux_srt.isChecked()
        pcfg.module.video_translator_flow_fixer_enabled = self.check_flow_fixer.isChecked()
        pcfg.module.video_translator_use_flow_fixer_for_corrections = self.check_flow_fixer_for_corrections.isChecked()
        idx = self.flow_fixer_combo.currentIndex()
        pcfg.module.video_translator_flow_fixer = "openai" if idx == 3 else ("openrouter" if idx == 2 else ("local_server" if idx == 1 else "none"))
        pcfg.module.video_translator_flow_fixer_server_url = (self.flow_fixer_url_edit.text() or "").strip() or "http://localhost:1234/v1"
        pcfg.module.video_translator_flow_fixer_model = (self.flow_fixer_model_edit.text() or "").strip()
        pcfg.module.video_translator_flow_fixer_max_tokens = self.flow_fixer_max_tokens_spin.value()
        pcfg.module.video_translator_flow_fixer_context_lines = self.flow_fixer_context_spin.value()
        pcfg.module.video_translator_flow_fixer_strict_single_line_review = self.check_flow_fixer_strict_single.isChecked()
        pcfg.module.video_translator_flow_fixer_timeout = float(getattr(pcfg.module, "video_translator_flow_fixer_timeout", 30.0))
        pcfg.module.video_translator_flow_fixer_enable_reasoning = self.check_flow_fixer_reasoning.isChecked()
        _idx_eff = self.flow_fixer_reasoning_effort_combo.currentIndex()
        pcfg.module.video_translator_flow_fixer_reasoning_effort = "high" if _idx_eff == 2 else ("low" if _idx_eff == 0 else "medium")
        pcfg.module.video_translator_flow_fixer_openrouter_apikey = (self.flow_fixer_openrouter_apikey_edit.text() or "").strip()
        pcfg.module.video_translator_flow_fixer_openrouter_model = (self.flow_fixer_openrouter_model_edit.text() or "").strip() or "google/gemma-3n-e2b-it:free"
        pcfg.module.video_translator_flow_fixer_openai_apikey = (self.flow_fixer_openai_apikey_edit.text() or "").strip()
        pcfg.module.video_translator_flow_fixer_openai_model = (self.flow_fixer_openai_model_edit.text() or "").strip() or "gpt-4o-mini"
        pcfg.module.video_translator_post_review_enabled = self.check_post_review.isChecked()
        pcfg.module.video_translator_post_review_use_main_translator = self.check_post_review_use_main_llm.isChecked()
        pcfg.module.video_translator_post_review_apply_on_cancel = self.check_post_review_on_cancel.isChecked()
        pcfg.module.video_translator_post_review_chunk_size = int(self.post_review_chunk_spin.value())
        pcfg.module.video_translator_post_review_context_lines = int(self.post_review_context_spin.value())
        pcfg.module.video_translator_post_review_enable_reasoning = self.check_post_review_reasoning.isChecked()
        _pre_idx = self.post_review_reasoning_effort_combo.currentIndex()
        pcfg.module.video_translator_post_review_reasoning_effort = (
            "high" if _pre_idx == 2 else ("low" if _pre_idx == 0 else "medium")
        )
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
        self.preview_subtitle_history = []
        self._last_preview_subtitle_lines = None

        soft_subs = self.check_soft_subs.isChecked()
        _idx = self.source_combo.currentIndex()
        src = "asr" if _idx == 1 else "existing_subs" if _idx == 2 else "ocr"
        self._progress_mode = (
            "two_pass_ocr"
            if (
                src == "ocr"
                and self.check_translate.isChecked()
                and (not soft_subs)
                and (not self.check_inpaint_only_soft.isChecked())
                and self.check_two_pass_ocr_burn_in.isChecked()
            )
            else "normal"
        )
        self._two_pass_stage_guess = 1
        self._last_progress_current = -1
        self.thread = VideoTranslateThread(
            inp,
            out,
            self.sample_spin.value(),
            self.check_detect.isChecked(),
            self.check_ocr.isChecked(),
            self.check_translate.isChecked(),
            False if (soft_subs or self.check_detect_no_inpaint.isChecked()) else self.check_inpaint.isChecked(),
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
            prefetch_frames=self.prefetch_spin.value(),
            two_stage_keyframes=self.check_two_stage.isChecked(),
            background_writer=self.check_background_writer.isChecked(),
            use_two_pass_ocr_burn_in=self.check_two_pass_ocr_burn_in.isChecked(),
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
        self.preview_subtitle_history = []
        self._last_preview_subtitle_lines = None
        soft_subs = self.check_soft_subs.isChecked()
        _idx = self.source_combo.currentIndex()
        src = "asr" if _idx == 1 else "existing_subs" if _idx == 2 else "ocr"
        self._progress_mode = (
            "two_pass_ocr"
            if (
                src == "ocr"
                and self.check_translate.isChecked()
                and (not soft_subs)
                and (not self.check_inpaint_only_soft.isChecked())
                and self.check_two_pass_ocr_burn_in.isChecked()
            )
            else "normal"
        )
        self._two_pass_stage_guess = 1
        self._last_progress_current = -1
        self.thread = VideoTranslateThread(
            inp, out,
            self.sample_spin.value(),
            self.check_detect.isChecked(),
            self.check_ocr.isChecked(),
            self.check_translate.isChecked(),
            False if (soft_subs or self.check_detect_no_inpaint.isChecked()) else self.check_inpaint.isChecked(),
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
            prefetch_frames=self.prefetch_spin.value(),
            two_stage_keyframes=self.check_two_stage.isChecked(),
            background_writer=self.check_background_writer.isChecked(),
            use_two_pass_ocr_burn_in=self.check_two_pass_ocr_burn_in.isChecked(),
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
            self._progress_mode = "normal"
            self._two_pass_stage_guess = 1
            self._last_progress_current = -1
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
                # Update full-screen window if open
                if self._preview_fullscreen_window is not None and self._preview_fullscreen_window.isVisible():
                    lbl = getattr(self._preview_fullscreen_window, "_label", None)
                    if lbl is not None:
                        lbl.setPixmap(
                            pix.scaled(lbl.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                        )
            except Exception as e:
                LOGGER.debug("Preview frame update: %s", e)
        lines = []
        for src, trans in (subtitle_lines or []):
            if (src or "").strip() or (trans or "").strip():
                lines.append("%s → %s" % ((src or "").strip() or "-", (trans or "").strip() or "-"))
        text = "\n".join(lines) if lines else self.tr("No text on this frame.")
        self.preview_text_edit.setPlainText(text)
        # Append to history only when subtitle content changed (avoid duplicate entries)
        if subtitle_lines is not None and subtitle_lines != self._last_preview_subtitle_lines:
            self._last_preview_subtitle_lines = list(subtitle_lines) if subtitle_lines else []
            self.preview_subtitle_history.append((frame_index, self._last_preview_subtitle_lines))

    def _show_preview_fullscreen(self):
        """Show current frame in a full-screen window. Esc to close."""
        pix = self.preview_frame_label.pixmap()
        if pix is None or pix.isNull():
            return
        if self._preview_fullscreen_window is not None and self._preview_fullscreen_window.isVisible():
            self._preview_fullscreen_window.close()
            return
        win = QDialog(self)
        win.setWindowTitle(self.tr("Preview — full screen"))
        win.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        layout = QVBoxLayout(win)
        layout.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel(win)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet("background-color: #000;")
        lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        lbl.setPixmap(
            pix.scaled(lbl.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        )
        layout.addWidget(lbl)
        win._label = lbl
        win._parent_dialog = self

        def refresh_pix():
            p = win._parent_dialog.preview_frame_label.pixmap()
            if p is not None and not p.isNull() and lbl.size().width() > 10:
                lbl.setPixmap(
                    p.scaled(lbl.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                )
        win.showFullScreen()
        QTimer.singleShot(50, refresh_pix)
        win.finished.connect(lambda: setattr(self, "_preview_fullscreen_window", None))
        QShortcut(QKeySequence(Qt.Key.Key_Escape), win, win.close)
        self._preview_fullscreen_window = win

    def _show_preview_history(self):
        """Open a dialog listing previous source text and translations for the current run."""
        d = QDialog(self)
        d.setWindowTitle(self.tr("Translation history"))
        d.setMinimumSize(480, 360)
        layout = QVBoxLayout(d)
        te = QPlainTextEdit(d)
        te.setReadOnly(True)
        te.setStyleSheet("font-size: 0.95em; background: #1e1e1e; color: #ddd;")
        te.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        if not self.preview_subtitle_history:
            te.setPlainText(self.tr("No history yet. Run the pipeline to see source and translations as they are detected."))
        else:
            parts = []
            for frame_index, pairs in self.preview_subtitle_history:
                parts.append("——— " + self.tr("Frame %s") % frame_index + " ———")
                for src, trans in pairs:
                    src_s = (src or "").strip() or "-"
                    trans_s = (trans or "").strip() or "-"
                    parts.append(self.tr("Source: %s") % src_s)
                    parts.append("→ " + trans_s)
                    parts.append("")
                parts.append("")
            te.setPlainText("\n".join(parts))
        layout.addWidget(te)
        close_btn = QPushButton(self.tr("Close"))
        close_btn.clicked.connect(d.accept)
        layout.addWidget(close_btn)
        d.exec()

    def _show_ab_compare_dialog(self):
        """A/B compare two models on sampled subtitle lines from current run history."""
        samples = []
        seen = set()
        for _frame_index, pairs in (self.preview_subtitle_history or []):
            for src, _trans in (pairs or []):
                s = str(src or "").strip()
                if not s or _is_invalid_subtitle_text(s) or s in seen:
                    continue
                seen.add(s)
                samples.append(s)
        preset_sets = {
            "zh_cn_terms": [
                "备案号：11904073220604077",
                "宗门大比开始！",
                "你先别急，听我解释。",
                "三日之后，山门见。",
                "此人修为深不可测。",
                "师尊，我已突破筑基。",
                "若有来生，再与你相见。",
                "快走！他们追上来了！",
                "你敢动她一下试试。",
                "天道无情，人有情。",
            ],
            "ja_anime_terms": [
                "行くぞ！",
                "ここは俺に任せろ。",
                "先輩、ありがとうございます。",
                "このままじゃ間に合わない。",
                "約束は守るって言っただろ。",
                "信じてるよ。",
                "本気で来い。",
                "終わりにしよう。",
                "絶対に負けない。",
                "また会おう。",
            ],
            "ko_terms": [
                "지금 가면 늦어.",
                "내가 막을게.",
                "약속은 지켜야지.",
                "정말 고마워.",
                "여기서 끝내자.",
                "절대 포기하지 마.",
                "그 사람을 믿어.",
                "시간이 없어.",
                "다 같이 가자.",
                "다음에 또 보자.",
            ],
            "en_dialog_terms": [
                "We don't have much time.",
                "Trust me on this one.",
                "I'll hold them off.",
                "Don't look back.",
                "You promised you'd come.",
                "This is our last chance.",
                "We're not done yet.",
                "Stay with me.",
                "Let's finish this together.",
                "See you on the other side.",
            ],
        }

        d = QDialog(self)
        d.setWindowTitle(self.tr("A/B compare translator models"))
        d.setMinimumSize(540, 280)
        lay = QVBoxLayout(d)
        grid = QGridLayout()
        grid.addWidget(QLabel(self.tr("Model A:")), 0, 0)
        model_a_edit = QLineEdit((getattr(pcfg.module, "video_translator_ab_model_a", None) or "").strip())
        model_a_edit.setPlaceholderText(self.tr("e.g. qwen/qwen3.5-4b-instruct"))
        grid.addWidget(model_a_edit, 0, 1)
        grid.addWidget(QLabel(self.tr("Model B:")), 1, 0)
        model_b_edit = QLineEdit((getattr(pcfg.module, "video_translator_ab_model_b", None) or "").strip())
        model_b_edit.setPlaceholderText(self.tr("e.g. mextrans-7b"))
        grid.addWidget(model_b_edit, 1, 1)
        grid.addWidget(QLabel(self.tr("Sample lines:")), 2, 0)
        sample_spin = QSpinBox()
        sample_spin.setRange(20, 1000)
        sample_spin.setValue(int(getattr(pcfg.module, "video_translator_ab_sample_size", 200) or 200))
        grid.addWidget(sample_spin, 2, 1)
        grid.addWidget(QLabel(self.tr("Test source:")), 3, 0)
        source_combo = QComboBox()
        source_combo.addItems([
            self.tr("Current run history"),
            self.tr("Preset: Chinese subtitle terms"),
            self.tr("Preset: Japanese anime lines"),
            self.tr("Preset: Korean dialogue lines"),
            self.tr("Preset: English dialogue lines"),
            self.tr("Custom pasted lines"),
        ])
        grid.addWidget(source_combo, 3, 1)
        lay.addLayout(grid)
        lay.addWidget(QLabel(self.tr("Custom lines (one per line):")))
        custom_edit = QPlainTextEdit(d)
        custom_edit.setPlaceholderText(self.tr("Paste your own terms/lines here, one per line."))
        custom_edit.setMaximumHeight(120)
        custom_edit.setPlainText((getattr(pcfg.module, "video_translator_ab_custom_lines", None) or "").strip())
        lay.addWidget(custom_edit)
        log = QPlainTextEdit(d)
        log.setReadOnly(True)
        log.setPlainText(self.tr("Click Run compare to generate side-by-side output."))
        lay.addWidget(log, 1)
        btn_row = QHBoxLayout()
        run_btn = QPushButton(self.tr("Run compare"))
        use_a_btn = QPushButton(self.tr("Use Model A"))
        use_b_btn = QPushButton(self.tr("Use Model B"))
        close_btn = QPushButton(self.tr("Close"))
        btn_row.addStretch()
        btn_row.addWidget(run_btn)
        btn_row.addWidget(use_a_btn)
        btn_row.addWidget(use_b_btn)
        save_custom_btn = QPushButton(self.tr("Save custom list"))
        btn_row.addWidget(save_custom_btn)
        btn_row.addWidget(close_btn)
        lay.addLayout(btn_row)
        close_btn.clicked.connect(d.accept)
        save_custom_btn.clicked.connect(
            lambda: (
                setattr(pcfg.module, "video_translator_ab_custom_lines", (custom_edit.toPlainText() or "").strip()),
                save_config(),
                self.status_label.setText(self.tr("Saved custom A/B list."))
            )
        )

        def _apply_winner(which: str):
            picked = (model_a_edit.text() if which == "a" else model_b_edit.text() or "").strip()
            if not picked:
                self.status_label.setText(self.tr("A/B compare: set model first."))
                return
            # Prefer setting translator override; fallback to model field if needed.
            try:
                tcfg = getattr(pcfg.module, "translator_params", None)
                tname = getattr(pcfg.module, "translator", None)
                if isinstance(tcfg, dict) and tname in tcfg and isinstance(tcfg[tname], dict):
                    tcfg[tname]["override model"] = picked
            except Exception:
                pass
            self.status_label.setText(self.tr("A/B compare: set winner model to {}").format(picked))
            save_config()

        use_a_btn.clicked.connect(lambda: _apply_winner("a"))
        use_b_btn.clicked.connect(lambda: _apply_winner("b"))

        def _run_compare():
            from qtpy.QtWidgets import QApplication
            from utils.textblock import TextBlock

            model_a = (model_a_edit.text() or "").strip()
            model_b = (model_b_edit.text() or "").strip()
            if not model_a or not model_b:
                log.setPlainText(self.tr("Please set both Model A and Model B."))
                return
            count = max(20, int(sample_spin.value()))
            source_idx = source_combo.currentIndex()
            if source_idx == 0:
                if not samples:
                    log.setPlainText(self.tr("No current run history yet. Choose a preset test source or run a translation first."))
                    return
                base_samples = samples
            elif source_idx == 1:
                base_samples = preset_sets["zh_cn_terms"]
            elif source_idx == 2:
                base_samples = preset_sets["ja_anime_terms"]
            elif source_idx == 3:
                base_samples = preset_sets["ko_terms"]
            elif source_idx == 4:
                base_samples = preset_sets["en_dialog_terms"]
            else:
                raw_custom = (custom_edit.toPlainText() or "").strip()
                base_samples = [ln.strip() for ln in raw_custom.splitlines() if ln.strip()]
                if not base_samples:
                    log.setPlainText(self.tr("Custom list is empty. Paste lines first or choose another source."))
                    return
            if len(base_samples) > count:
                step = max(1, len(base_samples) // count)
                eval_samples = [base_samples[i] for i in range(0, len(base_samples), step)][:count]
            else:
                eval_samples = list(base_samples)

            pcfg.module.video_translator_ab_model_a = model_a
            pcfg.module.video_translator_ab_model_b = model_b
            pcfg.module.video_translator_ab_sample_size = int(count)
            save_config()

            cfg = pcfg.module
            translator = _get_video_module("translator", cfg.translator)
            if translator is None:
                log.setPlainText(self.tr("Could not initialize translator module for A/B compare."))
                return
            if hasattr(translator, "load_model"):
                translator.load_model()

            orig_override = None
            try:
                if hasattr(translator, "get_param_value"):
                    orig_override = translator.get_param_value("override model")
            except Exception:
                orig_override = None

            def _translate_with_model(model_name: str):
                out = []
                try:
                    if hasattr(translator, "set_param_value"):
                        translator.set_param_value("override model", model_name)
                except Exception:
                    pass
                batch_size = 20
                for i in range(0, len(eval_samples), batch_size):
                    chunk = eval_samples[i : i + batch_size]
                    blks = []
                    for s in chunk:
                        b = TextBlock()
                        b.text = [s]
                        b.translation = ""
                        blks.append(b)
                    try:
                        translator.translate_textblk_lst(blks)
                    except Exception:
                        for _ in chunk:
                            out.append("")
                        continue
                    for b in blks:
                        out.append(str(getattr(b, "translation", "") or "").strip())
                    QApplication.processEvents()
                return out

            self.status_label.setText(self.tr("A/B compare running..."))
            QApplication.processEvents()
            trans_a = _translate_with_model(model_a)
            trans_b = _translate_with_model(model_b)
            try:
                if hasattr(translator, "set_param_value"):
                    translator.set_param_value("override model", orig_override or "")
            except Exception:
                pass

            disagreements = 0
            lines = []
            for i, src in enumerate(eval_samples):
                a = trans_a[i] if i < len(trans_a) else ""
                b = trans_b[i] if i < len(trans_b) else ""
                if a != b:
                    disagreements += 1
                lines.append(f"[{i+1}] SOURCE: {src}")
                lines.append(f"    A ({model_a}): {a}")
                lines.append(f"    B ({model_b}): {b}")
                lines.append("")
            agree = max(0, len(eval_samples) - disagreements)
            header = [
                f"Samples: {len(eval_samples)}",
                f"Agree: {agree}",
                f"Disagree: {disagreements}",
                f"Disagreement rate: {(100.0 * disagreements / max(1, len(eval_samples))):.1f}%",
                "",
            ]
            text = "\n".join(header + lines)
            log.setPlainText(text)
            out_base = (self.output_edit.text() or "").strip() or (self.input_edit.text() or "").strip()
            if out_base:
                try:
                    base = osp.splitext(out_base)[0]
                    out_path = base + ".ab_compare.txt"
                    with open(out_path, "w", encoding="utf-8") as f:
                        f.write(text)
                    self.status_label.setText(self.tr("A/B compare done. Saved: {}").format(out_path))
                except Exception:
                    self.status_label.setText(self.tr("A/B compare done."))
            else:
                self.status_label.setText(self.tr("A/B compare done."))

        run_btn.clicked.connect(_run_compare)
        d.exec()

    def _on_progress(self, current: int, total: int):
        if total > 0:
            self.progress_bar.setValue(int(100 * current / total))
            self.progress_bar.setRange(0, 100)
        else:
            self.progress_bar.setRange(0, 0)
        stage_prefix = ""
        disp_current = current
        disp_total = total if total > 0 else "?"
        if self._progress_mode == "two_pass_ocr":
            if total > 0:
                half = max(1, int(total // 2))
                if current <= half:
                    stage_prefix = self.tr("Pass 1/2")
                    disp_current = current
                    disp_total = half
                else:
                    stage_prefix = self.tr("Pass 2/2")
                    disp_current = max(0, current - half)
                    disp_total = max(1, total - half)
            else:
                # Unknown total: infer stage switch when progress counter restarts in pass 2.
                if self._last_progress_current >= 0 and current < self._last_progress_current:
                    self._two_pass_stage_guess = 2
                stage_prefix = self.tr("Pass 1/2") if self._two_pass_stage_guess == 1 else self.tr("Pass 2/2")
                self._last_progress_current = current
        if self._batch_jobs and self._batch_index < len(self._batch_jobs):
            if stage_prefix:
                self.status_label.setText(self.tr("File {} / {}: {} frame {} / {}").format(
                    self._batch_index + 1,
                    len(self._batch_jobs),
                    stage_prefix,
                    disp_current,
                    disp_total,
                ))
            else:
                self.status_label.setText(self.tr("File {} / {}: frame {} / {}").format(
                    self._batch_index + 1, len(self._batch_jobs), current, total if total > 0 else "?"))
        else:
            if stage_prefix:
                self.status_label.setText(self.tr("{} frame {} / {}").format(stage_prefix, disp_current, disp_total))
            else:
                self.status_label.setText(self.tr("Frame {} / {}").format(current, total if total > 0 else "?"))

    def _on_finished_ok(self, path: str):
        self._progress_mode = "normal"
        self._two_pass_stage_guess = 1
        self._last_progress_current = -1
        if not (self._batch_jobs and self._batch_index <= len(self._batch_jobs)):
            self.status_label.setText(self.tr("Done. Output: {}").format(path))

    def _on_failed(self, msg: str):
        self._progress_mode = "normal"
        self._two_pass_stage_guess = 1
        self._last_progress_current = -1
        self.status_label.setText(self.tr("Error: {}").format(msg))

    def _on_thread_finished(self):
        if not self._batch_jobs:
            self.run_btn.setEnabled(True)
            self.cancel_btn.setEnabled(False)

    def closeEvent(self, event: QCloseEvent):
        """Close (X / Close button) hides the dialog and keeps the run alive.

        Only Cancel should stop the pipeline.
        """
        try:
            self._save_options()
        except Exception:
            pass
        try:
            self._vt_was_maximized = bool(self.isMaximized())
            if not self._vt_was_maximized:
                self._vt_saved_geometry = self.saveGeometry()
        except Exception:
            pass
        event.ignore()
        self.hide()

    def reject(self):
        """Escape key / reject should behave like Close (hide), not cancel."""
        try:
            self._save_options()
        except Exception:
            pass
        try:
            self._vt_was_maximized = bool(self.isMaximized())
            if not self._vt_was_maximized:
                self._vt_saved_geometry = self.saveGeometry()
        except Exception:
            pass
        self.hide()

    def showEvent(self, event: QShowEvent):
        """Restore geometry/maximized state after being hidden."""
        try:
            if self._vt_saved_geometry is not None and not self._vt_was_maximized:
                self.restoreGeometry(self._vt_saved_geometry)
        except Exception:
            pass
        super().showEvent(event)
        # Force relayout after show/hide/maximize. Some Qt backends keep the old viewport size
        # until the next event loop tick, which can make the UI appear "stuck in a corner".
        def _force_relayout():
            try:
                if self._scroll is not None and self._scroll.widget() is not None:
                    self._scroll.widget().updateGeometry()
                    self._scroll.viewport().updateGeometry()
                if self.layout() is not None:
                    self.layout().activate()
                self.updateGeometry()
                self.update()
            except Exception:
                pass

        # Restoring maximized needs to happen after show() on some platforms.
        if getattr(self, "_vt_was_maximized", False):
            QTimer.singleShot(0, self.showMaximized)
            QTimer.singleShot(0, _force_relayout)
            QTimer.singleShot(50, _force_relayout)
        else:
            QTimer.singleShot(0, _force_relayout)

    def resizeEvent(self, event: QResizeEvent):
        """Keep scroll contents synced to viewport size (prevents 'UI stuck in corner')."""
        try:
            if getattr(self, "_scroll", None) is not None:
                w = self._scroll.viewport().width()
                if self._scroll.widget() is not None and w > 0:
                    # Force the scroll content to track the viewport width so layouts recompute.
                    self._scroll.widget().setMinimumWidth(w)
                    self._scroll.widget().updateGeometry()
        except Exception:
            pass
        super().resizeEvent(event)

    def _cancel_run(self):
        """Cancel button: stop the running pipeline."""
        if self.thread and self.thread.isRunning():
            self.thread.cancel()
