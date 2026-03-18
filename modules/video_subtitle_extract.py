"""
Load existing subtitles for video translator: sidecar files (.srt, .ass, .vtt)
or embedded subtitle stream via FFmpeg. Used when Source = Existing subtitles.
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
from typing import List, Optional, Tuple


def _parse_ts_srt(s: str) -> float:
    """Parse SRT timestamp HH:MM:SS,mmm to seconds."""
    s = (s or "").strip().replace(",", ".")
    parts = s.split(":")
    if len(parts) != 3:
        return 0.0
    try:
        h, m, rest = int(parts[0]), int(parts[1]), float(parts[2])
        return h * 3600 + m * 60 + rest
    except ValueError:
        return 0.0


def parse_srt(path: str) -> List[Tuple[float, float, str]]:
    """Parse SRT file. Returns list of (start_sec, end_sec, text)."""
    result = []
    if not path or not os.path.isfile(path):
        return result
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    # SRT blocks: optional index line, timestamp line, text line(s), blank
    block_pat = re.compile(
        r"(?:\d+\s*\n)?"
        r"(\d{2}:\d{2}:\d{2}[,\.]\d+)\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d+)\s*\n"
        r"((?:(?!\n\s*\n).)*)",
        re.DOTALL,
    )
    for m in block_pat.finditer(content):
        start_sec = _parse_ts_srt(m.group(1))
        end_sec = _parse_ts_srt(m.group(2))
        text = (m.group(3) or "").strip().replace("\n", " ").strip()
        if text or start_sec > 0 or end_sec > 0:
            result.append((start_sec, end_sec, text))
    return result


def _parse_ts_vtt(s: str) -> float:
    """Parse WebVTT timestamp HH:MM:SS.mmm to seconds."""
    s = (s or "").strip()
    parts = s.split(":")
    if len(parts) != 3:
        return 0.0
    try:
        h, m, rest = int(parts[0]), int(parts[1]), float(parts[2])
        return h * 3600 + m * 60 + rest
    except ValueError:
        return 0.0


def parse_vtt(path: str) -> List[Tuple[float, float, str]]:
    """Parse WebVTT file. Returns list of (start_sec, end_sec, text)."""
    result = []
    if not path or not os.path.isfile(path):
        return result
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    # Skip WEBVTT header and optional note
    idx = content.find("\n\n")
    if idx >= 0:
        content = content[idx + 2 :]
    # Cue: optional id, timestamp line, text
    cue_pat = re.compile(
        r"(?:\d+\s*\n)?"
        r"(\d{2}:\d{2}:\d{2}\.\d+)\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d+)(?:\s+.*?)?\s*\n"
        r"((?:(?!\n\s*\n).)*)",
        re.DOTALL,
    )
    for m in cue_pat.finditer(content):
        start_sec = _parse_ts_vtt(m.group(1))
        end_sec = _parse_ts_vtt(m.group(2))
        text = (m.group(3) or "").strip().replace("\n", " ").strip()
        if text or start_sec > 0 or end_sec > 0:
            result.append((start_sec, end_sec, text))
    return result


def _parse_ts_ass(s: str) -> float:
    """Parse ASS timestamp H:MM:SS.cc (centiseconds) to seconds."""
    s = (s or "").strip()
    parts = s.split(":")
    if len(parts) != 4:
        return 0.0
    try:
        h, m, sec, cs = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
        return h * 3600 + m * 60 + sec + cs / 100.0
    except ValueError:
        return 0.0


def parse_ass(path: str) -> List[Tuple[float, float, str]]:
    """Parse ASS file Dialogue lines. Returns list of (start_sec, end_sec, text)."""
    result = []
    if not path or not os.path.isfile(path):
        return result
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line.lower().startswith("dialogue:"):
                continue
            # Format: Dialogue: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
            parts = line.split(",", 9)
            if len(parts) < 10:
                continue
            start_sec = _parse_ts_ass(parts[1].strip())
            end_sec = _parse_ts_ass(parts[2].strip())
            text = (parts[9] or "").replace("\\N", "\n").strip()
            result.append((start_sec, end_sec, text))
    return result


def get_existing_subtitle_path(video_path: str) -> Optional[str]:
    """Check for sidecar .srt, .ass, .vtt next to video. Returns path if found."""
    if not video_path or not os.path.isfile(video_path):
        return None
    base, _ = os.path.splitext(video_path)
    for ext in (".srt", ".ass", ".vtt"):
        path = base + ext
        if os.path.isfile(path):
            return path
    return None


def extract_embedded_subtitles(
    video_path: str,
    ffmpeg_path: str = "ffmpeg",
    output_path: Optional[str] = None,
) -> Optional[str]:
    """Extract first embedded subtitle stream to SRT. Returns path to SRT or None."""
    if not video_path or not os.path.isfile(video_path):
        return None
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".srt")
        os.close(fd)
    cmd = [
        ffmpeg_path, "-y", "-i", video_path,
        "-map", "0:s:0", "-c:s", "srt",
        output_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
        if os.path.isfile(output_path) and os.path.getsize(output_path) > 0:
            return output_path
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        pass
    if output_path and output_path.startswith(tempfile.gettempdir()) and os.path.isfile(output_path):
        try:
            os.remove(output_path)
        except Exception:
            pass
    return None


def load_existing_subtitles(
    video_path: str,
    ffmpeg_path: str = "ffmpeg",
) -> Optional[List[Tuple[float, float, str]]]:
    """Try sidecar .srt/.ass/.vtt first, then embedded subtitle stream. Returns segments (start_sec, end_sec, text) or None."""
    sidecar = get_existing_subtitle_path(video_path)
    if sidecar:
        ext = os.path.splitext(sidecar)[1].lower()
        if ext == ".srt":
            segs = parse_srt(sidecar)
        elif ext == ".vtt":
            segs = parse_vtt(sidecar)
        elif ext == ".ass":
            segs = parse_ass(sidecar)
        else:
            segs = []
        if segs:
            return segs
    extracted = extract_embedded_subtitles(video_path, ffmpeg_path=ffmpeg_path)
    if extracted:
        try:
            segs = parse_srt(extracted)
            if segs:
                if extracted.startswith(tempfile.gettempdir()):
                    try:
                        os.remove(extracted)
                    except Exception:
                        pass
                return segs
        except Exception:
            if extracted.startswith(tempfile.gettempdir()):
                try:
                    os.remove(extracted)
                except Exception:
                    pass
    return None
