"""
Parse and write SubRip (.srt) and bracketed timestamp text (.txt) subtitle cues.

Bracketed text format (one cue per line):
  [00:01:02,345 --> 00:01:05,678] Subtitle text here
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Literal, Tuple

FormatHint = Literal["auto", "srt", "txt"]


@dataclass
class SubtitleCue:
    """One timed subtitle line."""

    start_ms: int
    end_ms: int
    text: str


_SRT_COMPONENT = r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})"
_SRT_LINE_RE = re.compile(
    rf"^\s*{_SRT_COMPONENT}\s*-->\s*{_SRT_COMPONENT}\b",
    re.IGNORECASE,
)
_TXT_CUE_RE = re.compile(
    rf"^\s*\[{_SRT_COMPONENT}\s*-->\s*{_SRT_COMPONENT}\]\s*(.*)$",
    re.IGNORECASE,
)


def _frac_to_ms(frac: str) -> int:
    frac = (frac or "").strip()
    if not frac:
        return 0
    if len(frac) < 3:
        frac = frac.ljust(3, "0")
    else:
        frac = frac[:3]
    return int(frac)


def srt_timestamp_to_ms(ts: str) -> int:
    """Parse SRT time token (HH:MM:SS,mmm or HH:MM:SS.mmm)."""
    ts = (ts or "").strip().replace(".", ",")
    m = re.match(rf"^{_SRT_COMPONENT}$", ts, re.IGNORECASE)
    if not m:
        raise ValueError(f"Invalid SRT timestamp: {ts!r}")
    h, mi, s, frac = m.groups()
    return int(h) * 3_600_000 + int(mi) * 60_000 + int(s) * 1_000 + _frac_to_ms(frac)


def ms_to_srt_timestamp(ms: int) -> str:
    if ms < 0:
        ms = 0
    h = ms // 3_600_000
    rem = ms % 3_600_000
    m = rem // 60_000
    rem %= 60_000
    s = rem // 1000
    frac = rem % 1000
    return f"{h:02d}:{m:02d}:{s:02d},{frac:03d}"


def parse_srt(content: str) -> List[SubtitleCue]:
    """Parse standard SRT (with optional numeric index lines)."""
    text = (content or "").lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    cues: List[SubtitleCue] = []
    i = 0
    n = len(lines)
    while i < n:
        while i < n and not lines[i].strip():
            i += 1
        if i >= n:
            break
        if re.match(r"^\d+\s*$", lines[i].strip()):
            i += 1
            if i >= n:
                break
        time_line = lines[i].strip()
        i += 1
        m = _SRT_LINE_RE.match(time_line)
        if not m:
            continue
        try:
            start = (
                int(m.group(1)) * 3_600_000
                + int(m.group(2)) * 60_000
                + int(m.group(3)) * 1_000
                + _frac_to_ms(m.group(4))
            )
            end = (
                int(m.group(5)) * 3_600_000
                + int(m.group(6)) * 60_000
                + int(m.group(7)) * 1_000
                + _frac_to_ms(m.group(8))
            )
        except (TypeError, ValueError):
            i += 1
            continue
        body_lines: List[str] = []
        while i < n and lines[i].strip():
            body_lines.append(lines[i])
            i += 1
        body = "\n".join(body_lines).strip()
        if body:
            cues.append(SubtitleCue(start_ms=start, end_ms=end, text=body))
        while i < n and not lines[i].strip():
            i += 1
    return cues


def parse_timestamped_txt(content: str) -> List[SubtitleCue]:
    """Parse lines like: [00:01:02,345 --> 00:01:05,678] Text"""
    text = (content or "").lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    cues: List[SubtitleCue] = []
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        m = _TXT_CUE_RE.match(line)
        if not m:
            continue
        try:
            start = (
                int(m.group(1)) * 3_600_000
                + int(m.group(2)) * 60_000
                + int(m.group(3)) * 1_000
                + _frac_to_ms(m.group(4))
            )
            end = (
                int(m.group(5)) * 3_600_000
                + int(m.group(6)) * 60_000
                + int(m.group(7)) * 1_000
                + _frac_to_ms(m.group(8))
            )
        except (TypeError, ValueError):
            continue
        body = (m.group(9) or "").strip()
        if body:
            cues.append(SubtitleCue(start_ms=start, end_ms=end, text=body))
    return cues


def detect_subtitle_text_format(content: str) -> Literal["srt", "txt"]:
    """Heuristic: bracketed lines vs classic SRT blocks."""
    lines = [ln for ln in (content or "").splitlines() if ln.strip()]
    if not lines:
        return "srt"
    txt_hits = sum(1 for ln in lines if _TXT_CUE_RE.match(ln.strip()))
    if txt_hits >= max(1, (len(lines) + 1) // 2):
        return "txt"
    return "srt"


def parse_subtitle_content(content: str, prefer: FormatHint = "auto") -> Tuple[List[SubtitleCue], Literal["srt", "txt"]]:
    """
    Parse subtitle text. ``prefer`` can force SRT or bracketed TXT; ``auto`` picks by content.
    """
    if prefer == "srt":
        return parse_srt(content), "srt"
    if prefer == "txt":
        return parse_timestamped_txt(content), "txt"
    fmt = detect_subtitle_text_format(content)
    if fmt == "txt":
        return parse_timestamped_txt(content), "txt"
    return parse_srt(content), "srt"


def write_srt(cues: List[SubtitleCue], lines: List[str]) -> str:
    """Build SRT document text. ``lines`` must align with ``cues``."""
    if len(cues) != len(lines):
        raise ValueError(f"cue count {len(cues)} != line count {len(lines)}")
    out: List[str] = []
    idx = 1
    for cue, txt in zip(cues, lines):
        t = (txt or "").strip()
        if not t:
            continue
        out.append(str(idx))
        out.append(
            f"{ms_to_srt_timestamp(cue.start_ms)} --> {ms_to_srt_timestamp(cue.end_ms)}"
        )
        out.append(t)
        out.append("")
        idx += 1
    return "\n".join(out).rstrip() + ("\n" if out else "")


def write_timestamped_txt(cues: List[SubtitleCue], lines: List[str]) -> str:
    """Build bracketed timestamp lines."""
    if len(cues) != len(lines):
        raise ValueError(f"cue count {len(cues)} != line count {len(lines)}")
    rows: List[str] = []
    for cue, txt in zip(cues, lines):
        t = (txt or "").strip()
        if not t:
            continue
        a = ms_to_srt_timestamp(cue.start_ms)
        b = ms_to_srt_timestamp(cue.end_ms)
        rows.append(f"[{a} --> {b}] {t}")
    return "\n".join(rows) + ("\n" if rows else "")
