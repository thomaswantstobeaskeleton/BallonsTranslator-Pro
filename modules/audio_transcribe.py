"""
Audio transcription for video translator (ASR).
Extract audio from video and transcribe with faster-whisper (optional dependency).
Inspired by VideoCaptioner (Whisper / faster-whisper).
"""
from __future__ import annotations

import os
import tempfile
import subprocess
from typing import List, Tuple, Optional

# Optional: faster-whisper (pip install faster-whisper)
try:
    from faster_whisper import WhisperModel
    HAS_FASTER_WHISPER = True
except ImportError:
    HAS_FASTER_WHISPER = False


def extract_audio_from_video(
    video_path: str,
    output_path: Optional[str] = None,
    ffmpeg_path: str = "ffmpeg",
) -> Optional[str]:
    """Extract audio from video to a WAV file. Returns path to WAV or None on failure."""
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
    cmd = [
        ffmpeg_path, "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        output_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=600)
        return output_path
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None


def transcribe_audio(
    audio_path: str,
    model_size: str = "base",
    device: str = "cuda",
    language: Optional[str] = None,
    compute_type: str = "float16",
    vad_filter: bool = True,
) -> List[Tuple[float, float, str]]:
    """
    Transcribe audio with faster-whisper. Returns list of (start_sec, end_sec, text).
    model_size: tiny, base, small, medium, large-v2, large-v3, etc.
    vad_filter: when True, filter non-speech segments to reduce hallucinations (VideoCaptioner-style).
    """
    if not HAS_FASTER_WHISPER:
        raise RuntimeError(
            "faster-whisper is required for ASR. Install with: pip install faster-whisper"
        )
    if device == "gpu":
        device = "cuda"
    if device != "cuda":
        compute_type = "int8"
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, info = model.transcribe(
        audio_path, language=language, word_timestamps=False, vad_filter=vad_filter
    )
    result = []
    for s in segments:
        result.append((s.start, s.end, (s.text or "").strip()))
    return result
