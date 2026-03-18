"""
Optional vocal separation before ASR (VideoCaptioner-style).
Uses demucs (pip install demucs) to separate vocals from music; improves ASR on noisy audio.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from typing import Optional


def separate_vocals(
    audio_path: str,
    output_path: Optional[str] = None,
    device: str = "cuda",
    model: str = "htdemucs",
) -> Optional[str]:
    """
    Separate vocals from audio using demucs (optional dependency).
    Returns path to vocals-only WAV, or None if demucs fails / not installed.
    """
    if not audio_path or not os.path.isfile(audio_path):
        return None
    out_dir = tempfile.mkdtemp(prefix="demucs_")
    try:
        cmd = [
            sys.executable, "-m", "demucs",
            "--two-stems=vocals",
            "-n", model,
            "-o", out_dir,
            audio_path,
        ]
        if device and device.lower() == "cpu":
            cmd.insert(-1, "--device=cpu")
        subprocess.run(cmd, check=True, capture_output=True, timeout=600)
        stem = os.path.splitext(os.path.basename(audio_path))[0]
        vocals_path = os.path.join(out_dir, model, stem, "vocals.wav")
        if os.path.isfile(vocals_path):
            if output_path is None:
                fd, output_path = tempfile.mkstemp(suffix=".wav")
                os.close(fd)
            shutil.copy2(vocals_path, output_path)
            return output_path
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        pass
    except Exception:
        pass
    finally:
        try:
            shutil.rmtree(out_dir, ignore_errors=True)
        except Exception:
            pass
    return None
