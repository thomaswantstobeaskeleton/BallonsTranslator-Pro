"""
CBR (Comic Book RAR) support. Optional dependency: rarfile.
Extract RAR/CBR to folder and open as project (same flow as CBZ).
"""
from __future__ import annotations

import os
import tempfile
import os.path as osp
from pathlib import Path
from typing import Optional, Tuple, List

from .logger import logger as LOGGER

CBR_EXT = (".cbr", ".rar")


def extract_cbr_to_folder(cbr_path: str, out_dir: Optional[str] = None) -> str:
    """
    Extract a CBR/RAR archive to out_dir. If out_dir is None, use a temp directory.
    Requires: pip install rarfile. WinRAR or 7-Zip (with UnRAR) must be in PATH.
    Returns the path to the extracted folder.
    """
    try:
        import rarfile
    except ImportError:
        raise RuntimeError(
            "CBR support requires the 'rarfile' package. Install with: pip install rarfile. "
            "You also need WinRAR or 7-Zip (with UnRAR) in your system PATH for RAR extraction."
        )
    cbr_path = osp.abspath(cbr_path)
    if not osp.isfile(cbr_path):
        raise FileNotFoundError(cbr_path)
    if out_dir is None:
        base = Path(tempfile.gettempdir()) / "ballonstranslator_cbr"
        base.mkdir(parents=True, exist_ok=True)
        out_dir = tempfile.mkdtemp(prefix=Path(cbr_path).stem + "_", dir=str(base))
    else:
        out_dir = osp.abspath(out_dir)
        os.makedirs(out_dir, exist_ok=True)
    with rarfile.RarFile(cbr_path, "r") as rf:
        rf.extractall(out_dir)
    LOGGER.info("Extracted CBR to %s", out_dir)
    return out_dir
