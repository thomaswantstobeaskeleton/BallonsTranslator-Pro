"""Environment + dependency doctor checks."""
from __future__ import annotations
import importlib.util
import os
from typing import List, Tuple


def _has(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except Exception:
        return False


def run_environment_doctor() -> List[Tuple[str, str, str]]:
    checks: List[Tuple[str, str, str]] = []
    checks.append(("python", "ok", os.sys.version.split()[0]))
    for mod in ["torch", "qtpy", "cv2", "numpy", "PIL"]:
        present = _has(mod)
        checks.append((f"module:{mod}", "ok" if present else "warn", "installed" if present else "missing"))
    hf = bool((os.environ.get("HF_TOKEN") or "").strip())
    checks.append(("auth:huggingface", "ok" if hf else "warn", "token set" if hf else "HF_TOKEN not set"))
    return checks
