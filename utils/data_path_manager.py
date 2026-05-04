from __future__ import annotations
import os
import shutil


def resolve_data_path(override_path: str = "") -> str:
    p = (override_path or "").strip()
    if p:
        return os.path.abspath(os.path.expanduser(p))
    return os.path.abspath("data")


def ensure_data_path(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def free_space_gb(path: str) -> float:
    target = path if os.path.isdir(path) else os.path.dirname(path) or "."
    st = shutil.disk_usage(target)
    return float(st.free) / float(1024 ** 3)
