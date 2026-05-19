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


def describe_data_path(override_path: str = "") -> dict:
    path = resolve_data_path(override_path)
    exists = os.path.isdir(path)
    free_gb = free_space_gb(path if exists else os.path.dirname(path) or ".")
    return {
        "path": path,
        "exists": exists,
        "free_gb": round(float(free_gb), 3),
    }


def migrate_data_path(src: str, dst: str, *, dry_run: bool = True) -> dict:
    src = resolve_data_path(src)
    dst = resolve_data_path(dst)
    moved = []
    if not os.path.isdir(src):
        return {"ok": False, "error": f"source_missing:{src}", "moved": moved, "dry_run": dry_run}
    os.makedirs(dst, exist_ok=True)
    for name in sorted(os.listdir(src)):
        s = os.path.join(src, name)
        d = os.path.join(dst, name)
        moved.append({"name": name, "src": s, "dst": d, "exists_dst": os.path.exists(d)})
        if dry_run:
            continue
        if os.path.exists(d):
            continue
        os.replace(s, d)
    return {"ok": True, "source": src, "dest": dst, "moved": moved, "dry_run": dry_run}
