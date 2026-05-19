from __future__ import annotations

import os
import os.path as osp
import zipfile
from typing import Callable, Dict, List, Tuple


def collect_files(root_dir: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for root, _dirs, files in os.walk(root_dir):
        for fn in sorted(files):
            full = osp.join(root, fn)
            rel = osp.relpath(full, root_dir)
            out.append((full, rel))
    return out


def write_archive_streaming(
    source_dir: str,
    archive_path: str,
    *,
    cancel_check: Callable[[], bool] | None = None,
    progress_hook: Callable[[Dict[str, object]], None] | None = None,
) -> Dict[str, object]:
    files = collect_files(source_dir)
    os.makedirs(osp.dirname(archive_path) or '.', exist_ok=True)
    total = len(files)
    written = 0
    progress: List[Dict[str, object]] = []
    cancelled = False
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for idx, (full, rel) in enumerate(files, start=1):
            if cancel_check is not None and bool(cancel_check()):
                cancelled = True
                break
            zf.write(full, rel)
            written += 1
            event = {'index': idx, 'total': total, 'relative_path': rel, 'progress': float(idx) / float(max(1, total))}
            progress.append(event)
            if progress_hook is not None:
                try:
                    progress_hook(dict(event))
                except Exception:
                    pass
    return {
        'archive_path': archive_path,
        'source_dir': source_dir,
        'total_files': total,
        'written_files': written,
        'cancelled': cancelled,
        'progress_events': progress,
    }
