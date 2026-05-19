from __future__ import annotations

import json
import os
import os.path as osp
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def build_export_manifest(
    project,
    out_dir: str,
    exported_paths: Sequence[Tuple[str, str]],
    missing_pages: Optional[Iterable[str]] = None,
    export_kind: str = "rendered_images",
    options: Optional[Dict] = None,
) -> Dict:
    """Build a stable export status manifest for batch/headless workflows."""
    missing = list(missing_pages or [])
    options = dict(options or {})
    export_sources = options.get("export_sources", {}) or {}
    pages = []
    for idx, (page_name, path) in enumerate(exported_paths or [], start=1):
        source_kind = str(export_sources.get(page_name, "rendered") or "rendered")
        pages.append({
            "index": idx,
            "page": page_name,
            "path": osp.abspath(path),
            "relative_path": osp.relpath(path, out_dir) if out_dir else path,
            "exists": osp.exists(path),
            "source_kind": source_kind,
            "used_fallback_source": source_kind != "rendered",
            "completion_state": project.get_page_completion_state(page_name) if project is not None and hasattr(project, "get_page_completion_state") else "",
        })
    renderer_info = options.get("renderer") or {}
    manifest = {
        "format": "ballonstranslator.export_manifest.v1",
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "project_dir": osp.abspath(getattr(project, "directory", "") or ""),
        "project_path": osp.abspath(getattr(project, "proj_path", "") or "") if getattr(project, "proj_path", "") else "",
        "export_kind": export_kind,
        "out_dir": osp.abspath(out_dir),
        "page_count": len(pages),
        "missing_count": len(missing),
        "pages": pages,
        "missing_pages": missing,
        "options": options,
        "warnings": [],
        "renderer": renderer_info,
    }
    fallback_count = sum(1 for page in pages if page.get("used_fallback_source"))
    if fallback_count:
        manifest["warnings"].append(f"{fallback_count} page(s) used inpainted/original fallback sources because rendered results were missing.")
    if missing:
        manifest["warnings"].append(f"{len(missing)} page(s) had no rendered result/source at export time.")
    return manifest




def mark_exported_pages(project, exported_paths: Sequence[Tuple[str, str]]) -> int:
    """Mark successfully exported pages as `exported` when project state supports it."""
    if project is None or not hasattr(project, "set_page_completion_state"):
        return 0
    marked = 0
    for page_name, path in exported_paths or []:
        if path and osp.exists(path):
            project.set_page_completion_state(page_name, "exported")
            marked += 1
    return marked


def write_export_manifest(
    project,
    out_dir: str,
    exported_paths: Sequence[Tuple[str, str]],
    missing_pages: Optional[Iterable[str]] = None,
    export_kind: str = "rendered_images",
    options: Optional[Dict] = None,
    filename: str = "export_manifest.json",
) -> Dict:
    os.makedirs(out_dir, exist_ok=True)
    manifest = build_export_manifest(project, out_dir, exported_paths, missing_pages, export_kind, options)
    path = osp.join(out_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    manifest["manifest_path"] = path
    return manifest
