import os
import os.path as osp
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .io_utils import IMG_EXT


def is_zip_path(path: str) -> bool:
    try:
        return osp.isfile(path) and Path(path).suffix.lower() == ".zip"
    except Exception:
        return False


def _dir_has_images(d: str) -> bool:
    try:
        for name in os.listdir(d):
            if Path(name).suffix.lower() in IMG_EXT:
                return True
    except OSError:
        return False
    return False


def _iter_image_dirs(root_dir: str) -> List[str]:
    """
    Return all directories (including root) that directly contain images.

    Note: BallonsTranslator projects are single-directory (non-recursive) image sets.
    """
    out: List[str] = []
    root_dir = osp.normpath(osp.abspath(root_dir))
    for cur, dirnames, _filenames in os.walk(root_dir):
        base = osp.basename(cur).lower()
        # Avoid treating output dirs as input projects
        if base in {"result", "mask", "inpainted"}:
            dirnames[:] = []
            continue
        if _dir_has_images(cur):
            out.append(osp.normpath(osp.abspath(cur)))
    out.sort(key=lambda p: (p.count(os.sep), p.lower()))
    return out


def _safe_mkdir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        pass


def _copy_result_images(project_dir: str, extracted_root: str, output_root: str) -> int:
    """
    Copy `<project_dir>/result/*.{img}` to `<output_root>/<relative_dir>/*.{img}`.

    Returns number of files copied.
    """
    src_result_dir = osp.join(project_dir, "result")
    if not osp.isdir(src_result_dir):
        return 0

    rel_dir = osp.relpath(project_dir, extracted_root)
    if rel_dir in {".", ""}:
        rel_dir = ""
    dst_dir = osp.join(output_root, rel_dir)
    _safe_mkdir(dst_dir)

    copied = 0
    try:
        for name in os.listdir(src_result_dir):
            if Path(name).suffix.lower() not in IMG_EXT:
                continue
            src = osp.join(src_result_dir, name)
            dst = osp.join(dst_dir, name)
            try:
                # copy2 preserves mtime which can be useful for debugging order
                import shutil

                shutil.copy2(src, dst)
                copied += 1
            except OSError:
                continue
    except OSError:
        return 0

    return copied


@dataclass
class ZipBatchJob:
    zip_path: str
    tempdir: tempfile.TemporaryDirectory
    extracted_root: str
    output_root: str
    project_dirs: List[str]
    remaining: int


def create_zip_batch_job(zip_path: str, output_root: Optional[str] = None) -> ZipBatchJob:
    """
    Extract `zip_path` to a temp directory and enumerate project directories.

    Output location defaults to `<zip_stem>_translated` alongside the zip.
    """
    zip_path = osp.normpath(osp.abspath(zip_path))
    if not is_zip_path(zip_path):
        raise ValueError(f"Not a zip file: {zip_path}")

    td = tempfile.TemporaryDirectory(prefix="ballonstrans_zip_")
    extracted_root = td.name

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extracted_root)

    if output_root is None or not str(output_root).strip():
        zip_stem = osp.splitext(osp.basename(zip_path))[0]
        output_root = osp.join(osp.dirname(zip_path), f"{zip_stem}_translated")
    output_root = osp.normpath(osp.abspath(output_root))
    _safe_mkdir(output_root)

    project_dirs = _iter_image_dirs(extracted_root)

    return ZipBatchJob(
        zip_path=zip_path,
        tempdir=td,
        extracted_root=extracted_root,
        output_root=output_root,
        project_dirs=project_dirs,
        remaining=len(project_dirs),
    )


class ZipBatchManager:
    """
    Tracks zip-derived project dirs and copies results on completion.
    """

    def __init__(self):
        self._jobs: List[ZipBatchJob] = []
        self._project_dir_to_job: Dict[str, ZipBatchJob] = {}

    def add_zip(self, zip_path: str, output_root: Optional[str] = None) -> ZipBatchJob:
        job = create_zip_batch_job(zip_path, output_root=output_root)
        self._jobs.append(job)
        for d in job.project_dirs:
            self._project_dir_to_job[d] = job
        return job

    def is_zip_project_dir(self, project_dir: str) -> bool:
        return osp.normpath(osp.abspath(project_dir)) in self._project_dir_to_job

    def on_project_finished(self, project_dir: str) -> Tuple[bool, Optional[ZipBatchJob], int]:
        """
        Copy results for this project_dir if it belongs to a zip job.

        Returns (handled, job, copied_files).
        """
        key = osp.normpath(osp.abspath(project_dir))
        job = self._project_dir_to_job.get(key)
        if job is None:
            return False, None, 0

        copied = _copy_result_images(key, job.extracted_root, job.output_root)
        job.remaining = max(0, int(job.remaining) - 1)
        # Remove mapping so we don't double-count
        try:
            del self._project_dir_to_job[key]
        except Exception:
            pass
        return True, job, copied

    def finalize_finished_jobs(self) -> List[ZipBatchJob]:
        """
        Cleanup any jobs that have no remaining projects.
        """
        finished: List[ZipBatchJob] = []
        still_running: List[ZipBatchJob] = []
        for job in self._jobs:
            if job.remaining <= 0:
                finished.append(job)
                try:
                    job.tempdir.cleanup()
                except Exception:
                    pass
            else:
                still_running.append(job)
        self._jobs = still_running
        return finished

    def cleanup_all(self) -> None:
        for job in self._jobs:
            try:
                job.tempdir.cleanup()
            except Exception:
                pass
        self._jobs.clear()
        self._project_dir_to_job.clear()

