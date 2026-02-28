"""
CUHK Manga Inpainting – Seamless Manga Inpainting with Semantics Awareness (SIGGRAPH 2021).
Requires MangaInpainting repo + checkpoints; uses image + mask + line map (line map is auto-generated if missing).
Install: clone https://github.com/msxie92/MangaInpainting, download checkpoints (see README), set repo path.
"""
import os
import sys
import subprocess
import tempfile
import shutil
import numpy as np
import cv2
from typing import List

from .base import InpainterBase, register_inpainter, TextBlock


def _simple_line_extraction(img: np.ndarray) -> np.ndarray:
    """Produce a simple structural line map (black lines on white) for manga. No extra deps."""
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = np.copy(img)
    # Emphasize edges; then invert so lines are black on white (manga line art style)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 50, 150)
    line_map = 255 - edges
    return line_map


@register_inpainter("cuhk_manga_inpaint")
class CuhkMangaInpainter(InpainterBase):
    """
    CUHK Seamless Manga Inpainting (SIGGRAPH 2021). Best on high-quality manga when line map is good.
    Requires MangaInpainting repo and checkpoints; line map is auto-generated from the image.
    """
    inpaint_by_block = False
    check_need_inpaint = True

    params = {
        "repo_path": {
            "type": "line_editor",
            "value": "",
            "description": "Path to MangaInpainting repo (contains test.py and src/).",
        },
        "checkpoints_path": {
            "type": "line_editor",
            "value": "",
            "description": "Path to checkpoints (e.g. repo/checkpoints/mangainpaintor). Leave empty to use repo_path/checkpoints/mangainpaintor.",
        },
        "model": {
            "type": "selector",
            "options": [1, 2, 3, 4],
            "value": 4,
            "description": "Model: 1=semantic, 2=manga, 3=manga+fixed semantic, 4=joint (default).",
        },
        "line_extraction": {
            "type": "selector",
            "options": ["simple", "canny"],
            "value": "simple",
            "description": "Line map: simple (Canny-based) or canny (stronger edges).",
        },
        "description": "CUHK Manga Inpainting (SIGGRAPH 2021). Set repo path and checkpoints.",
    }
    _load_model_keys = set()

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self._repo_path = None
        self._checkpoints_path = None

    def _get_paths(self):
        repo = (self.params.get("repo_path") or {}).get("value", "").strip()
        ckpt = (self.params.get("checkpoints_path") or {}).get("value", "").strip()
        if not repo:
            return None, None
        if not ckpt:
            ckpt = os.path.join(repo, "checkpoints", "mangainpaintor")
        return repo, ckpt

    def _inpaint(self, img: np.ndarray, mask: np.ndarray, textblock_list: List[TextBlock] = None) -> np.ndarray:
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask_bin = (mask > 127).astype(np.uint8) * 255
        repo_path, ckpt_path = self._get_paths()
        if not repo_path or not os.path.isdir(repo_path):
            self.logger.warning("CUHK Manga: repo path not set or invalid. Set repo_path in inpainter settings.")
            return img.copy()
        if not os.path.isdir(ckpt_path):
            self.logger.warning("CUHK Manga: checkpoints path not found. Download from MangaInpainting README.")
            return img.copy()
        test_py = os.path.join(repo_path, "test.py")
        if not os.path.isfile(test_py):
            self.logger.warning("CUHK Manga: test.py not found in repo path.")
            return img.copy()

        line_map = _simple_line_extraction(img)
        if (self.params.get("line_extraction") or {}).get("value") == "canny":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
            line_map = 255 - cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 30, 100)

        tmp = tempfile.mkdtemp(prefix="cuhk_manga_")
        try:
            input_dir = os.path.join(tmp, "input")
            mask_dir = os.path.join(tmp, "mask")
            line_dir = os.path.join(tmp, "line")
            out_dir = os.path.join(tmp, "out")
            for d in (input_dir, mask_dir, line_dir, out_dir):
                os.makedirs(d, exist_ok=True)
            base_name = "in.png"
            cv2.imwrite(os.path.join(input_dir, base_name), img)
            cv2.imwrite(os.path.join(mask_dir, base_name), mask_bin)
            cv2.imwrite(os.path.join(line_dir, base_name), line_map)

            model_val = 4
            m = self.params.get("model")
            if isinstance(m, dict):
                model_val = int(m.get("value", 4))
            cmd = [
                sys.executable,
                test_py,
                "--path", ckpt_path,
                "--model", str(model_val),
                "--input", input_dir,
                "--mask", mask_dir,
                "--line", line_dir,
                "--output", out_dir,
            ]
            env = os.environ.copy()
            env["PYTHONPATH"] = repo_path if not env.get("PYTHONPATH") else repo_path + os.pathsep + env["PYTHONPATH"]
            result = subprocess.run(
                cmd,
                cwd=repo_path,
                env=env,
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode != 0:
                self.logger.error(f"CUHK Manga test.py failed: {result.stderr or result.stdout}")
                return img.copy()
            # Output: save_images writes to results_path/fld_name/name (name = in.png)
            out_path = os.path.join(out_dir, base_name)
            if not os.path.isfile(out_path):
                found = None
                for root, _, files in os.walk(out_dir):
                    for f in files:
                        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                            found = os.path.join(root, f)
                            break
                    if found:
                        break
                out_path = found or out_path
            if not os.path.isfile(out_path):
                self.logger.error("CUHK Manga: no output file produced.")
                return img.copy()
            out_img = cv2.imread(out_path)
            if out_img is None:
                out_img = cv2.imread(out_path, cv2.IMREAD_UNCHANGED)
            if out_img is not None and (out_img.shape[0] != img.shape[0] or out_img.shape[1] != img.shape[1]):
                out_img = cv2.resize(out_img, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
            if out_img is not None:
                return out_img
        except subprocess.TimeoutExpired:
            self.logger.error("CUHK Manga: test.py timed out.")
        except Exception as e:
            self.logger.error(f"CUHK Manga inpainting failed: {e}")
        finally:
            try:
                shutil.rmtree(tmp, ignore_errors=True)
            except Exception:
                pass
        return img.copy()
