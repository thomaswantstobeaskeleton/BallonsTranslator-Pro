"""
DPText-DETR – Transformer-based scene text detection (AAAI 2023).
Optional: requires cloning ymy-k/DPText-DETR and setting repo path.
Detection only; use with any OCR. See docs/INSTALL_EXTRA_DETECTORS.md.
"""
import os
import sys
import tempfile
import numpy as np
import cv2
from typing import Tuple, List

from .base import register_textdetectors, TextDetectorBase, TextBlock, ProjImgTrans
from ..base import DEVICE_SELECTOR

_DPTEXT_DETR_AVAILABLE = False


@register_textdetectors("dptext_detr")
class DPTextDETRDetector(TextDetectorBase):
    """
    DPText-DETR – scene text detection (dynamic point queries). Optional: set repo_path to cloned ymy-k/DPText-DETR.
    Detection only; pair with any OCR. See docs/INSTALL_EXTRA_DETECTORS.md for setup.
    """
    params = {
        "repo_path": {
            "type": "line_editor",
            "value": "",
            "description": "Path to cloned DPText-DETR repo (ymy-k/DPText-DETR). Leave empty to disable.",
        },
        "device": DEVICE_SELECTOR(),
        "score_thresh": {
            "type": "line_editor",
            "value": 0.5,
            "description": "Min detection score (0.3–0.7).",
        },
        "description": "DPText-DETR (optional). Set repo_path. See docs/INSTALL_EXTRA_DETECTORS.md.",
    }
    _load_model_keys = set()

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self._repo_path = None
        self._available = False

    def all_model_loaded(self) -> bool:
        repo = (self.params.get("repo_path") or {}).get("value", "").strip()
        if not repo:
            return False
        if repo != self._repo_path:
            self._available = os.path.isdir(repo) and os.path.isfile(os.path.join(repo, "README.md"))
            self._repo_path = repo
        return self._available

    def _detect(self, img: np.ndarray, proj: ProjImgTrans = None) -> Tuple[np.ndarray, List[TextBlock]]:
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        blk_list: List[TextBlock] = []
        if not self.all_model_loaded():
            self.logger.warning(
                "DPText-DETR: set repo_path to cloned ymy-k/DPText-DETR. See docs/INSTALL_EXTRA_DETECTORS.md."
            )
            return mask, blk_list
        repo_path = (self.params.get("repo_path") or {}).get("value", "").strip()
        tmp_dir = tempfile.mkdtemp()
        tmp_img = os.path.join(tmp_dir, "input.png")
        out_path = os.path.join(tmp_dir, "result.json")
        try:
            cv2.imwrite(tmp_img, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            # DPText-DETR may have eval/demo script; adapt to actual CLI
            demo_scripts = [
                os.path.join(repo_path, "demo.py"),
                os.path.join(repo_path, "tools", "demo.py"),
                os.path.join(repo_path, "eval.py"),
            ]
            demo_py = None
            for p in demo_scripts:
                if os.path.isfile(p):
                    demo_py = p
                    break
            if not demo_py:
                self.logger.warning("DPText-DETR: no demo/eval script found in repo.")
                return mask, blk_list
            import subprocess
            env = os.environ.copy()
            env["PYTHONPATH"] = repo_path + os.pathsep + env.get("PYTHONPATH", "")
            proc = subprocess.run(
                [sys.executable, demo_py, "--image", tmp_img, "--output", out_path],
                cwd=repo_path,
                env=env,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if proc.returncode != 0:
                # Try without --output in case script uses different args
                proc = subprocess.run(
                    [sys.executable, demo_py, tmp_img],
                    cwd=repo_path,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
            if proc.returncode != 0:
                self.logger.warning(f"DPText-DETR run failed: {proc.stderr[:500] if proc.stderr else 'no stderr'}")
                return mask, blk_list
            if os.path.isfile(out_path):
                import json
                with open(out_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                # Some repos print to stdout; try parsing
                data = {}
            boxes = data.get("boxes", data.get("polys", data.get("results", [])))
            scores = data.get("scores", [])
            if isinstance(boxes, list) and boxes:
                thresh = 0.5
                try:
                    t = self.params.get("score_thresh", {})
                    if isinstance(t, dict):
                        thresh = float(t.get("value", 0.5))
                except (TypeError, ValueError):
                    pass
                for i, box in enumerate(boxes):
                    score = float(scores[i]) if i < len(scores) else 1.0
                    if score < thresh:
                        continue
                    if isinstance(box, (list, tuple)) and len(box) >= 4:
                        pts = np.array(box, dtype=np.int32)
                        if pts.ndim == 1:
                            pts = pts.reshape(-1, 2)
                        x1, y1 = int(pts[:, 0].min()), int(pts[:, 1].min())
                        x2, y2 = int(pts[:, 0].max()), int(pts[:, 1].max())
                        if x2 <= x1 or y2 <= y1:
                            continue
                        blk = TextBlock(xyxy=[x1, y1, x2, y2], lines=[pts.tolist()])
                        blk.language = "unknown"
                        blk._detected_font_size = max(y2 - y1, 12)
                        blk_list.append(blk)
                        cv2.fillPoly(mask, [pts.astype(np.int32)], 255)
        except Exception as e:
            self.logger.warning(f"DPText-DETR: {e}")
        finally:
            try:
                import shutil
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass
        return mask, blk_list
