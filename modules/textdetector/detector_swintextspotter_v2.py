"""
SwinTextSpotter v2 – end-to-end scene text spotting (detection + recognition).
Optional: requires cloning mxin262/SwinTextSpotterv2 and setting repo path.
Use as detector (boxes only) or full spotter (boxes + text). See docs/INSTALL_EXTRA_DETECTORS.md.
"""
import os
import sys
import tempfile
import numpy as np
import cv2
from typing import Tuple, List

from .base import register_textdetectors, TextDetectorBase, TextBlock, ProjImgTrans
from ..base import DEVICE_SELECTOR

@register_textdetectors("swintextspotter_v2")
class SwinTextSpotterV2Detector(TextDetectorBase):
    """
    SwinTextSpotter v2 – end-to-end text spotting. Optional: set repo_path to cloned SwinTextSpotterv2.
    Returns detection boxes; if spotter outputs text, those can be used (run with a no-op OCR or use as detector only).
    See docs/INSTALL_EXTRA_DETECTORS.md for setup.
    """
    params = {
        "repo_path": {
            "type": "line_editor",
            "value": "",
            "description": "Path to cloned SwinTextSpotterv2 repo (mxin262/SwinTextSpotterv2). Leave empty to disable.",
        },
        "device": DEVICE_SELECTOR(),
        "description": "SwinTextSpotter v2 (optional). Set repo_path. See docs/INSTALL_EXTRA_DETECTORS.md.",
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
            self._available = os.path.isdir(repo) and os.path.isfile(os.path.join(repo, "demo", "demo.py"))
            self._repo_path = repo
        return self._available

    def _detect(self, img: np.ndarray, proj: ProjImgTrans = None) -> Tuple[np.ndarray, List[TextBlock]]:
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        blk_list: List[TextBlock] = []
        if not self.all_model_loaded():
            self.logger.warning(
                "SwinTextSpotter v2: set repo_path to cloned SwinTextSpotterv2. See docs/INSTALL_EXTRA_DETECTORS.md."
            )
            return mask, blk_list
        # Subprocess fallback: run repo's demo if it accepts image path and writes output
        repo_path = (self.params.get("repo_path") or {}).get("value", "").strip()
        tmp_dir = tempfile.mkdtemp()
        tmp_img = os.path.join(tmp_dir, "input.png")
        out_dir = os.path.join(tmp_dir, "out")
        os.makedirs(out_dir, exist_ok=True)
        try:
            cv2.imwrite(tmp_img, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            demo_py = os.path.join(repo_path, "demo", "demo.py")
            if not os.path.isfile(demo_py):
                self.logger.warning("SwinTextSpotter v2: demo/demo.py not found in repo.")
                return mask, blk_list
            import subprocess
            env = os.environ.copy()
            env["PYTHONPATH"] = repo_path + os.pathsep + env.get("PYTHONPATH", "")
            proc = subprocess.run(
                [sys.executable, demo_py, "--image", tmp_img, "--output", out_dir],
                cwd=repo_path,
                env=env,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if proc.returncode != 0:
                self.logger.warning(f"SwinTextSpotter v2 demo failed: {proc.stderr[:500] if proc.stderr else proc.stdout[:500]}")
                return mask, blk_list
            # Parse output: repo may write visualizations or JSON; adapt to actual output format
            out_json = os.path.join(out_dir, "input.json")
            if os.path.isfile(out_json):
                import json
                with open(out_json, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Common keys: boxes, polygons, texts
                boxes = data.get("boxes", data.get("polys", []))
                texts = data.get("texts", [])
                if isinstance(boxes, list) and boxes:
                    for i, box in enumerate(boxes):
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
                            if i < len(texts) and texts[i]:
                                blk.text = [str(texts[i])]
                            blk._detected_font_size = max(y2 - y1, 12)
                            blk_list.append(blk)
                            cv2.fillPoly(mask, [pts.astype(np.int32)], 255)
        except Exception as e:
            self.logger.warning(f"SwinTextSpotter v2: {e}")
        finally:
            try:
                import shutil
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass
        return mask, blk_list
