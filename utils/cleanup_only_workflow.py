from __future__ import annotations

import os
import os.path as osp
from typing import Dict, Iterable, List, Tuple

from PIL import Image
import numpy as np


def _build_rect_mask(blocks, width: int, height: int):
    mask = np.zeros((height, width), dtype=np.uint8)
    for b in blocks or []:
        x1,y1,x2,y2 = [int(v) for v in getattr(b, "xyxy", [0,0,0,0])]
        x1,y1 = max(0,x1), max(0,y1)
        x2,y2 = min(width, max(x1+1,x2)), min(height, max(y1+1,y2))
        mask[y1:y2, x1:x2] = 255
    return mask


def run_cleanup_only_pages(project, detector, inpainter, pages: Iterable[str], *, out_dir: str = "") -> Dict[str, object]:
    pages = [str(p) for p in (pages or []) if str(p)]
    exported: List[Tuple[str, str]] = []
    warnings: List[str] = []
    processed = 0
    failed = 0
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    for page in pages:
        try:
            img = project.read_img(page)
            _mask, blks = detector.detect(img, project)
            mask = _build_rect_mask(blks or [], img.shape[1], img.shape[0])
            clean = inpainter.inpaint(img, mask)
            project.save_inpainted(page, clean)
            if out_dir:
                out_path = osp.join(out_dir, f"{osp.splitext(osp.basename(page))[0]}_clean.png")
                arr = np.asarray(clean)
                Image.fromarray(arr).save(out_path)
                exported.append((page, out_path))
            processed += 1
        except Exception as e:
            failed += 1
            warnings.append(f"{page}: {e}")
    return {
        'ok': failed == 0,
        'processed': processed,
        'failed': failed,
        'warnings': warnings,
        'exported': exported,
    }
