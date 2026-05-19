from __future__ import annotations

import os
import os.path as osp
from typing import Dict, Iterable, List, Tuple

from PIL import Image
import numpy as np
from utils.mask_cleanup_quality import adaptive_mask_expand, merge_masks_with_confidence


def _build_rect_mask(blocks, width: int, height: int):
    mask = np.zeros((height, width), dtype=np.uint8)
    for b in blocks or []:
        x1,y1,x2,y2 = [int(v) for v in getattr(b, "xyxy", [0,0,0,0])]
        x1,y1 = max(0,x1), max(0,y1)
        x2,y2 = min(width, max(x1+1,x2)), min(height, max(y1+1,y2))
        mask[y1:y2, x1:x2] = 255
    return mask


def run_cleanup_only_pages(project, detector, inpainter, pages: Iterable[str], *, out_dir: str = "", halo_threshold: float = 0.18, inside_radius: int = 1, outside_radius: int = 2, detector_confidence: float = 1.0) -> Dict[str, object]:
    pages = [str(p) for p in (pages or []) if str(p)]
    exported: List[Tuple[str, str]] = []
    warnings: List[str] = []
    processed = 0
    failed = 0
    halo_flags: List[str] = []
    halo_stats: List[Dict[str, object]] = []
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    for page in pages:
        try:
            img = project.read_img(page)
            det_mask, blks = detector.detect(img, project)
            rect_mask = _build_rect_mask(blks or [], img.shape[1], img.shape[0])
            mask = merge_masks_with_confidence(rect_mask, det_mask, confidence=detector_confidence)
            mask = adaptive_mask_expand(mask, inside_radius=inside_radius, outside_radius=outside_radius)
            try:
                from modules.mask_diagnostics import build_mask_diagnostics
                diag = build_mask_diagnostics(mask, threshold=127, dilate_iter=1) or {}
                stats = (diag.get("stats") or {})
            except Exception:
                stats = {}
            halo_ratio = float(stats.get("edge_halo_ratio", 0.0) or 0.0)
            halo_stats.append({"page": page, "edge_halo_ratio": halo_ratio, "mask_fill_ratio": float(stats.get("mask_fill_ratio", 0.0) or 0.0)})
            if halo_ratio >= float(halo_threshold):
                halo_flags.append(page)
                warnings.append(f"{page}: edge halo ratio {halo_ratio:.3f} >= threshold {float(halo_threshold):.3f}")
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
        'halo_flags': halo_flags,
        'halo_stats': halo_stats,
        'mask_policy': {'inside_radius': int(inside_radius), 'outside_radius': int(outside_radius), 'detector_confidence': float(detector_confidence)},
    }
