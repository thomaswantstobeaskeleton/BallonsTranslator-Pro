from __future__ import annotations

from typing import Any, Dict, List


def _area(xyxy):
    if not xyxy or len(xyxy) != 4:
        return 0.0
    return max(0.0, float(xyxy[2] - xyxy[0]) * float(xyxy[3] - xyxy[1]))


def _iou(a, b):
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = _area(a) + _area(b) - inter
    return inter / ua if ua > 0 else 0.0


def align_translations_by_iou(raw_blocks: List[Any], translated_blocks: List[Any], *, min_iou: float = 0.2) -> Dict[str, Any]:
    matched = 0
    changed = 0
    rows = []
    for idx, raw_blk in enumerate(raw_blocks or []):
        ra = getattr(raw_blk, 'xyxy', None)
        if not ra or len(ra) != 4:
            continue
        best_i, best_txt = 0.0, ''
        for tb in translated_blocks or []:
            ta = getattr(tb, 'xyxy', None)
            if not ta or len(ta) != 4:
                continue
            i = _iou(ra, ta)
            if i > best_i:
                best_i = i
                best_txt = (getattr(tb, 'get_text', lambda: '')() or '').strip() or str(getattr(tb, 'translation', '') or '').strip()
        if best_i >= float(min_iou) and best_txt:
            matched += 1
            old = str(getattr(raw_blk, 'translation', '') or '')
            if old != best_txt:
                raw_blk.translation = best_txt
                changed += 1
            rows.append({'index': idx, 'iou': round(best_i, 4), 'text': best_txt})
    return {'matched': matched, 'changed': changed, 'rows': rows}
