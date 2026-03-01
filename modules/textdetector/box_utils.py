"""Shared utilities for text block box manipulation (expand, inset, etc.)."""
import copy
from typing import List

from .base import TextBlock


def expand_blocks(blk_list: List[TextBlock], padding: int, img_w: int, img_h: int) -> List[TextBlock]:
    """
    Expand each block outward by padding pixels on all sides.
    Reduces clipped punctuation (e.g. ? !) and character edges at box boundaries.
    Recommended padding: 4–6 pixels.
    """
    if padding <= 0 or not blk_list:
        return blk_list
    out = []
    for blk in blk_list:
        x1, y1, x2, y2 = blk.xyxy
        x1n = max(0, x1 - padding)
        y1n = max(0, y1 - padding)
        x2n = min(img_w, x2 + padding)
        y2n = min(img_h, y2 + padding)
        if x2n <= x1n or y2n <= y1n:
            out.append(blk)
            continue
        new_blk = copy.copy(blk)
        new_blk.xyxy = [x1n, y1n, x2n, y2n]
        new_blk.lines = [[[x1n, y1n], [x2n, y1n], [x2n, y2n], [x1n, y2n]]]
        out.append(new_blk)
    return out
