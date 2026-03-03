"""
Collision-based text block merging (Dango-style).
Use when OCR or detector returns many small/word-level boxes that should be
grouped into reading-order blocks. Optional; off by default.
"""

from typing import List, Tuple
import numpy as np

from .textblock import TextBlock


def _box_xyxy(blk: TextBlock) -> Tuple[float, float, float, float]:
    """Get (x1, y1, x2, y2) from a TextBlock."""
    xyxy = getattr(blk, 'xyxy', None)
    if xyxy and len(xyxy) >= 4:
        return float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
    if getattr(blk, 'lines', None) and len(blk.lines) > 0:
        pts = np.vstack(blk.lines)
        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)
        return x1, y1, x2, y2
    return 0.0, 0.0, 0.0, 0.0


def _collision(
    x1: float, y1: float, x2: float, y2: float,
    x1b: float, y1b: float, x2b: float, y2b: float,
) -> bool:
    """AABB collision."""
    return x1 < x2b and x2 > x1b and y1 < y2b and y2 > y1b


def merge_blocks_horizontal(
    blk_list: List[TextBlock],
    gap_ratio: float = 1.5,
    add_space_between: bool = False,
) -> List[TextBlock]:
    """
    Merge blocks that are close vertically (same line or nearby lines) using
    expanded bbox collision. Good for horizontal text (LTR/TTB).
    gap_ratio: expand each box vertically by (height * gap_ratio) for collision; 1.5 = Dango-style.
    add_space_between: insert space when concatenating text (e.g. for English).
    """
    if not blk_list:
        return []
    if len(blk_list) == 1:
        return list(blk_list)

    used = [False] * len(blk_list)
    merged: List[TextBlock] = []

    for i, blk in enumerate(blk_list):
        if used[i]:
            continue
        x1, y1, x2, y2 = _box_xyxy(blk)
        h = max(1e-3, y2 - y1)
        expand = h * (gap_ratio - 1.0) / 2.0
        by1, by2 = y1 - expand, y2 + expand
        group = [blk]
        used[i] = True

        changed = True
        while changed:
            changed = False
            for j, other in enumerate(blk_list):
                if used[j]:
                    continue
                ox1, oy1, ox2, oy2 = _box_xyxy(other)
                if _collision(x1, by1, x2, by2, ox1, oy1, ox2, oy2):
                    group.append(other)
                    used[j] = True
                    x1 = min(x1, ox1)
                    y1 = min(y1, oy1)
                    x2 = max(x2, ox2)
                    y2 = max(y2, oy2)
                    h = max(h, y2 - y1)
                    expand = h * (gap_ratio - 1.0) / 2.0
                    by1, by2 = y1 - expand, y2 + expand
                    changed = True
                    break

        # Build merged block: union box, concatenate text/translation, keep first block's style
        first = group[0]
        texts = []
        trans = []
        for b in group:
            t = getattr(b, 'text', None) or []
            if isinstance(t, list):
                texts.extend(t)
            else:
                texts.append(str(t))
            tr = getattr(b, 'translation', None) or ''
            if tr:
                trans.append(tr)
        # Single union bbox
        all_xyxy = [_box_xyxy(b) for b in group]
        x1 = min(a[0] for a in all_xyxy)
        y1 = min(a[1] for a in all_xyxy)
        x2 = max(a[2] for a in all_xyxy)
        y2 = max(a[3] for a in all_xyxy)

        delimiter = ' ' if add_space_between else ''
        new_text = delimiter.join(str(t).strip() for t in texts if str(t).strip())
        new_translation = '\n'.join(tr.strip() for tr in trans if tr.strip()) if trans else ''

        new_blk = TextBlock(
            xyxy=[x1, y1, x2, y2],
            lines=first.lines,
            text=[new_text] if new_text else first.text,
            translation=new_translation or first.translation,
        )
        if hasattr(first.fontformat, 'copy'):
            new_blk.fontformat = first.fontformat.copy()
        else:
            import copy
            new_blk.fontformat = copy.deepcopy(first.fontformat)
        if hasattr(first, 'language'):
            new_blk.language = first.language
        if hasattr(first, '_detected_font_size') and getattr(first, '_detected_font_size', -1) > 0:
            new_blk._detected_font_size = first._detected_font_size
        if hasattr(first, 'label'):
            new_blk.label = first.label
        merged.append(new_blk)

    return merged


def merge_blocks_vertical(
    blk_list: List[TextBlock],
    gap_ratio: float = 0.5,
) -> List[TextBlock]:
    """
    Merge blocks that are close horizontally (same column). Sort RTL then TTB.
    gap_ratio: expand each box horizontally by (width * gap_ratio) for collision.
    """
    if not blk_list:
        return []
    if len(blk_list) == 1:
        return list(blk_list)

    # Sort by center x descending (right-to-left)
    def center_x(b: TextBlock) -> float:
        x1, y1, x2, y2 = _box_xyxy(b)
        return (x1 + x2) / 2.0

    ordered = sorted(blk_list, key=center_x, reverse=True)
    used = [False] * len(ordered)
    merged: List[TextBlock] = []

    for i, blk in enumerate(ordered):
        if used[i]:
            continue
        x1, y1, x2, y2 = _box_xyxy(blk)
        w = max(1e-3, x2 - x1)
        expand = w * gap_ratio / 2.0
        bx1, bx2 = x1 - expand, x2 + expand
        group = [blk]
        used[i] = True

        for j, other in enumerate(ordered):
            if used[j]:
                continue
            ox1, oy1, ox2, oy2 = _box_xyxy(other)
            if _collision(bx1, y1, bx2, y2, ox1, oy1, ox2, oy2):
                group.append(other)
                used[j] = True
                y1 = min(y1, oy1)
                y2 = max(y2, oy2)
                x1 = min(x1, ox1)
                x2 = max(x2, ox2)
                w = max(w, x2 - x1)
                expand = w * gap_ratio / 2.0
                bx1, bx2 = x1 - expand, x2 + expand

        # Sort group top-to-bottom
        group.sort(key=lambda b: (_box_xyxy(b)[1] + _box_xyxy(b)[3]) / 2.0)

        first = group[0]
        all_xyxy = [_box_xyxy(b) for b in group]
        x1 = min(a[0] for a in all_xyxy)
        y1 = min(a[1] for a in all_xyxy)
        x2 = max(a[2] for a in all_xyxy)
        y2 = max(a[3] for a in all_xyxy)
        texts = []
        trans = []
        for b in group:
            t = getattr(b, 'text', None) or []
            if isinstance(t, list):
                texts.extend(t)
            else:
                texts.append(str(t))
            tr = getattr(b, 'translation', None) or ''
            if tr:
                trans.append(tr)
        new_text = ''.join(str(t).strip() for t in texts if str(t).strip())
        new_translation = '\n'.join(tr.strip() for tr in trans if tr.strip()) if trans else ''

        new_blk = TextBlock(
            xyxy=[x1, y1, x2, y2],
            lines=first.lines,
            text=[new_text] if new_text else first.text,
            translation=new_translation or first.translation,
        )
        if hasattr(first.fontformat, 'copy'):
            new_blk.fontformat = first.fontformat.copy()
        else:
            import copy
            new_blk.fontformat = copy.deepcopy(first.fontformat)
        new_blk.vertical = True
        if hasattr(first, 'language'):
            new_blk.language = first.language
        if hasattr(first, '_detected_font_size') and getattr(first, '_detected_font_size', -1) > 0:
            new_blk._detected_font_size = first._detected_font_size
        if hasattr(first, 'label'):
            new_blk.label = first.label
        merged.append(new_blk)

    # Sort merged blocks top-to-bottom, then right-to-left
    merged.sort(key=lambda b: (_box_xyxy(b)[1] + _box_xyxy(b)[3]) / 2.0)
    merged.sort(key=center_x, reverse=True)
    return merged
