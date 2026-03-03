import re
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


OSB_LABELS = {"text_free", "osb", "outside", "caption", "sfx"}
BUBBLE_LABELS = {"bubble", "text_bubble"}


def _iou_xyxy(a: List[int], b: List[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = float((ix2 - ix1) * (iy2 - iy1))
    area_a = float(max(0, ax2 - ax1) * max(0, ay2 - ay1))
    area_b = float(max(0, bx2 - bx1) * max(0, by2 - by1))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _contains_xyxy(outer: List[int], inner: List[int]) -> bool:
    return outer[0] <= inner[0] and outer[1] <= inner[1] and outer[2] >= inner[2] and outer[3] >= inner[3]


def split_blocks_by_label(blk_list: list) -> Tuple[list, list, list]:
    """
    Split blocks into (bubble_like, osb_like, other) using `blk.label` when present.
    """
    bubble_like = []
    osb_like = []
    other = []
    for b in blk_list or []:
        lab = (getattr(b, "label", None) or "").strip().lower()
        if lab in BUBBLE_LABELS:
            bubble_like.append(b)
        elif lab in OSB_LABELS:
            osb_like.append(b)
        else:
            other.append(b)
    return bubble_like, osb_like, other


def _overlap_xyxy(a: List[int], b: List[int]) -> bool:
    """True if boxes overlap (intersection area > 0)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)


def expand_bubble_boxes_with_osb(
    bubble_blocks: list,
    osb_blocks: list,
    im_w: int,
    im_h: int,
) -> None:
    """
    Section 14: Expand bubble boxes to fully contain overlapping OSB text boxes
    so mask refinement (e.g. SAM) includes all text pixels; reduces "text cut off
    at bubble edge" failures. Modifies bubble blocks in place.
    """
    if not bubble_blocks or not osb_blocks or im_w <= 0 or im_h <= 0:
        return
    for bub in bubble_blocks:
        xyxy = getattr(bub, "xyxy", None)
        if not xyxy or len(xyxy) != 4:
            continue
        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
        for osb in osb_blocks:
            obb = getattr(osb, "xyxy", None)
            if not obb or len(obb) != 4:
                continue
            if not _overlap_xyxy(xyxy, obb):
                continue
            ox1, oy1, ox2, oy2 = int(obb[0]), int(obb[1]), int(obb[2]), int(obb[3])
            x1 = min(x1, ox1)
            y1 = min(y1, oy1)
            x2 = max(x2, ox2)
            y2 = max(y2, oy2)
        x1 = max(0, min(x1, im_w - 1))
        y1 = max(0, min(y1, im_h - 1))
        x2 = max(0, min(x2, im_w))
        y2 = max(0, min(y2, im_h))
        if x2 > x1 and y2 > y1:
            bub.xyxy = [x1, y1, x2, y2]


def filter_osb_overlapping_bubbles(
    osb_blocks: list,
    bubble_blocks: list,
    iou_threshold: float = 0.10,
) -> list:
    """
    Drop OSB blocks that are likely inside/overlapping speech bubbles.
    """
    if not osb_blocks or not bubble_blocks:
        return osb_blocks or []
    out = []
    for b in osb_blocks:
        bb = getattr(b, "xyxy", None)
        if not bb or len(bb) != 4:
            out.append(b)
            continue
        keep = True
        for bub in bubble_blocks:
            pb = getattr(bub, "xyxy", None)
            if not pb or len(pb) != 4:
                continue
            if _contains_xyxy(pb, bb) or _iou_xyxy(pb, bb) >= iou_threshold:
                keep = False
                break
        if keep:
            out.append(b)
    return out


def _boxes_touch(a: List[int], b: List[int], gap: int) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ax1 -= gap
    ay1 -= gap
    ax2 += gap
    ay2 += gap
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)


def group_osb_blocks(osb_blocks: list, gap_px: int = 24) -> list:
    """
    Group nearby OSB boxes by simple adjacency (expanded bbox intersection), return merged blocks.
    """
    if not osb_blocks or len(osb_blocks) < 2:
        return osb_blocks or []

    boxes = [getattr(b, "xyxy", None) for b in osb_blocks]
    n = len(osb_blocks)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri = find(i)
        rj = find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(n):
        bi = boxes[i]
        if not bi or len(bi) != 4:
            continue
        for j in range(i + 1, n):
            bj = boxes[j]
            if not bj or len(bj) != 4:
                continue
            if _boxes_touch(bi, bj, int(gap_px)):
                union(i, j)

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    if len(groups) == n:
        return osb_blocks

    merged = []
    for inds in groups.values():
        if len(inds) == 1:
            merged.append(osb_blocks[inds[0]])
            continue
        blks = [osb_blocks[i] for i in inds]
        xs1 = [b.xyxy[0] for b in blks]
        ys1 = [b.xyxy[1] for b in blks]
        xs2 = [b.xyxy[2] for b in blks]
        ys2 = [b.xyxy[3] for b in blks]
        x1, y1, x2, y2 = int(min(xs1)), int(min(ys1)), int(max(xs2)), int(max(ys2))
        # Create a simple merged rectangle block. Keep OSB label.
        from utils.textblock import TextBlock

        pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
        nb = TextBlock(xyxy=[x1, y1, x2, y2], lines=[pts.tolist()])
        nb.label = "osb"
        merged.append(nb)
    return merged


_PAGE_NUM_RE = re.compile(r"^\s*\d{1,4}\s*$")


def is_page_number_text(text: str) -> bool:
    if text is None:
        return False
    return bool(_PAGE_NUM_RE.match(str(text)))


def is_margin_candidate(xyxy: List[int], im_w: int, im_h: int, margin_ratio: float = 0.08) -> bool:
    if im_w <= 0 or im_h <= 0:
        return False
    x1, y1, x2, y2 = xyxy
    mx = int(round(im_w * float(margin_ratio)))
    my = int(round(im_h * float(margin_ratio)))
    return x1 <= mx or y1 <= my or x2 >= (im_w - mx) or y2 >= (im_h - my)


def filter_page_number_blocks_after_ocr(
    blk_list: list,
    im_w: int,
    im_h: int,
    margin_ratio: float = 0.08,
) -> Tuple[list, list]:
    """
    Remove page-number-like OSB blocks after OCR has populated `blk.text`.
    Returns (new_blk_list, removed_blocks).
    """
    if not blk_list:
        return blk_list, []
    kept = []
    removed = []
    for b in blk_list:
        lab = (getattr(b, "label", None) or "").strip().lower()
        if lab in OSB_LABELS or lab == "text_free":
            xyxy = getattr(b, "xyxy", None)
            if xyxy and len(xyxy) == 4 and is_margin_candidate(xyxy, im_w, im_h, margin_ratio=margin_ratio):
                txt = getattr(b, "translation", None)  # wrong field sometimes used by some OCRs
                if not txt:
                    try:
                        txt = b.get_text()
                    except Exception:
                        txt = ""
                if is_page_number_text(txt):
                    removed.append(b)
                    continue
        kept.append(b)
    return kept, removed


def probe_osb_style(img_rgb: np.ndarray, xyxy: List[int]) -> Optional[Dict]:
    """
    Estimate background luminance around the box and propose a readable fg + stroke.
    Returns dict with fg_rgb, stroke_rgb, stroke_width.
    """
    if img_rgb is None or img_rgb.size == 0:
        return None
    h, w = img_rgb.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(1, min(x2, w))
    y2 = max(1, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return None

    # Sample a thin border around the bbox (outside-ish background)
    bw = max(2, int(round(0.08 * (x2 - x1))))
    bh = max(2, int(round(0.08 * (y2 - y1))))
    xa1 = max(0, x1 - bw)
    ya1 = max(0, y1 - bh)
    xa2 = min(w, x2 + bw)
    ya2 = min(h, y2 + bh)
    roi = img_rgb[ya1:ya2, xa1:xa2]
    if roi.size == 0:
        return None

    # Build a ring mask: outer - inner
    ring = np.ones((roi.shape[0], roi.shape[1]), dtype=np.uint8) * 255
    ix1 = x1 - xa1
    iy1 = y1 - ya1
    ix2 = x2 - xa1
    iy2 = y2 - ya1
    cv2.rectangle(ring, (ix1, iy1), (ix2 - 1, iy2 - 1), 0, thickness=-1)
    ys, xs = np.where(ring > 0)
    if ys.size < 16:
        return None
    px = roi[ys, xs].astype(np.float32)
    bg = np.median(px, axis=0)
    # Luminance
    lum = float(0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2])

    if lum < 110:
        fg = np.array([255, 255, 255], dtype=np.uint8)
        stroke = np.array([0, 0, 0], dtype=np.uint8)
        sw = 0.18
    else:
        fg = np.array([0, 0, 0], dtype=np.uint8)
        stroke = np.array([255, 255, 255], dtype=np.uint8)
        sw = 0.16
    return {"fg_rgb": fg, "stroke_rgb": stroke, "stroke_width": sw}


def apply_osb_style_defaults(img_rgb: np.ndarray, osb_blocks: list) -> None:
    if not osb_blocks:
        return
    for b in osb_blocks:
        xyxy = getattr(b, "xyxy", None)
        if not xyxy or len(xyxy) != 4:
            continue
        sty = probe_osb_style(img_rgb, xyxy)
        if not sty:
            continue
        try:
            b.fg_colors = sty["fg_rgb"].tolist()
            b.stroke_width = float(sty["stroke_width"])
            # Keep stroke color in shadow_color for now if available (FontFormat supports shadow_color; stroke_color isn't exposed everywhere)
            if hasattr(b, "shadow_color"):
                b.shadow_color = sty["stroke_rgb"].tolist()
        except Exception:
            continue

