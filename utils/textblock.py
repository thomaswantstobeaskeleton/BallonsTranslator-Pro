from typing import List, Tuple, Callable
import numpy as np
from shapely.geometry import Polygon
import math
import copy
import cv2
import re

from .imgproc_utils import union_area, xywh2xyxypoly, rotate_polygons, color_difference
from .structures import Union, List, Dict, field, nested_dataclass
from .split_text_region import split_textblock as split_text_region
from .fontformat import FontFormat, LineSpacingType, TextAlignment, fix_fontweight_qt
from .textblock_mask import canny_flood
from .textlines_merge import sort_pnts, Quadrilateral, merge_bboxes_text_region


LANG_LIST = ['eng', 'ja', 'unknown']
LANGCLS2IDX = {'eng': 0, 'ja': 1, 'unknown': 2}

# https://ayaka.shn.hk/hanregex/
# https://medium.com/the-artificial-impostor/detecting-chinese-characters-in-unicode-strings-4ac839ba313a
CJKPATTERN = re.compile(r'[\uac00-\ud7a3\u3040-\u30ff\u4e00-\u9FFF]')


@nested_dataclass
class TextBlock:
    xyxy: List = field(default_factory = lambda: [0, 0, 0, 0])
    lines: List = field(default_factory = lambda: [])
    language: str = 'unknown'
    # font_size: float = -1.
    distance: np.ndarray = None
    angle: int = 0
    vec: List = None
    norm: float = -1
    merged: bool = False
    text: List = field(default_factory = lambda : [])
    translation: str = ""
    rich_text: str = ""
    _bounding_rect: List = None
    src_is_vertical: bool = None
    _detected_font_size: float = -1
    det_model: str = None
    label: str = None # ysg yolo label

    region_mask: np.ndarray = None
    region_inpaint_dict: Dict = None
    # Text eraser mask (#1093): same size as block in px, 255=show 0=erase (hole)
    text_mask: np.ndarray = None
    # PR #1070: foreground image overlay (path relative to project directory, e.g. overlays/xxx.png)
    foreground_image_path: str = None
    overlay_opacity: float = 1.0  # 0..1 opacity for foreground overlay image
    skew_x: float = 0.0  # horizontal skew (shear) for block content, typically -0.5..0.5
    skew_y: float = 0.0  # vertical skew (shear) for block content
    # PR #1105: interactive quad/mesh warp. warp_mode: 'none' | 'quad' | 'mesh'
    warp_mode: str = 'none'
    warp_quad: List = field(default_factory=lambda: [[0, 0], [1, 0], [1, 1], [0, 1]])  # 4 corners normalized [x,y]
    warp_mesh_size: List = field(default_factory=lambda: [2, 2])  # [nx, ny] for mesh
    warp_mesh: List = None  # list of normalized [x,y] points for mesh (None = use warp_quad only)

    fontformat: FontFormat = field(default_factory=lambda: FontFormat())

    # 字体识别相关属性
    _detected_font_name: str = ""  # 识别出的字体名称
    _detected_font_confidence: float = 0.0  # 识别置信度

    deprecated_attributes: dict = field(default_factory = lambda: dict())

    @property
    def vertical(self):
        return self.fontformat.vertical
    
    @vertical.setter
    def vertical(self, value: bool):
        self.fontformat.vertical = value

    @property
    def font_size(self):
        return self.fontformat.font_size
    
    @font_size.setter
    def font_size(self, value: float):
        self.fontformat.font_size = value

    @property
    def line_spacing(self):
        return self.fontformat.line_spacing

    @line_spacing.setter
    def line_spacing(self, value: float):
        self.fontformat.line_spacing = value

    @property
    def letter_spacing(self):
        return self.fontformat.letter_spacing

    @letter_spacing.setter
    def letter_spacing(self, value: float):
        self.fontformat.letter_spacing = value

    @property
    def font_family(self):
        return self.fontformat.font_family

    @font_family.setter
    def font_family(self, value: str):
        self.fontformat.font_family = value

    @property
    def font_weight(self):
        return self.fontformat.font_weight

    @font_weight.setter
    def font_weight(self, value: int):
        self.fontformat.font_weight = value

    @property
    def bold(self):
        return self.fontformat.bold

    @bold.setter
    def bold(self, value: bool):
        self.fontformat.bold = value

    @property
    def italic(self):
        return self.fontformat.italic

    @italic.setter
    def italic(self, value: bool):
        self.fontformat.italic = value

    @property
    def underline(self):
        return self.fontformat.underline

    @underline.setter
    def underline(self, value: bool):
        self.fontformat.underline = value

    @property
    def strikethrough(self):
        return getattr(self.fontformat, 'strikethrough', False)

    @strikethrough.setter
    def strikethrough(self, value: bool):
        self.fontformat.strikethrough = value

    @property
    def stroke_width(self):
        return self.fontformat.stroke_width

    @stroke_width.setter
    def stroke_width(self, value: float):
        self.fontformat.stroke_width = value

    @property
    def opacity(self):
        return self.fontformat.opacity

    @opacity.setter
    def opacity(self, value: float):
        self.fontformat.opacity = value

    @property
    def shadow_radius(self):
        return self.fontformat.shadow_radius

    @shadow_radius.setter
    def shadow_radius(self, value: float):
        self.fontformat.shadow_radius = value

    @property
    def shadow_strength(self):
        return self.fontformat.shadow_strength

    @shadow_strength.setter
    def shadow_strength(self, value: float):
        self.fontformat.shadow_strength = value

    @property
    def shadow_color(self):
        return self.fontformat.shadow_color

    @shadow_color.setter
    def shadow_color(self, value: float):
        self.fontformat.shadow_color = value

    @property
    def shadow_offset(self):
        return self.fontformat.shadow_offset

    @shadow_offset.setter
    def shadow_offset(self, value: float):
        self.fontformat.shadow_offset = value

    @property
    def fg_colors(self):
        return self.fontformat.frgb

    @fg_colors.setter
    def fg_colors(self, value: Union[np.ndarray, List]):
        self.fontformat.frgb = value

    @property
    def bg_colors(self):
       return self.fontformat.srgb

    @bg_colors.setter
    def bg_colors(self, value: np.ndarray):
        self.fontformat.srgb = value

    @property
    def alignment(self):
       return self.fontformat.alignment

    @alignment.setter
    def alignment(self, value: int):
        self.fontformat.alignment = value

    def __post_init__(self):
        if self.xyxy is not None:
            self.xyxy = [int(num) for num in self.xyxy]
        if self.distance is not None:
            self.distance = np.array(self.distance, np.float32)
        if self.vec is not None:
            self.vec = np.array(self.vec, np.float32)
        if self.text_mask is not None and isinstance(self.text_mask, (list, tuple)):
            self.text_mask = np.array(self.text_mask, dtype=np.uint8)
        if self.src_is_vertical is None:
            self.src_is_vertical = self.vertical
        
        if self.rich_text:
            self.rich_text = fix_fontweight_qt(self.rich_text)

        da = self.deprecated_attributes
        if len(da) > 0:
            if 'accumulate_color' in da:
                self.fg_colors = np.array([da['fg_r'], da['fg_g'], da['fg_b']], dtype=np.float32)
                self.bg_colors = np.array([da['bg_r'], da['bg_g'], da['bg_b']], dtype=np.float32)
                nlines = len(self)
                if da['accumulate_color'] and len(self) > 0:
                    self.fg_colors /= nlines
                    self.bg_colors /= nlines

            deprecated_blk_fmt_keys = {'vertical': None, 'line_spacing': None, 'letter_spacing': None, 'bold': None, 'underline': None, 'italic': None,
                'opacity': None, 'shadow_radius': None, 'shadow_strength': None, 'shadow_color': None, 'shadow_offset': None,
                 'font_size': 'size', 'font_family': None, '_alignment': 'alignment', 'default_stroke_width': 'stroke_width', 'font_weight': None,
                 'fg_colors': 'frgb', 'bg_colors': 'srgb'
            }
            for src_k, v in da.items():
                if src_k in deprecated_blk_fmt_keys:
                    if deprecated_blk_fmt_keys[src_k] is None:
                        tgt_k = src_k
                    else:
                        tgt_k = deprecated_blk_fmt_keys[src_k]
                    setattr(self.fontformat, tgt_k, v)
            self.font_weight = fix_fontweight_qt(self.font_weight)

        del self.deprecated_attributes

    @property
    def detected_font_size(self):
        if self._detected_font_size > 0:
            return self._detected_font_size
        return self.font_size

    def adjust_bbox(self, with_bbox=False, x_range=None, y_range=None):
        lines = self.lines_array().astype(np.int32)
        if with_bbox:
            self.xyxy[0] = min(lines[..., 0].min(), self.xyxy[0])
            self.xyxy[1] = min(lines[..., 1].min(), self.xyxy[1])
            self.xyxy[2] = max(lines[..., 0].max(), self.xyxy[2])
            self.xyxy[3] = max(lines[..., 1].max(), self.xyxy[3])
        else:
            self.xyxy[0] = lines[..., 0].min()
            self.xyxy[1] = lines[..., 1].min()
            self.xyxy[2] = lines[..., 0].max()
            self.xyxy[3] = lines[..., 1].max()

        if x_range is not None:
            self.xyxy[0] = np.clip(self.xyxy[0], x_range[0], x_range[1])
            self.xyxy[2] = np.clip(self.xyxy[2], x_range[0], x_range[1])
        if y_range is not None:
            self.xyxy[1] = np.clip(self.xyxy[1], y_range[0], y_range[1])
            self.xyxy[3] = np.clip(self.xyxy[3], y_range[0], y_range[1])

    def sort_lines(self):
        if self.distance is not None:
            idx = np.argsort(self.distance)
            self.distance = self.distance[idx]
            lines = np.array(self.lines, dtype=np.int32)
            self.lines = lines[idx].tolist()

    def lines_array(self, dtype=np.float64):
        return np.array(self.lines, dtype=dtype)

    def set_lines_by_xywh(self, xywh: np.ndarray, angle=0, x_range=None, y_range=None, adjust_bbox=False):
        if isinstance(xywh, List):
            xywh = np.array(xywh)
        lines = xywh2xyxypoly(np.array([xywh]))
        if angle != 0:
            cx, cy = xywh[0], xywh[1]
            cx += xywh[2] / 2.
            cy += xywh[3] / 2.
            lines = rotate_polygons([cx, cy], lines, angle)

        lines = lines.reshape(-1, 4, 2)
        if x_range is not None:
            lines[..., 0] = np.clip(lines[..., 0], x_range[0], x_range[1])
        if y_range is not None:
            lines[..., 1] = np.clip(lines[..., 1], y_range[0], y_range[1])
        self.lines = lines.tolist()

        if adjust_bbox:
            self.adjust_bbox()

    def aspect_ratio(self) -> float:
        min_rect = self.min_rect()
        middle_pnts = (min_rect[:, [1, 2, 3, 0]] + min_rect) / 2
        norm_v = np.linalg.norm(middle_pnts[:, 2] - middle_pnts[:, 0])
        norm_h = np.linalg.norm(middle_pnts[:, 1] - middle_pnts[:, 3])
        return norm_v / norm_h

    def center(self) -> np.ndarray:
        xyxy = np.array(self.xyxy)
        return (xyxy[:2] + xyxy[2:]) / 2

    def unrotated_polygons(self, ids=None) -> np.ndarray:
        angled = self.angle != 0
        center = self.center()
        polygons = self.lines_array().reshape(-1, 8)
        if ids is not None:
            polygons = polygons[ids]
        if angled:
            polygons = rotate_polygons(center, polygons, self.angle)
        return angled, center, polygons
    
    def min_rect(self, rotate_back=True, ids=None) -> List[int]:
        angled, center, polygons = self.unrotated_polygons(ids=ids)
        min_x = polygons[:, ::2].min()
        min_y = polygons[:, 1::2].min()
        max_x = polygons[:, ::2].max()
        max_y = polygons[:, 1::2].max()
        min_bbox = np.array([[min_x, min_y, max_x, min_y, max_x, max_y, min_x, max_y]])
        if angled and rotate_back:
            min_bbox = rotate_polygons(center, min_bbox, -self.angle)
        return min_bbox.reshape(-1, 4, 2).astype(np.int64)

    def normalizd_width_list(self, normalize=True):
        angled, center, polygons = self.unrotated_polygons()
        width_list = []
        for polygon in polygons:
            width_list.append((polygon[[2, 4]] - polygon[[0, 6]]).mean())
        sum_width = sum(width_list)
        if normalize:
            width_list = np.array(width_list)
            width_list = width_list / sum_width
            width_list = width_list.tolist()
        return width_list, sum_width

    # equivalent to qt's boundingRect, ignore angle
    def bounding_rect(self) -> List[int]:
        if self._bounding_rect is None:
        # if True:
            min_bbox = self.min_rect(rotate_back=False)[0]
            x, y = min_bbox[0]
            w, h = min_bbox[2] - min_bbox[0]
            return [int(x), int(y), int(w), int(h)]
        return self._bounding_rect

    def __getattribute__(self, name: str):
        if name == 'pts':
            return self.lines_array()
        # else:
        return object.__getattribute__(self, name)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        return self.lines[idx]

    def to_dict(self, deep_copy=False):
        blk_dict = vars(self)
        if deep_copy:
            blk_dict = copy.deepcopy(blk_dict)
        return blk_dict

    def get_transformed_region(self, img: np.ndarray, idx: int, textheight: int, maxwidth: int = None) -> np.ndarray :
        im_h, im_w = img.shape[:2]

        line = np.round(np.array(self.lines[idx])).astype(np.int64)
        
        if not self.src_is_vertical and self.det_model == 'ctd':
            # ctd detected horizontal bbox is smaller than GT
            expand_size = max(int(self._detected_font_size * 0.1), 3)
            rad = np.deg2rad(self.angle)
            shifted_vec = np.array([[[-1, -1],[1, -1],[1, 1],[-1, 1]]])
            shifted_vec = shifted_vec * np.array([[[np.sin(rad), np.cos(rad)]]]) * expand_size
            line = line + shifted_vec
            line[..., 0] = np.clip(line[..., 0], 0, im_w)
            line[..., 1] = np.clip(line[..., 1], 0, im_h)
            line = np.round(line[0]).astype(np.int64)

        x1, y1, x2, y2 = line[:, 0].min(), line[:, 1].min(), line[:, 0].max(), line[:, 1].max()
        
        x1 = np.clip(x1, 0, im_w)
        y1 = np.clip(y1, 0, im_h)
        x2 = np.clip(x2, 0, im_w)
        y2 = np.clip(y2, 0, im_h)
        img_croped = img[y1: y2, x1: x2]
        
        direction = 'v' if self.src_is_vertical else 'h'

        src_pts = line.copy()
        src_pts[:, 0] -= x1
        src_pts[:, 1] -= y1
        middle_pnt = (src_pts[[1, 2, 3, 0]] + src_pts) / 2
        vec_v = middle_pnt[2] - middle_pnt[0]   # vertical vectors of textlines
        vec_h = middle_pnt[1] - middle_pnt[3]   # horizontal vectors of textlines
        norm_v = np.linalg.norm(vec_v)
        norm_h = np.linalg.norm(vec_h)

        if textheight is None:
            if direction == 'h' :
                textheight = int(norm_v)
            else:
                textheight = int(norm_h)
        
        if norm_v <= 0 or norm_h <= 0:
            print('invalid textpolygon to target img')
            return np.zeros((textheight, textheight, 3), dtype=np.uint8)
        ratio = norm_v / norm_h

        if direction == 'h' :
            h = int(textheight)
            w = int(round(textheight / ratio))
            dst_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).astype(np.float32)
            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if M is None:
                print('invalid textpolygon to target img')
                return np.zeros((textheight, textheight, 3), dtype=np.uint8)
            region = cv2.warpPerspective(img_croped, M, (w, h))
        elif direction == 'v' :
            w = int(textheight)
            h = int(round(textheight * ratio))
            dst_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).astype(np.float32)
            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if M is None:
                print('invalid textpolygon to target img')
                return np.zeros((textheight, textheight, 3), dtype=np.uint8)
            region = cv2.warpPerspective(img_croped, M, (w, h))
            region = cv2.rotate(region, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if maxwidth is not None:
            h, w = region.shape[: 2]
            if w > maxwidth:
                region = cv2.resize(region, (maxwidth, h))

        return region

    def get_text(self) -> str:
        if isinstance(self.text, str):
            return self.text
        text = ''
        for t in self.text:
            if text and t:
                if text[-1].isalpha() and t[0].isalpha() \
                    and CJKPATTERN.search(text[-1]) is None \
                    and CJKPATTERN.search(t[0]) is None:
                    text += ' '
            text += t

        return text.strip()

    def set_font_colors(self, fg_colors = None, bg_colors = None):
        if fg_colors is not None:
            self.fg_colors = fg_colors
        if bg_colors is not None:
            self.bg_colors = bg_colors

    def update_font_colors(self, fg_colors: np.ndarray, bg_colors: np.ndarray):
        nlines = len(self)
        if nlines > 0:
            if not isinstance(fg_colors, np.ndarray):
                fg_colors = np.array(fg_colors, dtype=np.float32)
            if not isinstance(bg_colors, np.ndarray):
                bg_colors = np.array(bg_colors, dtype=np.float32)
            if not isinstance(self.fg_colors, np.ndarray):
                self.fg_colors = np.array(self.fg_colors, dtype=np.float32)
            if not isinstance(self.bg_colors, np.ndarray):
                self.bg_colors = np.array(self.bg_colors, dtype=np.float32)
            self.fg_colors += fg_colors / nlines
            self.bg_colors += bg_colors / nlines

    def get_font_colors(self, bgr=False):

        frgb = np.array(self.fg_colors).astype(np.int32)
        brgb = np.array(self.bg_colors).astype(np.int32)

        if bgr:
            frgb = frgb[::-1]
            brgb = brgb[::-1]

        return frgb, brgb

    def xywh(self):
        x, y, w, h = self.xyxy
        return [x, y, w-x, h-y]
    
    def recalulate_alignment(self):
        angled, center, polygons = self.unrotated_polygons()
        polygons = polygons.reshape(-1, 4, 2)
        
        left_std = np.std(polygons[:, 0, 0])
        right_std = np.std(polygons[:, 1, 0])
        center_std = np.std((polygons[:, 0, 0] + polygons[:, 1, 0]) / 2) * 0.7
        
        if left_std < right_std and left_std < center_std:
            self.alignment = TextAlignment.Left
        elif right_std < left_std and right_std < center_std:
            self.alignment = TextAlignment.Right
        else:
            self.alignment = TextAlignment.Center

    def recalulate_stroke_width(self, color_diff_tol=15, stroke_width: float = 0.2, min_stroke_on_bg: float = 0.12):
        """Set stroke so text is readable on background. When fg/bg are similar (e.g. text on art),
        use a minimum outline so text outside bubbles still has an outline."""
        if color_difference(*self.get_font_colors()) < color_diff_tol:
            self.stroke_width = max(0.0, min_stroke_on_bg)  # was 0.; use min outline for text on similar bg
        else:
            self.stroke_width = stroke_width

    def adjust_pos(self, dx: int, dy: int):
        self.xyxy[0] += dx
        self.xyxy[1] += dy
        self.xyxy[2] += dx
        self.xyxy[3] += dy
        if self._bounding_rect is not None:
            self._bounding_rect[0] += dx
            self._bounding_rect[1] += dy

    def line_coord_valid(self, rect):
        if self.det_model is None:
            return False
        if rect is None:
            rect = self.bounding_rect()

        min_bbox = self.min_rect(rotate_back=True)[0]
        x1, y1 = min_bbox[0]
        x2, y2 = min_bbox[2]
        w = x2 - x1
        h = y2 - y1
        if w < 1 or h < 1:
            return False
        rx1, ry1, rx2, ry2 = rect
        rx2 += rx1
        ry2 += ry1
        intersect = max(min(x2, rx2) - max(x1, rx1), 0) * max(min(y2, ry2) - max(y1, ry1), 0)
        if intersect == 0:
            return False
        if intersect / (w * h) < 0.6:
            return False
        return True


def _box_area_xyxy(xyxy) -> float:
    if not xyxy or len(xyxy) != 4:
        return 0.0
    return max(0, (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1]))


def _iou_blocks(a: TextBlock, b: TextBlock) -> float:
    """IoU of two blocks by xyxy."""
    inter = union_area(a.xyxy, b.xyxy)
    if inter <= 0:
        return 0.0
    area_a = _box_area_xyxy(a.xyxy)
    area_b = _box_area_xyxy(b.xyxy)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _ioa_xyxy(inner_xyxy, outer_xyxy) -> float:
    """Intersection over area of inner (IoA): how much of inner is inside outer."""
    inter = union_area(inner_xyxy, outer_xyxy)
    if inter <= 0:
        return 0.0
    area_inner = _box_area_xyxy(inner_xyxy)
    return inter / area_inner if area_inner > 0 else 0.0


def remove_contained_boxes(blk_list: List[TextBlock]) -> List[TextBlock]:
    """
    Section 14: Remove boxes that are fully contained inside another (by IoA).
    Keeps outer boxes; removes nested/duplicate inner boxes.
    """
    if not blk_list or len(blk_list) <= 1:
        return list(blk_list)
    out = []
    for blk in blk_list:
        xyxy = getattr(blk, "xyxy", None)
        if not xyxy or len(xyxy) != 4:
            out.append(blk)
            continue
        contained = False
        for other in blk_list:
            if other is blk:
                continue
            oxyxy = getattr(other, "xyxy", None)
            if not oxyxy or len(oxyxy) != 4:
                continue
            if _box_area_xyxy(oxyxy) <= _box_area_xyxy(xyxy):
                continue
            if _ioa_xyxy(xyxy, oxyxy) >= 0.99:
                contained = True
                break
        if not contained:
            out.append(blk)
    return out


def deduplicate_primary_boxes(
    blk_list: List[TextBlock],
    iou_threshold: float = 0.5,
) -> List[TextBlock]:
    """
    Section 14: Remove duplicates via IoU; keep the one with larger area when IoU >= threshold.
    (No confidence in TextBlock, so we use area as proxy for "keep best".)
    """
    if not blk_list or len(blk_list) <= 1:
        return list(blk_list)
    # Sort by area descending so we keep larger when duplicate
    with_area = [(blk, _box_area_xyxy(getattr(blk, "xyxy", []))) for blk in blk_list]
    with_area.sort(key=lambda x: x[1], reverse=True)
    out = []
    for blk, _ in with_area:
        xyxy = getattr(blk, "xyxy", None)
        if not xyxy or len(xyxy) != 4:
            out.append(blk)
            continue
        is_dup = False
        for kept in out:
            if _iou_blocks(blk, kept) >= iou_threshold:
                is_dup = True
                break
        if not is_dup:
            out.append(blk)
    return out


def _spatial_sort_hybrid(
    regions: List[TextBlock],
    right_to_left: bool,
    gap_threshold_ratio: float = 0.4,
) -> List[TextBlock]:
    """
    Section 13: Hybrid spatial sorting for slightly offset bubbles.
    Group items into bands by vertical overlap / center distance, then into columns
    within bands; tuned thresholds keep offset bubbles together (prevents incorrect
    reordering on imperfect layouts).
    """
    if len(regions) <= 1:
        return list(regions)
    heights = [r.xyxy[3] - r.xyxy[1] for r in regions]
    med_h = float(np.median(heights)) if heights else 50.0
    gap = max(8, gap_threshold_ratio * med_h)

    # Sort by top (y1) then left (x1) for deterministic band building
    by_top = sorted(regions, key=lambda r: (r.xyxy[1], r.xyxy[0]))
    bands = []
    for r in by_top:
        y1, y2 = r.xyxy[1], r.xyxy[3]
        placed = False
        for band in bands:
            for b in band:
                by1, by2 = b.xyxy[1], b.xyxy[3]
                if not (y2 < by1 - gap or y1 > by2 + gap):
                    band.append(r)
                    placed = True
                    break
            if placed:
                break
        if not placed:
            bands.append([r])

    bands_sorted = sorted(bands, key=lambda band: min(b.xyxy[1] for b in band))
    out = []
    for band in bands_sorted:
        band_sorted = sorted(
            band,
            key=lambda b: (-b.center()[0] if right_to_left else b.center()[0], b.center()[1]),
        )
        out.extend(band_sorted)
    return out


def sort_regions(regions: List[TextBlock], right_to_left=None) -> List[TextBlock]:
    # Section 13: use hybrid spatial sort (bands + columns) for better reading order
    # when bubbles are slightly offset; preserves top-to-bottom then column order.
    nr = len(regions)
    if nr == 0:
        return []
    if right_to_left is None:
        nv = sum(1 for r in regions if getattr(r, "vertical", False))
        right_to_left = nv / nr > 0
    return _spatial_sort_hybrid(regions, right_to_left)


def examine_textblk(blk: TextBlock, im_w: int, im_h: int, sort: bool = False) -> None:
    lines = blk.lines_array()
    middle_pnts = (lines[:, [1, 2, 3, 0]] + lines) / 2
    vec_v = middle_pnts[:, 2] - middle_pnts[:, 0]   # vertical vectors of textlines
    vec_h = middle_pnts[:, 1] - middle_pnts[:, 3]   # horizontal vectors of textlines
    # if sum of vertical vectors is longer, then text orientation is vertical, and vice versa.
    center_pnts = (lines[:, 0] + lines[:, 2]) / 2
    v = np.sum(vec_v, axis=0)
    h = np.sum(vec_h, axis=0)
    norm_v, norm_h = np.linalg.norm(v), np.linalg.norm(h)
    vertical = blk.src_is_vertical
    # calcuate distance between textlines and origin 
    if vertical:
        primary_vec, primary_norm = v, norm_v
        distance_vectors = center_pnts - np.array([[im_w, 0]], dtype=np.float64)   # vertical manga text is read from right to left, so origin is (imw, 0)
        font_size = int(round(norm_h / len(lines)))
    else:
        primary_vec, primary_norm = h, norm_h
        distance_vectors = center_pnts - np.array([[0, 0]], dtype=np.float64)
        font_size = int(round(norm_v / len(lines)))
    
    rotation_angle = int(math.atan2(primary_vec[1], primary_vec[0]) / math.pi * 180)     # rotation angle of textlines
    distance = np.linalg.norm(distance_vectors, axis=1)     # distance between textlinecenters and origin
    rad_matrix = np.arccos(np.einsum('ij, j->i', distance_vectors, primary_vec) / (distance * primary_norm))
    distance = np.abs(np.sin(rad_matrix) * distance)
    blk.lines = lines.astype(np.int32).tolist()
    blk.distance = distance
    blk.angle = rotation_angle
    if vertical:
        blk.angle -= 90
    # Force horizontal text to 0° so it never renders slanted/sideways
    if not vertical:
        blk.angle = 0
    elif abs(blk.angle) < 3:
        blk.angle = 0
    # Normalize font_size so text fits inside bubbles and rarely overflows.
    max_dim = max(im_w, im_h)
    num_lines = max(1, len(lines))
    if max_dim > 800:
        scale = 700.0 / max_dim
        font_size = max(11, min(60, int(round(font_size * scale))))
    bx1, by1, bx2, by2 = blk.xyxy
    block_h = by2 - by1
    if block_h > 0:
        # ~22% of (block height / num lines) so text stays inside bubble
        block_based_size = 0.22 * block_h / num_lines
        font_size = max(font_size, min(60, int(round(block_based_size))))
    font_size = max(11, min(60, font_size))
    blk.font_size = font_size
    blk.vec = primary_vec
    blk.norm = primary_norm
    if sort:
        blk.sort_lines()

def try_merge_textline(blk: TextBlock, blk2: TextBlock, fntsize_tol=1.7, distance_tol=2, canvas=None) -> bool:
    if blk2.merged:
        return False
    fntsize_div = blk.font_size / blk2.font_size
    num_l1, num_l2 = len(blk), len(blk2)
    fntsz_avg = (blk.font_size * num_l1 + blk2.font_size * num_l2) / (num_l1 + num_l2)
    vec_prod = blk.vec @ blk2.vec
    vec_sum = blk.vec + blk2.vec
    cos_vec = vec_prod / blk.norm / blk2.norm
    # distance = blk2.distance[-1] - blk.distance[-1]
    # distance_p1 = np.linalg.norm(np.array(blk2.lines[-1][0]) - np.array(blk.lines[-1][0]))
    minrect1 = blk.min_rect(ids=[-1])[0]
    xyxy1 = [*minrect1[0], *minrect1[2]]
    minrect2 = blk2.min_rect(ids=[0])[0]
    xyxy2 = [*minrect2[0], *minrect2[2]]
    distance_x = max(xyxy1[0], xyxy2[0]) - min(xyxy1[2], xyxy2[2])
    distance_y = max(xyxy1[1], xyxy2[1]) - min(xyxy1[3], xyxy2[3])
    w1 = xyxy1[2] - xyxy1[0]
    w2 = xyxy2[2] - xyxy2[0]
    h1 = xyxy1[3] - xyxy1[1]
    h2 = xyxy2[3] - xyxy2[1] 

    l1, l2 = Polygon(blk.lines[-1]), Polygon(blk2.lines[0])
    if not l1.intersects(l2):
        if blk.vertical:
            if distance_x > fntsz_avg * 0.8:
                return False
            # Allow blk2 below blk (distance_y > 0) if gap is small so clipped bottom line merges into one box
            if distance_y > 0:
                if distance_y > fntsz_avg * 1.5:
                    return False
            else:
                if abs(distance_y) / min(h1, h2) < 0.4:
                    return False
        else:
            if distance_x > 0:
                return False
            fntsz_thr = 0.5
            if fntsz_avg < 24:
                fntsz_thr = 0.6
            if distance_y > fntsz_avg * fntsz_thr:
                return False
            if abs(distance_x) / min(w1, w2) < 0.3:
                return False
        if fntsize_div > fntsize_tol or 1 / fntsize_div > fntsize_tol:
            return False
        if abs(cos_vec) < 0.866:   # cos30
            return False
        # if distance > distance_tol * fntsz_avg:
        #     return False

    # merge
    for line in blk2.lines:
        blk.lines.append(line)
    blk.vec = vec_sum
    blk.angle = int(round(np.rad2deg(math.atan2(vec_sum[1], vec_sum[0]))))
    if blk.vertical:
        blk.angle -= 90
    blk.norm = np.linalg.norm(vec_sum)
    blk.distance = np.append(blk.distance, blk2.distance[-1])
    blk.font_size = fntsz_avg
    blk2.merged = True
    return True

def merge_textlines(blk_list: List[TextBlock], canvas=None, fntsize_tol=1.7) -> List[TextBlock]:
    if len(blk_list) < 2:
        return blk_list
    merged_list = []
    for ii, current_blk in enumerate(blk_list):
        if current_blk.merged:
            continue
        for jj, blk in enumerate(blk_list[ii+1:]):
            try_merge_textline(current_blk, blk, canvas=canvas, fntsize_tol=fntsize_tol)
        merged_list.append(current_blk)
    for blk in merged_list:
        blk.adjust_bbox(with_bbox=False)
    return merged_list

def split_textblk(blk: TextBlock):
    font_size, distance, lines = blk.font_size, blk.distance, blk.lines
    l0 = np.array(blk.lines[0])
    lines.sort(key=lambda line: np.linalg.norm(np.array(line[0]) - l0[0]))
    distance_tol = font_size * 2
    current_blk = copy.deepcopy(blk)
    current_blk.lines = [l0]
    sub_blk_list = [current_blk]
    textblock_splitted = False
    for jj, line in enumerate(lines[1:]):
        l1, l2 = Polygon(lines[jj]), Polygon(line)
        split = False
        if not l1.intersects(l2):
            line_disance = abs(distance[jj+1] - distance[jj])
            if line_disance > distance_tol:
                split = True
            elif blk.vertical and abs(blk.angle) < 15:
                if len(current_blk.lines) > 1 or line_disance > font_size:
                    split = abs(lines[jj][0][1] - line[0][1]) > font_size
        if split:
            current_blk = copy.deepcopy(current_blk)
            current_blk.lines = [line]
            sub_blk_list.append(current_blk)
        else:
            current_blk.lines.append(line)
    if len(sub_blk_list) > 1:
        textblock_splitted = True
        for current_blk in sub_blk_list:
            current_blk.adjust_bbox(with_bbox=False)
    return textblock_splitted, sub_blk_list

def group_output(blks, lines, im_w, im_h, mask=None, sort_blklist=True, canvas=None, merge_fntsize_tol_hor=None, merge_fntsize_tol_ver=None) -> List[TextBlock]:
    blk_list: List[TextBlock] = []
    scattered_lines = {'ver': [], 'hor': []}
    for bbox, cls, conf in zip(*blks):
        # cls could give wrong result
        blk_list.append(TextBlock(bbox, language=LANG_LIST[cls]))

    # step1: filter & assign lines to textblocks
    bbox_score_thresh = 0.4
    mask_score_thresh = 0.1
    for ii, line in enumerate(lines):
        line, is_vertical = sort_pnts(line)
        bx1, bx2 = line[:, 0].min(), line[:, 0].max()
        by1, by2 = line[:, 1].min(), line[:, 1].max()
        bbox_score, bbox_idx = -1, -1
        line_area = (by2-by1) * (bx2-bx1)
        for jj, blk in enumerate(blk_list):
            score = union_area(blk.xyxy, [bx1, by1, bx2, by2]) / line_area
            if bbox_score < score:
                bbox_score = score
                bbox_idx = jj
        if bbox_score > bbox_score_thresh:
            blk_list[bbox_idx].lines.append(line)
            blk_list[bbox_idx].adjust_bbox(with_bbox=True)
        else:   # if no textblock was assigned, check whether there is "enough" textmask
            if mask is not None:
                mask_score = mask[by1: by2, bx1: bx2].mean() / 255
                if mask_score < mask_score_thresh:
                    continue
            blk = TextBlock([bx1, by1, bx2, by2], [line])
            blk.vertical = blk.src_is_vertical = is_vertical
            examine_textblk(blk, im_w, im_h, sort=False)
            if blk.vertical:
                scattered_lines['ver'].append(blk)
            else:
                scattered_lines['hor'].append(blk)

    # step2: filter textblocks, sort & split textlines
    final_blk_list = []
    for blk in blk_list:
        # filter textblocks 
        if len(blk.lines) == 0:
            bx1, by1, bx2, by2 = blk.xyxy
            if mask is not None:
                mask_score = mask[by1: by2, bx1: bx2].mean() / 255
                if mask_score < mask_score_thresh:
                    continue
            xywh = np.array([[bx1, by1, bx2-bx1, by2-by1]])
            blk.lines = xywh2xyxypoly(xywh).reshape(-1, 4, 2).tolist()
        else:
            blk.adjust_bbox(with_bbox=False)
        examine_textblk(blk, im_w, im_h, sort=True)
        
        # split manga text if there is a distance gap
        textblock_splitted = False
        if len(blk.lines) > 1:
            if blk.language == 'ja':
                textblock_splitted = True
            elif blk.vertical:
                textblock_splitted = True
        # if textblock_splitted:
        #     textblock_splitted, sub_blk_list = split_textblk(blk)
        # else:
        sub_blk_list = [blk]
        # modify textblock to fit its textlines
        if not textblock_splitted:
            for blk in sub_blk_list:
                blk.adjust_bbox(with_bbox=True)
        final_blk_list += sub_blk_list

    _final_blk_list = []
    for blk in final_blk_list:
        if blk.vertical:
            scattered_lines['ver'].append(blk)
        else:
            _final_blk_list.append(blk)
    final_blk_list = _final_blk_list

    # step3: merge scattered lines, sort textblocks by "grid"
    scattered_lines['ver'].sort(key=lambda blk: blk.center()[0], reverse=True)
    scattered_lines['hor'].sort(key=lambda blk: blk.center()[1])
    fntsize_tol_hor = merge_fntsize_tol_hor if merge_fntsize_tol_hor is not None else 2.0
    fntsize_tol_ver = merge_fntsize_tol_ver if merge_fntsize_tol_ver is not None else 1.7
    final_blk_list += merge_textlines(scattered_lines['hor'], canvas=canvas, fntsize_tol=fntsize_tol_hor)
    final_blk_list += merge_textlines(scattered_lines['ver'], fntsize_tol=fntsize_tol_ver)
    if sort_blklist:
        final_blk_list = sort_regions(final_blk_list, )
    for blk in final_blk_list:
        blk.distance = None


    if len(final_blk_list) > 1:
        _final_blks = [final_blk_list[0]]
        for blk in final_blk_list[1:]:
            ax1, ay1, ax2, ay2 = blk.xyxy
            keep_blk = True
            aarea = (ax2 - ax1) * (ay2 - ay1) + 1e-6
            for eb in _final_blks:
                bx1, by1, bx2, by2 = eb.xyxy
                x1 = max(ax1, bx1)
                y1 = max(ay1, by1)
                x2 = min(ax2, bx2)
                y2 = min(ay2, by2)
                if y2 < y1 or x2 < x1:
                    continue
                inter_area = (y2 - y1) * (x2 - x1)
                if inter_area / aarea > 0.9:
                    keep_blk = False
                    break
            if keep_blk:
                _final_blks.append(blk)
        final_blk_list = _final_blks

    # Merge vertical blocks stacked in same column (e.g. main block + clipped bottom fragment)
    ver_blks = [b for b in final_blk_list if b.vertical and len(b.lines) > 0]
    if len(ver_blks) >= 2:
        ver_blks.sort(key=lambda b: b.center()[0], reverse=True)
        merged_ver = []
        used = set()
        for i, a in enumerate(ver_blks):
            if i in used:
                continue
            ax1, ay1, ax2, ay2 = a.xyxy
            ca_x = (ax1 + ax2) / 2
            fnt = max(getattr(a, '_detected_font_size', a.font_size), 8)
            for j, b in enumerate(ver_blks):
                if j <= i or j in used:
                    continue
                bx1, by1, bx2, by2 = b.xyxy
                cb_x = (bx1 + bx2) / 2
                if abs(ca_x - cb_x) > fnt * 1.5:
                    continue
                gap = by1 - ay2 if by1 >= ay2 else by2 - ay1
                if 0 <= gap <= fnt * 2.2:
                    for line in b.lines:
                        a.lines.append(line)
                    used.add(j)
                    a.adjust_bbox(with_bbox=True)
            merged_ver.append(a)
        if merged_ver:
            others = [b for b in final_blk_list if not b.vertical]
            final_blk_list = others + merged_ver
            if sort_blklist:
                final_blk_list = sort_regions(final_blk_list)

    for blk in final_blk_list:
        if blk.language != 'ja' and not blk.vertical:
            num_lines = len(blk.lines)
            if num_lines == 0:
                continue
        blk._detected_font_size = blk.font_size
            
    return final_blk_list

def visualize_textblocks(canvas, blk_list:  List[TextBlock]):
    lw = max(round(sum(canvas.shape) / 2 * 0.003), 2)  # line width
    for ii, blk in enumerate(blk_list):
        bx1, by1, bx2, by2 = blk.xyxy
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (127, 255, 127), lw)
        lines = blk.lines_array(dtype=np.int32)
        for jj, line in enumerate(lines):
            cv2.putText(canvas, str(jj), line[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,127,0), 1)
            cv2.polylines(canvas, [line], True, (0,127,255), 2)
        cv2.polylines(canvas, [blk.min_rect()], True, (127,127,0), 2)
        center = [int((bx1 + bx2)/2), int((by1 + by2)/2)]
        cv2.putText(canvas, str(blk.angle), center, cv2.FONT_HERSHEY_SIMPLEX, 1, (127,127,255), 2)
        cv2.putText(canvas, str(ii), (bx1, by1 + lw + 2), 0, lw / 6, (255,127,127), max(lw-7, 1), cv2.LINE_AA)
    return canvas

def collect_textblock_regions(img: np.ndarray, textblk_lst: List[TextBlock], text_height=48, maxwidth=8100, split_textblk = False, seg_func: Callable = None):
    regions = []
    textblk_lst_indices = []
    for blk_idx, textblk in enumerate(textblk_lst):
        for ii in range(len(textblk)):
            if split_textblk and len(textblk) == 1:
                seg_func = canny_flood
                region = textblk.get_transformed_region(img, ii, None, maxwidth=None)
                mask  = seg_func(region)[0]
                split_lines = split_text_region(mask)[0]
                for jj, line in enumerate(split_lines):
                    bottom = line[3]
                    if len(split_lines) == 1:
                        bottom = region.shape[0]
                    r = region[line[1]: bottom]
                    h, w = r.shape[:2]
                    tgt_h, tgt_w = text_height, min(maxwidth, int(text_height / h * w))
                    if tgt_h != h or tgt_w != w:
                        r = cv2.resize(r, (tgt_w, tgt_h), interpolation=cv2.INTER_LINEAR)
                    regions.append(r)
                    textblk_lst_indices.append(blk_idx)
                #     cv2.imwrite(f'local_region{jj}.jpg', r)
                # cv2.imwrite('local_mask.jpg', mask)
                # cv2.imwrite('local_region.jpg',region)
            else:
                textblk_lst_indices.append(blk_idx)
                region = textblk.get_transformed_region(img, ii, text_height, maxwidth=maxwidth)
                regions.append(region)

    return regions, textblk_lst_indices


def mit_merge_textlines(textlines: List[Quadrilateral], width: int, height: int, verbose: bool = False) -> List[TextBlock]:
    # from https://github.com/zyddnys/manga-image-translator
    quadrilateral_lst = []
    for line in textlines:
        if not isinstance(line, Quadrilateral):
            line = Quadrilateral(np.array(line), '',  1.)
        quadrilateral_lst.append(line)
    textlines = quadrilateral_lst

    text_regions: List[TextBlock] = []
    textlines_total_area = sum([txtln.area for txtln in textlines])
    for (txtlns, fg_color, bg_color) in merge_bboxes_text_region(textlines, width, height):
        total_logprobs = 0
        for txtln in txtlns:
            total_logprobs += np.log(txtln.prob) * txtln.area
        
        total_logprobs /= textlines_total_area
        font_size = int(min([txtln.font_size for txtln in txtlns]))
        angle = np.rad2deg(np.mean([txtln.angle for txtln in txtlns])) - 90
        if abs(angle) < 3:
            angle = 0
        lines = [txtln.pts for txtln in txtlns]
        texts = [txtln.text for txtln in txtlns]
        ffmt = FontFormat(font_size=font_size, frgb=fg_color, srgb=bg_color)

        nv = 0
        for txtln in txtlns:
            if txtln.direction == 'v':
                nv += 1
        is_vertical = nv >= len(txtlns) // 2
        region = TextBlock(
            lines=lines, text=texts, angle=angle, fontformat=ffmt, 
            _detected_font_size=font_size, src_is_vertical=is_vertical, vertical=is_vertical)
        region.adjust_bbox()
        if region.src_is_vertical:
            region.alignment = 1
        else:
            region.recalulate_alignment()
        text_regions.append(region)

    return text_regions
