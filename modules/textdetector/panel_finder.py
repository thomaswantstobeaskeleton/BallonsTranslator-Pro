r"""
Finds panel order for manga page.
>> python .\modules\textdetector\panel_finder.py <path-to-images>
"""
import json
import sys
from pathlib import Path
from typing import List

import cv2 as cv
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from shapely import Polygon
from shapely.ops import nearest_points

KERNEL_SIZE = 7
BORDER_SIZE = 10


def panel_process_image(img: Image.Image):
    """Preprocesses an image to make it easier to find panels.

    Args:
        img: The image to preprocess.

    Returns:
        The preprocessed image.
    """

    img_gray = cv.cvtColor(np.array(img), cv.COLOR_RGB2GRAY)
    img_gray = cv.GaussianBlur(img_gray, (KERNEL_SIZE, KERNEL_SIZE), 0)
    img_gray = cv.threshold(img_gray, 200, 255, cv.THRESH_BINARY)[1]

    # Add black border to image, to help with finding contours
    img_gray = cv.copyMakeBorder(
        img_gray,
        BORDER_SIZE,
        BORDER_SIZE,
        BORDER_SIZE,
        BORDER_SIZE,
        cv.BORDER_CONSTANT,
        value=255,
    )
    # Invert image
    img_gray = cv.bitwise_not(img_gray)
    return img_gray


def remove_contained_contours(polygons):
    """Removes polygons from a list if any completely contain the other.

    Args:
        polygons: A list of polygons.

    Returns:
        A list of polygons with any contained polygons removed.
    """

    # Create a new list to store the filtered polygons.
    filtered_polygons = []

    # Iterate over the polygons.
    for polygon in polygons:
        # Check if the polygon contains any of the other polygons.
        contains = False
        for other_polygon in polygons:
            # Check if the polygon contains the other polygon and that the polygons
            if np.array_equal(other_polygon, polygon):
                continue
            rect1 = cv.boundingRect(other_polygon)
            rect2 = cv.boundingRect(polygon)
            # Check if rect2 is completely within rect1
            if (
                rect2[0] >= rect1[0]
                and rect2[1] >= rect1[1]
                and rect2[0] + rect2[2] <= rect1[0] + rect1[2]
                and rect2[1] + rect2[3] <= rect1[1] + rect1[3]
            ):
                contains = True
                break

        # If the polygon does not contain any of the other polygons, add it to the
        # filtered list.
        if not contains:
            filtered_polygons.append(polygon)

    return filtered_polygons


def calc_panel_contours(im: Image.Image):
    img_gray = panel_process_image(im)
    contours_raw, hierarchy = cv.findContours(
        img_gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )
    contours = contours_raw
    min_area = 10000
    contours = [i for i in contours if cv.contourArea(i) > min_area]
    contours = [cv.convexHull(i) for i in contours]
    contours = remove_contained_contours(contours)

    # Remap the contours to the original image
    contours = [i + np.array([[-BORDER_SIZE, -BORDER_SIZE]]) for i in contours]

    # Sort the contours by their y-coordinate.
    contours = order_panels(contours, img_gray)
    return contours


def calc_panel_bboxes_xyxy(img: Image.Image):
    contours = calc_panel_contours(img)
    panel_bboxes = [cv.boundingRect(c) for c in contours]
    panel_bboxes_xyxy = [xywh_to_xyxy(i) for i in panel_bboxes]
    return panel_bboxes_xyxy


def draw_contours(im, contours):
    """Debugging, draws the contours on the image."""
    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
    ]

    im_contour = np.array(im)

    for i, contour in enumerate(range(len(contours))):
        color = colors[i % len(colors)]
        im_contour = cv.drawContours(im_contour, contours, i, color, 4, cv.LINE_AA)
        # Draw a number at the top left of contour
        x, y, _, _ = cv.boundingRect(contours[i])
        cv.putText(
            im_contour,
            str(i),
            (x + 50, y + 50),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
            cv.LINE_AA,
        )

    img = Image.fromarray(im_contour)

    return img


def order_panels_xyxy(panel_bboxes_xyxy: List[List[int]], right_to_left: bool) -> List[int]:
    """
    Return indices of panels in reading order.

    Strategy:
    - Group panels that overlap vertically.
    - Sort groups top-to-bottom by y.
    - Sort within each group by x (rtl = descending, ltr = ascending), with y as tie-breaker.
    """
    if not panel_bboxes_xyxy:
        return []
    bboxes_xywh = [xyxy_to_xywh(b) for b in panel_bboxes_xyxy]
    groups = generate_vertical_bounding_box_groups_indices(bboxes_xywh)

    def group_key(g):
        ys = [bboxes_xywh[i][1] for i in g]
        return float(min(ys)) if ys else 0.0

    ordered_indices: List[int] = []
    for group in sorted(groups, key=group_key):
        def within_key(i: int):
            x, y, w, h = bboxes_xywh[i]
            return (-x if right_to_left else x, y, w * h)

        ordered_indices.extend(sorted(group, key=within_key))
    return ordered_indices


def _intersection_area_xyxy(a: List[int], b: List[int]) -> int:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    if x2 <= x1 or y2 <= y1:
        return 0
    return int((x2 - x1) * (y2 - y1))


def assign_boxes_to_panels(text_bboxes_xyxy: List[List[int]], panel_bboxes_xyxy: List[List[int]]) -> List[int]:
    """
    Assign each text box to a panel index.

    Uses max intersection first; if no intersection, uses nearest distance.
    Returns list of panel indices (same length as text_bboxes_xyxy). If no panels, all -1.
    """
    if not text_bboxes_xyxy:
        return []
    if not panel_bboxes_xyxy:
        return [-1 for _ in text_bboxes_xyxy]

    out: List[int] = []
    for box in text_bboxes_xyxy:
        areas = [_intersection_area_xyxy(box, p) for p in panel_bboxes_xyxy]
        best = int(np.argmax(areas)) if areas else -1
        if best >= 0 and areas[best] > 0:
            out.append(best)
            continue

        try:
            text_poly = polygon_from_xyxy(*box)
            best_dist = float("inf")
            best_idx = 0
            for i, p in enumerate(panel_bboxes_xyxy):
                panel_poly = polygon_from_xyxy(*p)
                nearest_pts = nearest_points(text_poly, panel_poly)
                dist = nearest_pts[0].distance(nearest_pts[1])
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
                    if dist == 0:
                        break
            out.append(best_idx)
        except Exception:
            out.append(0)
    return out


def reorder_textblocks_by_panels(
    img: np.ndarray,
    blk_list: list,
    reading_direction: str = "auto",
) -> list:
    """
    Section 13: Panel-aware reading order (bin to panels, then spatial sort).
    Detect panels, sort panels in Z-flow; assign bubbles to panels (or nearest);
    within each panel use hybrid spatial sort (bands + columns); append unassigned at end.

    reading_direction: "auto" | "rtl" | "ltr"
    """
    if blk_list is None or len(blk_list) < 2:
        return blk_list
    if img is None or img.size == 0:
        return blk_list

    from utils.textblock import sort_regions

    rd = (reading_direction or "auto").strip().lower()
    if rd not in {"auto", "rtl", "ltr"}:
        rd = "auto"
    if rd == "auto":
        nr = len(blk_list)
        nv = sum(1 for b in blk_list if getattr(b, "vertical", False))
        right_to_left = (nv / nr) > 0.5 if nr > 0 else False
    else:
        right_to_left = (rd == "rtl")

    try:
        if img.ndim == 3 and img.shape[2] == 4:
            img_rgb = cv.cvtColor(img, cv.COLOR_RGBA2RGB)
        else:
            img_rgb = img
        pil_img = Image.fromarray(img_rgb.astype(np.uint8), mode="RGB")
    except Exception:
        return sort_regions(blk_list, right_to_left=right_to_left)

    try:
        panel_bboxes = calc_panel_bboxes_xyxy(pil_img)
    except Exception:
        return sort_regions(blk_list, right_to_left=right_to_left)
    if not panel_bboxes:
        return sort_regions(blk_list, right_to_left=right_to_left)

    panel_order = order_panels_xyxy(panel_bboxes, right_to_left=right_to_left)
    if not panel_order:
        return sort_regions(blk_list, right_to_left=right_to_left)

    text_boxes = [getattr(b, "xyxy", None) for b in blk_list]
    if any(b is None or len(b) != 4 for b in text_boxes):
        return sort_regions(blk_list, right_to_left=right_to_left)

    assigned = assign_boxes_to_panels(text_boxes, panel_bboxes)
    panel_to_blks = {i: [] for i in range(len(panel_bboxes))}
    extra: List = []
    for blk, pi in zip(blk_list, assigned):
        if pi is None or int(pi) < 0 or int(pi) >= len(panel_bboxes):
            extra.append(blk)
        else:
            panel_to_blks[int(pi)].append(blk)

    out: List = []
    for pi in panel_order:
        out.extend(sort_regions(panel_to_blks.get(int(pi), []), right_to_left=right_to_left))
    if extra:
        out.extend(sort_regions(extra, right_to_left=right_to_left))
    return out


def save_draw_contours(path: Path | str):
    path = Path(path)

    pth_out = path.parent / (path.stem + "-contours")

    if not pth_out.exists():
        pth_out.mkdir()

    # Glob get all images in folder

    pths = [i for i in path.iterdir() if i.suffix in [".png", ".jpg", ".jpeg"]]
    for t in pths:
        print(t)
        im = Image.open(t)
        contours = calc_panel_contours(im)

        img_panels = draw_contours(im, contours)
        f_name = t.stem + t.suffix
        img_panels.save(pth_out / f_name)


def order_panels(contours, img_gray):
    """Orders the panels in a comic book page.

    Args:
      contours: A list of contours, where each contour is a list of points.

    Returns:
      A list of contours, where each contour is a list of points, ordered by
      their vertical position.
    """

    # Get the bounding boxes for each contour.
    bounding_boxes = [cv.boundingRect(contour) for contour in contours]

    # Generate groups of vertically overlapping bounding boxes.
    groups_indices = generate_vertical_bounding_box_groups_indices(bounding_boxes)

    c = []

    for group in groups_indices:
        # Reorder contours based on reverse z-order,

        cs = [bounding_boxes[i] for i in group]

        order_scores = order_read_direction_scores(cs)
        # Sort the list based on the location score value
        combined_list = list(zip(group, order_scores))
        sorted_list = sorted(combined_list, key=lambda x: x[1], reverse=True)

        c.extend(sorted_list)

    ordered_contours = [contours[i[0]] for i in c]
    return ordered_contours


def order_read_direction_scores(cs):
    """
    Smaller means read first, larger means read last
    """
    order_scores = [1 * (-i[1]) + i[0] * 1 for i in cs]
    return order_scores


def generate_vertical_bounding_box_groups_indices(bounding_boxes):
    """Generates groups of vertically overlapping bounding boxes.

    Args:
      bounding_boxes: A list of bounding boxes, where each bounding box is a tuple
        of (x, y, width, height).

    Returns:
      A list of groups, where each group is a list of bounding boxes that overlap
      vertically.
    """

    # Operate on indices Sort the bounding boxes by their y-coordinate.

    bbox_inds = np.argsort([i[1] for i in bounding_boxes])

    # generate groups of vertically overlapping bounding boxes
    groups = [[bbox_inds[0]]]
    for i in bbox_inds[1:]:
        is_old_group = False
        bbox = bounding_boxes[i]
        start1 = bbox[1]
        end1 = bbox[1] + bbox[3]
        for n, group in enumerate(groups):
            for ind in group:
                _bbox = bounding_boxes[ind]
                start2 = _bbox[1]
                end2 = _bbox[1] + _bbox[3]

                # Check for any partial overlapping
                if check_overlap((start1, end1), (start2, end2)):
                    groups[n] = group + [i]
                    is_old_group = True
                    break

            if is_old_group:
                break
        else:
            groups.append([i])
    return groups


def check_overlap(range1, range2):
    # Check if range1 is before range2
    if range1[1] < range2[0]:
        return False
    # Check if range1 is after range2
    elif range1[0] > range2[1]:
        return False
    # If neither of the above conditions are met, the ranges must overlap
    else:
        return True


# Convert xyxy bounding boxes to shapely polygons
def polygon_from_xyxy(x, y, x2, y2):
    return Polygon([(x, y), (x2, y), (x2, y2), (x, y2)])


def closest_text_to_panel_index(text_bboxes_xyxy, panel_bboxes_xyxy):
    closest_boxes = []

    # Iterate over each text bounding box
    for t_index, text_box in enumerate(text_bboxes_xyxy):
        # Initialize minimum distance to a large number
        min_dist = float("inf")
        # Initialize nearest box
        # Convert text bounding box to Polygon
        text_poly = polygon_from_xyxy(*text_box)
        # Iterate over each panel bounding box

        p_index = 0
        for p_index, panel_box in enumerate(panel_bboxes_xyxy):
            # Convert panel bounding box to Polygon
            panel_poly = polygon_from_xyxy(*panel_box)
            # Find the nearest points between the text and panel bounding boxes
            nearest_pts = nearest_points(text_poly, panel_poly)
            # Calculate the distance between the nearest points
            dist = nearest_pts[0].distance(nearest_pts[1])
            # If the distance is less than the minimum distance
            if dist < min_dist:
                # Update the minimum distance
                min_dist = dist
                # Update the nearest box
                if not dist:
                    break
        # Append the nearest box to the list of closest boxes
        closest_boxes.append((p_index, t_index))
    order_indices_dict = {i: [] for i in range(len(panel_bboxes_xyxy))}
    for order_index in closest_boxes:
        order_indices_dict[order_index[0]].append(order_index[1])
    return order_indices_dict


def xywh_to_xyxy(xywh):
    return [xywh[0], xywh[1], xywh[0] + xywh[2], xywh[1] + xywh[3]]


def xyxy_to_xywh(xyxy):
    return [xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]]


def reorder_boxes_indices(text_bboxes_xyxy, panel_bboxes_xyxy):
    panel_text_order = closest_text_to_panel_index(text_bboxes_xyxy, panel_bboxes_xyxy)
    box_orders = []
    for i in range(len(panel_bboxes_xyxy)):
        text_inds = panel_text_order[i]

        orders = order_read_direction_scores(
            [xyxy_to_xywh(i) for i in [text_bboxes_xyxy[i] for i in text_inds]],
        )
        # print(orders)
        bbox_inds = np.argsort(orders)[::-1]
        box_orders.extend([text_inds[i] for i in bbox_inds])
    return box_orders


def draw_bboxes(img, text_bboxes_xyxy, panel_bboxes_xyxy):
    image_ = img.copy()
    # Create a drawing object
    draw = ImageDraw.Draw(image_)

    # Draw black boxes on the image
    for i, box in enumerate(text_bboxes_xyxy):
        # draw.rectangle(xywh_to_xyxy(box), fill="black")
        draw.rectangle(box, outline="red")
        draw.text(
            box[:2],
            str(i),
            fill="red",
            stroke_width=2,
            font=ImageFont.truetype("arial.ttf", 50),
        )

    for i, box in enumerate(panel_bboxes_xyxy):
        # draw.rectangle(xywh_to_xyxy(box), fill="black")
        draw.rectangle(box, outline="blue")
        draw.text(
            box[:2],
            str(i),
            fill="blue",
            stroke_width=2,
            font=ImageFont.truetype("arial.ttf", 50),
        )

    # Show the image
    return image_


def extract_text_info_from_ballons(data):
    pages = data["pages"]
    extracted_data = {
        k1: [
            {k: v for k, v in d.items() if k in ["text", "xyxy", "_bounding_rect"]}
            for d in pages[k1]
        ]
        for k1 in pages.keys()
    }
    return extracted_data


def text_bboxes_from_ballons(text_info):
    text_bboxes_xyxy = [i["xyxy"] for i in text_info]
    return text_bboxes_xyxy


def save_panel_text_order(path: Path | str):
    path = Path(path)
    path_json = path / (f"imgtrans_{path.stem}" + ".json")
    pth_out = path.parent / (path.stem + "-panel-text-order")

    if not pth_out.exists():
        pth_out.mkdir()

    # Glob get all images in folder
    with open(path_json, encoding="utf8") as f:
        data = json.load(f)

    pages = data["pages"]
    pages_keys = list(pages.keys())

    for k in pages_keys:
        page_info = pages[k]
        text_bboxes = text_bboxes_from_ballons(page_info)
        img = Image.open(path / k)
        panel_bboxes = calc_panel_bboxes_xyxy(img)

        text_reorderered_index = reorder_boxes_indices(text_bboxes, panel_bboxes)
        text_bboxes = [text_bboxes[i] for i in text_reorderered_index]

        img_out = draw_bboxes(img, text_bboxes, panel_bboxes)
        img_out.save(pth_out / k)

def reorder_text_block_data(path: Path | str):
    path = Path(path)
    path_json = path / (f"imgtrans_{path.stem}" + ".json")

    # Glob get all images in folder
    with open(path_json, encoding="utf8") as f:
        data = json.load(f)

    pages = data["pages"]
    pages_keys = list(pages.keys())

    pages_reordered = {}
    for k in pages_keys:
        page_info = pages[k]
        text_bboxes = text_bboxes_from_ballons(page_info)
        img = Image.open(path / k)
        panel_bboxes = calc_panel_bboxes_xyxy(img)

        text_reorderered_index = reorder_boxes_indices(text_bboxes, panel_bboxes)
        pages_reordered[k] = [page_info[i] for i in text_reorderered_index]

    data["pages"] = pages_reordered

    with open(path_json, 'w', encoding="utf8") as f:
        json.dump(data, f)


if __name__ == "__main__":
    save_draw_contours(sys.argv[1])
    save_panel_text_order(sys.argv[1])
