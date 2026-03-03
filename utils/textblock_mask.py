import cv2
import numpy as np
from typing import Tuple, List, Optional
from .imgproc_utils import draw_connected_labels
from .stroke_width_calculator import strokewidth_check
from .logger import logger as LOGGER

opencv_inpaint = lambda img, mask: cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)


def _cleaning_flags_from_kwargs(kwargs: dict) -> Tuple[bool, bool]:
    """Resolve Section 16 cleaning flags from kwargs or config (pcfg.module)."""
    try:
        from .config import pcfg
        cleaning_otsu = kwargs.get("cleaning_otsu_retry")
        if cleaning_otsu is None:
            cleaning_otsu = getattr(pcfg.module, "cleaning_otsu_retry", True)
        cleaning_adaptive = kwargs.get("cleaning_adaptive_shrink")
        if cleaning_adaptive is None:
            cleaning_adaptive = getattr(pcfg.module, "cleaning_adaptive_shrink_junction", True)
        return bool(cleaning_otsu), bool(cleaning_adaptive)
    except Exception:
        return bool(kwargs.get("cleaning_otsu_retry", True)), bool(kwargs.get("cleaning_adaptive_shrink", True))


def _build_junction_zone_mask(
    crop_h: int,
    crop_w: int,
    self_xyxy: List[float],
    other_xyxy_list: List[List[float]],
    margin_px: int = 12,
) -> np.ndarray:
    """
    Binary mask (crop_h x crop_w), 255 in junction zones: strips along crop edges
    that face a neighboring block. Used so we apply smaller shrink there (no pinching).
    """
    if not other_xyxy_list or margin_px <= 0:
        return np.zeros((crop_h, crop_w), dtype=np.uint8)
    try:
        x1, y1, x2, y2 = float(self_xyxy[0]), float(self_xyxy[1]), float(self_xyxy[2]), float(self_xyxy[3])
    except (IndexError, TypeError):
        return np.zeros((crop_h, crop_w), dtype=np.uint8)
    junction = np.zeros((crop_h, crop_w), dtype=np.uint8)
    use_left = use_right = use_top = use_bottom = False
    for o in other_xyxy_list:
        if not o or len(o) < 4:
            continue
        ox1, oy1, ox2, oy2 = float(o[0]), float(o[1]), float(o[2]), float(o[3])
        if not (oy2 <= y1 or oy1 >= y2):
            if ox2 <= x1 + margin_px and x1 - ox2 <= margin_px * 2:
                use_left = True
            if ox1 >= x2 - margin_px and ox1 - x2 <= margin_px * 2:
                use_right = True
        if not (ox2 <= x1 or ox1 >= x2):
            if oy2 <= y1 + margin_px and y1 - oy2 <= margin_px * 2:
                use_top = True
            if oy1 >= y2 - margin_px and oy1 - y2 <= margin_px * 2:
                use_bottom = True
    m = min(margin_px, crop_w // 2, crop_h // 2)
    if m <= 0:
        return junction
    if use_left:
        junction[:, :m] = 255
    if use_right:
        junction[:, crop_w - m:] = 255
    if use_top:
        junction[:m, :] = 255
    if use_bottom:
        junction[crop_h - m:, :] = 255
    return junction


def _adaptive_shrink_ballon(
    ballon_mask: np.ndarray,
    junction_mask: np.ndarray,
    full_shrink_px: int = 1,
    junction_shrink_px: int = 0,
) -> np.ndarray:
    """Erode ballon_mask with full_shrink_px; in junction zones use junction_shrink_px (smaller)."""
    if full_shrink_px <= 0:
        return ballon_mask
    if junction_mask is None or junction_mask.size == 0 or np.max(junction_mask) == 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * full_shrink_px + 1,) * 2)
        return cv2.erode(ballon_mask, kernel)
    kernel_full = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * full_shrink_px + 1,) * 2)
    eroded_full = cv2.erode(ballon_mask, kernel_full)
    if junction_shrink_px <= 0:
        return np.where(junction_mask > 0, ballon_mask, eroded_full).astype(np.uint8)
    kernel_j = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * junction_shrink_px + 1,) * 2)
    eroded_j = cv2.erode(ballon_mask, kernel_j)
    return np.where(junction_mask > 0, eroded_j, eroded_full).astype(np.uint8)


def classify_bubble_colored(
    img: np.ndarray,
    ballon_mask: np.ndarray,
    text_mask: np.ndarray,
    min_interior_pixels: int = 64,
) -> bool:
    """
    Section 17: Classify bubble interior via grayscale histogram + ratios.
    Returns True if the bubble is colored/gradient (not plain white/light), so we should
    inpaint only the text mask instead of median-filling the whole balloon.
    """
    if img is None or img.size == 0 or ballon_mask is None or text_mask is None:
        return False
    if img.shape[:2] != ballon_mask.shape[:2] or ballon_mask.shape != text_mask.shape:
        return False
    interior = (ballon_mask > 127) & (text_mask <= 127)
    n = int(np.sum(interior))
    if n < min_interior_pixels:
        return False
    if len(img.shape) == 3 and img.shape[2] >= 3:
        rgb = img[:, :, :3]
        if img.shape[2] == 4:
            rgb = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        px = rgb[interior]
        gray = 0.299 * px[:, 0] + 0.587 * px[:, 1] + 0.114 * px[:, 2]
    else:
        gray = np.asarray(img, dtype=np.float64)
        if gray.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = gray[interior]
    g_mean = float(np.mean(gray))
    g_std = float(np.std(gray))
    if g_mean < 240:
        return True
    if g_std > 25:
        return True
    if len(img.shape) == 3 and img.shape[2] >= 3:
        px = img[:, :, :3][interior]
        ch_std = np.std(px, axis=0)
        if np.max(ch_std) > 18:
            return True
    return False


def show_img_by_dict(imgdicts):
    for keyname in imgdicts.keys():
        cv2.imshow(keyname, imgdicts[keyname])
    cv2.waitKey(0)

# 计算文本rgb均值
def letter_calculator(img, mask, bground_rgb, show_process=False, use_otsu=False):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    aver_bground_rgb = 0.299 * bground_rgb[0] + 0.587 * bground_rgb[1] + 0.114 * bground_rgb[2]
    thresh_low = 127
    if use_otsu:
        retval, threshed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        retval, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    if aver_bground_rgb < thresh_low:
        threshed = 255 - threshed
    threshed = 255 - threshed

    threshed = cv2.bitwise_and(threshed, mask)
    le_region = np.where(threshed == 255)
    mat_region = img[le_region]

    if mat_region.shape[0] == 0 and not use_otsu:
        # Second chance: Otsu thresholding (section 16)
        retval, threshed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if aver_bground_rgb < thresh_low:
            threshed = 255 - threshed
        threshed = 255 - threshed
        threshed = cv2.bitwise_and(threshed, mask)
        le_region = np.where(threshed == 255)
        mat_region = img[le_region]

    if mat_region.shape[0] == 0:
        return [-1, -1, -1], threshed

    letter_rgb = np.mean(mat_region, axis=0).astype(int).tolist()
    
    if show_process:
        cv2.imshow("thresh", threshed)
        # ocr_protest(threshed)
        imgcp = np.copy(img)
        imgcp *= 0
        imgcp += 127
        imgcp[le_region] = letter_rgb
        cv2.imshow("letter_img", imgcp)
        # cv2.waitKey(0)
        
    return letter_rgb, threshed

# 预处理让文本颜色提取准确点
def usm(src):
    # Handle RGBA images by converting to RGB for processing
    if len(src.shape) == 3 and src.shape[2] == 4:
        src = cv2.cvtColor(src, cv2.COLOR_RGBA2RGB)
        
    blur_img = cv2.GaussianBlur(src, (0, 0), 5)
    usm = cv2.addWeighted(src, 1.5, blur_img, -0.5, 0)
    h, w = src.shape[:2]
    result = np.zeros([h, w*2, 3], dtype=src.dtype)
    result[0:h,0:w,:] = src
    result[0:h,w:2*w,:] = usm
    return usm

# 计算文本rgb均值方法2，可能用中位数代替均值会好点
def textrgb_calculator(img, text_mask, show_process=False):
    text_mask = cv2.erode(text_mask, (3, 3), iterations=1)
    usm_img = usm(img)
    overall_meanrgb = np.mean(usm_img[np.where(text_mask==255)], axis=0)
    if show_process:
        colored_text_board = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8) + 127
        colored_text_board[np.where(text_mask==255)] = overall_meanrgb
        cv2.imshow("usm", usm_img)
        cv2.imshow("textcolor", colored_text_board)
    return overall_meanrgb.astype(np.uint8)

# 计算背景rgb均值和标准差
def bground_calculator(buble_img, back_ground_mask, dilate=True):
    kernel = np.ones((3,3),np.uint8)
    if dilate:
        back_ground_mask = cv2.dilate(back_ground_mask, kernel, iterations = 1)
    bground_region = np.where(back_ground_mask==0)
    sd = -1
    if len(bground_region[0]) != 0:
        pix_array = buble_img[bground_region]
        bground_aver = np.mean(pix_array, axis=0).astype(int)
        pix_array - bground_aver
        gray = cv2.cvtColor(buble_img, cv2.COLOR_RGB2GRAY)
        gray_pixarray = gray[bground_region]
        gray_aver = np.mean(gray_pixarray)
        gray_pixarray = gray_pixarray - gray_aver
        gray_pixarray = np.power(gray_pixarray, 2)
        # gray_pixarray = np.sqrt(gray_pixarray)
        sd = np.mean(gray_pixarray)
    else: bground_aver = np.array([-1, -1, -1])

    return bground_aver, bground_region, sd

# 输入：文本块roi，分割出文本mask，根据mask计算文本bgr均值和标准差，决定纯色覆盖/inpaint修复
def canny_flood(img, show_process=False, inpaint_sdthresh=10, force_otsu=False, **kwargs):
    cleaning_otsu_retry, cleaning_adaptive_shrink = _cleaning_flags_from_kwargs(kwargs)
    # cv2.setNumThreads(4)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    kernel = np.ones((3,3),np.uint8)
    orih, oriw = img.shape[0], img.shape[1]
    
    # Handle RGBA images by converting to RGB for processing
    if len(img.shape) == 3 and img.shape[2] == 4:
        # Convert RGBA to RGB for processing
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    scaleR = 1
    if orih > 300 and oriw > 300:
        scaleR = 0.6
    elif orih < 120 or oriw < 120:
        scaleR = 1.4

    if scaleR != 1:
        h, w = img.shape[0], img.shape[1]
        orimg = np.copy(img)
        img = cv2.resize(img, (int(w*scaleR), int(h*scaleR)), interpolation=cv2.INTER_AREA)
    h, w = img.shape[0], img.shape[1]
    img_area = h * w

    cpimg = cv2.GaussianBlur(img,(3,3),cv2.BORDER_DEFAULT)
    detected_edges = cv2.Canny(cpimg, 70, 140, L2gradient=True, apertureSize=3)
    cv2.rectangle(detected_edges, (0, 0), (w-1, h-1), WHITE, 1, cv2.LINE_8)

    cons, hiers = cv2.findContours(detected_edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    cv2.rectangle(detected_edges, (0, 0), (w-1, h-1), BLACK, 1, cv2.LINE_8)

    ballon_mask, outer_index = np.zeros((h, w), np.uint8), -1

    min_retval = np.inf
    mask = np.zeros((h, w), np.uint8)
    difres = 10
    seedpnt = (int(w/2), int(h/2))
    for ii in range(len(cons)):
        rect = cv2.boundingRect(cons[ii])
        if rect[2]*rect[3] < img_area*0.4:
            continue
        
        mask = cv2.drawContours(mask, cons, ii, (255), 2)
        cpmask = np.copy(mask)
        cv2.rectangle(mask, (0, 0), (w-1, h-1), WHITE, 1, cv2.LINE_8)
        retval, _, _, rect = cv2.floodFill(cpmask, mask=None, seedPoint=seedpnt,  flags=4, newVal=(127), loDiff=(difres, difres, difres), upDiff=(difres, difres, difres))

        if retval <= img_area * 0.3:
            mask = cv2.drawContours(mask, cons, ii, (0), 2)
        if retval < min_retval and retval > img_area * 0.3:
            min_retval = retval
            ballon_mask = cpmask

    ballon_mask = 127 - ballon_mask
    ballon_mask = cv2.dilate(ballon_mask, kernel,iterations = 1)
    outer_area, _, _, rect = cv2.floodFill(ballon_mask, mask=None, seedPoint=seedpnt,  flags=4, newVal=(30), loDiff=(difres, difres, difres), upDiff=(difres, difres, difres))
    ballon_mask = 30 - ballon_mask    
    retval, ballon_mask = cv2.threshold(ballon_mask, 1, 255, cv2.THRESH_BINARY)
    ballon_mask = cv2.bitwise_not(ballon_mask, ballon_mask)

    # Section 16: adaptive shrink in junction zones (smaller shrink near conjoined bubbles)
    self_xyxy = kwargs.get("self_xyxy")
    other_xyxy_list = kwargs.get("neighbor_xyxy_list") or []
    if self_xyxy and len(self_xyxy) >= 4 and (other_xyxy_list or cleaning_adaptive_shrink):
        jmask = _build_junction_zone_mask(h, w, self_xyxy, other_xyxy_list, margin_px=12)
        if np.max(jmask) > 0:
            ballon_mask = _adaptive_shrink_ballon(ballon_mask, jmask, full_shrink_px=1, junction_shrink_px=0)

    detected_edges = cv2.dilate(detected_edges, kernel, iterations = 1)
    for ii in range(2):
        ballon_mask_prev = ballon_mask.copy()
        detected_edges = cv2.bitwise_and(detected_edges, ballon_mask)
        mask = np.copy(detected_edges)
        bgarea1, _, _, rect = cv2.floodFill(mask, mask=None, seedPoint=(0, 0),  flags=4, newVal=(127), loDiff=(difres, difres, difres), upDiff=(difres, difres, difres))
        bgarea2, _, _, rect = cv2.floodFill(mask, mask=None, seedPoint=(detected_edges.shape[1]-1, detected_edges.shape[0]-1),  flags=4, newVal=(127), loDiff=(difres, difres, difres), upDiff=(difres, difres, difres))
        txt_area = min(img_area - bgarea1, img_area - bgarea2)
        ratio_ob = txt_area / outer_area
        ballon_mask = cv2.erode(ballon_mask, kernel,iterations = 1)
        if self_xyxy and len(self_xyxy) >= 4 and (other_xyxy_list or cleaning_adaptive_shrink):
            jmask = _build_junction_zone_mask(h, w, self_xyxy, other_xyxy_list, margin_px=12)
            if np.max(jmask) > 0:
                ballon_mask = np.where(jmask > 0, ballon_mask_prev, ballon_mask).astype(np.uint8)
        if ratio_ob < 0.85:
            break

    mask = 127 - mask
    retval, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    if scaleR != 1:
        img = orimg
        ballon_mask = cv2.resize(ballon_mask, (oriw, orih))
        mask = cv2.resize(mask, (oriw, orih))

    bg_mask = cv2.bitwise_or(mask, 255-ballon_mask)
    mask = cv2.bitwise_and(mask, ballon_mask)

    bground_aver, bground_region, sd = bground_calculator(img, bg_mask)
    inner_rect = None
    threshed = np.zeros((img.shape[0], img.shape[1]), np.uint8)

    if bground_aver[0] != -1:
        letter_aver, threshed = letter_calculator(img, mask, bground_aver, show_process=show_process, use_otsu=force_otsu)
        if letter_aver[0] != -1:
            mask = cv2.dilate(threshed, kernel, iterations=1)
            inner_rect = cv2.boundingRect(cv2.findNonZero(mask))
    else: letter_aver = [0, 0, 0]

    if sd != -1 and sd < inpaint_sdthresh:
        need_inpaint = False
    else:
        need_inpaint = True
    is_colored = False
    try:
        if getattr(__import__("utils.config", fromlist=["pcfg"]).pcfg.module, "colored_bubble_handling", True):
            is_colored = classify_bubble_colored(img, ballon_mask, mask, min_interior_pixels=64)
            if is_colored:
                need_inpaint = True
    except Exception:
        pass
    if show_process:
        print(f"\nneed_inpaint: {need_inpaint}, sd: {sd}, {type(inner_rect)}")
        show_img_by_dict({"outermask": ballon_mask, "detect": detected_edges, "mask": mask})


    if isinstance(inner_rect, tuple):
        inner_rect = [ii for ii in inner_rect]
    if inner_rect is None:
        inner_rect = [-1, -1, -1, -1]
    else:
        inner_rect.append(-1)
    
    bground_aver = bground_aver.astype(np.uint8)
    bub_dict = {"rgb": letter_aver,
                "bground_rgb": bground_aver,
                "inner_rect": inner_rect,
                "need_inpaint": need_inpaint,
                "is_colored_bubble": is_colored}
    
    # Section 16: Otsu retry if fixed threshold gave empty/invalid result
    if not force_otsu and cleaning_otsu_retry:
        bad = (
            (mask is None or mask.size == 0 or np.sum(mask > 0) == 0)
            or (isinstance(letter_aver, (list, tuple)) and len(letter_aver) >= 1 and letter_aver[0] == -1)
        )
        if bad:
            return canny_flood(
                img, show_process=show_process, inpaint_sdthresh=inpaint_sdthresh,
                force_otsu=True, **{k: v for k, v in kwargs.items() if k not in ("force_otsu",)}
            )
    
    return mask, ballon_mask, bub_dict

# 输入：文本块roi，分割出文本mask，根据mask计算文本bgr均值和标准差，决定纯色覆盖/inpaint修复
def connected_canny_flood(img, show_process=False, inpaint_sdthresh=10, apply_strokewidth_check=0, force_otsu=False, **kwargs):
    cleaning_otsu_retry, cleaning_adaptive_shrink = _cleaning_flags_from_kwargs(kwargs)

    # Handle RGBA images by converting to RGB for processing
    if len(img.shape) == 3 and img.shape[2] == 4:
        # Convert RGBA to RGB for processing
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    # 寻找最可能是气泡的外轮廓mask
    def find_outermask(img):
        connectivity = 4
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity, cv2.CV_16U)
        drawtext = np.zeros((img.shape[0], img.shape[1]), np.uint8)
        
        max_ind = np.argmax(stats[:, 4])
        maxbbox_area, sec_ind = -1, -1
        for ind, stat in enumerate(stats):
            if ind != max_ind:
                bbarea = stat[2] * stat[3]
                if bbarea > maxbbox_area:
                    maxbbox_area = bbarea
                    sec_ind = ind
        drawtext[np.where(labels==max_ind)] = 255
        
        cv2.rectangle(drawtext, (0, 0), (img.shape[1]-1, img.shape[0]-1), (0, 0, 0), 1, cv2.LINE_8)
        cons, hiers = cv2.findContours(drawtext, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        img_area = img.shape[0] * img.shape[1]

        rects = np.array([cv2.boundingRect(cnt) for cnt in cons])
        rect_area = np.array([rect[2] * rect[3] for rect in rects])
        quali_ind = np.where(rect_area > img_area * 0.3)[0]
        ballon_mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
        for ind in quali_ind:
            ballon_mask = cv2.drawContours(ballon_mask, cons, ind, (255), 2)
        
        seedpnt = (int(ballon_mask.shape[1]/2), int(ballon_mask.shape[0]/2))
        difres = 10
        retval, _, _, rect = cv2.floodFill(ballon_mask, mask=None, seedPoint=seedpnt,  flags=4, newVal=(127), loDiff=(difres, difres, difres), upDiff=(difres, difres, difres))
        ballon_mask = 255 - cv2.threshold(ballon_mask - 127, 1, 255, cv2.THRESH_BINARY)[1]
        return num_labels, labels, stats, centroids, ballon_mask

    # BGR直接转灰度图可能导致文本区域和背景难以区分，比如测试样例中的黑底红字
    # 但是总有一个通道文本和背景容易区分
    # 返回最容易区分的那个通道
    def ccctest(img, crop_r=0.1):
        # img = usm(img)
        maxh = 100
        if img.shape[0] > maxh:
            scaleR = maxh / img.shape[0]
            im = cv2.resize(img, (int(img.shape[1]*scaleR), int(img.shape[0]*scaleR)), interpolation=cv2.INTER_AREA)
        else:
            im = img

        textlabel_counter = 0
        reverse = False
        c_ind = 0

        num_labels, labels, stats, centroids, pseduo_outermask = find_outermask(cv2.threshold(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), 1, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)[1])
        grayim = np.expand_dims(np.array(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)), axis=2)
        im = np.append(im, grayim, axis=2)
        outer_cords = np.where(pseduo_outermask==255)
        for bgr_ind in range(4):
            channel = im[:, :, bgr_ind]
            ret, thresh = cv2.threshold(channel, 1, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)

            tmp_reverse = False
            
            if np.mean(thresh[outer_cords]) > 160:
                thresh = 255 - thresh
                tmp_reverse = True

            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, 4, cv2.CV_16U)
            # draw_connected_labels(num_labels, labels, stats, centroids)
            # cv2.waitKey(0)
            max_ind = np.argmax(stats[:, 4])
            maxr, minr = 0.5, 0.001
            maxw, maxh = stats[max_ind][2] * maxr, stats[max_ind][3] * maxr
            minarea = im.shape[0] * im.shape[1] * minr

            tmp_counter = 0
            for stat in stats:
                bboxarea = stat[2] * stat[3]
                if stat[2] < maxw and stat[3] < maxh and bboxarea > minarea:
                    tmp_counter += 1
            if tmp_counter > textlabel_counter:
                textlabel_counter = tmp_counter
                c_ind = bgr_ind
                reverse = tmp_reverse
        return c_ind, reverse
    
    channel_index, reverse = ccctest(img)
    chanel = img[:, :, channel_index] if channel_index < 3 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(chanel, 1, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
    
    # Optional: fixed threshold first, then Otsu on retry (section 16)
    if force_otsu:
        pass  # already Otsu above
    # else we keep Otsu for connected path; retry below if result bad
    
    # reverse to get white text on black bg
    if reverse:
        thresh = 255 - thresh
    num_labels, labels, stats, centroids, ballon_mask = find_outermask(thresh)
    img_area = img.shape[0] * img.shape[1]

    # Section 16: adaptive shrink in junction zones
    self_xyxy = kwargs.get("self_xyxy")
    other_xyxy_list = kwargs.get("neighbor_xyxy_list") or []
    if self_xyxy and len(self_xyxy) >= 4 and (other_xyxy_list or cleaning_adaptive_shrink):
        jmask = _build_junction_zone_mask(ballon_mask.shape[0], ballon_mask.shape[1], self_xyxy, other_xyxy_list, margin_px=12)
        if np.max(jmask) > 0:
            ballon_mask = _adaptive_shrink_ballon(ballon_mask, jmask, full_shrink_px=1, junction_shrink_px=0)

    text_mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    max_ind = np.argmax(stats[:, 4])
    for lab in (range(num_labels)):
        stat = stats[lab]
        if lab != max_ind and stat[4] < img_area * 0.4:
            labcord = np.where(labels==lab)
            text_mask[labcord] = 255

    text_mask = cv2.bitwise_and(text_mask, ballon_mask)
    if apply_strokewidth_check > 0:
        text_mask = strokewidth_check(text_mask, labels, num_labels, stats, debug_type=show_process-1)
        
    text_color = textrgb_calculator(img, text_mask, show_process=show_process)
    inner_rect = cv2.boundingRect(cv2.findNonZero(cv2.dilate(text_mask, (3, 3), iterations=1)))
    inner_rect = [ii for ii in inner_rect]
    inner_rect.append(-1)

    bg_mask = cv2.bitwise_or(text_mask, 255-ballon_mask)

    bground_aver, bground_region, sd = bground_calculator(img, bg_mask)

    mask = cv2.GaussianBlur(text_mask,(3,3),cv2.BORDER_DEFAULT)
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    if sd != -1 and sd < inpaint_sdthresh:
        need_inpaint = False
    else:
        need_inpaint = True
    is_colored = False
    try:
        if getattr(__import__("utils.config", fromlist=["pcfg"]).pcfg.module, "colored_bubble_handling", True):
            is_colored = classify_bubble_colored(img, ballon_mask, text_mask, min_interior_pixels=64)
            if is_colored:
                need_inpaint = True
    except Exception:
        pass

    if show_process:
        print(f"\nuse inpaint: {need_inpaint}, sd: {sd}, {type(inner_rect)}")
        draw_connected_labels(num_labels, labels, stats, centroids)
        show_img_by_dict({"thresh": thresh, "ori": img, "outer": ballon_mask, "text": text_mask, "bgmask": bg_mask})

    bground_aver = bground_aver.astype(np.uint8)
    bub_dict = {"rgb": text_color,
                "bground_rgb": bground_aver,
                "inner_rect": inner_rect,
                "need_inpaint": need_inpaint,
                "is_colored_bubble": is_colored}
    
    # Section 16: Otsu retry (retry whole cleaning with force_otsu)
    if not force_otsu and cleaning_otsu_retry:
        bad = (
            (mask is None or mask.size == 0 or np.sum(mask > 0) == 0)
            or (text_color is not None and np.all(np.array(text_color) == 0))
        )
        if bad:
            return connected_canny_flood(
                img, show_process=show_process, inpaint_sdthresh=inpaint_sdthresh,
                apply_strokewidth_check=apply_strokewidth_check, force_otsu=True,
                **{k: v for k, v in kwargs.items() if k != "force_otsu"}
            )
    
    return mask, ballon_mask, bub_dict


def existing_mask(img, mask: np.ndarray):
    bub_dict = {"rgb": [0, 0, 0],"bground_rgb": [255, 255, 255],"need_inpaint": True, "is_colored_bubble": False}
    return mask, mask, bub_dict


_SAM_REFINER = None
_SAM_REFINER_INIT_ERR = None
_SAM_REFINER_WARNED = False


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(v)))


def _bbox_from_mask(mask: np.ndarray, pad: int = 0) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        return -1, -1, -1, -1
    y1 = int(ys.min())
    y2 = int(ys.max()) + 1
    x1 = int(xs.min())
    x2 = int(xs.max()) + 1
    if pad > 0:
        h, w = mask.shape[:2]
        x1 = _clamp(x1 - pad, 0, w - 1)
        y1 = _clamp(y1 - pad, 0, h - 1)
        x2 = _clamp(x2 + pad, 1, w)
        y2 = _clamp(y2 + pad, 1, h)
    return x1, y1, x2, y2


def _point_from_mask(mask: np.ndarray) -> Tuple[int, int]:
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        h, w = mask.shape[:2]
        return int(w // 2), int(h // 2)
    return int(xs.mean()), int(ys.mean())


class _SamMaskRefiner:
    def __init__(self, model_id: str, device: str = "") -> None:
        self.model_id = (model_id or "").strip() or "facebook/sam2.1-hiera-large"
        self._device_pref = (device or "").strip().lower()
        self._model = None
        self._processor = None
        self._backend = None  # "sam2" | "sam3"

    def _resolve_device(self):
        import torch

        if self._device_pref in {"cuda", "cpu"}:
            return self._device_pref
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _ensure_loaded(self):
        if self._model is not None and self._processor is not None:
            return

        # Lazy import so the app still works without SAM-capable transformers.
        model_id = self.model_id
        device = self._resolve_device()

        try:
            if "sam3" in model_id.lower():
                from transformers import Sam3TrackerModel, Sam3TrackerProcessor

                self._backend = "sam3"
                self._processor = Sam3TrackerProcessor.from_pretrained(model_id)
                self._model = Sam3TrackerModel.from_pretrained(model_id)
            else:
                from transformers import Sam2Model, Sam2Processor

                self._backend = "sam2"
                self._processor = Sam2Processor.from_pretrained(model_id)
                self._model = Sam2Model.from_pretrained(model_id)
        except Exception as e:
            raise RuntimeError(
                "SAM refine balloon requires a newer `transformers` build with SAM2/SAM3 classes, "
                f"and access to the model ({model_id}). Original error: {e}"
            ) from e

        try:
            self._model.to(device)
        except Exception:
            pass

        try:
            self._model.eval()
        except Exception:
            pass

    def _select_best_mask(self, cand_masks: np.ndarray, coarse_mask: np.ndarray) -> np.ndarray:
        """
        cand_masks: (K, H, W) bool/uint8
        coarse_mask: (H, W) uint8
        """
        if cand_masks is None or cand_masks.size == 0:
            return None
        coarse = (coarse_mask > 0)
        best_iou = -1.0
        best = None
        for k in range(int(cand_masks.shape[0])):
            m = cand_masks[k].astype(bool)
            inter = float(np.logical_and(m, coarse).sum())
            union = float(np.logical_or(m, coarse).sum())
            iou = inter / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou = iou
                best = m
        return best

    def refine(self, img_rgb: np.ndarray, coarse_mask: np.ndarray, pad_px: int = 12) -> np.ndarray:
        self._ensure_loaded()

        if coarse_mask is None or coarse_mask.size == 0:
            return None

        # Normalize inputs
        if img_rgb.ndim == 3 and img_rgb.shape[2] == 4:
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGBA2RGB)

        h, w = img_rgb.shape[:2]
        x1, y1, x2, y2 = _bbox_from_mask(coarse_mask, pad=int(pad_px))
        if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
            return None

        from PIL import Image
        import torch

        image = Image.fromarray(img_rgb.astype(np.uint8), mode="RGB")
        device = getattr(self._model, "device", None) or self._resolve_device()

        if self._backend == "sam3":
            px, py = _point_from_mask(coarse_mask)
            input_points = [[[[int(px), int(py)]]]]
            input_labels = [[[1]]]
            inputs = self._processor(
                images=image,
                input_points=input_points,
                input_labels=input_labels,
                return_tensors="pt",
            )
        else:
            input_boxes = torch.tensor([[[float(x1), float(y1), float(x2), float(y2)]]])
            inputs = self._processor(images=image, input_boxes=input_boxes, return_tensors="pt")

        try:
            inputs = inputs.to(device)
        except Exception:
            pass

        with torch.no_grad():
            outputs = self._model(**inputs)

        # post_process_masks returns a list per image; each is typically (num_objects, num_masks, H, W)
        try:
            pp = self._processor.post_process_masks(outputs.pred_masks, inputs["original_sizes"])
        except Exception:
            # Some builds expect CPU tensors
            pp = self._processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])
        masks = pp[0]

        # Flatten to (K, H, W)
        if hasattr(masks, "cpu"):
            masks_np = masks.cpu().numpy()
        else:
            masks_np = np.asarray(masks)

        if masks_np.ndim == 4:
            # (Nobj, Nmasks, H, W) -> take first object, all candidate masks
            masks_np = masks_np[0]
        elif masks_np.ndim == 3:
            pass
        else:
            return None

        # Threshold
        cand = masks_np > 0.5
        best = self._select_best_mask(cand, coarse_mask)
        if best is None:
            return None

        refined = (best.astype(np.uint8) * 255)
        # Ensure same size as input
        if refined.shape[0] != h or refined.shape[1] != w:
            refined = cv2.resize(refined, (w, h), interpolation=cv2.INTER_NEAREST)
        return refined


def _get_sam_refiner() -> _SamMaskRefiner:
    global _SAM_REFINER, _SAM_REFINER_INIT_ERR
    if _SAM_REFINER is not None:
        return _SAM_REFINER
    if _SAM_REFINER_INIT_ERR is not None:
        raise _SAM_REFINER_INIT_ERR

    try:
        from utils.config import pcfg

        model_id = getattr(pcfg.drawpanel, "sam_maskrefine_model_id", "") or "facebook/sam2.1-hiera-large"
        device = getattr(pcfg.drawpanel, "sam_maskrefine_device", "") or ""
    except Exception:
        model_id = "facebook/sam2.1-hiera-large"
        device = ""

    try:
        _SAM_REFINER = _SamMaskRefiner(model_id=model_id, device=device)
        return _SAM_REFINER
    except Exception as e:
        _SAM_REFINER_INIT_ERR = e
        raise


def sam_refine_ballon_mask(img, show_process=False, inpaint_sdthresh=10, **kwargs):
    """
    Mask-seg method: run existing canny+flood, then refine the balloon mask with SAM2/SAM3.

    - If SAM isn't available (missing/old transformers, model access), falls back to `canny_flood`.
    - Returns (text_mask, balloon_mask, bub_dict) like other methods.
    """
    # Start from the current stable heuristic
    text_mask, ballon_mask, bub_dict = canny_flood(img, show_process=show_process, inpaint_sdthresh=inpaint_sdthresh, **kwargs)

    if ballon_mask is None or ballon_mask.size == 0:
        return text_mask, ballon_mask, bub_dict

    try:
        from utils.config import pcfg

        pad_px = int(getattr(pcfg.drawpanel, "sam_maskrefine_padding_px", 12) or 12)
    except Exception:
        pad_px = 12

    try:
        refiner = _get_sam_refiner()
        refined = refiner.refine(img, ballon_mask, pad_px=pad_px)
        if refined is not None and refined.size > 0:
            ballon_mask = refined
            if text_mask is not None and text_mask.shape == ballon_mask.shape:
                text_mask = cv2.bitwise_and(text_mask, ballon_mask)
    except Exception as e:
        global _SAM_REFINER_WARNED
        if not _SAM_REFINER_WARNED:
            _SAM_REFINER_WARNED = True
            LOGGER.warning(f"SAM refine balloon unavailable; falling back to Canny+flood. ({e})")

    return text_mask, ballon_mask, bub_dict


def extract_ballon_mask(img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Given original img and text mask (cropped)
    return ballon mask & non text mask
    '''
    # Handle RGBA images by converting to RGB for processing
    if len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        
    img = cv2.GaussianBlur(img,(3,3),cv2.BORDER_DEFAULT)
    h, w = img.shape[:2]
    text_sum = np.sum(mask)
    if text_sum == 0:
        return None, None
    cannyed = cv2.Canny(img, 70, 140, L2gradient=True, apertureSize=3)
    e_size = 1
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * e_size + 1, 2 * e_size + 1),(e_size, e_size))
    cannyed = cv2.dilate(cannyed, element, iterations=1)
    br = cv2.boundingRect(cv2.findNonZero(mask))
    br_xyxy = [br[0], br[1], br[0] + br[2], br[1] + br[3]]

    # draw the bounding rect in case there is no closed ballon
    cv2.rectangle(cannyed, (0, 0), (w-1, h-1), (255, 255, 255), 1, cv2.LINE_8)
    cannyed = cv2.bitwise_and(cannyed, 255 - mask)

    cons, _ = cv2.findContours(cannyed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    min_ballon_area = w * h
    ballon_mask = None
    non_text_mask = None
    # minimum contour which covers all text mask must be the ballon
    for ii, con in enumerate(cons):
        br_c = cv2.boundingRect(con)
        br_c = [br_c[0], br_c[1], br_c[0] + br_c[2], br_c[1] + br_c[3]]
        if br_c[0] > br_xyxy[0] or br_c[1] > br_xyxy[1] or br_c[2] < br_xyxy[2] or br_c[3] < br_xyxy[3]:
            continue
        tmp = np.zeros_like(cannyed)
        cv2.drawContours(tmp, cons, ii, (255, 255, 255), -1, cv2.LINE_8)
        if cv2.bitwise_and(tmp, mask).sum() >= text_sum:
            con_area = cv2.contourArea(con)
            if con_area < min_ballon_area:
                min_ballon_area = con_area
                ballon_mask = tmp
    if ballon_mask is not None:
        non_text_mask = cv2.bitwise_and(ballon_mask, 255 - mask)
    #     cv2.imshow('ballon', ballon_mask)
    #     cv2.imshow('non_text', non_text_mask)
    # cv2.imshow('im', img)
    # cv2.imshow('msk', mask)
    # cv2.imshow('canny', cannyed)
    # cv2.waitKey(0)

    return ballon_mask, non_text_mask