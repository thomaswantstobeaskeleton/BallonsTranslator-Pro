"""
Pipeline Handlers — composable, stateless pipeline step classes.

Replaces monolithic pipeline methods in ModuleThread with discrete handlers
that can be assembled into different pipeline configurations (imgtrans,
webtoon, batch, block-level, etc.).

Each handler implements:
  __call__(context: PipelineContext) -> PipelineContext

Context carries mutable state forward; handlers are pure functions
over context, making them testable and chainable.

Inspired by Comic Translate's pipeline step pattern and modern
functional pipeline composition.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable
import copy

import numpy as np
from qtpy.QtCore import QObject, Signal

from utils.textblock import TextBlock, sort_regions, remove_contained_boxes, deduplicate_primary_boxes
from utils.imgproc_utils import enlarge_window, union_area, get_block_mask
from utils.proj_imgtrans import ProjImgTrans
from utils.config import pcfg
from utils.logger import logger as LOGGER
from utils.cancellation import CancelFlag
from modules.ocr.base import normalize_block_text, ensure_blocks_have_lines
from modules.textdetector.outside_text_processor import filter_osb_overlapping_bubbles
from ui.funcmaps import get_maskseg_method


@dataclass
class PipelineContext:
    """
    Mutable context passed through each pipeline handler.

    Fields are optional so handlers can be run independently.
    """
    imgtrans_proj: Optional[ProjImgTrans] = None
    page_key: Optional[str] = None
    page_index: int = 0

    # Image arrays
    img_array: Optional[np.ndarray] = None
    inpainted_array: Optional[np.ndarray] = None
    mask_array: Optional[np.ndarray] = None

    # Blocks
    block_list: List[TextBlock] = field(default_factory=list)
    blk_ids: List[int] = field(default_factory=list)

    # Progress / control
    progress_pct: float = 0.0
    cancel_flag: Optional[CancelFlag] = None
    stop_requested: bool = False

    # Metadata
    extra: Dict = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def check_cancelled(self) -> bool:
        if self.stop_requested:
            return True
        if self.cancel_flag is not None and self.cancel_flag.is_set():
            return True
        return False


class PipelineHandler:
    """Base class for a pipeline step handler."""

    name: str = "handler"

    def __call__(self, ctx: PipelineContext) -> PipelineContext:
        """Execute the handler and return (possibly mutated) context."""
        raise NotImplementedError


class DetectHandler(PipelineHandler):
    """
    Text detection handler.

    Args:
        textdetector: A TextDetectorBase instance.
        config: PipelineConfig or dict with detection settings.
    """

    name = "detect"

    def __init__(
        self,
        textdetector,
        extra_textdetectors=None,
        merge_overlap_thres: float = 0.3,
        enlarge_offset: int = 0,
    ) -> None:
        self.textdetector = textdetector
        self.extra_textdetectors = extra_textdetectors or []
        self.merge_overlap_thres = merge_overlap_thres
        self.enlarge_offset = enlarge_offset

    def __call__(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.check_cancelled():
            return ctx
        img = ctx.img_array
        if img is None:
            ctx.errors.append("DetectHandler: img_array is None")
            return ctx

        try:
            blk_list = self.textdetector.detect(img)
        except Exception as e:
            LOGGER.error("Detection failed: %s", e)
            ctx.errors.append(f"Detection failed: {e}")
            return ctx

        # Extra detectors (e.g. outside-text processor)
        for extra_det in self.extra_textdetectors:
            try:
                extra_blks = extra_det.detect(img)
                if extra_blks:
                    blk_list.extend(extra_blks)
            except Exception as e:
                LOGGER.warning("Extra detector failed: %s", e)

        # Post-process
        if blk_list:
            blk_list = sort_regions(blk_list)
            if self.merge_overlap_thres > 0:
                blk_list = remove_contained_boxes(blk_list)
                blk_list = deduplicate_primary_boxes(blk_list, overlap_threshold=self.merge_overlap_thres)
            if self.enlarge_offset > 0:
                for blk in blk_list:
                    blk.xyxy = enlarge_window(
                        blk.xyxy, img.shape[1], img.shape[0], offset=self.enlarge_offset
                    )

        ctx.block_list = blk_list
        return ctx


class OCRHandler(PipelineHandler):
    """
    OCR handler.

    Args:
        ocr: An OCRBase instance.
    """

    name = "ocr"

    def __init__(self, ocr, split_textblk: bool = True) -> None:
        self.ocr = ocr
        self.split_textblk = split_textblk

    def __call__(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.check_cancelled():
            return ctx
        img = ctx.img_array
        blk_list = ctx.block_list
        if img is None or not blk_list:
            return ctx

        try:
            self.ocr.run_ocr(img, blk_list, split_textblk=self.split_textblk)
        except Exception as e:
            LOGGER.error("OCR failed: %s", e)
            ctx.errors.append(f"OCR failed: {e}")
            return ctx

        # Post-process OCR results
        for blk in blk_list:
            normalize_block_text(blk)
            ensure_blocks_have_lines(blk)

        return ctx


class TranslateHandler(PipelineHandler):
    """
    Translation handler.

    Args:
        translator: A BaseTranslator instance.
    """

    name = "translate"

    def __init__(
        self,
        translator,
        page_context_builder=None,  # Optional: modules.translation.page_context_builder
        set_context_fn=None,
        target_lang: str = "English",
    ) -> None:
        self.translator = translator
        self.page_context_builder = page_context_builder
        self.set_context_fn = set_context_fn
        self.target_lang = target_lang

    def __call__(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.check_cancelled():
            return ctx
        blk_list = ctx.block_list
        if not blk_list:
            return ctx

        # Set context if provided (page-level glossary, series context, etc.)
        if self.set_context_fn is not None:
            try:
                self.set_context_fn(ctx)
            except Exception as e:
                LOGGER.warning("Failed to set translation context: %s", e)

        try:
            self.translator.translate_textblk_lst(blk_list)
        except Exception as e:
            LOGGER.error("Translation failed: %s", e)
            ctx.errors.append(f"Translation failed: {e}")

        return ctx


class InpaintHandler(PipelineHandler):
    """
    Inpainting handler.

    Args:
        inpainter: An InpainterBase instance.
    """

    name = "inpaint"

    def __init__(self, inpainter, maskseg_method=None) -> None:
        self.inpainter = inpainter
        self.maskseg_method = maskseg_method or get_maskseg_method()

    def _cleaning_kwargs_for_block(
        self, blk_list: List[TextBlock], idx: int, x1: int, y1: int, x2: int, y2: int
    ) -> dict:
        """Compute self_xyxy and neighbor_xyxy_list in crop coords."""
        blk = blk_list[idx]
        crop_w, crop_h = x2 - x1, y2 - y1
        try:
            bx1, by1, bx2, by2 = float(blk.xyxy[0]), float(blk.xyxy[1]), float(blk.xyxy[2]), float(blk.xyxy[3])
        except (IndexError, TypeError):
            return {}
        self_xyxy = [bx1 - x1, by1 - y1, bx2 - x1, by2 - y1]
        margin = 20
        neighbor_xyxy_list = []
        for j, other in enumerate(blk_list):
            if j == idx:
                continue
            try:
                ox1, oy1, ox2, oy2 = float(other.xyxy[0]), float(other.xyxy[1]), float(other.xyxy[2]), float(other.xyxy[3])
            except (IndexError, TypeError):
                continue
            if not (ox2 < x1 - margin or ox1 > x2 + margin or oy2 < y1 - margin or oy1 > y2 + margin):
                nc = [
                    max(0.0, ox1 - x1), max(0.0, oy1 - y1),
                    min(float(crop_w), ox2 - x1), min(float(crop_h), oy2 - y1)
                ]
                if nc[2] > nc[0] and nc[3] > nc[1]:
                    neighbor_xyxy_list.append(nc)
        return {"self_xyxy": self_xyxy, "neighbor_xyxy_list": neighbor_xyxy_list}

    def __call__(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.check_cancelled():
            return ctx
        img = ctx.img_array
        mask = ctx.mask_array
        blk_list = ctx.block_list
        if img is None or not blk_list:
            return ctx

        im_h, im_w = img.shape[:2]
        progress_prod = 100.0 / len(blk_list) if len(blk_list) > 0 else 0

        for ii, blk in enumerate(blk_list):
            if ctx.check_cancelled():
                break
            xyxy = enlarge_window(blk.xyxy, im_w, im_h)
            xyxy = np.array(xyxy)
            x1, y1, x2, y2 = xyxy.astype(np.int64)
            blk.region_inpaint_dict = None
            if y2 - y1 > 2 and x2 - x1 > 2:
                im = np.copy(img[y1:y2, x1:x2])
                crop_mask = mask[y1:y2, x1:x2] if mask is not None else None
                cleaning_kwargs = self._cleaning_kwargs_for_block(blk_list, ii, int(x1), int(y1), int(x2), int(y2))
                try:
                    inpaint_mask_array, _, _ = self.maskseg_method(im, mask=crop_mask, **cleaning_kwargs)
                except Exception as e:
                    LOGGER.warning("Mask segmentation failed for block %d: %s", ii, e)
                    continue
                if inpaint_mask_array is not None and inpaint_mask_array.size > 0:
                    kernel = np.ones((7, 7), np.uint8)
                    inpaint_mask_array = cv2.dilate(
                        (inpaint_mask_array > 127).astype(np.uint8) * 255, kernel, iterations=1
                    )
                proc_mask = self._post_process_mask(inpaint_mask_array)
                if proc_mask is not None and proc_mask.sum() > 0:
                    try:
                        inpainted = self.inpainter.inpaint(im, proc_mask)
                        blk.region_inpaint_dict = {
                            'img': im, 'mask': proc_mask,
                            'inpaint_rect': [int(x1), int(y1), int(x2), int(y2)],
                            'inpainted': inpainted
                        }
                    except Exception as e:
                        LOGGER.warning("Inpainting failed for block %d: %s", ii, e)
            ctx.progress_pct = int((ii + 1) * progress_prod)

        return ctx

    @staticmethod
    def _post_process_mask(mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if mask is None:
            return None
        # Simple post-processing: binary threshold and cleanup
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        return binary


class LayoutHandler(PipelineHandler):
    """
    Text layout handler: compute font sizes, alignment, and positioning.

    Args:
        layout_engine: A callable taking (blk_list, img_shape) -> None.
    """

    name = "layout"

    def __init__(self, layout_engine=None) -> None:
        self.layout_engine = layout_engine

    def __call__(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.check_cancelled():
            return ctx
        if self.layout_engine is None:
            return ctx
        img = ctx.img_array
        blk_list = ctx.block_list
        if img is None or not blk_list:
            return ctx
        try:
            self.layout_engine(blk_list, img.shape)
        except Exception as e:
            LOGGER.warning("Layout engine failed: %s", e)
        return ctx


class PipelineComposer:
    """
    Compose multiple handlers into a single callable pipeline.

    Usage:
        pipeline = PipelineComposer([
            DetectHandler(detector),
            OCRHandler(ocr),
            TranslateHandler(translator),
            InpaintHandler(inpainter),
        ])
        ctx = PipelineContext(img_array=img, imgtrans_proj=proj, page_key=key)
        ctx = pipeline(ctx)
    """

    def __init__(self, handlers: List[PipelineHandler], name: str = "pipeline") -> None:
        self.handlers = handlers
        self.name = name

    def __call__(self, ctx: PipelineContext) -> PipelineContext:
        for handler in self.handlers:
            if ctx.check_cancelled():
                break
            LOGGER.debug("Running pipeline handler: %s", handler.name)
            ctx = handler(ctx)
        return ctx

    def __or__(self, other: "PipelineComposer") -> "PipelineComposer":
        """Compose two pipelines: p1 | p2"""
        return PipelineComposer(self.handlers + other.handlers)


# --- Pre-built pipeline presets ---

def build_imgtrans_pipeline(
    textdetector,
    ocr,
    translator,
    inpainter,
    extra_textdetectors=None,
    layout_engine=None,
) -> PipelineComposer:
    """Standard 4-step pipeline: detect -> OCR -> translate -> inpaint -> layout."""
    return PipelineComposer([
        DetectHandler(textdetector, extra_textdetectors=extra_textdetectors),
        OCRHandler(ocr),
        TranslateHandler(translator),
        InpaintHandler(inpainter),
        LayoutHandler(layout_engine),
    ], name="imgtrans")


def build_block_pipeline(
    ocr,
    translator,
    inpainter,
    layout_engine=None,
) -> PipelineComposer:
    """Block-level pipeline (blocks already known): OCR -> translate -> inpaint -> layout."""
    return PipelineComposer([
        OCRHandler(ocr),
        TranslateHandler(translator),
        InpaintHandler(inpainter),
        LayoutHandler(layout_engine),
    ], name="blktrans")


def build_detect_only_pipeline(textdetector) -> PipelineComposer:
    """Detection-only pipeline."""
    return PipelineComposer([DetectHandler(textdetector)], name="detect_only")


def build_ocr_only_pipeline(ocr) -> PipelineComposer:
    """OCR-only pipeline (blocks already known)."""
    return PipelineComposer([OCRHandler(ocr)], name="ocr_only")
