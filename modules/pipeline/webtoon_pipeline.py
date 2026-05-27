"""
Webtoon Batch Pipeline — streaming pipeline for long-scroll / webtoon content.

Extends the standard imgtrans pipeline with:
  - Page streaming: process one page at a time, unload previous to cap RAM
  - WebtoonManager integration: viewport-aware loading for the active view
  - Long-image slicing: optionally slice extremely tall images into chunks
  - Progress tracking with per-page granularity

Usage (inside ModuleManager or a dedicated thread):
    pipeline = WebtoonBatchPipeline(module_manager)
    pipeline.run(proj, pages_to_process=page_keys)
    # Signals:
    pipeline.page_finished.emit(page_key, page_index, total)
    pipeline.progress_changed.emit(current, total, stage_name)
    pipeline.finished.emit()
"""

from typing import List, Optional, Dict, Callable
import copy

import numpy as np
from qtpy.QtCore import QObject, Signal, QThread

from utils.textblock import TextBlock
from utils.proj_imgtrans import ProjImgTrans
from utils.logger import logger as LOGGER
from utils.cancellation import CancelFlag
from utils.config import pcfg
from utils.image_upscale import apply_initial_upscale

from .handlers import (
    PipelineContext,
    DetectHandler,
    OCRHandler,
    TranslateHandler,
    InpaintHandler,
    LayoutHandler,
    PipelineComposer,
)


class WebtoonBatchPipeline(QThread):
    """
    Streaming batch pipeline for webtoon / long-scroll projects.

    Unlike the standard ModuleThread pipeline which holds all pages in memory,
    this pipeline:
      1. Loads one page image at a time
      2. Runs detect → OCR → translate → inpaint → layout
      3. Saves results to the project
      4. Unloads the image array to free RAM before moving to the next page
    """

    page_started = Signal(str, int, int)  # page_key, current_index, total
    page_finished = Signal(str, int, int)  # page_key, current_index, total
    progress_changed = Signal(int, int, str)  # current, total, stage_name
    stage_started = Signal(str, str)  # page_key, stage_name
    stage_finished = Signal(str, str)  # page_key, stage_name
    finished = Signal()
    error_occurred = Signal(str, str)  # page_key, error_message

    def __init__(
        self,
        textdetector=None,
        ocr=None,
        translator=None,
        inpainter=None,
        extra_textdetectors=None,
        layout_engine=None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.textdetector = textdetector
        self.ocr = ocr
        self.translator = translator
        self.inpainter = inpainter
        self.extra_textdetectors = extra_textdetectors or []
        self.layout_engine = layout_engine

        self.imgtrans_proj: Optional[ProjImgTrans] = None
        self.pages_to_process: List[str] = []
        self.cancel_flag: Optional[CancelFlag] = None
        self.stop_requested = False
        self._pause_requested = False

        # Slice very tall images into chunks of this max height (0 = disable)
        self.max_slice_height = int(getattr(pcfg, 'webtoon_max_slice_height', 0) or 0)

    def run(
        self,
        imgtrans_proj: ProjImgTrans,
        pages_to_process: Optional[List[str]] = None,
        cancel_flag: Optional[CancelFlag] = None,
    ) -> None:
        """Execute the pipeline (call via start(), not directly)."""
        self.imgtrans_proj = imgtrans_proj
        self.cancel_flag = cancel_flag
        self.stop_requested = False

        all_pages = list(imgtrans_proj.pages.keys())
        if pages_to_process:
            self.pages_to_process = [p for p in pages_to_process if p in all_pages]
        else:
            self.pages_to_process = all_pages

        total = len(self.pages_to_process)
        if total == 0:
            LOGGER.info("WebtoonBatchPipeline: no pages to process")
            self.finished.emit()
            return

        LOGGER.info("WebtoonBatchPipeline: starting %d pages", total)

        # Build pipeline composer
        pipeline = self._build_pipeline()

        for idx, page_key in enumerate(self.pages_to_process):
            if self._should_stop():
                LOGGER.info("WebtoonBatchPipeline: stopped by user at page %d", idx)
                break

            self.page_started.emit(page_key, idx + 1, total)
            self.progress_changed.emit(idx + 1, total, "page_start")

            try:
                self._process_single_page(page_key, pipeline)
            except Exception as e:
                LOGGER.error("WebtoonBatchPipeline: failed on page %s: %s", page_key, e)
                self.error_occurred.emit(page_key, str(e))

            self.page_finished.emit(page_key, idx + 1, total)
            self.progress_changed.emit(idx + 1, total, "page_finished")

        self.finished.emit()

    def _build_pipeline(self) -> PipelineComposer:
        """Assemble detect → OCR → translate → inpaint → layout handlers."""
        cfg_module = pcfg.module
        handlers = []
        if cfg_module.enable_detect and self.textdetector is not None:
            handlers.append(
                DetectHandler(
                    self.textdetector,
                    extra_textdetectors=self.extra_textdetectors,
                    merge_overlap_thres=getattr(cfg_module, "merge_overlap_thres", 0.3),
                )
            )
        if cfg_module.enable_ocr and self.ocr is not None:
            handlers.append(OCRHandler(self.ocr))
        if cfg_module.enable_translate and self.translator is not None:
            handlers.append(TranslateHandler(self.translator))
        if cfg_module.enable_inpaint and self.inpainter is not None:
            handlers.append(InpaintHandler(self.inpainter))
        if self.layout_engine is not None:
            handlers.append(LayoutHandler(self.layout_engine))
        return PipelineComposer(handlers, name="webtoon")

    def _process_single_page(self, page_key: str, pipeline: PipelineComposer) -> None:
        """Process one page: load image, run pipeline, save results, unload."""
        proj = self.imgtrans_proj
        if proj is None:
            return

        # Load image
        img = proj.read_img(page_key)
        if img is None:
            raise ValueError(f"Could not read image for page: {page_key}")

        orig_h, orig_w = img.shape[:2]
        scale = 1.0

        # Optional initial upscale
        if getattr(pcfg.module, 'image_upscale_initial', False):
            try:
                factor = float(getattr(pcfg.module, 'image_upscale_initial_factor', 2.0) or 2.0)
                policy = (getattr(pcfg.module, 'upscale_policy_initial', None) or 'lanczos').strip().lower()
                if factor > 1.0 and policy != 'none':
                    img = apply_initial_upscale(img, factor=factor, policy=policy)
                    scale = factor
            except Exception as e:
                LOGGER.warning("Webtoon initial upscale failed: %s", e)

        # Long-image slicing
        slices = self._slice_image_if_needed(img)
        all_blks: List[TextBlock] = []

        for slice_idx, slice_img in enumerate(slices):
            if self._should_stop():
                break

            ctx = PipelineContext(
                imgtrans_proj=proj,
                page_key=page_key,
                page_index=slice_idx,
                img_array=slice_img,
                cancel_flag=self.cancel_flag,
                stop_requested=self.stop_requested,
            )

            # Run pipeline on slice
            ctx = pipeline(ctx)

            # Offset block coordinates for sliced images
            if len(slices) > 1 and slice_idx > 0:
                offset_y = sum(s.shape[0] for s in slices[:slice_idx])
                for blk in ctx.block_list:
                    blk.offset(0, offset_y / scale)

            all_blks.extend(ctx.block_list)

            # Unload slice memory immediately
            del slice_img

        # Store results in project
        if all_blks:
            proj.pages[page_key] = all_blks

        # Explicitly drop image references to cap RAM before next page
        del img

    def _slice_image_if_needed(self, img: np.ndarray) -> List[np.ndarray]:
        """Slice extremely tall images into manageable vertical chunks."""
        if self.max_slice_height <= 0:
            return [img]
        h, w = img.shape[:2]
        if h <= self.max_slice_height:
            return [img]

        slices = []
        stride = self.max_slice_height
        overlap = int(getattr(pcfg, 'webtoon_slice_overlap', 200))
        start = 0
        while start < h:
            end = min(start + stride, h)
            slices.append(img[start:end, :])
            if end >= h:
                break
            start = end - overlap
        LOGGER.info("Sliced tall image (%d px) into %d chunks", h, len(slices))
        return slices

    def _should_stop(self) -> bool:
        if self.stop_requested:
            return True
        if self.cancel_flag is not None and self.cancel_flag.is_set():
            return True
        return False

    def request_stop(self) -> None:
        self.stop_requested = True
        if self.cancel_flag is not None:
            self.cancel_flag.request()

    def request_pause(self) -> None:
        self._pause_requested = True

    def request_resume(self) -> None:
        self._pause_requested = False
