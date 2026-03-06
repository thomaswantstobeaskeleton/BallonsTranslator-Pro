"""
Batch translation script (#1094).
Runs detect -> OCR -> translate -> inpaint on one or more project folders using the same
pipeline as the GUI (config from config/config.json). Saves project after each page.

Usage:
  python scripts/batch_translate.py [OPTIONS] [DIRS...]
  DIRS: one or more project directories. If omitted, --dir is required.

Examples:
  python scripts/batch_translate.py --dir ./my_chapter
  python scripts/batch_translate.py --dir ./ch1 ./ch2 --no-inpaint
"""
from __future__ import annotations

import argparse
import os
import sys
import os.path as osp
from pathlib import Path
from typing import List, Optional
import logging

# Run from project root
PATH_ROOT = Path(__file__).resolve().parent.parent
if str(PATH_ROOT) not in sys.path:
    sys.path.insert(0, str(PATH_ROOT))
os.chdir(PATH_ROOT)

import utils.shared as shared
shared.PROGRAM_PATH = str(PATH_ROOT)

from utils.config import load_config, pcfg, RunStatus
from utils.proj_imgtrans import ProjImgTrans
from utils.textblock import examine_textblk, remove_contained_boxes, deduplicate_primary_boxes
from utils.logger import logger as LOGGER
from modules import (
    init_module_registries,
    TEXTDETECTORS,
    OCR,
    TRANSLATORS,
    INPAINTERS,
    BaseTranslator,
    InpainterBase,
    OCRBase,
    TextDetectorBase,
)
from modules.base import GPUINTENSIVE_SET, soft_empty_cache
from modules.inpaint.base import build_mask_with_resolved_overlaps

try:
    from utils.image_upscale import apply_initial_upscale, downscale_to_size
except ImportError:
    apply_initial_upscale = None
    def downscale_to_size(img, w, h):
        return img

def _get_module(module_type: str, name: str):
    if module_type == "textdetector":
        reg, cfg = TEXTDETECTORS, pcfg.module.textdetector_params
    elif module_type == "ocr":
        reg, cfg = OCR, pcfg.module.ocr_params
    elif module_type == "translator":
        reg, cfg = TRANSLATORS, pcfg.module.translator_params
    elif module_type == "inpainter":
        reg, cfg = INPAINTERS, pcfg.module.inpainter_params
    else:
        raise ValueError(module_type)
    if name not in reg.module_dict:
        raise RuntimeError(f"Module {module_type}/{name} not found.")
    params = cfg.get(name) or {}
    if params:
        return reg.module_dict[name](**params)
    return reg.module_dict[name]()


def run_batch_pipeline(
    proj: ProjImgTrans,
    detector: Optional[TextDetectorBase],
    ocr: Optional[OCRBase],
    translator: Optional[BaseTranslator],
    inpainter: Optional[InpainterBase],
    enable_detect: bool,
    enable_ocr: bool,
    enable_translate: bool,
    enable_inpaint: bool,
    log: Optional[logging.Logger] = None,
) -> None:
    log = log or LOGGER
    cfg = pcfg.module
    pages_to_iterate = list(proj.pages.keys())

    for page_num, imgname in enumerate(pages_to_iterate, 1):
        log.info("Page %d/%d: %s", page_num, len(pages_to_iterate), imgname)
        try:
            img = proj.read_img(imgname)
        except Exception as e:
            log.error("Failed to read image %s: %s", imgname, e)
            continue
        orig_h, orig_w = img.shape[:2]
        scale = 1.0
        if getattr(cfg, "image_upscale_initial", False) and apply_initial_upscale:
            try:
                factor = float(getattr(cfg, "image_upscale_initial_factor", 2.0) or 2.0)
                policy = (getattr(cfg, "upscale_policy_initial", None) or "lanczos").strip().lower()
                if factor > 1.0 and policy != "none":
                    img = apply_initial_upscale(img, factor=factor, policy=policy)
                    scale = factor
            except Exception as e:
                log.warning("Initial upscale failed: %s", e)

        mask = None
        blk_list = []

        if enable_detect and detector:
            try:
                mask, blk_list = detector.detect(img, proj)
                im_h, im_w = img.shape[:2]
                for blk in blk_list:
                    if getattr(blk, "lines", None) and len(blk.lines) > 0:
                        examine_textblk(blk, im_w, im_h, sort=True)
                blk_list = remove_contained_boxes(blk_list)
                blk_list = deduplicate_primary_boxes(blk_list, iou_threshold=0.5)
                proj.pages[imgname] = blk_list
                proj.update_page_progress(imgname, RunStatus.FIN_DET)
            except Exception as e:
                log.error("Detection failed for %s: %s", imgname, e, exc_info=True)
                blk_list = proj.pages.get(imgname, [])

        if blk_list is None:
            blk_list = proj.pages.get(imgname, [])

        if enable_ocr and ocr and blk_list:
            try:
                if getattr(ocr, "restore_to_device", None):
                    ocr.restore_to_device()
                ocr.run_ocr(img, blk_list)
                proj.update_page_progress(imgname, RunStatus.FIN_OCR)
                if enable_inpaint and getattr(ocr, "device", None) in GPUINTENSIVE_SET and getattr(ocr, "offload_to_cpu", None):
                    ocr.offload_to_cpu()
                    soft_empty_cache()
            except Exception as e:
                log.error("OCR failed for %s: %s", imgname, e, exc_info=True)

            im_h, im_w = img.shape[:2]
            if mask is None:
                mask = build_mask_with_resolved_overlaps(blk_list, im_w, im_h)
            if mask is not None:
                mask_to_save = downscale_to_size(mask, orig_w, orig_h) if scale > 1 else mask
                proj.save_mask(imgname, mask_to_save)

        if enable_translate and translator and blk_list:
            try:
                if hasattr(translator, "set_translation_context"):
                    translator.set_translation_context(
                        previous_pages=[],
                        project_glossary=proj.translation_glossary,
                        series_context_path=proj.series_context_path or "",
                    )
                setattr(translator, "_current_page_key", imgname)
                translator.translate_textblk_lst(blk_list)
                if hasattr(translator, "append_page_to_series_context") and getattr(proj, "series_context_path", None):
                    src = [b.get_text() for b in blk_list]
                    trans = [getattr(b, "translation", "") or "" for b in blk_list]
                    translator.append_page_to_series_context(proj.series_context_path, src, trans)
                proj.update_page_progress(imgname, RunStatus.FIN_TRANSLATE)
            except Exception as e:
                log.warning("Translation failed for %s: %s (continuing)", imgname, e)
                for blk in blk_list:
                    if getattr(blk, "translation", None) is None or str(blk.translation).strip() == "":
                        blk.translation = "[Translation failed]"

        if enable_inpaint and inpainter and mask is not None:
            try:
                if getattr(inpainter, "restore_to_device", None):
                    inpainter.restore_to_device()
                inpainted = inpainter.inpaint(img, mask)
                inpainted_to_save = downscale_to_size(inpainted, orig_w, orig_h) if scale > 1 else inpainted
                proj.save_inpainted(imgname, inpainted_to_save)
                proj.update_page_progress(imgname, RunStatus.FIN_INPAINT)
            except Exception as e:
                log.error("Inpaint failed for %s: %s", imgname, e, exc_info=True)

        proj.save()
    log.info("Batch finished: %s", proj.directory)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch detect/OCR/translate/inpaint on project folder(s).")
    parser.add_argument("dirs", nargs="*", help="Project directory (folder of images).")
    parser.add_argument("--dir", "-d", action="append", default=[], dest="dirs_opt", help="Add project directory.")
    parser.add_argument("--config", "-c", default=osp.join(shared.PROGRAM_PATH, "config", "config.json"), help="Config JSON path.")
    parser.add_argument("--no-detect", action="store_true", help="Skip text detection.")
    parser.add_argument("--no-ocr", action="store_true", help="Skip OCR.")
    parser.add_argument("--no-translate", action="store_true", help="Skip translation.")
    parser.add_argument("--no-inpaint", action="store_true", help="Skip inpainting.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging.")
    args = parser.parse_args()

    dirs = list(args.dirs) + list(args.dirs_opt or [])
    if not dirs:
        parser.error("Provide at least one project directory (positional or --dir).")

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not osp.exists(args.config):
        LOGGER.error("Config not found: %s", args.config)
        sys.exit(1)

    load_config(args.config)
    init_module_registries()

    cfg = pcfg.module
    enable_detect = cfg.enable_detect and not args.no_detect
    enable_ocr = cfg.enable_ocr and not args.no_ocr
    enable_translate = cfg.enable_translate and not args.no_translate
    enable_inpaint = cfg.enable_inpaint and not args.no_inpaint

    detector = _get_module("textdetector", cfg.textdetector) if enable_detect else None
    ocr = _get_module("ocr", cfg.ocr) if enable_ocr else None
    translator = _get_module("translator", cfg.translator) if enable_translate else None
    inpainter = _get_module("inpainter", cfg.inpainter) if enable_inpaint else None

    if not getattr(cfg, "load_model_on_demand", True):
        for m in (detector, ocr, inpainter):
            if m and hasattr(m, "load_model"):
                m.load_model()

    for d in dirs:
        d = osp.abspath(d)
        if not osp.isdir(d):
            LOGGER.warning("Not a directory: %s", d)
            continue
        LOGGER.info("Processing project: %s", d)
        proj = ProjImgTrans(d)
        run_batch_pipeline(
            proj,
            detector=detector,
            ocr=ocr,
            translator=translator,
            inpainter=inpainter,
            enable_detect=enable_detect,
            enable_ocr=enable_ocr,
            enable_translate=enable_translate,
            enable_inpaint=enable_inpaint,
        )


if __name__ == "__main__":
    main()
