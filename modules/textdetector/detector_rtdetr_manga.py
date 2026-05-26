"""
RT-DETR Manga109s detector — tori29umai/rtdetrv4-x-manga109s for comic/manga text detection.
Requires: pip install transformers torch
Model: https://huggingface.co/tori29umai/rtdetrv4-x-manga109s
"""
import numpy as np
import cv2
from typing import Tuple, List

from .base import register_textdetectors, TextDetectorBase, TextBlock, DEVICE_SELECTOR
from .box_utils import expand_blocks
from utils.textblock import mit_merge_textlines, sort_regions, examine_textblk, sort_pnts
from utils.imgproc_utils import xywh2xyxypoly

from utils.logger import logger as _LOGGER

_RT_DETR_AVAILABLE = False
try:
    from transformers import AutoModelForObjectDetection, AutoImageProcessor
    import torch
    _RT_DETR_AVAILABLE = True
except ImportError as _e:
    _LOGGER.warning("RT-DETR detector dependencies missing: %s", _e)


HF_REPO_ID = "tori29umai/rtdetrv4-x-manga109s"


@register_textdetectors("rtdetr_manga")
class RTDetrMangaDetector(TextDetectorBase):
    """
    RT-DETRv4 fine-tuned on Manga109s for robust comic/manga text and bubble detection.
    Auto-downloads model from Hugging Face on first use via transformers caching.
    """
    params = {
        "model_id": {
            "type": "line_editor",
            "value": HF_REPO_ID,
            "description": "Hugging Face model id. Leave default for Manga109s fine-tuned RT-DETR.",
        },
        "confidence threshold": {
            "display_name": "Confidence threshold",
            "type": "line_editor",
            "value": 0.3,
        },
        "IoU threshold": {
            "display_name": "IoU threshold",
            "type": "line_editor",
            "value": 0.5,
        },
        "device": {**DEVICE_SELECTOR(), "display_name": "Device"},
        "font size multiplier": {
            "display_name": "Font size multiplier",
            "type": "line_editor",
            "value": 1.0,
        },
        "font size max": {
            "display_name": "Max font size",
            "type": "line_editor",
            "value": -1,
        },
        "font size min": {
            "display_name": "Min font size",
            "type": "line_editor",
            "value": -1,
        },
        "merge text lines": {
            "display_name": "Merge text lines", "type": "checkbox", "value": True
        },
        "box_padding": {
            "type": "line_editor",
            "value": 5,
            "display_name": "Box padding",
            "description": "Pixels to add around each detected box (all sides). Reduces clipped punctuation.",
        },
        "description": "RT-DETRv4 Manga109s (tori29umai/rtdetrv4-x-manga109s). Auto-downloads from Hugging Face. Fine-tuned for manga text/bubble detection.",
    }

    download_file_on_load = True
    _load_model_keys = {"model", "processor"}

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.model = None
        self.processor = None

    @classmethod
    def is_environment_compatible(cls) -> bool:
        try:
            import transformers
            from packaging import version
            return version.parse(transformers.__version__) >= version.parse("4.39.0")
        except Exception:
            return False

    def _load_model(self):
        if not _RT_DETR_AVAILABLE:
            raise RuntimeError(
                "RT-DETR detector requires transformers and torch. "
                "Install: pip install transformers torch"
            )
        model_id = self.get_param_value("model_id") or HF_REPO_ID
        _LOGGER.info("RT-DETR: loading model '%s'...", model_id)
        device = self.get_param_value("device") or "cpu"
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForObjectDetection.from_pretrained(model_id)
        self.model.to(device)
        self.model.eval()
        _LOGGER.info("RT-DETR: model loaded on %s", device)

    def _detect(self, img: np.ndarray, proj=None) -> Tuple[np.ndarray, List[TextBlock]]:
        if not _RT_DETR_AVAILABLE:
            raise RuntimeError("transformers / torch not installed.")

        im_h, im_w = img.shape[:2]
        device = self.get_param_value("device") or "cpu"
        conf_thr = float(self.get_param_value("confidence threshold"))

        inputs = self.processor(images=img, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([[im_h, im_w]])
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=conf_thr
        )[0]

        mask = np.zeros((im_h, im_w), dtype=np.uint8)
        detected_items = []

        scores = results["scores"].cpu().numpy()
        labels = results["labels"].cpu().numpy()
        boxes = results["boxes"].cpu().numpy()

        for score, label, box in zip(scores, labels, boxes):
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(im_w, x2), min(im_h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
            pts = xywh2xyxypoly(np.array([[x1, y1, x2 - x1, y2 - y1]])).reshape(4, 2).tolist()
            detected_items.append({"pts": pts, "label": str(label)})

        blk_list = []
        if self.get_param_value("merge text lines"):
            pts_only_list = [item["pts"] for item in detected_items]
            blk_list = mit_merge_textlines(pts_only_list, width=im_w, height=im_h)
        else:
            for item in detected_items:
                pts_sorted, is_vertical = sort_pnts(item["pts"])
                blk = TextBlock(lines=[pts_sorted], src_is_vertical=is_vertical, label=item["label"])
                blk.vertical = is_vertical
                blk.adjust_bbox()
                examine_textblk(blk, im_w, im_h)
                blk_list.append(blk)

        blk_list = sort_regions(blk_list)

        fnt_rsz = self.get_param_value("font size multiplier")
        fnt_max = self.get_param_value("font size max")
        fnt_min = self.get_param_value("font size min")
        for blk in blk_list:
            sz = blk._detected_font_size * fnt_rsz
            if fnt_max > 0:
                sz = min(fnt_max, sz)
            if fnt_min > 0:
                sz = max(fnt_min, sz)
            blk.font_size = sz
            blk._detected_font_size = sz

        pad_val = 0
        bp = self.params.get("box_padding", {})
        if isinstance(bp, dict):
            v = bp.get("value", 5)
            try:
                pad_val = max(0, min(24, int(v) if v not in (None, "") else 5))
            except (TypeError, ValueError):
                pad_val = 5
        if pad_val > 0:
            blk_list = expand_blocks(blk_list, pad_val, im_w, im_h)

        return mask, blk_list

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        if param_key in ("model_id", "device") and hasattr(self, "model") and self.model is not None:
            del self.model
            self.model = None
            del self.processor
            self.processor = None
