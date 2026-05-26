"""
AnimeText_yolo detector — deepghs/AnimeText_yolo (YOLO12) for robust anime/manga scene text detection.
Requires:  pip install ultralytics huggingface-hub
Model:    https://huggingface.co/deepghs/AnimeText_yolo
"""
import os
import os.path as osp
from typing import Tuple, List

import numpy as np
import cv2

from .base import register_textdetectors, TextDetectorBase, TextBlock, DEVICE_SELECTOR
from .box_utils import expand_blocks
from utils.textblock import mit_merge_textlines, sort_regions, examine_textblk, sort_pnts
from utils.imgproc_utils import xywh2xyxypoly


from utils.logger import logger as _LOGGER

_ANIME_TEXT_YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO
    from huggingface_hub import hf_hub_download, list_repo_files
    _ANIME_TEXT_YOLO_AVAILABLE = True
except ImportError as _e:
    _LOGGER.warning("AnimeText YOLO dependencies missing: %s", _e)


HF_REPO_ID = "deepghs/AnimeText_yolo"


def _find_yolo_model_file() -> str:
    """Return path to the best YOLO model file from the HF repo, downloading if needed."""
    try:
        files = list_repo_files(HF_REPO_ID, repo_type="model")
        # Prefer .pt, then .safetensors
        candidates = [f for f in files if f.endswith(".pt") or f.endswith(".safetensors")]
        if not candidates:
            _LOGGER.warning("AnimeText YOLO: no .pt/.safetensors files found in repo %s", HF_REPO_ID)
            return None
        # Heuristic: pick the largest / most specific name (often best)
        candidates.sort(key=lambda x: ("best" in x.lower(), "last" not in x.lower(), len(x)), reverse=True)
        chosen = candidates[0]
        _LOGGER.info("AnimeText YOLO: downloading model file '%s' from %s", chosen, HF_REPO_ID)
        return hf_hub_download(repo_id=HF_REPO_ID, filename=chosen, repo_type="model")
    except ImportError as e:
        _LOGGER.error("AnimeText YOLO: huggingface_hub not available: %s", e)
        return None
    except Exception as e:
        _LOGGER.error("AnimeText YOLO: failed to download model from Hugging Face: %s", e)
        return None


@register_textdetectors("animetext_yolo")
class AnimeTextYoloDetector(TextDetectorBase):
    """
    AnimeText YOLO12 — trained on the AnimeText dataset for robust complex anime scene text detection.
    Auto-downloads model weights from Hugging Face (deepghs/AnimeText_yolo) on first use.
    """
    params = {
        "model path": {
            "type": "line_editor",
            "value": "",
            "description": "Path to a local .pt/.safetensors YOLO model. Leave empty to auto-download from HF.",
        },
        "confidence threshold": {
            "display_name": "Confidence threshold",
            "type": "line_editor",
            "value": 0.25,
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
        "description": "AnimeText YOLO12 (deepghs/AnimeText_yolo). Auto-downloads from Hugging Face. Best for anime/manga scenes with complex text.",
    }

    download_file_on_load = True
    _load_model_keys = {"model"}

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self._model_path = None

    def _load_model(self):
        if not _ANIME_TEXT_YOLO_AVAILABLE:
            raise RuntimeError(
                "AnimeText YOLO requires ultralytics and huggingface-hub. "
                "Install: pip install ultralytics huggingface-hub"
            )
        model_path = self.get_param_value("model path")
        if model_path and osp.exists(model_path):
            self._model_path = model_path
        else:
            cached = _find_yolo_model_file()
            if not cached:
                raise RuntimeError(
                    "Could not download AnimeText YOLO model from Hugging Face. "
                    "Check internet connection or set 'model path' to a local .pt file."
                )
            self._model_path = cached
        self.model = YOLO(self._model_path)
        device = self.get_param_value("device")
        if device:
            self.model.to(device)

    def _detect(self, img: np.ndarray, proj=None) -> Tuple[np.ndarray, List[TextBlock]]:
        if not _ANIME_TEXT_YOLO_AVAILABLE:
            raise RuntimeError("ultralytics / huggingface-hub not installed.")

        conf = float(self.get_param_value("confidence threshold"))
        iou = float(self.get_param_value("IoU threshold"))

        result = self.model.predict(
            source=img,
            save=False,
            show=False,
            verbose=False,
            conf=conf,
            iou=iou,
            agnostic_nms=True,
        )[0]

        im_h, im_w = img.shape[:2]
        mask = np.zeros_like(img[..., 0])
        detected_items = []

        # Standard boxes
        dets = result.boxes
        if dets is not None and len(dets.cls) > 0:
            for i in range(len(dets.cls)):
                xyxy = dets.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = xyxy.astype(int)
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                pts = xywh2xyxypoly(np.array([[x1, y1, x2 - x1, y2 - y1]])).reshape(4, 2).tolist()
                detected_items.append({"pts": pts, "label": "text"})

        # Oriented boxes
        dets = result.obb
        if dets is not None and len(dets.cls) > 0:
            for i in range(len(dets.cls)):
                pts = dets.xyxyxyxy[i].cpu().numpy().astype(int)
                cv2.fillPoly(mask, [pts], 255)
                detected_items.append({"pts": pts.tolist(), "label": "text"})

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
        if param_key == "model path" and hasattr(self, "model"):
            del self.model
