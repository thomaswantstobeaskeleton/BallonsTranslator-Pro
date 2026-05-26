"""
Anime Big LaMa — TorchScript inpainting model fine-tuned on anime/manga.
Model: df1412/anime-big-lama (Hugging Face).
Requires: pip install torch huggingface-hub
"""
import os
import os.path as osp
import numpy as np
import cv2
from typing import List

from .base import InpainterBase, register_inpainter, TextBlock
from utils.imgproc_utils import resize_keepasp

from utils.logger import logger as _LOGGER

_ANIME_BIG_LAMA_AVAILABLE = False
try:
    import torch
    from huggingface_hub import hf_hub_download
    _ANIME_BIG_LAMA_AVAILABLE = True
except ImportError as _e:
    _LOGGER.warning("Anime Big LaMa dependencies missing: %s", _e)


HF_REPO_ID = "df1412/anime-big-lama"
HF_MODEL_FILENAME = "anime-manga-big-lama.pt"


@register_inpainter("anime_big_lama")
class AnimeBigLamaInpainter(InpainterBase):
    """
    Anime Big LaMa — TorchScript model for anime/manga inpainting.
    Auto-downloads from Hugging Face on first use.
    """
    inpaint_by_block = False
    check_need_inpaint = True

    download_file_list = [{
        "url": f"https://huggingface.co/{HF_REPO_ID}/resolve/main/{HF_MODEL_FILENAME}",
        "files": f"data/models/{HF_MODEL_FILENAME}",
    }]
    params = {
        "model_path": {
            "type": "line_editor",
            "value": f"data/models/{HF_MODEL_FILENAME}",
            "description": "Path to anime-manga-big-lama.pt (auto-downloaded from HF).",
        },
        "inpaint_size": {
            "type": "selector",
            "options": [512, 768, 1024, 1536, 2048],
            "value": 1024,
            "description": "Max side for inference.",
        },
        "description": "Anime Big LaMa (df1412/anime-big-lama). TorchScript model fine-tuned for anime/manga inpainting.",
    }
    _load_model_keys = {"model"}

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.model = None

    def _load_model(self):
        if not _ANIME_BIG_LAMA_AVAILABLE:
            raise RuntimeError(
                "Anime Big LaMa requires torch and huggingface-hub. "
                "Install: pip install torch huggingface-hub"
            )
        path = (self.params.get("model_path") or {}).get("value", f"data/models/{HF_MODEL_FILENAME}")
        if isinstance(path, str):
            path = path.strip()
        if not path or not osp.isfile(path):
            _LOGGER.info("Anime Big LaMa: downloading model from %s...", HF_REPO_ID)
            try:
                path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_MODEL_FILENAME)
            except Exception as e:
                raise RuntimeError(
                    f"Could not download Anime Big LaMa model: {e}"
                ) from e
        _LOGGER.info("Anime Big LaMa: loading model from %s", path)
        self.model = torch.jit.load(path, map_location="cpu")
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        _LOGGER.info("Anime Big LaMa: model loaded")

    def _inpaint(self, img: np.ndarray, mask: np.ndarray, textblock_list: List[TextBlock] = None) -> np.ndarray:
        if self.model is None:
            return img.copy()
        im_h, im_w = img.shape[:2]
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask_bin = (mask > 127).astype(np.float32)

        inpaint_size = int((self.params.get("inpaint_size") or {}).get("value", 1024))
        max_side = max(im_h, im_w)
        scale = 1.0
        if max_side > inpaint_size:
            scale = inpaint_size / max_side
            new_w = int(im_w * scale)
            new_h = int(im_h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            mask_bin = cv2.resize(mask_bin, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        h, w = img.shape[:2]
        img_f = img.astype(np.float32) / 255.0
        img_t = torch.from_numpy(np.transpose(img_f, (2, 0, 1))).unsqueeze(0)
        mask_t = torch.from_numpy(mask_bin).unsqueeze(0).unsqueeze(0)

        if torch.cuda.is_available() and next(self.model.parameters()).is_cuda:
            img_t = img_t.cuda()
            mask_t = mask_t.cuda()

        with torch.no_grad():
            result = self.model(img_t, mask_t)

        result_np = result.squeeze(0).cpu().numpy()
        result_np = np.transpose(result_np, (1, 2, 0))
        result_np = np.clip(result_np * 255.0, 0, 255).astype(np.uint8)

        if result_np.shape[:2] != (im_h, im_w):
            result_np = cv2.resize(result_np, (im_w, im_h), interpolation=cv2.INTER_LINEAR)

        mask_3 = (mask > 127).astype(np.float32)[:, :, np.newaxis]
        return (result_np.astype(np.float32) * mask_3 + img.astype(np.float32) * (1 - mask_3)).astype(np.uint8)
