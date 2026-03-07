"""
Surya OCR – multilingual recognition (90+ languages).
Uses pre-cropped regions from the text detector; good for Chinese, English, and mixed script.
Requires: pip install surya-ocr  (Python 3.10+, PyTorch).
Supports current surya-ocr API (RecognitionPredictor + FoundationPredictor).
"""
from typing import List
import re
import numpy as np
import cv2
from .base import OCRBase, register_OCR, DEFAULT_DEVICE, DEVICE_SELECTOR, TextBlock
from utils.io_utils import trim_ocr_repetition

# Unicode replacement character and common OCR garbage that renders as empty squares in many fonts
_REPLACEMENT_CHAR = "\uFFFD"
_VISIBLE_PLACEHOLDER = "\u25A1"  # □ (white square) - visible in CJK fonts


def _normalize_ocr_text(text: str, chinese_only: bool = False) -> str:
    """Replace replacement chars so they don't render as empty; optionally strip stray katakana when Chinese-only."""
    if not text:
        return text
    if _REPLACEMENT_CHAR in text:
        text = text.replace(_REPLACEMENT_CHAR, _VISIBLE_PLACEHOLDER)
    if chinese_only:
        # Strip stray katakana that OCR sometimes outputs for garbled/cut-off Chinese (e.g. "山クト" for "3")
        text = re.sub(r"[\u30A0-\u30FF]+", "", text)  # katakana
        text = re.sub(r"\s+", " ", text).strip()
    return text

_SURYA_AVAILABLE = False
_surya_predictor = None
_surya_task_name = None

try:
    from surya.recognition import RecognitionPredictor
    from surya.foundation import FoundationPredictor
    from surya.common.surya.schema import TaskNames
    from PIL import Image
    _SURYA_AVAILABLE = True
    _surya_predictor = RecognitionPredictor
    _surya_task_name = TaskNames.ocr_with_boxes
    # Workaround: some transformers/surya-ocr versions leave SuryaDecoderConfig without pad_token_id
    try:
        from surya.common.surya.decoder import SuryaDecoderConfig
        _orig_getattr = getattr(SuryaDecoderConfig, "__getattribute__", object.__getattribute__)
        def _patched_getattr(self, name):
            if name == "pad_token_id":
                try:
                    return _orig_getattr(self, name)
                except AttributeError:
                    return 0
            return _orig_getattr(self, name)
        SuryaDecoderConfig.__getattribute__ = _patched_getattr
    except Exception:
        pass
    # Workaround: model config may have rope_type='default' but ROPE_INIT_FUNCTIONS expects keys that
    # use transformers' rope_parameters (with "factor"), which SuryaDecoderConfig doesn't have. Provide
    # a simple default RoPE init using only config.rope_theta.
    # Note: surya.common.surya.decoder imports ROPE_INIT_FUNCTIONS from transformers.modeling_rope_utils,
    # so this patches the shared dict. Other OCRs (e.g. PaddleOCRVLManga) call rope_fn(config) with one arg;
    # accept optional device so both rope_fn(config, device) and rope_fn(config) work.
    try:
        import torch
        import surya.common.surya.decoder as _surya_dec

        def _default_rope_init(config, device=None):
            """Basic RoPE (no scaling factor) for configs with rope_type='default' and no rope_parameters."""
            if device is None or (hasattr(device, "type") and device.type == "meta"):
                device = torch.device("cpu")
            base = getattr(config, "rope_theta", 10000.0)
            head_dim = getattr(config, "head_dim", None) or (
                config.hidden_size // config.num_attention_heads
            )
            dim = int(head_dim * getattr(config, "partial_rotary_factor", 1.0))
            # Create on CPU to avoid "Cannot copy out of meta tensor" when model uses meta device for lazy init
            create_device = torch.device("cpu")
            inv_freq = 1.0 / (
                base
                ** (
                    torch.arange(0, dim, 2, dtype=torch.int64, device=create_device)
                    .float()
                    / dim
                )
            )
            inv_freq = inv_freq.to(device)
            return inv_freq, 1.0

        if "default" not in _surya_dec.ROPE_INIT_FUNCTIONS:
            _surya_dec.ROPE_INIT_FUNCTIONS["default"] = _default_rope_init
    except Exception:
        pass
    # Workaround: newer transformers expect model.all_tied_weights_keys (dict); SuryaModel only has _tied_weights_keys
    # (list) and may not set all_tied_weights_keys, causing AttributeError in _finalize_model_loading.
    try:
        from surya.common.surya import SuryaModel as _SuryaModel
        _orig_surya_init = _SuryaModel.__init__

        def _surya_init_patch(self, *args, **kwargs):
            _orig_surya_init(self, *args, **kwargs)
            if not hasattr(self, "all_tied_weights_keys"):
                self.all_tied_weights_keys = {}

        _SuryaModel.__init__ = _surya_init_patch
        # Newer transformers removed _tie_or_clone_weights from PreTrainedModel; SuryaModel._tie_weights uses it.
        if not hasattr(_SuryaModel, "_tie_or_clone_weights") or not callable(
            getattr(_SuryaModel, "_tie_or_clone_weights", None)
        ):
            import torch as _torch

            def _tie_or_clone_weights(module, linear_layer, embed_layer):
                target_device = embed_layer.weight.device
                linear_layer.weight = embed_layer.weight
                if getattr(linear_layer, "bias", None) is not None:
                    if linear_layer.bias.device.type == "meta" and target_device.type != "meta":
                        linear_layer.bias.data = linear_layer.bias.data.to(target_device)
                    linear_layer.bias.data.zero_()
                # Move any other params still on meta (e.g. after from_pretrained) — skip weight, we just tied it
                for name, param in linear_layer.named_parameters(recurse=False):
                    if name != "weight" and param.device.type == "meta" and target_device.type != "meta":
                        setattr(linear_layer, name, _torch.nn.Parameter(param.data.to(target_device)))

            _SuryaModel._tie_or_clone_weights = _tie_or_clone_weights
        # Newer transformers call tie_weights(missing_keys=..., recompute_mapping=False); SuryaModel.tie_weights() has no args.
        _orig_tie = _SuryaModel.tie_weights
        _SuryaModel.tie_weights = lambda self, missing_keys=None, recompute_mapping=True, **kw: _orig_tie(self)
    except Exception:
        pass
except ImportError as e:
    import logging
    logging.getLogger("BallonTranslator").debug(
        f"Surya OCR not available: {e}. Install with: pip install surya-ocr (Python 3.10+, PyTorch)."
    )


def _cv2_to_pil_rgb(img: np.ndarray) -> "Image.Image":
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)


def _materialize_meta_tensors(module, device):
    """Move any parameters/buffers on meta device to the target device (transformers/surya compatibility).
    Meta tensors cannot be .to(device)'d; we use empty_like to materialize when needed.
    Walks the full tree by full name so we hit every parameter/buffer that forward() uses.
    """
    import torch
    dev = torch.device(device) if isinstance(device, str) else device
    if dev.type == "meta":
        return

    def _set_param_or_buffer(parent, leaf_name, tensor, is_param):
        if is_param:
            try:
                with torch.no_grad():
                    p = getattr(parent, leaf_name)
                    if p is not None and p.device.type == "meta":
                        p.data = p.data.to(dev)
            except (NotImplementedError, RuntimeError, AttributeError):
                new_t = torch.empty_like(tensor, device=dev, dtype=tensor.dtype)
                param = torch.nn.Parameter(new_t)
                if hasattr(parent, "_parameters") and leaf_name in parent._parameters:
                    parent._parameters[leaf_name] = param
                else:
                    setattr(parent, leaf_name, param)
        else:
            new_t = torch.empty_like(tensor, device=dev, dtype=tensor.dtype)
            if hasattr(parent, "_buffers") and leaf_name in parent._buffers:
                parent._buffers[leaf_name] = new_t
            else:
                parent.register_buffer(leaf_name, new_t)

    for name, param in list(module.named_parameters()):
        if param.device.type != "meta":
            continue
        if "." in name:
            parent = module.get_submodule(name.rsplit(".", 1)[0])
            leaf = name.rsplit(".", 1)[1]
        else:
            parent, leaf = module, name
        _set_param_or_buffer(parent, leaf, param, is_param=True)

    for name, buf in list(module.named_buffers()):
        if buf.device.type != "meta":
            continue
        if "." in name:
            parent = module.get_submodule(name.rsplit(".", 1)[0])
            leaf = name.rsplit(".", 1)[1]
        else:
            parent, leaf = module, name
        _set_param_or_buffer(parent, leaf, buf, is_param=False)


if _SURYA_AVAILABLE:
    # Surya OCR disabled: recognition fails with "Tensor on device meta" with current surya-ocr/checkpoint.
    # Use surya_det for text detection only; pair with another OCR (e.g. rapidocr, mit48px, paddle_rec_v5).
    # class SuryaOCR left unregistered so it does not appear in OCR options.
    class _SuryaOCRUnused(OCRBase):
        """
        Surya OCR: 90+ languages, line-level recognition.
        Best for: Chinese, English, multilingual manhua/comics when you want an alternative to mit48px or PaddleOCR.
        """
        lang_map = {
            "Chinese (Simplified)": ["zh"],
            "Chinese (Traditional)": ["zh"],
            "Chinese + English": ["zh", "en"],
            "English": ["en"],
            "Japanese": ["ja"],
            "Korean": ["ko"],
            "Multilingual (zh, en, ja, ko)": ["zh", "en", "ja", "ko"],
        }
        # Common Latin misrecognitions when the actual script is Chinese (model has no language hint in API)
        _LATIN_TO_CHINESE_FIXES = {"Wg": "王", "Wo": "我", "Ol": "了", "On": "们", "Og": "公"}
        params = {
            "language": {
                "type": "selector",
                "options": list(lang_map.keys()),
                "value": "Chinese + English",
                "description": "Language for display; set to Chinese (Simplified) for Chinese-only to reduce Latin misreads (e.g. 'Wg'→王).",
            },
            "fix_latin_misread": {
                "type": "checkbox",
                "value": True,
                "description": "When language is Chinese-only, fix common Latin misrecognitions (e.g. Wg→王).",
            },
            "device": DEVICE_SELECTOR(),
            "batch_size": {
                "value": 16,
                "description": "Batch size for recognition (reduce if OOM).",
            },
            "crop_padding": {
                "type": "line_editor",
                "value": 6,
                "description": "Pixels to add around each box when cropping for OCR (0–24). Reduces clipped text at edges (e.g. with CTD).",
            },
            "description": "Surya OCR – 90+ languages (pip install surya-ocr)",
        }
        _load_model_keys = {"_recognizer"}

        def __init__(self, **params) -> None:
            super().__init__(**params)
            self.device = self.params["device"]["value"]
            self._recognizer = None

        def _load_model(self):
            if self._recognizer is not None:
                return
            try:
                import torch
                foundation = FoundationPredictor(device=self.device)
                _materialize_meta_tensors(foundation.model, self.device)
                foundation.model.to(self.device)
                _dev = foundation.model.device
                # Recreate tensors that may have been created on meta when model had meta params
                if hasattr(foundation, "device_pad_token") and foundation.device_pad_token.device.type == "meta":
                    foundation.device_pad_token = torch.tensor(
                        foundation.processor.pad_token_id, device=_dev, dtype=torch.long
                    )
                if hasattr(foundation, "device_beacon_token") and foundation.device_beacon_token.device.type == "meta":
                    foundation.device_beacon_token = torch.tensor(
                        foundation.processor.beacon_token_id, device=_dev, dtype=torch.long
                    )
                if hasattr(foundation, "special_token_ids") and foundation.special_token_ids.device.type == "meta":
                    foundation.special_token_ids = torch.tensor(
                        [foundation.model.config.image_token_id] + foundation.model.config.register_token_ids,
                        device=_dev,
                    )
                self._recognizer = _surya_predictor(foundation)
            except Exception as e:
                raise RuntimeError(
                    f"Surya OCR failed to load model: {e}. "
                    "Ensure surya-ocr is installed: pip install surya-ocr (Python 3.10+, PyTorch)."
                ) from e

        def _ocr_blk_list(self, img: np.ndarray, blk_list: List[TextBlock], *args, **kwargs) -> None:
            im_h, im_w = img.shape[:2]
            pad = max(0, int(self.params.get("crop_padding", {}).get("value", 6)))
            images_pil = []
            indices = []
            for idx, blk in enumerate(blk_list):
                x1, y1, x2, y2 = blk.xyxy
                if not (0 <= x1 < x2 <= im_w and 0 <= y1 < y2 <= im_h):
                    blk.text = [""]
                    continue
                # Pad crop so text at box edges (e.g. last line) is not clipped; helps when CTD bbox is tight
                x1a = max(0, x1 - pad)
                y1a = max(0, y1 - pad)
                x2a = min(im_w, x2 + pad)
                y2a = min(im_h, y2 + pad)
                # Vertical text is often clipped at bottom; add extra padding there
                if getattr(blk, "src_is_vertical", False):
                    extra_bottom = max(pad, min(int((y2 - y1) * 0.12), 24))
                    y2a = min(im_h, y2a + extra_bottom)
                crop = img[y1a:y2a, x1a:x2a]
                if crop.size == 0:
                    blk.text = [""]
                    continue
                images_pil.append(_cv2_to_pil_rgb(crop))
                indices.append(idx)

            if not images_pil:
                return
            batch_size = max(1, int(self.params.get("batch_size", {}).get("value", 16)))
            # One bbox per crop: full image so the recognizer processes the whole crop as one line.
            bboxes = [[[0, 0, im.size[0], im.size[1]]] for im in images_pil]
            task_names = [_surya_task_name] * len(images_pil)

            # Ensure no tensors are still on meta (lazy init or late-loaded submodules)
            model = getattr(self._recognizer, "model", None) or getattr(
                getattr(self._recognizer, "foundation_predictor", None), "model", None
            )
            if model is not None:
                _materialize_meta_tensors(model, self.device)

            try:
                results = self._recognizer(
                    images_pil,
                    task_names=task_names,
                    bboxes=bboxes,
                    recognition_batch_size=batch_size,
                    drop_repeated_text=True,
                )
            except Exception as e:
                self.logger.error(f"Surya recognition error: {e}")
                for idx in indices:
                    blk_list[idx].text = [""]
                return

            lang_opt = self.params.get("language", {}).get("value", "Chinese + English")
            fix_latin = self.params.get("fix_latin_misread", {}).get("value", True)
            chinese_only = lang_opt in ("Chinese (Simplified)", "Chinese (Traditional)")

            for idx, result in zip(indices, results):
                parts = []
                if result.text_lines:
                    for line in result.text_lines:
                        t = (line.text or "").strip()
                        if t:
                            parts.append(t)
                text = "\n".join(parts) if parts else ""
                text = trim_ocr_repetition(text)
                text = _normalize_ocr_text(text, chinese_only=chinese_only)
                if chinese_only and fix_latin and text and text.isascii() and len(text) <= 4:
                    for lat, ch in _SuryaOCRUnused._LATIN_TO_CHINESE_FIXES.items():
                        text = text.replace(lat, ch)
                blk_list[idx].text = [text]

        def ocr_img(self, img: np.ndarray) -> str:
            blk = TextBlock(xyxy=[0, 0, img.shape[1], img.shape[0]])
            blk.lines = [[[0, 0], [img.shape[1], 0], [img.shape[1], img.shape[0]], [0, img.shape[0]]]]
            self._ocr_blk_list(img, [blk])
            return blk.get_text()
