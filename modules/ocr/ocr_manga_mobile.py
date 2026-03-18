"""
Manga OCR Mobile – Lightweight Japanese manga OCR via TFLite (bluolightning/manga-ocr-mobile).
Encoder+decoder .tflite + HF tokenizer; good for mobile/edge. Optional: pip install tflite-runtime (or tensorflow).
"""
import os
import re
from typing import List
import numpy as np
import cv2

from .base import OCRBase, register_OCR, DEFAULT_DEVICE, DEVICE_SELECTOR, TextBlock

try:
    import jaconv
except ImportError:
    jaconv = None

try:
    import jaconv
except ImportError:
    jaconv = None

_interpreter_enc = _interpreter_dec = _tokenizer = None

def _post_process(text: str) -> str:
    text = "".join(text.split())
    text = text.replace("…", "...")
    if jaconv:
        text = jaconv.h2z(text, ascii=True, digit=True)
    return text


def _load_tflite_and_tokenizer(cache_dir: str):
    """Load TFLite encoder/decoder and tokenizer. Returns (True, None) on success, (False, error_message) on failure."""
    global _interpreter_enc, _interpreter_dec, _tokenizer
    try:
        from utils.model_manager import get_model_manager
        root = get_model_manager().snapshot_download("bluolightning/manga-ocr-mobile", cache_dir=cache_dir)
        base = os.path.join(root, "v1_fp16")
        enc_path = os.path.join(base, "encoder.tflite")
        dec_path = os.path.join(base, "decoder.tflite")
        tok_path = os.path.join(base, "tokenizer")
        if not os.path.isfile(enc_path) or not os.path.isfile(dec_path):
            return False, (
                f"Model files not found in {base!r} (encoder.tflite and decoder.tflite required). "
                "Check that the Hugging Face repo bluolightning/manga-ocr-mobile contains v1_fp16/."
            )
        try:
            import tflite_runtime.interpreter as tflite
        except ImportError:
            try:
                import tensorflow.lite as tflite
            except ImportError as e:
                return False, (
                    "tflite_runtime and tensorflow are not installed. "
                    "Install one of: pip install tflite-runtime  (or  pip install tensorflow)"
                ) + f" — {e}"
        _interpreter_enc = tflite.Interpreter(model_path=enc_path)
        _interpreter_enc.allocate_tensors()
        _interpreter_dec = tflite.Interpreter(model_path=dec_path)
        _interpreter_dec.allocate_tensors()
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(tok_path)
        return True, None
    except Exception as e:
        return False, str(e)


def _run_manga_mobile(img: np.ndarray, interpreter_enc, interpreter_dec, tokenizer) -> str:
    """Run encoder + decoder TFLite and decode. Assumes encoder input [1, 32, W, 3] (NHWC)."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    # Resize to height 32 (common for line OCR)
    target_h = 32
    scale = target_h / max(h, 1)
    new_w = max(1, int(w * scale))
    img = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_LINEAR)
    # NHWC float32 in [0,1] or normalized
    inp = img.astype(np.float32) / 255.0
    inp = np.expand_dims(inp, axis=0)

    enc_in = interpreter_enc.get_input_details()[0]
    dec_in = interpreter_enc.get_output_details()[0]
    if enc_in["shape"][1] != target_h or enc_in["dtype"] != np.float32:
        # Try NCHW or different size
        if len(enc_in["shape"]) == 4 and enc_in["shape"][1] == 3:
            inp = np.transpose(inp, (0, 3, 1, 2))
        if enc_in["shape"][2] != target_h:
            target_h_in = enc_in["shape"][2]
            new_w_in = max(1, int(new_w * target_h_in / target_h))
            inp = cv2.resize(img, (new_w_in, target_h_in), interpolation=cv2.INTER_LINEAR)
            inp = inp.astype(np.float32) / 255.0
            inp = np.expand_dims(inp, axis=0)
            if len(enc_in["shape"]) == 4 and enc_in["shape"][1] == 3:
                inp = np.transpose(inp, (0, 3, 1, 2))

    interpreter_enc.set_tensor(enc_in["index"], inp.astype(enc_in["dtype"]))
    interpreter_enc.invoke()
    enc_out = interpreter_enc.get_tensor(dec_in["index"])

    dec_details = interpreter_dec.get_input_details()
    out_details = interpreter_dec.get_output_details()
    # Feed encoder output and run decoder (often autoregressive; simplify: single step if possible)
    for d in dec_details:
        idx = d["index"]
        name = d.get("name", "")
        shape = d["shape"]
        if enc_out.shape == shape or (enc_out.size == np.prod(shape)):
            interpreter_dec.set_tensor(idx, enc_out.reshape(shape).astype(d["dtype"]))
        elif "input_ids" in name or "decoder" in name:
            # Decoder input_ids: start token
            pad_id = getattr(tokenizer, "pad_token_id", 0) or 0
            start = getattr(tokenizer, "bos_token_id", tokenizer.cls_token_id) or pad_id
            batch = 1
            seq = shape[-1] if len(shape) > 1 else 1
            interpreter_dec.set_tensor(idx, np.full(shape, start, dtype=d["dtype"]))
        else:
            interpreter_dec.set_tensor(idx, np.zeros(shape, dtype=d["dtype"]))
    interpreter_dec.invoke()
    out = interpreter_dec.get_tensor(out_details[0]["index"])
    # out might be [1, seq_len] token ids
    if out.size == 0:
        return ""
    ids = out.flatten()
    if hasattr(tokenizer, "decode"):
        text = tokenizer.decode(ids, skip_special_tokens=True)
    else:
        text = "".join(chr(int(x)) for x in ids if 0 < int(x) < 0x110000)
    return _post_process(text)


# Lazy load on first use (no load at import)


@register_OCR("manga_ocr_mobile")
class MangaOCRMobile(OCRBase):
    """
    Manga OCR Mobile: lightweight Japanese manga OCR (TFLite, bluolightning/manga-ocr-mobile).
    Optional: pip install tflite-runtime huggingface_hub transformers. Falls back to manga_ocr if unavailable.
    """
    params = {
        "cache_dir": {
            "type": "line_editor",
            "value": "",
            "description": "Hugging Face cache dir (empty = default).",
        },
        "crop_padding": {
            "type": "line_editor",
            "value": 6,
            "description": "Pixels to add around each box when cropping (0–24).",
        },
        "device": DEVICE_SELECTOR(),
        "description": "Manga OCR Mobile (TFLite). Install: pip install tflite-runtime huggingface_hub transformers",
    }
    _load_model_keys = {"enc", "dec", "tokenizer"}

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self._enc = self._dec = self._tokenizer = None
        self._loaded = False

    def _load_model(self):
        global _interpreter_enc, _interpreter_dec, _tokenizer
        if self._loaded and _interpreter_enc is not None:
            return
        cache = (self.params.get("cache_dir") or {}).get("value", "") or None
        ok, err = _load_tflite_and_tokenizer(cache)
        if not ok:
            hint = "Install: pip install tflite-runtime huggingface_hub transformers"
            raise RuntimeError(
                f"Manga OCR Mobile: failed to load TFLite model from bluolightning/manga-ocr-mobile. {hint}"
                + (f" — {err}" if err else "")
            )
        self._enc = _interpreter_enc
        self._dec = _interpreter_dec
        self._tokenizer = _tokenizer
        self._loaded = True

    def ocr_img(self, img: np.ndarray) -> str:
        if not self.all_model_loaded():
            self.load_model()
        return _run_manga_mobile(img, self._enc, self._dec, self._tokenizer)

    def _ocr_blk_list(self, img: np.ndarray, blk_list: List[TextBlock], *args, **kwargs) -> None:
        im_h, im_w = img.shape[:2]
        pad_val = 6
        try:
            pad_val = max(0, min(24, int((self.params.get("crop_padding") or {}).get("value", 6))))
        except (TypeError, ValueError):
            pass
        for blk in blk_list:
            x1, y1, x2, y2 = blk.xyxy
            x1 = max(0, min(int(round(float(x1))), im_w - 1))
            y1 = max(0, min(int(round(float(y1))), im_h - 1))
            x2 = max(x1 + 1, min(int(round(float(x2))), im_w))
            y2 = max(y1 + 1, min(int(round(float(y2))), im_h))
            if pad_val > 0:
                x1 = max(0, x1 - pad_val)
                y1 = max(0, y1 - pad_val)
                x2 = min(im_w, x2 + pad_val)
                y2 = min(im_h, y2 + pad_val)
            if x2 > x1 and y2 > y1 and x1 >= 0 and y1 >= 0 and x2 <= im_w and y2 <= im_h:
                region = img[y1:y2, x1:x2]
                blk.text = [self.ocr_img(region)]
            else:
                blk.text = [""]
