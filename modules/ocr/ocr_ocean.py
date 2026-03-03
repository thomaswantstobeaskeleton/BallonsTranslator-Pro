"""
Ocean-OCR – 3B MLLM OCR (Hugging Face). First MLLM to outperform dedicated OCR (PaddleOCR, TextIn).
Document, scene text, handwritten. Quality over speed.
Requires: pip install transformers torch pillow accelerate easydict.
Optional: flash-attn (faster); on Windows a stub is used if flash-attn is not installed.
"""
from typing import List
import os
import re
import sys
import tempfile
import traceback
import numpy as np
import cv2
from .base import OCRBase, register_OCR, DEVICE_SELECTOR, TextBlock

_OCEAN_AVAILABLE = False
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from PIL import Image
    import torch
    _OCEAN_AVAILABLE = True
except ImportError as e:
    import logging
    logging.getLogger("BallonTranslator").debug(
        f"Ocean-OCR not available: {e}. Install: pip install transformers torch pillow accelerate easydict"
    )


def _cv2_to_pil_rgb(img: np.ndarray) -> "Image.Image":
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)


# Prefixes/phrases that Ocean (or similar MLLMs) may echo; strip them so only OCR content remains.
# Each regex has one capturing group (.*) for the content after the prefix.
# CJK character set (Chinese, Japanese Kanji) for stray-character cleanup
_OCEAN_CJK = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7a3]")

_OCEAN_RESPONSE_PREFIXES = [
    # Instruction echoes (model repeating "read the text" style prompts)
    re.compile(r"^Read\s+the\s+main\s+text\s+in\s+the\s+image\.?\s*(.*)", re.IGNORECASE | re.DOTALL),
    re.compile(r"^Read\s+the\s+text\s+in\s+the\s+image\.?\s*(.*)", re.IGNORECASE | re.DOTALL),
    re.compile(r"^The\s+text\s+in\s+the\s+image\s+reads\s*:\s*[\"']?\s*(.*)", re.IGNORECASE | re.DOTALL),
    re.compile(r"^The\s+text\s+in\s+the\s+image\s+is\s*:\s*[\"']?(.*)", re.IGNORECASE | re.DOTALL),
    re.compile(r"^The\s+following\s+is\s+the\s+text\s+content\s+in\s+the\s+image\s*:\s*[\"']?(.*)", re.IGNORECASE | re.DOTALL),
    re.compile(r"^The\s+text\s+reads\s*:\s*[\"']?\s*(.*)", re.IGNORECASE | re.DOTALL),
    re.compile(r"^The\s+text\s+content\s+in\s+this\s+picture\s+is\s+as\s+follows\s*:\s*[\"']?(.*)", re.IGNORECASE | re.DOTALL),
    re.compile(r"^The\s+image\s+contains\s+(?:small\s+)?(?:a\s+)?(?:single\s+)?(?:line\s+of\s+)?(?:character|text)\s*,?\s*(?:which\s+reads?\s*:?\s*)?[\"']?\s*(.*)", re.IGNORECASE | re.DOTALL),
    re.compile(r"^The\s+image\s+contains\s+a\s+single\s+character\s*,?\s*(?:in\s+Chinese\s*,?)?\s*which\s+is\s*[\"']?\s*(.*)", re.IGNORECASE | re.DOTALL),
    re.compile(r"^The\s+image\s+contains\s+Chinese\s+characters\s*,?\s*\.?\s*The\s+text\s+reads\s*:\s*[\"']?\s*(.*)", re.IGNORECASE | re.DOTALL),
    re.compile(r"^The\s+image\s+contains\s+Chinese\s+characters\s*,?\s*which\s+are\s*[\"']?\s*(.*)", re.IGNORECASE | re.DOTALL),
    re.compile(r"^The\s+image\s+contains\s+(?:a\s+single\s+)?(?:character|text)\s*,?\s*(?:which\s+is\s*|:)\s*[\"']?(.*)", re.IGNORECASE | re.DOTALL),
    re.compile(r"^Extract\s+all\s+text\s+(?:information\s+)?from\s+(?:this\s+)?(?:the\s+)?image\s*\.?\s*:?\s*[\"']?(.*)", re.IGNORECASE | re.DOTALL),
    re.compile(r"^图像中的文字是\s*[：:\s]*[\"']?(.*)", re.DOTALL),
    re.compile(r"^从图片中提取所有文本信息\s*[：:\.\s]*[\"']?(.*)", re.DOTALL),
    re.compile(r"^以下是图片中的文字内容\s*[：:\s]*[\"']?(.*)", re.DOTALL),
    re.compile(r"^以下是图片中的文字信息\s*[：:\s]*[\"']?(.*)", re.DOTALL),
]


def _strip_ocean_response(raw: str, prompt: str) -> str:
    """Extract only the OCR text from Ocean's full response (prompt + wrapper phrases)."""
    if not raw or not isinstance(raw, str):
        return raw or ""
    text = raw.strip()
    if not text:
        return ""
    # Remove the exact prompt from the start
    if prompt:
        prompt_stripped = prompt.strip()
        for prefix in (prompt_stripped, prompt_stripped.rstrip("?")):
            if text.startswith(prefix):
                text = text[len(prefix):].lstrip(". :").strip()
                break
    # Strip known wrapper phrases (English and Chinese); repeat to handle nested echoes
    for _ in range(5):
        stripped = False
        for pattern in _OCEAN_RESPONSE_PREFIXES:
            m = pattern.match(text)
            if m and m.group(1) is not None:
                candidate = m.group(1).strip()
                if candidate != text:
                    text = candidate
                    stripped = True
                    break
        if not stripped:
            break
    # Strip one layer of surrounding quotes if present
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'\u201c\u201d\u2018\u2019":
        text = text[1:-1].strip()
    # Remove trailing phonetic in parens e.g. " (o)." or " (o)"
    text = re.sub(r"\s*\([a-zA-Z]\)\.?\s*$", "", text).strip()
    text = text.strip()
    # Fix stray single CJK glued to English (e.g. "the丑"): if text is mostly Latin and has one CJK at start/end, remove it
    if len(text) > 1 and _OCEAN_CJK.search(text):
        cjk_count = len(_OCEAN_CJK.findall(text))
        non_cjk_len = len(_OCEAN_CJK.sub("", text))
        if cjk_count == 1 and non_cjk_len >= 3:
            if _OCEAN_CJK.match(text[-1]):
                text = text[:-1].rstrip()
            elif _OCEAN_CJK.match(text[0]):
                text = text[1:].lstrip()
    return text


if _OCEAN_AVAILABLE:

    @register_OCR("ocean_ocr")
    class OceanOCROCR(OCRBase):
        """
        Ocean-OCR 3B: SOTA OCR MLLM (document, scene, handwritten). Quality over speed.
        Uses guoxy25/Ocean-OCR; trust_remote_code. Per-block via temp image path.
        """
        params = {
            "model_name": {
                "type": "line_editor",
                "value": "guoxy25/Ocean-OCR",
                "description": "Hugging Face model id (Ocean-OCR 3B).",
            },
            "device": DEVICE_SELECTOR(),
            "crop_padding": {
                "type": "line_editor",
                "value": 4,
                "description": "Pixels to add around each box when cropping (0–24).",
            },
            "prompt": {
                "type": "line_editor",
                "value": "Extract all main text from this image. Read only the primary, foreground text (e.g. speech bubble or caption). Ignore any faint, blurred, or background text that overlaps the same region. Output only the recognized text, no explanation.",
                "description": "OCR prompt. Prefer foreground-only to avoid mixing background text (e.g. EN+CN in one box).",
            },
            "max_new_tokens": {
                "type": "line_editor",
                "value": 2048,
                "description": "Max new tokens per block.",
            },
            "use_bf16": {
                "type": "checkbox",
                "value": True,
                "description": "Use bfloat16 when available.",
            },
            "description": "Ocean-OCR 3B – SOTA OCR MLLM (HF, trust_remote_code).",
        }
        _load_model_keys = {"model", "tokenizer"}

        def __init__(self, **params) -> None:
            super().__init__(**params)
            self.device = self.params["device"]["value"]
            self.model = None
            self.tokenizer = None
            self._model_name = None

        def _load_model(self):
            model_name = (self.params.get("model_name") or {}).get("value", "guoxy25/Ocean-OCR") or "guoxy25/Ocean-OCR"
            if self.model is not None and self._model_name == model_name:
                return
            self._model_name = model_name
            # Stub for flash_attn when not installed (e.g. Windows); Ocean uses use_flash=False when seqlens is None.
            _stub_dir = os.path.join(os.path.dirname(__file__), "flash_attn_stub")
            if os.path.isdir(_stub_dir) and _stub_dir not in sys.path:
                sys.path.insert(0, _stub_dir)
            # Ocean-OCR custom code (processor_ocean, etc.) lives in the HF repo; add to path so
            # transformers' check_imports can find it when loading trust_remote_code.
            try:
                from utils.model_manager import get_model_manager
                ocean_repo_path = get_model_manager().snapshot_download(model_name)
                if ocean_repo_path and ocean_repo_path not in sys.path:
                    sys.path.insert(0, ocean_repo_path)
            except Exception as e:
                self.logger.debug(f"Ocean-OCR: could not add repo to path: {e}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True, resume_download=True
            )
            use_bf16 = self.params.get("use_bf16", {}).get("value", True)
            dtype = torch.bfloat16
            if not (torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)()):
                dtype = torch.float16
            if not use_bf16:
                dtype = torch.float16
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=None,
                trust_remote_code=True,
                torch_dtype=dtype,
            )
            self.model.to(self.device)
            self.model.eval()
            if hasattr(self.model, "bind_processor"):
                # Ocean's OceanAudioProcessor asserts torchaudio backends; we only use image OCR.
                try:
                    import torchaudio
                    if len(torchaudio.list_audio_backends()) == 0:
                        torchaudio.list_audio_backends = lambda: ["soundfile"]
                except Exception:
                    pass
                self.model.bind_processor(self.tokenizer, training=False)

        def _run_one(self, pil_img: "Image.Image") -> str:
            tmp_path = None
            try:
                fd, tmp_path = tempfile.mkstemp(suffix=".png")
                os.close(fd)
                pil_img.save(tmp_path)
                prompt = (self.params.get("prompt") or {}).get("value", "Can you pull all textual information from the images?") or "Can you pull all textual information from the images?"
                # Ocean processor expects image ref between image_start_tag and image_end_tag (regex in extract_replace_multimodal).
                proc = self.model.processor
                start_tag = getattr(proc, "image_start_tag", "") or ""
                end_tag = getattr(proc, "image_end_tag", "") or ""
                local_path = os.path.abspath(tmp_path).replace("\\", "/")
                image_ref = start_tag + '{"local": "' + local_path + '"}' + end_tag
                full_input = " " + image_ref + " " + prompt + " "
                if not hasattr(self.model, "processor"):
                    self.logger.warning("Ocean-OCR: model has no processor (bind_processor may not have set it).")
                    return ""
                ret = self.model.processor(full_input)
                input_ids = ret.input_ids
                images = ret.images
                if images is None or (isinstance(images, list) and len(images) == 0):
                    self.logger.warning("Ocean-OCR: no images from processor.")
                    return ""
                # Processor may return numpy; model expects tensors on device.
                def _to_tensor(x, device):
                    if x is None:
                        return None
                    if isinstance(x, torch.Tensor):
                        return x.to(device)
                    t = torch.as_tensor(x)
                    return t.to(device)
                img_list = images if isinstance(images, list) else [images]
                images_t = [_to_tensor(t, self.device) for t in img_list]
                input_ids_t = _to_tensor(
                    input_ids if isinstance(input_ids, list) else input_ids,
                    self.device,
                )
                if input_ids_t is None:
                    self.logger.warning("Ocean-OCR: processor returned no input_ids.")
                    return ""
                if input_ids_t.dim() == 1:
                    input_ids_t = input_ids_t.unsqueeze(0)
                enc_len = getattr(ret, "encoder_length", None)
                br_len = getattr(ret, "bridge_length", None)
                encoder_length_t = _to_tensor(enc_len, self.device) if enc_len is not None else None
                bridge_length_t = _to_tensor(br_len, self.device) if br_len is not None else None
                images_grid = getattr(ret, "images_grid", None)
                if images_grid is not None and isinstance(images_grid, list):
                    images_grid = [_to_tensor(g, self.device) for g in images_grid]
                max_new_tokens = 2048
                mt = self.params.get("max_new_tokens", {})
                if isinstance(mt, dict):
                    try:
                        max_new_tokens = max(64, min(4096, int(mt.get("value", 2048))))
                    except (TypeError, ValueError):
                        pass
                # Satisfy transformers: pass pad_token_id and attention_mask to avoid warnings
                pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
                if pad_token_id is None:
                    pad_token_id = getattr(self.tokenizer, "eos_token_id", None)
                attention_mask = (input_ids_t != (pad_token_id or 0)).long()
                if pad_token_id is None:
                    attention_mask = torch.ones_like(input_ids_t, dtype=torch.long, device=input_ids_t.device)

                # Patch Tensor.numpy so CUDA tensors call .cpu() first (avoids "can't convert cuda tensor to numpy" inside generate()).
                _orig_numpy = torch.Tensor.numpy
                def _numpy_cpu_first(self, *args, **kwargs):
                    t = self.cpu() if self.is_cuda else self
                    return _orig_numpy(t, *args, **kwargs)
                torch.Tensor.numpy = _numpy_cpu_first
                gen_kw = dict(
                    inputs=input_ids_t,
                    images=images_t,
                    labels=None,
                    audios=None,
                    encoder_length=encoder_length_t,
                    bridge_length=bridge_length_t,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    top_k=5,
                    top_p=0.85,
                    temperature=0,
                    num_return_sequences=1,
                    repetition_penalty=1.05,
                    use_cache=False,  # Ocean expects tuple cache; avoid DynamicCache (not subscriptable)
                    images_grid=images_grid,
                )
                if attention_mask is not None:
                    gen_kw["attention_mask"] = attention_mask
                if pad_token_id is not None:
                    gen_kw["pad_token_id"] = pad_token_id
                try:
                    with torch.inference_mode():
                        out = self.model.generate(**gen_kw)
                except TypeError:
                    # Model may not accept attention_mask/pad_token_id; retry without to avoid breaking
                    gen_kw.pop("attention_mask", None)
                    gen_kw.pop("pad_token_id", None)
                    with torch.inference_mode():
                        out = self.model.generate(**gen_kw)
                finally:
                    torch.Tensor.numpy = _orig_numpy
                # Decode: handle None, tuple (seq, scores), or tensor from generate().
                if out is None:
                    return ""
                if isinstance(out, (list, tuple)) and len(out) == 2 and isinstance(out[0], torch.Tensor):
                    out = out[0]
                elif isinstance(out, (list, tuple)) and len(out) >= 1 and out[0] is None:
                    return ""
                if isinstance(out, torch.Tensor):
                    out = out.cpu().tolist()
                if not isinstance(out, list):
                    out = list(out) if out is not None else []
                if not out:
                    return ""
                if isinstance(out[0], int):
                    out = [out]
                outputs = ""
                for res in out:
                    if res is None:
                        continue
                    ids = res.cpu().tolist() if isinstance(res, torch.Tensor) else (res if isinstance(res, list) else list(res))
                    decoded = self.tokenizer.decode(ids, skip_special_tokens=True)
                    outputs += decoded
                return _strip_ocean_response(outputs.strip(), prompt)
            except Exception as e:
                self.logger.warning(f"Ocean-OCR failed: {e}")
                self.logger.debug(traceback.format_exc())
                return ""
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

        def _ocr_blk_list(self, img: np.ndarray, blk_list: List[TextBlock], *args, **kwargs) -> None:
            im_h, im_w = img.shape[:2]
            pad = 0
            cp = self.params.get("crop_padding", {})
            if isinstance(cp, dict):
                try:
                    pad = max(0, min(24, int(cp.get("value", 0))))
                except (TypeError, ValueError):
                    pass
            for blk in blk_list:
                x1, y1, x2, y2 = blk.xyxy
                if pad > 0:
                    x1 = max(0, x1 - pad)
                    y1 = max(0, y1 - pad)
                    x2 = min(im_w, x2 + pad)
                    y2 = min(im_h, y2 + pad)
                if not (x1 < x2 and y1 < y2):
                    blk.text = [""]
                    continue
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    blk.text = [""]
                    continue
                # Ocean processor rejects "image too small" (e.g. 14x14); skip to avoid warning and empty result
                min_side = 24
                if crop.shape[0] < min_side or crop.shape[1] < min_side:
                    self.logger.debug(
                        "Ocean-OCR: crop too small (%s), skipping (min %s px).",
                        (crop.shape[1], crop.shape[0]),
                        min_side,
                    )
                    blk.text = [""]
                    continue
                pil_img = _cv2_to_pil_rgb(crop)
                text = self._run_one(pil_img)
                blk.text = [text if text else ""]

        def ocr_img(self, img: np.ndarray) -> str:
            pil_img = _cv2_to_pil_rgb(img)
            return self._run_one(pil_img)

        def updateParam(self, param_key: str, param_content):
            super().updateParam(param_key, param_content)
            if param_key == "device":
                self.device = self.params["device"]["value"]
                if self.model is not None:
                    self.model.to(self.device)
            elif param_key in ("model_name", "use_bf16"):
                self.model = None
                self.tokenizer = None
                self.processor = None
                self._model_name = None
