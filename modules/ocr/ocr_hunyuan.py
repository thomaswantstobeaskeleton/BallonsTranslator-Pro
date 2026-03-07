"""
HunyuanOCR – Tencent 1B end-to-end OCR VLM (tencent/HunyuanOCR).
100+ languages; text spotting, document parsing, extraction. Chat-style image + prompt -> text.
Requires: pip install transformers torch pillow (transformers 4.49+ with HunyuanVL support or trust_remote_code).
"""
from typing import List
import os
import tempfile
import numpy as np
import cv2
from .base import OCRBase, register_OCR, DEFAULT_DEVICE, DEVICE_SELECTOR, TextBlock

_HUNYUAN_AVAILABLE = False
get_class_from_dynamic_module = None
try:
    from transformers import AutoProcessor, AutoModelForImageTextToText, AutoConfig
    from PIL import Image
    import torch
    _HUNYUAN_AVAILABLE = True
    try:
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
    except ImportError:
        get_class_from_dynamic_module = None
except ImportError as e:
    import logging
    logging.getLogger("BallonTranslator").debug(
        f"HunyuanOCR not available: {e}. Install: pip install transformers torch pillow"
    )

# Repo that ships hunyuan_vl modeling .py (tencent/HunyuanOCR has weights only).
_HUNYUAN_OCR_CODE_REPO = "lvyufeng/HunyuanOCR"


def _cv2_to_pil_rgb(img: np.ndarray) -> "Image.Image":
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)


def _clean_repeated_substrings(text: str) -> str:
    """Clean repeated substrings in model output (from HunyuanOCR recipe)."""
    n = len(text)
    if n < 8000:
        return text
    for length in range(2, n // 10 + 1):
        candidate = text[-length:]
        count = 0
        i = n - length
        while i >= 0 and text[i : i + length] == candidate:
            count += 1
            i -= length
        if count >= 10:
            return text[: n - length * (count - 1)]
    return text


if _HUNYUAN_AVAILABLE:

    @register_OCR("hunyuan_ocr")
    class HunyuanOCROCR(OCRBase):
        """
        HunyuanOCR 1B via Hugging Face. 100+ languages; spotting, document parsing, extraction.
        Use with any detector. Prompt: extract text from image (or spotting with coords).
        """
        params = {
            "model_name": {
                "type": "line_editor",
                "value": "tencent/HunyuanOCR",
                "description": "Hugging Face model id. Requires transformers with HunyuanVL or trust_remote_code.",
            },
            "device": DEVICE_SELECTOR(),
            "crop_padding": {
                "type": "line_editor",
                "value": 4,
                "description": "Pixels to add around each box when cropping (0–24).",
            },
            "max_new_tokens": {
                "type": "line_editor",
                "value": 256,
                "description": "Max tokens per block.",
            },
            "prompt_type": {
                "type": "selector",
                "options": ["extract_text", "spotting_cn", "spotting_en"],
                "value": "extract_text",
                "description": "extract_text: plain text. spotting_cn/en: text with coordinates (CN/EN prompt).",
            },
            "use_bf16": {
                "type": "checkbox",
                "value": True,
                "description": "Use bfloat16 when available.",
            },
            "description": "HunyuanOCR 1B (Tencent). 100+ languages, document OCR.",
        }
        _load_model_keys = {"processor", "model"}

        _PROMPTS = {
            "extract_text": "提取图中的文字。",
            "spotting_cn": "检测并识别图片中的文字，将文本坐标格式化输出。",
            "spotting_en": "Detect and recognize text in the image, output text with coordinates.",
        }

        def __init__(self, **params) -> None:
            super().__init__(**params)
            self.device = self.params["device"]["value"]
            self.processor = None
            self.model = None
            self._model_name = None

        def _load_model(self):
            model_name = (
                (self.params.get("model_name") or {}).get("value", "tencent/HunyuanOCR")
                or "tencent/HunyuanOCR"
            )
            if self.processor is not None and self._model_name == model_name:
                return
            self._model_name = model_name
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            use_bf16 = self.params.get("use_bf16", {}).get("value", True)
            dtype = (
                torch.bfloat16
                if (
                    use_bf16
                    and torch.cuda.is_available()
                    and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
                )
                else torch.float16
            )
            config = None
            try:
                config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            except (KeyError, ValueError) as e:
                err_str = str(e)
                if "hunyuan_vl" in err_str or "hunyuan-vl" in err_str.lower():
                    if get_class_from_dynamic_module is not None:
                        try:
                            config_class = get_class_from_dynamic_module(
                                "configuration_hunyuan_vl.HunYuanVLConfig",
                                _HUNYUAN_OCR_CODE_REPO,
                            )
                            config = config_class.from_pretrained(model_name)
                        except (OSError, AttributeError, Exception):
                            raise ValueError(
                                "tencent/HunyuanOCR requires transformers 4.49+. "
                                "Upgrade with: pip install --upgrade 'transformers>=4.49'"
                            ) from e
                    else:
                        raise ValueError(
                            "tencent/HunyuanOCR requires transformers 4.49+. "
                            "Upgrade with: pip install --upgrade 'transformers>=4.49'"
                        ) from e
                else:
                    raise
            load_kw = {"trust_remote_code": True}
            if config is not None:
                load_kw["config"] = config
            try:
                self.model = AutoModelForImageTextToText.from_pretrained(
                    model_name, torch_dtype=dtype, **load_kw
                )
            except (KeyError, ValueError) as model_err:
                err_str = str(model_err)
                is_hunyuan = (
                    config is not None and getattr(config, "model_type", None) == "hunyuan_vl"
                ) or "hunyuan_vl" in err_str or "hunyuan-vl" in err_str.lower()
                if get_class_from_dynamic_module is not None and config is not None and is_hunyuan:
                    try:
                        model_class = get_class_from_dynamic_module(
                            "modeling_hunyuan_vl.HunYuanVLForConditionalGeneration",
                            _HUNYUAN_OCR_CODE_REPO,
                        )

                        # Patch post_init to ignore tied-weights issues on older transformers and ensure
                        # minimal tied-weights metadata is present.
                        orig_post_init = getattr(model_class, "post_init", None)
                        if orig_post_init is not None:
                            def _safe_post_init(self, *args, **kwargs):  # type: ignore[override]
                                try:
                                    return orig_post_init(self, *args, **kwargs)
                                except AttributeError as e:
                                    msg = str(e)
                                    if "list' object has no attribute 'keys'" in msg or "all_tied_weights_keys" in msg:
                                        if not hasattr(self, "_tp_plan"):
                                            self._tp_plan = {}
                                        if not hasattr(self, "_ep_plan"):
                                            self._ep_plan = {}
                                        if not hasattr(self, "_pp_plan"):
                                            self._pp_plan = {}
                                        if not hasattr(self, "all_tied_weights_keys"):
                                            self.all_tied_weights_keys = {}
                                        if not hasattr(self, "_keep_in_fp32_modules"):
                                            self._keep_in_fp32_modules = set()
                                        if not hasattr(self, "_keep_in_fp32_modules_strict"):
                                            self._keep_in_fp32_modules_strict = set()
                                        if not hasattr(self, "_no_split_modules"):
                                            self._no_split_modules = set()
                                        return
                                    raise

                            model_class.post_init = _safe_post_init  # type: ignore[assignment]

                        # Patch _finalize_model_loading to tolerate missing all_tied_weights_keys on this model.
                        orig_finalize = getattr(model_class, "_finalize_model_loading", None)
                        if orig_finalize is not None:
                            def _safe_finalize_model_loading(model, load_config, loading_info):  # type: ignore[override]
                                try:
                                    return orig_finalize(model, load_config, loading_info)
                                except AttributeError as e:
                                    if "all_tied_weights_keys" in str(e):
                                        if not hasattr(model, "all_tied_weights_keys"):
                                            model.all_tied_weights_keys = {}
                                        return loading_info
                                    raise

                            model_class._finalize_model_loading = staticmethod(_safe_finalize_model_loading)  # type: ignore[assignment]

                        self.model = model_class.from_pretrained(
                            model_name, config=config, torch_dtype=dtype
                        )
                    except Exception as load_e:
                        self.logger.warning(f"HunyuanOCR model load via code repo failed: {load_e}")
                        if "dtype" in str(load_e):
                            try:
                                self.model = model_class.from_pretrained(model_name, config=config)
                            except Exception:
                                raise load_e
                        else:
                            raise
                else:
                    self.logger.warning(f"HunyuanOCR load with dtype failed: {model_err}; trying default dtype.")
                    self.model = AutoModelForImageTextToText.from_pretrained(
                        model_name, **load_kw
                    )
            except Exception as e:
                self.logger.warning(f"HunyuanOCR load with dtype failed: {e}; trying default dtype.")
                self.model = AutoModelForImageTextToText.from_pretrained(
                    model_name, **load_kw
                )
            self.model.to(self.device)

        def _run_one(self, pil_img: "Image.Image") -> str:
            tmp_path = None
            try:
                fd, tmp_path = tempfile.mkstemp(suffix=".png")
                os.close(fd)
                pil_img.save(tmp_path)
                pt = self.params.get("prompt_type", {}).get("value", "extract_text")
                prompt_text = self._PROMPTS.get(pt, self._PROMPTS["extract_text"])
                messages = [
                    {"role": "system", "content": ""},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": tmp_path},
                            {"type": "text", "text": prompt_text},
                        ],
                    },
                ]
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self.processor(
                    text=[text],
                    images=[pil_img],
                    padding=True,
                    return_tensors="pt",
                )
                inputs = {k: (v.to(self.model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
                max_tokens = 256
                mt = self.params.get("max_new_tokens", {})
                if isinstance(mt, dict):
                    try:
                        max_tokens = max(32, min(2048, int(mt.get("value", 256))))
                    except (TypeError, ValueError):
                        pass
                with torch.inference_mode():
                    generated_ids = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
                if "input_ids" in inputs:
                    in_ids = inputs["input_ids"]
                else:
                    in_ids = getattr(inputs, "input_ids", inputs.get("inputs"))
                if in_ids is not None and in_ids.dim() >= 1:
                    in_len = in_ids.shape[-1]
                    out_ids = generated_ids[0][in_len:]
                else:
                    out_ids = generated_ids[0]
                decoded = self.processor.decode(
                    out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                return _clean_repeated_substrings(decoded).strip()
            except Exception as e:
                self.logger.warning(f"HunyuanOCR failed: {e}")
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
                val = cp.get("value", 0)
            else:
                val = 0
            try:
                pad = max(0, min(24, int(val)))
            except (TypeError, ValueError):
                pass
            for blk in blk_list:
                x1, y1, x2, y2 = blk.xyxy
                x1 = max(0, min(int(round(float(x1))), im_w - 1))
                y1 = max(0, min(int(round(float(y1))), im_h - 1))
                x2 = max(x1 + 1, min(int(round(float(x2))), im_w))
                y2 = max(y1 + 1, min(int(round(float(y2))), im_h))
                if pad > 0:
                    x1 = max(0, x1 - pad)
                    y1 = max(0, y1 - pad)
                    x2 = min(im_w, x2 + pad)
                    y2 = min(im_h, y2 + pad)
                if not (x1 < x2 and y1 < y2 and x2 <= im_w and y2 <= im_h and x1 >= 0 and y1 >= 0):
                    blk.text = [""]
                    continue
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
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
                self.processor = None
                self.model = None
                self._model_name = None
