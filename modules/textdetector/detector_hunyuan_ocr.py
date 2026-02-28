"""
HunyuanOCR as text detector – full-image spotting (detection + recognition in one run).
Returns boxes and text when the model output is parseable. Pair with none_ocr to keep spotter text;
or use with any OCR to re-run recognition on crops.
Requires: same as HunyuanOCR OCR (transformers, torch, pillow, trust_remote_code).
"""
import os
import re
import json
import tempfile
import numpy as np
import cv2
from typing import Tuple, List, Any

from .base import register_textdetectors, TextDetectorBase, TextBlock, ProjImgTrans
from ..base import DEVICE_SELECTOR

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
except ImportError:
    import logging
    logging.getLogger("BallonTranslator").debug("HunyuanOCR detector: transformers/torch/pillow not available.")

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


def _parse_spotting_output(raw: str) -> List[Tuple[List[Tuple[int, int]], str]]:
    """
    Parse HunyuanOCR spotting output into list of (polygon_points, text).
    Tries: JSON array of {box/poly, text}; then regex for [[x,y,...]], "text" or similar.
    Returns list of (pts, text); pts as [(x,y), ...] or 4 corners; text may be "".
    """
    raw = raw.strip()
    if not raw:
        return []
    out: List[Tuple[List[Tuple[int, int]], str]] = []

    # Try full string as JSON array
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                box = item.get("box", item.get("poly", item.get("bbox", item.get("points"))))
                text = item.get("text", item.get("content", ""))
                if isinstance(text, (list, tuple)):
                    text = " ".join(str(t) for t in text)
                if box is None:
                    continue
                if isinstance(box, (list, tuple)):
                    pts = []
                    for p in box:
                        if isinstance(p, (list, tuple)) and len(p) >= 2:
                            pts.append((int(p[0]), int(p[1])))
                        elif isinstance(p, (int, float)):
                            if len(box) >= 4 and len(pts) < 2:
                                # flat [x1,y1,x2,y2]
                                pts = [(int(box[0]), int(box[1])), (int(box[2]), int(box[3]))]
                            break
                    if len(pts) >= 2:
                        if len(pts) == 2:
                            x1, y1, x2, y2 = pts[0][0], pts[0][1], pts[1][0], pts[1][1]
                            pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                        out.append((pts, str(text)))
            if out:
                return out
    except (json.JSONDecodeError, TypeError):
        pass

    # Try to find JSON array inside the string
    match = re.search(r'\[[\s\S]*\]', raw)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, list) and data:
                for item in data:
                    if isinstance(item, dict):
                        box = item.get("box", item.get("poly", item.get("bbox")))
                        text = item.get("text", item.get("content", ""))
                        if box and isinstance(box, (list, tuple)):
                            pts = []
                            for p in box:
                                if isinstance(p, (list, tuple)) and len(p) >= 2:
                                    pts.append((int(p[0]), int(p[1])))
                            if len(pts) >= 2:
                                if len(pts) == 2:
                                    pts = [(pts[0][0], pts[0][1]), (pts[1][0], pts[0][1]), (pts[1][0], pts[1][1]), (pts[0][0], pts[1][1])]
                                out.append((pts, str(text)))
                if out:
                    return out
        except (json.JSONDecodeError, TypeError):
            pass

    # Regex: "[[x1,y1,x2,y2,...]], \"text\"" or "[x1,y1,x2,y2]: text"
    for m in re.finditer(r'\[[\d\s,]+\][\s:,\-]*["\']?([^"\']*)["\']?', raw):
        box_str = m.group(0).split(']')[0] + ']'
        text = m.group(1).strip().strip('"\'')
        try:
            coords = json.loads(box_str)
            if isinstance(coords, list) and len(coords) >= 4:
                arr = np.array(coords, dtype=np.int32)
                if arr.ndim == 1:
                    if len(arr) >= 4:
                        x1, y1 = int(arr[0]), int(arr[1])
                        x2, y2 = int(arr[2]), int(arr[3])
                        pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                        out.append((pts, text))
                elif arr.ndim == 2 and arr.shape[0] >= 2:
                    pts = [(int(arr[i, 0]), int(arr[i, 1])) for i in range(min(4, len(arr)))]
                    if len(pts) == 2:
                        pts = [(pts[0][0], pts[0][1]), (pts[1][0], pts[0][1]), (pts[1][0], pts[1][1]), (pts[0][0], pts[1][1])]
                    out.append((pts, text))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    return out


if _HUNYUAN_AVAILABLE:

    @register_textdetectors("hunyuan_ocr_det")
    class HunyuanOCRDetector(TextDetectorBase):
        """
        HunyuanOCR as detector: full-image spotting (detection + recognition in one run).
        Returns boxes and text when parseable. Use with none_ocr to keep spotter text; or any OCR to re-recognize crops.
        """
        params = {
            "model_name": {
                "type": "line_editor",
                "value": "tencent/HunyuanOCR",
                "description": "Hugging Face model id (same as HunyuanOCR OCR).",
            },
            "device": DEVICE_SELECTOR(),
            "max_new_tokens": {
                "type": "line_editor",
                "value": 2048,
                "description": "Max tokens for full-page spotting (increase for dense pages).",
            },
            "use_bf16": {
                "type": "checkbox",
                "value": True,
                "description": "Use bfloat16 when available.",
            },
            "description": "HunyuanOCR full-image spotting (det + rec). Pair with none_ocr to keep text.",
        }
        _load_model_keys = {"processor", "model"}

        SPOTTING_PROMPT_CN = "检测并识别图片中的文字，将文本坐标格式化输出。"
        SPOTTING_PROMPT_EN = "Detect and recognize text in the image, output text with coordinates in JSON format."

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
                        except (OSError, AttributeError, Exception) as dyn_e:
                            self.logger.warning(f"HunyuanOCR dynamic config load failed: {dyn_e}")
                            raise ValueError(
                                "The tencent/HunyuanOCR model uses architecture 'hunyuan_vl', which requires "
                                "transformers 4.49 or newer. Upgrade with:\n"
                                "  pip install --upgrade 'transformers>=4.49'\n"
                                "or install from source:\n"
                                "  pip install git+https://github.com/huggingface/transformers.git"
                            ) from e
                    else:
                        raise ValueError(
                            "The tencent/HunyuanOCR model uses architecture 'hunyuan_vl', which requires "
                            "transformers 4.49 or newer. Upgrade with:\n"
                            "  pip install --upgrade 'transformers>=4.49'\n"
                            "or install from source:\n"
                            "  pip install git+https://github.com/huggingface/transformers.git"
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
                    # Load model class from code repo and weights from model_name
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
                                        # Fallback: initialize the attributes post_init usually sets so later
                                        # loading steps (e.g. _adjust_tied_keys_with_tied_pointers) don't crash.
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
                    self.logger.warning(f"HunyuanOCR detector load failed: {model_err}; trying default dtype.")
                    self.model = AutoModelForImageTextToText.from_pretrained(
                        model_name, **load_kw
                    )
            except Exception as e:
                self.logger.warning(f"HunyuanOCR detector load failed: {e}; trying default dtype.")
                self.model = AutoModelForImageTextToText.from_pretrained(
                    model_name, **load_kw
                )
            self.model.to(self.device)

        def _run_spotting(self, pil_img: "Image.Image") -> str:
            tmp_path = None
            try:
                fd, tmp_path = tempfile.mkstemp(suffix=".png")
                os.close(fd)
                pil_img.save(tmp_path)
                messages = [
                    {"role": "system", "content": ""},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": tmp_path},
                            {"type": "text", "text": self.SPOTTING_PROMPT_CN},
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
                max_tokens = 2048
                mt = self.params.get("max_new_tokens", {})
                if isinstance(mt, dict):
                    try:
                        max_tokens = max(512, min(8192, int(mt.get("value", 2048))))
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
                self.logger.warning(f"HunyuanOCR spotting failed: {e}")
                return ""
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

        def _detect(self, img: np.ndarray, proj: ProjImgTrans = None) -> Tuple[np.ndarray, List[TextBlock]]:
            h, w = img.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            blk_list: List[TextBlock] = []
            if img.ndim == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            pil_img = _cv2_to_pil_rgb(img)
            raw = self._run_spotting(pil_img)
            if not raw:
                # Spotting failed on this environment (e.g. sparse permute issue) – fall back to
                # a single full-page box so downstream OCR can still run.
                full_pts = [(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)]
                blk = TextBlock(xyxy=[0, 0, w - 1, h - 1], lines=[full_pts])
                blk.language = "unknown"
                blk._detected_font_size = max(h // 20, 12)
                blk.text = [""]
                blk_list.append(blk)
                cv2.fillPoly(mask, [np.array(full_pts, dtype=np.int32)], 255)
                return mask, blk_list
            parsed = _parse_spotting_output(raw)
            for pts, text in parsed:
                if len(pts) < 3:
                    continue
                arr = np.array(pts, dtype=np.int32)
                x1 = int(arr[:, 0].min())
                y1 = int(arr[:, 1].min())
                x2 = int(arr[:, 0].max())
                y2 = int(arr[:, 1].max())
                if x2 <= x1 or y2 <= y1:
                    continue
                blk = TextBlock(xyxy=[x1, y1, x2, y2], lines=[pts])
                blk.language = "unknown"
                blk._detected_font_size = max(y2 - y1, 12)
                blk.text = [text] if text else [""]
                blk_list.append(blk)
                cv2.fillPoly(mask, [arr], 255)
            return mask, blk_list

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
