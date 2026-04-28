import re
import time
import base64
import json
import cv2
import numpy as np
from typing import List, Optional

import openai
import httpx

from .base import register_OCR, OCRBase, TextBlock
from utils.ocr_preprocess import preprocess_for_ocr


@register_OCR("llm_ocr")
class LLM_OCR(OCRBase):
    lang_map = {
        "Auto Detect": None,
        "Afrikaans": "af",
        "Albanian": "sq",
        "Amharic": "am",
        "Arabic": "ar",
        "Armenian": "hy",
        "Assamese": "as",
        "Azerbaijani": "az",
        "Bangla": "bn",
        "Basque": "eu",
        "Belarusian": "be",
        "Bengali": "bn",
        "Bosnian": "bs",
        "Breton": "br",
        "Bulgarian": "bg",
        "Burmese": "my",
        "Catalan": "ca",
        "Cebuano": "ceb",
        "Cherokee": "chr",
        "Chinese (Simplified)": "zh-CN",
        "Chinese (Traditional)": "zh-TW",
        "Corsican": "co",
        "Croatian": "hr",
        "Czech": "cs",
        "Danish": "da",
        "Dutch": "nl",
        "English": "en",
        "Esperanto": "eo",
        "Estonian": "et",
        "Faroese": "fo",
        "Filipino": "fil",
        "Finnish": "fi",
        "French": "fr",
        "Frisian": "fy",
        "Galician": "gl",
        "Georgian": "ka",
        "German": "de",
        "Greek": "el",
        "Gujarati": "gu",
        "Haitian Creole": "ht",
        "Hausa": "ha",
        "Hawaiian": "haw",
        "Hebrew": "he",
        "Hindi": "hi",
        "Hmong": "hmn",
        "Hungarian": "hu",
        "Icelandic": "is",
        "Igbo": "ig",
        "Indonesian": "id",
        "Interlingua": "ia",
        "Irish": "ga",
        "Italian": "it",
        "Japanese": "ja",
        "Javanese": "jv",
        "Kannada": "kn",
        "Kazakh": "kk",
        "Khmer": "km",
        "Korean": "ko",
        "Kurdish": "ku",
        "Kyrgyz": "ky",
        "Lao": "lo",
        "Latin": "la",
        "Latvian": "lv",
        "Lithuanian": "lt",
        "Luxembourgish": "lb",
        "Macedonian": "mk",
        "Malagasy": "mg",
        "Malay": "ms",
        "Malayalam": "ml",
        "Maltese": "mt",
        "Maori": "mi",
        "Marathi": "mr",
        "Mongolian": "mn",
        "Nepali": "ne",
        "Norwegian": "no",
        "Occitan": "oc",
        "Oriya": "or",
        "Pashto": "ps",
        "Persian": "fa",
        "Polish": "pl",
        "Portuguese": "pt",
        "Punjabi": "pa",
        "Quechua": "qu",
        "Romanian": "ro",
        "Russian": "ru",
        "Samoan": "sm",
        "Scots Gaelic": "gd",
        "Serbian (Cyrillic)": "sr-Cyrl",
        "Serbian (Latin)": "sr-Latn",
        "Shona": "sn",
        "Sindhi": "sd",
        "Sinhala": "si",
        "Slovak": "sk",
        "Slovenian": "sl",
        "Somali": "so",
        "Spanish": "es",
        "Sundanese": "su",
        "Swahili": "sw",
        "Swedish": "sv",
        "Tagalog": "tl",
        "Tajik": "tg",
        "Tamil": "ta",
        "Tatar": "tt",
        "Telugu": "te",
        "Thai": "th",
        "Tibetan": "bo",
        "Tigrinya": "ti",
        "Tongan": "to",
        "Turkish": "tr",
        "Ukrainian": "uk",
        "Urdu": "ur",
        "Uyghur": "ug",
        "Uzbek": "uz",
        "Vietnamese": "vi",
        "Welsh": "cy",
        "Xhosa": "xh",
        "Yiddish": "yi",
        "Yoruba": "yo",
        "Zulu": "zu",
    }

    popular_models = [
        "OAI: gpt-4o-mini",
        "OAI: gpt-4-vision-preview",
        "OAI: gpt-4o",
        "OAI: gpt-4",
        "GGL: gemini-1.5-pro-latest",
        "GGL: gemini-1.5-flash-latest",
        # OpenRouter vision models (use with provider=OpenRouter; get key at openrouter.ai)
        # Free vision models (image in, text out, $0): https://openrouter.ai/models?fmt=cards&input_modalities=image&max_price=0&output_modalities=text
        "OpenRouter: openrouter/free",
        "OpenRouter: google/gemma-3-4b-it:free",
        "OpenRouter: google/gemma-3-12b-it:free",
        "OpenRouter: google/gemma-3-27b-it:free",
        "OpenRouter: mistralai/mistral-small-3.1-24b-instruct:free",
        "OpenRouter: nvidia/nemotron-nano-12b-v2-vl:free",
        "OpenRouter: qwen/qwen3-vl-30b-a3b-thinking",
        "OpenRouter: qwen/qwen3-vl-235b-a22b-thinking",
        # Paid OpenRouter vision models
        "OpenRouter: openai/gpt-4o",
        "OpenRouter: openai/gpt-4o-mini",
        "OpenRouter: google/gemini-2.0-flash-001",
        "OpenRouter: google/gemini-2.0-flash-exp",
        "OpenRouter: google/gemini-1.5-flash",
        "OpenRouter: google/gemini-1.5-pro",
        "OpenRouter: qwen/qwen2.5-vl-72b-instruct",
        "OpenRouter: qwen/qwen3.5-flash-02-23",
        "OpenRouter: anthropic/claude-sonnet-4",
        "OpenRouter: anthropic/claude-3-5-sonnet",
        "OpenRouter: meta-llama/llama-3.2-11b-vision-instruct",
        "OLL: (override model field)",
    ]

    params = {
        "provider": {
            "type": "selector",
            "options": ["OpenAI", "Google", "OpenRouter", "Ollama"],
            "value": "OpenAI",
            "description": "Select the LLM provider.",
        },
        "api_key": {
            "value": "",
            "description": "API key to use if multiple keys are not provided.",
        },
        "multiple_keys": {
            "type": "editor",
            "value": "",
            "description": "API keys separated by semicolons (;). Requests will rotate.",
        },
        "endpoint": {
            "value": "",
            "description": "Base URL for the API. Leave empty for provider default.",
        },
        "model": {
            "type": "selector",
            "options": popular_models,
            "value": "OAI: gpt-4o-mini",
            "description": "Select the model to use.",
        },
        "override_model": {
            "value": "",
            "description": "Specify a custom model name to override the selected one.",
        },
        "language": {
            "type": "selector",
            "options": list(lang_map.keys()),
            "value": "Japanese",
            "description": "Language for OCR.",
        },
        "detail_level": {
            "type": "selector",
            "options": ["auto", "low", "high"],
            "value": "auto",
            "description": "Controls image detail level for vision models.",
        },
        "upscale_min_side": {
            "type": "line_editor",
            "value": 0,
            "description": "If >0, upscale crop so longer side >= this before sending to VLM (e.g. 512). Helps tiny text. 0 = use global default from Config.",
        },
        "prompt": {
            "type": "editor",
            "value": "Perform OCR on the provided manga image snippet. The language is **{language}**.\nRecognize all text, including handwritten sound effects (SFX).\n**CRITICAL INSTRUCTION:** If you see jumbled characters, it is likely vertical text that was read horizontally. First, mentally reconstruct the correct vertical text.\n**OUTPUT FORMATTING:** All recognized text from the image must be consolidated into a **single, continuous horizontal line**. Do not use newlines.\nYour final output must be ONLY the recognized text. No explanations.",
            "description": "The main prompt for the OCR task. Use {language} placeholder.",
        },
        "system_prompt": {
            "type": "editor",
            "value": (
                "You are a specialized OCR engine for manga and comics.\n"
                "- Transcribe all text exactly. Ignore ruby/furigana readings and emphasis markers as separate content; output the main text only.\n"
                "- Output a single continuous horizontal line per image. No line numbers or labels unless you use strict numbered format (1: ... 2: ...) when multiple segments are requested.\n"
                "- Return only the raw recognized text, no explanations."
            ),
            "description": "Section 18: Dedicated OCR transcription prompt. Optional; leave empty to use the default above.",
        },
        "proxy": {
            "value": "",
            "description": "Proxy address (e.g., http(s)://user:password@host:port)",
        },
        "delay": {"value": 1.0, "description": "Delay in seconds between requests."},
        "requests_per_minute": {
            "value": 15,
            "description": "Maximum number of requests per minute per key.",
        },
        "max_response_tokens": {
            "value": 4096,
            "description": "Maximum number of tokens in the LLM's response.",
        },
        "temperature": {
            "value": 0.0,
            "description": "Sampling temperature. Set to 0 for deterministic OCR/translate output.",
        },
        "translate_prompt": {
            "type": "editor",
            "value": (
                "Translate the text in this comic/manga image into {target_language}. "
                "Return ONLY the translated text (no quotes, no markdown). Keep line breaks if the text clearly has multiple lines."
            ),
            "description": "Prompt used for one-step image->translation when translation_mode=one_step_vlm.",
        },
        "description": "OCR using various vision-capable LLMs.",
    }

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.last_request_time = 0
        self.client = None
        self.request_count_minute = 0
        self.minute_start_time = time.time()
        self.key_usage = {}
        self.current_key_index = 0

    def _initialize_client(self, api_key_to_use: str):
        endpoint = self.endpoint
        provider = self._effective_provider_for_model()
        if not endpoint:
            if provider == "OpenAI":
                endpoint = "https://api.openai.com/v1"
            elif provider == "Google":
                endpoint = "https://generativelanguage.googleapis.com/v1beta/openai"
            elif provider == "OpenRouter":
                endpoint = "https://openrouter.ai/api/v1"
            elif provider == "Ollama":
                endpoint = "http://localhost:11434/v1"

        http_client = None
        if self.proxy:
            try:
                proxy_mounts = {"all://": httpx.HTTPTransport(proxy=self.proxy)}
                http_client = httpx.Client(mounts=proxy_mounts)
            except Exception as e:
                self.logger.error(f"Failed to initialize proxy '{self.proxy}': {e}.")

        masked_key = (
            api_key_to_use[:4] + "..." + api_key_to_use[-4:]
            if len(api_key_to_use) > 8
            else api_key_to_use
        )
        self.logger.debug(
            f"Initializing client for {provider} with key {masked_key} at endpoint {endpoint}"
        )

        self.client = openai.OpenAI(
            api_key=api_key_to_use, base_url=endpoint, http_client=http_client
        )

    def _effective_provider_for_model(self) -> str:
        try:
            if self.override_model:
                return self.provider
            m = (self.model or "").strip()
            if ": " in m:
                prefix = m.split(": ", 1)[0].strip().upper()
                if prefix == "OAI":
                    return "OpenAI"
                if prefix == "GGL":
                    return "Google"
                if prefix == "OPENROUTER":
                    return "OpenRouter"
                if prefix == "OLL":
                    return "Ollama"
            return self.provider
        except Exception:
            return self.provider

    # --- Property Getters (similar to translator) ---
    @property
    def provider(self) -> str:
        return self.get_param_value("provider")

    @property
    def api_key(self) -> str:
        return self.get_param_value("api_key")

    @property
    def multiple_keys_list(self) -> List[str]:
        keys_str = self.get_param_value("multiple_keys")
        if not isinstance(keys_str, str):
            return []
        return [
            key.strip()
            for key in keys_str.strip().replace("\n", ";").split(";")
            if key.strip()
        ]

    @property
    def endpoint(self) -> Optional[str]:
        return self.get_param_value("endpoint") or None

    @property
    def model(self) -> str:
        return self.get_param_value("model")

    @property
    def override_model(self) -> Optional[str]:
        return self.get_param_value("override_model") or None

    @property
    def language(self) -> str:
        return self.get_param_value("language")

    @property
    def detail_level(self) -> str:
        return self.get_param_value("detail_level")

    @property
    def prompt(self) -> str:
        return self.get_param_value("prompt")

    @property
    def system_prompt(self) -> str:
        return self.get_param_value("system_prompt")

    @property
    def proxy(self) -> str:
        return self.get_param_value("proxy")

    @property
    def requests_per_minute(self) -> int:
        return int(self.get_param_value("requests_per_minute"))

    @property
    def max_response_tokens(self) -> int:
        return int(self.get_param_value("max_response_tokens"))

    @property
    def request_delay(self) -> float:
        try:
            return float(self.get_param_value("delay"))
        except (ValueError, TypeError):
            return 1.0

    @property
    def temperature(self) -> float:
        try:
            return float(self.get_param_value("temperature"))
        except (ValueError, TypeError):
            return 0.0

    def _respect_delay(self):
        # This logic is identical to the one in LLM_API_Translator
        current_time = time.time()
        rpm = self.requests_per_minute
        if rpm > 0:
            if current_time - self.minute_start_time >= 60:
                self.request_count_minute = 0
                self.minute_start_time = current_time
            if self.request_count_minute >= rpm:
                wait_time = 60.1 - (current_time - self.minute_start_time)
                if wait_time > 0:
                    self.logger.warning(
                        f"Global RPM limit ({rpm}) reached. Waiting {wait_time:.2f}s."
                    )
                    time.sleep(wait_time)
                self.request_count_minute = 0
                self.minute_start_time = time.time()

        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.request_delay:
            sleep_time = self.request_delay - time_since_last_request
            if self.debug_mode:
                self.logger.debug(f"Global delay: Waiting {sleep_time:.3f}s.")
            time.sleep(sleep_time)

        self.last_request_time = time.time()
        self.request_count_minute += 1

    def _respect_key_limit(self, key: str) -> bool:
        # This logic is identical to the one in LLM_API_Translator
        rpm = self.requests_per_minute
        if rpm <= 0:
            return True
        now = time.time()
        count, start_time = self.key_usage.get(key, (0, now))
        if now - start_time >= 60:
            count, start_time = 0, now
        if count >= rpm:
            wait_time = 60.1 - (now - start_time)
            if wait_time > 0:
                self.logger.warning(
                    f"RPM limit ({rpm}) for key {key[:6]}... reached. Waiting {wait_time:.2f}s."
                )
                time.sleep(wait_time)
            self.key_usage[key] = (0, time.time())
            return False
        return True

    def _select_api_key(self) -> Optional[str]:
        # This logic is identical to the one in LLM_API_Translator
        if self._effective_provider_for_model() == "Ollama":
            return "local-llm"
        api_keys = self.multiple_keys_list
        single_key = self.api_key
        if not api_keys and not single_key:
            self.logger.error("No API keys provided.")
            return None

        if not api_keys:
            if self._respect_key_limit(single_key):
                now = time.time()
                count, start_time = self.key_usage.get(single_key, (0, now))
                self.key_usage[single_key] = (count + 1, start_time)
                return single_key
            return None

        start_index = self.current_key_index
        for i in range(len(api_keys)):
            index = (start_index + i) % len(api_keys)
            key = api_keys[index]
            if self._respect_key_limit(key):
                now = time.time()
                count, start_time = self.key_usage.get(key, (0, now))
                self.key_usage[key] = (count + 1, start_time)
                self.current_key_index = (index + 1) % len(api_keys)
                return key
        self.logger.error("All API keys are rate-limited.")
        return None

    def ocr(self, img_base64: str, prompt_override: str = None) -> str:
        provider = self._effective_provider_for_model()
        api_key_to_use = self._select_api_key()
        if not api_key_to_use and provider != "Ollama":
            return "[ERROR: No available API key]"

        # Re-initialize client if key is different from the last one used
        if not self.client or self.client.api_key != api_key_to_use:
            self._initialize_client(api_key_to_use)

        self._respect_delay()
        try:
            lang_name = self.language
            prompt_text = (prompt_override or self.prompt).format(language=lang_name)

            image_content_part = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
            }

            if provider in ["OpenAI", "Google", "OpenRouter", "Ollama"]:
                detail_setting = self.detail_level
                if detail_setting in ["low", "high"]:
                    image_content_part["image_url"]["detail"] = detail_setting

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        image_content_part,
                    ],
                }
            ]
            if self.system_prompt:
                messages.insert(0, {"role": "system", "content": self.system_prompt})

            model_name = self.override_model or self.model
            if ": " in model_name:
                model_name = model_name.split(": ", 1)[1]

            self.logger.debug(f"OCR request with model: {model_name}")

            response = self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=self.max_response_tokens,
                temperature=self.temperature,
            )

            if response.choices and response.choices[0].message.content:
                full_text = (
                    response.choices[0].message.content.replace("\n", " ").strip()
                )
                self.logger.debug(f"OCR result: {full_text}")
                return full_text
            else:
                self.logger.warning("No text found in OCR response.")
                return ""
        except Exception as e:
            self.logger.error(f"OCR error: {e}")
            return f"[ERROR: {type(e).__name__}]"

    def run_ocr_translate(self, img: np.ndarray, blk_list: List[TextBlock], source_lang: str, target_lang: str, *args, **kwargs) -> List[TextBlock]:
        """
        One-step VLM translation: fill blk.translation directly from image crops.
        """
        if img.ndim == 3 and img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        im_h, im_w = img.shape[:2]
        upscale_min = 0
        try:
            upscale_min = int(self.get_param_value("upscale_min_side") or 0)
        except (TypeError, ValueError):
            pass
        if upscale_min <= 0:
            try:
                from utils.config import pcfg
                upscale_min = int(getattr(pcfg.module, "ocr_upscale_min_side", 0) or 0)
            except Exception:
                pass
        prompt_tmpl = (self.get_param_value("translate_prompt") or "").strip()
        if not prompt_tmpl:
            prompt_tmpl = "Translate the text in this image into {target_language}. Return only the translated text."
        prompt = prompt_tmpl.format(target_language=str(target_lang))
        for blk in blk_list:
            x1, y1, x2, y2 = blk.xyxy
            x1 = max(0, min(int(round(float(x1))), im_w - 1))
            y1 = max(0, min(int(round(float(y1))), im_h - 1))
            x2 = max(x1 + 1, min(int(round(float(x2))), im_w))
            y2 = max(y1 + 1, min(int(round(float(y2))), im_h))
            if 0 <= x1 < x2 <= im_w and 0 <= y1 < y2 <= im_h:
                cropped_img = img[y1:y2, x1:x2]
                cropped_img = preprocess_for_ocr(cropped_img, recipe="none", upscale_min_side=upscale_min)
                try:
                    ok, buffer = cv2.imencode(".jpg", cropped_img)
                except Exception:
                    ok, buffer = False, None
                if not ok or buffer is None:
                    blk.translation = ""
                    continue
                img_base64 = base64.b64encode(buffer).decode("utf-8")
                result = self.ocr(img_base64, prompt_override=prompt)
                blk.translation = result or ""
            else:
                blk.translation = ""
        return blk_list

    def _ocr_blk_list(
        self, img: np.ndarray, blk_list: List[TextBlock], *args, **kwargs
    ):
        im_h, im_w = img.shape[:2]
        upscale_min = 0
        try:
            upscale_min = int(self.get_param_value("upscale_min_side") or 0)
        except (TypeError, ValueError):
            pass
        if upscale_min <= 0:
            try:
                from utils.config import pcfg
                upscale_min = int(getattr(pcfg.module, "ocr_upscale_min_side", 0) or 0)
            except Exception:
                pass
        for blk in blk_list:
            x1, y1, x2, y2 = blk.xyxy
            x1 = max(0, min(int(round(float(x1))), im_w - 1))
            y1 = max(0, min(int(round(float(y1))), im_h - 1))
            x2 = max(x1 + 1, min(int(round(float(x2))), im_w))
            y2 = max(y1 + 1, min(int(round(float(y2))), im_h))
            if 0 <= x1 < x2 <= im_w and 0 <= y1 < y2 <= im_h:
                cropped_img = img[y1:y2, x1:x2]
                cropped_img = preprocess_for_ocr(cropped_img, recipe="none", upscale_min_side=upscale_min)
                _, buffer = cv2.imencode(".jpg", cropped_img)
                img_base64 = base64.b64encode(buffer).decode("utf-8")
                result = self.ocr(img_base64, prompt_override=kwargs.get("prompt"))
                blk.text = [result] if result else [""]
            else:
                blk.text = [""]

    def ocr_img(self, img: np.ndarray, prompt: str = "") -> str:
        _, buffer = cv2.imencode(".jpg", img)
        img_base64 = base64.b64encode(buffer).decode("utf-8")
        return self.ocr(img_base64, prompt_override=prompt)

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        if param_key in ["api_key", "multiple_keys", "endpoint", "proxy", "provider"]:
            self.client = None  # Force re-initialization on next call
        if param_key in ["requests_per_minute", "delay"]:
            self.request_count_minute = 0
            self.minute_start_time = time.time()
            self.last_request_time = 0
