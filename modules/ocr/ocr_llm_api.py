import re
import time
import base64
import json
import cv2
import numpy as np
from typing import List, Optional

from openai import OpenAI
import httpx

from .base import register_OCR, OCRBase, TextBlock


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
        "OAI: gpt-4",
        "OAI: gpt-3.5-turbo",
        "GGL: gemini-1.5-pro-latest",
        "GGL: gemini-2.0-flash-exp",
        "GGL: gemini-2.0-flash",
    ]

    params = {
        "provider": {
            "type": "selector",
            "options": ["OpenAI", "Google"],
            "value": "OpenAI",
            "description": "Select the LLM provider.",
        },
        "api_key": {"value": "", "description": "Your API key."},
        "endpoint": {
            "value": "",
            "description": "Base URL for the API. Leave empty for provider default.",
        },
        "model": {
            "type": "selector",
            "options": popular_models,
            "value": "OAI: gpt-4o-mini",
            "description": "Select the model to use. (Provider prefix indicates the provider).",
        },
        "override_model": {
            "value": "",
            "description": "Specify a custom model name to override the selected model.",
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
            "description": "Controls image detail level for vision models (e.g., OpenAI, Google). Affects token usage and accuracy.",
        },
        "prompt": {
            "type": "editor",
            "value": "Perform OCR on the provided manga image snippet. The language is **{language}**.\nRecognize all text, including handwritten sound effects (SFX).\n**CRITICAL INSTRUCTION:** If you see jumbled characters, it is likely vertical text that was read horizontally. First, mentally reconstruct the correct vertical text.\n**OUTPUT FORMATTING:** All recognized text from the image must be consolidated into a **single, continuous horizontal line**. Do not use newlines.\nYour final output must be ONLY the recognized text. No explanations.",
            "description": "The main prompt for the OCR task. Use {language} placeholder.",
        },
        "system_prompt": {
            "type": "editor",
            "value": "You are a specialized OCR engine for manga and comics. Your primary function is to accurately extract and consolidate all recognized text from an image into a **single, continuous horizontal line**. You must return only the raw, recognized text. You do not interpret, translate, or explain the content. You are designed to intelligently handle common OCR errors, such as reconstructing jumbled characters that result from misreading vertical text.",
            "description": "Optional system prompt to guide the model's behavior.",
        },
        "proxy": {
            "value": "",
            "description": "Proxy address (e.g., http(s)://user:password@host:port)",
        },
        "delay": {"value": 1.0, "description": "Delay in seconds between requests."},
        "requests_per_minute": {
            "value": 15,
            "description": "Maximum number of requests per minute (0 for no limit).",
        },
        "max_response_tokens": {
            "value": 10240,
            "description": "Maximum number of tokens in the LLM's response.",
        },
        "description": "OCR using various LLMs compatible with the OpenAI API.",
    }

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.last_request_time = 0
        self.client = None
        self.request_count_minute = 0
        self.minute_start_time = time.time()

    def _initialize_client(self):
        transport = None
        if self.proxy:
            try:
                if self.debug_mode:
                    self.logger.info(f"Using proxy: {self.proxy}")
                transport = httpx.HTTPTransport(proxy=self.proxy)
            except Exception as e:
                self.logger.error(
                    f"Invalid proxy configuration: {self.proxy}. Error: {e}"
                )
                transport = None

        endpoint = self.endpoint
        provider = self.provider
        if not endpoint:
            if provider == "OpenAI":
                endpoint = "https://api.openai.com/v1"
            elif provider == "Google":
                endpoint = "https://generativelanguage.googleapis.com/v1beta/openai"
            else:
                endpoint = "https://api.openai.com/v1"

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=endpoint,
            http_client=httpx.Client(transport=transport) if transport else None,
        )
        if self.debug_mode:
            self.logger.info("LLM client initialized.")

    @property
    def provider(self):
        return self.get_param_value("provider")

    @property
    def request_delay(self):
        try:
            return float(self.get_param_value("delay"))
        except (ValueError, TypeError):
            return 1.0

    @property
    def api_key(self):
        return self.get_param_value("api_key")

    @property
    def endpoint(self):
        return self.get_param_value("endpoint")

    @property
    def model(self):
        return self.get_param_value("model")

    @property
    def override_model(self):
        return self.get_param_value("override_model")

    @property
    def language(self):
        lang_name = self.get_param_value("language")
        return self.lang_map.get(lang_name)

    @property
    def detail_level(self):
        return self.get_param_value("detail_level")

    @property
    def prompt(self):
        return self.get_param_value("prompt")

    @property
    def system_prompt(self):
        return self.get_param_value("system_prompt")

    @property
    def proxy(self):
        return self.get_param_value("proxy")

    @property
    def requests_per_minute(self):
        return int(self.get_param_value("requests_per_minute"))

    @property
    def max_response_tokens(self):
        try:
            return int(self.get_param_value("max_response_tokens"))
        except (ValueError, TypeError):
            return 4096

    def _respect_delay(self):
        current_time = time.time()
        if self.requests_per_minute > 0:
            if current_time - self.minute_start_time >= 60:
                self.request_count_minute = 0
                self.minute_start_time = current_time
            if self.request_count_minute >= self.requests_per_minute:
                wait_time = 62 - (current_time - self.minute_start_time)
                if wait_time > 0:
                    if self.debug_mode:
                        self.logger.info(
                            f"Reached request limit. Waiting {wait_time:.2f} seconds."
                        )
                    time.sleep(wait_time)
                self.request_count_minute = 0
                self.minute_start_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if self.debug_mode:
            self.logger.info(
                f"Time since last request: {time_since_last_request} seconds"
            )
        if time_since_last_request < self.request_delay:
            sleep_time = self.request_delay - time_since_last_request
            if self.debug_mode:
                self.logger.info(f"Waiting {sleep_time} seconds before next request")
            time.sleep(sleep_time)
        self.last_request_time = time.time()
        if self.requests_per_minute > 0:
            self.request_count_minute += 1

    def ocr(self, img_base64: str, prompt_override: str = None) -> str:
        if self.client is None:
            if self.debug_mode:
                self.logger.debug("Client is not initialized. Initializing now.")
            self._initialize_client()
        if self.debug_mode:
            self.logger.debug(f"Starting OCR on image")
        self._respect_delay()
        try:
            lang_name = self.get_param_value("language")
            prompt_text = (prompt_override if prompt_override else self.prompt).format(
                language=lang_name
            )

            image_content_part = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
            }

            provider = self.provider
            if provider in ["OpenAI", "Google"]:
                detail_setting = self.detail_level
                if detail_setting in ["low", "high"]:
                    image_content_part["image_url"]["detail"] = detail_setting
                    if self.debug_mode:
                        self.logger.debug(
                            f"Adding '{provider}' parameter: detail={detail_setting}"
                        )

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

            model_name = (
                self.override_model
                or self.model
                or ("gpt-4o-mini" if provider == "OpenAI" else "gemini-1.5-pro-latest")
            )
            if ": " in model_name:
                model_name = model_name.split(": ", 1)[1]
            if self.debug_mode:
                self.logger.info(f"Using model: {model_name}")

            request_payload = {
                "model": model_name,
                "messages": messages,
                "max_tokens": self.max_response_tokens,
            }

            response = self.client.chat.completions.create(**request_payload)

            if self.debug_mode:
                self.logger.debug(
                    f"Raw API response: {response.model_dump_json(indent=2)}"
                )
            if response.choices:
                full_text = response.choices[0].message.content
                if full_text is None:
                    if self.debug_mode:
                        self.logger.warning("OCR response content is None.")
                    return ""

                full_text = full_text.replace("\n", " ").strip()

                if self.debug_mode:
                    self.logger.debug(f"OCR result: {full_text}")
                return full_text
            else:
                if self.debug_mode:
                    self.logger.warning("No text found in OCR response choices.")
                return ""
        except Exception as e:
            self.logger.error(f"OCR error: {e}")
            return ""

    def _ocr_blk_list(
        self, img: np.ndarray, blk_list: List[TextBlock], *args, **kwargs
    ):
        im_h, im_w = img.shape[:2]
        if self.debug_mode:
            self.logger.debug(f"Image dimensions: {im_h}x{im_w}")
        for blk in blk_list:
            x1, y1, x2, y2 = blk.xyxy
            if self.debug_mode:
                self.logger.debug(f"Processing block: ({x1}, {y1}, {x2}, {y2})")
            if (
                y2 <= im_h
                and x2 <= im_w
                and x1 >= 0
                and y1 >= 0
                and x1 < x2
                and y1 < y2
            ):
                cropped_img = img[y1:y2, x1:x2]
                _, buffer = cv2.imencode(".jpg", cropped_img)
                img_base64 = base64.b64encode(buffer).decode("utf-8")
                if self.debug_mode:
                    self.logger.debug(f"Cropped image dimensions: {cropped_img.shape}")
                blk.text = self.ocr(
                    img_base64, prompt_override=kwargs.get("prompt", "")
                )
            else:
                if self.debug_mode:
                    self.logger.warning("Invalid text block coordinates")
                blk.text = ""

    def ocr_img(self, img: np.ndarray, prompt: str = "") -> str:
        _, buffer = cv2.imencode(".jpg", img)
        img_base64 = base64.b64encode(buffer).decode("utf-8")
        return self.ocr(img_base64, prompt_override=prompt)

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        if param_key in [
            "api_key",
            "endpoint",
            "proxy",
            "provider",
            "model",
            "override_model",
            "max_response_tokens",
        ]:
            self.client = None
        if param_key in ["requests_per_minute", "delay"]:
            current_time = time.time()
            self.request_count_minute = 0
            self.minute_start_time = current_time
            self.last_request_time = current_time
