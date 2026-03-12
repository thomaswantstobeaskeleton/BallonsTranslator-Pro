# Qwen-MT: Alibaba's machine translation API (DashScope, OpenAI-compatible).
# Models: qwen-mt-plus (best quality), qwen-mt-flash (balanced), qwen-mt-lite (fast, 31 langs).
# API: https://dashscope-intl.aliyuncs.com/compatible-mode/v1 (or Beijing region).
# Set DASHSCOPE_API_KEY or use the API key param.

import os
from typing import List, Dict

import openai

from .base import BaseTranslator, register_translator
from .exceptions import MissingTranslatorParams


DEFAULT_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
DEFAULT_BASE_URL_CN = "https://dashscope.aliyuncs.com/compatible-mode/v1"


@register_translator("Qwen_MT")
class QwenMTTranslator(BaseTranslator):
    concate_text = False
    cht_require_convert = True
    params: Dict = {
        "api_key": {
            "value": "",
            "description": "DashScope API key (Alibaba Model Studio). Or set env DASHSCOPE_API_KEY.",
        },
        "model": {
            "type": "selector",
            "options": [
                "qwen-mt-plus",
                "qwen-mt-flash",
                "qwen-mt-lite",
            ],
            "value": "qwen-mt-flash",
            "description": "qwen-mt-plus: best quality. qwen-mt-flash: balanced. qwen-mt-lite: fastest (31 langs).",
        },
        "base_url": {
            "type": "selector",
            "options": ["International (Singapore)", "China (Beijing)"],
            "value": "International (Singapore)",
            "description": "API region: International or China.",
        },
        "delay": {
            "value": 0.2,
            "description": "Delay in seconds between requests.",
        },
    }

    def _setup_translator(self):
        # Map app language names to Qwen-MT API language names (92 languages supported)
        self.lang_map["简体中文"] = "Simplified Chinese"
        self.lang_map["繁體中文"] = "Traditional Chinese"
        self.lang_map["日本語"] = "Japanese"
        self.lang_map["English"] = "English"
        self.lang_map["한국어"] = "Korean"
        self.lang_map["Tiếng Việt"] = "Vietnamese"
        self.lang_map["Français"] = "French"
        self.lang_map["Español"] = "Spanish"
        self.lang_map["Deutsch"] = "German"
        self.lang_map["ไทย"] = "Thai"
        self.lang_map["Bahasa Indonesia"] = "Indonesian"
        self.lang_map["العربية"] = "Arabic"
        self.lang_map["русский язык"] = "Russian"
        self.lang_map["Italiano"] = "Italian"
        self.lang_map["Português"] = "Portuguese"
        self.lang_map["Polski"] = "Polish"
        self.lang_map["Türk dili"] = "Turkish"
        self.lang_map["Nederlands"] = "Dutch"
        self.lang_map["čeština"] = "Czech"
        self.lang_map["magyar nyelv"] = "Hungarian"
        self.lang_map["limba română"] = "Romanian"
        self.lang_map["украї́нська мо́ва"] = "Ukrainian"
        self.lang_map["Ελληνικά"] = "Greek"
        self.lang_map["Svenska"] = "Swedish"
        self.lang_map["Dansk"] = "Danish"
        self.lang_map["Norsk"] = "Norwegian"
        self.lang_map["Suomi"] = "Finnish"

    @property
    def api_key(self) -> str:
        key = self.get_param_value("api_key") or ""
        if not key:
            key = os.environ.get("DASHSCOPE_API_KEY", "")
        return key.strip()

    @property
    def model(self) -> str:
        return self.get_param_value("model") or "qwen-mt-flash"

    @property
    def base_url(self) -> str:
        region = self.get_param_value("base_url") or "International (Singapore)"
        return DEFAULT_BASE_URL_CN if "China" in region else DEFAULT_BASE_URL

    @property
    def delay(self) -> float:
        try:
            return float(self.get_param_value("delay") or 0.2)
        except (TypeError, ValueError):
            return 0.2

    def _translate(self, src_list: List[str]) -> List[str]:
        if not self.api_key:
            raise MissingTranslatorParams("Qwen-MT requires an API key. Set it in params or DASHSCOPE_API_KEY.")
        import time
        client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        source_lang = self.lang_map.get(self.lang_source) or "auto"
        target_lang = self.lang_map.get(self.lang_target)
        if not target_lang:
            target_lang = "English"
        results = []
        for i, text in enumerate(src_list):
            if self.delay > 0 and i > 0:
                time.sleep(self.delay)
            text = (text or "").strip()
            if not text:
                results.append("")
                continue
            try:
                completion = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": text}],
                    extra_body={
                        "translation_options": {
                            "source_lang": source_lang,
                            "target_lang": target_lang,
                        }
                    },
                )
                if completion.choices and completion.choices[0].message.content:
                    results.append(completion.choices[0].message.content.strip())
                else:
                    results.append("")
            except Exception as e:
                self.logger.error("Qwen-MT request failed: %s", e)
                results.append("")
        return results
