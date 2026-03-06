import time
from typing import Dict, List, Optional

import httpx

from .base import BaseTranslator, register_translator
from .exceptions import CriticalTranslationError
from utils.logger import logger as LOGGER


@register_translator("Cohere_Command_R+")
class CohereCommandRPlusTranslator(BaseTranslator):
    """
    Translator using Cohere Command R+ via the Chat API.

    API docs: https://docs.cohere.com/docs/command-r-plus
    Endpoint: POST https://api.cohere.ai/v2/chat
    """

    concate_text = False
    cht_require_convert = True

    params: Dict = {
        "api_key": {
            "value": "",
            "description": "Cohere API key (from https://coral.cohere.com/).",
        },
        "endpoint": {
            "value": "https://api.cohere.ai/v2/chat",
            "description": "Cohere Chat API endpoint. Leave default unless you use a proxy.",
        },
        "model": {
            "type": "line_editor",
            "value": "command-r-plus-08-2024",
            "description": "Cohere model ID (default Command R+ 08-2024).",
        },
        "temperature": {
            "value": 0.2,
            "description": "Sampling temperature (0–1). Lower = more literal translations.",
        },
        "max_tokens": {
            "value": 1024,
            "description": "Maximum output tokens per request.",
        },
        "delay": {
            "value": 0.3,
            "description": "Delay in seconds between requests to respect rate limits.",
        },
    }

    def _setup_translator(self):
        # Map UI language names to Cohere hint strings.
        self.lang_map = {
            "简体中文": "Simplified Chinese",
            "繁體中文": "Traditional Chinese",
            "日本語": "Japanese",
            "English": "English",
            "한국어": "Korean",
            "Tiếng Việt": "Vietnamese",
            "Français": "French",
            "Deutsch": "German",
            "Italiano": "Italian",
            "Português": "Portuguese",
            "Brazilian Portuguese": "Brazilian Portuguese",
            "Español": "Spanish",
            "русский язык": "Russian",
            "Türk dili": "Turkish",
            "Thai": "Thai",
            "Arabic": "Arabic",
            "Hindi": "Hindi",
        }

    @property
    def _api_key(self) -> str:
        key = (self.get_param_value("api_key") or "").strip()
        if not key:
            raise CriticalTranslationError(
                "Cohere API key is empty. Set it in Config → Translator."
            )
        return key

    @property
    def _endpoint(self) -> str:
        ep = (self.get_param_value("endpoint") or "").strip()
        if not ep:
            ep = "https://api.cohere.ai/v2/chat"
        return ep

    @property
    def _model(self) -> str:
        return (self.get_param_value("model") or "command-r-plus-08-2024").strip()

    @property
    def _temperature(self) -> float:
        try:
            return float(self.get_param_value("temperature") or 0.2)
        except (TypeError, ValueError):
            return 0.2

    @property
    def _max_tokens(self) -> int:
        try:
            v = int(self.get_param_value("max_tokens") or 1024)
            return max(64, min(4096, v))
        except (TypeError, ValueError):
            return 1024

    @property
    def _delay(self) -> float:
        try:
            return float(self.get_param_value("delay") or 0.3)
        except (TypeError, ValueError):
            return 0.3

    def _build_system_prompt(self) -> str:
        src = self.lang_map.get(self.lang_source, self.lang_source)
        tgt = self.lang_map.get(self.lang_target, self.lang_target)
        return (
            "You are an expert comic and manga translator.\n"
            f"Translate from {src} to {tgt}.\n"
            "Preserve the meaning and tone, but keep sentences natural and readable.\n"
            "Return ONLY the translated text, with line breaks mirrored from the input. "
            "Do not add explanations or extra commentary."
        )

    def _call_api(self, text: str) -> str:
        client = httpx.Client(timeout=60.0)
        try:
            payload = {
                "model": self._model,
                "temperature": self._temperature,
                "max_tokens": self._max_tokens,
                "messages": [
                    {"role": "system", "content": self._build_system_prompt()},
                    {"role": "user", "content": text},
                ],
            }
            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            }
            resp = client.post(self._endpoint, json=payload, headers=headers)
            if resp.status_code == 429:
                raise CriticalTranslationError(
                    f"Cohere rate limit (429). Try increasing delay or lowering page size. Response: {resp.text}"
                )
            if resp.status_code >= 400:
                raise CriticalTranslationError(
                    f"Cohere API error {resp.status_code}: {resp.text}"
                )
            data = resp.json()
            msg = data.get("message") or {}
            content = msg.get("content") or []
            if not content:
                raise CriticalTranslationError(
                    f"Cohere API returned empty content: {data}"
                )
            # content is a list of blocks; take first text block.
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    return block.get("text", "").strip()
            # Fallback: try first element as string.
            if isinstance(content[0], str):
                return content[0].strip()
            raise CriticalTranslationError(
                f"Unexpected Cohere response format: {data}"
            )
        except CriticalTranslationError:
            raise
        except Exception as e:
            LOGGER.error("Cohere Command R+ request failed: %s", e)
            raise CriticalTranslationError(str(e))
        finally:
            try:
                client.close()
            except Exception:
                pass

    def _translate(self, src_list: List[str]) -> List[str]:
        results: List[str] = []
        delay = self._delay
        for idx, block in enumerate(src_list):
            text = block or ""
            if not text.strip():
                results.append(text)
                continue
            if delay > 0 and idx > 0:
                time.sleep(delay)
            translated = self._call_api(text)
            results.append(translated)
        return results

