import re
import time
import json
import traceback
from typing import List, Dict, Optional, Type

import httpx
import openai
from pydantic import BaseModel, Field, ValidationError

from utils.proxy_utils import create_httpx_client
from utils.series_context_store import (
    get_series_context_dir,
    load_series_glossary,
    load_recent_context,
    append_page_to_series_context as store_append_page,
    merge_glossary_no_dupes,
)
from .base import BaseTranslator, register_translator
from .exceptions import CriticalTranslationError


class InvalidNumTranslations(Exception):
    """Exception raised when the number of translations does not match the number of sources."""

    pass


class TranslationElement(BaseModel):
    id: int = Field(..., description="The original numeric ID of the text snippet.")
    translation: str = Field(
        ..., description="The translated text corresponding to the id."
    )


class TranslationResponse(BaseModel):
    translations: List[TranslationElement] = Field(
        ..., description="A list of all translated elements."
    )


@register_translator("LLM_API_Translator")
class LLM_API_Translator(BaseTranslator):
    concate_text = False
    cht_require_convert = True
    params: Dict = {
        "provider": {
            "type": "selector",
            "options": ["OpenAI", "Google", "Grok", "OpenRouter", "LLM Studio"],
            "value": "OpenAI",
            "description": "Select the LLM provider.",
        },
        "apikey": {
            "value": "",
            "description": "Single API key to use if multiple keys are not provided.",
        },
        "multiple_keys": {
            "type": "editor",
            "value": "",
            "description": "API keys separated by semicolons (;). Requests will rotate through these keys.",
        },
        "model": {
            "type": "selector",
            "options": [
                "OAI: gpt-4o",
                "OAI: gpt-4-turbo",
                "OAI: gpt-3.5-turbo",
                "GGL: gemini-1.5-pro-latest",
                "GGL: gemini-2.5-flash",
                "GGL: gemini-2.5-flash-lite",
                "XAI: grok-4",
                "XAI: grok-3",
                "XAI: grok-3-mini",
                # OpenRouter free text models (https://openrouter.ai/models?fmt=cards&max_price=0&order=most-popular&output_modalities=text&input_modalities=text)
                "OpenRouter: openrouter/free",
                "OpenRouter: meta-llama/llama-3.2-3b-instruct:free",
                "OpenRouter: meta-llama/llama-3.3-70b-instruct:free",
                "OpenRouter: google/gemma-3n-e2b-it:free",
                "OpenRouter: google/gemma-3n-e4b-it:free",
                "OpenRouter: stepfun/step-3.5-flash:free",
                "OpenRouter: qwen/qwen3-4b:free",
                "OpenRouter: qwen/qwen3-next-80b-a3b-instruct:free",
                "OpenRouter: qwen/qwen3-235b-a22b-thinking-2507",
                "OpenRouter: nvidia/nemotron-nano-9b-v2:free",
                "OpenRouter: nvidia/nemotron-3-nano-30b-a3b:free",
                "OpenRouter: liquid/lfm-2.5-1.2b-instruct:free",
                "OpenRouter: liquid/lfm-2.5-1.2b-thinking:free",
                "OpenRouter: z-ai/glm-4.5-air:free",
                "OpenRouter: arcee-ai/trinity-mini:free",
                "OpenRouter: arcee-ai/trinity-large-preview:free",
                "OpenRouter: nousresearch/hermes-3-llama-3.1-405b:free",
                "OpenRouter: upstage/solar-pro-3:free",
                "OpenRouter: openai/gpt-oss-20b:free",
                "OpenRouter: openai/gpt-oss-120b:free",
                "OpenRouter: qwen/qwen3-coder:free",
                "OpenRouter: cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
                "LLMS: (override model field)",
            ],
            "value": "OAI: gpt-4o",
            "description": "Select a model that supports JSON Mode for structured output.",
        },
        "override model": {
            "value": "",
            "description": "Specify a custom model name to override the selected model.",
        },
        "endpoint": {
            "value": "",
            "description": "Base URL for the API. Leave empty for provider default.",
        },
        "system_prompt": {
            "type": "editor",
            "value": (
                "You are an expert translator for comics and dialogue. Translate the given text snippets naturally and concisely. "
                "Use consistent terminology for the same concept (e.g. stick to one term for recurring ideas). "
                "Cohesion and styling: keep dialogue tone consistent and match character voice across snippets. "
                "Output MUST be valid JSON only: a single object with key 'translations', a list of objects with 'id' (integer) and 'translation' (string). "
                "No markdown, no code fences, no extra text.\n"
                "Example: {\"translations\": [{\"id\": 1, \"translation\": \"Translated text.\"}]}"
            ),
            "description": "Section 18: Dedicated translation system prompt (cohesion rules, styling guide). For z-ai/glm-4.5-air: keep short; emphasize JSON-only and consistent terms.",
        },
        "translation_glossary": {
            "type": "editor",
            "value": "",
            "description": "Terminology for consistency (one per line). Format: source -> target or source = target. E.g. 丹田 -> dantian. Used on every page.",
        },
        "context_previous_pages": {
            "type": "line_editor",
            "value": 1,
            "description": "Include this many previous pages (source+translation) as context for consistency. 0 = off. 1–2 recommended.",
        },
        "context_next_page": {
            "type": "checkbox",
            "value": False,
            "description": "Include the next page's source text as context (helps with continuity; upstream #1142). Off by default.",
        },
        "series_context_prompt": {
            "type": "editor",
            "value": "",
            "description": "Optional. E.g. 'This is a cultivation manhua. Keep cultivation terms and character names consistent.'",
        },
        "series_context_path": {
            "type": "line_editor",
            "value": "default",
            "description": "Folder or series ID for cross-chapter consistency (e.g. urban_immortal_cultivator). Uses data/translation_context/<id>/glossary.txt and recent_context.json. Default: 'default'.",
        },
        "context_max_chars": {
            "type": "line_editor",
            "value": 2000,
            "description": "Max characters for previous-context block (to fit model context limit). 0 = no limit. When over limit: if 'summarize when over limit' is on, the model summarizes context; otherwise recent context is kept and the rest is dropped.",
        },
        "context_trim_mode": {
            "type": "selector",
            "options": ["full", "compact"],
            "value": "compact",
            "description": "full = use all lines per page (up to limit). compact = use at most 2 lines per page to save tokens.",
        },
        "summarize_context_when_over_limit": {
            "type": "checkbox",
            "value": True,
            "description": "When context exceeds context_max_chars, ask the model to summarize it (preserving key terms and style) instead of truncating. Uses one extra API call.",
        },
        "keyword_replacements": {
            "type": "editor",
            "value": "",
            "description": "Post-translation word substitutions (one per line). Format: original -> replacement. Whole-word only, e.g. 'powers' becomes 'abilities' but 'superpowers' is unchanged. Add more lines as needed.",
        },
        "invalid repeat count": {
            "value": 2,
            "description": "Number of retries if the count of translations mismatches the source count.",
        },
        "max requests per minute": {
            "value": 20,
            "description": "Maximum requests per minute for EACH API key.",
        },
        "delay": {
            "value": 0.3,
            "description": "Global delay in seconds between requests.",
        },
        "max tokens": {
            "value": 4096,
            "description": "Maximum tokens for the response.",
        },
        "temperature": {
            "value": 0.1,
            "description": "Sampling temperature. Lower values are recommended for structured output.",
        },
        "top p": {
            "value": 1.0,
            "description": "Top P for sampling.",
        },
        "retry attempts": {
            "value": 3,
            "description": "Number of retry attempts on API connection or parsing failures.",
        },
        "retry timeout": {
            "value": 15,
            "description": "Timeout between retry attempts (seconds).",
        },
        "rate limit delay": {
            "value": 45,
            "description": "Extra wait (seconds) when API returns 429 rate limit, then retry. Increase if free-tier limits persist.",
        },
        "proxy": {
            "value": "",
            "description": "Proxy address (e.g., http(s)://user:password@host:port or socks4/5://user:password@host:port)",
        },
        "frequency penalty": {
            "value": 0.0,
            "description": "Frequency penalty (OpenAI).",
        },
        "presence penalty": {"value": 0.0, "description": "Presence penalty (OpenAI)."},
    }

    def _setup_translator(self):
        self.lang_map = {
            "简体中文": "Simplified Chinese",
            "繁體中文": "Traditional Chinese",
            "日本語": "Japanese",
            "English": "English",
            "한국어": "Korean",
            "Tiếng Việt": "Vietnamese",
            "čeština": "Czech",
            "Français": "French",
            "Deutsch": "German",
            "magyar nyelv": "Hungarian",
            "Italiano": "Italian",
            "Polski": "Polish",
            "Português": "Portuguese",
            "limba română": "Romanian",
            "русский язык": "Russian",
            "Español": "Spanish",
            "Türk dili": "Turkish",
            "украї́нська мо́ва": "Ukrainian",
            "Thai": "Thai",
            "Arabic": "Arabic",
            "Malayalam": "Malayalam",
            "Tamil": "Tamil",
            "Hindi": "Hindi",
        }
        self.token_count = 0
        self.token_count_last = 0
        self.current_key_index = 0
        self.last_request_time = 0
        self.request_count_minute = 0
        self.minute_start_time = time.time()
        self.key_usage = {}
        self.client = None
        self._translation_context_previous_pages = None
        self._translation_project_glossary = None
        self._translation_series_context_path = ""

    def _get_series_context_path(self) -> str:
        """Resolve series context path (from param or set via set_translation_context)."""
        path = getattr(self, "_translation_series_context_path", None) or ""
        if (path or "").strip():
            return get_series_context_dir(path)
        param = (self.get_param_value("series_context_path") or "").strip()
        if param:
            return get_series_context_dir(param)
        return ""

    def _get_glossary_entries(self) -> List[tuple]:
        """Parse translation_glossary param into list of (source, target). Same format as keyword_replacements."""
        raw = self.get_param_value("translation_glossary") or ""
        out = []
        for line in raw.strip().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            for sep in ("->", "→", "=", ":"):
                if sep in line:
                    parts = line.split(sep, 1)
                    if len(parts) == 2:
                        k, v = parts[0].strip(), parts[1].strip()
                        if k and v:
                            out.append((k, v))
                        break
        return out

    def set_translation_context(self, previous_pages=None, project_glossary=None, series_context_path=None, next_page=None):
        """Set previous pages (list of {sources, translations}), optional next_page {sources: [...]}, and project glossary for consistency."""
        self._translation_context_previous_pages = previous_pages or []
        self._translation_context_next_page = next_page if next_page and isinstance(next_page, dict) else None
        self._translation_project_glossary = project_glossary or []
        if series_context_path is not None:
            self._translation_series_context_path = (series_context_path or "").strip()
        # Also store for base cache key
        try:
            super().set_translation_context(previous_pages, project_glossary, series_context_path, next_page)
        except Exception:
            pass

    def append_page_to_series_context(self, series_context_path: str, sources: List[str], translations: List[str]) -> None:
        """Append one translated page to the series context store for cross-chapter consistency."""
        path = get_series_context_dir((series_context_path or "").strip())
        if not path or not sources:
            return
        n_prev = max(5, self.context_previous_pages_count * 3)
        store_append_page(path, sources, translations, max_stored_pages=n_prev)

    def _build_system_prompt(self) -> str:
        """System prompt plus optional glossary (translator + project + series) and series context."""
        parts = [self.system_prompt]
        translator_glossary = self._get_glossary_entries()
        proj_glossary = getattr(self, "_translation_project_glossary", None) or []
        proj_as_tuples = [
            (g["source"], g["target"])
            for g in proj_glossary
            if isinstance(g, dict) and g.get("source") and g.get("target")
        ]
        series_path = self._get_series_context_path()
        series_glossary = load_series_glossary(series_path) if series_path else []
        glossary_entries = merge_glossary_no_dupes(
            translator_glossary,
            proj_as_tuples,
            series_glossary,
        )
        if glossary_entries:
            # Section 18: Rosetta-style models get glossary as JSON list for systematic injection
            model_name = (self.override_model or self.model or "").strip()
            if ": " in model_name:
                model_name = model_name.split(": ", 1)[1]
            if "rosetta" in model_name.lower():
                gl_list = json.dumps([{"source": s, "target": t} for s, t in glossary_entries], ensure_ascii=False)
                parts.append(f"\n\nGlossary (use these exact translations when the source term appears):\n{gl_list}")
            else:
                gl_str = "; ".join(f"{s} -> {t}" for s, t in glossary_entries)
                parts.append(f"\n\nUse these exact translations for terms when they appear: {gl_str}")
        series = (self.get_param_value("series_context_prompt") or "").strip()
        if series:
            parts.append(f"\n\nContext: {series}")
        return "\n".join(parts)

    @property
    def context_previous_pages_count(self) -> int:
        try:
            return max(0, min(5, int(self.get_param_value("context_previous_pages"))))
        except (TypeError, ValueError):
            return 0

    def _initialize_client(self, api_key_to_use: str, provider_override: Optional[str] = None) -> bool:
        endpoint = self.endpoint
        provider = provider_override or self.provider
        if not endpoint:
            if provider == "Google":
                endpoint = "https://generativelanguage.googleapis.com/v1beta/openai"
            elif provider == "OpenAI":
                endpoint = "https://api.openai.com/v1"
            elif provider == "OpenRouter":
                endpoint = "https://openrouter.ai/api/v1"
            elif provider == "Grok":
                endpoint = "https://api.x.ai/v1"

        proxy = self.proxy
        timeout = httpx.Timeout(30.0, read=120.0)
        try:
            http_client = create_httpx_client(proxy, timeout=timeout)
        except Exception as e:
            self.logger.error(
                f"Failed to initialize proxy '{proxy}': {e}. Proceeding without proxy."
            )
            http_client = httpx.Client(timeout=timeout)

        masked_key = (
            api_key_to_use[:4] + "..." + api_key_to_use[-4:]
            if len(api_key_to_use) > 8
            else api_key_to_use
        )
        self.logger.debug(
            f"Initializing client for {provider} with key {masked_key} at endpoint {endpoint}"
        )

        try:
            self.client = openai.OpenAI(
                api_key=api_key_to_use, base_url=endpoint, http_client=http_client
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {e}")
            self.client = None
            return False

    def is_deterministic(self) -> bool:
        try:
            return float(self.temperature) <= 0.0
        except Exception:
            return False

    def _effective_provider_for_model(self) -> str:
        """
        If user picked a model with a provider prefix (e.g. "GGL: ...") but provider selector mismatches,
        route to the implied provider to avoid sending Gemini model names to OpenAI endpoints, etc.
        """
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
                if prefix == "XAI":
                    return "Grok"
                if prefix == "OPENROUTER":
                    return "OpenRouter"
            return self.provider
        except Exception:
            return self.provider

    def _build_generation_config(self, provider: str, model_name: str) -> dict:
        """
        Section 18: Central (provider, model) -> API param names, reasoning toggles, max token caps.
        Returns dict to merge into chat completion api_args.
        """
        cfg = {}
        if provider == "OpenAI":
            cfg["frequency_penalty"] = self.frequency_penalty
            cfg["presence_penalty"] = self.presence_penalty
        max_tok = self.max_tokens
        if provider == "Google" and max_tok > 8192:
            max_tok = 8192
        cfg["max_tokens"] = max_tok
        return cfg

    # --- Property getters ---
    @property
    def provider(self) -> str:
        return self.get_param_value("provider")

    @property
    def apikey(self) -> str:
        return self.get_param_value("apikey")

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
    def model(self) -> str:
        return self.get_param_value("model")

    @property
    def override_model(self) -> Optional[str]:
        return self.get_param_value("override model") or None

    @property
    def endpoint(self) -> Optional[str]:
        return self.get_param_value("endpoint") or None

    @property
    def temperature(self) -> float:
        return float(self.get_param_value("temperature"))

    @property
    def top_p(self) -> float:
        return float(self.get_param_value("top p"))

    @property
    def max_tokens(self) -> int:
        return int(self.get_param_value("max tokens"))

    @property
    def retry_attempts(self) -> int:
        return int(self.get_param_value("retry attempts"))

    @property
    def retry_timeout(self) -> int:
        return int(self.get_param_value("retry timeout"))

    @property
    def rate_limit_delay(self) -> int:
        return max(15, int(self.get_param_value("rate limit delay") or 45))

    @property
    def proxy(self) -> str:
        return self.get_param_value("proxy")

    @property
    def system_prompt(self) -> str:
        return self.get_param_value("system_prompt")

    @property
    def invalid_repeat_count(self) -> int:
        return int(self.get_param_value("invalid repeat count"))

    @property
    def frequency_penalty(self) -> float:
        return float(self.get_param_value("frequency penalty"))

    @property
    def presence_penalty(self) -> float:
        return float(self.get_param_value("presence penalty"))

    @property
    def max_rpm(self) -> int:
        return int(self.get_param_value("max requests per minute"))

    @property
    def global_delay(self) -> float:
        return float(self.get_param_value("delay"))

    def _assemble_prompts(self, queries: List[str], to_lang: str):
        from_lang = self.lang_map.get(self.lang_source, self.lang_source)

        input_elements = [
            {"id": i + 1, "source": query} for i, query in enumerate(queries)
        ]
        input_json_str = json.dumps(input_elements, ensure_ascii=False, indent=2)

        prompt_parts = []
        prev_pages = getattr(self, "_translation_context_previous_pages", None) or []
        n_prev = self.context_previous_pages_count
        # If using series store and we have few in-memory pages, seed from stored recent context (e.g. end of previous chapter)
        series_path = self._get_series_context_path()
        if n_prev > 0 and series_path:
            stored = load_recent_context(series_path, max_pages=n_prev)
            if stored:
                prev_pages = stored + prev_pages
        if n_prev > 0 and prev_pages:
            # Use last N pages only; each entry is {"sources": [...], "translations": [...]}
            context_entries = prev_pages[-n_prev:]
            trim_mode = (self.get_param_value("context_trim_mode") or "compact").strip().lower()
            if trim_mode != "full":
                trim_mode = "compact"
            lines = []
            for i, page in enumerate(context_entries):
                srcs = page.get("sources") or []
                trans = page.get("translations") or []
                if not srcs and not trans:
                    continue
                pairs = []
                for s, t in zip(srcs, trans):
                    s = (s or "").strip()
                    t = (t or "").strip()
                    if s or t:
                        pairs.append(f"  [{s}] -> [{t}]")
                if pairs:
                    if trim_mode == "compact":
                        # Keep at most 2 lines per page to stay within context limits
                        if len(pairs) <= 2:
                            lines.append("Page context:\n" + "\n".join(pairs))
                        else:
                            lines.append("Page context:\n" + "\n".join([pairs[0], pairs[-1]]))
                    else:
                        lines.append("Page context:\n" + "\n".join(pairs[:10]))
            if lines:
                context_block = (
                    "Previous context (for terminology and style consistency):\n"
                    + "\n".join(lines[-6:])
                )
                try:
                    max_chars = int(self.get_param_value("context_max_chars") or 0)
                except (TypeError, ValueError):
                    max_chars = 2000
                if max_chars > 0 and len(context_block) > max_chars:
                    summarize = self.get_param_value("summarize_context_when_over_limit")
                    if summarize:
                        context_block = self._request_context_summary(
                            context_block, max_chars
                        )
                    else:
                        context_block = "...\n" + context_block[-max_chars:]
                prompt_parts.append(context_block)
        next_page = None
        if self.get_param_value("context_next_page"):
            next_page = getattr(self, "_translation_context_next_page", None)
        if next_page and isinstance(next_page, dict):
            srcs = next_page.get("sources") or []
            if srcs:
                next_preview = " ".join((s or "").strip() for s in srcs[:5])[:500]
                if next_preview:
                    prompt_parts.append("Next page (for context): " + next_preview.strip())
        prompt_parts.append(
            f"Please translate the following text snippets from {from_lang} to {to_lang}. "
            f"The input is provided as a JSON array. Respond with a JSON object in the specified format.\n\n"
            f"INPUT:\n{input_json_str}"
        )
        prompt = "\n\n".join(prompt_parts)

        yield prompt, len(queries)

    def _translate_single_item_fallback(
        self, src_list: List[str], to_lang: str, error_placeholder: str
    ) -> List[str]:
        """When batch request fails after retries, try translating each snippet one-by-one."""
        results = []
        for src in src_list:
            self._respect_delay()
            try:
                for prompt, num_src in self._assemble_prompts([src], to_lang=to_lang):
                    parsed = self._request_translation(prompt, expected_count=1)
                    if (
                        parsed
                        and parsed.translations
                        and len(parsed.translations) >= 1
                    ):
                        results.append(
                            self._apply_keyword_substitutions(
                                parsed.translations[0].translation
                            )
                        )
                    else:
                        results.append(error_placeholder)
                    break
            except Exception:
                results.append(error_placeholder)
            time.sleep(max(0, self.global_delay))
        return results

    def _respect_delay(self):
        current_time = time.time()
        rpm = self.max_rpm
        delay = self.global_delay
        if rpm > 0:
            if current_time - self.minute_start_time >= 60:
                self.request_count_minute = 0
                self.minute_start_time = current_time
            if self.request_count_minute >= rpm:
                wait_time = 60.1 - (current_time - self.minute_start_time)
                if wait_time > 0:
                    self.logger.warning(
                        f"Global RPM limit ({rpm}) reached. Waiting {wait_time:.2f} seconds."
                    )
                    time.sleep(wait_time)
                self.request_count_minute = 0
                self.minute_start_time = time.time()

        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < delay:
            sleep_time = delay - time_since_last_request
            if hasattr(self, "debug_mode") and self.debug_mode:
                self.logger.debug(f"Global delay: Waiting {sleep_time:.3f} seconds.")
            time.sleep(sleep_time)

        self.last_request_time = time.time()
        self.request_count_minute += 1

    def _respect_key_limit(self, key: str) -> bool:
        rpm = self.max_rpm
        if rpm <= 0:
            return True
        now = time.time()
        count, start_time = self.key_usage.get(key, (0, now))
        if now - start_time >= 60:
            count, start_time = 0, now
            self.key_usage[key] = (count, start_time)
        if count >= rpm:
            wait_time = 60.1 - (now - start_time)
            if wait_time > 0:
                self.logger.warning(
                    f"RPM limit ({rpm}) reached for key {key[:6]}... Waiting {wait_time:.2f} seconds."
                )
                time.sleep(wait_time)
            self.key_usage[key] = (0, time.time())
            return False
        return True

    def _select_api_key(self) -> Optional[str]:
        api_keys = self.multiple_keys_list
        single_key = self.apikey
        if not api_keys and not single_key:
            self.logger.error("No API keys provided in parameters.")
            return None

        def _warn_if_invalid(provider: str, key: str) -> None:
            try:
                from utils.validation import validate_api_key
                ok, msg = validate_api_key(provider, key, strict=False)
                if not ok and msg:
                    self.logger.warning("API key validation: %s", msg)
            except Exception:
                pass

        provider = self._effective_provider_for_model()

        if not api_keys:
            if self._respect_key_limit(single_key):
                _warn_if_invalid(provider, single_key)
                now = time.time()
                count, start_time = self.key_usage.get(single_key, (0, now))
                if now - start_time >= 60:
                    count = 0
                    start_time = now
                self.key_usage[single_key] = (count + 1, start_time)
                return single_key
            return None

        start_index = self.current_key_index
        for i in range(len(api_keys)):
            index = (start_index + i) % len(api_keys)
            key = api_keys[index]
            if self._respect_key_limit(key):
                _warn_if_invalid(provider, key)
                now = time.time()
                count, start_time = self.key_usage.get(key, (0, now))
                self.key_usage[key] = (count + 1, start_time)
                self.current_key_index = (index + 1) % len(api_keys)
                return key
        self.logger.error("All available API keys are currently rate-limited.")
        return None

    def _get_keyword_replacements(self) -> List[tuple]:
        """Parse keyword_replacements param into list of (original, replacement) for whole-word substitution."""
        raw = self.get_param_value("keyword_replacements") or ""
        out = []
        for line in raw.strip().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            for sep in ("->", "→", "=", ":"):
                if sep in line:
                    parts = line.split(sep, 1)
                    if len(parts) == 2:
                        k, v = parts[0].strip(), parts[1].strip()
                        if k and v:
                            out.append((k, v))
                        break
        return out

    def _apply_keyword_substitutions(self, text: str) -> str:
        """Apply keyword_replacements to text (whole-word match)."""
        if not text:
            return text
        for original, replacement in self._get_keyword_replacements():
            # Whole-word: \b so "powers" matches but "superpowers" does not
            text = re.sub(r"\b" + re.escape(original) + r"\b", replacement, text)
        return text

    def _first_json_object(self, s: str) -> str:
        """Extract the first complete {...} object (brace-balanced). Handles 'Extra data' when model returns concatenated JSON."""
        start = s.find("{")
        if start == -1:
            return s
        depth = 0
        for i in range(start, len(s)):
            if s[i] == "{":
                depth += 1
            elif s[i] == "}":
                depth -= 1
                if depth == 0:
                    return s[start : i + 1]
        return s[start:]

    def _repair_json_unescaped_quotes(self, s: str) -> str:
        """Escape unescaped double-quotes inside \"translation\": \"...\" values so json.loads can parse."""
        out = []
        i = 0
        key = '"translation": "'
        while i < len(s):
            if s[i:i + len(key)] == key:
                out.append(key)
                i += len(key)
                while i < len(s):
                    if s[i] == "\\":
                        out.append(s[i])
                        i += 1
                        if i < len(s):
                            out.append(s[i])
                            i += 1
                        continue
                    if s[i] == '"':
                        j = i + 1
                        while j < len(s) and s[j] in " \t\n\r":
                            j += 1
                        if j < len(s) and s[j] in "},]":
                            out.append('"')
                            i += 1
                            break
                        out.append('\\"')
                        i += 1
                        continue
                    out.append(s[i])
                    i += 1
                continue
            out.append(s[i])
            i += 1
        return "".join(out)

    def _extract_translations_from_broken_json(self, raw: str) -> List[dict]:
        """Extract id/translation pairs when JSON is malformed (e.g. unescaped \" inside strings)."""
        out = []
        patterns = [
            (re.compile(r'"id"\s*:\s*(\d+)\s*,\s*"translation"\s*:\s*"(.*?)"\s*}\s*(?:,\s*|\s*])', re.DOTALL), True),   # id first
            (re.compile(r'"translation"\s*:\s*"(.*?)"\s*,\s*"id"\s*:\s*(\d+)\s*}\s*(?:,\s*|\s*])', re.DOTALL), False),  # translation first
        ]
        for pattern, id_first in patterns:
            for m in pattern.finditer(raw):
                if id_first:
                    tid, text = int(m.group(1)), m.group(2)
                else:
                    text, tid = m.group(1), int(m.group(2))
                text = text.replace("\\\"", "\"").replace("\\\\", "\\")
                out.append({"id": tid, "translation": text})
            if out:
                break
        return out if out else []

    def _parse_numbered_list_response(self, raw: str, expected_count: Optional[int] = None) -> Optional[List[dict]]:
        """
        Section 18: Parse "1: ..." / "1. ..." style responses; fill missing IDs with [missing] placeholder.
        Returns list of {"id": i, "translation": str}. If expected_count given, list has ids 1..expected_count;
        otherwise uses max id found in raw.
        """
        raw = (raw or "").strip()
        if not raw:
            return None
        pattern = re.compile(r"^\s*(\d+)\s*[.:)]\s*(.*)$", re.MULTILINE)
        found = {}
        for m in pattern.finditer(raw):
            idx = int(m.group(1))
            text = (m.group(2) or "").strip()
            if 1 <= idx <= (expected_count or 999):
                found[idx] = text
        if not found:
            return None
        count = expected_count if expected_count is not None else max(found.keys())
        count = max(count, max(found.keys()))
        placeholder = "[missing]"
        return [
            {"id": i, "translation": found.get(i, placeholder)}
            for i in range(1, count + 1)
        ]

    def _request_raw_completion(
        self, messages: List[dict], max_tokens: int = 1024
    ) -> Optional[str]:
        """Perform a single chat completion without JSON response format. Returns content or None."""
        model_name = self.override_model or self.model
        if ": " in model_name:
            model_name = model_name.split(": ", 1)[1]
        api_args = {
            "model": model_name,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": max_tokens,
        }
        try:
            completion = self.client.chat.completions.create(**api_args)
        except Exception as e:
            self.logger.warning(f"Context summary request failed: {e}")
            return None
        if completion is None:
            return None
        if isinstance(completion, str):
            return completion.strip() or None
        if (
            completion.choices
            and completion.choices[0].message
            and completion.choices[0].message.content
        ):
            return completion.choices[0].message.content.strip()
        return None

    def request_custom_completion(
        self, system_prompt: str, user_prompt: str, max_tokens: int = 4096
    ) -> Optional[str]:
        """
        One-off completion with custom system and user prompts (e.g. for ensemble judge).
        Uses same API key/client as normal translation. Returns content string or None.
        """
        provider = self._effective_provider_for_model()
        current_api_key = "lm-studio" if provider == "LLM Studio" else self._select_api_key()
        if not current_api_key and provider != "LLM Studio":
            self.logger.warning("No API key for custom completion.")
            return None
        if provider == "LLM Studio" and not self.endpoint:
            return None
        if not self._initialize_client(current_api_key, provider_override=provider):
            return None
        self._respect_delay()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self._request_raw_completion(messages, max_tokens=max_tokens)

    def _request_context_summary(self, long_context: str, max_chars: int) -> str:
        """Ask the model to summarize long_context to under max_chars; on failure, truncate."""
        current_api_key = "lm-studio"
        provider = self._effective_provider_for_model()
        if provider != "LLM Studio":
            current_api_key = self._select_api_key()
            if not current_api_key:
                self.logger.warning("No API key for context summary; truncating.")
                return "...\n" + long_context[-max_chars:]
        if provider == "LLM Studio" and not self.endpoint:
            return "...\n" + long_context[-max_chars:]
        if not self._initialize_client(current_api_key, provider_override=provider):
            return "...\n" + long_context[-max_chars:]
        self._respect_delay()
        system = (
            "You are a helper that shortens translation context. Given a block of previous source->translation pairs "
            "used for terminology consistency, produce a shorter version that preserves: exact term translations "
            "(keep [source] -> [translation] for important terms and names), character names, and tone. "
            "Output only the summarized context, no other commentary. Stay under the requested character limit."
        )
        user = (
            f"Summarize the following translation context to under {max_chars} characters, "
            f"keeping key terms and style:\n\n{long_context}"
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        summary = self._request_raw_completion(messages, max_tokens=min(2048, max_chars + 500))
        if summary and len(summary) > 0:
            if len(summary) > max_chars:
                summary = summary[: max_chars - 3].rstrip() + "..."
            return summary
        return "...\n" + long_context[-max_chars:]

    def _request_translation(self, prompt: str, expected_count: Optional[int] = None) -> Optional[TranslationResponse]:
        provider = self._effective_provider_for_model()
        current_api_key = "lm-studio"
        if provider != "LLM Studio":
            current_api_key = self._select_api_key()
            if not current_api_key:
                raise ConnectionError("No available API key found.")

        if provider == "LLM Studio" and not self.endpoint:
            raise ValueError(
                "Endpoint must be specified when using the LLM Studio provider (e.g., http://localhost:1234/v1)."
            )

        if not self._initialize_client(current_api_key, provider_override=provider):
            raise ConnectionError("Failed to initialize API client.")

        self._respect_delay()

        model_name = self.override_model or self.model
        if ": " in model_name:
            model_name = model_name.split(": ", 1)[1]

        # Section 9: remember last-used model per provider for config / UI
        try:
            from utils import config as utils_config
            if not hasattr(utils_config.pcfg, "translator_last_model_by_provider"):
                utils_config.pcfg.translator_last_model_by_provider = {}
            utils_config.pcfg.translator_last_model_by_provider[provider] = model_name
        except Exception:
            pass

        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": prompt},
        ]

        api_args = {
            "model": model_name,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        api_args.update(self._build_generation_config(provider, model_name))

        # Section 18: OpenRouter reasoning models may need higher max_tokens
        if provider == "OpenRouter":
            try:
                from utils.endpoints.openrouter import openrouter_is_reasoning_model
                if openrouter_is_reasoning_model(current_api_key, model_name, self.proxy):
                    api_args["max_tokens"] = max(api_args.get("max_tokens", 4096), 8192)
            except Exception:
                pass

        if provider == "LLM Studio":
            self.logger.debug("Using 'json_schema' mode for LLM Studio.")
            api_args["response_format"] = {
                "type": "json_schema",
                "json_schema": {"schema": TranslationResponse.model_json_schema()},
            }
        elif provider in ["OpenAI", "Grok", "Google", "OpenRouter"]:
            self.logger.debug(f"Using 'json_object' mode for {provider}.")
            api_args["response_format"] = {"type": "json_object"}

        try:
            completion = self.client.chat.completions.create(**api_args)
        except Exception as e:
            err_str = str(e).lower()
            # Some models (e.g. StepFun via OpenRouter) don't support response_format json_object; retry without it
            if (
                provider in ["OpenAI", "Grok", "Google", "OpenRouter"]
                and "response_format" in err_str
                and "not supported" in err_str
                and "response_format" in api_args
            ):
                self.logger.warning(
                    f"Model does not support json_object response_format. Retrying without it."
                )
                api_args = {k: v for k, v in api_args.items() if k != "response_format"}
                try:
                    completion = self.client.chat.completions.create(**api_args)
                except Exception as e2:
                    self.logger.error(f"API request failed: {e2}")
                    raise
            else:
                self.logger.error(f"API request failed: {e}")
                raise

        if completion is None:
            self.logger.warning("API returned None.")
            return None
        # Some providers return a raw string instead of an object with .choices (#1098)
        raw_content = None
        if isinstance(completion, str):
            raw_content = completion.strip()
        elif getattr(completion, "choices", None) and completion.choices:
            msg = getattr(completion.choices[0], "message", None)
            if msg and getattr(msg, "content", None):
                raw_content = msg.content.strip()
        if not raw_content:
            self.logger.warning("API returned no content or unexpected response shape.")
            return None
        json_to_parse = raw_content

        match = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```", json_to_parse, re.DOTALL
        )
        if match:
            self.logger.debug(
                "Markdown code block detected. Extracting JSON content."
            )
            json_to_parse = match.group(1)
        else:
            start = json_to_parse.find("{")
            end = json_to_parse.rfind("}")
            if start != -1 and end != -1 and end > start:
                json_to_parse = json_to_parse[start : end + 1]
        try:
                data_to_validate = json.loads(json_to_parse)
                # Normalize Grok-style {"1": "text", "2": "..."} to {"translations": [...]} before validate (#1031)
                if isinstance(data_to_validate, dict) and "translations" not in data_to_validate:
                    if all(str(k).strip().isdigit() for k in data_to_validate.keys()):
                        data_to_validate = {
                            "translations": [
                                {"id": int(k), "translation": v if isinstance(v, str) else (str(v) if v is not None else "")}
                                for k, v in sorted(data_to_validate.items(), key=lambda x: int(str(x[0]).strip()) if str(x[0]).strip().isdigit() else 0)
                            ]
                        }
                validated_response = TranslationResponse.model_validate(
                    data_to_validate
                )
            except (ValidationError, json.JSONDecodeError) as e:
                validated_response = None
                self.logger.warning(
                    f"Initial Pydantic validation failed: {e}. Attempting repair and fallbacks."
                )
                # 0) If "Extra data", model likely returned concatenated JSON; use first object only
                if "Extra data" in str(e):
                    try:
                        first_only = self._first_json_object(json_to_parse)
                        if first_only != json_to_parse:
                            data_to_validate = json.loads(first_only)
                            validated_response = TranslationResponse.model_validate(
                                data_to_validate
                            )
                            self.logger.info(
                                "Parsed first JSON object only (response had extra concatenated data)."
                            )
                    except (ValidationError, json.JSONDecodeError):
                        pass
                # 1) Try repairing unescaped quotes inside "translation": " values
                if validated_response is None:
                    try:
                        repaired = self._repair_json_unescaped_quotes(json_to_parse)
                        data_to_validate = json.loads(repaired)
                        validated_response = TranslationResponse.model_validate(
                            data_to_validate
                        )
                        self.logger.info(
                            "Successfully parsed response after repairing unescaped quotes in JSON."
                        )
                    except (ValidationError, json.JSONDecodeError):
                        pass
                # 2) Try simple format (e.g. {"1": "text"} or list of items)
                if validated_response is None:
                    try:
                        simple_data = json.loads(json_to_parse)
                        fixed_translations = []
                        if isinstance(simple_data, dict) and all(
                            k.isdigit() for k in simple_data.keys()
                        ):
                            fixed_translations = [
                                {"id": int(k), "translation": v if isinstance(v, str) else (str(v) if v is not None else "")}
                                for k, v in sorted(simple_data.items(), key=lambda x: int(x[0]) if str(x[0]).strip().isdigit() else 0)
                            ]
                        elif isinstance(simple_data, list):
                            fixed_translations = simple_data
                        if fixed_translations:
                            fixed_data = {"translations": fixed_translations}
                            validated_response = TranslationResponse.model_validate(
                                fixed_data
                            )
                            self.logger.info(
                                "Successfully parsed response after fixing simple format."
                            )
                    except (ValidationError, json.JSONDecodeError):
                        pass
                # 3) Last resort: regex extraction (handles unescaped " in values)
                if validated_response is None:
                    try:
                        fixed_translations = self._extract_translations_from_broken_json(
                            self._repair_json_unescaped_quotes(json_to_parse)
                        )
                        if not fixed_translations:
                            fixed_translations = self._extract_translations_from_broken_json(
                                json_to_parse
                            )
                        if fixed_translations:
                            fixed_data = {"translations": fixed_translations}
                            validated_response = TranslationResponse.model_validate(
                                fixed_data
                            )
                            self.logger.info(
                                "Successfully parsed response after regex extraction (malformed JSON repaired)."
                            )
                        else:
                            raise e
                    except Exception as repair_err:
                        pass
                # 4) Section 18: numbered-list "1: ..." style with missing-item placeholders
                if validated_response is None and expected_count is not None and expected_count > 0:
                    try:
                        numbered = self._parse_numbered_list_response(raw_content, expected_count)
                        if numbered and len(numbered) >= 1:
                            validated_response = TranslationResponse(translations=[
                                TranslationElement(id=x["id"], translation=x["translation"])
                                for x in numbered
                            ])
                            self.logger.info(
                                "Successfully parsed response as numbered list (missing entries filled with placeholder)."
                            )
                    except Exception:
                        pass
                if validated_response is None:
                    self.logger.error(
                        "Pydantic validation or JSON parsing failed even after attempting fix."
                    )
                    self.logger.debug(f"Raw content from API: {raw_content}")
                    raise e
        else:
            self.logger.warning("No valid message content in API response.")
            return None

        if hasattr(completion, "usage") and completion.usage:
            self.token_count += completion.usage.total_tokens
            self.token_count_last = completion.usage.total_tokens
        else:
            self.token_count_last = 0

        return validated_response

    def _translate(self, src_list: List[str]) -> List[str]:
        if not src_list:
            return []

        RETRYABLE_EXCEPTIONS = (
            openai.RateLimitError,
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.InternalServerError,
            openai.APIStatusError,
            httpx.RequestError,
        )

        translations = []
        to_lang = self.lang_map.get(self.lang_target, self.lang_target)

        for prompt, num_src in self._assemble_prompts(src_list, to_lang=to_lang):
            api_retry_attempt = 0
            mismatch_retry_attempt = 0

            while True:
                try:
                    parsed_response = self._request_translation(prompt, expected_count=num_src)

                    if not parsed_response or not parsed_response.translations:
                        raise ValueError(
                            "Received empty or invalid parsed response from API."
                        )

                    translations_dict = {
                        item.id: item.translation
                        for item in parsed_response.translations
                    }
                    ordered_translations = [
                        translations_dict.get(i, "") for i in range(1, num_src + 1)
                    ]
                    got = len(parsed_response.translations)
                    if got != num_src:
                        self.logger.warning(
                            f"Translation count mismatch: expected {num_src}, got {got}. Using available entries (missing filled with empty)."
                        )
                        # If we got 1 and expected N, try newline split as improvement over empty slots
                        if got == 1 and num_src > 1:
                            one = parsed_response.translations[0].translation
                            lines = [s.strip() for s in one.split("\n") if s.strip()]
                            if len(lines) >= num_src:
                                ordered_translations = lines[:num_src]
                                self.logger.info(
                                    "Recovered from single newline-separated response."
                                )
                            elif len(lines) > 0:
                                # Use lines we have, pad rest with ""
                                ordered_translations = lines[:num_src] + [""] * (
                                    num_src - len(lines)
                                )

                    # Apply keyword substitutions (e.g. powers -> abilities)
                    ordered_translations = [
                        self._apply_keyword_substitutions(t) for t in ordered_translations
                    ]
                    translations.extend(ordered_translations)
                    self.logger.info(
                        f"Successfully translated batch of {num_src}. Tokens used: {self.token_count_last}"
                    )
                    break

                except RETRYABLE_EXCEPTIONS as e:
                    api_retry_attempt += 1
                    err_msg = str(e)
                    self.logger.warning(
                        f"API Error (retryable): {type(e).__name__} - {e}. Attempt {api_retry_attempt}/{self.retry_attempts}."
                    )
                    if isinstance(e, (openai.APIConnectionError, httpx.RequestError)):
                        self.logger.info(
                            "Connection error: check firewall/proxy, VPN, or DNS. If behind a proxy, set 'Proxy' in translator settings (e.g. http://proxy:port). Test: curl -I https://openrouter.ai"
                        )
                    if self.provider == "OpenRouter" and "404" in err_msg and "data policy" in err_msg.lower():
                        self.logger.info(
                            "OpenRouter 404: enable 'Free model publication' (or your model) at https://openrouter.ai/settings/privacy"
                        )
                    is_rate_limit = self.provider == "OpenRouter" and "429" in err_msg and ("rate limit" in err_msg.lower() or "Rate limit" in err_msg)
                    if is_rate_limit:
                        self.logger.info(
                            "Rate limit (429). Waiting %s s then retrying (%s/%s).",
                            self.rate_limit_delay, api_retry_attempt, self.retry_attempts,
                        )
                    if api_retry_attempt >= self.retry_attempts:
                        raise CriticalTranslationError(
                            f"Failed to connect to API after {self.retry_attempts} attempts.",
                            cause=e,
                        )
                    # Use longer delay for rate limit so upstream can recover
                    delay = self.rate_limit_delay if is_rate_limit else self.retry_timeout
                    time.sleep(delay)

                except json.JSONDecodeError as e:
                    api_retry_attempt += 1
                    self.logger.warning(
                        f"Malformed JSON from API (retrying). Attempt {api_retry_attempt}/{self.retry_attempts}."
                    )
                    if api_retry_attempt >= self.retry_attempts:
                        self.logger.warning(
                            "Malformed JSON after retries. Trying each snippet one-by-one (single-item fallback)."
                        )
                        fallback = self._translate_single_item_fallback(
                            src_list, to_lang, "[ERROR: Invalid JSON]"
                        )
                        translations.extend(fallback)
                        break
                    time.sleep(self.retry_timeout)

                except ValueError as e:
                    if "empty or invalid parsed response" in str(e):
                        api_retry_attempt += 1
                        self.logger.warning(
                            f"Empty or invalid API response. Attempt {api_retry_attempt}/{self.retry_attempts}."
                        )
                        if api_retry_attempt >= self.retry_attempts:
                            raise CriticalTranslationError(
                                "Empty or invalid API response after retries.",
                                cause=e,
                            )
                        time.sleep(self.retry_timeout)
                    else:
                        page = getattr(self, '_current_page_key', None)
                        self.logger.error(
                            f"Fatal Error: An unrecoverable error occurred: {type(e).__name__} - {e}"
                            + (f" (page: {page})" if page else "")
                        )
                        self.logger.debug(traceback.format_exc())
                        translations.extend([f"[ERROR: {type(e).__name__}]"] * num_src)
                        break

                except (
                    ValidationError,
                    openai.BadRequestError,
                    openai.AuthenticationError,
                ) as e:
                    err_msg = str(e).lower()
                    is_auth = isinstance(e, openai.AuthenticationError)
                    is_quota = "quota" in err_msg or "limit exceeded" in err_msg
                    if is_auth or is_quota:
                        raise CriticalTranslationError(
                            "Authentication or quota error; cannot continue.",
                            cause=e,
                        )
                    page = getattr(self, '_current_page_key', None)
                    self.logger.error(
                        f"Fatal Error: An unrecoverable error occurred: {type(e).__name__} - {e}"
                        + (f" (page: {page})" if page else "")
                    )
                    self.logger.debug(traceback.format_exc())
                    translations.extend([f"[ERROR: {type(e).__name__}]"] * num_src)
                    break

        return translations

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)

        if param_key in ["proxy", "multiple_keys", "apikey", "provider", "endpoint"]:
            self.client = None
            if param_key == "apikey" or param_key == "provider":
                try:
                    from utils.endpoints.openrouter import openrouter_clear_models_cache
                    openrouter_clear_models_cache()
                except Exception:
                    pass
