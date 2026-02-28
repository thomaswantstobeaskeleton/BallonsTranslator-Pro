"""
Modified From PyDeepLX

Author: Vincent Young
Date: 2023-04-27 00:44:01
... (остальная часть оригинального заголовка)
"""

import random
import time
import json
import httpx
from langdetect import detect
import brotli
import gzip
import re
from typing import Dict, List, Optional

from modules.translators.base import BaseTranslator, register_translator
from utils.logger import logger as LOGGER


deeplAPI_base = "https://www2.deepl.com/jsonrpc"
deepl_client_params = "client=chrome-extension,1.28.0"
headers = {
    "Content-Type": "application/json",
    "User-Agent": "DeepL/1627620 CFNetwork/3826.500.62.2.1 Darwin/24.4.0",
    "Accept": "*/*",
    "X-App-Os-Name": "iOS",
    "X-App-Os-Version": "18.4.0",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "X-App-Device": "iPhone16,2",
    "Referer": "https://www.deepl.com/",
    "X-Product": "translator",
    "X-App-Build": "1627620",
    "X-App-Version": "25.1",
}


class TooManyRequestsException(Exception):
    def __str__(self):
        return "Error: Too many requests, your IP has been blocked by DeepL temporarily, please don't request it frequently in a short time."


def detectLang(translateText: str) -> str:
    try:
        language = detect(translateText)
        return language.upper()
    except:
        return "EN"


def getICount(translateText: str) -> int:
    return translateText.count("i")


def getRandomNumber() -> int:
    src = random.Random(time.time())
    num = src.randint(8300000, 8399999)
    return num * 1000


def getTimestamp(iCount: int) -> int:
    ts = int(time.time() * 1000)
    if iCount == 0:
        return ts
    iCount += 1
    return ts - ts % iCount + iCount


def format_post_data(post_data_dict, id_val):
    post_data_str = json.dumps(post_data_dict, ensure_ascii=False)
    if (id_val + 5) % 29 == 0 or (id_val + 3) % 13 == 0:
        post_data_str = post_data_str.replace('"method":"', '"method" : "', 1)
    else:
        post_data_str = post_data_str.replace('"method":"', '"method": "', 1)
    return post_data_str


def is_richtext(text: str) -> bool:
    return bool(re.search(r"<[^>]+>", text))


def deepl_split_text(
    text: str, tag_handling: bool = None, proxy_mounts=None
) -> dict:  # Изменено: proxy_mounts
    source_lang = "auto"
    text_type = "richtext" if (tag_handling or is_richtext(text)) else "plaintext"
    postData = {
        "jsonrpc": "2.0",
        "method": "LMT_split_text",
        "params": {
            "commonJobParams": {"mode": "translate"},
            "lang": {"lang_user_selected": source_lang},
            "texts": [text],
            "textType": text_type,
        },
        "id": getRandomNumber(),
    }
    postDataStr = format_post_data(postData, getRandomNumber())
    url = f"{deeplAPI_base}?{deepl_client_params}&method=LMT_split_text"
    return make_deepl_request(url, postDataStr, proxy_mounts)  # Изменено: proxy_mounts


def make_deepl_request(url, postDataStr, proxy_mounts):  # Изменено: proxy_mounts
    client = httpx.Client(
        headers=headers, mounts=proxy_mounts, timeout=30, verify=False
    )  # Изменено: mounts=proxy_mounts
    try:
        LOGGER.debug(f"Request JSON: {postDataStr}")
        resp = client.post(url=url, content=postDataStr)
        if not resp.is_success:
            LOGGER.error(
                f"Request failed with status code: {resp.status_code}, response text: {resp.text}"
            )
            return {"error": resp.text, "status_code": resp.status_code}
        try:
            return resp.json()
        except json.JSONDecodeError:
            try:
                return json.loads(gzip.decompress(resp.content))
            except Exception:
                try:
                    return resp.json()
                except:
                    try:
                        return json.loads(brotli.decompress(resp.content))
                    except Exception as e:
                        LOGGER.error(
                            f"Decompression error: {e}, content: {resp.content[:100]}"
                        )
                        return {"error": "Failed to decompress response"}

    except httpx.HTTPError as e:
        LOGGER.error(f"HTTPError: {e}")
        LOGGER.error(f"Request URL: {url}")
        LOGGER.error(f"Request Data: {postDataStr}")
        return {"error": str(e)}


def deepl_response_to_deeplx(data: dict) -> dict:
    alternatives = []
    if (
        "result" in data
        and "translations" in data["result"]
        and len(data["result"]["translations"]) > 0
    ):
        num_beams = len(data["result"]["translations"][0].get("beams", []))
        for i in range(num_beams):
            alternative_str = ""
            for translation in data["result"]["translations"]:
                beams = translation.get("beams", [])
                if i < len(beams):
                    sentences = beams[i].get("sentences", [])
                    if sentences:
                        alternative_str += sentences[0].get("text", "")
                alternatives.append(alternative_str)
    source_lang = data.get("result", {}).get("source_lang", "unknown")
    target_lang = data.get("result", {}).get("target_lang", "unknown")
    main_translation = " ".join(
        translation.get("beams", [{}])[0].get("sentences", [{}])[0].get("text", "")
        for translation in data.get("result", {}).get("translations", [])
    )
    return {
        "alternatives": alternatives,
        "code": 200,
        "data": main_translation,
        "id": data.get("id", None),
        "method": "Free",
        "source_lang": source_lang,
        "target_lang": target_lang,
    }


def translate_core(
    text,
    sourceLang,
    targetLang,
    tagHandling,
    dl_session="",
    proxy_mounts=None,  # Изменено: proxy_mounts
):
    if not text:
        return {"code": 404, "message": "No text to translate"}

    split_result_json = deepl_split_text(
        text, tagHandling in ("html", "xml"), proxy_mounts
    )  # Изменено: proxy_mounts
    if "error" in split_result_json:
        status = split_result_json.get("status_code", 0)
        if status == 429:
            return {"code": 429, "message": split_result_json["error"]}
        return {"code": 503, "message": split_result_json["error"]}

    if sourceLang == "auto" or not sourceLang:
        sourceLang_detected = (
            split_result_json.get("result", {}).get("lang", {}).get("detected")
        )
        if sourceLang_detected:
            sourceLang = sourceLang_detected.lower()
        else:
            sourceLang = detectLang(text).lower()

    i_count = getICount(text)

    jobs = []
    try:
        chunks = split_result_json["result"]["texts"][0]["chunks"]
    except (KeyError, IndexError, TypeError):
        return {"code": 503, "message": "Unexpected response structure from split_text"}

    for idx, chunk in enumerate(chunks):
        sentence = chunk["sentences"][0]
        context_before = [chunks[idx - 1]["sentences"][0]["text"]] if idx > 0 else []
        context_after = (
            [chunks[idx + 1]["sentences"][0]["text"]] if idx < len(chunks) - 1 else []
        )

        jobs.append(
            {
                "kind": "default",
                "preferred_num_beams": 4,
                "raw_en_context_before": context_before,
                "raw_en_context_after": context_after,
                "sentences": [
                    {
                        "prefix": sentence["prefix"],
                        "text": sentence["text"],
                        "id": idx + 1,
                    }
                ],
            }
        )

    targetLang_code = targetLang.upper()
    has_regional_variant = False
    if "-" in targetLang:
        targetLang_code = targetLang.split("-")[0].upper()
        has_regional_variant = True

    current_tag_handling = "plaintext"
    postData = {
        "jsonrpc": "2.0",
        "method": "LMT_handle_jobs",
        "id": getRandomNumber(),
        "params": {
            "commonJobParams": {
                "mode": "translate",
                "formality": "undefined",
                "transcribeAs": "romanize",
                "advancedMode": False,
                "textType": current_tag_handling,
                "wasSpoken": False,
            },
            "lang": {
                "source_lang_user_selected": "auto",
                "target_lang": targetLang_code,
                "source_lang_computed": sourceLang.upper(),
            },
            "jobs": jobs,
            "timestamp": getTimestamp(i_count),
        },
    }

    if has_regional_variant:
        postData["params"]["commonJobParams"]["regionalVariant"] = targetLang

    postDataStr = format_post_data(postData, getRandomNumber())
    LOGGER.debug(f"Request JSON before sending: {postDataStr}")
    url = f"{deeplAPI_base}?{deepl_client_params}&method=LMT_handle_jobs"
    translate_result_json = make_deepl_request(
        url, postDataStr, proxy_mounts
    )  # Изменено: proxy_mounts

    if "error" in translate_result_json:
        status = translate_result_json.get("status_code", 0)
        if status == 429:
            return {"code": 429, "message": translate_result_json["error"]}
        return {"code": 503, "message": translate_result_json["error"]}

    deeplx_result = deepl_response_to_deeplx(translate_result_json)
    return deeplx_result


def translate(
    text,
    sourceLang=None,
    targetLang=None,
    numberAlternative=0,
    printResult=False,
    proxy_mounts=None,  # Изменено: proxy_mounts
):
    tagHandling = "plaintext"  # Явно задаем plaintext
    result_json = translate_core(
        text, sourceLang, targetLang, tagHandling, proxy_mounts=proxy_mounts
    )  # Изменено: proxy_mounts

    if result_json and result_json["code"] == 200:
        if printResult:
            print(result_json["data"])
        return result_json["data"]
    else:
        error_message = (
            result_json.get("message", "Unknown error")
            if result_json
            else "Request failed"
        )
        LOGGER.error(f"Translation error: {error_message}")
        raise Exception(f"Translation failed: {error_message}")


@register_translator("DeepL Free")
class DeepLX(BaseTranslator):
    cht_require_convert = True
    params: Dict = {
        "delay": 0.0,
        "429_retry_seconds": 60.0,
        "proxy": {
            "value": "",
            "description": "Proxy address (e.g., http(s)://user:password@host:port or socks4/5://user:password@host:port)",
        },
    }
    concate_text = False

    def _setup_translator(self):
        self.lang_map = {
            "简体中文": "zh",
            "日本語": "ja",
            "English": "en",
            "Français": "fr",
            "Deutsch": "de",
            "Italiano": "it",
            "Português": "pt",
            "Brazilian Portuguese": "pt-br",
            "русский язык": "ru",
            "Español": "es",
            "български език": "bg",
            "Český Jazyk": "cs",
            "Dansk": "da",
            "Ελληνικά": "el",
            "Eesti": "et",
            "Suomi": "fi",
            "Magyar": "hu",
            "Lietuvių": "lt",
            "latviešu": "lv",
            "Nederlands": "nl",
            "Polski": "pl",
            "Română": "ro",
            "Slovenčina": "sk",
            "Slovenščina": "sl",
            "Svenska": "sv",
            "Indonesia": "id",
            "украї́нська мо́ва": "uk",
            "한국어": "ko",
            "Arabic": "ar",
            "繁體中文": "zh-TW",
        }
        self.textblk_break = "\n"

    def __init__(
        self, source="auto", target="en", raise_unsupported_lang=True, **params
    ):
        self.proxy_str = params.get("proxy", {}).get(
            "value"
        )  # Сохраняем прокси как строку
        self.proxy_mounts = self._create_proxy_mounts(
            self.proxy_str
        )  # Создаем mounts сразу при инициализации
        super().__init__(source, target, raise_unsupported_lang=raise_unsupported_lang)

    def _create_proxy_mounts(self, proxy_str: Optional[str]) -> Optional[Dict]:
        if not proxy_str:  # Если proxy_str пустая или None
            return None  # Возвращаем None, если прокси не нужен

        proxy_mounts = {}
        if proxy_str.startswith("socks"):  # Обработка SOCKS прокси
            proxy_mounts["http://"] = httpx.HTTPTransport(proxy=proxy_str)
            proxy_mounts["https://"] = httpx.HTTPTransport(proxy=proxy_str)
        else:  # Обработка HTTP/HTTPS прокси (предполагаем HTTP схему для прокси URL)
            proxy_mounts["http://"] = httpx.HTTPTransport(proxy=proxy_str)
            proxy_mounts["https://"] = httpx.HTTPTransport(proxy=proxy_str)
        return proxy_mounts

    @property
    def proxy(self):  # property proxy теперь возвращает mounts
        return self.proxy_mounts

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        if param_key == "proxy":
            self.proxy_str = param_content["value"]  # Обновляем строку прокси
            self.proxy_mounts = self._create_proxy_mounts(
                self.proxy_str
            )  # Пересоздаем mounts

    def _translate(self, src_list: List[str]) -> List[str]:
        result = []
        source = self.lang_map[self.lang_source]
        target = self.lang_map[self.lang_target]
        proxy_mounts = self.proxy  # Используем property proxy, чтобы получить mounts

        delay_sec = float(self.params.get("delay", {}).get("value", 0) or 0)
        retry_429_sec = float(self.params.get("429_retry_seconds", {}).get("value", 60) or 60)
        retry_429_sec = max(10.0, min(300.0, retry_429_sec))

        for text_block in src_list:
            translated_lines = []
            lines = text_block.split("\n")
            for line in lines:
                if delay_sec > 0:
                    time.sleep(delay_sec)
                last_err = None
                for attempt in range(2):
                    try:
                        tl = translate(
                            line, source, target, proxy_mounts=proxy_mounts
                        )
                        translated_lines.append(tl)
                        break
                    except Exception as e:
                        is_429 = (
                            "429" in str(e)
                            or "Too many requests" in str(e).lower()
                        )
                        if is_429 and attempt == 0:
                            LOGGER.warning(
                                f"DeepL 429 Too Many Requests; waiting {retry_429_sec:.0f}s then retrying."
                            )
                            time.sleep(retry_429_sec)
                        else:
                            LOGGER.error(
                                f"Translation failed for line: '{line}'. Error: {e}"
                            )
                            translated_lines.append("")
                            break
            result.append("\n".join(translated_lines))
        return result
