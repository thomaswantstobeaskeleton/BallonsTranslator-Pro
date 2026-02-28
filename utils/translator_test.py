"""
Test whether the current translator is usable (e.g. API key, network).
"""
from typing import Tuple

from modules.translators.base import BaseTranslator
from modules.translators.exceptions import (
    TranslatorSetupFailure,
    InvalidSourceOrTargetLanguage,
    TranslationNotFound,
)
from utils.logger import logger as LOGGER


def test_translator(translator: BaseTranslator) -> Tuple[bool, str, str]:
    """
    Test if the translator works.

    Args:
        translator: Translator instance to test.

    Returns:
        (success, source_text, result_or_error_message)
    """
    test_text = _test_text_for_lang(translator.lang_source)
    try:
        LOGGER.info("Testing translator %s, source: %s, target: %s", translator.name, translator.lang_source, translator.lang_target)
        result = translator.translate(test_text)
        if not result or (isinstance(result, str) and result.strip() == ""):
            return False, test_text, "Translation result was empty. Check translator settings."
        if isinstance(result, list):
            result = result[0] if result else ""
        LOGGER.info("Translation result: %s", result)
        return True, test_text, result
    except TranslatorSetupFailure as e:
        LOGGER.error("Translator setup failed: %s", e)
        return False, test_text, str(e)
    except InvalidSourceOrTargetLanguage as e:
        LOGGER.error("Unsupported language: %s", e)
        return False, test_text, str(e)
    except TranslationNotFound as e:
        LOGGER.error("No translation found: %s", e)
        return False, test_text, str(e)
    except Exception as e:
        LOGGER.error("Translator test failed: %s", e)
        return False, test_text, "Translation test failed: " + str(e)


def _test_text_for_lang(lang_source: str) -> str:
    if lang_source == "日本語":
        return "気泡翻訳はオープンソースで無料、深層学習技術に基づく漫画翻訳ツールです。"
    if lang_source == "English":
        return "Balloon Translation is an open-source, free manga translation tool based on deep learning technology."
    if lang_source == "简体中文":
        return "气泡翻译器是一个开源免费,基于深度学习技术的漫画翻译工具."
    if lang_source == "繁體中文":
        return "氣泡翻譯器是一個開源免費,基於深度學習技術的漫畫翻譯工具."
    if lang_source == "한국어":
        return "말풍선 번역은 오픈 소스, 무료, 딥러닝 기술 기반의 만화 번역 도구입니다."
    return "Balloon Translation is an open-source, free manga translation tool based on deep learning technology."
