import urllib.request
from ordered_set import OrderedSet
from typing import Dict, List, Union, Set, Callable
import time, requests, re, uuid, base64, hmac, functools, json, copy
from collections import OrderedDict

from .exceptions import InvalidSourceOrTargetLanguage, TranslatorSetupFailure, MissingTranslatorParams, TranslatorNotValid
from utils.textblock import TextBlock
from ..base import BaseModule, DEVICE_SELECTOR
from utils.registry import Registry
from utils.io_utils import text_is_empty, normalize_line_breaks
from utils.logger import logger as LOGGER
from utils.translation_cache import TranslationCache
from utils.config import pcfg

TRANSLATORS = Registry('translators')
register_translator = TRANSLATORS.register_module

PROXY = urllib.request.getproxies()


def sanitize_translation_text(text: str) -> str:
    """Normalize HTML line breaks and trim excessive trailing repetition from LLM/API output."""
    if not text or not isinstance(text, str):
        return text
    # Unescape literal \n and \r\n from API (e.g. "Mortal\nworld" as two chars -> real newline)
    text = text.replace("\\n", "\n").replace("\\r\\n", "\n").replace("\\r", "\n")
    text = normalize_line_breaks(text)
    # Trim excessive trailing repetition (e.g. "one one one one one" or "一个一个..." from stuck generation)
    text = text.rstrip()
    if len(text) < 4:
        return text
    max_repeat = 3  # allow at most this many repetitions of the same unit at the end
    for unit_len in range(1, min(len(text) // 2, 21)):  # unit from 1 to ~20 chars
        unit = text[-unit_len:]
        if not unit.strip():  # skip all-whitespace units
            continue
        count = 0
        i = len(text)
        while i >= unit_len and text[i - unit_len:i] == unit:
            count += 1
            i -= unit_len
        if count > max_repeat:
            return text[:i + unit_len * max_repeat].rstrip()
    return text

LANGMAP_GLOBAL = {
    'Auto': '',
    '简体中文': '',
    '繁體中文': '',
    '日本語': '',
    'English': '',
    '한국어': '',
    'Tiếng Việt': '',
    'čeština': '',
    'Nederlands': '',
    'Français': '',
    'Deutsch': '',
    'magyar nyelv': '',
    'Italiano': '',
    'Polski': '',
    'Português': '',
    'Brazilian Portuguese': '',
    'limba română': '',
    'русский язык': '',
    'Español': '',
    'Türk dili': '',
    'украї́нська мо́ва': '',  
    'Thai': '',
    'Arabic': '',
    'Hindi': '',
    'Malayalam': '',
    'Tamil': '',
}

# Latin (English) labels for UI display next to non-Latin names (#1132)
LANG_LATIN_DISPLAY = {
    'Auto': 'Auto',
    '简体中文': 'Chinese Simplified',
    '繁體中文': 'Chinese Traditional',
    '日本語': 'Japanese',
    'English': 'English',
    '한국어': 'Korean',
    'Tiếng Việt': 'Vietnamese',
    'čeština': 'Czech',
    'Nederlands': 'Dutch',
    'Français': 'French',
    'Deutsch': 'German',
    'magyar nyelv': 'Hungarian',
    'Italiano': 'Italian',
    'Polski': 'Polish',
    'Português': 'Portuguese',
    'Brazilian Portuguese': 'Brazilian Portuguese',
    'limba română': 'Romanian',
    'русский язык': 'Russian',
    'Español': 'Spanish',
    'Türk dili': 'Turkish',
    'украї́нська мо́ва': 'Ukrainian',
    'Thai': 'Thai',
    'Arabic': 'Arabic',
    'Hindi': 'Hindi',
    'Malayalam': 'Malayalam',
    'Tamil': 'Tamil',
}


def lang_display_label(key: str) -> str:
    """Return display string for language selector: 'key (Latin)' when Latin differs, else key. (#1132)"""
    latin = LANG_LATIN_DISPLAY.get(key)
    if latin and latin != key:
        return f"{key} ({latin})"
    return key


def lang_display_to_key(display: str) -> str:
    """Extract internal language key from combobox display string. (#1132)"""
    if " (" in display:
        return display.split(" (", 1)[0].strip()
    return display.strip()


SYSTEM_LANG = ''
SYSTEM_LANGMAP = {
    'zh-CN': '简体中文'        
}


def check_language_support(check_type: str = 'source'):
    
    def decorator(set_lang_method):
        @functools.wraps(set_lang_method)
        def wrapper(self, lang: str = ''):
            if check_type == 'source':
                supported_lang_list = self.supported_src_list
            else:
                supported_lang_list = self.supported_tgt_list
            if not lang in supported_lang_list:
                msg = '\n'.join(supported_lang_list)
                raise InvalidSourceOrTargetLanguage(f'Invalid {check_type}: {lang}\n', message=msg)
            return set_lang_method(self, lang)
        return wrapper

    return decorator


class BaseTranslator(BaseModule):

    concate_text = True
    cht_require_convert = False

    _postprocess_hooks = OrderedDict()
    _preprocess_hooks = OrderedDict()
    
    def __init__(self,
                 lang_source: str, 
                 lang_target: str,
                 raise_unsupported_lang: bool = True,
                 **params) -> None:
        super().__init__(**params)
        self.name = ''
        for key in TRANSLATORS.module_dict:
            if TRANSLATORS.module_dict[key] == self.__class__:
                self.name = key
                break
        self.textblk_break = '\n##\n'
        self.lang_source: str = lang_source
        self.lang_target: str = lang_target
        self.lang_map: Dict = LANGMAP_GLOBAL.copy()
        
        try:
            self.setup_translator()
        except Exception as e:
            if isinstance(e, MissingTranslatorParams):
                raise e
            else:
                raise TranslatorSetupFailure(e)
            
        # enable traditional chinese by converting from simplified chinese
        if self.cht_require_convert and not self.lang_map['繁體中文']:
            self.lang_map['繁體中文'] = self.lang_map['简体中文']

        self.valid_lang_list = [lang for lang in self.lang_map if self.lang_map[lang] != '']

        try:
            self.set_source(lang_source)
            self.set_target(lang_target)
        except InvalidSourceOrTargetLanguage as e:
            if raise_unsupported_lang:
                raise e
            if not self.supported_src_list or not self.supported_tgt_list:
                raise e
            lang_source = self.supported_src_list[0]
            lang_target = self.supported_tgt_list[0]
            self.set_source(lang_source)
            self.set_target(lang_target)

    def _setup_translator(self):
        raise NotImplementedError

    def setup_translator(self):
        self._setup_translator()

    @check_language_support(check_type='source')
    def set_source(self, lang: str):
        self.lang_source = lang

    @check_language_support(check_type='target')
    def set_target(self, lang: str):
        self.lang_target = lang

    def _translate(self, src_list: List[str]) -> List[str]:
        raise NotImplementedError

    def translate(self, text: Union[str, List]) -> Union[str, List]:
        if text_is_empty(text):
            return text

        is_list = isinstance(text, List)
        concate_text = is_list and self.concate_text
        text_source = self.textlist2text(text) if concate_text else text
        
        src_is_list = isinstance(text_source, List)
        if src_is_list: 
            text_trans = self._translate(text_source)
        else:
            text_trans = self._translate([text_source])[0]
        
        if text_trans is None:
            if is_list:
                text_trans = [''] * len(text)
            else:
                text_trans = ''
        elif concate_text:
            text_trans = self.text2textlist(text_trans)
            
        if is_list:
            try:
                assert len(text_trans) == len(text)
            except:
                LOGGER.error('This translator seems to messed up the translation which resulted in inconsistent translated line count.\n \
                             Set concate_text to False or change textblk_break in the source code may solve the problem.')
                raise

        return text_trans

    def textlist2text(self, text_list: List[str]) -> str:
        # some translators automatically strip '\n'
        # so we insert '\n###\n' between concated text instead of '\n' to avoid mismatch
        return self.textblk_break.join(text_list)

    def text2textlist(self, text: str) -> List[str]:
        breaker = self.textblk_break.replace('\n', '') or '\n'
        text_list = text.split(breaker)
        return [text.lstrip().rstrip() for text in text_list]

    def translate_textblk_lst(self, textblk_lst: List[TextBlock]):
        '''
        only textblks with non-empty source text would be passed to translator
        '''
        non_empty_ids = []
        text_list = []
        translations = []
        for ii, blk in enumerate(textblk_lst):
            text = blk.get_text()
            if text.strip() != '':
                non_empty_ids.append(ii)
                text_list.append(text)
            translations.append(text)

        # non_empty_txtlst_str = ',\n'.join(text_list)
        # LOGGER.debug(f'non empty src text list: \n[{non_empty_txtlst_str}]')

        for callback_name, callback in self._preprocess_hooks.items():
            callback(translations = translations, textblocks = textblk_lst, translator = self, source_text = text_list)

        if len(text_list) > 0:
            # Optional deterministic translation caching
            use_cache = bool(getattr(pcfg.module, "translation_cache_enabled", False))
            det_only = bool(getattr(pcfg.module, "translation_cache_deterministic_only", True))
            cache_hit = None
            cache = None
            cache_key = None
            if use_cache and (not det_only or self.is_deterministic()):
                try:
                    cache = TranslationCache()
                    cache_key = self._build_translation_cache_key(text_list)
                    cache_hit = cache.get(cache_key)
                except Exception:
                    cache_hit = None

            if cache_hit and isinstance(cache_hit, dict) and cache_hit.get("translations") and cache_hit.get("sources") == text_list:
                _translations = cache_hit.get("translations")
            else:
                _translations = self.translate(text_list)
                if cache is not None and cache_key is not None:
                    try:
                        cache.set(cache_key, {"sources": text_list, "translations": _translations})
                    except Exception:
                        pass
            for ii, idx in enumerate(non_empty_ids):
                translations[idx] = _translations[ii]

        for callback_name, callback in self._postprocess_hooks.items():
            callback(translations = translations, textblocks = textblk_lst, translator = self)

        for tr, blk in zip(translations, textblk_lst):
            blk.translation = sanitize_translation_text(tr)

    def set_translation_context(self, previous_pages=None, project_glossary=None, series_context_path=None, next_page=None):
        """Optional: set cross-page context and project glossary before translate_textblk_lst. next_page: optional {"sources": [...]} for next page context. Base implementation does nothing."""
        # Store a context blob for cache keys (subclasses may override and still call super()).
        try:
            self._cache_translation_context = {
                "previous_pages": previous_pages or [],
                "project_glossary": project_glossary or [],
                "series_context_path": (series_context_path or "") if series_context_path is not None else None,
                "next_page": next_page if next_page and isinstance(next_page, dict) else None,
            }
        except Exception:
            self._cache_translation_context = None

    def is_deterministic(self) -> bool:
        """
        Whether translations are expected to be deterministic for the current settings.
        LLM-based translators should override (e.g. temperature==0).
        """
        return True

    def _cache_relevant_params(self) -> dict:
        """Return a stable, non-secret subset of params for cache keys."""
        out = {}
        for k, v in (getattr(self, "params", {}) or {}).items():
            if k.lower() in {"apikey", "api_key", "multiple_keys", "multiple keys", "key", "keys", "proxy"}:
                continue
            if isinstance(v, dict) and "value" in v:
                out[k] = v.get("value")
            else:
                out[k] = v
        return out

    def _build_translation_cache_key(self, sources: List[str]) -> dict:
        return {
            "translator": getattr(self, "name", self.__class__.__name__),
            "src": getattr(self, "lang_source", ""),
            "tgt": getattr(self, "lang_target", ""),
            "params": self._cache_relevant_params(),
            "context": getattr(self, "_cache_translation_context", None),
            "sources": sources,
        }

    def append_page_to_series_context(self, series_context_path: str, sources: list, translations: list) -> None:
        """Optional: append translated page to series store. Base implementation does nothing."""
        pass

    def supported_languages(self) -> List[str]:
        return self.valid_lang_list

    @property
    def supported_tgt_list(self) -> List[str]:
        return self.valid_lang_list

    @property
    def supported_src_list(self) -> List[str]:
        return self.valid_lang_list
        
    def delay(self) -> float:
        if 'delay' in self.params:
            delay = self.params['delay']
            if delay:
                try:
                    return float(delay)
                except:
                    pass
        return 0.


@register_translator('None')
class TransNone(BaseTranslator):

    concate_text = False
    cht_require_convert = True
    params: Dict = {
        'description': 'Return existing translation'
    }

    def _setup_translator(self):
        for k in self.lang_map.keys():
            self.lang_map[k] = 'dummy language'
        
    def _translate(self, src_list: List[str]) -> List[str]:
        return copy.copy(src_list)
    
def transhook_copy_original(translations: List[str] = None, textblocks: List[TextBlock] = None, translator: BaseTranslator = None, **kwargs):
    if textblocks is not None and isinstance(translator, TransNone):
        for ii, _ in enumerate(translations):
            translations[ii] = textblocks[ii].translation

TransNone.register_postprocess_hooks({'copy_original': transhook_copy_original})


@register_translator('Copy Source')
class TransSource(BaseTranslator):

    concate_text = False
    cht_require_convert = True
    params: Dict = {
        'description': 'Return source text as translation'
    }

    def _setup_translator(self):
        for k in self.lang_map.keys():
            self.lang_map[k] = 'dummy language'
        self.register_preprocess_hooks
        
    def _translate(self, src_list: List[str]) -> List[str]:
        return copy.copy(src_list)