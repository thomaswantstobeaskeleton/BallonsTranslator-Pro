# Translator chain (#926): run multiple translators in sequence (e.g. Japanese -> English -> Chinese).
from .base import *
from typing import List
from utils.config import pcfg

cfg_module = pcfg.module


@register_translator('Chain')
class TransChain(BaseTranslator):
    """Run a sequence of translators. E.g. Japanese -> English (Google) -> Chinese (LLM)."""
    concate_text = True
    cht_require_convert = False
    params: Dict = {
        'chain_translators': {
            'type': 'text',
            'value': 'google,trans_llm_api',
            'description': 'Comma-separated translator names (e.g. google,trans_llm_api). First translates source->intermediate, last translates intermediate->target.',
        },
        'chain_intermediate_langs': {
            'type': 'text',
            'value': 'English',
            'description': 'Comma-separated intermediate language(s). Must be n-1 for n translators (e.g. one for 2-step chain).',
        },
        'description': 'Run multiple translators in sequence. Set chain_translators (e.g. google,trans_llm_api) and chain_intermediate_langs (e.g. English).',
    }

    def _setup_translator(self):
        for k in self.lang_map.keys():
            self.lang_map[k] = k

    def _get_chain_config(self):
        raw = (self.get_param_value('chain_translators') or 'google').strip()
        names = [n.strip() for n in raw.split(',') if n.strip()]
        raw_langs = (self.get_param_value('chain_intermediate_langs') or '').strip()
        intermediate = [l.strip() for l in raw_langs.split(',') if l.strip()]
        return names, intermediate

    def _translate(self, src_list: List[str]) -> List[str]:
        names, intermediate_langs = self._get_chain_config()
        if not names:
            return src_list
        # Build language sequence: source, [intermediates], target
        if len(intermediate_langs) != max(0, len(names) - 1):
            self.logger.warning(
                "Chain: intermediate_langs count should be len(chain_translators)-1. Using first intermediate for all steps."
            )
        lang_sequence = [self.lang_source]
        for i in range(len(names) - 1):
            lang_sequence.append(intermediate_langs[i] if i < len(intermediate_langs) else (intermediate_langs[0] if intermediate_langs else 'English'))
        lang_sequence.append(self.lang_target)

        exclude = {'Chain', 'None', 'Copy Source', 'Ensemble (3+1)'}
        current = list(src_list)
        for i, tname in enumerate(names):
            if tname in exclude:
                continue
            trans_cls = TRANSLATORS.module_dict.get(tname)
            if trans_cls is None:
                self.logger.warning(f"Chain: translator '{tname}' not found, skipping.")
                continue
            src_lang = lang_sequence[i]
            tgt_lang = lang_sequence[i + 1]
            params = (cfg_module.translator_params or {}).get(tname)
            try:
                if params is not None:
                    trans = trans_cls(src_lang, tgt_lang, raise_unsupported_lang=False, **params)
                else:
                    trans = trans_cls(src_lang, tgt_lang, raise_unsupported_lang=False)
            except Exception as e:
                self.logger.error(f"Chain: failed to create '{tname}': {e}")
                break
            try:
                current = trans.translate(current)
            except Exception as e:
                self.logger.error(f"Chain: step '{tname}' failed: {e}")
                break
            if not isinstance(current, list):
                current = [current] if isinstance(current, str) else []
            if len(current) != len(src_list):
                self.logger.warning(f"Chain: step '{tname}' returned length {len(current)}, expected {len(src_list)}.")
        return current
