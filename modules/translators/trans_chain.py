# Translator chain (#926): run multiple translators in sequence (e.g. Japanese -> English -> Chinese).
from .base import *
from typing import List
from utils.config import pcfg
from utils.series_context_store import get_series_context_dir, append_page_to_series_context as store_append_page, ensure_series_dir

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
        'chain_llm_review_mode': {
            'type': 'checkbox',
            'value': True,
            'description': 'When the final step is LLM_API_Translator, pass both original text and first-step draft so the LLM can review/correct it (2-step translation validation).',
        },
        'chain_llm_review_instruction': {
            'type': 'editor',
            'value': 'Improve the draft translation using the source text. Keep meaning accurate, natural, and concise; preserve names and terms consistently.',
            'description': 'Instruction prepended when chain_llm_review_mode is enabled.',
        },
        'description': 'Run multiple translators in sequence. Set chain_translators (e.g. google,trans_llm_api) and chain_intermediate_langs (e.g. English).',
    }

    def _setup_translator(self):
        for k in self.lang_map.keys():
            self.lang_map[k] = k

    def append_page_to_series_context(self, series_context_path: str, sources: List[str], translations: List[str]) -> None:
        """Append final chain output to series context store."""
        path = get_series_context_dir((series_context_path or "").strip())
        if not path or not sources:
            return
        ensure_series_dir(path)
        store_append_page(path, sources, translations, max_stored_pages=15)

    def _get_chain_config(self):
        raw = (self.get_param_value('chain_translators') or 'google').strip()
        names = [n.strip() for n in raw.split(',') if n.strip()]
        raw_langs = (self.get_param_value('chain_intermediate_langs') or '').strip()
        intermediate = [l.strip() for l in raw_langs.split(',') if l.strip()]
        return names, intermediate


    def _as_bool(self, value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        return str(value).strip().lower() in {'1', 'true', 'yes', 'on'}

    def _build_llm_review_inputs(self, sources: List[str], drafts: List[str], review_instruction: str) -> List[str]:
        payloads: List[str] = []
        for src, draft in zip(sources, drafts):
            payloads.append(
                f"{review_instruction}\n\n"
                f"[Source Text]\n{src}\n\n"
                f"[Draft Translation]\n{draft}\n\n"
                "Return only the improved final translation text."
            )
        return payloads

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

        exclude = {'Chain', 'None', 'Copy Source', 'Ensemble (3+1)', 'Chimera (multi-source)'}
        current = list(src_list)
        original_sources = list(src_list)
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
                    # Flatten selector params so sub-translators get scalar values
                    flat = {k: v.get("value", v) if isinstance(v, dict) and "value" in v else v for k, v in params.items() if not (isinstance(k, str) and k.startswith("__"))}
                    trans = trans_cls(src_lang, tgt_lang, raise_unsupported_lang=False, **flat)
                else:
                    trans = trans_cls(src_lang, tgt_lang, raise_unsupported_lang=False)
            except Exception as e:
                self.logger.error(f"Chain: failed to create '{tname}': {e}")
                break
            # Forward all context/settings the pipeline gives the main translator (same as LLM API translator gets)
            ctx = getattr(self, "_cache_translation_context", None)
            if ctx and hasattr(trans, "set_translation_context"):
                try:
                    trans.set_translation_context(
                        previous_pages=ctx.get("previous_pages") or [],
                        project_glossary=ctx.get("project_glossary") or [],
                        series_context_path=ctx.get("series_context_path"),
                        next_page=ctx.get("next_page"),
                    )
                except Exception:
                    pass
            if ctx is not None:
                try:
                    setattr(trans, "_cache_translation_context", ctx)
                except Exception:
                    pass
            for attr in ("_current_page_key", "_current_page_image"):
                if hasattr(self, attr):
                    try:
                        setattr(trans, attr, getattr(self, attr))
                    except Exception:
                        pass
            step_inputs = current
            if (
                i == len(names) - 1
                and tname == 'LLM_API_Translator'
                and self._as_bool(self.get_param_value('chain_llm_review_mode'))
                and len(original_sources) == len(current)
            ):
                review_instruction = (self.get_param_value('chain_llm_review_instruction') or '').strip()
                if review_instruction:
                    step_inputs = self._build_llm_review_inputs(original_sources, current, review_instruction)
            try:
                current = trans.translate(step_inputs)
            except Exception as e:
                self.logger.error(f"Chain: step '{tname}' failed: {e}")
                break
            if not isinstance(current, list):
                current = [current] if isinstance(current, str) else []
            if len(current) != len(src_list):
                self.logger.warning(f"Chain: step '{tname}' returned length {len(current)}, expected {len(src_list)}.")
        return current
