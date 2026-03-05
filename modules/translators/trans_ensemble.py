"""
Ensemble translator: 3 small translation models + 1 LLM judge.
Gets 3 candidate translations per text, then asks the judge to pick the best or merge.
Default candidates (local, auto-download): nllb200, opus_mt, t5_mt.
Default judge: LLM_API_Translator (configure API key and model in its params).
"""
import copy
import json
import re
from typing import List, Dict, Optional

from .base import (
    BaseTranslator,
    register_translator,
    LANGMAP_GLOBAL,
    sanitize_translation_text,
)
from utils.config import pcfg
from utils.logger import logger as LOGGER


def _valid_candidate_translators():
    """Translator names that can be used as candidates (exclude ensemble, None, Copy Source)."""
    from modules import GET_VALID_TRANSLATORS
    exclude = {'Ensemble (3+1)', 'None', 'Copy Source'}
    return [t for t in GET_VALID_TRANSLATORS() if t not in exclude]


DEFAULT_JUDGE_SYSTEM = """You are a translation judge. You will be given a source text and 3 candidate translations (A, B, C).
Output the best translation, or a merged version if combining phrases from several is better.
Preserve meaning and natural style. For dialogue, keep tone consistent.
Reply with valid JSON only: {"translations": [{"id": 1, "translation": "your chosen or merged text"}]}.
One object per item. No markdown, no code fences, no explanation outside the JSON."""


@register_translator('Ensemble (3+1)')
class EnsembleTranslator(BaseTranslator):
    """Three small translators + one LLM judge to pick or merge the best translation."""
    concate_text = False
    cht_require_convert = True
    params: Dict = {
        'candidate_1': {
            'type': 'selector',
            'options': [],  # filled at runtime from all registered translators
            'value': 'google',
            'description': 'First candidate. For Chinese→English: Google, nllb200, or LLM_API_Translator (OpenRouter) work well.',
        },
        'candidate_2': {
            'type': 'selector',
            'options': [],
            'value': 'nllb200',
            'description': 'Second candidate. nllb200 is strong for many languages; LLM_API_Translator supports OpenRouter models.',
        },
        'candidate_3': {
            'type': 'selector',
            'options': [],
            'value': 'LLM_API_Translator',
            'description': 'Third candidate. Use LLM_API_Translator with OpenRouter (e.g. google/gemma-3n-e2b-it:free, qwen/qwen3-4b:free) for Zh→En.',
        },
        'judge_translator': {
            'type': 'selector',
            'options': [],
            'value': 'LLM_API_Translator',
            'description': 'Judge: LLM that picks/merges the best. Set provider=OpenRouter and model in LLM_API_Translator params (e.g. OpenRouter free models for Zh→En).',
        },
        'judge_system_prompt': {
            'type': 'editor',
            'value': DEFAULT_JUDGE_SYSTEM,
            'description': 'System prompt for the judge (must ask for JSON with translations array).',
        },
        'delay': {
            'value': 0.5,
            'description': 'Delay in seconds between judge API calls (avoid rate limits).',
        },
    }

    def _setup_translator(self):
        # Support same languages as typical LLM (broad)
        self.lang_map = LANGMAP_GLOBAL.copy()
        for k in list(self.lang_map.keys()):
            if self.lang_map[k] == '':
                self.lang_map[k] = k
        # Fill selector options from registry
        valid = _valid_candidate_translators()
        for key in ('candidate_1', 'candidate_2', 'candidate_3', 'judge_translator'):
            if key in self.params and isinstance(self.params[key], dict):
                opts = self.params[key].get('options')
                if not opts or len(opts) != len(valid):
                    self.params[key]['options'] = valid
        self._candidates = []
        self._judge = None

    def _get_merged_params(self) -> Dict:
        from modules import GET_VALID_TRANSLATORS, TRANSLATORS
        from modules.base import merge_config_module_params
        cfg = pcfg.module
        raw = getattr(cfg, 'translator_params', None) or {}
        return merge_config_module_params(
            copy.deepcopy(raw),
            GET_VALID_TRANSLATORS(),
            TRANSLATORS.get,
        )

    def _build_candidates_and_judge(self):
        if self._candidates:
            return
        from modules import GET_VALID_TRANSLATORS, TRANSLATORS
        merged = self._get_merged_params()
        c1_name = (self.get_param_value('candidate_1') or '').strip()
        c2_name = (self.get_param_value('candidate_2') or '').strip()
        c3_name = (self.get_param_value('candidate_3') or '').strip()
        judge_name = (self.get_param_value('judge_translator') or '').strip()
        valid = GET_VALID_TRANSLATORS()
        self._candidates = []
        for name in (c1_name, c2_name, c3_name):
            if not name or name not in valid:
                self._candidates.append(None)
                continue
            try:
                Klass = TRANSLATORS.get(name)
                params = merged.get(name, {})
                if isinstance(params, dict):
                    params = {k: v for k, v in params.items() if not (isinstance(k, str) and k.startswith('__'))}
                inst = Klass(
                    lang_source=self.lang_source,
                    lang_target=self.lang_target,
                    raise_unsupported_lang=False,
                    **params,
                )
                self._candidates.append(inst)
            except Exception as e:
                LOGGER.warning('Ensemble: could not create candidate %s: %s', name, e)
                self._candidates.append(None)
        self._judge = None
        if judge_name and judge_name in valid:
            try:
                Klass = TRANSLATORS.get(judge_name)
                params = merged.get(judge_name, {})
                if isinstance(params, dict):
                    params = {k: v for k, v in params.items() if not (isinstance(k, str) and k.startswith('__'))}
                self._judge = Klass(
                    lang_source=self.lang_source,
                    lang_target=self.lang_target,
                    raise_unsupported_lang=False,
                    **params,
                )
            except Exception as e:
                LOGGER.warning('Ensemble: could not create judge %s: %s', judge_name, e)

    def _translate_one_candidate(self, candidate, src_list: List[str], candidate_label: str = '') -> List[str]:
        if candidate is None or not callable(getattr(candidate, 'translate', None)):
            if candidate_label:
                LOGGER.debug('Ensemble candidate %s skipped (not available or no translate method)', candidate_label)
            return ['[candidate failed]'] * len(src_list)
        try:
            return candidate.translate(src_list)
        except Exception as e:
            LOGGER.warning('Ensemble candidate %s translate failed: %s', candidate_label or '?', e)
            return ['[candidate failed]'] * len(src_list)

    def _parse_judge_json(self, raw: str, expected_count: int) -> Optional[List[str]]:
        if not raw or not raw.strip():
            return None
        raw = raw.strip()
        start = raw.find('{')
        if start == -1:
            return None
        depth = 0
        for i in range(start, len(raw)):
            if raw[i] == '{':
                depth += 1
            elif raw[i] == '}':
                depth -= 1
                if depth == 0:
                    raw = raw[start : i + 1]
                    break
        try:
            data = json.loads(raw)
            trans = data.get('translations') or data.get('translation')
            if isinstance(trans, str):
                trans = [trans]
            if not trans or not isinstance(trans, list):
                return None
            id_to_text = {}
            for t in trans:
                if isinstance(t, dict):
                    tid = t.get('id', len(id_to_text) + 1)
                    id_to_text[tid] = (t.get('translation') or '').strip()
                elif isinstance(t, str):
                    id_to_text[len(id_to_text) + 1] = t.strip()
            out = [id_to_text.get(i + 1, '') for i in range(expected_count)]
            if len(out) == expected_count:
                return out
            return None
        except (json.JSONDecodeError, TypeError):
            pattern = re.compile(r'^\s*(\d+)\s*[.:)]\s*(.*)$', re.MULTILINE)
            found = {}
            for m in pattern.finditer(raw):
                found[int(m.group(1))] = (m.group(2) or '').strip()
            if found and max(found.keys()) >= 1:
                return [found.get(i + 1, '') for i in range(expected_count)]
            return None

    def _translate(self, src_list: List[str]) -> List[str]:
        if not src_list:
            return []
        self._build_candidates_and_judge()
        c1_name = (self.get_param_value('candidate_1') or '').strip() or '1'
        c2_name = (self.get_param_value('candidate_2') or '').strip() or '2'
        c3_name = (self.get_param_value('candidate_3') or '').strip() or '3'
        t1 = self._translate_one_candidate(
            self._candidates[0] if len(self._candidates) > 0 else None, src_list, c1_name)
        t2 = self._translate_one_candidate(
            self._candidates[1] if len(self._candidates) > 1 else None, src_list, c2_name)
        t3 = self._translate_one_candidate(
            self._candidates[2] if len(self._candidates) > 2 else None, src_list, c3_name)
        if self._judge is None:
            LOGGER.warning('Ensemble: no judge configured; using first candidate.')
            return t1
        judge_system = self.get_param_value('judge_system_prompt') or DEFAULT_JUDGE_SYSTEM
        delay = float(self.get_param_value('delay') or 0.5)
        user_parts = []
        for i, src in enumerate(src_list):
            user_parts.append(
                f"Item {i + 1}:\nSource: {src}\n"
                f"Translation A: {t1[i] if i < len(t1) else ''}\n"
                f"Translation B: {t2[i] if i < len(t2) else ''}\n"
                f"Translation C: {t3[i] if i < len(t3) else ''}\n"
            )
        user_prompt = "\n".join(user_parts) + "\nOutput JSON with one translation per item (id 1 to N)."
        if not hasattr(self._judge, 'request_custom_completion'):
            LOGGER.warning('Ensemble: judge does not support request_custom_completion; using first candidate.')
            return t1
        import time
        if delay > 0:
            time.sleep(delay)
        raw = self._judge.request_custom_completion(judge_system, user_prompt, max_tokens=4096)
        parsed = self._parse_judge_json(raw, len(src_list)) if raw else None
        if parsed and len(parsed) == len(src_list):
            return parsed
        LOGGER.warning('Ensemble: judge response parse failed or count mismatch; using first candidate.')
        return t1

    @property
    def supported_src_list(self) -> List[str]:
        return list(LANGMAP_GLOBAL.keys())

    @property
    def supported_tgt_list(self) -> List[str]:
        return list(LANGMAP_GLOBAL.keys())

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        if param_key in ('candidate_1', 'candidate_2', 'candidate_3', 'judge_translator'):
            self._candidates = []
            self._judge = None


# Fill selector options at load time so UI has a fallback; init_translator_registries() refreshes with full list after all translators load
try:
    _opts = _valid_candidate_translators()
    for _k in ('candidate_1', 'candidate_2', 'candidate_3', 'judge_translator'):
        if _k in EnsembleTranslator.params and isinstance(EnsembleTranslator.params[_k], dict):
            EnsembleTranslator.params[_k]['options'] = _opts
except Exception:
    pass
