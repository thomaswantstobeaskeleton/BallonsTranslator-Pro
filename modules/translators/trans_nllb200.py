"""
NLLB-200 – Facebook's No Language Left Behind 200-language translation (Hugging Face).
Local GPU/CPU. Use distilled 600M for less VRAM or 1.3B for quality.
Requires: pip install transformers torch
"""
from .base import *
from .exceptions import MissingTranslatorParams
import os
from typing import List, Dict

# NLLB uses BCP-47 style codes (e.g. eng_Latn, jpn_Jpan). Map UI names to these.
NLLB_LANG_MAP: Dict[str, str] = {
    'English': 'eng_Latn',
    '简体中文': 'zho_Hans',
    '繁體中文': 'zho_Hant',
    '日本語': 'jpn_Jpan',
    '한국어': 'kor_Hang',
    'Français': 'fra_Latn',
    'Deutsch': 'deu_Latn',
    'Español': 'spa_Latn',
    'Italiano': 'ita_Latn',
    'Português': 'por_Latn',
    'русский язык': 'rus_Cyrl',
    'Arabic': 'arb_Arab',
    'Hindi': 'hin_Deva',
    'Thai': 'tha_Thai',
    'Tiếng Việt': 'vie_Latn',
    'Indonesian': 'ind_Latn',
    'Nederlands': 'nld_Latn',
    'Polski': 'pol_Latn',
    'Türk dili': 'tur_Latn',
    'Persian': 'pes_Arab',
    'Bengali': 'ben_Beng',
    'Malay': 'zsm_Latn',
    'Swahili': 'swh_Latn',
    'Amharic': 'amh_Ethi',
    'Burmese': 'mya_Mymr',
    'Czech': 'ces_Latn',
    'Greek': 'ell_Grek',
    'Romanian': 'ron_Latn',
    'Hungarian': 'hun_Latn',
    'Swedish': 'swe_Latn',
    'Ukrainian': 'ukr_Cyrl',
    'Catalan': 'cat_Latn',
    'Danish': 'dan_Latn',
    'Finnish': 'fin_Latn',
    'Norwegian': 'nob_Latn',
    'Slovak': 'slk_Latn',
    'Bulgarian': 'bul_Cyrl',
    'Croatian': 'hrv_Latn',
    'Serbian': 'srp_Cyrl',
}

NLLB_SUPPORTED = list(NLLB_LANG_MAP.keys())


@register_translator('nllb200')
class NLLB200Translator(BaseTranslator):
    """NLLB-200: 200-language translation via Hugging Face. Local only."""
    concate_text = True
    _load_model_keys = {"model", "tokenizer"}
    params: Dict = {
        'model_name': {
            'type': 'selector',
            'options': [
                'facebook/nllb-200-distilled-600M',
                'facebook/nllb-200-distilled-1.3B',
            ],
            'value': 'facebook/nllb-200-distilled-600M',
            'description': 'NLLB model. 600M = low VRAM (~3GB), default. 1.3B = better quality, more VRAM.',
        },
        'device': DEVICE_SELECTOR(),
    }

    def _setup_translator(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        model_name = self.params.get('model_name', {}).get('value', 'facebook/nllb-200-distilled-600M')
        device = self.params.get('device', {}).get('value', 'cpu')
        if device in ('cuda', 'gpu') and torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        self._device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        self._model_name = model_name
        for k, v in NLLB_LANG_MAP.items():
            self.lang_map[k] = v

    def _translate(self, src_list: List[str]) -> List[str]:
        if not src_list:
            return []
        # Translator params can invalidate loaded objects (updateParam). Reload lazily.
        if not hasattr(self, 'model') or self.model is None or not hasattr(self, 'tokenizer') or self.tokenizer is None:
            self.setup_translator()
        src_lang = NLLB_LANG_MAP.get(self.lang_source) or self.lang_map.get(self.lang_source)
        tgt_lang = NLLB_LANG_MAP.get(self.lang_target) or self.lang_map.get(self.lang_target)
        if not src_lang or not tgt_lang:
            raise MissingTranslatorParams(
                f"NLLB200: unsupported language pair {self.lang_source} -> {self.lang_target}. "
                f"Supported: {', '.join(NLLB_SUPPORTED[:20])}..."
            )
        import torch
        self.tokenizer.src_lang = src_lang
        # forced_bos_token_id must be int; convert_tokens_to_ids may return list in some tokenizer versions
        forced_bos_id = None
        if hasattr(self.tokenizer, 'lang_code_to_id'):
            forced_bos_id = self.tokenizer.lang_code_to_id.get(tgt_lang)
        if forced_bos_id is None:
            raw = self.tokenizer.convert_tokens_to_ids(tgt_lang)
            forced_bos_id = raw[0] if isinstance(raw, (list, tuple)) else raw
        if not isinstance(forced_bos_id, int):
            raise MissingTranslatorParams(
                f"NLLB200: could not resolve token ID for target language {tgt_lang}. "
                "Try a different language pair or update transformers."
            )
        inputs = self.tokenizer(
            src_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_id,
                max_length=512,
                num_beams=5,
            )
        return self.tokenizer.batch_decode(out, skip_special_tokens=True)

    @property
    def supported_src_list(self) -> List[str]:
        return NLLB_SUPPORTED

    @property
    def supported_tgt_list(self) -> List[str]:
        return NLLB_SUPPORTED

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        if param_key in ('model_name', 'device'):
            if hasattr(self, 'model'):
                delattr(self, 'model')
            if hasattr(self, 'tokenizer'):
                delattr(self, 'tokenizer')
            if hasattr(self, '_model_name'):
                delattr(self, '_model_name')
