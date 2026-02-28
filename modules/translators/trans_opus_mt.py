"""
OPUS-MT – Helsinki-NLP multilingual translation (Hugging Face).
One model per language pair; lightweight and fast. Many pairs available.
Requires: pip install transformers torch
"""
from .base import *
from .exceptions import MissingTranslatorParams
from typing import List, Dict, Tuple

# (source_display_name, target_display_name) -> Hugging Face model id
OPUS_MT_PAIRS: Dict[Tuple[str, str], str] = {
    ('日本語', 'English'): 'Helsinki-NLP/opus-mt-ja-en',
    ('English', '日本語'): 'Helsinki-NLP/opus-mt-en-ja',
    ('简体中文', 'English'): 'Helsinki-NLP/opus-mt-zh-en',
    ('English', '简体中文'): 'Helsinki-NLP/opus-mt-en-zh',
    ('繁體中文', 'English'): 'Helsinki-NLP/opus-mt-zh-en',
    ('English', '繁體中文'): 'Helsinki-NLP/opus-mt-en-zh',
    ('한국어', 'English'): 'Helsinki-NLP/opus-mt-ko-en',
    ('English', '한국어'): 'Helsinki-NLP/opus-mt-en-ko',
    ('Français', 'English'): 'Helsinki-NLP/opus-mt-fr-en',
    ('English', 'Français'): 'Helsinki-NLP/opus-mt-en-fr',
    ('Deutsch', 'English'): 'Helsinki-NLP/opus-mt-de-en',
    ('English', 'Deutsch'): 'Helsinki-NLP/opus-mt-en-de',
    ('Español', 'English'): 'Helsinki-NLP/opus-mt-es-en',
    ('English', 'Español'): 'Helsinki-NLP/opus-mt-en-es',
    ('Italiano', 'English'): 'Helsinki-NLP/opus-mt-it-en',
    ('English', 'Italiano'): 'Helsinki-NLP/opus-mt-en-it',
    ('Português', 'English'): 'Helsinki-NLP/opus-mt-pt-en',
    ('English', 'Português'): 'Helsinki-NLP/opus-mt-en-pt',
    ('русский язык', 'English'): 'Helsinki-NLP/opus-mt-ru-en',
    ('English', 'русский язык'): 'Helsinki-NLP/opus-mt-en-ru',
    ('Tiếng Việt', 'English'): 'Helsinki-NLP/opus-mt-vi-en',
    ('English', 'Tiếng Việt'): 'Helsinki-NLP/opus-mt-en-vi',
    ('Nederlands', 'English'): 'Helsinki-NLP/opus-mt-nl-en',
    ('English', 'Nederlands'): 'Helsinki-NLP/opus-mt-en-nl',
    ('Polski', 'English'): 'Helsinki-NLP/opus-mt-pl-en',
    ('English', 'Polski'): 'Helsinki-NLP/opus-mt-en-pl',
    ('Türk dili', 'English'): 'Helsinki-NLP/opus-mt-tr-en',
    ('English', 'Türk dili'): 'Helsinki-NLP/opus-mt-en-tr',
    ('Arabic', 'English'): 'Helsinki-NLP/opus-mt-ar-en',
    ('English', 'Arabic'): 'Helsinki-NLP/opus-mt-en-ar',
    ('Indonesian', 'English'): 'Helsinki-NLP/opus-mt-id-en',
    ('English', 'Indonesian'): 'Helsinki-NLP/opus-mt-en-id',
    ('Thai', 'English'): 'Helsinki-NLP/opus-mt-th-en',
    ('English', 'Thai'): 'Helsinki-NLP/opus-mt-en-th',
    ('Czech', 'English'): 'Helsinki-NLP/opus-mt-cs-en',
    ('English', 'Czech'): 'Helsinki-NLP/opus-mt-en-cs',
    ('Greek', 'English'): 'Helsinki-NLP/opus-mt-el-en',
    ('English', 'Greek'): 'Helsinki-NLP/opus-mt-en-el',
    ('Hungarian', 'English'): 'Helsinki-NLP/opus-mt-hu-en',
    ('English', 'Hungarian'): 'Helsinki-NLP/opus-mt-en-hu',
    ('Swedish', 'English'): 'Helsinki-NLP/opus-mt-sv-en',
    ('English', 'Swedish'): 'Helsinki-NLP/opus-mt-en-sv',
    ('Romanian', 'English'): 'Helsinki-NLP/opus-mt-ro-en',
    ('English', 'Romanian'): 'Helsinki-NLP/opus-mt-en-ro',
    ('Danish', 'English'): 'Helsinki-NLP/opus-mt-da-en',
    ('English', 'Danish'): 'Helsinki-NLP/opus-mt-en-da',
    ('Finnish', 'English'): 'Helsinki-NLP/opus-mt-fi-en',
    ('English', 'Finnish'): 'Helsinki-NLP/opus-mt-en-fi',
    ('Norwegian', 'English'): 'Helsinki-NLP/opus-mt-nb-en',
    ('English', 'Norwegian'): 'Helsinki-NLP/opus-mt-en-nb',
    ('Ukrainian', 'English'): 'Helsinki-NLP/opus-mt-uk-en',
    ('English', 'Ukrainian'): 'Helsinki-NLP/opus-mt-en-uk',
}

# All languages that appear in any pair (for supported_src_list / supported_tgt_list)
OPUS_MT_LANGS = sorted(set(l for pair in OPUS_MT_PAIRS for l in pair))


@register_translator('opus_mt')
class OPUSMTTranslator(BaseTranslator):
    """OPUS-MT: Helsinki-NLP translation models. One model per language pair."""
    concate_text = True
    _load_model_keys = {"model", "tokenizer"}
    params: Dict = {
        'device': DEVICE_SELECTOR(),
    }

    def _get_model_id(self) -> str:
        key = (self.lang_source, self.lang_target)
        model_id = OPUS_MT_PAIRS.get(key)
        if not model_id:
            raise MissingTranslatorParams(
                f"OPUS-MT: no model for {self.lang_source} -> {self.lang_target}. "
                "Choose a supported pair (e.g. Japanese <-> English, Chinese <-> English)."
            )
        return model_id

    def _setup_translator(self):
        import torch
        from transformers import MarianMTModel, MarianTokenizer
        model_id = self._get_model_id()
        device = self.params.get('device', {}).get('value', 'cpu')
        if device in ('cuda', 'gpu') and torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        self._device = device
        self._model_id = model_id
        self.tokenizer = MarianTokenizer.from_pretrained(model_id)
        self.model = MarianMTModel.from_pretrained(model_id)
        self.model.to(device)
        self.model.eval()

    def _translate(self, src_list: List[str]) -> List[str]:
        if not src_list:
            return []
        model_id = self._get_model_id()
        if getattr(self, '_model_id', None) != model_id:
            self._model_id = model_id
            if hasattr(self, 'model'):
                delattr(self, 'model')
                delattr(self, 'tokenizer')
            self._setup_translator()
        import torch
        inputs = self.tokenizer(
            src_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model.generate(**inputs, max_length=512, num_beams=5)
        return self.tokenizer.batch_decode(out, skip_special_tokens=True)

    @property
    def supported_src_list(self) -> List[str]:
        return OPUS_MT_LANGS

    @property
    def supported_tgt_list(self) -> List[str]:
        return OPUS_MT_LANGS

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        if param_key == 'device' and hasattr(self, 'model'):
            delattr(self, 'model')
            delattr(self, 'tokenizer')
