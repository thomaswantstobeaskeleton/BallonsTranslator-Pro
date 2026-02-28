"""
T5 MT – Google T5 for translation via Hugging Face (prompt-based).
Use t5-small or t5-base; prompt: "translate X to Y: <text>".
Requires: pip install transformers torch
"""
from .base import *
from .exceptions import MissingTranslatorParams
from typing import List, Dict

# Display name -> short name for T5 prompt (e.g. "translate English to Japanese: ...")
T5_LANG_PROMPT_NAMES: Dict[str, str] = {
    'English': 'English',
    '简体中文': 'Chinese',
    '繁體中文': 'Chinese',
    '日本語': 'Japanese',
    '한국어': 'Korean',
    'Français': 'French',
    'Deutsch': 'German',
    'Español': 'Spanish',
    'Italiano': 'Italian',
    'Português': 'Portuguese',
    'русский язык': 'Russian',
    'Arabic': 'Arabic',
    'Hindi': 'Hindi',
    'Thai': 'Thai',
    'Tiếng Việt': 'Vietnamese',
    'Indonesian': 'Indonesian',
    'Nederlands': 'Dutch',
    'Polski': 'Polish',
    'Türk dili': 'Turkish',
}

T5_SUPPORTED = list(T5_LANG_PROMPT_NAMES.keys())


@register_translator('t5_mt')
class T5MTTranslator(BaseTranslator):
    """T5: Prompt-based translation (e.g. t5-small, t5-base). General-purpose."""
    concate_text = True
    _load_model_keys = {"model", "tokenizer"}
    params: Dict = {
        'model_name': {
            'type': 'selector',
            'options': [
                'google-t5/t5-small',
                'google-t5/t5-base',
            ],
            'value': 'google-t5/t5-small',
            'description': 'T5 model (small = faster/less VRAM, base = better quality).',
        },
        'device': DEVICE_SELECTOR(),
    }

    def _setup_translator(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        model_name = self.params.get('model_name', {}).get('value', 'google-t5/t5-small')
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
        for k in T5_SUPPORTED:
            self.lang_map[k] = T5_LANG_PROMPT_NAMES.get(k, k)

    def _translate(self, src_list: List[str]) -> List[str]:
        if not src_list:
            return []
        src_name = T5_LANG_PROMPT_NAMES.get(self.lang_source) or self.lang_map.get(self.lang_source)
        tgt_name = T5_LANG_PROMPT_NAMES.get(self.lang_target) or self.lang_map.get(self.lang_target)
        if not src_name or not tgt_name:
            raise MissingTranslatorParams(
                f"T5 MT: unsupported language pair {self.lang_source} -> {self.lang_target}."
            )
        import torch
        prefix = f"translate {src_name} to {tgt_name}: "
        inputs = self.tokenizer(
            [prefix + t for t in src_list],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model.generate(**inputs, max_length=512, num_beams=4)
        return self.tokenizer.batch_decode(out, skip_special_tokens=True)

    @property
    def supported_src_list(self) -> List[str]:
        return T5_SUPPORTED

    @property
    def supported_tgt_list(self) -> List[str]:
        return T5_SUPPORTED

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        if param_key in ('model_name', 'device') and hasattr(self, 'model'):
            delattr(self, 'model')
            delattr(self, 'tokenizer')
