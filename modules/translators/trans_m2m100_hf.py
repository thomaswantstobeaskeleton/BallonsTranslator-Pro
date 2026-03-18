"""
M2M100 (Hugging Face) – Meta's many-to-many translation via transformers. 100 languages.
Use 418M for low VRAM (~1–2GB) or 1.2B for better quality. Direct language pairs (no English pivot).
Requires: pip install transformers torch sentencepiece
"""
from .base import BaseTranslator, register_translator, DEVICE_SELECTOR
from .exceptions import MissingTranslatorParams
from typing import List, Dict

# App language display name -> M2M100 2-letter code (tokenizer.get_lang_id / src_lang)
M2M100_HF_LANG_MAP: Dict[str, str] = {
    "简体中文": "zh",
    "繁體中文": "zh",
    "English": "en",
    "日本語": "ja",
    "한국어": "ko",
    "Français": "fr",
    "Deutsch": "de",
    "Español": "es",
    "Italiano": "it",
    "Português": "pt",
    "русский язык": "ru",
    "Arabic": "ar",
    "Thai": "th",
    "Tiếng Việt": "vi",
    "Hindi": "hi",
    "Polski": "pl",
    "čeština": "cs",
    "Nederlands": "nl",
    "Türk dili": "tr",
    "украї́нська мо́ва": "uk",
    "Bengali": "bn",
    "Tamil": "ta",
    "Persian": "fa",
    "Urdu": "ur",
    "Malay": "ms",
    "Indonesian": "id",
    "Filipino": "tl",
    "Khmer": "km",
    "Burmese": "my",
    "Gujarati": "gu",
    "Telugu": "te",
    "Marathi": "mr",
    "Hebrew": "he",
    "Swahili": "sw",
    "Amharic": "am",
    "Afrikaans": "af",
    "Albanian": "sq",
    "Azerbaijani": "az",
    "Belarusian": "be",
    "Bosnian": "bs",
    "Bulgarian": "bg",
    "Catalan": "ca",
    "Cebuano": "ceb",
    "Central Khmer": "km",
    "Chinese": "zh",
    "Croatian": "hr",
    "Danish": "da",
    "Estonian": "et",
    "Finnish": "fi",
    "Gaelic": "gd",
    "Galician": "gl",
    "Georgian": "ka",
    "Greek": "el",
    "Hungarian": "hu",
    "Icelandic": "is",
    "Irish": "ga",
    "Javanese": "jv",
    "Kannada": "kn",
    "Kazakh": "kk",
    "Lao": "lo",
    "Latvian": "lv",
    "Lithuanian": "lt",
    "Macedonian": "mk",
    "Malagasy": "mg",
    "Malayalam": "ml",
    "Mongolian": "mn",
    "Nepali": "ne",
    "Norwegian": "no",
    "Romanian": "ro",
    "Serbian": "sr",
    "Slovak": "sk",
    "Slovenian": "sl",
    "Somali": "so",
    "Swedish": "sv",
    "Tagalog": "tl",
    "Uzbek": "uz",
    "Welsh": "cy",
    "Xhosa": "xh",
    "Yoruba": "yo",
    "Zulu": "zu",
}
M2M100_HF_SUPPORTED = list(M2M100_HF_LANG_MAP.keys())


@register_translator("M2M100_HF")
class M2M100HFTranslator(BaseTranslator):
    """M2M100 (Hugging Face): Meta many-to-many translation. 418M = low VRAM; 1.2B = better quality."""

    concate_text = False
    cht_require_convert = True
    _load_model_keys = {"model", "tokenizer"}
    params: Dict = {
        "model_name": {
            "type": "selector",
            "options": [
                "facebook/m2m100_418M",
                "facebook/m2m100_1.2B",
            ],
            "value": "facebook/m2m100_418M",
            "description": "418M = low VRAM (~1–2GB). 1.2B = better quality, more VRAM.",
        },
        "device": DEVICE_SELECTOR(),
    }

    def _setup_translator(self):
        import torch
        from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

        model_name = self.params.get("model_name", {}).get("value", "facebook/m2m100_418M")
        device = self.params.get("device", {}).get("value", "cpu")
        if device in ("cuda", "gpu") and torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        self._device = device
        self.tokenizer = M2M100Tokenizer.from_pretrained(model_name)
        self.model = M2M100ForConditionalGeneration.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        self.lang_map.clear()
        for k, v in M2M100_HF_LANG_MAP.items():
            self.lang_map[k] = v

    def _translate(self, src_list: List[str]) -> List[str]:
        if not src_list:
            return []
        if not hasattr(self, "model") or self.model is None or not hasattr(self, "tokenizer"):
            self.setup_translator()

        src_code = M2M100_HF_LANG_MAP.get(self.lang_source) or self.lang_map.get(self.lang_source)
        tgt_code = M2M100_HF_LANG_MAP.get(self.lang_target) or self.lang_map.get(self.lang_target)
        if not src_code or not tgt_code:
            raise MissingTranslatorParams(
                f"M2M100 HF: unsupported pair {self.lang_source} -> {self.lang_target}. "
                f"Supported: {', '.join(M2M100_HF_SUPPORTED[:25])}..."
            )

        import torch

        self.tokenizer.src_lang = src_code
        inputs = self.tokenizer(
            src_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        forced_bos_id = self.tokenizer.get_lang_id(tgt_code)
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
        return M2M100_HF_SUPPORTED

    @property
    def supported_tgt_list(self) -> List[str]:
        return M2M100_HF_SUPPORTED

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        if param_key in ("model_name", "device"):
            for attr in ("model", "tokenizer"):
                if hasattr(self, attr):
                    delattr(self, attr)
