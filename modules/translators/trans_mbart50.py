"""
mBART-50 – Meta's multilingual translation (50 languages). Many-to-many, no English pivot.
facebook/mbart-large-50-many-to-many-mmt. Good for Chinese↔English and 48 other languages.
Requires: pip install transformers torch
"""
from .base import BaseTranslator, register_translator, DEVICE_SELECTOR
from .exceptions import MissingTranslatorParams
from typing import List, Dict

# App language display name -> mBART50 tokenizer lang code (e.g. en_XX, zh_CN)
MBART50_LANG_MAP: Dict[str, str] = {
    "简体中文": "zh_CN",
    "繁體中文": "zh_CN",
    "English": "en_XX",
    "日本語": "ja_XX",
    "한국어": "ko_KR",
    "Français": "fr_XX",
    "Deutsch": "de_DE",
    "Español": "es_XX",
    "Italiano": "it_IT",
    "Português": "pt_XX",
    "русский язык": "ru_RU",
    "Arabic": "ar_AR",
    "Thai": "th_TH",
    "Tiếng Việt": "vi_VN",
    "Hindi": "hi_IN",
    "Polski": "pl_PL",
    "čeština": "cs_CZ",
    "Nederlands": "nl_XX",
    "Türk dili": "tr_TR",
    "украї́нська мо́ва": "uk_UA",
    "Bengali": "bn_IN",
    "Tamil": "ta_IN",
    "Persian": "fa_IR",
    "Urdu": "ur_PK",
    "Malay": "id_ID",  # mBART50 has id_ID; use for Malay if needed; no ms
    "Indonesian": "id_ID",
    "Filipino": "tl_XX",
    "Khmer": "km_KH",
    "Burmese": "my_MM",
    "Gujarati": "gu_IN",
    "Telugu": "te_IN",
    "Marathi": "mr_IN",
    "Hebrew": "he_IL",
    "Nepali": "ne_NP",
    "Swahili": "sw_KE",
    "Swedish": "sv_SE",
    "Romanian": "ro_RO",
    "Finnish": "fi_FI",
    "Estonian": "et_EE",
    "Lithuanian": "lt_LT",
    "Latvian": "lv_LV",
    "Sinhala": "si_LK",
    "Afrikaans": "af_ZA",
    "Azerbaijani": "az_AZ",
    "Croatian": "hr_HR",
    "Georgian": "ka_GE",
    "Kazakh": "kk_KZ",
    "Mongolian": "mn_MN",
    "Pashto": "ps_AF",
    "Galician": "gl_ES",
    "Slovenian": "sl_SI",
    "Xhosa": "xh_ZA",
    "Malayalam": "ml_IN",
}
# Malay: mBART50 has no ms_XX; use id_ID as fallback for Malay
if "Malay" in MBART50_LANG_MAP:
    MBART50_LANG_MAP["Malay"] = "id_ID"
MBART50_SUPPORTED = list(MBART50_LANG_MAP.keys())


@register_translator("mBART50")
class MBart50Translator(BaseTranslator):
    """mBART-50: Meta multilingual translation (50 languages). Many-to-many, MT-centric. Good Zh↔En."""

    concate_text = False
    cht_require_convert = True
    _load_model_keys = {"model", "tokenizer"}
    params: Dict = {
        "device": DEVICE_SELECTOR(),
    }

    def _setup_translator(self):
        import torch
        from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

        model_id = "facebook/mbart-large-50-many-to-many-mmt"
        device = self.params.get("device", {}).get("value", "cpu")
        if device in ("cuda", "gpu") and torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        self._device = device
        self.tokenizer = MBart50TokenizerFast.from_pretrained(model_id)
        self.model = MBartForConditionalGeneration.from_pretrained(model_id)
        self.model.to(device)
        self.model.eval()
        self.lang_map.clear()
        for k, v in MBART50_LANG_MAP.items():
            self.lang_map[k] = v

    def _translate(self, src_list: List[str]) -> List[str]:
        if not src_list:
            return []
        if not hasattr(self, "model") or self.model is None or not hasattr(self, "tokenizer"):
            self.setup_translator()

        src_code = MBART50_LANG_MAP.get(self.lang_source) or self.lang_map.get(self.lang_source)
        tgt_code = MBART50_LANG_MAP.get(self.lang_target) or self.lang_map.get(self.lang_target)
        if not src_code or not tgt_code:
            raise MissingTranslatorParams(
                f"mBART50: unsupported pair {self.lang_source} -> {self.lang_target}. "
                f"Supported: {', '.join(MBART50_SUPPORTED[:20])}..."
            )
        if tgt_code not in self.tokenizer.lang_code_to_id:
            raise MissingTranslatorParams(f"mBART50: target code {tgt_code} not in tokenizer.")
        forced_bos_id = self.tokenizer.lang_code_to_id[tgt_code]

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
        return MBART50_SUPPORTED

    @property
    def supported_tgt_list(self) -> List[str]:
        return MBART50_SUPPORTED

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        if param_key == "device":
            for attr in ("model", "tokenizer", "_device"):
                if hasattr(self, attr):
                    delattr(self, attr)
