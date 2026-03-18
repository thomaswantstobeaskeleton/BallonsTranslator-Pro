"""
Tencent Hunyuan MT: HY-MT 1.5 7B (direct translation) and Hunyuan-MT-Chimera 7B (ensemble refiner).
Both support 33 languages. Chimera uses HY-MT 1.5 7B to generate 6 candidates, then refines with Chimera.
Requires: pip install transformers torch (transformers >= 4.56 recommended)
"""
from .base import BaseTranslator, register_translator, DEVICE_SELECTOR
from .exceptions import MissingTranslatorParams
from typing import List, Dict

# UI language name -> model language code (shared by 1.5 and Chimera)
HY_MT_LANG_MAP: Dict[str, str] = {
    "简体中文": "zh",
    "繁體中文": "zh-Hant",
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
    "Hebrew": "he",
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
    "Cantonese": "yue",
    "Tibetan": "bo",
    "Kazakh": "kk",
    "Mongolian": "mn",
    "Uyghur": "ug",
}

HY_MT_SUPPORTED = list(HY_MT_LANG_MAP.keys())

# Target language display names for prompts (Chinese names from official tables)
HY_MT_TARGET_DISPLAY: Dict[str, str] = {
    "zh": "中文",
    "zh-Hant": "繁体中文",
    "en": "英语",
    "ja": "日语",
    "ko": "韩语",
    "fr": "法语",
    "de": "德语",
    "es": "西班牙语",
    "it": "意大利语",
    "pt": "葡萄牙语",
    "ru": "俄语",
    "ar": "阿拉伯语",
    "th": "泰语",
    "vi": "越南语",
    "hi": "印地语",
    "pl": "波兰语",
    "cs": "捷克语",
    "nl": "荷兰语",
    "tr": "土耳其语",
    "uk": "乌克兰语",
    "he": "希伯来语",
    "bn": "孟加拉语",
    "ta": "泰米尔语",
    "fa": "波斯语",
    "ur": "乌尔都语",
    "ms": "马来语",
    "id": "印尼语",
    "tl": "菲律宾语",
    "km": "高棉语",
    "my": "缅甸语",
    "gu": "古吉拉特语",
    "te": "泰卢固语",
    "mr": "马拉地语",
    "yue": "粤语",
    "bo": "藏语",
    "kk": "哈萨克语",
    "mn": "蒙古语",
    "ug": "维吾尔语",
}


def _ensure_lang_codes(lang_source: str, lang_target: str):
    src = HY_MT_LANG_MAP.get(lang_source) or lang_source
    tgt = HY_MT_LANG_MAP.get(lang_target) or lang_target
    if not src or not tgt:
        raise MissingTranslatorParams(
            f"Unsupported language pair {lang_source} -> {lang_target}. "
            f"Supported: {', '.join(HY_MT_SUPPORTED[:15])}..."
        )
    return src, tgt


def _run_generation(model, tokenizer, prompt: str, device: str, add_generation_prompt: bool = True):
    """Tokenize prompt, generate, decode and strip prompt to get model reply."""
    import torch
    try:
        messages = [{"role": "user", "content": prompt}]
        tokenized = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_tensors="pt",
        )
    except Exception:
        tokenized = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).input_ids
    if isinstance(tokenized, dict):
        inp = {k: v.to(device) for k, v in tokenized.items()}
    else:
        tokenized = tokenized.to(device)
        inp = {"input_ids": tokenized}
    with torch.no_grad():
        out = model.generate(
            **inp,
            max_new_tokens=2048,
            do_sample=True,
            top_k=20,
            top_p=0.6,
            repetition_penalty=1.05,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    if isinstance(out, (list, tuple)):
        out = out[0] if out else None
    if out is None:
        return ""
    text = tokenizer.decode(out, skip_special_tokens=True)
    if prompt in text:
        text = text.split(prompt, 1)[-1]
    return text.strip()


# ---------- HY-MT 1.5 7B (direct translation) ----------

def _hy_mt15_prompt(source_text: str, src_code: str, tgt_code: str, tgt_display: str) -> str:
    """Official HY-MT1.5 prompt: ZH<=>XX uses Chinese instruction, else English."""
    if (src_code in ("zh", "zh-Hant")) or (tgt_code in ("zh", "zh-Hant")):
        return f"将以下文本翻译为{tgt_display}，注意只需要输出翻译后的结果，不要额外解释：\n\n{source_text}"
    return f"Translate the following segment into {tgt_display}, without additional explanation.\n\n{source_text}"


@register_translator("HY_MT_1_5_7B")
class HYMT15Translator(BaseTranslator):
    """HY-MT 1.5: Tencent Hunyuan Translation 1.5. Choose 1.8B (low RAM/VRAM) or 7B. 33 languages."""

    concate_text = False
    cht_require_convert = True
    _load_model_keys = {"model", "tokenizer"}
    params: Dict = {
        "model_name": {
            "type": "selector",
            "options": [
                "tencent/HY-MT1.5-1.8B",
                "tencent/HY-MT1.5-1.8B-FP8",
            ],
            "value": "tencent/HY-MT1.5-1.8B",
            "description": "1.8B only (≤1.5B class). Low RAM/VRAM (~2–3GB). FP8 = quantized.",
        },
        "device": DEVICE_SELECTOR(),
    }

    def _setup_translator(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model_name = self.params.get("model_name", {}).get("value", "tencent/HY-MT1.5-1.8B")
        device = self.params.get("device", {}).get("value", "cpu")
        if device in ("cuda", "gpu") and torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        self._device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        )
        self.model.to(device)
        self.model.eval()
        self.lang_map.clear()
        for k, v in HY_MT_LANG_MAP.items():
            self.lang_map[k] = v

    def _translate(self, src_list: List[str]) -> List[str]:
        if not src_list:
            return []
        if not hasattr(self, "model") or self.model is None or not hasattr(self, "tokenizer") or self.tokenizer is None:
            self.setup_translator()
        src_code, tgt_code = _ensure_lang_codes(self.lang_source, self.lang_target)
        tgt_display = HY_MT_TARGET_DISPLAY.get(tgt_code, tgt_code)
        results = []
        for source_text in src_list:
            prompt = _hy_mt15_prompt(source_text, src_code, tgt_code, tgt_display)
            text = _run_generation(self.model, self.tokenizer, prompt, self._device)
            results.append(text if text else source_text)
        return results

    @property
    def supported_src_list(self) -> List[str]:
        return HY_MT_SUPPORTED

    @property
    def supported_tgt_list(self) -> List[str]:
        return HY_MT_SUPPORTED

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        if param_key in ("model_name", "device"):
            for attr in ("model", "tokenizer", "_model_name"):
                if hasattr(self, attr):
                    delattr(self, attr)


# ---------- Hunyuan-MT-Chimera 7B (6 candidates from 1.5 + Chimera refine) ----------

def _chimera_prompt(source_text: str, candidates: List[str], source_lang: str, target_lang: str) -> str:
    """Official Chimera prompt: 6 candidate translations + source -> refined translation."""
    # Use English names for source/target in the prompt (Chimera README uses <source_language> <target_language>)
    src_name = HY_MT_TARGET_DISPLAY.get(source_lang, source_lang)
    tgt_name = HY_MT_TARGET_DISPLAY.get(target_lang, target_lang)
    lines = [
        f"Analyze the following multiple {tgt_name} translations of the {src_name} segment surrounded in triple backticks and generate a single refined {tgt_name} translation. Only output the refined translation, do not explain.",
        "",
        f"The {src_name} segment:",
        f"```{source_text}```",
        "",
        f"The multiple {tgt_name} translations:",
    ]
    for i, c in enumerate(candidates[:6], 1):
        lines.append(f"{i}. ```{c}```")
    return "\n".join(lines)


@register_translator("Hunyuan_MT_Chimera_7B")
class HunyuanMTChimeraTranslator(BaseTranslator):
    """Hunyuan-MT-Chimera-7B: Ensemble refiner. 6 candidates from HY-MT 1.5 (use 1.8B for low VRAM), then Chimera 7B refines. Chimera refiner needs ~7B VRAM."""

    concate_text = False
    cht_require_convert = True
    _load_model_keys = {"model", "tokenizer", "candidate_model", "candidate_tokenizer"}
    params: Dict = {
        "chimera_model": {
            "type": "selector",
            "options": [
                "tencent/Hunyuan-MT-Chimera-7B (BF16, ~14GB VRAM)",
                "tencent/Hunyuan-MT-Chimera-7B-fp8 (FP8, ~7GB VRAM)",
                "tencent/Hunyuan-MT-Chimera-7B (8-bit, ~7GB VRAM)",
                "tencent/Hunyuan-MT-Chimera-7B (4-bit, ~4GB VRAM)",
            ],
            "value": "tencent/Hunyuan-MT-Chimera-7B (4-bit, ~4GB VRAM)",
            "description": "Chimera refiner. 8-bit/4-bit need bitsandbytes. VRAM is approximate.",
        },
        "candidate_model": {
            "type": "selector",
            "options": [
                "tencent/HY-MT1.5-1.8B",
                "tencent/HY-MT1.5-1.8B-FP8",
            ],
            "value": "tencent/HY-MT1.5-1.8B",
            "description": "Model to generate 6 candidates. 1.8B only (≤1.5B). Chimera refiner is 7B.",
        },
        "device": DEVICE_SELECTOR(),
    }

    def _setup_translator(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        device = self.params.get("device", {}).get("value", "cpu")
        if device in ("cuda", "gpu") and torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        self._device = device
        # Candidate generator (HY-MT 1.5 7B)
        cand_name = self.params.get("candidate_model", {}).get("value", "tencent/HY-MT1.5-1.8B")
        self.candidate_tokenizer = AutoTokenizer.from_pretrained(cand_name, trust_remote_code=True)
        self.candidate_model = AutoModelForCausalLM.from_pretrained(
            cand_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        )
        self.candidate_model.to(device)
        self.candidate_model.eval()
        # Chimera refiner
        chimera_option = self.params.get("chimera_model", {}).get("value", "tencent/Hunyuan-MT-Chimera-7B (4-bit, ~4GB VRAM)")
        chimera_name = chimera_option.split(" (")[0].strip()
        use_4bit = "4-bit" in chimera_option and device == "cuda"
        use_8bit = "8-bit" in chimera_option and device == "cuda"
        self.tokenizer = AutoTokenizer.from_pretrained(chimera_name, trust_remote_code=True)
        if use_4bit:
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    chimera_name,
                    trust_remote_code=True,
                    quantization_config=quantization_config,
                    device_map="auto",
                )
                self._device = next(self.model.parameters()).device
            except Exception as e:
                from utils.logger import logger as LOGGER
                LOGGER.warning("Chimera 4-bit load failed (install bitsandbytes?): %s. Falling back to fp16.", e)
                use_4bit = False
        if use_8bit and not use_4bit:
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    chimera_name,
                    trust_remote_code=True,
                    quantization_config=quantization_config,
                    device_map="auto",
                )
                self._device = next(self.model.parameters()).device
            except Exception as e:
                from utils.logger import logger as LOGGER
                LOGGER.warning("Chimera 8-bit load failed (install bitsandbytes?): %s. Falling back to fp16.", e)
                use_8bit = False
        if not use_4bit and not use_8bit:
            self.model = AutoModelForCausalLM.from_pretrained(
                chimera_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            )
            self.model.to(device)
        self.model.eval()
        self.lang_map.clear()
        for k, v in HY_MT_LANG_MAP.items():
            self.lang_map[k] = v

    def _translate(self, src_list: List[str]) -> List[str]:
        if not src_list:
            return []
        if not hasattr(self, "model") or self.model is None or not hasattr(self, "candidate_model") or self.candidate_model is None:
            self.setup_translator()
        src_code, tgt_code = _ensure_lang_codes(self.lang_source, self.lang_target)
        tgt_display = HY_MT_TARGET_DISPLAY.get(tgt_code, tgt_code)
        results = []
        for source_text in src_list:
            # 6 candidates from HY-MT 1.5 (vary seed for diversity)
            candidates = []
            for seed in range(6):
                import torch
                prompt = _hy_mt15_prompt(source_text, src_code, tgt_code, tgt_display)
                g = torch.Generator(device=self._device).manual_seed(seed)
                text = _run_generation_with_seed(
                    self.candidate_model, self.candidate_tokenizer, prompt, self._device, generator=g
                )
                if text:
                    candidates.append(text)
            # Dedupe while preserving order; pad to 6 if needed
            seen = set()
            unique = []
            for c in candidates:
                if c not in seen:
                    seen.add(c)
                    unique.append(c)
            while len(unique) < 6:
                pad = unique[-1] if unique else source_text
                unique.append(pad)
            candidates = unique[:6]
            # Chimera refine
            chimera_prompt = _chimera_prompt(source_text, candidates, src_code, tgt_code)
            refined = _run_generation(self.model, self.tokenizer, chimera_prompt, self._device)
            results.append(refined if refined else (candidates[0] if candidates else source_text))
        return results

    @property
    def supported_src_list(self) -> List[str]:
        return HY_MT_SUPPORTED

    @property
    def supported_tgt_list(self) -> List[str]:
        return HY_MT_SUPPORTED

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        if param_key in ("chimera_model", "candidate_model", "device"):
            for attr in ("model", "tokenizer", "candidate_model", "candidate_tokenizer"):
                if hasattr(self, attr):
                    delattr(self, attr)


def _run_generation_with_seed(model, tokenizer, prompt: str, device: str, generator=None):
    """Like _run_generation but pass generator for reproducible sampling."""
    import torch
    try:
        messages = [{"role": "user", "content": prompt}]
        tokenized = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    except Exception:
        tokenized = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).input_ids
    if isinstance(tokenized, dict):
        inp = {k: v.to(device) for k, v in tokenized.items()}
    else:
        tokenized = tokenized.to(device)
        inp = {"input_ids": tokenized}
    with torch.no_grad():
        out = model.generate(
            **inp,
            max_new_tokens=2048,
            do_sample=True,
            top_k=20,
            top_p=0.6,
            repetition_penalty=1.05,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            generator=generator,
        )
    if isinstance(out, (list, tuple)):
        out = out[0] if out else None
    if out is None:
        return ""
    text = tokenizer.decode(out, skip_special_tokens=True)
    if prompt in text:
        text = text.split(prompt, 1)[-1]
    return text.strip()
