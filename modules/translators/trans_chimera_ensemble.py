"""
Chimera (multi-source): use multiple translators (e.g. Google, DeepL, nllb200, OpenRouter) as candidates,
then refine with local Hunyuan-MT-Chimera-7B. Good for Chinese→English and other pairs.
Select up to 6 candidate translators; Chimera outputs one refined translation per segment.
"""
import copy
import gc
from typing import List, Dict

from .base import BaseTranslator, register_translator, DEVICE_SELECTOR, LANGMAP_GLOBAL
from .exceptions import MissingTranslatorParams
from utils.config import pcfg
from utils.logger import logger as LOGGER
from utils.series_context_store import (
    get_series_context_dir,
    append_page_to_series_context as store_append_page,
    ensure_series_dir,
)

# Reuse Chimera prompt and lang maps from Hunyuan MT
from .trans_hunyuan_mt import (
    _chimera_prompt,
    HY_MT_LANG_MAP,
    HY_MT_TARGET_DISPLAY,
    HY_MT_SUPPORTED,
)

# Translators we cannot use as candidates (recursion / not applicable)
CHIMERA_EXCLUDE = {
    "Chimera (multi-source)",
    "Chain",
    "Ensemble (3+1)",
    "None",
    "Copy Source",
    "Hunyuan_MT_Chimera_7B",
}


def _valid_chimera_candidates():
    from modules import GET_VALID_TRANSLATORS
    base = [t for t in GET_VALID_TRANSLATORS() if t not in CHIMERA_EXCLUDE]
    return ["(skip)"] + base


# Defaults: local / no-API-key translators only (nllb200, opus_mt, t5_mt, HY_MT_1_5_7B).
# For API-based options (Google, DeepL, OpenRouter, Baidu, etc.) select them in the dropdown.
CHIMERA_NO_API_DEFAULT_CANDIDATES = [
    "nllb200",
    "opus_mt",
    "t5_mt",
    "HY_MT_1_5_7B",
    "(skip)",
    "(skip)",
]


def _run_chimera_generation(model, tokenizer, prompt: str, device: str) -> str:
    """Tokenize prompt, generate, decode and strip prompt to get Chimera output."""
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


@register_translator("Chimera (multi-source)")
class ChimeraEnsembleTranslator(BaseTranslator):
    """Multiple translators (Google, DeepL, OpenRouter, etc.) → Hunyuan-MT-Chimera-7B refiner. Good for Zh→En."""

    concate_text = False
    cht_require_convert = True
    _load_model_keys = {"model", "tokenizer"}
    params: Dict = {
        "candidate_1": {
            "type": "selector",
            "options": [],
            "value": "nllb200",
            "description": "Candidate 1. NLLB-200 600M (≤1.5B). Set to 1.3B in nllb200 params for quality.",
        },
        "candidate_2": {
            "type": "selector",
            "options": [],
            "value": "opus_mt",
            "description": "Candidate 2. OPUS-MT zh-en / MarianMT (Helsinki-NLP).",
        },
        "candidate_3": {
            "type": "selector",
            "options": [],
            "value": "t5_mt",
            "description": "Candidate 3. T5-small/base (Google).",
        },
        "candidate_4": {
            "type": "selector",
            "options": [],
            "value": "mBART50",
            "description": "Candidate 4. mBART50 many-to-many (Meta, ~600M).",
        },
        "candidate_5": {
            "type": "selector",
            "options": [],
            "value": "M2M100_HF",
            "description": "Candidate 5. M2M100 418M or 1.2B (Meta). Default 418M in params.",
        },
        "candidate_6": {
            "type": "selector",
            "options": [],
            "value": "(skip)",
            "description": "Candidate 6. Or add HY_MT_1_5_7B (1.8B), google, DeepL, etc.",
        },
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
        "device": DEVICE_SELECTOR(),
    }

    def _setup_translator(self):
        self.lang_map.clear()
        for k, v in HY_MT_LANG_MAP.items():
            self.lang_map[k] = v
        valid = _valid_chimera_candidates()
        for key in ("candidate_1", "candidate_2", "candidate_3", "candidate_4", "candidate_5", "candidate_6"):
            if key in self.params and isinstance(self.params[key], dict):
                opts = self.params[key].get("options")
                if not opts or len(opts) != len(valid):
                    self.params[key]["options"] = valid
        self._candidates = []

    def _get_merged_params(self) -> Dict:
        from modules import GET_VALID_TRANSLATORS, TRANSLATORS
        from modules.base import merge_config_module_params
        cfg = pcfg.module
        raw = getattr(cfg, "translator_params", None) or {}
        return merge_config_module_params(
            copy.deepcopy(raw),
            GET_VALID_TRANSLATORS(),
            TRANSLATORS.get,
        )

    def _build_candidates(self):
        """Fill candidate list for backward compat; prefer _create_one_candidate + one-at-a-time in _translate."""
        if self._candidates:
            return
        from modules import GET_VALID_TRANSLATORS, TRANSLATORS
        merged = self._get_merged_params()
        valid = GET_VALID_TRANSLATORS()
        self._candidates = []
        for i in range(6):
            key = f"candidate_{i + 1}"
            name = (self.get_param_value(key) or "").strip()
            if not name or name == "(skip)" or name in CHIMERA_EXCLUDE or name not in valid:
                self._candidates.append(None)
                continue
            try:
                Klass = TRANSLATORS.get(name)
                params = merged.get(name, {})
                if isinstance(params, dict):
                    params = {k: v for k, v in params.items() if not (isinstance(k, str) and k.startswith("__"))}
                params = self._flatten_params_for_candidate(params)
                inst = Klass(
                    lang_source=self.lang_source,
                    lang_target=self.lang_target,
                    raise_unsupported_lang=False,
                    **params,
                )
                self._candidates.append(inst)
            except Exception as e:
                LOGGER.warning("Chimera (multi-source): could not create %s: %s", name, e)
                self._candidates.append(None)

    @staticmethod
    def _flatten_params_for_candidate(params: Dict) -> Dict:
        """Resolve dict params to their 'value' so candidates that use self.params[key].strip() etc. get strings."""
        if not isinstance(params, dict):
            return params
        out = {}
        for k, v in params.items():
            if isinstance(k, str) and k.startswith("__"):
                continue
            if isinstance(v, dict) and "value" in v:
                out[k] = v["value"]
            else:
                out[k] = v
        return out

    def _create_one_candidate(self, name: str):
        """Create a single candidate translator (caller must delete it after use to free VRAM)."""
        from modules import GET_VALID_TRANSLATORS, TRANSLATORS
        if not name or name == "(skip)" or name in CHIMERA_EXCLUDE:
            return None
        valid = GET_VALID_TRANSLATORS()
        if name not in valid:
            return None
        merged = self._get_merged_params()
        try:
            Klass = TRANSLATORS.get(name)
            params = merged.get(name, {})
            if isinstance(params, dict):
                params = {k: v for k, v in params.items() if not (isinstance(k, str) and k.startswith("__"))}
            params = self._flatten_params_for_candidate(params)
            return Klass(
                lang_source=self.lang_source,
                lang_target=self.lang_target,
                raise_unsupported_lang=False,
                **params,
            )
        except Exception as e:
            LOGGER.warning("Chimera (multi-source): could not create %s: %s", name, e)
            return None

    def _forward_translation_context(self, candidate, candidate_name: str = "") -> None:
        """Forward all context/settings the pipeline gives the main translator so sub-translators (e.g. LLM API) get the same."""
        ctx = getattr(self, "_cache_translation_context", None)
        if ctx and hasattr(candidate, "set_translation_context"):
            try:
                candidate.set_translation_context(
                    previous_pages=ctx.get("previous_pages") or [],
                    project_glossary=ctx.get("project_glossary") or [],
                    series_context_path=ctx.get("series_context_path"),
                    next_page=ctx.get("next_page"),
                )
            except Exception:
                pass
        if ctx and candidate_name:
            prev = ctx.get("previous_pages") or []
            glossary = ctx.get("project_glossary") or []
            series_path = ctx.get("series_context_path") or ""
            LOGGER.info(
                "Chimera: forwarding translation context to candidate %s (series_path=%s, %d previous pages, %d project glossary entries)",
                candidate_name, series_path or "(none)", len(prev), len(glossary),
            )
        if ctx is not None:
            try:
                setattr(candidate, "_cache_translation_context", ctx)
            except Exception:
                pass
        for attr in ("_current_page_key", "_current_page_image"):
            if hasattr(self, attr):
                try:
                    setattr(candidate, attr, getattr(self, attr))
                except Exception:
                    pass

    def append_page_to_series_context(self, series_context_path: str, sources: List[str], translations: List[str]) -> None:
        """Append refined page to series context store for cross-chapter consistency."""
        path = get_series_context_dir((series_context_path or "").strip())
        if not path or not sources:
            return
        ensure_series_dir(path)
        store_append_page(path, sources, translations, max_stored_pages=15)

    def _translate_one_candidate(self, candidate, src_list: List[str], candidate_name: str = "") -> List[str]:
        if candidate is None or not callable(getattr(candidate, "_translate", None)):
            return []
        self._forward_translation_context(candidate, candidate_name)
        try:
            out = candidate._translate(src_list)
            if not isinstance(out, list):
                out = [out] if out is not None else []
            # Normalize length so Chimera gets one result per segment (avoid base.translate assert).
            n = len(src_list)
            if len(out) > n:
                out = out[:n]
            elif len(out) < n:
                pad = out[-1] if out else ""
                out = out + [pad] * (n - len(out))
            return out
        except Exception as e:
            LOGGER.warning("Chimera candidate translate failed: %s", e)
            return []

    @staticmethod
    def _release_candidate_gpu(inst):
        """Aggressively free GPU memory used by a candidate translator (model/tokenizer)."""
        for attr in ("model", "tokenizer", "pipeline", "generator"):
            if hasattr(inst, attr):
                try:
                    setattr(inst, attr, None)
                except Exception:
                    pass
        try:
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _safe_param(self, key: str, default: str) -> str:
        """Get param value whether it's stored as dict with 'value' or as plain string (avoids 'dict'/'str' attribute errors)."""
        if not getattr(self, "params", None) or key not in self.params:
            return default
        p = self.params[key]
        if isinstance(p, dict) and "value" in p:
            v = p["value"]
        else:
            v = p
        return str(v).strip() if v not in (None, "") else default

    def _ensure_chimera_loaded(self):
        if getattr(self, "_chimera_load_failed", False):
            return
        if hasattr(self, "model") and self.model is not None and hasattr(self, "tokenizer") and self.tokenizer is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        device = self._safe_param("device", "cpu")
        if device in ("cuda", "gpu") and torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        self._device = device
        chimera_default = "tencent/Hunyuan-MT-Chimera-7B (4-bit, ~4GB VRAM)"
        chimera_option = self._safe_param("chimera_model", chimera_default) or chimera_default
        chimera_name = chimera_option.split(" (")[0].strip() if isinstance(chimera_option, str) else chimera_default.split(" (")[0].strip()
        # On CUDA always try 4-bit first so the model loads in 4-bit when possible (saves VRAM).
        use_4bit = device == "cuda"
        use_8bit = "8-bit" in chimera_option and device == "cuda"
        self.tokenizer = AutoTokenizer.from_pretrained(chimera_name, trust_remote_code=True)
        def _is_oom(err):
            s = str(err).lower()
            return "out of memory" in s or "cuda" in s and "memory" in s

        if use_4bit:
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                # Cap GPU memory so 4-bit load doesn't OOM on 11GB after candidates; leave ~1.5GB headroom
                max_memory = None
                if torch.cuda.is_available():
                    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                    alloc_gb = max(4.0, min(total_gb - 1.5, 9.0))
                    max_memory = {0: f"{alloc_gb:.1f}GiB", "cpu": "24GiB"}
                self.model = AutoModelForCausalLM.from_pretrained(
                    chimera_name,
                    trust_remote_code=True,
                    quantization_config=quantization_config,
                    device_map="auto",
                    max_memory=max_memory,
                    low_cpu_mem_usage=True,
                )
                self._device = next(self.model.parameters()).device
            except Exception as e:
                if _is_oom(e):
                    raise RuntimeError(
                        "Chimera 4-bit load ran out of GPU memory. Free VRAM by using fewer/smaller candidate translators, "
                        "or close other GPU apps. On 11GB GPUs use only lightweight candidates (e.g. nllb200, opus_mt, t5_mt)."
                    ) from e
                LOGGER.warning("Chimera 4-bit load failed (install bitsandbytes?): %s. Falling back to fp16.", e)
                use_4bit = False
        if use_8bit and not use_4bit:
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                max_memory = None
                if torch.cuda.is_available():
                    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                    alloc_gb = max(6.0, min(total_gb - 1.5, 10.0))
                    max_memory = {0: f"{alloc_gb:.1f}GiB", "cpu": "24GiB"}
                self.model = AutoModelForCausalLM.from_pretrained(
                    chimera_name,
                    trust_remote_code=True,
                    quantization_config=quantization_config,
                    device_map="auto",
                    max_memory=max_memory,
                    low_cpu_mem_usage=True,
                )
                self._device = next(self.model.parameters()).device
            except Exception as e:
                if _is_oom(e):
                    raise RuntimeError(
                        "Chimera 8-bit load ran out of GPU memory. Free VRAM or use fewer/smaller candidates."
                    ) from e
                LOGGER.warning("Chimera 8-bit load failed (install bitsandbytes?): %s. Falling back to fp16.", e)
                use_8bit = False
        if not use_4bit and not use_8bit:
            if device == "cuda":
                try:
                    total = torch.cuda.get_device_properties(0).total_memory
                    if total < 14 * (1024 ** 3):
                        raise RuntimeError(
                            "Chimera fp16/bf16 needs ~14GB VRAM. This GPU has %.1f GB. "
                            "Use Chimera model option '4-bit' or 'FP8' in translator settings."
                            % (total / (1024 ** 3),)
                        ) from None
                except RuntimeError:
                    raise
                except Exception:
                    pass
            self.model = AutoModelForCausalLM.from_pretrained(
                chimera_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
            )
            self.model.to(device)
        self.model.eval()

    def _translate(self, src_list: List[str]) -> List[str]:
        # Load Chimera refiner before doing anything else (clean GPU for 4-bit; avoid OOM from fragmentation after candidates).
        try:
            self._ensure_chimera_loaded()
        except RuntimeError as e:
            err = str(e).lower()
            if "memory" in err or "vram" in err or "out of memory" in err:
                LOGGER.warning(
                    "Chimera refiner could not load (OOM). Using candidate translations only."
                )
                self._chimera_load_failed = True
                if hasattr(self, "model"):
                    self.model = None
                if hasattr(self, "tokenizer"):
                    self.tokenizer = None
            else:
                raise
        if not src_list:
            return []
        src_code = HY_MT_LANG_MAP.get(self.lang_source) or self.lang_map.get(self.lang_source)
        tgt_code = HY_MT_LANG_MAP.get(self.lang_target) or self.lang_map.get(self.lang_target)
        if not src_code or not tgt_code:
            raise MissingTranslatorParams(
                f"Chimera (multi-source): unsupported language pair {self.lang_source} -> {self.lang_target}. "
                f"Supported: {', '.join(HY_MT_SUPPORTED[:15])}..."
            )
        # Run one candidate at a time; refiner (if loaded) stays in VRAM, so use lightweight candidates on 11GB.
        all_candidates = []
        for i in range(6):
            key = f"candidate_{i + 1}"
            name = (self.get_param_value(key) or "").strip()
            inst = self._create_one_candidate(name)
            if inst is None:
                continue
            try:
                t = self._translate_one_candidate(inst, src_list, candidate_name=name)
                if t:
                    all_candidates.append(t)
            finally:
                self._release_candidate_gpu(inst)
                del inst
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
        if not all_candidates:
            LOGGER.warning("Chimera (multi-source): no candidates produced translations; returning source.")
            return list(src_list)
        chimera_available = getattr(self, "model", None) is not None and getattr(self, "tokenizer", None) is not None
        if not chimera_available:
            # OOM or refiner unavailable: return first available candidate translation per segment (no refinement).
            results = []
            for i in range(len(src_list)):
                fallback = src_list[i]
                for row in all_candidates:
                    if i < len(row) and row[i] and str(row[i]).strip() and row[i] != "[candidate failed]":
                        fallback = row[i].strip()
                        break
                results.append(fallback)
            return results
        # Per-segment: build list of up to 6 candidate strings, pad to 6, run Chimera
        results = []
        for i, source_text in enumerate(src_list):
            candidates = []
            for row in all_candidates:
                if i < len(row) and row[i] and row[i].strip() and row[i] != "[candidate failed]":
                    candidates.append(row[i].strip())
            # Dedupe order-preserving, then pad to 6
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
            prompt = _chimera_prompt(source_text, candidates, src_code, tgt_code)
            refined = _run_chimera_generation(self.model, self.tokenizer, prompt, self._device)
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
        if param_key.startswith("candidate_") or param_key in ("chimera_model", "device"):
            self._candidates = []
            if param_key in ("chimera_model", "device"):
                if getattr(self, "_chimera_load_failed", False):
                    self._chimera_load_failed = False
                for attr in ("model", "tokenizer", "_device"):
                    if hasattr(self, attr):
                        delattr(self, attr)


def _refresh_chimera_ensemble_options():
    """Refresh Chimera (multi-source) candidate dropdown options after all translators are loaded."""
    try:
        from modules.translators import TRANSLATORS
        chimera = TRANSLATORS.module_dict.get("Chimera (multi-source)")
        if chimera is not None and hasattr(chimera, "params"):
            valid = _valid_chimera_candidates()
            for key in ("candidate_1", "candidate_2", "candidate_3", "candidate_4", "candidate_5", "candidate_6"):
                if key in chimera.params and isinstance(chimera.params[key], dict):
                    chimera.params[key]["options"] = valid
    except Exception as e:
        LOGGER.warning("Could not refresh Chimera (multi-source) candidate options: %s", e)


# Fill candidate options at load time; init_module_registries('translator') calls _refresh_chimera_ensemble_options() for full list
try:
    _opts = _valid_chimera_candidates()
    for _k in ("candidate_1", "candidate_2", "candidate_3", "candidate_4", "candidate_5", "candidate_6"):
        if _k in ChimeraEnsembleTranslator.params and isinstance(ChimeraEnsembleTranslator.params[_k], dict):
            ChimeraEnsembleTranslator.params[_k]["options"] = _opts
except Exception:
    pass
