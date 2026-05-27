"""
Local LLM translator via llama.cpp (llama-cpp-python).

Supports GGUF models from Hugging Face:
- Sakura-7B / Sakura-13B / Sakura-14B (SakuraLLM family)
- Sugoi-14B / Sugoi-v4
- vntl-llama3-8b
- Qwen2.5-7B-Instruct / Qwen3.5-4b

Auto-downloads models via hf_hub_download on first use.
Jinja prompt templates live in data/prompts/.
"""

import os
import re
from typing import List, Optional

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None

from .base import BaseTranslator, register_translator


def _ensure_model_path(model_id: str, filename: str) -> str:
    """Download GGUF from HF if not present locally, return local path."""
    if hf_hub_download is None:
        raise RuntimeError("huggingface-hub is required. Install: pip install huggingface-hub")

    # First check if user already has it in data/models
    local_path = os.path.join("data", "models", filename)
    if os.path.isfile(local_path):
        return os.path.abspath(local_path)

    # Otherwise download via HF hub cache
    path = hf_hub_download(repo_id=model_id, filename=filename)
    return path


@register_translator("llama_cpp")
class LLamaCppTranslator(BaseTranslator):
    """
    Local LLM translator using llama.cpp.
    Recommended for privacy-conscious users or offline workflows.
    """

    params = {
        "model_id": {
            "type": "selector",
            "options": [
                "SakuraLLM/Sakura-13B-LNovel-v0.9-GGUF",
                "SakuraLLM/Sakura-7B-LNovel-v0.9-GGUF",
                "SakuraLLM/Sakura-14B-LNovel-v0.9-GGUF",
                "gnklaxx/sugoi-v4-13b-GGUF",
                "vnteam/vntl-llama3-8b-GGUF",
                "Qwen/Qwen2.5-7B-Instruct-GGUF",
            ],
            "value": "SakuraLLM/Sakura-13B-LNovel-v0.9-GGUF",
            "description": "Hugging Face repo ID for the GGUF model.",
        },
        "gguf_filename": {
            "type": "line_editor",
            "value": "sakura-13b-lnovel-v0_9-Q4_K_M.gguf",
            "description": "Specific GGUF filename inside the repo (e.g. Q4_K_M, Q5_K_M).",
        },
        "n_ctx": {
            "type": "selector",
            "options": [2048, 4096, 8192, 16384, 32768],
            "value": 4096,
            "description": "Context window size.",
        },
        "n_gpu_layers": {
            "type": "selector",
            "options": [0, 10, 20, 30, 40, 50, 60, 70, 80],
            "value": 0,
            "description": "Number of layers to offload to GPU (0 = CPU only).",
        },
        "temperature": {
            "type": "selector",
            "options": [0.0, 0.1, 0.3, 0.5, 0.7, 1.0],
            "value": 0.3,
            "description": "Sampling temperature (0 = deterministic).",
        },
        "chat_template": {
            "type": "selector",
            "options": ["auto", "sakura", "alpaca", "chatml", "gemma", "qwen"],
            "value": "auto",
            "description": "Prompt template. 'auto' detects from model_id.",
        },
        "use_page_context": {
            "type": "checkbox",
            "value": True,
            "description": "Feed entire page text as context (vs. block-by-block).",
        },
        "description": "Local LLM via llama.cpp (GGUF). Auto-downloads from Hugging Face.",
    }

    concate_text = False
    translate_by_textblock = False

    def __init__(self, lang_source: str, lang_target: str, raise_unsupported_lang: bool = True, **params) -> None:
        self._llm: Optional["Llama"] = None
        self._model_path: Optional[str] = None
        super().__init__(lang_source, lang_target, raise_unsupported_lang, **params)

    def _setup_translator(self):
        if Llama is None:
            raise RuntimeError(
                "llama-cpp-python is required for the local LLM translator. "
                "Install: pip install llama-cpp-python"
            )

    def _load_model(self):
        model_id = self.get_param_value("model_id")
        filename = self.get_param_value("gguf_filename")
        n_ctx = self.get_param_value("n_ctx")
        n_gpu_layers = self.get_param_value("n_gpu_layers")

        self._model_path = _ensure_model_path(model_id, filename)

        self._llm = Llama(
            model_path=self._model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )

    def _detect_template(self) -> str:
        """Auto-detect chat template from model_id if set to 'auto'."""
        template = self.get_param_value("chat_template")
        if template != "auto":
            return template

        model_id = self.get_param_value("model_id").lower()
        if "sakura" in model_id:
            return "sakura"
        if "qwen" in model_id:
            return "qwen"
        if "gemma" in model_id:
            return "gemma"
        if "alpaca" in model_id or "sugoi" in model_id:
            return "alpaca"
        return "chatml"

    def _format_prompt(self, src_list: List[str]) -> str:
        """Format source text list into a prompt using the selected template."""
        template = self._detect_template()
        source_lang = self.lang_source
        target_lang = self.lang_target

        # Build block list
        block_texts = []
        for idx, text in enumerate(src_list):
            block_texts.append(f"{idx + 1}. {text}")
        blocks_str = "\n".join(block_texts)

        # Template-specific formatting
        if template == "sakura":
            # SakuraLLM expects a simple instruction with Japanese source
            prompt = (
                f"以下の{source_lang}の漫画テキストを{target_lang}に翻訳してください。\n\n"
                f"{blocks_str}\n\n"
                f"翻訳結果:"
            )
        elif template == "qwen":
            # Qwen chat format
            prompt = (
                f"<|im_start|>system\n"
                f"You are a professional manga translator from {source_lang} to {target_lang}.\n"
                f"Preserve character voice, honorifics, and emotional tone.\n"
                f"Return exactly {len(src_list)} translations, one per block.\n"
                f"<|im_end|>\n"
                f"<|im_start|>user\n"
                f"Translate the following manga text blocks:\n\n{blocks_str}\n\n"
                f"<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
        elif template == "gemma":
            prompt = (
                f"<start_of_turn>user\n"
                f"Translate these {source_lang} manga text blocks to {target_lang}:\n\n{blocks_str}\n"
                f"<end_of_turn>\n"
                f"<start_of_turn>model\n"
            )
        elif template == "alpaca":
            prompt = (
                f"### Instruction:\n"
                f"Translate the following {source_lang} manga text blocks to {target_lang}.\n"
                f"Preserve character voice and emotional tone.\n\n"
                f"### Input:\n{blocks_str}\n\n"
                f"### Response:\n"
            )
        else:  # chatml / default
            prompt = (
                f"<|system|>\n"
                f"You are a professional manga translator from {source_lang} to {target_lang}.\n"
                f"Preserve character voice, honorifics, and emotional tone.\n"
                f"Return exactly {len(src_list)} translations, one per block.\n"
                f"<|user|>\n"
                f"Translate the following manga text blocks:\n\n{blocks_str}\n"
                f"<|assistant|>\n"
            )

        return prompt

    def _parse_response(self, response: str, expected_count: int) -> List[str]:
        """Parse LLM output into a list of translations."""
        text = response.strip()

        # Try JSON extraction first
        import json
        try:
            # Find JSON object in response
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                data = json.loads(match.group())
                if isinstance(data, dict) and "translations" in data:
                    return data["translations"]
        except Exception:
            pass

        # Try numbered list extraction: "1. text\n2. text"
        numbered = re.findall(r'^\s*\d+\.\s*(.+)$', text, re.MULTILINE)
        if len(numbered) == expected_count:
            return numbered

        # Try line-by-line split
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if len(lines) == expected_count:
            return lines

        # Fallback: split by common delimiters
        for delimiter in ["\n###\n", "\n---\n", "\n##\n"]:
            if delimiter in text:
                parts = [p.strip() for p in text.split(delimiter) if p.strip()]
                if len(parts) == expected_count:
                    return parts

        # Last resort: return raw lines, pad with empty strings if needed
        if len(lines) >= expected_count:
            return lines[:expected_count]
        return lines + [""] * (expected_count - len(lines))

    def _translate(self, src_list: List[str]) -> List[str]:
        if not src_list:
            return []

        if self._llm is None:
            self._load_model()

        prompt = self._format_prompt(src_list)
        temperature = float(self.get_param_value("temperature"))

        output = self._llm(
            prompt,
            max_tokens=-1,  # use context window
            temperature=temperature,
            stop=["<|im_end|>", "<end_of_turn>", "### Instruction:", "<|user|>"],
        )

        response = output["choices"][0]["text"]
        return self._parse_response(response, len(src_list))

    def is_deterministic(self) -> bool:
        """LLM is deterministic only when temperature is 0."""
        try:
            return float(self.get_param_value("temperature")) == 0.0
        except Exception:
            return False
