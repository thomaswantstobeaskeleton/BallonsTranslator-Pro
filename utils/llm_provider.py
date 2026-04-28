from typing import Optional, Tuple

MODEL_PREFIX_TO_PROVIDER = {
    "OAI": "OpenAI",
    "GGL": "Google",
    "XAI": "Grok",
    "OPENROUTER": "OpenRouter",
    "OLL": "Ollama",
    "LLMS": "LLM Studio",
}

_PROVIDER_NORMALIZED = {v.upper(): v for v in MODEL_PREFIX_TO_PROVIDER.values()}
PROVIDER_DEFAULT_ENDPOINTS = {
    "OpenAI": "https://api.openai.com/v1",
    "Google": "https://generativelanguage.googleapis.com/v1beta/openai",
    "OpenRouter": "https://openrouter.ai/api/v1",
    "Grok": "https://api.x.ai/v1",
    "LLM Studio": "http://localhost:1234/v1",
    "Ollama": "http://localhost:11434/v1",
}
LOCAL_PROVIDERS = {"LLM Studio", "Ollama"}


def is_local_provider(provider: Optional[str]) -> bool:
    return (provider or "") in LOCAL_PROVIDERS


def default_endpoint_for_provider(provider: Optional[str]) -> Optional[str]:
    return PROVIDER_DEFAULT_ENDPOINTS.get(provider or "")


def resolve_endpoint(provider: Optional[str], endpoint: Optional[str]) -> Optional[str]:
    normalized = (endpoint or "").strip()
    if normalized:
        return normalized
    return default_endpoint_for_provider(provider)


def parse_provider_prefixed_model(model_id: Optional[str]) -> Tuple[Optional[str], str]:
    """Return (provider_from_prefix, bare_model_id).

    Accepts forms like "OAI: gpt-4o" and "OpenRouter: model/id".
    Raises ValueError for malformed/unknown prefixed identifiers.
    """
    model_text = (model_id or "").strip()
    if not model_text:
        return None, ""

    if ":" not in model_text:
        return None, model_text

    prefix, bare_model = model_text.split(":", 1)
    prefix = prefix.strip()
    bare_model = bare_model.strip()
    if not prefix or not bare_model:
        raise ValueError(f"Invalid provider/model id combination: '{model_text}'")

    mapped_provider = MODEL_PREFIX_TO_PROVIDER.get(prefix.upper())
    if mapped_provider:
        return mapped_provider, bare_model

    normalized_provider = _PROVIDER_NORMALIZED.get(prefix.upper())
    if normalized_provider:
        return normalized_provider, bare_model

    raise ValueError(f"Unknown provider prefix '{prefix}' in '{model_text}'")


def effective_provider(
    selected_provider: Optional[str],
    model_id: Optional[str],
    override_model: Optional[str] = None,
) -> str:
    if (override_model or "").strip():
        return selected_provider or ""
    parsed_provider, _ = parse_provider_prefixed_model(model_id)
    return parsed_provider or (selected_provider or "")


def normalize_model_name(model_id: Optional[str], override_model: Optional[str] = None) -> str:
    if (override_model or "").strip():
        return override_model.strip()
    _provider, bare_model = parse_provider_prefixed_model(model_id)
    return bare_model
