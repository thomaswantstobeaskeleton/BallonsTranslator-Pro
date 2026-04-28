import pytest

from utils.llm_provider import (
    effective_provider,
    is_local_provider,
    normalize_model_name,
    resolve_endpoint,
)


def test_openai_style_model_id_keeps_provider_and_normalizes_model_name():
    provider = effective_provider("OpenAI", "OAI: gpt-4o")
    model_name = normalize_model_name("OAI: gpt-4o")
    assert provider == "OpenAI"
    assert model_name == "gpt-4o"


def test_oll_prefix_routes_to_ollama_and_bypasses_api_key():
    provider = effective_provider("OpenAI", "OLL: qwen2.5vl:latest")
    assert provider == "Ollama"
    assert is_local_provider(provider)


def test_empty_endpoint_uses_provider_default():
    endpoint = resolve_endpoint("Ollama", "")
    assert endpoint == "http://localhost:11434/v1"


def test_invalid_provider_model_combinations_raise():
    with pytest.raises(ValueError):
        normalize_model_name("OAI:")

    with pytest.raises(ValueError):
        normalize_model_name("BADPREFIX: gpt-4o")
