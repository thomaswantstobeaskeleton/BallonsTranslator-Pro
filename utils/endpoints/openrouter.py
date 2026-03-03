"""
Section 18: OpenRouter helpers — reasoning-model detection via live /models metadata.
"""
from typing import Optional

# Cache: result of GET /models. None = not fetched yet.
_OPENROUTER_MODELS_CACHE: Optional[dict] = None


def openrouter_fetch_models(api_key: str, proxy: Optional[str] = None) -> dict:
    """GET OpenRouter /models, return response JSON. Caches in memory."""
    global _OPENROUTER_MODELS_CACHE
    if _OPENROUTER_MODELS_CACHE is not None:
        return _OPENROUTER_MODELS_CACHE
    try:
        import httpx
        from utils.proxy_utils import create_httpx_client
        url = "https://openrouter.ai/api/v1/models"
        headers = {"Authorization": f"Bearer {api_key}"}
        timeout = httpx.Timeout(30.0)
        client = create_httpx_client(proxy, timeout=timeout) if proxy else httpx.Client(timeout=timeout)
        try:
            r = client.get(url, headers=headers)
            r.raise_for_status()
            data = r.json()
            _OPENROUTER_MODELS_CACHE = data
            return data
        finally:
            client.close()
    except Exception:
        _OPENROUTER_MODELS_CACHE = {}
        return {}


def openrouter_is_reasoning_model(api_key: str, model_id: str, proxy: Optional[str] = None) -> bool:
    """
    Return True if the OpenRouter model has include_reasoning (reasoning-capable).
    Uses cached /models response.
    """
    if not api_key or not model_id:
        return False
    model_id = (model_id or "").strip()
    if ": " in model_id:
        model_id = model_id.split(": ", 1)[1].strip()
    data = openrouter_fetch_models(api_key, proxy)
    models = data.get("data") or []
    for m in models:
        mid = (m.get("id") or "").strip()
        if not mid:
            continue
        if mid == model_id or model_id.startswith(mid + "/") or mid.startswith(model_id + ":"):
            arch = m.get("architecture") or {}
            return bool(arch.get("include_reasoning") or m.get("include_reasoning"))
    return False


def openrouter_clear_models_cache() -> None:
    """Clear the /models cache (e.g. after API key change)."""
    global _OPENROUTER_MODELS_CACHE
    _OPENROUTER_MODELS_CACHE = None
