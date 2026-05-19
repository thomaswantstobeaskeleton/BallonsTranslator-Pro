from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple
from urllib.parse import urljoin
import json
import urllib.request

from utils.llm_provider import default_endpoint_for_provider


@dataclass
class ProviderCheckResult:
    ok: bool
    provider: str
    endpoint: str
    status_code: int
    detail: str = ""


def provider_endpoint_preset(provider: str) -> str:
    return (default_endpoint_for_provider(provider) or "").strip()


def _default_fetch(url: str, headers: Optional[Dict[str, str]] = None, timeout: float = 8.0) -> Tuple[int, str]:
    req = urllib.request.Request(url, headers=headers or {}, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return int(getattr(resp, "status", 200) or 200), resp.read().decode("utf-8", errors="replace")


def check_provider_connection(
    provider: str,
    endpoint: str,
    api_key: str = "",
    timeout_sec: float = 8.0,
    fetcher: Optional[Callable[[str, Optional[Dict[str, str]], float], Tuple[int, str]]] = None,
) -> ProviderCheckResult:
    provider = (provider or "").strip()
    endpoint = (endpoint or provider_endpoint_preset(provider) or "").strip().rstrip("/")
    if not endpoint:
        return ProviderCheckResult(False, provider, endpoint, 0, "missing endpoint")
    fetch = fetcher or _default_fetch
    headers: Dict[str, str] = {"Accept": "application/json"}
    if api_key.strip() and provider not in {"Ollama", "LLM Studio"}:
        headers["Authorization"] = f"Bearer {api_key.strip()}"

    # Lightweight provider-specific probes.
    path = "/v1/models"
    if provider == "Ollama":
        path = "/api/tags"
    elif provider == "LLM Studio":
        path = "/v1/models"

    url = urljoin(endpoint + "/", path.lstrip("/"))
    try:
        status, body = fetch(url, headers, float(timeout_sec))
        if status < 200 or status >= 300:
            return ProviderCheckResult(False, provider, endpoint, status, f"HTTP {status}")
        # best-effort JSON parse for human detail
        detail = "ok"
        try:
            payload = json.loads(body or "{}")
            if isinstance(payload, dict) and isinstance(payload.get("data"), list):
                detail = f"ok ({len(payload['data'])} model(s))"
            elif isinstance(payload, dict) and isinstance(payload.get("models"), list):
                detail = f"ok ({len(payload['models'])} model(s))"
        except Exception:
            pass
        return ProviderCheckResult(True, provider, endpoint, status, detail)
    except Exception as e:
        return ProviderCheckResult(False, provider, endpoint, 0, str(e))
