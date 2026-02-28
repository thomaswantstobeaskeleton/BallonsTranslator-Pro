"""
Normalize proxy URL for use with httpx/requests (e.g. add http:// if missing).
Used by LLM API translator and other modules that accept a proxy string.
"""
from typing import Optional, Dict, Union

import httpx


def normalize_proxy_url(proxy_url: Optional[str]) -> Optional[str]:
    """
    Ensure proxy URL has a scheme prefix so httpx/requests accept it.

    Args:
        proxy_url: Proxy URL (e.g. "127.0.0.1:7897" or "http://proxy:port").

    Returns:
        Normalized URL with scheme, or None if input is empty.
    """
    if not proxy_url or not isinstance(proxy_url, str):
        return proxy_url
    s = proxy_url.strip()
    if not s:
        return None
    if "://" not in s:
        return "http://" + s
    return s


def create_httpx_transport(
    proxy: Optional[Union[str, Dict[str, str]]],
) -> Optional[Dict[str, httpx.HTTPTransport]]:
    """
    Create httpx transport mounts for the given proxy.

    Args:
        proxy: Proxy URL string or dict mapping scheme to URL.

    Returns:
        Mounts dict for httpx client, or None if no proxy.
    """
    if not proxy:
        return None
    mounts = {}
    if isinstance(proxy, str):
        url = normalize_proxy_url(proxy)
        if url:
            mounts["http://"] = httpx.HTTPTransport(proxy=url)
            mounts["https://"] = httpx.HTTPTransport(proxy=url)
    elif isinstance(proxy, dict):
        for scheme in ("http://", "https://"):
            if scheme in proxy:
                url = normalize_proxy_url(proxy[scheme])
                if url:
                    mounts[scheme] = httpx.HTTPTransport(proxy=url)
    return mounts if mounts else None


def create_httpx_client(
    proxy: Optional[Union[str, Dict[str, str]]] = None,
    **kwargs,
) -> httpx.Client:
    """
    Create an httpx client with optional proxy. Pass timeout, etc. via kwargs.

    Args:
        proxy: Proxy URL or dict of scheme -> URL.
        **kwargs: Passed to httpx.Client (e.g. timeout=httpx.Timeout(...)).

    Returns:
        Configured httpx.Client.
    """
    mounts = create_httpx_transport(proxy)
    if mounts:
        kwargs = dict(kwargs)
        kwargs["mounts"] = mounts
    return httpx.Client(**kwargs)
