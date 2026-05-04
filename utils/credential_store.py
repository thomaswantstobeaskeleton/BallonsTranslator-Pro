"""Credential storage wrapper with optional OS keyring backend."""
from __future__ import annotations

from typing import Optional

SERVICE = "ballonstranslator-pro"


def _keyring_module():
    try:
        import keyring  # type: ignore
        return keyring
    except Exception:
        return None


def has_keyring() -> bool:
    return _keyring_module() is not None


def set_secret(key: str, value: str) -> bool:
    kr = _keyring_module()
    if kr is None:
        return False
    try:
        kr.set_password(SERVICE, key, value or "")
        return True
    except Exception:
        return False


def get_secret(key: str) -> Optional[str]:
    kr = _keyring_module()
    if kr is None:
        return None
    try:
        return kr.get_password(SERVICE, key)
    except Exception:
        return None
