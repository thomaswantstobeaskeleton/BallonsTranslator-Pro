from __future__ import annotations

from typing import Any, List, Tuple

SECRET_FIELD_SPECS: List[Tuple[str | None, str, str]] = [
    ("module", "layout_review_api_key", "layout_review_api_key"),
    ("module", "video_translator_flow_fixer_openrouter_apikey", "video_flow_fixer_openrouter_apikey"),
    ("module", "video_translator_flow_fixer_openai_apikey", "video_flow_fixer_openai_apikey"),
    (None, "automation_api_key", "automation_api_key"),
]


def migrate_secrets_to_keyring_if_possible(pcfg: Any, *, has_keyring, set_secret) -> bool:
    if not has_keyring() or bool(getattr(pcfg, "credential_migration_done", False)):
        return False
    allow_plain = bool(getattr(pcfg, "credential_use_plaintext_fallback", False))
    migrated = False
    for owner, field_name, secret_key in SECRET_FIELD_SPECS:
        target = getattr(pcfg, owner) if owner else pcfg
        value = str(getattr(target, field_name, "") or "").strip()
        if not value:
            continue
        if set_secret(secret_key, value):
            migrated = True
            if not allow_plain:
                setattr(target, field_name, "")
    if migrated:
        setattr(pcfg, "credential_migration_done", True)
    return migrated


def scrub_plaintext_secrets_for_save(pcfg: Any, *, has_keyring) -> None:
    if bool(getattr(pcfg, "credential_use_plaintext_fallback", False)):
        return
    if not has_keyring():
        return
    for owner, field_name, _ in SECRET_FIELD_SPECS:
        target = getattr(pcfg, owner) if owner else pcfg
        if getattr(target, field_name, ""):
            setattr(target, field_name, "")
