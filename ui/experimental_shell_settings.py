from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

EXPERIMENTAL_APP_SHELL_KEY = "enable_experimental_app_shell"
EXPERIMENTAL_MODE_RAIL_KEY = "enable_experimental_mode_rail"
EXPERIMENTAL_EDITOR_INSPECTOR_KEY = "enable_experimental_editor_inspector"
EXPERIMENTAL_JOB_DRAWER_KEY = "enable_experimental_job_drawer"
EXPERIMENTAL_SHELL_MODE_KEY = "experimental_shell_initial_mode"
EXPERIMENTAL_SHELL_SPLITTER_KEY = "experimental_shell_splitter_state"

VALID_SHELL_MODES = {
    "home",
    "editor",
    "live",
    "quick_image",
    "downloader",
    "batch",
    "assist",
    "models",
    "settings",
    "diagnostics",
}


@dataclass(frozen=True)
class ExperimentalShellSettings:
    enable_app_shell: bool = False
    enable_mode_rail: bool = False
    enable_editor_inspector: bool = False
    enable_job_drawer: bool = False
    initial_mode: str = "home"
    splitter_state: str = ""

    def as_config_patch(self) -> dict:
        return {
            EXPERIMENTAL_APP_SHELL_KEY: self.enable_app_shell,
            EXPERIMENTAL_MODE_RAIL_KEY: self.enable_mode_rail,
            EXPERIMENTAL_EDITOR_INSPECTOR_KEY: self.enable_editor_inspector,
            EXPERIMENTAL_JOB_DRAWER_KEY: self.enable_job_drawer,
            EXPERIMENTAL_SHELL_MODE_KEY: self.initial_mode,
            EXPERIMENTAL_SHELL_SPLITTER_KEY: self.splitter_state,
        }


def _get_bool(source: Any, key: str, default: bool) -> bool:
    if isinstance(source, Mapping):
        return bool(source.get(key, default))
    return bool(getattr(source, key, default))


def _get_str(source: Any, key: str, default: str) -> str:
    if isinstance(source, Mapping):
        value = source.get(key, default)
    else:
        value = getattr(source, key, default)
    return str(value or default)


def normalize_shell_mode(mode: str) -> str:
    value = str(mode or "home").strip().lower()
    return value if value in VALID_SHELL_MODES else "home"


def read_experimental_shell_settings(source: Any) -> ExperimentalShellSettings:
    return ExperimentalShellSettings(
        enable_app_shell=_get_bool(source, EXPERIMENTAL_APP_SHELL_KEY, False),
        enable_mode_rail=_get_bool(source, EXPERIMENTAL_MODE_RAIL_KEY, False),
        enable_editor_inspector=_get_bool(source, EXPERIMENTAL_EDITOR_INSPECTOR_KEY, False),
        enable_job_drawer=_get_bool(source, EXPERIMENTAL_JOB_DRAWER_KEY, False),
        initial_mode=normalize_shell_mode(_get_str(source, EXPERIMENTAL_SHELL_MODE_KEY, "home")),
        splitter_state=_get_str(source, EXPERIMENTAL_SHELL_SPLITTER_KEY, ""),
    )


def ensure_experimental_shell_defaults(config_obj: Any) -> ExperimentalShellSettings:
    """Apply in-memory defaults to a config-like object and return normalized settings.

    This is intentionally separate from `utils.config.ProgramConfig` so the
    experimental shell can iterate without risky broad config rewrites.  A later
    migration patch can add these fields to ProgramConfig and call this helper
    during load.
    """
    settings = read_experimental_shell_settings(config_obj)
    for key, value in settings.as_config_patch().items():
        if not hasattr(config_obj, key):
            try:
                setattr(config_obj, key, value)
            except Exception:
                pass
    return settings


def shell_feature_flags(settings: ExperimentalShellSettings | Mapping[str, Any] | Any) -> dict:
    if not isinstance(settings, ExperimentalShellSettings):
        settings = read_experimental_shell_settings(settings)
    return {
        "app_shell": settings.enable_app_shell,
        "mode_rail": settings.enable_app_shell or settings.enable_mode_rail,
        "editor_inspector": settings.enable_app_shell or settings.enable_editor_inspector,
        "job_drawer": settings.enable_app_shell or settings.enable_job_drawer,
    }


def allowed_shell_modes() -> Iterable[str]:
    return tuple(sorted(VALID_SHELL_MODES))
