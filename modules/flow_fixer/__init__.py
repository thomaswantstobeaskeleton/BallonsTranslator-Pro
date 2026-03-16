"""
Flow fixer: optional local post-processing for video subtitle flow/continuity.
Runs after translation; no cloud API calls. Used by the Video translator.
"""
from typing import Any, Dict, List, Tuple

from utils.registry import Registry

FLOW_FIXERS = Registry("flow_fixer")
register_flow_fixer = FLOW_FIXERS.register_module


class BaseFlowFixer:
    """Base for subtitle flow fixers. improve_flow returns revised previous entries and new translations."""

    def improve_flow(
        self,
        previous_entries: List[Dict[str, Any]],
        new_translations: List[str],
        target_lang: str = "en",
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Optionally revise previous subtitle entries and the new segment for better flow.
        Return (revised_previous_entries, revised_new_translations). Default: return inputs unchanged.
        """
        return previous_entries, new_translations


# Register built-in fixers (import side-effect so FLOW_FIXERS has "none", "local_server", "openrouter", "openai")
from .none_flow_fixer import NoOpFlowFixer  # noqa: E402
from .local_server_flow_fixer import LocalServerFlowFixer  # noqa: E402
from .openrouter_flow_fixer import OpenRouterFlowFixer  # noqa: E402
from .openai_flow_fixer import OpenAIFlowFixer  # noqa: E402


def get_flow_fixer(name: str, **params) -> BaseFlowFixer:
    """Build a flow fixer instance by name. Returns no-op fixer if name is empty or 'none' or unknown."""
    if not name or (isinstance(name, str) and name.strip().lower() == "none"):
        name = "none"
    cls = FLOW_FIXERS.get(name)
    if cls is None:
        cls = FLOW_FIXERS.get("none")
    if cls is None:
        return NoOpFlowFixer()
    return cls(**params)


__all__ = [
    "FLOW_FIXERS",
    "BaseFlowFixer",
    "register_flow_fixer",
    "get_flow_fixer",
    "NoOpFlowFixer",
    "LocalServerFlowFixer",
    "OpenRouterFlowFixer",
    "OpenAIFlowFixer",
]
