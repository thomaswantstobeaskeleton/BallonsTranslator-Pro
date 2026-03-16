"""No-op flow fixer: returns inputs unchanged."""
from typing import Any, Dict, List, Tuple

from . import BaseFlowFixer, register_flow_fixer


@register_flow_fixer(name="none")
class NoOpFlowFixer(BaseFlowFixer):
    """Flow fixer that returns previous_entries and new_translations unchanged."""

    def improve_flow(
        self,
        previous_entries: List[Dict[str, Any]],
        new_translations: List[str],
        target_lang: str = "en",
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        return previous_entries, new_translations
