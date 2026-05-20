"""Curated documentation catalog used by README and automation APIs."""
from __future__ import annotations

from typing import Dict, List


def build_docs_catalog() -> Dict[str, List[Dict[str, str]]]:
    return {
        "start_here": [
            {"path": "docs/TROUBLESHOOTING.md", "highlight": "Install/startup troubleshooting and common fixes."},
            {"path": "docs/GPU_ACCELERATION.md", "highlight": "GPU backends, vendor notes, and acceleration setup guidance."},
        ],
        "quality_translation_lettering": [
            {"path": "docs/QUALITY_RANKINGS.md", "highlight": "Quality/performance expectations for major module combinations."},
            {"path": "docs/MODELS_REFERENCE.md", "highlight": "Model/module reference list and practical selection notes."},
            {"path": "docs/TRANSLATION_CONTEXT_AND_GLOSSARY.md", "highlight": "Translation context packaging, glossary usage, and consistency guidance."},
            {"path": "docs/INDESIGN_LPTXT_WORKFLOW.md", "highlight": "InDesign and LPTXT handoff workflow for professional lettering."},
            {"path": "docs/TRANSLATION_ASSIST_PLAN.md", "highlight": "Translation Assist capabilities, compare modes/scopes, and roadmap status."},
        ],
        "automation_realtime_plans": [
            {"path": "docs/LOCAL_AUTOMATION_API.md", "highlight": "Local automation routes, examples, and MCP-style control surface."},
            {"path": "docs/REALTIME_TRANSLATION_MODE_PLAN.md", "highlight": "Realtime screen translation mode phases and current constraints."},
            {"path": "docs/ALTERNATIVES_FEATURE_GAP_IMPLEMENTATION_PLAN.md", "highlight": "Feature-gap implementation strategy and progress checkpoints."},
            {"path": "docs/FEATURE_PARITY_MATRIX.md", "highlight": "Feature parity status matrix across related tool categories."},
        ],
        "raw_sources_and_extensibility": [
            {"path": "docs/RAW_SOURCE_PROVIDER_EXPANSION_PLAN.md", "highlight": "Raw-source provider expansion roadmap and governance boundaries."},
            {"path": "docs/MANGA_SOURCE_PLUGIN_API.md", "highlight": "Provider plugin API surface and compliance expectations."},
            {"path": "docs/EXTERNAL_SOURCE_AUDIT.md", "highlight": "Local audit report for optional external source integrations."},
        ],
    }
