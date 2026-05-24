from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class DashboardActionRoute:
    mode: str
    action: str
    action_id: str = ""
    handler_name: str = ""
    title: str = ""
    description: str = ""
    safe_to_wire: bool = True

    def route_key(self) -> Tuple[str, str]:
        return (self.mode, self.action)


DASHBOARD_ACTION_ROUTES: Dict[Tuple[str, str], DashboardActionRoute] = {
    ("editor", "primary"): DashboardActionRoute("editor", "primary", "file.open_folder", "_welcome_open_folder", "Open project/folder"),
    ("editor", "run_pipeline"): DashboardActionRoute("editor", "run_pipeline", "pipeline.run", "run_imgtrans", "Run full pipeline"),
    ("editor", "selected_ocr"): DashboardActionRoute("editor", "selected_ocr", "ocr.rerun_selected", "on_open_ocr_crop_inspector_requested", "OCR selected blocks"),
    ("editor", "layout_review"): DashboardActionRoute("editor", "layout_review", "qa.layout_review", "shortcutLayoutReviewSelected", "Review layout"),
    ("editor", "export_proof"): DashboardActionRoute("editor", "export_proof", "export.proof_pack", "on_export_lettering_proof_pack", "Export proof pack"),
    ("live", "primary"): DashboardActionRoute("live", "primary", "live.open", "on_open_realtime_translator", "Start live region picker"),
    ("live", "pick_region"): DashboardActionRoute("live", "pick_region", "live.pick_region", "on_open_realtime_translator", "Select live region"),
    ("live", "chrome_profile"): DashboardActionRoute("live", "chrome_profile", "live.chrome_profile", "on_open_realtime_translator", "Chrome Manhua preset"),
    ("live", "overlay"): DashboardActionRoute("live", "overlay", "live.overlay_settings", "on_open_realtime_translator", "Overlay settings"),
    ("live", "privacy"): DashboardActionRoute("live", "privacy", "live.privacy", "on_open_realtime_translator", "Live privacy controls"),
    ("assist", "primary"): DashboardActionRoute("assist", "primary", "translation.assist", "on_open_translation_assist_dock", "Open Translation Assist"),
    ("assist", "compare"): DashboardActionRoute("assist", "compare", "translation.compare_providers", "on_open_translation_assist_dock", "Compare providers"),
    ("assist", "tm"): DashboardActionRoute("assist", "tm", "translation.memory", "on_open_translation_assist_dock", "Translation memory"),
    ("assist", "glossary"): DashboardActionRoute("assist", "glossary", "translation.glossary", "on_open_translation_assist_dock", "Glossary violations"),
    ("assist", "sfx"): DashboardActionRoute("assist", "sfx", "translation.sfx_dictionary", "on_open_translation_assist_dock", "SFX dictionary"),
    ("downloader", "primary"): DashboardActionRoute("downloader", "primary", "tools.raw_downloader", "on_open_manga_source", "Search sources"),
    ("downloader", "search"): DashboardActionRoute("downloader", "search", "tools.raw_downloader", "on_open_manga_source", "Search sources"),
    ("downloader", "queue"): DashboardActionRoute("downloader", "queue", "batch.queue", "on_open_batch_queue", "Chapter queue"),
    ("downloader", "health"): DashboardActionRoute("downloader", "health", "diagnostics.source_health", "on_environment_doctor", "Source health"),
    ("downloader", "import"): DashboardActionRoute("downloader", "import", "file.open_folder", "_welcome_open_folder", "Import to project"),
    ("quick_image", "primary"): DashboardActionRoute("quick_image", "primary", "file.open_images", "leftBar.onOpenImages", "Open image files"),
    ("batch", "primary"): DashboardActionRoute("batch", "primary", "batch.queue", "on_open_batch_queue", "Open batch queue"),
    ("models", "primary"): DashboardActionRoute("models", "primary", "models.manage", "on_open_manage_models", "Manage models"),
    ("settings", "primary"): DashboardActionRoute("settings", "primary", "settings.open", "setupConfigUI", "Open settings"),
    ("diagnostics", "primary"): DashboardActionRoute("diagnostics", "primary", "diagnostics.environment_doctor", "on_environment_doctor", "Environment doctor"),
}


def dashboard_route_for(mode: str, action: str) -> Optional[DashboardActionRoute]:
    return DASHBOARD_ACTION_ROUTES.get((str(mode or ""), str(action or "")))


def dashboard_action_id_for(mode: str, action: str) -> str:
    route = dashboard_route_for(mode, action)
    return route.action_id if route else ""


def dashboard_handler_for(mode: str, action: str) -> str:
    route = dashboard_route_for(mode, action)
    return route.handler_name if route else ""


def safe_dashboard_routes() -> Dict[Tuple[str, str], DashboardActionRoute]:
    return {key: route for key, route in DASHBOARD_ACTION_ROUTES.items() if route.safe_to_wire}
