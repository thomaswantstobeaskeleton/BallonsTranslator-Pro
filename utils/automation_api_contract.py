"""Automation API route and job contract helpers.

This module is intentionally UI-free so route discovery, docs generation, and
headless tests can share one stable contract without importing PyQt.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set


MCP_COMMAND_ROUTES: Set[str] = {
    "open_project",
    "project_status",
    "list_pages",
    "apply_edit",
    "run_pipeline",
    "render",
    "export",
    "undo",
    "redo",
    "realtime_status",
    "realtime_start",
    "realtime_stop",
    "realtime_translate_now",
    "translation_assist_block",
    "translation_assist_apply_candidate",
    "docs_catalog",
}

JOB_TASK_ALIASES: Mapping[str, str] = {
    "run_pipeline": "run_pipeline",
    "pipeline_run": "run_pipeline",
    "render": "render_page",
    "render_page": "render_page",
    "render_current_page": "render_page",
    "export": "export",
    "batch_export": "batch_export",
    "export_archive": "batch_export",
    "proof_pack": "proof_pack",
    "export_lettering_proof": "proof_pack",
    "lettering_proof": "proof_pack",
}

JOB_TASKS: Set[str] = set(JOB_TASK_ALIASES.values())


@dataclass(frozen=True)
class RouteDiscovery:
    routes: List[str]
    get_routes: List[str] = field(default_factory=lambda: ["health", "routes", "events"])
    mcp_routes: List[str] = field(default_factory=list)
    job_routes: List[str] = field(default_factory=list)

    def to_payload(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "routes": self.routes,
            "count": len(self.routes),
            "methods": {
                "GET": self.get_routes,
                "POST": self.routes,
            },
            "mcp_routes": self.mcp_routes,
            "job_routes": self.job_routes,
            "event_stream": "/events?job_id=<job_id>",
        }


def normalize_job_task(task: str) -> str:
    key = str(task or "").strip().lower()
    if key not in JOB_TASK_ALIASES:
        valid = ", ".join(sorted(JOB_TASK_ALIASES))
        raise ValueError(f"task must be one of: {valid}")
    return JOB_TASK_ALIASES[key]


def build_route_discovery(handlers: Mapping[str, Any], *, get_routes: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    routes = sorted(str(k) for k in handlers.keys())
    get_list = sorted(set(str(r) for r in (get_routes or ["health", "routes", "events", "mcp/commands", "realtime/events"])))
    discovery = RouteDiscovery(
        routes=routes,
        get_routes=get_list,
        mcp_routes=[r for r in routes if r in MCP_COMMAND_ROUTES],
        job_routes=[r for r in routes if r.startswith("job_") or r == "jobs_list"],
    )
    return discovery.to_payload()
