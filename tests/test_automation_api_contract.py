import pytest

from utils.automation_api_contract import build_route_discovery, normalize_job_task


def test_route_discovery_marks_mcp_and_job_routes():
    payload = build_route_discovery({
        "project_status": lambda body: {},
        "job_status": lambda body: {},
        "apply_edit": lambda body: {},
        "z_extra": lambda body: {},
    })
    assert payload["routes"] == ["apply_edit", "job_status", "project_status", "z_extra"]
    assert payload["methods"]["GET"] == ["events", "health", "mcp/commands", "realtime/events", "routes"]
    assert payload["mcp_routes"] == ["apply_edit", "project_status"]
    assert payload["job_routes"] == ["job_status"]
    assert payload["event_stream"] == "/events?job_id=<job_id>"


def test_normalize_job_task_aliases_and_rejects_unknown():
    assert normalize_job_task("render_current_page") == "render_page"
    assert normalize_job_task("proof_pack") == "proof_pack"
    assert normalize_job_task("export_archive") == "batch_export"
    with pytest.raises(ValueError):
        normalize_job_task("not_a_task")
