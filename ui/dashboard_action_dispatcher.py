from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .dashboard_action_router import DashboardActionRoute, dashboard_route_for


@dataclass(frozen=True)
class DashboardDispatchResult:
    mode: str
    action: str
    handled: bool
    route_found: bool
    invoked: str = ""
    message: str = ""


def _find_action_registry(owner: Any):
    """Best-effort lookup for the action registry without depending on MainWindow."""
    for attr in ("action_registry", "_action_registry"):
        reg = getattr(owner, attr, None)
        if reg is not None:
            return reg
    for holder_name in ("titleBar", "titlebar", "title_bar"):
        holder = getattr(owner, holder_name, None)
        if holder is None:
            continue
        for attr in ("action_registry", "_action_registry"):
            reg = getattr(holder, attr, None)
            if reg is not None:
                return reg
    return None


def _trigger_action_by_id(owner: Any, action_id: str) -> bool:
    if not action_id:
        return False
    registry = _find_action_registry(owner)
    if registry is None or not hasattr(registry, "all_records"):
        return False
    try:
        for record in registry.all_records():
            if getattr(record, "action_id", "") == action_id:
                action_obj = getattr(record, "action", None)
                if action_obj is not None and hasattr(action_obj, "trigger"):
                    action_obj.trigger()
                    return True
    except Exception:
        return False
    return False


def _invoke_handler(owner: Any, handler_name: str) -> bool:
    if not handler_name:
        return False
    handler = getattr(owner, handler_name, None)
    if handler is None or not callable(handler):
        return False
    handler()
    return True


def dispatch_dashboard_action(owner: Any, mode: str, action: str) -> DashboardDispatchResult:
    """Safely dispatch a dashboard action to the existing app when possible.

    Resolution order:
    1. Action registry ID, if present and registered.
    2. Existing handler method name on the owner.
    3. Structured unhandled result for future wiring.

    This lets the modern dashboards become useful incrementally without forcing
    a risky all-at-once MainWindow rewrite.
    """
    route: Optional[DashboardActionRoute] = dashboard_route_for(mode, action)
    if route is None:
        return DashboardDispatchResult(mode, action, False, False, message="No route is registered for this dashboard action.")

    if route.action_id and _trigger_action_by_id(owner, route.action_id):
        return DashboardDispatchResult(mode, action, True, True, invoked=f"action:{route.action_id}", message=route.title)

    try:
        if route.handler_name and _invoke_handler(owner, route.handler_name):
            return DashboardDispatchResult(mode, action, True, True, invoked=f"handler:{route.handler_name}", message=route.title)
    except Exception as exc:
        return DashboardDispatchResult(mode, action, False, True, invoked=f"handler:{route.handler_name}", message=f"Dashboard action failed: {exc}")

    if not route.safe_to_wire:
        return DashboardDispatchResult(
            mode,
            action,
            False,
            True,
            message=f"{route.title or action} is intentionally marked as not wired yet.",
        )
    return DashboardDispatchResult(
        mode,
        action,
        False,
        True,
        message=f"{route.title or action} has a route but no matching handler/action was found.",
    )


def dispatch_message_for_result(result: DashboardDispatchResult) -> str:
    if result.handled:
        return result.message or f"Handled {result.mode}:{result.action}"
    if result.message:
        return result.message
    return f"Dashboard action is not wired yet: {result.mode}:{result.action}"
