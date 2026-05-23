from ui.dashboard_action_dispatcher import dispatch_dashboard_action, dispatch_message_for_result
from ui.dashboard_action_router import dashboard_action_id_for, dashboard_handler_for, safe_dashboard_routes


class FakeAction:
    def __init__(self):
        self.triggered = False

    def trigger(self):
        self.triggered = True


class FakeRecord:
    def __init__(self, action_id, action):
        self.action_id = action_id
        self.action = action


class FakeRegistry:
    def __init__(self, records):
        self._records = records

    def all_records(self):
        return list(self._records)


class FakeOwner:
    pass


def test_dispatch_uses_action_registry_first():
    owner = FakeOwner()
    action = FakeAction()
    owner._action_registry = FakeRegistry([FakeRecord("live.open", action)])

    result = dispatch_dashboard_action(owner, "live", "primary")

    assert result.handled is True
    assert result.route_found is True
    assert result.invoked == "action:live.open"
    assert action.triggered is True


def test_dispatch_falls_back_to_handler_name():
    owner = FakeOwner()
    owner.called = False

    def handler():
        owner.called = True

    owner.on_open_realtime_translator = handler

    result = dispatch_dashboard_action(owner, "live", "primary")

    assert result.handled is True
    assert result.invoked == "handler:on_open_realtime_translator"
    assert owner.called is True


def test_dispatch_reports_missing_route():
    owner = FakeOwner()
    result = dispatch_dashboard_action(owner, "missing", "action")

    assert result.handled is False
    assert result.route_found is False
    assert "No route" in dispatch_message_for_result(result)


def test_editor_layout_review_route_uses_current_handler():
    owner = FakeOwner()
    owner.called = False

    def handler():
        owner.called = True

    owner.shortcutLayoutReviewSelected = handler

    result = dispatch_dashboard_action(owner, "editor", "layout_review")

    assert result.handled is True
    assert result.route_found is True
    assert result.invoked == "handler:shortcutLayoutReviewSelected"
    assert owner.called is True


def test_router_has_safe_routes_and_known_action_ids():
    routes = safe_dashboard_routes()
    assert ("live", "primary") in routes
    assert ("editor", "layout_review") in routes
    assert dashboard_action_id_for("assist", "primary") == "translation.assist"
    assert dashboard_handler_for("assist", "primary") == "on_open_translation_assist_dock"
