from ui.action_registry import ActionRegistry


class _DummyShortcut:
    def __init__(self, value: str):
        self._v = value

    def toString(self):
        return self._v


class _DummyIcon:
    def __init__(self, null=True):
        self._null = null

    def isNull(self):
        return self._null


class _DummyAction:
    def __init__(self, text: str, shortcut: str = "", tooltip: str = ""):
        self._text = text
        self._shortcut = _DummyShortcut(shortcut)
        self._tooltip = tooltip

    def text(self):
        return self._text

    def shortcut(self):
        return self._shortcut

    def toolTip(self):
        return self._tooltip

    def icon(self):
        return _DummyIcon(True)


def test_action_registry_unique_ids_and_discoverability():
    reg = ActionRegistry()
    a1 = _DummyAction("Open Project", "Ctrl+O")
    a2 = _DummyAction("Export All")
    reg.register_qaction(top_level="File", menu_path="File", qaction=a1)
    reg.register_qaction(top_level="File", menu_path="File > Export", qaction=a2)

    assert len(reg.all_records()) == 2
    assert not reg.has_duplicate_ids()
    labels = [label for label, _, _ in reg.discoverable_actions()]
    assert any("Open Project" in x for x in labels)
    assert any("Export All" in x for x in labels)


def test_action_registry_detects_shortcut_conflicts():
    reg = ActionRegistry()
    reg.register_qaction(top_level="Edit", menu_path="Edit", qaction=_DummyAction("Undo", "Ctrl+Z"))
    reg.register_qaction(top_level="Edit", menu_path="Edit", qaction=_DummyAction("Also Undo", "Ctrl+Z"))

    conflicts = reg.duplicate_shortcuts()
    assert "Ctrl+Z" in conflicts
    assert len(conflicts["Ctrl+Z"]) == 2


def test_action_registry_hides_unavailable_by_default_and_includes_when_requested():
    class _DisabledDummy(_DummyAction):
        def __init__(self, text: str, enabled: bool):
            super().__init__(text)
            self._enabled = enabled

        def isEnabled(self):
            return self._enabled

    reg = ActionRegistry()
    reg.register_qaction(top_level="Tools", menu_path="Tools", qaction=_DisabledDummy("Enabled action", True))
    reg.register_qaction(top_level="Tools", menu_path="Tools", qaction=_DisabledDummy("Disabled action", False))

    visible_default = [x[0] for x in reg.discoverable_actions()]
    visible_all = [x[0] for x in reg.discoverable_actions(show_unavailable=True)]

    assert any("Enabled action" in t for t in visible_default)
    assert not any("Disabled action" in t for t in visible_default)
    assert any("Disabled action" in t for t in visible_all)


def test_action_registry_to_rows_contains_expected_fields():
    reg = ActionRegistry()
    reg.register_qaction(top_level="View", menu_path="View", qaction=_DummyAction("Toggle Panel", "Ctrl+1", "Toggle panel visibility"))
    rows = reg.to_rows()
    assert len(rows) == 1
    row = rows[0]
    assert row["action_id"].startswith("view.")
    assert row["top_level"] == "View"
    assert row["label"] == "Toggle Panel"
    assert row["shortcut"] == "Ctrl+1"
    assert row["tooltip"] == "Toggle panel visibility"


def test_action_registry_record_for_action_lookup():
    a = _DummyAction("Find Me", "Ctrl+F")
    reg = ActionRegistry()
    rec = reg.register_qaction(top_level="Edit", menu_path="Edit", qaction=a)
    got = reg.record_for_action(a)
    assert got is not None
    assert got.action_id == rec.action_id


def test_action_registry_summary_stats_reports_counts():
    reg = ActionRegistry()
    reg.register_qaction(top_level="File", menu_path="File", qaction=_DummyAction("Open", "Ctrl+O"))
    reg.register_qaction(top_level="Edit", menu_path="Edit", qaction=_DummyAction("Copy", "Ctrl+C"))
    stats = reg.summary_stats()
    assert stats["total_actions"] == 2
    assert stats["enabled_actions"] == 2
    assert stats["disabled_actions"] == 0
    assert isinstance(stats["categories"], dict)
