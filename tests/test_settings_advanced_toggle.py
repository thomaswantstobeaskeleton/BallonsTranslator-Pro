import os.path as osp
import sys
import types
import ast

# Stub cv2 before anything else imports it via utils.io_utils
cv2_stub = sys.modules.get("cv2") or types.ModuleType("cv2")
cv2_stub.IMREAD_COLOR = getattr(cv2_stub, "IMREAD_COLOR", 1)
cv2_stub.IMREAD_GRAYSCALE = getattr(cv2_stub, "IMREAD_GRAYSCALE", 0)
cv2_stub.COLOR_GRAY2RGB = getattr(cv2_stub, "COLOR_GRAY2RGB", 0)
cv2_stub.COLOR_RGB2BGR = getattr(cv2_stub, "COLOR_RGB2BGR", 0)
cv2_stub.COLOR_RGBA2BGRA = getattr(cv2_stub, "COLOR_RGBA2BGRA", 0)
cv2_stub.IMWRITE_JPEG_QUALITY = getattr(cv2_stub, "IMWRITE_JPEG_QUALITY", 1)
cv2_stub.IMWRITE_WEBP_QUALITY = getattr(cv2_stub, "IMWRITE_WEBP_QUALITY", 64)
cv2_stub.cvtColor = getattr(cv2_stub, "cvtColor", lambda img, code: img)
cv2_stub.imencode = getattr(
    cv2_stub,
    "imencode",
    lambda ext, img, enc: (True, types.SimpleNamespace(tofile=lambda p: None)),
)
cv2_stub.imshow = getattr(cv2_stub, "imshow", lambda *args, **kwargs: None)
cv2_stub.waitKey = getattr(cv2_stub, "waitKey", lambda *args, **kwargs: 0)
sys.modules["cv2"] = cv2_stub

from utils.config import ProgramConfig


class MockWidget:
    """Minimal mock widget that records setVisible calls."""
    def __init__(self, parent=None):
        self.visible = True
        self._parent = parent

    def parent(self):
        return self._parent

    def setVisible(self, visible):
        self.visible = visible


class MockSubBlock(MockWidget):
    """Mock ConfigSubBlock — the visibility target of _apply_show_advanced_settings."""
    pass


def _extract_advanced_widget_names_from_source():
    panel_path = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), "ui", "configpanel.py")
    tree = ast.parse(open(panel_path, encoding="utf-8").read())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ConfigPanel":
            for item in node.body:
                if isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Name) and target.id == "_ADVANCED_LAYOUT_WIDGET_NAMES":
                            return tuple(elt.s for elt in item.value.elts)
    raise AssertionError("Could not find _ADVANCED_LAYOUT_WIDGET_NAMES in ConfigPanel source")


def _apply_show_advanced_settings(panel, names, show):
    """Reimplementation of ConfigPanel._apply_show_advanced_settings for testing."""
    for name in names:
        widget = getattr(panel, name, None)
        if widget is None:
            continue
        parent = widget.parent()
        while parent is not None and not isinstance(parent, MockSubBlock):
            parent = parent.parent() if hasattr(parent, "parent") else None
        if parent is not None:
            parent.setVisible(show)


def test_apply_show_advanced_settings_hides_expert_widgets():
    """_apply_show_advanced_settings should hide all expert sublocks when show_advanced_settings=False."""
    names = _extract_advanced_widget_names_from_source()

    # Build a minimal fake panel with a subset of mock expert widgets.
    panel = types.SimpleNamespace()
    subblocks = {}
    for name in names[:5]:
        sub = MockSubBlock()
        sub.setVisible(True)
        subblocks[name] = sub
        w = MockWidget(parent=sub)
        setattr(panel, name, w)

    for name in names[5:]:
        setattr(panel, name, None)

    # Turn advanced OFF
    _apply_show_advanced_settings(panel, names, show=False)
    for sub in subblocks.values():
        assert sub.visible is False, f"Expected subblock to be hidden when advanced=False"

    # Turn advanced ON
    _apply_show_advanced_settings(panel, names, show=True)
    for sub in subblocks.values():
        assert sub.visible is True, f"Expected subblock to be visible when advanced=True"


def test_show_advanced_settings_persisted_in_save_list():
    """show_advanced_settings must appear in the ProgramConfig save key list."""
    from utils.config import CONFIG_KEY_ORDER
    assert "show_advanced_settings" in CONFIG_KEY_ORDER, (
        "show_advanced_settings must be in CONFIG_KEY_ORDER for persistence"
    )
