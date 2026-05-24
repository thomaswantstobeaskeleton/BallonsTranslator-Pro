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


def test_show_advanced_settings_default():
    """show_advanced_settings defaults to False so first-time users see Basic mode."""
    defaults = {f.name for f in ProgramConfig.__dataclass_fields__.values()}
    assert "show_advanced_settings" in defaults, "show_advanced_settings must be a ProgramConfig field"
    cfg = ProgramConfig()
    assert cfg.show_advanced_settings is False


def _extract_advanced_widget_names_from_source():
    """Parse configpanel.py source to find _ADVANCED_LAYOUT_WIDGET_NAMES without importing Qt."""
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


def test_advanced_widget_names_list_is_tuple_of_strings():
    """The canonical list of expert knob names should be a non-empty tuple of strings."""
    names = _extract_advanced_widget_names_from_source()
    assert isinstance(names, tuple)
    assert len(names) > 0
    assert all(isinstance(n, str) for n in names)
    expected = {
        "layout_short_line_penalty_spin",
        "layout_height_overflow_penalty_spin",
        "layout_stub_penalty_1word_spin",
        "optimize_line_breaks_checker",
        "layout_font_binary_search_checker",
    }
    assert expected.issubset(set(names)), f"Missing expected advanced widgets: {expected - set(names)}"


def test_new_svg_icon_files_exist():
    """All new dedicated icons created in Phase 3 must exist on disk."""
    icons_dir = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), "icons")
    required = [
        "menu_context_menu.svg",
        "menu_keyboard.svg",
        "menu_theme_customizer.svg",
        "menu_about.svg",
        "menu_update_github.svg",
        "menu_spellcheck.svg",
        "menu_import_styles.svg",
        "menu_export_styles.svg",
        "config_models.svg",
        "config_layout_engine.svg",
        "config_typesetting.svg",
        "config_canvas.svg",
        "config_integrations.svg",
        "config_general.svg",
    ]
    missing = [f for f in required if not osp.isfile(osp.join(icons_dir, f))]
    assert not missing, f"Missing icon files: {missing}"


def test_configpanel_source_contains_new_block_setup():
    """ConfigPanel.__init__ source should reference the 4 new top-level block names."""
    panel_path = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), "ui", "configpanel.py")
    src = open(panel_path, encoding="utf-8").read()
    for name in ("modelsConfigPanel", "layoutConfigPanel", "typesettingConfigPanel", "generalConfigPanel"):
        assert name in src, f"ConfigPanel source missing {name}"
