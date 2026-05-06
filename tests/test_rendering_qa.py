from utils.fontformat import FontFormat
from utils.rendering_qa import analyze_text_block, apply_project_rendering_fixes


class Cfg:
    render_fallback_fonts_latin = ""
    render_fallback_fonts_cjk = ""
    render_fallback_fonts_korean = ""
    render_fallback_fonts_rtl = ""
    render_fallback_fonts_emoji = ""


class Block:
    def __init__(self, text, xyxy, fmt):
        self.translation = text
        self.rich_text = ""
        self.text = []
        self.xyxy = xyxy
        self.fontformat = fmt


class Project:
    def __init__(self, blocks):
        self.pages = {"p.png": blocks}


def test_rendering_qa_suggests_vertical_mode_for_tall_cjk_box():
    fmt = FontFormat()
    fmt.writing_mode = "horizontal_ltr"
    blk = Block("こんにちは世界?!", [0, 0, 50, 220], fmt)
    diag = analyze_text_block(blk, "p.png", 0, config_obj=Cfg())
    assert "horizontal_cjk_in_tall_box" in diag["warnings"]
    assert any(s["action"] == "switch_writing_mode" for s in diag["suggestions"])


def test_apply_project_rendering_fixes_normalizes_vertical_punctuation_and_effects():
    fmt = FontFormat()
    fmt.writing_mode = "vertical_rl"
    fmt.frgb = [245, 245, 245]
    fmt.stroke_width = 0.0
    blk = Block("本当??", [0, 0, 60, 220], fmt)
    result = apply_project_rendering_fixes(Project([blk]), config_obj=Cfg())
    assert result["applied_count"] == 1
    assert "⁇" in blk.translation
    assert blk.fontformat.stroke_width > 0


def test_apply_project_rendering_fixes_honors_selected_rows_and_actions():
    fmt = FontFormat()
    fmt.writing_mode = "horizontal_ltr"
    blk = Block("こんにちは世界", [0, 0, 50, 220], fmt)
    project = Project([blk])
    skipped = apply_project_rendering_fixes(project, config_obj=Cfg(), selected_fixes=[])
    assert skipped["applied_count"] == 0
    assert blk.fontformat.writing_mode == "horizontal_ltr"
    applied = apply_project_rendering_fixes(
        project,
        config_obj=Cfg(),
        selected_fixes=[{"page": "p.png", "index": 0, "actions": ["switch_writing_mode"]}],
    )
    assert applied["applied_count"] == 1
    assert blk.fontformat.writing_mode == "vertical_rl"
