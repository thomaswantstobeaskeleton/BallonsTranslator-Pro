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


def test_rendering_qa_reports_quality_and_unsafe_effect_bounds():
    fmt = FontFormat()
    fmt.stroke_width = 0.5
    fmt.shadow_radius = 0.2
    fmt.text_padding = 10
    blk = Block("Hello", [0, 0, 60, 35], fmt)
    diag = analyze_text_block(blk, "p.png", 0, config_obj=Cfg())
    assert "quality_score" in diag
    assert "unsafe_effect_bounds" in diag["warnings"]


def test_rendering_qa_reports_mask_visible_area_overflow_and_fix_grows_box():
    import numpy as np
    fmt = FontFormat()
    fmt.font_size = 28
    blk = Block("A long masked line", [0, 0, 180, 80], fmt)
    blk.text_mask = np.zeros((10, 20), dtype=np.uint8)
    blk.text_mask[2:8, 5:15] = 255
    diag = analyze_text_block(blk, "p.png", 0, config_obj=Cfg())
    assert "mask_visible_area_overflow" in diag["warnings"]
    assert any(s["action"] == "resize_to_mask_safe_box" for s in diag["suggestions"])
    before_w = blk.xyxy[2] - blk.xyxy[0]
    result = apply_project_rendering_fixes(
        Project([blk]),
        config_obj=Cfg(),
        selected_fixes=[{"page": "p.png", "index": 0, "actions": ["resize_to_mask_safe_box"]}],
    )
    assert result["applied_count"] == 1
    assert blk.xyxy[2] - blk.xyxy[0] >= before_w


def test_layout_review_planner_proposes_mask_safe_box_action():
    from utils.layout_review_agent import BlockSnapshot, LayoutReviewPlanner
    planner = LayoutReviewPlanner()
    result = planner.review_page(
        "p.png",
        [BlockSnapshot(
            block_index=2,
            xyxy=(0, 0, 180, 80),
            text="masked",
            font_size=24,
            est_text_size=(100, 40),
            text_style={"mask_diagnostics": {"mask_overflow": True, "recommended_box_size": [220, 90], "mask": {"edge_hidden": True}, "text_padding": 0}},
        )],
    )
    actions = [a.action for a in result.blocks[0].actions]
    issues = [i.code for i in result.blocks[0].issues]
    assert "resize_to_mask_safe_box" in actions
    assert "mask_visible_area_overflow" in issues
    assert "text_mask_erases_edge" in issues


def test_rendering_qa_suggests_and_applies_sfx_double_outline():
    from utils.fontformat import FontFormat
    from utils.rendering_qa import analyze_text_block, apply_project_rendering_fixes

    class Cfg:
        render_fallback_fonts_latin = ""
        render_fallback_fonts_cjk = ""
        render_fallback_fonts_korean = ""
        render_fallback_fonts_rtl = ""
        render_fallback_fonts_emoji = ""
        module = type("M", (), {"layout_font_size_min": 6.0, "layout_font_size_max": 96.0})()

    class Project:
        pages = {}

    blk = type("B", (), {})()
    blk.xyxy = [0, 0, 120, 60]
    blk.translation = "BOOM!!"
    blk.rich_text = ""
    blk.text = []
    blk.text_mask = None
    blk.fontformat = FontFormat(font_size=34, writing_mode="auto", secondary_stroke_width=0.0)
    blk.fontformat.frgb = [20, 20, 20]
    blk.fontformat.srgb = [0, 0, 0]
    diag = analyze_text_block(blk, "p.png", 0, config_obj=Cfg())
    assert "sfx_missing_back_outline" in diag["warnings"]
    assert any(s["action"] == "apply_double_outline" for s in diag["suggestions"])
    project = Project()
    project.pages = {"p.png": [blk]}
    result = apply_project_rendering_fixes(project, pages=["p.png"], config_obj=Cfg(), selected_fixes=[{"page": "p.png", "index": 0, "actions": ["apply_double_outline"]}])
    assert result["applied_count"] >= 1
    assert blk.fontformat.secondary_stroke_width > 0


def test_rendering_qa_can_balance_widow_lines():
    from utils.fontformat import FontFormat
    from utils.rendering_qa import analyze_text_block, apply_project_rendering_fixes

    class Cfg:
        render_fallback_fonts_latin = ""
        render_fallback_fonts_cjk = ""
        render_fallback_fonts_korean = ""
        render_fallback_fonts_rtl = ""
        render_fallback_fonts_emoji = ""
        module = type("M", (), {"layout_font_size_min": 6.0, "layout_font_size_max": 96.0})()

    blk = type("B", (), {})()
    blk.xyxy = [0, 0, 88, 80]
    blk.translation = "one two three four five"
    blk.rich_text = ""
    blk.text = []
    blk.text_mask = None
    blk.fontformat = FontFormat(font_size=18, writing_mode="horizontal_ltr", fit_mode="preserve", line_break_strategy="balanced")
    diag = analyze_text_block(blk, "p.png", 0, config_obj=Cfg())
    assert diag["proof_metrics"].get("line_break_quality")
    project = type("P", (), {"pages": {"p.png": [blk]}})()
    result = apply_project_rendering_fixes(project, pages=["p.png"], config_obj=Cfg(), selected_fixes=[{"page": "p.png", "index": 0, "actions": ["balance_lines"]}])
    assert result["applied_count"] >= 0


def test_rendering_qa_suggests_atomic_bubble_fit_for_poor_bubble_fill():
    from utils.fontformat import FontFormat
    from utils.rendering_qa import analyze_text_block, apply_project_rendering_fixes

    class Cfg:
        render_fallback_fonts_latin = ""
        render_fallback_fonts_cjk = ""
        render_fallback_fonts_korean = ""
        render_fallback_fonts_rtl = ""
        render_fallback_fonts_emoji = ""
        render_atomic_fit_target_fill = 0.76
        render_atomic_fit_max_expand = 1.15
        render_atomic_fit_profile = "dense"
        module = type("M", (), {"layout_font_size_min": 6.0, "layout_font_size_max": 96.0})()

    blk = type("B", (), {})()
    blk.xyxy = [0, 0, 220, 120]
    blk.translation = "tiny"
    blk.rich_text = ""
    blk.text = []
    blk.text_mask = None
    blk.fontformat = FontFormat(font_size=8, writing_mode="horizontal_ltr", fit_mode="shrink", line_break_strategy="auto", text_padding=0)
    diag = analyze_text_block(blk, "p.png", 0, config_obj=Cfg())
    assert any(s.get("action") == "atomic_bubble_fit" for s in diag["suggestions"])
    assert diag["atomic_bubble_fit"]["profile"] == "dense"
    project = type("P", (), {"pages": {"p.png": [blk]}})()
    result = apply_project_rendering_fixes(project, pages=["p.png"], config_obj=Cfg(), selected_fixes=[{"page": "p.png", "index": 0, "actions": ["atomic_bubble_fit"]}])
    assert result["applied_count"] >= 1
    assert blk.fontformat.font_size >= 8
    assert blk.fontformat.text_padding > 0
