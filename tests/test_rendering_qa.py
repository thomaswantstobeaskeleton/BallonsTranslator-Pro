from utils.fontformat import FontFormat
from utils.rendering_qa import analyze_text_block, apply_project_rendering_fixes, build_project_rendering_qa


class Cfg:
    render_fallback_fonts_latin = "Arial"
    render_fallback_fonts_cjk = "Noto Sans CJK JP"
    render_fallback_fonts_korean = "Noto Sans CJK KR"
    render_fallback_fonts_rtl = "Noto Naskh Arabic"
    render_fallback_fonts_emoji = "Noto Color Emoji"


class Block:
    def __init__(self, text, xyxy, fmt):
        self.translation = text
        self.rich_text = ""
        self.text = []
        self.xyxy = xyxy
        self.fontformat = fmt


class Project:
    def __init__(self, pages):
        self.pages = pages


def test_rendering_qa_detects_overflow_and_vertical_policy():
    fmt = FontFormat(font_size=32, writing_mode="vertical_rl", line_break_strategy="loose")
    blk = Block("これはテストです?!", [0, 0, 30, 80], fmt)
    diag = analyze_text_block(blk, "p1", 0, Cfg())
    assert "weak_vertical_line_break_strategy" in diag["warnings"]
    assert any(s["action"] == "set_line_break_strategy" for s in diag["suggestions"])


def test_project_rendering_fixes_apply_safe_style_changes():
    fmt = FontFormat(font_size=32, writing_mode="vertical_rl", line_break_strategy="loose", stroke_width=0.1, text_padding=0)
    project = Project({"p1": [Block("これはテストです?!", [0, 0, 30, 80], fmt)]})
    result = apply_project_rendering_fixes(project, config_obj=Cfg())
    fixed = project.pages["p1"][0].fontformat
    assert result["applied_count"] == 1
    assert fixed.line_break_strategy == "cjk_strict"
    assert fixed.text_padding >= 2.0
    assert fixed.font_size <= 32


def test_project_rendering_qa_summary_counts_pages_and_issues():
    project = Project({"p1": [Block("Hello", [0, 0, 200, 80], FontFormat(font_size=12))]})
    report = build_project_rendering_qa(project, config_obj=Cfg())
    assert report["summary"]["pages"] == 1
    assert report["summary"]["textboxes"] == 1

from utils.rendering_qa import flatten_rendering_qa_rows, rendering_qa_to_markdown


def test_rendering_qa_markdown_and_flatten_rows_include_actions():
    fmt = FontFormat(font_size=32, writing_mode="vertical_rl", line_break_strategy="loose")
    project = Project({"p1": [Block("これはテストです?!", [0, 0, 30, 80], fmt)]})
    report = build_project_rendering_qa(project, config_obj=Cfg())
    rows = flatten_rendering_qa_rows(report)
    assert rows[0]["page"] == "p1"
    assert "set_line_break_strategy" in rows[0]["suggestions"]
    md = rendering_qa_to_markdown(report)
    assert "Typography QA Report" in md
    assert "weak_vertical_line_break_strategy" in md
