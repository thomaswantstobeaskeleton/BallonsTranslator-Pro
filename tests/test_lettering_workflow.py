from types import SimpleNamespace

from utils.fontformat import FontFormat
from utils.lettering_workflow import build_lettering_workflow_plan, next_rendering_issue
from utils.rendering_qa import build_project_rendering_qa


def _cfg():
    return SimpleNamespace(
        render_fallback_fonts_latin="",
        render_fallback_fonts_cjk="Noto Sans CJK JP",
        render_fallback_fonts_korean="",
        render_fallback_fonts_rtl="",
        render_fallback_fonts_emoji="",
        vertical_cjk_rotate_latin=True,
        vertical_cjk_punctuation_hang=True,
        module=SimpleNamespace(layout_font_size_min=6.0, layout_font_size_max=96.0),
    )


def test_lettering_workflow_plans_polish_smart_fit_and_proof_steps():
    blk = SimpleNamespace(
        xyxy=[0, 0, 42, 120],
        translation="こんにちは?!",
        rich_text="",
        text=[],
        text_mask=None,
        fontformat=FontFormat(font_size=28, writing_mode="auto", fit_mode="preserve", text_padding=0),
    )
    project = SimpleNamespace(pages={"p.png": [blk]})
    plan = build_lettering_workflow_plan(project, pages=["p.png"], config_obj=_cfg())
    step_ids = [step["id"] for step in plan["steps"]]
    assert "polish_typography" in step_ids
    assert "smart_fit" in step_ids
    assert "export_lettering_proof" in step_ids
    assert plan["selected_fixes"]


def test_next_rendering_issue_wraps_current_page_rows():
    blk1 = SimpleNamespace(xyxy=[0, 0, 40, 25], translation="Very long text here", rich_text="", text=[], text_mask=None, fontformat=FontFormat(font_size=24, fit_mode="preserve"))
    blk2 = SimpleNamespace(xyxy=[0, 0, 40, 25], translation="Another long text", rich_text="", text=[], text_mask=None, fontformat=FontFormat(font_size=24, fit_mode="preserve"))
    project = SimpleNamespace(pages={"p.png": [blk1, blk2]})
    report = build_project_rendering_qa(project, pages=["p.png"], include_ok=False, config_obj=_cfg())
    first = next_rendering_issue(report, "p.png", -1)
    second = next_rendering_issue(report, "p.png", first["index"])
    wrapped = next_rendering_issue(report, "p.png", 99)
    assert first["found"] and second["found"]
    assert wrapped["wrapped"] is True
