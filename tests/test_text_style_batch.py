from utils.fontformat import FontFormat
from utils.text_style_batch import apply_text_style_batch, normalize_text_style_updates


class Block:
    def __init__(self, auto=False):
        self.fontformat = FontFormat(font_family="Old", font_size=20, alignment=0)
        self.fontformat.auto_fit_font_size = auto


class Project:
    pages = {"p1.png": [Block(auto=True), Block(auto=False)], "p2.png": [Block(auto=True)]}


def test_normalize_text_style_updates_keeps_safe_renderer_fields():
    out = normalize_text_style_updates({
        "font_family": "Noto Sans",
        "writing_mode": "vertical-rl",
        "fit_mode": "balance",
        "line_break_strategy": "cjk-strict",
        "stroke_width": "0.08",
        "unknown": "ignored",
    })
    assert out["font_family"] == "Noto Sans"
    assert out["writing_mode"] == "vertical_rl"
    assert out["fit_mode"] == "balance"
    assert out["line_break_strategy"] == "cjk_strict"
    assert "unknown" not in out


def test_apply_text_style_batch_can_target_only_auto_sized_blocks():
    project = Project()
    result = apply_text_style_batch(project, {"font_family": "Noto Sans", "alignment": 1}, only_auto_sized=True)
    assert result["changed"] == 2
    assert project.pages["p1.png"][0].fontformat.font_family == "Noto Sans"
    assert project.pages["p1.png"][1].fontformat.font_family == "Old"
    assert project.pages["p2.png"][0].fontformat.alignment == 1
