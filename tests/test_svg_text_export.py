from utils.fontformat import FontFormat
from utils.svg_text_export import build_svg_text_handoff


class Block:
    xyxy = [10, 20, 80, 160]
    translation = "第12話?!"
    fontformat = FontFormat(font_family="Noto Sans CJK JP", font_size=18, writing_mode="vertical_rl", alignment=1)

    def get_text(self):
        return "src"


class Project:
    directory = "/tmp/missing_project"
    current_img = "p1.png"
    pages = {"p1.png": [Block()]}
    _image_info = {"p1.png": {"width": 100, "height": 200}}


def test_svg_text_handoff_writes_editable_vertical_text(tmp_path):
    manifest = build_svg_text_handoff(Project(), "p1.png", str(tmp_path))
    svg = (tmp_path / "p1.svg").read_text(encoding="utf-8")
    assert manifest["format"] == "ballonstranslator.svg_text_handoff.v1"
    assert manifest["text_layers"][0]["writing_mode"] == "vertical_rl"
    assert 'writing-mode="vertical-rl"' in svg
    assert "text_001" in svg
    assert "data-tate-chu-yoko" in svg
    assert manifest["text_layers"][0]["vertical_layout_plan"]["tate_chu_yoko_groups"]
    assert manifest["text_layers"][0]["font_runs"]
    assert manifest["warnings"]
