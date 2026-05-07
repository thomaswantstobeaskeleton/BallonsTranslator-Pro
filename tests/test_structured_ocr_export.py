import sys
import types

cv2_stub = sys.modules.get("cv2") or types.ModuleType("cv2")
cv2_stub.__file__ = getattr(cv2_stub, "__file__", "cv2_stub.py")
cv2_stub.__getattr__ = lambda name: (_ for _ in ()).throw(AttributeError(name)) if name.startswith("__") else 1
sys.modules["cv2"] = cv2_stub

from utils.fontformat import FontFormat
from utils.structured_ocr_export import build_structured_ocr_export


class Block:
    xyxy = [1, 2, 30, 40]
    lines = [[[1, 2], [30, 2]]]
    angle = 0
    translation = "Hello"
    label = "speech"
    confidence = 0.75
    fontformat = FontFormat(font_family="Arial", font_size=24, alignment=1)

    def get_text(self):
        return "こんにちは"


class BlockAt(Block):
    def __init__(self, x):
        self.xyxy = [x, 2, x + 30, 40]


class Project:
    directory = "/tmp/proj"
    current_img = "p1.png"
    pages = {"p1.png": [Block()]}
    _image_info = {"p1.png": {"width": 100, "height": 200, "ignored": False, "finish_code": 15, "completion_state": "reviewed"}}

    def get_page_completion_state(self, page):
        return self._image_info[page]["completion_state"]


def test_structured_ocr_export_contains_page_block_geometry_and_font():
    out = build_structured_ocr_export(Project())
    assert out["schema"] == "ballonstranslator.structured_ocr.v1"
    page = out["pages"][0]
    assert page["completion_state"] == "reviewed"
    assert page["blocks"][0]["source_text"] == "こんにちは"
    assert page["blocks"][0]["font"]["alignment"] == 1
    assert page["blocks"][0]["xyxy"] == [1.0, 2.0, 30.0, 40.0]


def test_structured_ocr_export_can_sort_blocks_by_reading_order():
    class P(Project):
        pages = {"p1.png": [BlockAt(1), BlockAt(80)]}

    out = build_structured_ocr_export(P(), reading_order="rtl")
    blocks = out["pages"][0]["blocks"]
    assert out["pages"][0]["reading_order"] == "rtl"
    assert blocks[0]["source_index"] == 1
    assert blocks[1]["source_index"] == 0
    assert blocks[0]["font"]["writing_mode"] == "auto"


def test_structured_ocr_export_includes_vertical_render_hints():
    class VBlock(Block):
        translation = "第12話?!"
        fontformat = FontFormat(font_family="Arial", font_size=24, writing_mode="vertical_rl", line_break_strategy="cjk_strict")

    class P(Project):
        pages = {"p1.png": [VBlock()]}

    out = build_structured_ocr_export(P())
    hints = out["pages"][0]["blocks"][0]["render_hints"]
    assert hints["resolved_writing_mode"] == "vertical_rl"
    assert hints["vertical_columns"]
    assert any(group["text"] == "12" for group in hints["tate_chu_yoko_groups"])
