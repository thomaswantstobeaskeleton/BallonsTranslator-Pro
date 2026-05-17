from types import SimpleNamespace

from utils.fontformat import FontFormat
from utils.layered_psd_export import build_layered_psd_handoff


class Project:
    def __init__(self, directory, block):
        self.directory = str(directory)
        self.pages = {"p.png": [block]}

    def get_inpainted_path(self, page_name):
        return self.directory + "/missing_clean.png"

    def get_mask_path(self, page_name):
        return self.directory + "/missing_mask.png"

    def get_result_path(self, page_name):
        return self.directory + "/missing_result.png"


def test_layered_psd_handoff_preserves_secondary_outline_metadata(tmp_path):
    (tmp_path / "p.png").write_bytes(b"not-really-an-image-but-copyable")
    fmt = FontFormat(font_size=24, stroke_width=0.08, secondary_stroke_width=0.22)
    fmt.secondary_srgb = [255, 240, 120]
    block = SimpleNamespace(xyxy=[0, 0, 120, 60], translation="BOOM!!", text=[], fontformat=fmt, text_mask=None)
    manifest = build_layered_psd_handoff(Project(tmp_path, block), "p.png", str(tmp_path / "out"))
    layer = manifest["text_layers"][0]
    assert layer["secondary_stroke_width"] == 0.22
    assert layer["secondary_stroke_rgb"] == [255, 240, 120]
    assert layer["proof_metrics"]["effect_margin"] > 0
    assert layer["font_runs"]
    assert "fallback_runs" in layer
    assert "secondary_stroke_width" in (tmp_path / "out" / "p_rebuild_psd.jsx").read_text()
