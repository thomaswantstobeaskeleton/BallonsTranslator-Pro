from types import SimpleNamespace

from utils.fontformat import FontFormat
from utils.lettering_proof_export import build_lettering_proof_pack


def test_lettering_proof_pack_writes_manifest_and_qa(tmp_path):
    page = tmp_path / "page.png"
    page.write_bytes(b"not-real-image-but-copyable")
    blk = SimpleNamespace(
        xyxy=[0, 0, 40, 120],
        text=["こんにちは?!"],
        translation="こんにちは?!",
        fontformat=FontFormat(font_size=18, writing_mode="auto"),
        get_text=lambda: "こんにちは?!",
    )
    project = SimpleNamespace(
        directory=str(tmp_path),
        pages={"page.png": [blk]},
        _image_info={"page.png": {"width": 100, "height": 160}},
        get_result_path=lambda name: str(tmp_path / "missing_result.png"),
        get_inpainted_path=lambda name: str(tmp_path / "missing_clean.png"),
        get_mask_path=lambda name: str(tmp_path / "missing_mask.png"),
    )
    manifest = build_lettering_proof_pack(project, "page.png", str(tmp_path / "proof"), config_obj=SimpleNamespace())
    assert manifest["format"].endswith("v1")
    assert manifest["svg_path"].endswith(".svg")
    assert manifest["typography_qa_json"].endswith("typography_qa.json")
    assert manifest["html_index"].endswith("lettering_proof_index.html")
    assert (tmp_path / "proof" / "page_lettering_proof" / "lettering_proof_index.html").exists()
    assert manifest["archive_path"].endswith("_lettering_proof.zip")
    assert (tmp_path / "proof" / "page_lettering_proof.zip").exists()
    assert "Portable ZIP archive" in (tmp_path / "proof" / "page_lettering_proof" / "lettering_proof_index.html").read_text(encoding="utf-8")
    assert "polish_typography" in manifest["next_actions"]
