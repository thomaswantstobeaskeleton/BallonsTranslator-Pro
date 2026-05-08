from pathlib import Path

from utils.export_manifest import mark_exported_pages, write_export_manifest


class Project:
    directory = "/tmp/project"
    proj_path = "/tmp/project/project.json"

    def __init__(self):
        self.states = {"p1.png": "reviewed"}

    def get_page_completion_state(self, page):
        return self.states.get(page, "todo")

    def set_page_completion_state(self, page, state):
        self.states[page] = state


def test_write_export_manifest_records_paths_and_missing_pages(tmp_path):
    out = tmp_path / "out"
    page = out / "001.png"
    out.mkdir()
    page.write_bytes(b"x")
    project = Project()
    manifest = write_export_manifest(project, str(out), [("p1.png", str(page))], ["p2.png"], options={"ext": ".png"})
    assert Path(manifest["manifest_path"]).exists()
    assert manifest["page_count"] == 1
    assert manifest["missing_count"] == 1
    assert manifest["pages"][0]["completion_state"] == "reviewed"
    assert manifest["warnings"]


def test_mark_exported_pages_updates_completion_state(tmp_path):
    out = tmp_path / "out"
    page = out / "001.png"
    out.mkdir()
    page.write_bytes(b"x")
    project = Project()
    assert mark_exported_pages(project, [("p1.png", str(page)), ("missing.png", str(out / "missing.png"))]) == 1
    assert project.get_page_completion_state("p1.png") == "exported"
    assert project.get_page_completion_state("missing.png") == "todo"


def test_export_manifest_records_unrendered_fallback_sources(tmp_path):
    out = tmp_path / "out"
    page = out / "002.png"
    out.mkdir()
    page.write_bytes(b"x")
    project = Project()
    manifest = write_export_manifest(
        project,
        str(out),
        [("p2.png", str(page))],
        [],
        options={"include_unrendered": True, "export_sources": {"p2.png": "original_fallback"}},
    )
    assert manifest["pages"][0]["source_kind"] == "original_fallback"
    assert manifest["pages"][0]["used_fallback_source"] is True
    assert "fallback" in manifest["warnings"][0]
