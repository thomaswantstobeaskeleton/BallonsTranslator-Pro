from utils.project_ops_protocol import ProjectOpSession, apply_ops, undo, redo


def test_project_ops_update_undo_redo():
    s = ProjectOpSession(pages={"p1": [{"translation": "A"}, {"translation": "B"}]})
    out = apply_ops(s, [{"op": "UpdateText", "page": "p1", "index": 1, "text": "BB"}])
    assert out["count"] == 1
    assert s.pages["p1"][1]["translation"] == "BB"
    assert undo(s) == 1
    assert s.pages["p1"][1]["translation"] == "B"
    assert redo(s) == 1
    assert s.pages["p1"][1]["translation"] == "BB"
