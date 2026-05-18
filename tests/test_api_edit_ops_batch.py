from utils.api_edit_ops import validate_batch_payload


def test_validate_batch_multiple_ops_parses_all():
    ops = validate_batch_payload({"ops": [
        {"op": "update_textbox", "index": 0, "text": "A"},
        {"op": "undo"},
        {"op": "redo"},
    ]})
    assert len(ops) == 3
    assert ops[0].op == "update_textbox"
    assert ops[1].op == "undo"
    assert ops[2].op == "redo"
