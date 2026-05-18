import pytest

from utils.api_edit_ops import validate_textbox_operation, validate_batch_payload, EditValidationError


def test_validate_update_textbox_ok():
    op = validate_textbox_operation({"op": "update_textbox", "index": 2, "text": "hello"})
    assert op.op == "update_textbox"
    assert op.index == 2
    assert op.text == "hello"


def test_validate_batch_requires_ops():
    with pytest.raises(EditValidationError):
        validate_batch_payload({})


def test_validate_rejects_unknown_operation():
    with pytest.raises(EditValidationError) as e:
        validate_textbox_operation({"op": "warp_everything"})
    assert e.value.code == "unsupported_operation"


def test_validate_delete_requires_index():
    with pytest.raises(EditValidationError) as e:
        validate_textbox_operation({"op": "delete_textbox"})
    assert e.value.code == "missing_field"
