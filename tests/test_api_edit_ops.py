import pytest

from utils.api_edit_ops import (
    validate_textbox_operation,
    validate_batch_payload,
    EditValidationError,
    ensure_block_stable_id,
    find_block_index_by_stable_id,
)


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


class _Blk:
    pass


def test_validate_update_accepts_block_id():
    op = validate_textbox_operation({"op": "update_textbox", "block_id": "tbx_abc", "text": "hello"})
    assert op.op == "update_textbox"
    assert op.block_id == "tbx_abc"
    assert op.index is None


def test_stable_id_helpers_assign_and_lookup():
    a = _Blk()
    b = _Blk()
    bid_a = ensure_block_stable_id(a)
    bid_b = ensure_block_stable_id(b)
    assert bid_a != bid_b
    assert find_block_index_by_stable_id([a, b], bid_b) == 1


def test_find_block_index_missing_raises_not_found():
    with pytest.raises(EditValidationError) as e:
        find_block_index_by_stable_id([_Blk()], "tbx_missing")
    assert e.value.code == "not_found"
