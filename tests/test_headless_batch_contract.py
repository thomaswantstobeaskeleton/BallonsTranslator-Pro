from utils.headless_batch_contract import parse_stage_set


def test_parse_stage_set_defaults_all():
    assert parse_stage_set("") == {"detect", "ocr", "translate", "inpaint"}


def test_parse_stage_set_rejects_invalid_entries():
    try:
        parse_stage_set("detect,foo")
        assert False, "expected ValueError"
    except ValueError as e:
        assert "invalid stage" in str(e)
