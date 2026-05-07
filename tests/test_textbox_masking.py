import numpy as np

from utils.textbox_masking import centered_resize_xyxy, mask_aware_textbox_diagnostics, visible_mask_rect


def test_visible_mask_rect_reports_erased_edges_and_scaled_visible_size():
    mask = np.zeros((10, 20), dtype=np.uint8)
    mask[2:8, 5:15] = 255
    diag = visible_mask_rect(mask, (200, 100))
    assert diag["has_mask"] is True
    assert diag["coverage"] == 0.3
    assert diag["edge_hidden"] is True
    assert diag["visible_rect"] == [50.0, 20.0, 150.0, 80.0]
    assert diag["visible_size"] == [100.0, 60.0]


def test_mask_aware_textbox_diagnostics_detects_visible_area_overflow():
    mask = np.zeros((10, 20), dtype=np.uint8)
    mask[2:8, 5:15] = 255
    diag = mask_aware_textbox_diagnostics((200, 100), (130, 40), mask, effect_margin=2, padding=1)
    assert diag["mask_overflow"] is True
    assert "x" in diag["mask_overflow_axes"]
    assert diag["visible_area_ratio"] < 1.0


def test_centered_resize_xyxy_keeps_center_and_only_grows():
    out = centered_resize_xyxy([10, 20, 30, 40], [40, 10])
    assert out == [0.0, 20.0, 40.0, 40.0]
