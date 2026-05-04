import numpy as np

from modules.mask_diagnostics import build_mask_diagnostics


def test_mask_diagnostics_emits_edge_halo_and_stats():
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 200
    out = build_mask_diagnostics(mask, threshold=100, dilate_iter=1)
    assert out is not None
    assert out["edge_halo"].shape == mask.shape
    stats = out["stats"]
    assert 0 <= stats["otsu_threshold"] <= 255
    assert 0.0 <= stats["mask_fill_ratio"] <= 1.0
    assert 0.0 <= stats["edge_halo_ratio"] <= 1.0
