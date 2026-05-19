import numpy as np

from utils.mask_cleanup_quality import adaptive_mask_expand, merge_masks_with_confidence


def test_adaptive_mask_expand_increases_coverage():
    m = np.zeros((16, 16), dtype=np.uint8)
    m[6:10, 6:10] = 255
    out = adaptive_mask_expand(m, inside_radius=1, outside_radius=2)
    assert out.sum() >= m.sum()


def test_merge_masks_with_confidence_gates_secondary():
    a = np.zeros((8, 8), dtype=np.uint8)
    b = np.zeros((8, 8), dtype=np.uint8)
    a[2:4, 2:4] = 255
    b[4:6, 4:6] = 255
    low = merge_masks_with_confidence(a, b, confidence=0.1)
    high = merge_masks_with_confidence(a, b, confidence=0.9)
    assert high.sum() > low.sum()
