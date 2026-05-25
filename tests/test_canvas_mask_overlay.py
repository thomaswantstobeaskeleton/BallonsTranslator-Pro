"""Tests for Canvas._build_mask_overlay fix (#1117)."""
import sys
import os.path as osp
import numpy as np

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

from ui.canvas import Canvas


def test_build_mask_overlay_zero_mask():
    """A zero mask should produce a fully transparent RGBA image."""
    mask = np.zeros((10, 10), dtype=np.uint8)
    rgba = Canvas._build_mask_overlay(mask, 0.5)
    assert rgba.shape == (10, 10, 4)
    assert rgba.dtype == np.uint8
    # Alpha channel should be all zeros
    assert np.all(rgba[:, :, 3] == 0)


def test_build_mask_overlay_full_mask():
    """A fully white mask at transparency=1 should produce alpha=255."""
    mask = np.full((10, 10), 255, dtype=np.uint8)
    rgba = Canvas._build_mask_overlay(mask, 1.0)
    assert np.all(rgba[:, :, 3] == 255)
    # Check magenta tint
    assert np.all(rgba[:, :, 0] == 200)
    assert np.all(rgba[:, :, 1] == 0)
    assert np.all(rgba[:, :, 2] == 200)


def test_build_mask_overlay_partial_transparency():
    """Partial transparency should scale alpha proportionally."""
    mask = np.full((10, 10), 128, dtype=np.uint8)
    rgba = Canvas._build_mask_overlay(mask, 0.5)
    expected_alpha = int(128 * 0.5)
    assert np.all(rgba[:, :, 3] == expected_alpha)


def test_build_mask_overlay_gradient_mask():
    """A gradient mask should preserve per-pixel alpha values."""
    mask = np.arange(0, 256, dtype=np.uint8).reshape(16, 16)
    rgba = Canvas._build_mask_overlay(mask, 0.5)
    expected_alpha = np.clip(mask * 0.5, 0, 255).astype(np.uint8)
    np.testing.assert_array_equal(rgba[:, :, 3], expected_alpha)
