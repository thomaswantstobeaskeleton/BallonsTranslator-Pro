import numpy as np
from utils.image_transform_ops import TransformOptions, transform_image_array


def test_transform_crop_resize_border_pipeline():
    img = np.zeros((20, 30, 3), dtype=np.uint8)
    img[:, :] = [10, 20, 30]
    opts = TransformOptions(crop=(5, 5, 10, 8), resize=(40, 16), border=2, border_color=(255, 0, 0))
    out = transform_image_array(img, opts)
    assert out.shape[1] == 44
    assert out.shape[0] == 20
    assert (out[0, 0][:3] == np.array([255, 0, 0])).all()


def test_transform_shrink_scale():
    img = np.ones((100, 50, 3), dtype=np.uint8) * 127
    out = transform_image_array(img, TransformOptions(scale=0.5))
    assert out.shape[0] == 50
    assert out.shape[1] == 25


def test_transform_brightness_contrast():
    img = np.ones((4, 4, 3), dtype=np.uint8) * 50
    out = transform_image_array(img, TransformOptions(brightness=20, contrast=2.0))
    assert int(out.mean()) > int(img.mean())


def test_transform_perspective_identity_like():
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    img[4:12, 4:12] = 255
    src = ((0, 0), (15, 0), (15, 15), (0, 15))
    dst = ((0, 0), (15, 0), (15, 15), (0, 15))
    out = transform_image_array(img, TransformOptions(perspective_src=src, perspective_dst=dst))
    assert out.shape == img.shape
