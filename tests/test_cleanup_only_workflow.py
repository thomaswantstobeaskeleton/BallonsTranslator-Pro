import numpy as np
from pathlib import Path

from utils.cleanup_only_workflow import run_cleanup_only_pages


class B:
    def __init__(self):
        self.xyxy = [1, 1, 4, 4]


class D:
    def detect(self, img, proj):
        return None, [B()]


class I:
    def inpaint(self, img, mask):
        out = img.copy()
        out[1:4, 1:4] = 255
        return out


class P:
    def __init__(self, tmp):
        self.tmp = Path(tmp)
    def read_img(self, page):
        return np.zeros((8, 8, 3), dtype=np.uint8)
    def save_inpainted(self, page, img):
        return None


def test_cleanup_only_exports_clean_images(tmp_path):
    proj = P(tmp_path)
    rst = run_cleanup_only_pages(proj, D(), I(), ['001.png'], out_dir=str(tmp_path / 'out'), inside_radius=1, outside_radius=2, detector_confidence=0.9)
    assert rst['processed'] == 1
    assert rst['failed'] == 0
    assert (tmp_path / 'out' / '001_clean.png').exists()


def test_cleanup_only_returns_mask_policy(tmp_path):
    proj = P(tmp_path)
    rst = run_cleanup_only_pages(proj, D(), I(), ['001.png'], out_dir=str(tmp_path / 'out2'))
    assert rst['mask_policy']['outside_radius'] >= rst['mask_policy']['inside_radius']
    assert isinstance(rst['halo_stats'], list)
    assert rst['halo_stats'][0]['page'] == '001.png'
