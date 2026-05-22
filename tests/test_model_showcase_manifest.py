import json
from pathlib import Path


def test_manifest_contains_showcase_package_and_models():
    manifest = json.loads(Path('data/model_manifest.json').read_text(encoding='utf-8'))
    packages = {p.get('id'): p for p in manifest.get('packages', [])}
    assert 'community_showcase' in packages
    module_keys = {m.get('module_key') for m in manifest.get('modules', [])}
    expected = {
        'ysg_comic_text_segmenter_v8m',
        'ysg_comic_speech_bubble_v8m',
        'mangalens_bubble_segmentation',
        'sam2_1_hiera_large',
        'sam3_hiera_large',
        'flux1_kontext_inpaint',
        'flux2_klein_inpaint',
        'paddleocr_vl_1_5',
        'realcugan_upscaler',
        'anime_sharp_v4_x2',
    }
    assert expected.issubset(module_keys)
