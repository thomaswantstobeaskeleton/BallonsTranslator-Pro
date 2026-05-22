from __future__ import annotations

from .base import register_textdetectors
from .detector_ysg import YSGYoloDetector


@register_textdetectors('mangalens_bubble_segmentation')
class MangaLensBubbleSegmentationDetector(YSGYoloDetector):
    """MangaLens bubble segmentation detector profile.

    Compatibility-first wrapper over ysgyolo backend with defaults tuned for
    speech-bubble segmentation style workflows.
    """

    params = dict(YSGYoloDetector.params)
    params['model path'] = dict(YSGYoloDetector.params['model path'])
    params['model path']['value'] = 'data/models/mangalens_bubble_segmentation.pt'

    params['label'] = dict(YSGYoloDetector.params['label'])
    params['label']['value'] = {
        'balloon': True,
        'bubble': True,
        'speech_bubble': True,
        'text_bubble': True,
        'other': False,
    }

    params['merge text lines'] = dict(YSGYoloDetector.params['merge text lines'])
    params['merge text lines']['value'] = False

    params['confidence threshold'] = dict(YSGYoloDetector.params['confidence threshold'])
    params['confidence threshold']['value'] = 0.25

    params['IoU threshold'] = dict(YSGYoloDetector.params['IoU threshold'])
    params['IoU threshold']['value'] = 0.5
