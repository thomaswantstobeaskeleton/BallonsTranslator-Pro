from utils.translated_image_alignment import align_translations_by_iou


class B:
    def __init__(self, xyxy, txt='', tr=''):
        self.xyxy = xyxy
        self._txt = txt
        self.translation = tr
    def get_text(self):
        return self._txt


def test_align_translations_by_iou_applies_best_match():
    raw = [B([0,0,10,10], tr=''), B([20,20,30,30], tr='x')]
    trn = [B([1,1,9,9], txt='hello'), B([21,21,29,29], txt='world')]
    out = align_translations_by_iou(raw, trn, min_iou=0.2)
    assert out['matched'] == 2
    assert raw[0].translation == 'hello'
    assert raw[1].translation == 'world'
