"""
TextMamba – Scene text detector with Mamba (arXiv 2512.06657, IJCNN 2025).
SOTA on CTW1500 (89.7%), TotalText (89.2%), ICDAR19ArT (78.5%). Linear complexity vs Transformers.
Implemented as a stub: official code is not yet public; selecting this detector raises a clear error.
When the authors release the repo, wire _load_model and _detect to their API. See: https://arxiv.org/abs/2512.06657
"""
import numpy as np
from typing import Tuple, List

from .base import register_textdetectors, TextDetectorBase, TextBlock, ProjImgTrans
from ..base import DEVICE_SELECTOR

_TEXTMAMBA_NOT_AVAILABLE_MSG = (
    "TextMamba official code is not yet released. "
    "Use mmocr_det, surya_det, or craft_det for scene text meanwhile. Paper: https://arxiv.org/abs/2512.06657"
)


@register_textdetectors("textmamba_det")
class TextMambaDetector(TextDetectorBase):
    """
    TextMamba: scene text detector with Mamba SSM (curved text, CTW1500/TotalText).
    Stub: official code TBD; selecting this detector raises an error with alternatives.
    """
    params = {
        "device": DEVICE_SELECTOR(),
        "det_score_thresh": {
            "type": "line_editor",
            "value": 0.3,
            "description": "Min detection score (when official code is integrated).",
        },
        "description": "TextMamba – official code not yet released. Use mmocr_det or surya_det. See arXiv:2512.06657.",
    }
    _load_model_keys = set()

    def __init__(self, **params) -> None:
        super().__init__(**params)

    def _load_model(self):
        raise RuntimeError(_TEXTMAMBA_NOT_AVAILABLE_MSG)

    def _detect(self, img: np.ndarray, proj: ProjImgTrans = None) -> Tuple[np.ndarray, List[TextBlock]]:
        raise RuntimeError(_TEXTMAMBA_NOT_AVAILABLE_MSG)
