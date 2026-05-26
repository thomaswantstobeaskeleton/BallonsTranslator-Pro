"""
Bubble + Text Joint Detector (RT-DETR-v2).

Wraps the Comic Translate joint detector model:
  ogkalu/comic-text-and-bubble-detector

Outputs TextBlock objects enriched with bubble_type:
  - "speech"   : Standard speech bubbles
  - "thought"  : Thought bubbles (cloud/dashed)
  - "caption"  : Rectangular caption/narration boxes
  - "sfx"      : Sound effect text (often outside bubbles)
  - "none"     : Free-floating text without a detected bubble

The model is trained on 11k images across manga, webtoon, and western comics.
"""

from typing import List, Optional, Tuple, Dict
import numpy as np

try:
    from transformers import AutoImageProcessor, AutoModelForObjectDetection
except ImportError:
    AutoImageProcessor = None
    AutoModelForObjectDetection = None

from utils.textblock import TextBlock
from .base import TextDetectorBase, register_textdetectors


HF_MODEL_ID = "ogkalu/comic-text-and-bubble-detector"

# COCO-style label mapping for the joint detector
LABEL_MAP = {
    0: "text",
    1: "bubble",
    2: "speech",
    3: "thought",
    4: "caption",
    5: "sfx",
}

# Bubble-type classification: which bubble label encompasses which text
BUBBLE_TYPE_MAP = {
    "speech": "speech",
    "thought": "thought",
    "caption": "caption",
    "sfx": "sfx",
    "bubble": "speech",  # generic bubble defaults to speech
}


def _compute_iou(box_a: List[float], box_b: List[float]) -> float:
    """Compute IoU between two XYXY boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def _assign_bubble_types(
    text_boxes: List[List[float]],
    bubble_boxes: List[Tuple[List[float], str]],
    iou_threshold: float = 0.1,
) -> List[str]:
    """
    Assign a bubble type to each text box based on overlapping bubble detections.

    Args:
        text_boxes: List of XYXY text bounding boxes.
        bubble_boxes: List of (XYXY box, label) tuples for bubble detections.
        iou_threshold: Minimum IoU to consider a text box inside a bubble.

    Returns:
        List of bubble type strings, one per text box.
    """
    types: List[str] = []
    for txt in text_boxes:
        best_type = "none"
        best_iou = 0.0
        for bub, label in bubble_boxes:
            iou = _compute_iou(txt, bub)
            if iou > best_iou:
                best_iou = iou
                best_type = BUBBLE_TYPE_MAP.get(label, "none")
        if best_iou < iou_threshold:
            best_type = "none"
        types.append(best_type)
    return types


@register_textdetectors("bubble_text_joint")
class BubbleTextJointDetector(TextDetectorBase):
    """
    Joint bubble + text detector using RT-DETR-v2.

    This detector simultaneously finds text regions and their enclosing bubbles,
    producing TextBlock objects annotated with bubble_type for downstream
    translation context and mask generation.
    """

    params = {
        "model_id": {
            "type": "line_editor",
            "value": HF_MODEL_ID,
            "description": "Hugging Face model ID for the joint detector.",
        },
        "score_threshold": {
            "type": "selector",
            "options": [0.1, 0.2, 0.3, 0.4, 0.5],
            "value": 0.3,
            "description": "Confidence threshold for detections.",
        },
        "bubble_iou_threshold": {
            "type": "selector",
            "options": [0.05, 0.1, 0.15, 0.2],
            "value": 0.1,
            "description": "Min IoU for assigning a text block to a bubble.",
        },
        "device": {
            "type": "selector",
            "options": ["cpu", "cuda"],
            "value": "cuda",
            "description": "Inference device.",
        },
        "description": "RT-DETR-v2 joint bubble+text detector (trained on 11k images).",
    }
    _load_model_keys = {"processor", "model"}

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.processor = None
        self.model = None

    def _load_model(self):
        if AutoImageProcessor is None or AutoModelForObjectDetection is None:
            raise RuntimeError(
                "transformers is required for the bubble+text joint detector. "
                "Install: pip install transformers"
            )

        model_id = self.get_param_value("model_id")
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForObjectDetection.from_pretrained(model_id)

        device = self.get_param_value("device")
        if device == "cuda" and hasattr(self.model, "to"):
            import torch
            if torch.cuda.is_available():
                self.model = self.model.to("cuda")

    def detect(self, img: np.ndarray) -> Tuple[List[TextBlock], Optional[np.ndarray]]:
        """
        Detect text blocks and bubbles, returning enriched TextBlock list + mask.

        Args:
            img: H×W×3 BGR or RGB numpy array.

        Returns:
            Tuple of (text_block_list, mask_array).
        """
        if self.processor is None or self.model is None:
            self.load_model()

        import torch

        # Ensure RGB
        if img.shape[2] == 4:
            img = img[:, :, :3]
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)

        inputs = self.processor(images=img, return_tensors="pt")
        device = self.get_param_value("device")
        if device == "cuda" and torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([img.shape[:2]])
        results = self.processor.post_process_object_detection(
            outputs,
            threshold=self.get_param_value("score_threshold"),
            target_sizes=target_sizes,
        )[0]

        scores = results["scores"].cpu().numpy()
        labels = results["labels"].cpu().numpy()
        boxes = results["boxes"].cpu().numpy()  # XYXY format

        # Separate text and bubble detections
        text_entries: List[Tuple[List[float], float]] = []
        bubble_entries: List[Tuple[List[float], str, float]] = []

        for box, label, score in zip(boxes, labels, scores):
            box_list = [float(v) for v in box]
            label_name = LABEL_MAP.get(int(label), "unknown")
            if label_name == "text":
                text_entries.append((box_list, float(score)))
            elif label_name in ("bubble", "speech", "thought", "caption"):
                bubble_entries.append((box_list, label_name, float(score)))

        # Assign bubble types to text boxes
        iou_thr = self.get_param_value("bubble_iou_threshold")
        bubble_boxes = [(b[0], b[1]) for b in bubble_entries]
        assigned_types = _assign_bubble_types(
            [t[0] for t in text_entries], bubble_boxes, iou_threshold=iou_thr
        )

        # Build TextBlock list
        text_blocks: List[TextBlock] = []
        for (xyxy, score), bubble_type in zip(text_entries, assigned_types):
            blk = TextBlock(
                xyxy=xyxy,
                score=float(score),
            )
            # Store bubble type in a custom attribute for downstream use
            blk.bubble_type = bubble_type  # type: ignore[attr-defined]
            text_blocks.append(blk)

        # Build mask from bubble regions (for inpainting)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        for (xyxy, _, _) in bubble_entries:
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = 255

        return text_blocks, mask

    def __call__(self, img: np.ndarray) -> Tuple[List[TextBlock], Optional[np.ndarray]]:
        """Callable interface for pipeline integration."""
        return self.detect(img)
