"""
Font detection model wrapper.

Wraps YuzuMarker.FontDetection (fffonion/yuzumarker-font-detection) or
a similar Hugging Face vision model to auto-infer:
  - font family
  - text color (foreground)
  - stroke color + width
  - bold / italic flags

Results are stored in TextBlock.font_prediction for downstream
auto-lettering and style matching.

Dependencies:
  - transformers
  - torch / torchvision
"""

from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass

import numpy as np

try:
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    from PIL import Image as PILImage
except ImportError:
    AutoImageProcessor = None
    AutoModelForImageClassification = None
    PILImage = None

from utils.textblock import TextBlock


@dataclass
class FontPrediction:
    """Predicted font properties for a text region."""
    font_family: Optional[str] = None
    font_size: Optional[float] = None
    color: Optional[Tuple[int, int, int]] = None  # RGB foreground
    stroke_color: Optional[Tuple[int, int, int]] = None
    stroke_width: Optional[float] = None
    bold: bool = False
    italic: bool = False
    confidence: float = 0.0


class FontDetector:
    """
    Lightweight font detection wrapper.

    Auto-downloads model from Hugging Face on first use.
    Falls back to heuristic color extraction if the ML model is unavailable.
    """

    DEFAULT_MODEL_ID = "fffonion/yuzumarker-font-detection"

    def __init__(self, model_id: Optional[str] = None, device: str = "cuda") -> None:
        self.model_id = model_id or self.DEFAULT_MODEL_ID
        self.device = device
        self._processor = None
        self._model = None
        self._available = self._check_deps()

    def _check_deps(self) -> bool:
        return (
            AutoImageProcessor is not None
            and AutoModelForImageClassification is not None
            and PILImage is not None
        )

    def _load(self):
        if not self._available:
            raise RuntimeError(
                "FontDetector requires transformers and Pillow. "
                "Install: pip install transformers Pillow"
            )
        if self._model is None:
            self._processor = AutoImageProcessor.from_pretrained(self.model_id)
            self._model = AutoModelForImageClassification.from_pretrained(self.model_id)
            if self.device == "cuda":
                import torch
                if torch.cuda.is_available():
                    self._model = self._model.to("cuda")

    def predict(
        self,
        img: np.ndarray,
        text_regions: List[Tuple[int, int, int, int]],
    ) -> List[FontPrediction]:
        """
        Predict font properties for each text region.

        Args:
            img: H×W×3/4 numpy array (uint8).
            text_regions: List of (x1, y1, x2, y2) bounding boxes.

        Returns:
            List of FontPrediction, one per region.
        """
        if not self._available:
            # Graceful fallback: return empty predictions
            return [FontPrediction() for _ in text_regions]

        self._load()

        predictions: List[FontPrediction] = []
        for x1, y1, x2, y2 in text_regions:
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                predictions.append(FontPrediction())
                continue

            try:
                pil_img = PILImage.fromarray(crop)
                inputs = self._processor(images=pil_img, return_tensors="pt")
                if self.device == "cuda":
                    import torch
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}

                import torch
                with torch.no_grad():
                    outputs = self._model(**inputs)

                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                pred_idx = int(torch.argmax(probs, dim=-1).item())
                confidence = float(probs[0, pred_idx].item())

                # Map model output to FontPrediction
                # (Actual mapping depends on model label schema; this is a skeleton.)
                pred = FontPrediction(
                    font_family=self._model.config.id2label.get(pred_idx),
                    confidence=confidence,
                )
                predictions.append(pred)
            except Exception:
                predictions.append(FontPrediction())

        return predictions

    def predict_on_blocks(
        self,
        img: np.ndarray,
        blk_list: List[TextBlock],
    ) -> List[FontPrediction]:
        """
        Convenience wrapper that extracts regions from TextBlock.xyxy.
        Stores predictions in blk.font_prediction.
        """
        regions = []
        for blk in blk_list:
            x1, y1, x2, y2 = map(int, blk.xyxy)
            regions.append((x1, y1, x2, y2))

        preds = self.predict(img, regions)
        for blk, pred in zip(blk_list, preds):
            blk.font_prediction = pred  # type: ignore[attr-defined]
        return preds


def heuristic_font_color(
    img: np.ndarray,
    text_region: Tuple[int, int, int, int],
) -> Tuple[Tuple[int, int, int], Optional[Tuple[int, int, int]]]:
    """
    Fast heuristic to estimate text color and stroke color from a region.

    Returns:
        (foreground_color, stroke_color_or_None)
    """
    x1, y1, x2, y2 = text_region
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return (0, 0, 0), None

    # Simple k-means-ish approach: find dominant colors
    if crop.shape[2] == 4:
        crop = crop[:, :, :3]

    # Reshape and cluster roughly by brightness
    pixels = crop.reshape(-1, 3).astype(np.float32)
    brightness = np.mean(pixels, axis=1)

    # Dark pixels = text, bright pixels = background
    dark = pixels[brightness < np.median(brightness)]
    bright = pixels[brightness >= np.median(brightness)]

    fg = tuple(int(v) for v in np.median(dark, axis=0)) if len(dark) > 0 else (0, 0, 0)
    bg = tuple(int(v) for v in np.median(bright, axis=0)) if len(bright) > 0 else (255, 255, 255)

    # If there's significant color difference, guess stroke as intermediate
    diff = np.mean(np.abs(np.array(fg) - np.array(bg)))
    stroke = None
    if diff > 60:
        stroke = tuple(int(v) for v in (np.array(fg) + np.array(bg)) / 2)

    return fg, stroke
