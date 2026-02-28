# Stub for Ocean-OCR processor_ocean when decord is not installed.
# Only used for video; BallonsTranslator Ocean OCR uses local image paths only.

import numpy as np


def cpu(device_id=0):
    """Dummy context for CPU video decoding."""
    return type("Context", (), {})()


class VideoReader:
    """Dummy VideoReader; real use requires pip install decord."""

    def __init__(self, path, ctx=None, **kwargs):
        self._path = path
        self._len = 0

    def __len__(self):
        return self._len

    def get_avg_fps(self):
        return 1.0

    def get_batch(self, indices):
        class Batch:
            def asnumpy(self):
                return np.array([])

        return Batch()
