# https://learn.microsoft.com/en-us/windows/powertoys/text-extractor#how-to-query-for-ocr-language-packs
from __future__ import annotations

import logging
import platform
from typing import TYPE_CHECKING

try:
    from .base import LOGGER
except Exception:  # pragma: no cover - fallback for very early optional-module import failures
    LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np
    from typing import List
    from .base import TextBlock


def _is_supported_windows_version() -> bool:
    if platform.system() != 'Windows':
        return False
    try:
        parts = tuple(int(part) for part in platform.version().split('.')[:3])
    except Exception:
        return False
    return parts >= (10, 0, 10240)


if _is_supported_windows_version():

    try:
        from winsdk.windows.media.ocr import OcrEngine
        from winsdk.windows.globalization import Language
        from winsdk.windows.storage.streams import DataWriter
        from winsdk.windows.graphics.imaging import SoftwareBitmap, BitmapPixelFormat
        import asyncio
        import cv2
        import numpy as np
        from typing import List

        from .base import register_OCR, OCRBase, TextBlock

        def get_supported_language_packs():
            return list(OcrEngine.available_recognizer_languages)

        def ocr(byte, width, height, lang='en'):
            writer = DataWriter()
            writer.write_bytes(byte)
            sb = SoftwareBitmap.create_copy_from_buffer(writer.detach_buffer(), BitmapPixelFormat.RGBA8, width, height)
            engine = OcrEngine.try_create_from_language(Language(lang))
            if engine is None:
                raise RuntimeError(f"Windows OCR engine is unavailable for language: {lang}")
            return engine.recognize_async(sb)

        async def coroutine(awaitable):
            return await awaitable

        winocr_available_recognizer_languages = get_supported_language_packs()

        if len(winocr_available_recognizer_languages) > 0:
            class WindowsOCR:
                lang = winocr_available_recognizer_languages[0].language_tag

                def __call__(self, img: np.ndarray) -> str:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
                    w, h = img.shape[1], img.shape[0]
                    return asyncio.run(coroutine(ocr(img.tobytes(), w, h, self.lang))).text

            languages_display_name = [lang.display_name for lang in winocr_available_recognizer_languages]
            languages_tag = [lang.language_tag for lang in winocr_available_recognizer_languages]

            @register_OCR('windows_ocr')
            class OCRWindows(OCRBase):
                params = {
                    'language': {
                        'type': 'selector',
                        'options': languages_display_name,
                        'value': languages_display_name[0],
                    }
                }
                language = languages_display_name[0]

                def __init__(self, **params) -> None:
                    super().__init__(**params)
                    self.engine = WindowsOCR()
                    self.engine.lang = self.get_engine_lang()

                def get_engine_lang(self) -> str:
                    language = self.params['language']['value']
                    tag_name = languages_tag[languages_display_name.index(language)]
                    return tag_name

                def ocr_img(self, img: np.ndarray) -> str:
                    return self.engine(img)

                def _ocr_blk_list(self, img: np.ndarray, blk_list: List[TextBlock], *args, **kwargs) -> None:
                    im_h, im_w = img.shape[:2]
                    for blk in blk_list:
                        x1, y1, x2, y2 = blk.xyxy
                        x1 = max(0, min(int(round(float(x1))), im_w - 1))
                        y1 = max(0, min(int(round(float(y1))), im_h - 1))
                        x2 = max(x1 + 1, min(int(round(float(x2))), im_w))
                        y2 = max(y1 + 1, min(int(round(float(y2))), im_h))
                        if y2 <= im_h and x2 <= im_w and x1 < x2 and y1 < y2:
                            blk.text = [self.engine(img[y1:y2, x1:x2])]
                        else:
                            self.logger.warning('invalid textbbox to target img')
                            blk.text = ['']

                def updateParam(self, param_key: str, param_content):
                    super().updateParam(param_key, param_content)
                    self.engine.lang = self.get_engine_lang()

        else:
            LOGGER.warning('No supported language packs found for Windows; Windows OCR will be unavailable.')
    except Exception as e:
        LOGGER.warning('Failed to initialize Windows OCR; it will be unavailable.', exc_info=True)
