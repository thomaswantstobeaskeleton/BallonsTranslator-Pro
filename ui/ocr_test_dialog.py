"""Dialog to test the current OCR module on a small sample image."""
import time
import os
import textwrap

import numpy as np
import cv2
from qtpy.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QFrame,
    QSizePolicy,
)
from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QPixmap, QImage

from utils.logger import logger as LOGGER


class OCRTestDialog(QDialog):
    """Dialog to test OCR on a generated sample image and show result/time."""

    def __init__(self, ocr_module, parent=None):
        super().__init__(parent)
        self.ocr_module = ocr_module
        self.setWindowTitle(self.tr("OCR Test"))
        self.setFixedSize(300, 200)

        self.test_image = self._create_test_image()
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(5)
        layout.setContentsMargins(5, 5, 5, 5)

        image_frame = QFrame()
        image_frame.setFrameShape(QFrame.Shape.StyledPanel)
        image_frame.setStyleSheet("background-color: #FFC266;")
        image_layout = QVBoxLayout(image_frame)
        image_layout.setContentsMargins(3, 3, 3, 3)

        h, w, c = self.test_image.shape
        q_img = QImage(self.test_image.data, w, h, w * c, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.image_label = QLabel()
        self.image_label.setPixmap(pixmap)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("padding: 0px; margin: 0px;")
        image_layout.addWidget(self.image_label)
        layout.addWidget(image_frame)

        result_frame = QFrame()
        result_frame.setFrameShape(QFrame.Shape.StyledPanel)
        result_frame.setStyleSheet("background-color: #99CCFF;")
        result_layout = QVBoxLayout(result_frame)
        result_layout.setContentsMargins(3, 3, 3, 3)
        result_layout.setSpacing(2)

        self.result_label = QLabel(self.tr("Result:"))
        self.result_label.setStyleSheet("color: #0066CC; font-weight: bold; font-size: 12px;")
        result_layout.addWidget(self.result_label)

        self.result_text = QLabel()
        self.result_text.setWordWrap(True)
        self.result_text.setStyleSheet("color: #0066CC; font-size: 12px;")
        self.result_text.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        result_layout.addWidget(self.result_text)

        status_layout = QHBoxLayout()
        status_layout.setContentsMargins(2, 1, 2, 1)
        self.test_status_label = QLabel()
        self.test_status_label.setStyleSheet(
            "color: #008080; font-size: 10px; font-weight: bold; text-decoration: underline;"
        )
        status_layout.addWidget(self.test_status_label)
        status_layout.addStretch(1)
        self.test_time_label = QLabel()
        self.test_time_label.setStyleSheet("color: #008080; font-size: 10px; text-decoration: underline;")
        status_layout.addWidget(self.test_time_label)
        result_layout.addLayout(status_layout)
        layout.addWidget(result_frame)

        QTimer.singleShot(100, self._run_test)

    def _create_test_image(self):
        """Build a small test image with Japanese-style text for OCR."""
        img = np.ones((100, 280, 3), dtype=np.uint8) * 255
        img[:, :, 0] = 153
        img[:, :, 1] = 204
        img[:, :, 2] = 255

        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            cv2.putText(img, "OCR Test Image", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            return img

        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        font_size = 8
        possible_fonts = [
            "C:\\Windows\\Fonts\\msgothic.ttc",
            "C:\\Windows\\Fonts\\YuGothR.ttc",
            "C:\\Windows\\Fonts\\meiryo.ttc",
            "C:\\Windows\\Fonts\\msmincho.ttc",
            "fonts/NotoSansJP-Regular.otf",
        ]
        font_path = None
        for path in possible_fonts:
            if os.path.exists(path):
                font_path = path
                break

        if font_path:
            try:
                font = ImageFont.truetype(font_path, font_size)
                text = (
                    "校門の外から携てた様子でやってきた二人組の男たちに声をかけられるとちらも中年くらいで、"
                    "一人はバカでかいビデオカメラのような機材を担いでいた。"
                )
                y_position = 15
                for line in textwrap.wrap(text, width=30):
                    draw.text((10, y_position), line, font=font, fill=(0, 0, 0))
                    y_position += font_size + 5
            except Exception as e:
                LOGGER.warning("PIL font draw failed: %s", e)
                draw.text((10, 15), "OCR Test Image", fill=(0, 0, 0))
        else:
            draw.text((10, 15), "OCR Test Image", fill=(0, 0, 0))
            draw.text((10, 40), "Japanese text test", fill=(0, 0, 0))

        return np.array(pil_img)

    def _run_test(self):
        self.result_text.setText(self.tr("Testing..."))
        self.test_status_label.setText("")
        self.test_time_label.setText("")

        timer = QTimer(self)
        timer.setSingleShot(True)
        timer.timeout.connect(lambda: self._do_test(timer))
        timer.start(100)

    def _do_test(self, timer):
        try:
            if not self.ocr_module:
                self.result_text.setText(self.tr("Error: OCR module not initialized."))
                self._set_status_fail()
                return

            start_time = time.time()
            try:
                result = self.ocr_module.ocr_img(self.test_image)
                if not isinstance(result, str):
                    result = str(result) if result is not None else ""
                elapsed = time.time() - start_time
                self.test_time_label.setText(self.tr("Time: %1 s").arg(f"{elapsed:.2f}"))

                if result and result.strip():
                    self.result_text.setText(result.strip())
                    self.test_status_label.setText(self.tr("Test passed!"))
                    self.test_status_label.setStyleSheet(
                        "color: #008080; font-size: 12px; font-weight: bold; text-decoration: underline;"
                    )
                else:
                    self.result_text.setText(self.tr("OCR returned no text."))
                    self._set_status_fail()
            except Exception as e:
                elapsed = time.time() - start_time
                self.test_time_label.setText(self.tr("Time: %1 s").arg(f"{elapsed:.2f}"))
                self.result_text.setText(self.tr("OCR failed: %1").arg(str(e)))
                self._set_status_fail()
                LOGGER.error("OCR test failed: %s", e)
        except Exception as e:
            self.result_text.setText(self.tr("Error: %1").arg(str(e)))
            self._set_status_fail()
            LOGGER.exception("OCR test error")
        finally:
            if timer and timer.isActive():
                timer.stop()
                timer.deleteLater()

    def _set_status_fail(self):
        self.test_status_label.setText(self.tr("Test failed!"))
        self.test_status_label.setStyleSheet(
            "color: #c0392b; font-size: 12px; font-weight: bold; text-decoration: underline;"
        )
