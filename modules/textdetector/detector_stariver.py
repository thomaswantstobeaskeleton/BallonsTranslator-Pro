import numpy as np
import cv2
from typing import Tuple, List
import requests
import base64

from .base import register_textdetectors, TextDetectorBase, TextBlock, ProjImgTrans
from utils.message import create_error_dialog, create_info_dialog

import json
import time
import os

@register_textdetectors('stariver_ocr')
class StariverDetector(TextDetectorBase):

    params = {
        'User': "Enter your username",
        'Password': "Enter your password. Stored in plain text; avoid on shared computers.",
        'expand_ratio': "0.01",
        "refine": {
            'type': 'checkbox',
            'value': True
        },
        "filtrate": {
            'type': 'checkbox',
            'value': True
        },
        "disable_skip_area": {
            'type': 'checkbox',
            'value': True
        },
        "detect_scale": "3",
        "merge_threshold": "2.0",
        "low_accuracy_mode": {
            'type': 'checkbox',
            'value': False
        },
        "force_expand": {
            'type': 'checkbox',
            'value': False
        },
        "font_size_offset": "0",
        "font_size_min(set to -1 to disable)": "-1",
        "font_size_max(set to -1 to disable)": "-1",
        "font_size_multiplier": "1.0",
        'update_token_btn': {
            'type': 'pushbtn',
            'value': '',
            'description': 'Clear stored token and request a new one',
            'display_name': 'Update token'
        },
        'description': 'Starriver Cloud (Tuanzi) OCR text detector'
    }

    @property
    def User(self):
        return self.params['User']

    @property
    def Password(self):
        return self.params['Password']

    @property
    def expand_ratio(self):
        return float(self.params['expand_ratio'])

    @property
    def refine(self):
        return self.params['refine']['value']

    @property
    def filtrate(self):
        return self.params['filtrate']['value']

    @property
    def disable_skip_area(self):
        return self.params['disable_skip_area']['value']

    @property
    def detect_scale(self):
        return int(self.params['detect_scale'])

    @property
    def merge_threshold(self):
        return float(self.params['merge_threshold'])

    @property
    def low_accuracy_mode(self):
        return self.params['low_accuracy_mode']['value']

    @property
    def force_expand(self):
        return self.params['force_expand']['value']

    @property
    def font_size_offset(self):
        return int(self.params['font_size_offset'])

    @property
    def font_size_min(self):
        return int(self.params['font_size_min(set to -1 to disable)'])

    @property
    def font_size_max(self):
        return int(self.params['font_size_max(set to -1 to disable)'])
    
    @property
    def font_size_multiplier(self):
        return float(self.params['font_size_multiplier'])

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.url = 'https://dl.ap-qz.starivercs.cn/v2/manga_trans/advanced/manga_ocr'
        self.debug = False
        self.token = ''
        self.token_obtained = False
        # 初始化时设置用户名和密码为空
        self.register_username = None
        self.register_password = None

    def get_token(self):
        response = requests.post('https://capiv1.ap-sh.starivercs.cn/OCR/Admin/Login', json={
            "User": self.User,
            "Password": self.Password
        }).json()
        if response.get('Status', -1) != "Success":
            error_msg = f'Starriver login failed. Error: {response.get("ErrorMsg", "")}'
            raise Exception(error_msg)
        token = response.get('Token', '')
        if token != '':
            self.logger.info(f'Starriver detector login successful, token prefix: {token[:10]}')

        return token

    def adjust_font_size(self, original_font_size):
        new_font_size = original_font_size + self.font_size_offset
        if self.font_size_min != -1:
            new_font_size = max(new_font_size, self.font_size_min)
        if self.font_size_max != -1:
            new_font_size = min(new_font_size, self.font_size_max)
        if self.font_size_multiplier != 1.0:
            new_font_size = int(new_font_size * self.font_size_multiplier)
        return new_font_size

    def _detect(self, img: np.ndarray, proj: ProjImgTrans = None) -> Tuple[np.ndarray, List[TextBlock]]:
        self.update_token_if_needed()  # Try to refresh token before sending request
        if not self.token or self.token == '':
            self.logger.error(
                'Starriver detector token is not set.')
            raise ValueError('Starriver detector token is not set.')
        if self.low_accuracy_mode:
            self.logger.info('Starriver detector is in low-accuracy mode.')
            short_side = 768
        else:
            short_side = 1536

        # Compute scale factor
        height, width = img.shape[:2]
        scale = short_side / min(height, width)

        # Compute new dimensions
        new_width = int(width * scale)
        new_height = int(height * scale)

        # Scale image
        if scale < 1:
            img_scaled = cv2.resize(
                img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        else:
            img_scaled = img

        # Log scale and dimensions
        self.logger.debug(f'Image scale: {scale}, size: {new_width}x{new_height}')

        # Encode image to base64
        img_encoded = cv2.imencode('.jpg', img_scaled)[1]
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')

        payload = {
            "token": self.token,
            "mask": True,
            "refine": self.refine,
            "filtrate": self.filtrate,
            "disable_skip_area": self.disable_skip_area,
            "detect_scale": self.detect_scale,
            "merge_threshold": self.merge_threshold,
            "low_accuracy_mode": self.low_accuracy_mode,
            "force_expand": self.force_expand,
            "image": img_base64
        }
        if self.debug:
            payload_log = {k: v for k, v in payload.items() if k != 'image'}
            self.logger.debug(f'Starriver detector request params: {payload_log}')
            self.save_debug_json(payload_log, 'request')

        response = requests.post(self.url, json=payload)
        if response.status_code != 200:
            self.logger.error(
                f'Starriver detector request failed, status code: {response.status_code}')
            if response.json().get('Code', -1) != 0:
                self.logger.error(
                    f'Starriver detector error: {response.json().get("Message", "")}')
                with open('stariver_ocr_error.txt', 'w', encoding='utf-8') as f:
                    f.write(response.text)
            raise ValueError('Starriver detector request failed.')
        response_data = response.json()['Data']

        if self.debug:
            self.save_debug_json(response_data, 'response')

        blk_list = []
        for block in response_data.get('text_block', []):
            if scale < 1:
                xyxy = [int(min(coord[0] for coord in block['block_coordinate'].values()) / scale),
                        int(min(
                            coord[1] for coord in block['block_coordinate'].values()) / scale),
                        int(max(
                            coord[0] for coord in block['block_coordinate'].values()) / scale),
                        int(max(coord[1] for coord in block['block_coordinate'].values()) / scale)]
                lines = [np.array([[coord[pos][0] / scale, coord[pos][1] / scale] for pos in ['upper_left', 'upper_right',
                                                                                              'lower_right', 'lower_left']], dtype=np.float32) for coord in block['coordinate']]
            else:
                xyxy = [int(min(coord[0] for coord in block['block_coordinate'].values())),
                        int(min(coord[1]
                            for coord in block['block_coordinate'].values())),
                        int(max(coord[0]
                            for coord in block['block_coordinate'].values())),
                        int(max(coord[1] for coord in block['block_coordinate'].values()))]
                lines = [np.array([[coord[pos][0], coord[pos][1]] for pos in ['upper_left', 'upper_right',
                                                                              'lower_right', 'lower_left']], dtype=np.float32) for coord in block['coordinate']]
            texts = [text.replace('<skip>', '')
                     for text in block.get('texts', [])]

            original_font_size = block.get('text_size', 0)
            scaled_font_size = original_font_size / \
                scale if scale < 1 else original_font_size
            font_size_recalculated = self.adjust_font_size(scaled_font_size)

            if self.debug:
                self.logger.debug(
                    f'Original font size: {original_font_size}, adjusted: {font_size_recalculated}')

            blk = TextBlock(
                xyxy=xyxy,
                lines=lines,
                language=block.get('language', 'unknown'),
                vertical=block.get('is_vertical', False),
                font_size=font_size_recalculated,

                text=texts,
                fg_colors=np.array(block.get('foreground_color', [
                                   0, 0, 0]), dtype=np.float32),
                bg_colors=np.array(block.get('background_color', [
                                   0, 0, 0]), dtype=np.float32)
            )
            blk_list.append(blk)
            if self.debug:
                self.logger.debug(f'Detected text block: {blk.to_dict()}')

        mask = self._decode_base64_mask(
            response_data['mask']) if response_data.get('mask', '') != '' else None
        if mask is None:
            self.logger.warning('Starriver detector did not detect any text')
            return None, []
        mask = self.expand_mask(mask)

        # scale back to original size
        if scale < 1:
            mask = cv2.resize(mask, (width, height),
                              interpolation=cv2.INTER_NEAREST)
        self.logger.debug(f'Result mask shape: {mask.shape}')
        return mask, blk_list

    @staticmethod
    def _decode_base64_mask(base64_str: str) -> np.ndarray:
        img_data = base64.b64decode(base64_str)
        img_array = np.frombuffer(img_data, dtype=np.uint8)
        mask = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        return mask

    def expand_mask(self, mask: np.ndarray, expand_ratio: float = 0.01) -> np.ndarray:
        """
        Expand mask region for better text extraction.
        :param mask: input mask
        :param expand_ratio: expansion ratio (default 0.01)
        :return: expanded mask
        """

        if expand_ratio == 0:
            return mask

        # Ensure mask is binary (0 and 255 only)
        mask = (mask > 0).astype(np.uint8) * 255

        # Get image dimensions
        height, width = mask.shape

        # Kernel size from image size and expand_ratio
        kernel_size = int(min(height, width) * expand_ratio)
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure odd kernel size

        # Create square kernel
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # Dilate
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)

        # Binarize expanded mask
        dilated_mask = (dilated_mask > 0).astype(np.uint8) * 255

        return dilated_mask

    def update_token_if_needed(self):
        token_updated = False
        if (self.User != self.register_username or 
            self.Password != self.register_password):
            if self.token_obtained == False:
                if "Enter your username" not in self.User and "Enter your password" not in self.Password:
                    if len(self.Password) > 7 and len(self.User) >= 1:
                        new_token = self.get_token()
                        if new_token:  # 确保新获取到有效token再更新信息
                            self.token = new_token
                            self.register_username = self.User
                            self.register_password = self.Password
                            self.token_obtained = True
                            self.logger.info("Token updated due to credential change.")
                            token_updated = True
        return token_updated

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)

        if param_key == 'update_token_btn':
            self.token_obtained = False  # 强制刷新token时，将标志位设置为False
            self.token = ''  # 强制刷新token时，将token置空
            self.register_username = None  # 强制刷新token时，将用户名置空
            self.register_password = None  # 强制刷新token时，将密码置空
            try:
                if self.update_token_if_needed():
                    create_info_dialog('Token updated successfully')
            except Exception as e:
                create_error_dialog(e, 'Token update failed', 'TokenUpdateFailed')

    def save_debug_json(self, data, prefix='debug'):
        timestamp = int(time.time())
        filename = f"{prefix}_{timestamp}.json"
        os.makedirs('debug_logs', exist_ok=True)
        filepath = os.path.join('debug_logs', filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        self.logger.debug(f"Debug JSON saved to {filepath}")