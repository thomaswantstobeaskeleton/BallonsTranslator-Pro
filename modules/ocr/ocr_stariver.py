import numpy as np
import json
import cv2
import requests
import base64
from typing import List

from .base import register_OCR, OCRBase, TextBlock
from utils.message import create_error_dialog, create_info_dialog


@register_OCR('stariver_ocr')
class OCRStariver(OCRBase):
    params = {
        'User': "Enter your username",
        'Password': "Enter your password. Stored in plain text; avoid on shared computers.",
        "refine":{
            'type': 'checkbox',
            'value': True
        },
        "filtrate":{
            'type': 'checkbox',
            'value': True
        },
        "disable_skip_area":{
            'type': 'checkbox',
            'value': True
        },
        "detect_scale": "3",
        "merge_threshold": "2",
        "force_expand":{
            'type': 'checkbox',
            'value': False,
            'description': 'Force expand image pixels; may slow down recognition'
        },
        "low_accuracy_mode":{
            'type': 'checkbox',
            'value': False,
        },
        'update_token_btn': {
            'type': 'pushbtn',
            'value': '',
            'description': 'Clear stored token and request a new one',
            'display_name': 'Update token'
        },
        'description': 'Starriver Cloud (Tuanzi) OCR API'
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
        return  self.params['refine']['value']
     
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
    def force_expand(self):
        return self.params['force_expand']['value']
    
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
            raise   Exception(error_msg)
        token = response.get('Token', '')
        if token != '':
            self.logger.info(f'Login successful, token prefix: {token[:10]}')

        return token

    def _ocr_blk_list(self, img: np.ndarray, blk_list: List[TextBlock], *args, **kwargs):
        self.update_token_if_needed() # 在向服务器发送请求前尝试更新 Token
        im_h, im_w = img.shape[:2]
        for blk in blk_list:
            x1, y1, x2, y2 = blk.xyxy
            x1 = max(0, min(int(round(float(x1))), im_w - 1))
            y1 = max(0, min(int(round(float(y1))), im_h - 1))
            x2 = max(x1 + 1, min(int(round(float(x2))), im_w))
            y2 = max(y1 + 1, min(int(round(float(y2))), im_h))
            if 0 <= x1 < x2 <= im_w and 0 <= y1 < y2 <= im_h:
                blk.text = [self.ocr(img[y1:y2, x1:x2])]
            else:
                self.logger.warning('invalid textbbox to target img')
                blk.text = ['']

    def ocr_img(self, img: np.ndarray) -> str:
        self.update_token_if_needed() # 在向服务器发送请求前尝试更新 Token
        self.logger.debug(f'ocr_img: {img.shape}')
        return self.ocr(img)

    def ocr(self, img: np.ndarray) -> str:
        
        payload = {
            "token": self.token,
            "mask": False,
            "refine": self.refine,
            "filtrate": self.filtrate,
            "disable_skip_area": self.disable_skip_area,
            "detect_scale": self.detect_scale,
            "merge_threshold": self.merge_threshold,
            "low_accuracy_mode": self.params['low_accuracy_mode']['value'],
            "force_expand": self.force_expand
        }

        enc_ext = '.png' if (img.ndim == 3 and img.shape[2] == 4) else '.jpg'
        img_base64 = base64.b64encode(
            cv2.imencode(enc_ext, img)[1]).decode('utf-8')
        payload["image"] = img_base64

        response = requests.post(self.url, data=json.dumps(payload))

        if response.status_code != 200:
            print(f'Starriver OCR request failed, status code: {response.status_code}')
            if response.json().get('Code', -1) != 0:
                print(f'Starriver OCR error: {response.json().get("Message", "")}')
                with open('stariver_ocr_error.txt', 'w', encoding='utf-8') as f:
                    f.write(response.text)
            raise ValueError('Starriver OCR request failed.')

        response_data = response.json()['Data']

        if self.debug:
            id = response.json().get('RequestID', '')
            file_name = f"stariver_ocr_response_{id}.json"
            print(f"Starriver OCR request successful, response saved to {file_name}")
            with open(file_name, 'w', encoding='utf-8') as f:
                json.dump(response_data, f, ensure_ascii=False, indent=4)

        texts_list = ["".join(block.get('texts', '')).strip()
                      for block in response_data.get('text_block', [])]
        texts_str = "".join(texts_list).replace('<skip>', '')
        return texts_str

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