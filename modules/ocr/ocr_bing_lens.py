import re
import numpy as np
import time
import cv2
import random
import string
from typing import List
import os
import base64
import uuid
import json

import httpx
from PIL import Image as PilImage
import io
import http.cookiejar as cookielib
from urllib.parse import urlparse, parse_qs

from .base import register_OCR, OCRBase, TextBlock

class BingOCRCore:
    API_ENDPOINT = 'https://www.bing.com/images/api/custom/knowledge'
    UPLOAD_ENDPOINT = 'https://www.bing.com/images/search?view=detailv2&iss=sbiupload&FORM=SBIIDP&sbisrc=ImgDropper&idpbck=1'
    HEADERS = {
        'accept': '*/*',
        'accept-language': 'ru,en;q=0.9,en-GB;q=0.8,en-US;q=0.7',
        'origin': 'https://www.bing.com',
        'referer': 'https://www.bing.com/images/search?view=detailV2&iss=SBIUPLOADGET&sbisrc=ImgDropper',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36 Edg/134.0.0.0'
    }

    def __init__(self, proxy=None):
        self.proxy = proxy
        self.cookie_jar = cookielib.CookieJar()

    def _send_request(self, url, headers, data=None, files=None, cookies=None, follow_redirects=False, timeout=10.0):
        try:
            client_kwargs = {}
            if self.proxy:
                if isinstance(self.proxy, str):
                    client_kwargs['proxy'] = self.proxy
                elif isinstance(self.proxy, dict):
                    mounts = {}
                    if 'http://' in self.proxy:
                        mounts["http://"] = httpx.HTTPTransport(proxy=self.proxy['http://'])
                    if 'https://' in self.proxy:
                        mounts["https://"] = httpx.HTTPTransport(proxy=self.proxy['https://'])
                    if mounts:
                        client_kwargs['mounts'] = mounts
                else:
                    raise ValueError("Proxy must be a string or a dictionary")
            client = httpx.Client(**client_kwargs, timeout=timeout) 
            response = client.post(url, headers=headers, data=data, files=files, cookies=cookies, follow_redirects=follow_redirects)
            return response
        except httpx.TimeoutException as e:
            raise Exception(f"Request to {url} timed out: {e}") 
        except httpx.HTTPError as e: # Обработка HTTP ошибок остается
            raise Exception(f"HTTP error {e.response.status_code} during request to {url}: {e.response.text}")
        except Exception as e:
            raise Exception(f"Request to {url} failed: {e}")

    def upload_image(self, image_path=None, image_buffer=None):
        try:
            image_base64 = None 

            if image_path: 
                with open(image_path, "rb") as image_file:
                    image_data = image_file.read()
                    image_base64 = base64.b64encode(image_data).decode('utf-8')
                img = PilImage.open(image_path) 
            elif image_buffer: 
                image_base64 = base64.b64encode(image_buffer).decode('utf-8')
                img = PilImage.open(io.BytesIO(image_buffer))
            else:
                raise ValueError("Either image_path or image_buffer must be provided")


            width, height = img.size
            file_size_bytes = len(image_buffer) if image_buffer else os.path.getsize(image_path)
            file_size_kb = round(file_size_bytes / 1024, 2)
            file_name = os.path.basename(image_path) if image_path else "image_from_buffer.jpg" 
            file_extension = os.path.splitext(image_path)[1][1:].lower() if image_path else "jpg" 

            sbifsz_value = f"{width}+x+{height}+%c2%b7+{file_size_kb}+kB+%c2%b7+{file_extension}"
            sbifnm_value = file_name
            thw_value = width
            thh_value = height
            expw_value = width
            exph_value = height

            upload_url = f'{self.UPLOAD_ENDPOINT}&sbifsz={sbifsz_value}&sbifnm={sbifnm_value}&thw={thw_value}&thh={thh_value}&ptime=26&dlen=29932&expw={expw_value}&exph={exph_value}'

            boundary_upload = f"----WebKitFormBoundary{uuid.uuid4().hex}"
            upload_headers = self.HEADERS.copy()
            upload_headers['content-type'] = f'multipart/form-data; boundary={boundary_upload}'

            upload_data = f'''{boundary_upload}\r\nContent-Disposition: form-data; name="imgurl"\r\n\r\n\r\n{boundary_upload}\r\nContent-Disposition: form-data; name="cbir"\r\n\r\nsbi\r\n{boundary_upload}\r\nContent-Disposition: form-data; name="imageBin"\r\n\r\n{image_base64}\r\n{boundary_upload}--\r\n'''

            upload_response = self._send_request(upload_url, upload_headers, data=upload_data.encode('utf-8'), follow_redirects=False) 

            if upload_response.status_code == 302: 
                redirect_url = upload_response.headers.get('Location')
                if not redirect_url:
                    raise Exception("Redirect 302 received but no Location header found.")
            else: 
                upload_response.raise_for_status() 
                redirect_url = None 

            if not redirect_url:
                raise Exception("No redirect URL received after image upload (not 302).")


            parsed_url = urlparse(redirect_url)
            query_params = parse_qs(parsed_url.query)
            image_insights_token = query_params.get('insightsToken')
            if not image_insights_token:
                raise Exception("Image insights token not found in redirect URL.")
            return image_insights_token[0], upload_response.cookies

        except Exception as e:
            raise Exception(f"Image upload failed: {e}")

    def get_ocr_json(self, image_insights_token, upload_cookies=None):
        api_url = self.API_ENDPOINT
        boundary_ocr = f"----WebKitFormBoundary{uuid.uuid4().hex}"
        api_headers = self.HEADERS.copy()
        api_headers['content-type'] = f'multipart/form-data; boundary={boundary_ocr}'
        api_headers['referer'] = f'https://www.bing.com/images/search?view=detailV2&insightstoken={image_insights_token}'

        api_data_json = {
            "imageInfo": {"imageInsightsToken": image_insights_token, "source": "Url"},
            "knowledgeRequest": {"invokedSkills": ["OCR"], "index": 1}
        }
        api_data = f'''{boundary_ocr}\r\nContent-Disposition: form-data; name="knowledgeRequest"\r\n\r\n{json.dumps(api_data_json)}\r\n{boundary_ocr}--\r\n'''

        try:
            api_response = self._send_request(api_url, api_headers, data=api_data.encode('utf-8'), cookies=upload_cookies)
            return api_response.json()
        except httpx.TimeoutException as e: 
            raise Exception(f"OCR API request timed out: {e}") 
        except httpx.HTTPError as e: 
            raise Exception(f"HTTP error {e.response.status_code} during OCR API request to {api_url}: {e.response.text}")
        except Exception as e: 
            raise Exception(f"OCR API request failed: {e}")


class BingOCR(BingOCRCore):
    def __init__(self, proxy=None):
        super().__init__(proxy=proxy)

    def scan_by_file(self, file_path):
        image_insights_token, upload_cookies = self.upload_image(image_path=file_path)
        ocr_json = self.get_ocr_json(image_insights_token, upload_cookies)
        return ocr_json

    def scan_by_buffer(self, buffer, filename=None): 
        image_insights_token, upload_cookies = self.upload_image(image_buffer=buffer) 
        ocr_json = self.get_ocr_json(image_insights_token, upload_cookies)
        return ocr_json


class BingOCRAPI:
    def __init__(self, proxy=None):
        self.bing_ocr = BingOCR(proxy=proxy)

    @staticmethod
    def extract_text_and_coordinates(ocr_json_data):
        text_with_coords = []
        try:
            ocr_tag = ocr_json_data['tags'][1]['actions'][0] 
            if ocr_tag['_type'] == 'ImageKnowledge/TextRecognitionAction':
                regions = ocr_tag['data']['regions']
                for region in regions:
                    for line in region['lines']:
                        line_text = line['text']
                        line_bbox = line['boundingBox']
                        text_with_coords.append({"text": line_text, "boundingBox": line_bbox}) 
        except (KeyError, IndexError, TypeError):
            return [] 
        return text_with_coords

    @staticmethod
    def stitch_text_smart(text_with_coords):
        if not text_with_coords:
            return ""

        def get_bbox_coords(bbox):
            return bbox['topLeft']['x'], bbox['topLeft']['y'], bbox['bottomRight']['x'], bbox['bottomRight']['y']

        sorted_elements = sorted(text_with_coords, key=lambda x: (get_bbox_coords(x['boundingBox'])[1], get_bbox_coords(x['boundingBox'])[0]))

        stitched_text = []
        current_y_start = None
        current_line = []

        for element in sorted_elements:
            bbox = get_bbox_coords(element['boundingBox'])
            y_start = bbox[1]
            text = element['text']

            if current_y_start is None or abs(y_start - current_y_start) > 0.03: 
                if current_line:
                    stitched_text.append(" ".join(current_line))
                    current_line = []
                current_y_start = y_start
            current_line.append(text)

        if current_line:
            stitched_text.append(" ".join(current_line))

        return "\n".join(stitched_text).strip()

    @staticmethod
    def stitch_text_sequential(text_with_coords):
        return " ".join([item['text'] for item in text_with_coords]).strip() if text_with_coords else ""

    def process_image(self, image_path=None, image_buffer=None, response_method="Full Text"):
        if image_path:
            ocr_json_data = self.bing_ocr.scan_by_file(image_path)
        elif image_buffer:
            ocr_json_data = self.bing_ocr.scan_by_buffer(image_buffer)
        else:
            raise ValueError("Either image_path or image_buffer must be provided")

        text_with_coords = BingOCRAPI.extract_text_and_coordinates(ocr_json_data)

        if response_method == "Full Text":
            return {
                'full_text': BingOCRAPI.stitch_text_smart(text_with_coords), 
                'text_with_coordinates': text_with_coords
            }
        elif response_method == "Coordinate sequence":
            return {
                'full_text': BingOCRAPI.stitch_text_sequential(text_with_coords),
                'text_with_coordinates': text_with_coords
            }
        elif response_method == "Location coordinates": 
            return {
                'full_text': BingOCRAPI.stitch_text_smart(text_with_coords),
                'text_with_coordinates': text_with_coords
            }
        else:
            raise ValueError("Invalid response method")


def format_bing_ocr_result(result):
    full_text = result.get("full_text", "")
    if not full_text:
        formatted_result = {
            "language": result.get("language", ""),
            "text_with_coordinates": [ 
                f"{item['text']}: {item['boundingBox']}"
                for item in result.get("text_with_coordinates", [])
            ]
        }
        return json.dumps(formatted_result, indent=4, ensure_ascii=False)
    else:
        return f"OCR Text: '{full_text}'" 


@register_OCR('bing_ocr')
class OCRBingAPI(OCRBase):
    params = {
        "delay": 1.0,
        'newline_handling': {
            'type': 'selector',
            'options': [
                'preserve',
                'remove'
            ],
            'value': 'preserve',
            'description': 'Choose how to handle newline characters in OCR result'
        },
        'no_uppercase': {
            'type': 'checkbox',
            'value': False,
            'description': 'Convert text to lowercase except the first letter of each sentence'
        },
        'response_method': {
            'type': 'selector',
            'options': [
                'Full Text',
                'Coordinate sequence',
                'Location coordinates'
            ],
            'value': 'Full Text',
            'description': 'Choose the method for extracting text from image'
        },
        'proxy': {
            'value': '',
            'description': 'Proxy address (e.g., http(s)://user:password@host:port or socks4/5://user:password@host:port)'
        },
        'description': 'OCR using Bing OCR API'
    }

    @property
    def request_delay(self):
        try:
            return float(self.get_param_value('delay'))
        except (ValueError, TypeError):
            return 1.0

    @property
    def newline_handling(self):
        return self.get_param_value('newline_handling')

    @property
    def no_uppercase(self):
        return self.get_param_value('no_uppercase')

    @property
    def response_method(self):
        return self.get_param_value('response_method')

    @property
    def proxy(self):
        return self.get_param_value('proxy')

    def __init__(self, **params) -> None:
        if 'delay' in params:
            try:
                params['delay'] = float(params['delay'])
            except (ValueError, TypeError):
                params['delay'] = 1.0
        super().__init__(**params)
        self.api = BingOCRAPI(proxy=self.proxy)
        self.last_request_time = 0

    def _ocr_blk_list(self, img: np.ndarray, blk_list: List[TextBlock], *args, **kwargs):
        im_h, im_w = img.shape[:2]
        if self.debug_mode:
            self.logger.debug(f'Image size: {im_h}x{im_w}')
        for blk in blk_list:
            x1, y1, x2, y2 = blk.xyxy
            x1 = max(0, min(int(round(float(x1))), im_w - 1))
            y1 = max(0, min(int(round(float(y1))), im_h - 1))
            x2 = max(x1 + 1, min(int(round(float(x2))), im_w))
            y2 = max(y1 + 1, min(int(round(float(y2))), im_h))
            if self.debug_mode:
                self.logger.debug(f'Processing block: ({x1, y1, x2, y2})')
            if x1 < x2 and y1 < y2:
                cropped_img = img[y1:y2, x1:x2]
                if self.debug_mode:
                    self.logger.debug(f'Cropped image size: {cropped_img.shape}')
                blk.text = [self.ocr(cropped_img)]
            else:
                if self.debug_mode:
                    self.logger.warning('Invalid text bbox to target image')
                blk.text = ['']

    def ocr_img(self, img: np.ndarray) -> str:
        if self.debug_mode:
            self.logger.debug(f'ocr_img: {img.shape}')
        return self.ocr(img)

    def ocr(self, img: np.ndarray) -> str:
        if self.debug_mode:
            self.logger.debug(f'Starting OCR on image of shape: {img.shape}')
        self._respect_delay()
        try:
            if img.size > 0:
                if self.debug_mode:
                    self.logger.debug(f'Input image size: {img.shape}')
                _, buffer = cv2.imencode('.jpg', img)
                result = self.api.process_image(image_buffer=buffer.tobytes(), response_method=self.response_method)
                if self.debug_mode:
                    formatted_result = format_bing_ocr_result(result) 
                    self.logger.debug(f'OCR result: {formatted_result}') 

                full_text = result['full_text']
                if self.newline_handling == 'remove':
                    full_text = full_text.replace('\n', ' ')

                full_text = self._apply_punctuation_and_spacing(full_text)

                if self.no_uppercase:
                    full_text = self._apply_no_uppercase(full_text)

                if isinstance(full_text, list):
                    return '\n'.join(full_text)
                else:
                    return full_text
            else:
                if self.debug_mode:
                    self.logger.warning('Empty image provided for OCR')
                return ''
        except Exception as e:
            if self.debug_mode:
                self.logger.error(f"OCR error: {str(e)}")
            return ''

    def _apply_no_uppercase(self, text: str) -> str:
        def process_sentence(sentence):
            words = sentence.split()
            if not words:
                return ''
            processed = [words[0].capitalize()] + [word.lower() for word in words[1:]]
            return ' '.join(processed)

        sentences = re.split(r'(?<=[.!?…])\s+', text)
        processed_sentences = [process_sentence(sentence) for sentence in sentences]

        return ' '.join(processed_sentences)

    def _apply_punctuation_and_spacing(self, text: str) -> str:
        text = re.sub(r'\s+([,.!?…])', r'\1', text)
        text = re.sub(r'([,.!?…])(?!\s)(?![,.!?…])', r'\1 ', text)
        text = re.sub(r'([,.!?…])\s+([,.!?…])', r'\1\2', text)
        return text.strip()

    def _respect_delay(self):
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if self.debug_mode:
            self.logger.info(f'Time since last request: {time_since_last_request} seconds')

        if time_since_last_request < self.request_delay:
            sleep_time = self.request_delay - time_since_last_request
            if self.debug_mode:
                self.logger.info(f'Sleeping for {sleep_time} seconds')
            time.sleep(sleep_time)
        self.last_request_time = time.time()

    def updateParam(self, param_key: str, param_content):
        if param_key == 'delay':
            try:
                param_content = float(param_content)
            except (ValueError, TypeError):
                param_content = 1.0 
        super().updateParam(param_key, param_content)
        if param_key == 'proxy':
            
            self.api.bing_ocr.proxy = self.proxy 
            self.api.bing_ocr.client = None 
